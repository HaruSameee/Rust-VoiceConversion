use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const RVC_N_FFT: usize = 1024;
pub const RVC_HOP_LENGTH: usize = 160;
pub const RVC_WIN_LENGTH: usize = RVC_N_FFT;
pub const RMVPE_SAMPLE_RATE: u32 = 16_000;
pub const RMVPE_MEL_BINS: usize = 128;
pub const RMVPE_MIN_FRAMES: usize = 32;
pub const RMVPE_FRAME_ALIGN: usize = 32;
// Favor stopband attenuation in 48k->16k downsampling for HuBERT/RMVPE input quality,
// while keeping realtime CPU usage reasonable.
const HQ_RESAMPLE_HALF_TAPS: isize = 60;
const HQ_RESAMPLE_PHASES: usize = 512;
const HQ_DOWNSAMPLE_CUTOFF_GUARD: f32 = 0.96;
static HQ_RESAMPLE_BANK_CACHE: OnceLock<Mutex<HashMap<(u32, u32), Arc<HqResampleBank>>>> =
    OnceLock::new();
static RMVPE_MEL_SCALE_LOGGED: OnceLock<()> = OnceLock::new();
static RMVPE_MEL_FLOOR_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct RmvpeMelInput {
    pub mel: Array3<f32>,
    pub valid_frames: usize,
}

#[derive(Debug, Clone)]
pub struct RmvpePaddedInput {
    pub padded: Vec<f32>,
    pub left_pad: usize,
    pub right_pad: usize,
    pub original_len: usize,
}

#[derive(Debug)]
struct HqResampleBank {
    phases: usize,
    half_taps: isize,
    kernel_len: usize,
    kernels: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct HqResampler {
    src_rate: u32,
    dst_rate: u32,
    ratio: f64,
    integer_downsample_step: Option<usize>,
    bank: Option<Arc<HqResampleBank>>,
}

pub fn normalize_peak(samples: &mut [f32], target_peak: f32) {
    let peak = samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, |acc, v| acc.max(v));
    if peak <= f32::EPSILON {
        return;
    }
    let gain = target_peak / peak;
    for sample in samples {
        *sample *= gain;
    }
}

pub fn normalize_for_onnx_input(samples: &[f32], target_peak: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<f32> = samples.to_vec();
    let peak = out
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, |acc, v| acc.max(v));
    // Streaming input should not be amplified per block; only attenuate when clipping.
    if peak > target_peak && peak > f32::EPSILON {
        let gain = target_peak / peak;
        for sample in &mut out {
            *sample *= gain;
        }
    }
    out
}

pub fn pad_for_rmvpe(samples: &[f32], hop_length: usize) -> RmvpePaddedInput {
    let left_pad = RVC_N_FFT / 2;
    let mut right_pad = RVC_N_FFT / 2;

    let mut padded = reflect_pad_center(samples, left_pad);
    if hop_length > 0 {
        let rem = padded.len() % hop_length;
        if rem != 0 {
            let extra = hop_length - rem;
            padded.extend(std::iter::repeat_n(0.0_f32, extra));
            right_pad += extra;
        }
    }

    RmvpePaddedInput {
        padded,
        left_pad,
        right_pad,
        original_len: samples.len(),
    }
}

pub fn trim_to_original_length(samples: &[f32], pad: &RmvpePaddedInput) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let start = pad.left_pad.min(samples.len());
    let end = (start + pad.original_len).min(samples.len());
    samples[start..end].to_vec()
}

pub fn postprocess_generated_audio(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = samples.to_vec();
    let peak = out
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, |acc, v| acc.max(v));
    if peak <= 1.0e-4 {
        out.fill(0.0);
        return out;
    }
    let gain = if peak > 0.99 { 0.99 / peak } else { 1.0 };
    // DC blocker + safety clip only.
    // Keep post-process as linear as possible to avoid adding metallic harmonics.
    let mut prev_x = 0.0_f32;
    let mut prev_y = 0.0_f32;
    let hp_alpha = 0.995_f32;
    for s in &mut out {
        let x = *s * gain;
        let y_hp = x - prev_x + hp_alpha * prev_y;
        prev_x = x;
        prev_y = y_hp;
        let v = y_hp;
        *s = v.clamp(-0.99, 0.99);
    }
    out
}

pub fn apply_rms_mix(input: &[f32], output: &[f32], rms_mix_rate: f32) -> Vec<f32> {
    if output.is_empty() {
        return Vec::new();
    }
    let mix = rms_mix_rate.clamp(0.0, 1.0);
    if mix >= 0.999 || input.is_empty() {
        return output.to_vec();
    }

    let in_env = rms_envelope(input);
    let out_env = rms_envelope(output);
    let in_len = in_env.len();
    let out_len = out_env.len();
    if in_len == 0 || out_len == 0 {
        return output.to_vec();
    }

    let mut out = output.to_vec();
    let in_last = (in_len - 1) as f32;
    let out_last = (out_len - 1).max(1) as f32;
    let src_scale = in_last / out_last;
    let gain_exp = 1.0 - mix;
    let gain_mode = if gain_exp <= 1.0e-6 {
        0_u8
    } else if (gain_exp - 1.0).abs() <= 1.0e-6 {
        1_u8
    } else if (gain_exp - 0.5).abs() <= 1.0e-6 {
        2_u8
    } else {
        3_u8
    };
    for (i, sample) in out.iter_mut().enumerate() {
        let src_pos = i as f32 * src_scale;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(in_len - 1);
        let frac = src_pos - left as f32;
        let in_rms = in_env[left] * (1.0 - frac) + in_env[right] * frac;
        let out_rms = out_env[i].max(1e-4);
        let ratio = (in_rms / out_rms).clamp(0.1, 10.0);
        let gain = match gain_mode {
            0 => 1.0,
            1 => ratio,
            2 => ratio.sqrt(),
            _ => ratio.powf(gain_exp),
        };
        *sample *= gain;
    }
    out
}

pub fn median_filter_pitch_track_inplace(pitch: &mut [f32], radius: usize) {
    if radius == 0 || pitch.len() < 3 {
        return;
    }
    let src = pitch.to_vec();
    let mut win = Vec::<f32>::with_capacity(radius * 2 + 1);
    for (i, dst) in pitch.iter_mut().enumerate() {
        if src[i] <= 0.0 {
            continue;
        }
        win.clear();
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(src.len());
        for &v in &src[lo..hi] {
            if v > 0.0 {
                win.push(v);
            }
        }
        if win.is_empty() {
            continue;
        }
        win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *dst = win[win.len() / 2];
    }
}

fn rms_envelope(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if samples.len() == 1 {
        return vec![samples[0].abs()];
    }

    let frame = samples.len().clamp(256, 1024);
    let hop = (frame / 4).max(1);
    let mut points = Vec::<(usize, f32)>::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame).min(samples.len());
        let seg = &samples[start..end];
        let rms = (seg.iter().map(|v| v * v).sum::<f32>() / seg.len().max(1) as f32).sqrt();
        points.push((start + (seg.len() / 2), rms));
        if end == samples.len() {
            break;
        }
        start += hop;
    }

    if points.len() == 1 {
        return vec![points[0].1; samples.len()];
    }

    let mut env = vec![0.0_f32; samples.len()];
    let mut p = 0usize;
    for (i, slot) in env.iter_mut().enumerate() {
        while p + 1 < points.len() && points[p + 1].0 <= i {
            p += 1;
        }
        if p + 1 >= points.len() {
            *slot = points[p].1;
        } else {
            let (x0, y0) = points[p];
            let (x1, y1) = points[p + 1];
            let denom = (x1.saturating_sub(x0)).max(1) as f32;
            let t = (i.saturating_sub(x0)) as f32 / denom;
            *slot = y0 * (1.0 - t) + y1 * t;
        }
    }
    env
}

pub fn resample_linear_into(samples: &[f32], src_rate: u32, dst_rate: u32, out: &mut Vec<f32>) {
    out.clear();
    if samples.is_empty() || src_rate == 0 || dst_rate == 0 {
        return;
    }
    if src_rate == dst_rate {
        out.extend_from_slice(samples);
        return;
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    out.reserve(out_len.saturating_sub(out.capacity()));

    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }
}

pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    let mut out = Vec::new();
    resample_linear_into(samples, src_rate, dst_rate, &mut out);
    out
}

impl HqResampler {
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        let ratio = if src_rate == 0 {
            1.0
        } else {
            dst_rate as f64 / src_rate as f64
        };
        let integer_downsample_step =
            if src_rate > dst_rate && dst_rate > 0 && src_rate % dst_rate == 0 {
                Some((src_rate / dst_rate) as usize)
            } else {
                None
            };
        let bank = if src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
            None
        } else {
            Some(get_hq_resample_bank(src_rate, dst_rate))
        };
        Self {
            src_rate,
            dst_rate,
            ratio,
            integer_downsample_step,
            bank,
        }
    }

    pub fn src_rate(&self) -> u32 {
        self.src_rate
    }

    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    pub fn is_identity(&self) -> bool {
        self.src_rate == self.dst_rate
    }

    fn output_len(&self, input_len: usize) -> usize {
        ((input_len as f64) * self.ratio).round().max(1.0) as usize
    }

    pub fn resample_into(&self, samples: &[f32], out: &mut Vec<f32>) {
        out.clear();
        if samples.is_empty() || self.src_rate == 0 || self.dst_rate == 0 {
            return;
        }
        if self.is_identity() {
            out.extend_from_slice(samples);
            return;
        }
        // Never use linear path for downsampling; it aliases without a low-pass.
        // For tiny upsampling blocks, linear is still acceptable and cheaper.
        if samples.len() < 64 && self.dst_rate >= self.src_rate {
            resample_linear_into(samples, self.src_rate, self.dst_rate, out);
            return;
        }

        let out_len = self.output_len(samples.len());
        out.reserve(out_len.saturating_sub(out.capacity()));
        let bank = self
            .bank
            .as_ref()
            .expect("non-identity resampler should have a filter bank");

        if let Some(step) = self.integer_downsample_step {
            let phase_base = 0usize;
            for i in 0..out_len {
                let center = i.saturating_mul(step) as isize;
                let mut acc = 0.0_f32;
                let mut wsum = 0.0_f32;
                for tap in 0..bank.kernel_len {
                    let k = tap as isize - bank.half_taps;
                    let idx = center + k;
                    if idx < 0 || idx >= samples.len() as isize {
                        continue;
                    }
                    let w = bank.kernels[phase_base + tap];
                    acc += samples[idx as usize] * w;
                    wsum += w;
                }
                out.push(if wsum.abs() > 1.0e-8 { acc / wsum } else { 0.0 });
            }
            return;
        }

        for i in 0..out_len {
            let src_pos = i as f64 / self.ratio;
            let center = src_pos.floor() as isize;
            let frac = (src_pos - center as f64) as f32;
            let mut phase_idx = (frac * bank.phases as f32).round() as usize;
            if phase_idx >= bank.phases {
                phase_idx = bank.phases - 1;
            }
            let phase_base = phase_idx * bank.kernel_len;

            let mut acc = 0.0_f32;
            let mut wsum = 0.0_f32;
            for tap in 0..bank.kernel_len {
                let k = tap as isize - bank.half_taps;
                let idx = center + k;
                if idx < 0 || idx >= samples.len() as isize {
                    continue;
                }
                let w = bank.kernels[phase_base + tap];
                acc += samples[idx as usize] * w;
                wsum += w;
            }
            out.push(if wsum.abs() > 1.0e-8 { acc / wsum } else { 0.0 });
        }
    }
}

pub fn resample_hq_into(resampler: &HqResampler, samples: &[f32], out: &mut Vec<f32>) {
    resampler.resample_into(samples, out);
}

fn get_hq_resample_bank(src_rate: u32, dst_rate: u32) -> Arc<HqResampleBank> {
    let cache = HQ_RESAMPLE_BANK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(bank) = guard.get(&(src_rate, dst_rate)) {
            return Arc::clone(bank);
        }
    }

    let built = Arc::new(build_hq_resample_bank(src_rate, dst_rate));
    if let Ok(mut guard) = cache.lock() {
        let entry = guard
            .entry((src_rate, dst_rate))
            .or_insert_with(|| Arc::clone(&built));
        return Arc::clone(entry);
    }
    built
}

fn build_hq_resample_bank(src_rate: u32, dst_rate: u32) -> HqResampleBank {
    let phases = HQ_RESAMPLE_PHASES;
    let half_taps = HQ_RESAMPLE_HALF_TAPS;
    let kernel_len = (half_taps * 2 + 1) as usize;
    let ratio = dst_rate as f32 / src_rate as f32;
    let cutoff = if ratio < 1.0 {
        (ratio * HQ_DOWNSAMPLE_CUTOFF_GUARD).clamp(0.01, 1.0)
    } else {
        1.0
    };
    let denom = (half_taps as f32 + 1.0).max(1.0);
    let pi = std::f32::consts::PI;
    let mut kernels = vec![0.0_f32; phases * kernel_len];

    for phase in 0..phases {
        let frac = phase as f32 / phases as f32;
        let base = phase * kernel_len;
        let mut wsum = 0.0_f32;
        for tap in 0..kernel_len {
            let k = tap as isize - half_taps;
            let x = (k as f32 - frac) * cutoff;
            let sinc = if x.abs() < 1.0e-6 {
                1.0
            } else {
                let pix = pi * x;
                pix.sin() / pix
            };
            let win_pos = ((k as f32 - frac) / denom).clamp(-1.0, 1.0);
            // Blackman window: stronger stopband attenuation than Hann.
            // win_pos is normalized to [-1, 1].
            let window =
                0.42 + 0.5 * (pi * win_pos).cos() + 0.08 * (2.0 * pi * win_pos).cos();
            let w = cutoff * sinc * window;
            kernels[base + tap] = w;
            wsum += w;
        }
        if wsum.abs() > 1.0e-8 {
            for tap in 0..kernel_len {
                kernels[base + tap] /= wsum;
            }
        }
    }

    HqResampleBank {
        phases,
        half_taps,
        kernel_len,
        kernels,
    }
}

pub fn rmvpe_mel_from_audio(samples: &[f32], src_rate: u32) -> RmvpeMelInput {
    let resampler = HqResampler::new(src_rate, RMVPE_SAMPLE_RATE);
    rmvpe_mel_from_audio_with_resampler(samples, src_rate, &resampler)
}

pub fn rmvpe_mel_from_audio_with_resampler(
    samples: &[f32],
    src_rate: u32,
    resampler: &HqResampler,
) -> RmvpeMelInput {
    if samples.is_empty() {
        return RmvpeMelInput {
            mel: Array3::zeros((1, RMVPE_MEL_BINS, RMVPE_MIN_FRAMES)),
            valid_frames: 0,
        };
    }

    let mut audio_16k = Vec::<f32>::new();
    if src_rate == RMVPE_SAMPLE_RATE {
        audio_16k.extend_from_slice(samples);
    } else {
        if resampler.src_rate() == src_rate && resampler.dst_rate() == RMVPE_SAMPLE_RATE {
            resample_hq_into(resampler, samples, &mut audio_16k);
        } else {
            let local = HqResampler::new(src_rate, RMVPE_SAMPLE_RATE);
            resample_hq_into(&local, samples, &mut audio_16k);
        }
    }
    if audio_16k.is_empty() {
        return RmvpeMelInput {
            mel: Array3::zeros((1, RMVPE_MEL_BINS, RMVPE_MIN_FRAMES)),
            valid_frames: 0,
        };
    }
    let (audio_16k_rms, audio_16k_peak) = signal_rms_peak(&audio_16k);

    let spec = stft_magnitude(&audio_16k, RVC_N_FFT, RVC_HOP_LENGTH);
    let frames = spec.ncols();
    if frames == 0 {
        return RmvpeMelInput {
            mel: Array3::zeros((1, RMVPE_MEL_BINS, RMVPE_MIN_FRAMES)),
            valid_frames: 0,
        };
    }

    let bins = spec.nrows();
    let mel_bank = mel_filter_bank(
        bins,
        RMVPE_SAMPLE_RATE,
        RMVPE_MEL_BINS,
        30.0_f32,
        8_000.0_f32,
    );
    let mut mel = Array2::<f32>::zeros((RMVPE_MEL_BINS, frames));
    let mut prelog_min = f32::INFINITY;
    let mut prelog_max = 0.0_f32;
    let mut prelog_sum = 0.0_f64;
    let mut log_min = f32::INFINITY;
    let mut log_max = f32::NEG_INFINITY;
    let mut mel_count: usize = 0;
    for m in 0..RMVPE_MEL_BINS {
        for t in 0..frames {
            let mut acc = 0.0_f32;
            for b in 0..bins {
                let mag = spec[(b, t)];
                let power = mag * mag;
                acc += mel_bank[(m, b)] * power;
            }
            let clamped = acc.max(1e-5);
            prelog_min = prelog_min.min(clamped);
            prelog_max = prelog_max.max(clamped);
            prelog_sum += clamped as f64;
            let logged = clamped.ln();
            log_min = log_min.min(logged);
            log_max = log_max.max(logged);
            mel_count += 1;
            mel[(m, t)] = logged;
        }
    }
    if mel_count > 0 {
        let prelog_mean = (prelog_sum / mel_count as f64) as f32;
        let all_floor = prelog_max <= 1.0001e-5;
        if all_floor {
            let seen = RMVPE_MEL_FLOOR_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if seen <= 8 || seen % 64 == 0 {
                eprintln!(
                    "[vc-signal] warning: rmvpe mel is floor-only (all {:.1e}); likely near-silence or gated input. src_rate={} len_16k={} rms={:.3e} peak={:.3e} count={}",
                    1.0e-5_f32,
                    src_rate,
                    audio_16k.len(),
                    audio_16k_rms,
                    audio_16k_peak,
                    seen
                );
            }
        } else if RMVPE_MEL_SCALE_LOGGED.set(()).is_ok() {
            eprintln!(
                "[vc-signal] rmvpe mel scale (power): prelog(min={:.3e} max={:.3e} mean={:.3e}) log(min={:.3} max={:.3}) bins={} frames={} rms={:.3e} peak={:.3e}",
                prelog_min,
                prelog_max,
                prelog_mean,
                log_min,
                log_max,
                RMVPE_MEL_BINS,
                frames,
                audio_16k_rms,
                audio_16k_peak
            );
        }
    }

    let aligned_frames = align_up(frames.max(RMVPE_MIN_FRAMES), RMVPE_FRAME_ALIGN);
    let mut out = Array3::<f32>::zeros((1, RMVPE_MEL_BINS, aligned_frames));
    for m in 0..RMVPE_MEL_BINS {
        for t in 0..aligned_frames {
            let src_t = t.min(frames - 1);
            out[(0, m, t)] = mel[(m, src_t)];
        }
    }
    RmvpeMelInput {
        mel: out,
        valid_frames: frames,
    }
}

pub fn resize_pitch_to_frames(pitch: &[f32], frames: usize) -> Vec<f32> {
    if frames == 0 {
        return Vec::new();
    }
    if pitch.is_empty() {
        return vec![0.0; frames];
    }
    if pitch.len() == frames {
        return pitch.to_vec();
    }
    if pitch.len() == 1 {
        return vec![pitch[0]; frames];
    }

    let mut out = Vec::with_capacity(frames);
    let scale = (pitch.len() - 1) as f64 / (frames - 1).max(1) as f64;
    for i in 0..frames {
        let src = i as f64 * scale;
        let left = src.floor() as usize;
        let right = (left + 1).min(pitch.len() - 1);
        let frac = (src - left as f64) as f32;
        out.push(pitch[left] * (1.0 - frac) + pitch[right] * frac);
    }
    out
}

pub fn coarse_pitch_from_f0(f0: &[f32]) -> Vec<i64> {
    let f0_min = 50.0_f32;
    let f0_max = 1100.0_f32;
    let f0_mel_min = 1127.0_f32 * (1.0 + f0_min / 700.0).ln();
    let f0_mel_max = 1127.0_f32 * (1.0 + f0_max / 700.0).ln();
    let mel_scale = 254.0_f32 / (f0_mel_max - f0_mel_min);

    let mut out = Vec::with_capacity(f0.len());
    for &v in f0 {
        if v <= 0.0 {
            out.push(1);
            continue;
        }
        let mel = 1127.0_f32 * (1.0 + v / 700.0).ln();
        let coarse = ((mel - f0_mel_min) * mel_scale + 1.0)
            .round()
            .clamp(1.0, 255.0) as i64;
        out.push(coarse);
    }
    out
}

pub fn stft_magnitude(signal: &[f32], n_fft: usize, hop_length: usize) -> Array2<f32> {
    if signal.is_empty() || n_fft == 0 || hop_length == 0 {
        return Array2::zeros((0, 0));
    }

    let pad = n_fft / 2;
    let padded = reflect_pad_center(signal, pad);
    if padded.len() < n_fft {
        return Array2::zeros((n_fft / 2 + 1, 0));
    }

    let frames = 1 + (padded.len() - n_fft) / hop_length;
    let bins = n_fft / 2 + 1;
    let mut spec = Array2::<f32>::zeros((bins, frames));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let window = hann_window_periodic(n_fft);
    let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];

    for frame_idx in 0..frames {
        let offset = frame_idx * hop_length;
        for i in 0..n_fft {
            buffer[i] = Complex32::new(padded[offset + i] * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..bins {
            spec[(bin, frame_idx)] = buffer[bin].norm();
        }
    }

    spec
}

pub fn stft_magnitude_rvc(signal: &[f32]) -> Array2<f32> {
    stft_magnitude(signal, RVC_N_FFT, RVC_HOP_LENGTH)
}

fn hann_window_periodic(size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(size);
    if size == 0 {
        return out;
    }
    let denom = size as f32;
    for i in 0..size {
        let phase = 2.0 * std::f32::consts::PI * i as f32 / denom;
        out.push(0.5 - 0.5 * phase.cos());
    }
    out
}

fn mel_filter_bank(
    stft_bins: usize,
    sample_rate: u32,
    mel_bins: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_fft = (stft_bins.saturating_sub(1)) * 2;
    let sr = sample_rate as f32;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.min(sr * 0.5));

    let mut mel_points = vec![0.0_f32; mel_bins + 2];
    for (i, v) in mel_points.iter_mut().enumerate() {
        let frac = i as f32 / (mel_bins + 1) as f32;
        *v = mel_to_hz_slaney(mel_min + frac * (mel_max - mel_min));
    }

    let mut fb = Array2::<f32>::zeros((mel_bins, stft_bins));
    let mut fft_freqs = vec![0.0_f32; stft_bins];
    for (i, f) in fft_freqs.iter_mut().enumerate() {
        *f = sr * (i as f32) / (n_fft as f32);
    }
    for m in 0..mel_bins {
        let left_hz = mel_points[m];
        let center_hz = mel_points[m + 1];
        let right_hz = mel_points[m + 2];
        let left_width = (center_hz - left_hz).max(1.0e-12);
        let right_width = (right_hz - center_hz).max(1.0e-12);
        if right_hz <= left_hz {
            continue;
        }

        for (b, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left_hz) / left_width;
            let upper = (right_hz - freq) / right_width;
            fb[(m, b)] = lower.min(upper).max(0.0);
        }

        // Match librosa(norm="slaney"):
        // scale each triangular filter by 2 / (f_right - f_left).
        let enorm = 2.0_f32 / (right_hz - left_hz).max(1.0e-12);
        for b in 0..stft_bins {
            fb[(m, b)] *= enorm;
        }
    }
    fb
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    // librosa default (htk=False) / Slaney Auditory Toolbox scale.
    // piecewise-linear below 1kHz, logarithmic above.
    let f_sp = 200.0_f32 / 3.0_f32;
    let min_log_hz = 1_000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0_f32;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0_f32 / 3.0_f32;
    let min_log_hz = 1_000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0_f32;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
}

fn signal_rms_peak(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0_f32;
    let mut peak = 0.0_f32;
    for &s in samples {
        sum_sq += s * s;
        peak = peak.max(s.abs());
    }
    ((sum_sq / samples.len() as f32).sqrt(), peak)
}

fn reflect_pad_center(signal: &[f32], pad: usize) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    if n == 1 {
        return vec![signal[0]; n + 2 * pad];
    }

    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..(n + 2 * pad) {
        let src = i as isize - pad as isize;
        let idx = reflect_index(src, n);
        out.push(signal[idx]);
    }
    out
}

fn reflect_index(index: isize, len: usize) -> usize {
    let period = (2 * (len - 1)) as isize;
    let mut x = index.rem_euclid(period);
    if x >= len as isize {
        x = period - x;
    }
    x as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_peak_scales_to_target() {
        let mut v = vec![0.25, -0.5, 0.1];
        normalize_peak(&mut v, 1.0);
        let peak = v.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));
        assert!((peak - 1.0).abs() < 1e-4);
    }

    #[test]
    fn resample_linear_identity_when_same_rate() {
        let x = vec![0.0_f32, 1.0, 0.0, -1.0];
        let y = resample_linear(&x, 16_000, 16_000);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_linear_changes_length_by_ratio() {
        let x = vec![0.0_f32; 160];
        let y = resample_linear(&x, 16_000, 48_000);
        assert_eq!(y.len(), 480);
    }

    #[test]
    fn resample_hq_identity_when_same_rate() {
        let x = vec![0.0_f32, 1.0, 0.0, -1.0];
        let resampler = HqResampler::new(16_000, 16_000);
        let mut y = Vec::new();
        resample_hq_into(&resampler, &x, &mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_hq_changes_length_by_ratio() {
        let x = vec![0.0_f32; 160];
        let resampler = HqResampler::new(16_000, 48_000);
        let mut y = Vec::new();
        resample_hq_into(&resampler, &x, &mut y);
        assert_eq!(y.len(), 480);
    }

    #[test]
    fn stft_returns_expected_shape_like_librosa() {
        let signal = vec![0.0_f32; 1024];
        let out = stft_magnitude(&signal, 256, 128);
        assert_eq!(out.shape(), &[129, 9]);
    }

    #[test]
    fn stft_rvc_uses_fixed_constants() {
        let signal = vec![0.0_f32; 16_000];
        let out = stft_magnitude_rvc(&signal);
        assert_eq!(out.shape(), &[513, 101]);
    }

    #[test]
    fn pitch_resize_matches_target_frames() {
        let pitch = vec![100.0_f32; 10];
        let resized = resize_pitch_to_frames(&pitch, 23);
        assert_eq!(resized.len(), 23);
    }

    #[test]
    fn coarse_pitch_range_is_valid() {
        let f0 = vec![0.0_f32, 50.0, 220.0, 880.0];
        let coarse = coarse_pitch_from_f0(&f0);
        assert!(coarse.iter().all(|v| (1..=255).contains(v)));
    }

    #[test]
    fn slaney_mel_known_points_match_librosa_definition() {
        // librosa(hkt=False) reference points:
        // 1000 Hz <-> 15 mel
        let mel_1k = hz_to_mel_slaney(1_000.0);
        assert!((mel_1k - 15.0).abs() < 1e-4);
        let hz_15 = mel_to_hz_slaney(15.0);
        assert!((hz_15 - 1_000.0).abs() < 1e-3);
    }
}
