use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const RVC_N_FFT: usize = 1024;
pub const RVC_HOP_LENGTH: usize = 160;
pub const RVC_WIN_LENGTH: usize = RVC_N_FFT;
pub const RMVPE_SAMPLE_RATE: u32 = 16_000;
pub const RMVPE_MEL_BINS: usize = 128;
pub const RMVPE_MIN_FRAMES: usize = 32;
pub const RMVPE_FRAME_ALIGN: usize = 32;

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
    // DC blocker + gentle peak limiter. Avoid always-on nonlinear shaping to reduce metallic tone.
    let mut prev_x = 0.0_f32;
    let mut prev_y = 0.0_f32;
    let hp_alpha = 0.995_f32;
    for s in &mut out {
        let x = *s * gain;
        let y_hp = x - prev_x + hp_alpha * prev_y;
        prev_x = x;
        prev_y = y_hp;
        let mut v = y_hp;
        let abs = v.abs();
        if abs > 0.95 {
            // Engage soft limiting only near clipping.
            v = v.signum() * ((2.2_f32 * abs).tanh() / 2.2_f32);
        }
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
    let gain_exp = 1.0 - mix;
    for (i, sample) in out.iter_mut().enumerate() {
        let src_pos = i as f32 * in_last / out_last;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(in_len - 1);
        let frac = src_pos - left as f32;
        let in_rms = in_env[left] * (1.0 - frac) + in_env[right] * frac;
        let out_rms = out_env[i].max(1e-4);
        let gain = (in_rms / out_rms).clamp(0.1, 10.0).powf(gain_exp);
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

pub fn resample_hq_into(samples: &[f32], src_rate: u32, dst_rate: u32, out: &mut Vec<f32>) {
    out.clear();
    if samples.is_empty() || src_rate == 0 || dst_rate == 0 {
        return;
    }
    if src_rate == dst_rate {
        out.extend_from_slice(samples);
        return;
    }
    // Small blocks are cheaper with linear interpolation and practically indistinguishable.
    if samples.len() < 64 {
        resample_linear_into(samples, src_rate, dst_rate, out);
        return;
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    out.reserve(out_len.saturating_sub(out.capacity()));

    // Windowed-sinc resampling. This is significantly cleaner than linear interpolation,
    // especially in downsampling paths (e.g. 48k -> 16k for HuBERT/RMVPE).
    const HALF_TAPS: isize = 16;
    let cutoff = (dst_rate.min(src_rate) as f32 / src_rate as f32).clamp(0.01, 1.0);
    let denom = (HALF_TAPS as f32 + 1.0).max(1.0);
    let pi = std::f32::consts::PI;

    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let center = src_pos.floor() as isize;
        let frac = (src_pos - center as f64) as f32;

        let mut acc = 0.0_f32;
        let mut wsum = 0.0_f32;
        for k in -HALF_TAPS..=HALF_TAPS {
            let idx = center + k;
            if idx < 0 || idx >= samples.len() as isize {
                continue;
            }
            let x = (k as f32 - frac) * cutoff;
            let sinc = if x.abs() < 1.0e-6 {
                1.0
            } else {
                let pix = pi * x;
                pix.sin() / pix
            };
            let win_pos = ((k as f32 - frac) / denom).clamp(-1.0, 1.0);
            let window = 0.5 * (1.0 + (pi * win_pos).cos());
            let w = cutoff * sinc * window;
            acc += samples[idx as usize] * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1.0e-8 { acc / wsum } else { 0.0 });
    }
}

pub fn rmvpe_mel_from_audio(samples: &[f32], src_rate: u32) -> RmvpeMelInput {
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
        resample_hq_into(samples, src_rate, RMVPE_SAMPLE_RATE, &mut audio_16k);
    }
    if audio_16k.is_empty() {
        return RmvpeMelInput {
            mel: Array3::zeros((1, RMVPE_MEL_BINS, RMVPE_MIN_FRAMES)),
            valid_frames: 0,
        };
    }

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
    for m in 0..RMVPE_MEL_BINS {
        for t in 0..frames {
            let mut acc = 0.0_f32;
            for b in 0..bins {
                acc += mel_bank[(m, b)] * spec[(b, t)];
            }
            mel[(m, t)] = acc.max(1e-5).ln();
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

pub fn frame_features_for_rvc(
    signal: &[f32],
    hop_length: usize,
    feature_dim: usize,
) -> Array3<f32> {
    let channels = feature_dim.max(1);
    if signal.is_empty() || hop_length == 0 {
        return Array3::zeros((1, 1, channels));
    }

    let frames = signal.len().div_ceil(hop_length).max(1);
    let mut out = Array3::<f32>::zeros((1, frames, channels));

    for t in 0..frames {
        let start = t * hop_length;
        let end = (start + hop_length).min(signal.len());
        let frame = &signal[start..end];
        if frame.is_empty() {
            continue;
        }

        let mean_abs = frame.iter().map(|v| v.abs()).sum::<f32>() / frame.len() as f32;
        let rms = (frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32).sqrt();
        let zcr = frame
            .windows(2)
            .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
            .count() as f32
            / frame.len() as f32;

        for c in 0..channels {
            let v = match c % 4 {
                0 => mean_abs,
                1 => rms,
                2 => zcr,
                _ => frame[(c / 4) % frame.len()],
            };
            out[(0, t, c)] = v;
        }
    }

    out
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
    let mel_min = hz_to_mel(fmin.max(0.0));
    let mel_max = hz_to_mel(fmax.min(sr * 0.5));

    let mut mel_points = vec![0.0_f32; mel_bins + 2];
    for (i, v) in mel_points.iter_mut().enumerate() {
        let frac = i as f32 / (mel_bins + 1) as f32;
        *v = mel_to_hz(mel_min + frac * (mel_max - mel_min));
    }

    let mut bins = vec![0usize; mel_bins + 2];
    for (i, hz) in mel_points.iter().copied().enumerate() {
        let b = ((n_fft + 1) as f32 * hz / sr).floor() as usize;
        bins[i] = b.min(stft_bins.saturating_sub(1));
    }

    let mut fb = Array2::<f32>::zeros((mel_bins, stft_bins));
    for m in 0..mel_bins {
        let left = bins[m];
        let center = bins[m + 1];
        let right = bins[m + 2];
        if left >= right {
            continue;
        }
        if center > left {
            for b in left..center {
                fb[(m, b)] = (b - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for b in center..right {
                fb[(m, b)] = (right - b) as f32 / (right - center) as f32;
            }
        }
    }
    fb
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0_f32 * (1.0_f32 + hz / 700.0_f32).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0_f32 * (10.0_f32.powf(mel / 2595.0_f32) - 1.0_f32)
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
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
        let mut y = Vec::new();
        resample_hq_into(&x, 16_000, 16_000, &mut y);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_hq_changes_length_by_ratio() {
        let x = vec![0.0_f32; 160];
        let mut y = Vec::new();
        resample_hq_into(&x, 16_000, 48_000, &mut y);
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
}
