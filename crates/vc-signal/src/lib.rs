use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const RVC_N_FFT: usize = 1024;
pub const RVC_HOP_LENGTH: usize = 160;
pub const RVC_WIN_LENGTH: usize = RVC_N_FFT;

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

    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    let mut out: Vec<f32> = samples.iter().map(|x| x - mean).collect();
    normalize_peak(&mut out, target_peak);
    out
}

pub fn pad_for_rmvpe(samples: &[f32], hop_length: usize) -> RmvpePaddedInput {
    let normalized = normalize_for_onnx_input(samples, 0.95);
    let left_pad = RVC_N_FFT / 2;
    let mut right_pad = RVC_N_FFT / 2;

    let mut padded = reflect_pad_center(&normalized, left_pad);
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
    let mut out = samples.to_vec();
    for s in &mut out {
        *s = s.clamp(-1.0, 1.0);
    }
    normalize_peak(&mut out, 0.98);
    out
}

pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return samples.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let v = samples[left] * (1.0 - frac) + samples[right] * frac;
        out.push(v);
    }

    out
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
