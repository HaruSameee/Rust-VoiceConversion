use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

pub const RVC_N_FFT: usize = 1024;
pub const RVC_HOP_LENGTH: usize = 160;
pub const RVC_WIN_LENGTH: usize = RVC_N_FFT;

/// 波形の最大振幅を `target_peak` にそろえる。
/// 無音に近い入力はそのまま返す。
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

/// シンプルな線形補間リサンプラ。
///
/// 高品質化より「低依存で軽いこと」を優先した実装です。
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

/// `librosa.stft` に寄せた振幅スペクトログラム（freq x time）を返す。
///
/// 返却形状: `[1 + n_fft / 2, n_frames]`
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

/// 周期版Hann窓（periodic）。
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

/// center=True 相当の反射パディング。
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

/// 反射境界でのインデックス変換。
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
}
