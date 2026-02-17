use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

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

pub fn stft_magnitude(signal: &[f32], fft_size: usize, hop_size: usize) -> Array2<f32> {
    if signal.len() < fft_size || fft_size == 0 || hop_size == 0 {
        return Array2::zeros((0, 0));
    }

    let frames = 1 + (signal.len() - fft_size) / hop_size;
    let bins = fft_size / 2 + 1;
    let mut spec = Array2::<f32>::zeros((frames, bins));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let window = hann_window(fft_size);
    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];

    for frame_idx in 0..frames {
        let offset = frame_idx * hop_size;
        for i in 0..fft_size {
            buffer[i] = Complex32::new(signal[offset + i] * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..bins {
            spec[(frame_idx, bin)] = buffer[bin].norm();
        }
    }

    spec
}

fn hann_window(size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(size);
    let denom = (size as f32).max(1.0) - 1.0;
    for i in 0..size {
        let phase = 2.0 * std::f32::consts::PI * i as f32 / denom.max(1.0);
        out.push(0.5 - 0.5 * phase.cos());
    }
    out
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
    fn stft_returns_expected_shape() {
        let signal = vec![0.0_f32; 1024];
        let out = stft_magnitude(&signal, 256, 128);
        assert_eq!(out.shape(), &[7, 129]);
    }
}
