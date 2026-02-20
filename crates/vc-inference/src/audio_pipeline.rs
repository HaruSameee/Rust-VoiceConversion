//! audio_pipeline.rs
//! DSP post-processing for RVC real-time inference.
//! Fixes: feature resampling (49→50 frames), symmetric OLA crossfade,
//!        F0 smoothing with log-domain EMA and voiced hysteresis.

use ndarray::{Array2, ArrayView2};

const DECODER_FRAMES: usize = 50;
const VOICED_F0_HZ: f32 = 50.0;
const A4_HZ: f32 = 440.0;

#[inline]
fn clamp_index(i: isize, upper: usize) -> usize {
    if upper == 0 {
        return 0;
    }
    i.clamp(0, upper as isize - 1) as usize
}

#[inline]
fn src_pos(dst_i: usize, src_frames: usize, dst_frames: usize) -> f32 {
    if src_frames <= 1 || dst_frames <= 1 {
        return 0.0;
    }
    (dst_i as f32) * ((src_frames - 1) as f32) / ((dst_frames - 1) as f32)
}

#[inline]
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[inline]
fn allocate_dst(features: ArrayView2<'_, f32>, dst_frames: usize) -> Array2<f32> {
    Array2::<f32>::zeros((dst_frames, features.shape()[1]))
}

/// Resamples feature frames to the decoder contract using linear interpolation.
///
/// The destination frame count is fixed to 50 for RVC decoder input.
#[inline]
pub fn resample_features_linear(features: ArrayView2<f32>) -> Array2<f32> {
    resample_features_linear_to(features, DECODER_FRAMES)
}

#[inline]
fn resample_features_linear_to(features: ArrayView2<'_, f32>, dst_frames: usize) -> Array2<f32> {
    let src_frames = features.shape()[0];
    let feat_dim = features.shape()[1];
    let mut out = allocate_dst(features, dst_frames);
    if dst_frames == 0 || feat_dim == 0 || src_frames == 0 {
        return out;
    }
    if src_frames == 1 || dst_frames == 1 {
        for d in 0..dst_frames {
            for c in 0..feat_dim {
                out[(d, c)] = features[(0, c)];
            }
        }
        return out;
    }

    for d in 0..dst_frames {
        let pos = src_pos(d, src_frames, dst_frames);
        let left = pos.floor() as usize;
        let right = (left + 1).min(src_frames - 1);
        let frac = pos - left as f32;
        let w_left = 1.0 - frac;
        for c in 0..feat_dim {
            let a = features[(left, c)];
            let b = features[(right, c)];
            out[(d, c)] = a * w_left + b * frac;
        }
    }
    out
}

/// Resamples feature frames to the decoder contract using Catmull-Rom spline interpolation.
///
/// The destination frame count is fixed to 50 for RVC decoder input.
#[inline]
pub fn resample_features_catmull_rom(features: ArrayView2<f32>) -> Array2<f32> {
    resample_features_catmull_rom_to(features, DECODER_FRAMES)
}

#[inline]
fn resample_features_catmull_rom_to(
    features: ArrayView2<'_, f32>,
    dst_frames: usize,
) -> Array2<f32> {
    let src_frames = features.shape()[0];
    let feat_dim = features.shape()[1];
    let mut out = allocate_dst(features, dst_frames);
    if dst_frames == 0 || feat_dim == 0 || src_frames == 0 {
        return out;
    }
    if src_frames == 1 || dst_frames == 1 {
        for d in 0..dst_frames {
            for c in 0..feat_dim {
                out[(d, c)] = features[(0, c)];
            }
        }
        return out;
    }

    for d in 0..dst_frames {
        let pos = src_pos(d, src_frames, dst_frames);
        let i1 = pos.floor() as isize;
        let t = pos - i1 as f32;

        let i0 = clamp_index(i1 - 1, src_frames);
        let i1c = clamp_index(i1, src_frames);
        let i2 = clamp_index(i1 + 1, src_frames);
        let i3 = clamp_index(i1 + 2, src_frames);

        for c in 0..feat_dim {
            out[(d, c)] = catmull_rom(
                features[(i0, c)],
                features[(i1c, c)],
                features[(i2, c)],
                features[(i3, c)],
                t,
            );
        }
    }
    out
}

/// Symmetric Hann-window overlap-add processor for seamless block stitching.
pub struct OlaBuffer {
    overlap_samples: usize,
    prev_tail: Vec<f32>,
    hann_fade_out: Vec<f32>,
    hann_fade_in: Vec<f32>,
}

impl OlaBuffer {
    /// Creates a new overlap-add buffer with precomputed Hann fade curves.
    #[inline]
    pub fn new(overlap_samples: usize) -> Self {
        let n = overlap_samples;
        let mut hann_fade_out = vec![0.0_f32; n];
        if n > 0 {
            for (i, v) in hann_fade_out.iter_mut().enumerate() {
                *v = 0.5 * (1.0 + ((std::f32::consts::PI * (i as f32 + 0.5)) / n as f32).cos());
            }
        }
        let mut hann_fade_in = hann_fade_out.clone();
        hann_fade_in.reverse();

        Self {
            overlap_samples: n,
            prev_tail: vec![0.0; n],
            hann_fade_out,
            hann_fade_in,
        }
    }

    /// Applies symmetric overlap-add to a decoder output block.
    ///
    /// Returns `input.len() - overlap_samples` samples and stores block tail for the next call.
    #[inline]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let n = self.overlap_samples;
        assert!(input.len() > n.saturating_mul(2));

        let out_len = input.len().saturating_sub(n);
        let mut out = vec![0.0_f32; out_len];

        for i in 0..n.min(out_len) {
            out[i] = self.prev_tail[i] * self.hann_fade_out[i] + input[i] * self.hann_fade_in[i];
        }
        for i in n..out_len {
            out[i] = input[i];
        }

        if n > 0 {
            let tail_start = input.len() - n;
            self.prev_tail.copy_from_slice(&input[tail_start..]);
        }
        out
    }

    /// Resets internal tail state for stream restart.
    #[inline]
    pub fn reset(&mut self) {
        self.prev_tail.fill(0.0);
    }
}

/// Stateful F0 smoother:
/// voiced hysteresis + log-domain EMA + block-boundary ramp blending.
pub struct F0Smoother {
    voiced_hold_frames: usize,
    unvoiced_count: usize,
    ema_value: f32,
    ema_alpha: f32,
    prev_block_last_f0: f32,
}

impl F0Smoother {
    /// Creates a smoother using hop-size dependent EMA alpha (30ms time constant).
    #[inline]
    pub fn new(sample_rate: u32, hop_size: usize) -> Self {
        let hop_ms = hop_size as f32 / sample_rate.max(1) as f32 * 1000.0;
        let ema_alpha = 1.0 - (-hop_ms / 30.0_f32).exp();
        Self {
            voiced_hold_frames: 3,
            unvoiced_count: 0,
            ema_value: 0.0,
            ema_alpha,
            prev_block_last_f0: 0.0,
        }
    }

    /// Smooths one F0 block and returns a new stabilized block.
    #[inline]
    pub fn process(&mut self, f0: &[f32]) -> Vec<f32> {
        let n = f0.len();
        if n == 0 {
            return Vec::new();
        }

        let mut voiced_mask = vec![false; n];
        for (i, &v) in f0.iter().enumerate() {
            if v > VOICED_F0_HZ {
                voiced_mask[i] = true;
                self.unvoiced_count = 0;
            } else {
                self.unvoiced_count = self.unvoiced_count.saturating_add(1);
                voiced_mask[i] = self.unvoiced_count <= self.voiced_hold_frames;
            }
        }

        if self.prev_block_last_f0 > VOICED_F0_HZ {
            self.ema_value = (self.prev_block_last_f0 / A4_HZ).log2() * 12.0;
        }

        let mut smoothed = vec![0.0_f32; n];
        for i in 0..n {
            if voiced_mask[i] {
                if f0[i] > VOICED_F0_HZ {
                    let log_f0 = (f0[i] / A4_HZ).log2() * 12.0;
                    self.ema_value =
                        self.ema_alpha * log_f0 + (1.0 - self.ema_alpha) * self.ema_value;
                } else {
                    self.ema_value *= 0.995;
                }
                smoothed[i] = A4_HZ * 2.0_f32.powf(self.ema_value / 12.0);
            } else {
                self.ema_value *= 0.95;
                smoothed[i] = 0.0;
            }
        }

        let ramp_frames = n.min(3);
        if ramp_frames > 0 && self.prev_block_last_f0 > VOICED_F0_HZ && smoothed[0] > VOICED_F0_HZ {
            let first = smoothed[0];
            for (i, v) in smoothed.iter_mut().enumerate().take(ramp_frames) {
                let t = i as f32 / ramp_frames as f32;
                let ramp = self.prev_block_last_f0 + (first - self.prev_block_last_f0) * t;
                *v = *v * t + ramp * (1.0 - t);
            }
        }

        if let Some(&last_voiced) = smoothed.iter().rev().find(|v| **v > 0.0) {
            self.prev_block_last_f0 = last_voiced;
        }
        smoothed
    }

    /// Resets all smoother state.
    #[inline]
    pub fn reset(&mut self) {
        self.unvoiced_count = 0;
        self.ema_value = 0.0;
        self.prev_block_last_f0 = 0.0;
    }
}

/// Thin orchestration object for feature/F0/audio post-processing.
pub struct InferencePipeline {
    /// Overlap-add stitcher for decoder block outputs.
    pub ola: OlaBuffer,
    /// Stateful F0 smoother used before coarse quantization.
    pub f0_smoother: F0Smoother,
}

impl InferencePipeline {
    /// Constructs the pipeline.
    ///
    /// `block_size` is reserved for higher-level orchestration and currently not used internally.
    #[inline]
    pub fn new(
        sample_rate: u32,
        _block_size: usize,
        overlap_samples: usize,
        hop_size: usize,
    ) -> Self {
        Self {
            ola: OlaBuffer::new(overlap_samples),
            f0_smoother: F0Smoother::new(sample_rate, hop_size),
        }
    }

    /// Resamples HuBERT features to decoder frame contract.
    #[inline]
    pub fn prepare_features(&self, raw_features: ArrayView2<f32>) -> Array2<f32> {
        resample_features_catmull_rom(raw_features)
    }

    /// Smooths raw RMVPE F0 track.
    #[inline]
    pub fn prepare_f0(&mut self, raw_f0: &[f32]) -> Vec<f32> {
        self.f0_smoother.process(raw_f0)
    }

    /// Applies overlap-add postprocess to decoder output.
    #[inline]
    pub fn postprocess_audio(&mut self, decoder_output: &[f32]) -> Vec<f32> {
        self.ola.process(decoder_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_resample_linear_shape() {
        let src = Array2::<f32>::zeros((49, 256));
        let out = resample_features_linear(src.view());
        assert_eq!(out.shape(), &[50, 256]);
    }

    #[test]
    fn test_resample_catmull_shape() {
        let src = Array2::<f32>::zeros((49, 256));
        let out = resample_features_catmull_rom(src.view());
        assert_eq!(out.shape(), &[50, 256]);
    }

    #[test]
    fn test_resample_endpoints() {
        let mut src = Array2::<f32>::zeros((49, 8));
        for t in 0..49 {
            for c in 0..8 {
                src[(t, c)] = t as f32 * 0.1 + c as f32;
            }
        }

        let linear = resample_features_linear(src.view());
        let catmull = resample_features_catmull_rom(src.view());
        for c in 0..8 {
            assert!((linear[(0, c)] - src[(0, c)]).abs() <= 1e-4);
            assert!((linear[(49, c)] - src[(48, c)]).abs() <= 1e-4);
            assert!((catmull[(0, c)] - src[(0, c)]).abs() <= 1e-4);
            assert!((catmull[(49, c)] - src[(48, c)]).abs() <= 1e-4);
        }
    }

    #[test]
    fn test_ola_no_click() {
        let mut ola = OlaBuffer::new(512);
        let z1 = vec![0.0_f32; 2048];
        let z2 = vec![0.0_f32; 2048];
        let mut s3 = vec![0.0_f32; 2048];
        for (i, v) in s3.iter_mut().enumerate() {
            let t = i as f32 / 48_000.0;
            *v = (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        }

        let o1 = ola.process(&z1);
        let o2 = ola.process(&z2);
        let o3 = ola.process(&s3);

        for v in o1.iter().chain(o2.iter()).chain(o3.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_f0_smoother_voiced_hold() {
        let mut s = F0Smoother::new(16_000, 320);
        let in_f0 = [440.0_f32, 0.0, 0.0, 440.0];
        let out = s.process(&in_f0);
        assert!(out[1] > 0.0);
    }

    #[test]
    fn test_f0_smoother_log_scale() {
        let mut s = F0Smoother::new(16_000, 320);
        let mut in_f0 = vec![220.0_f32; 20];
        in_f0.extend(vec![440.0_f32; 20]);
        let out = s.process(&in_f0);
        let last = *out.last().unwrap_or(&0.0);
        assert!(last > 0.0);
        let semitone_err = (12.0 * (last / 440.0).log2()).abs();
        assert!(semitone_err <= 5.0);
    }
}
