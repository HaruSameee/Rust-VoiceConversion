use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft};

const DEFAULT_FFT_SIZE: usize = 512;
const DEFAULT_HOP_SIZE: usize = 256;
const DEFAULT_FLOOR: f32 = 0.01;
const SILENT_PROFILE_EMA: f32 = 0.05;

pub struct NoiseSuppress {
    fft_size: usize,
    hop_size: usize,
    noise_profile: Vec<f32>,
    is_learning: bool,
    learn_frames: usize,
    learn_count: usize,
    overshoot: f32,
    floor: f32,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    spectrum: Vec<Complex32>,
    overlap: Vec<f32>,
    norm: Vec<f32>,
}

impl NoiseSuppress {
    pub fn new(sample_rate: u32, learn_sec: f32, suppress_db: f32) -> Self {
        let fft_size = DEFAULT_FFT_SIZE;
        let hop_size = DEFAULT_HOP_SIZE;
        let frames_per_sec = sample_rate.max(1) as f32 / hop_size as f32;
        let learn_frames = (learn_sec.max(0.1) * frames_per_sec).ceil().max(1.0) as usize;
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);
        let window = build_sqrt_hann(fft_size);
        Self {
            fft_size,
            hop_size,
            noise_profile: vec![0.0; fft_size],
            is_learning: true,
            learn_frames,
            learn_count: 0,
            overshoot: overshoot_from_db(suppress_db),
            floor: DEFAULT_FLOOR,
            window,
            fft,
            ifft,
            spectrum: vec![Complex32::new(0.0, 0.0); fft_size],
            overlap: Vec::new(),
            norm: Vec::new(),
        }
    }

    pub fn process(&mut self, audio: &mut [f32], is_silent: bool) {
        if audio.is_empty() {
            return;
        }

        let frame_count = 1 + audio.len().saturating_sub(1) / self.hop_size;
        let required = audio.len().saturating_add(self.fft_size);
        self.overlap.clear();
        self.overlap.resize(required, 0.0);
        self.norm.clear();
        self.norm.resize(required, 0.0);

        let was_learning = self.is_learning;
        for frame_idx in 0..frame_count {
            let frame_start = frame_idx * self.hop_size;
            self.load_windowed_frame(audio, frame_start);
            self.fft.process(&mut self.spectrum);

            let mut power = vec![0.0_f32; self.fft_size];
            for (dst, bin) in power.iter_mut().zip(self.spectrum.iter()) {
                *dst = bin.norm_sqr();
            }

            if self.is_learning {
                self.accumulate_learning(&power);
                continue;
            }

            if is_silent {
                self.update_profile_ema(&power, SILENT_PROFILE_EMA);
            }

            self.apply_subtraction(&power);
            self.ifft.process(&mut self.spectrum);
            self.overlap_add(frame_start, audio.len());
        }

        if was_learning {
            return;
        }

        for (i, sample) in audio.iter_mut().enumerate() {
            let norm = self.norm[i];
            if norm > 1.0e-6 {
                *sample = self.overlap[i] / norm;
            }
        }
    }

    pub fn update_noise_profile(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
        }
        let frame_count = 1 + audio.len().saturating_sub(1) / self.hop_size;
        for frame_idx in 0..frame_count {
            let frame_start = frame_idx * self.hop_size;
            self.load_windowed_frame(audio, frame_start);
            self.fft.process(&mut self.spectrum);
            let mut power = vec![0.0_f32; self.fft_size];
            for (dst, bin) in power.iter_mut().zip(self.spectrum.iter()) {
                *dst = bin.norm_sqr();
            }
            if self.is_learning {
                self.accumulate_learning(&power);
            } else {
                self.update_profile_ema(&power, SILENT_PROFILE_EMA);
            }
        }
    }

    pub fn finish_learning(&mut self) {
        self.is_learning = false;
    }

    pub fn is_learning(&self) -> bool {
        self.is_learning
    }

    pub fn learn_count(&self) -> usize {
        self.learn_count
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    fn load_windowed_frame(&mut self, audio: &[f32], frame_start: usize) {
        for i in 0..self.fft_size {
            let src = audio.get(frame_start + i).copied().unwrap_or(0.0);
            self.spectrum[i] = Complex32::new(src * self.window[i], 0.0);
        }
    }

    fn accumulate_learning(&mut self, power: &[f32]) {
        let count = self.learn_count as f32;
        for (dst, &src) in self.noise_profile.iter_mut().zip(power.iter()) {
            *dst = (*dst * count + src) / (count + 1.0);
        }
        self.learn_count = self.learn_count.saturating_add(1);
        if self.learn_count >= self.learn_frames {
            self.finish_learning();
        }
    }

    fn update_profile_ema(&mut self, power: &[f32], alpha: f32) {
        for (dst, &src) in self.noise_profile.iter_mut().zip(power.iter()) {
            *dst = *dst * (1.0 - alpha) + src * alpha;
        }
    }

    fn apply_subtraction(&mut self, power: &[f32]) {
        for (i, bin) in self.spectrum.iter_mut().enumerate() {
            let power_in = power[i];
            if power_in <= 1.0e-12 {
                *bin = Complex32::new(0.0, 0.0);
                continue;
            }
            let power_out = (power_in - self.overshoot * self.noise_profile[i])
                .max(self.floor * power_in);
            let scale = (power_out / power_in).sqrt();
            *bin *= scale;
        }
    }

    fn overlap_add(&mut self, frame_start: usize, audio_len: usize) {
        let scale = 1.0 / self.fft_size as f32;
        let max_end = frame_start.saturating_add(self.fft_size).min(audio_len + self.fft_size);
        for dst in frame_start..max_end {
            let idx = dst - frame_start;
            let sample = self.spectrum[idx].re * scale * self.window[idx];
            self.overlap[dst] += sample;
            self.norm[dst] += self.window[idx] * self.window[idx];
        }
    }
}

fn build_sqrt_hann(size: usize) -> Vec<f32> {
    let denom = size.max(1) as f32;
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / denom;
            (0.5 - 0.5 * phase.cos()).sqrt()
        })
        .collect()
}

fn overshoot_from_db(suppress_db: f32) -> f32 {
    let db = suppress_db.clamp(-30.0, 0.0);
    let t = ((-db) - 6.0) / 12.0;
    0.5 + t.clamp(0.0, 1.0) * 0.2
}

#[cfg(test)]
mod tests {
    use super::NoiseSuppress;

    #[test]
    fn noise_suppress_preserves_length() {
        let mut ns = NoiseSuppress::new(48_000, 0.1, -12.0);
        ns.finish_learning();
        let mut audio = vec![0.0_f32; 2_400];
        audio[100] = 1.0;
        ns.process(&mut audio, false);
        assert_eq!(audio.len(), 2_400);
    }

    #[test]
    fn noise_suppress_finishes_learning() {
        let mut ns = NoiseSuppress::new(48_000, 0.01, -12.0);
        let mut audio = vec![0.0_f32; 4_800];
        ns.process(&mut audio, true);
        assert!(!ns.is_learning());
    }
}
