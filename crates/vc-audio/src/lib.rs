use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use vc_core::{InferenceEngine, VoiceChanger};
use vc_signal::resample_linear;

pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()?
        .map(|d| d.name().unwrap_or_else(|_| "unknown-input".to_string()))
        .collect();
    Ok(devices)
}

pub fn list_output_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host
        .output_devices()?
        .map(|d| d.name().unwrap_or_else(|_| "unknown-output".to_string()))
        .collect();
    Ok(devices)
}

pub fn default_sample_rate() -> Result<u32> {
    let host = cpal::default_host();
    let input = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let config = input.default_input_config()?;
    Ok(config.sample_rate().0)
}

/// 実行中のリアルタイム音声エンジンへのハンドル。
///
/// レベル取得と停止制御だけを公開し、ストリームの詳細は内部に隠します。
pub struct RealtimeAudioEngine {
    running: Arc<AtomicBool>,
    input_rms: Arc<AtomicU32>,
    input_peak: Arc<AtomicU32>,
    handle: tokio::task::JoinHandle<()>,
}

impl RealtimeAudioEngine {
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn levels(&self) -> (f32, f32) {
        (
            f32::from_bits(self.input_rms.load(Ordering::Relaxed)),
            f32::from_bits(self.input_peak.load(Ordering::Relaxed)),
        )
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    pub fn stop_and_abort(self) {
        self.stop();
        self.handle.abort();
    }
}

/// `VoiceChanger` を `cpal` 入出力に接続して起動する。
///
/// 処理の流れ:
/// 1. 入力をモノラル化
/// 2. 必要ならモデルの想定レートへリサンプル
/// 3. `VoiceChanger::process_frame`
/// 4. 出力デバイスのレートへ戻して再生
pub fn spawn_voice_changer_stream<E>(
    mut voice_changer: VoiceChanger<E>,
    model_sample_rate: u32,
    block_size: usize,
) -> Result<RealtimeAudioEngine>
where
    E: InferenceEngine,
{
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("default input device not found"))?;
    let output_device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("default output device not found"))?;

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    if input_config.sample_format() != cpal::SampleFormat::F32
        || output_config.sample_format() != cpal::SampleFormat::F32
    {
        return Err(anyhow!(
            "only f32 input/output sample format is supported in this prototype"
        ));
    }

    let input_stream_config: cpal::StreamConfig = input_config.clone().into();
    let output_stream_config: cpal::StreamConfig = output_config.clone().into();
    let input_rate = input_stream_config.sample_rate.0;
    let output_rate = output_stream_config.sample_rate.0;
    let input_channels = input_stream_config.channels as usize;
    let output_channels = output_stream_config.channels as usize;

    let latency_samples = (block_size.max(1) * output_channels * 12).max(output_channels * 1024);
    let ring = HeapRb::<f32>::new(latency_samples * 2);
    let (mut producer, mut consumer) = ring.split();
    for _ in 0..latency_samples {
        let _ = producer.try_push(0.0);
    }

    let running = Arc::new(AtomicBool::new(true));
    let input_rms = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let input_peak = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let running_in = Arc::clone(&running);
    let input_rms_in = Arc::clone(&input_rms);
    let input_peak_in = Arc::clone(&input_peak);
    let mut level_meter = LevelMeter::new(0.92);

    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        if !running_in.load(Ordering::Relaxed) {
            return;
        }

        let mono = downmix_to_mono(data, input_channels);
        let model_in = if input_rate != model_sample_rate {
            resample_linear(&mono, input_rate, model_sample_rate)
        } else {
            mono
        };

        let model_out = match voice_changer.process_frame(&model_in) {
            Ok(v) => v,
            Err(_) => model_in,
        };

        let output_mono = if model_sample_rate != output_rate {
            resample_linear(&model_out, model_sample_rate, output_rate)
        } else {
            model_out
        };
        level_meter.push_block(&output_mono);
        input_rms_in.store(level_meter.rms().to_bits(), Ordering::Relaxed);
        input_peak_in.store(level_meter.peak().to_bits(), Ordering::Relaxed);

        let output_interleaved = upmix_from_mono(&output_mono, output_channels);
        for sample in output_interleaved {
            if producer.try_push(sample).is_err() {
                break;
            }
        }
    };

    let running_out = Arc::clone(&running);
    let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        if !running_out.load(Ordering::Relaxed) {
            data.fill(0.0);
            return;
        }
        for sample in data {
            *sample = consumer.try_pop().unwrap_or(0.0);
        }
    };

    let input_stream = input_device.build_input_stream(
        &input_stream_config,
        input_data_fn,
        |err| eprintln!("input stream error: {err}"),
        None,
    )?;
    let output_stream = output_device.build_output_stream(
        &output_stream_config,
        output_data_fn,
        |err| eprintln!("output stream error: {err}"),
        None,
    )?;

    let running_loop = Arc::clone(&running);
    let handle = tokio::spawn(async move {
        if input_stream.play().is_err() || output_stream.play().is_err() {
            running_loop.store(false, Ordering::Relaxed);
            return;
        }

        while running_loop.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(30)).await;
        }

        drop(output_stream);
        drop(input_stream);
    });

    Ok(RealtimeAudioEngine {
        running,
        input_rms,
        input_peak,
        handle,
    })
}

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }

    let frames = data.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for frame_idx in 0..frames {
        let start = frame_idx * channels;
        let mut sum = 0.0_f32;
        for ch in 0..channels {
            sum += data[start + ch];
        }
        out.push(sum / channels as f32);
    }
    out
}

fn upmix_from_mono(mono: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return mono.to_vec();
    }

    let mut out = Vec::with_capacity(mono.len() * channels);
    for &sample in mono {
        for _ in 0..channels {
            out.push(sample);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct LevelMeter {
    smoothing: f32,
    rms: f32,
    peak: f32,
}

impl LevelMeter {
    pub fn new(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 0.9999),
            rms: 0.0,
            peak: 0.0,
        }
    }

    pub fn push_block(&mut self, block: &[f32]) {
        if block.is_empty() {
            return;
        }

        let mut sum = 0.0_f32;
        let mut peak = 0.0_f32;
        for s in block {
            let x = s.abs();
            sum += x * x;
            peak = peak.max(x);
        }
        let rms_now = (sum / block.len() as f32).sqrt();
        self.rms = self.smoothing * self.rms + (1.0 - self.smoothing) * rms_now;
        self.peak = self.smoothing * self.peak + (1.0 - self.smoothing) * peak;
    }

    pub fn rms(&self) -> f32 {
        self.rms
    }

    pub fn peak(&self) -> f32 {
        self.peak
    }
}
