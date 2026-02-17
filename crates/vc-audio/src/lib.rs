use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use vc_core::{InferenceEngine, VoiceChanger};
use vc_signal::resample_linear_into;

const MIN_INFERENCE_BLOCK: usize = 8_192;
const OUTPUT_CROSSFADE_SAMPLES: usize = 256;

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

pub struct RealtimeAudioEngine {
    running: Arc<AtomicBool>,
    input_rms: Arc<AtomicU32>,
    input_peak: Arc<AtomicU32>,
    stream_thread: Option<thread::JoinHandle<()>>,
    worker_thread: Option<thread::JoinHandle<()>>,
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

    pub fn stop_and_abort(mut self) {
        self.stop();
        if let Some(handle) = self.stream_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

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
    eprintln!(
        "[vc-audio] input_rate={} output_rate={} input_ch={} output_ch={} model_rate={} block_size={}",
        input_rate, output_rate, input_channels, output_channels, model_sample_rate, block_size
    );

    let (input_tx, input_rx) = sync_channel::<Vec<f32>>(64);
    let (output_tx, output_rx) = sync_channel::<Vec<f32>>(64);

    let running = Arc::new(AtomicBool::new(true));
    let input_rms = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let input_peak = Arc::new(AtomicU32::new(0.0_f32.to_bits()));

    let input_data_fn = build_input_callback(Arc::clone(&running), input_channels, input_tx);
    let output_data_fn = build_output_callback(Arc::clone(&running), output_rx);

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

    let running_worker = Arc::clone(&running);
    let input_rms_worker = Arc::clone(&input_rms);
    let input_peak_worker = Arc::clone(&input_peak);
    let worker_thread = thread::spawn(move || {
        let mut level_meter = LevelMeter::new(0.92);
        worker_loop(
            &mut voice_changer,
            input_rx,
            output_tx,
            model_sample_rate,
            block_size,
            input_rate,
            output_rate,
            output_channels,
            &running_worker,
            &input_rms_worker,
            &input_peak_worker,
            &mut level_meter,
        );
    });

    let running_stream = Arc::clone(&running);
    let stream_thread = thread::spawn(move || {
        if input_stream.play().is_err() || output_stream.play().is_err() {
            running_stream.store(false, Ordering::Relaxed);
            return;
        }

        while running_stream.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(20));
        }

        drop(output_stream);
        drop(input_stream);
    });

    Ok(RealtimeAudioEngine {
        running,
        input_rms,
        input_peak,
        stream_thread: Some(stream_thread),
        worker_thread: Some(worker_thread),
    })
}

fn build_input_callback(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[f32], &cpal::InputCallbackInfo) + Send + 'static {
    let mut dropped = 0usize;
    move |data: &[f32], _: &cpal::InputCallbackInfo| {
        if !running.load(Ordering::Relaxed) {
            return;
        }
        let mono = downmix_to_mono(data, input_channels);
        match input_tx.try_send(mono) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                dropped += 1;
                if dropped % 64 == 0 {
                    eprintln!("[vc-audio] input queue full (dropped={dropped})");
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                eprintln!("[vc-audio] input queue disconnected");
                running.store(false, Ordering::Relaxed);
            }
        }
    }
}

fn build_output_callback(
    running: Arc<AtomicBool>,
    output_rx: Receiver<Vec<f32>>,
) -> impl FnMut(&mut [f32], &cpal::OutputCallbackInfo) + Send + 'static {
    let mut pending = VecDeque::<f32>::new();
    let mut underruns = 0usize;
    let mut last_sample = 0.0_f32;
    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        if !running.load(Ordering::Relaxed) {
            data.fill(0.0);
            return;
        }

        while pending.len() < data.len() {
            match output_rx.try_recv() {
                Ok(block) => {
                    pending.extend(block);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    eprintln!("[vc-audio] output queue disconnected");
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }
        }

        let mut callback_underrun = false;
        for sample in data {
            if let Some(v) = pending.pop_front() {
                *sample = v;
                last_sample = v;
            } else {
                // Smoothly decay towards silence to avoid periodic clicks/tones on underrun.
                last_sample *= 0.995;
                if last_sample.abs() < 1.0e-5 {
                    last_sample = 0.0;
                }
                *sample = last_sample;
                callback_underrun = true;
            }
        }
        if callback_underrun {
            underruns += 1;
            if underruns % 64 == 0 {
                eprintln!("[vc-audio] output underrun callbacks={underruns}");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_loop<E>(
    voice_changer: &mut VoiceChanger<E>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: SyncSender<Vec<f32>>,
    model_sample_rate: u32,
    block_size: usize,
    input_rate: u32,
    output_rate: u32,
    output_channels: usize,
    running: &Arc<AtomicBool>,
    input_rms: &Arc<AtomicU32>,
    input_peak: &Arc<AtomicU32>,
    level_meter: &mut LevelMeter,
) where
    E: InferenceEngine,
{
    let model_block_size = block_size.max(MIN_INFERENCE_BLOCK);
    let mut model_rate_buf = Vec::<f32>::new();
    let mut model_input_queue = VecDeque::<f32>::new();
    let mut model_block = Vec::<f32>::with_capacity(model_block_size);
    let mut output_rate_buf = Vec::<f32>::new();
    let mut previous_output_tail = Vec::<f32>::new();
    let mut processed_blocks: u64 = 0;
    let mut last_heartbeat = Instant::now();
    let block_budget = Duration::from_secs_f64(model_block_size as f64 / model_sample_rate as f64);
    eprintln!(
        "[vc-audio] inference_block_size={} io_block_size={} block_budget_ms={:.2}",
        model_block_size,
        block_size,
        block_budget.as_secs_f64() * 1000.0
    );
    // Prime output queue with short silence to hide startup gaps while first inference block is prepared.
    let prefill_frames = block_size.max(480) * output_channels * 4;
    let _ = output_tx.try_send(vec![0.0; prefill_frames]);

    while running.load(Ordering::Relaxed) {
        let mono = match input_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                running.store(false, Ordering::Relaxed);
                break;
            }
        };

        if input_rate != model_sample_rate {
            resample_linear_into(&mono, input_rate, model_sample_rate, &mut model_rate_buf);
            model_input_queue.extend(model_rate_buf.iter().copied());
        } else {
            model_input_queue.extend(mono.iter().copied());
        }

        while model_input_queue.len() >= model_block_size {
            model_block.clear();
            for _ in 0..model_block_size {
                if let Some(sample) = model_input_queue.pop_front() {
                    model_block.push(sample);
                }
            }

            let start = Instant::now();
            let model_out = match voice_changer.process_frame(&model_block) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[vc-audio] process_frame failed: {e}");
                    vec![0.0; model_block.len()]
                }
            };

            let output_mono: &[f32] = if model_sample_rate != output_rate {
                resample_linear_into(
                    &model_out,
                    model_sample_rate,
                    output_rate,
                    &mut output_rate_buf,
                );
                &output_rate_buf
            } else {
                &model_out
            };
            let mut smoothed_output = output_mono.to_vec();
            let crossfade = OUTPUT_CROSSFADE_SAMPLES
                .min(smoothed_output.len())
                .min(previous_output_tail.len());
            if crossfade > 0 {
                let prev_start = previous_output_tail.len() - crossfade;
                for i in 0..crossfade {
                    let a = previous_output_tail[prev_start + i];
                    let b = smoothed_output[i];
                    let t = (i + 1) as f32 / (crossfade + 1) as f32;
                    smoothed_output[i] = a * (1.0 - t) + b * t;
                }
            }
            let keep = OUTPUT_CROSSFADE_SAMPLES.min(smoothed_output.len());
            previous_output_tail.clear();
            if keep > 0 {
                previous_output_tail
                    .extend_from_slice(&smoothed_output[smoothed_output.len() - keep..]);
            }

            level_meter.push_block(&smoothed_output);
            input_rms.store(level_meter.rms().to_bits(), Ordering::Relaxed);
            input_peak.store(level_meter.peak().to_bits(), Ordering::Relaxed);

            let output_interleaved = upmix_from_mono(&smoothed_output, output_channels);
            match output_tx.try_send(output_interleaved) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    eprintln!("[vc-audio] output queue full");
                }
                Err(TrySendError::Disconnected(_)) => {
                    eprintln!("[vc-audio] output queue disconnected");
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }

            processed_blocks += 1;
            let elapsed = start.elapsed();
            if elapsed > block_budget {
                eprintln!(
                    "[vc-audio] slow block: elapsed={:.2}ms budget={:.2}ms queue={}",
                    elapsed.as_secs_f64() * 1000.0,
                    block_budget.as_secs_f64() * 1000.0,
                    model_input_queue.len()
                );
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "[vc-audio] heartbeat blocks={} queue={} rms={:.4} peak={:.4}",
                    processed_blocks,
                    model_input_queue.len(),
                    level_meter.rms(),
                    level_meter.peak()
                );
                last_heartbeat = Instant::now();
            }
        }
    }
    eprintln!("[vc-audio] worker loop stopped");
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
