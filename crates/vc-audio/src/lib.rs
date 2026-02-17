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

#[derive(Debug, Clone)]
pub struct AudioStreamOptions {
    pub model_sample_rate: u32,
    pub block_size: usize,
    pub input_device_name: Option<String>,
    pub output_device_name: Option<String>,
    pub extra_inference_ms: u32,
    pub response_threshold: f32,
    pub fade_in_ms: u32,
    pub fade_out_ms: u32,
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
    options: AudioStreamOptions,
) -> Result<RealtimeAudioEngine>
where
    E: InferenceEngine,
{
    let host = cpal::default_host();
    let model_sample_rate = options.model_sample_rate;
    let block_size = options.block_size;
    let extra_inference_ms = options.extra_inference_ms;
    let response_threshold = normalize_response_threshold(options.response_threshold);
    let fade_in_ms = options.fade_in_ms;
    let fade_out_ms = options.fade_out_ms;

    let input_device = find_input_device(&host, options.input_device_name.as_deref())?;
    let output_device = find_output_device(&host, options.output_device_name.as_deref())?;

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
    let extra_samples =
        ((model_sample_rate as f64) * (extra_inference_ms as f64 / 1000.0)).round() as usize;
    let inference_block_size = block_size.max(MIN_INFERENCE_BLOCK);
    let target_buffer_samples = inference_block_size.saturating_add(extra_samples);
    let needed_input_frames = ((target_buffer_samples as f64) * (input_rate as f64)
        / (model_sample_rate as f64))
        .round()
        .max(1.0) as usize;
    let est_callback_frames = ((input_rate as f64) * 0.01).round().max(64.0) as usize;
    let callbacks_per_infer = needed_input_frames.div_ceil(est_callback_frames);
    let input_queue_capacity = callbacks_per_infer.saturating_mul(2).max(64);
    let output_queue_capacity = callbacks_per_infer.saturating_mul(2).max(64);
    eprintln!(
        "[vc-audio] input_rate={} output_rate={} input_ch={} output_ch={} model_rate={} block_size={}",
        input_rate, output_rate, input_channels, output_channels, model_sample_rate, block_size
    );
    eprintln!(
        "[vc-audio] input_device='{}' output_device='{}' extra_inference_ms={} threshold={:.4} fade_in_ms={} fade_out_ms={}",
        input_device.name().unwrap_or_else(|_| "unknown-input".to_string()),
        output_device.name().unwrap_or_else(|_| "unknown-output".to_string()),
        extra_inference_ms,
        response_threshold,
        fade_in_ms,
        fade_out_ms
    );
    eprintln!(
        "[vc-audio] queue_capacity input={} output={} (est_cb={} frames, infer_block={} target_buffer={})",
        input_queue_capacity,
        output_queue_capacity,
        est_callback_frames,
        inference_block_size,
        target_buffer_samples
    );

    let (input_tx, input_rx) = sync_channel::<Vec<f32>>(input_queue_capacity);
    let (output_tx, output_rx) = sync_channel::<Vec<f32>>(output_queue_capacity);

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
            extra_inference_ms,
            response_threshold,
            fade_in_ms,
            fade_out_ms,
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

fn find_input_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    if let Some(name) = name.map(str::trim).filter(|s| !s.is_empty()) {
        let mut devices = host.input_devices()?;
        if let Some(dev) = devices.find(|d| {
            d.name()
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
        }) {
            return Ok(dev);
        }
        return Err(anyhow!("input device not found: '{name}'"));
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("default input device not found"))
}

fn find_output_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device> {
    if let Some(name) = name.map(str::trim).filter(|s| !s.is_empty()) {
        let mut devices = host.output_devices()?;
        if let Some(dev) = devices.find(|d| {
            d.name()
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
        }) {
            return Ok(dev);
        }
        return Err(anyhow!("output device not found: '{name}'"));
    }
    host.default_output_device()
        .ok_or_else(|| anyhow!("default output device not found"))
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
    extra_inference_ms: u32,
    response_threshold: f32,
    fade_in_ms: u32,
    fade_out_ms: u32,
    running: &Arc<AtomicBool>,
    input_rms: &Arc<AtomicU32>,
    input_peak: &Arc<AtomicU32>,
    level_meter: &mut LevelMeter,
) where
    E: InferenceEngine,
{
    let extra_samples =
        ((model_sample_rate as f64) * (extra_inference_ms as f64 / 1000.0)).round() as usize;
    let io_block_size = block_size.max(1);
    let inference_block_size = io_block_size.max(MIN_INFERENCE_BLOCK);
    let target_buffer_samples = inference_block_size.saturating_add(extra_samples);
    let mut model_rate_buf = Vec::<f32>::new();
    let mut model_input_queue = VecDeque::<f32>::new();
    let mut model_block = Vec::<f32>::with_capacity(inference_block_size);
    let mut output_rate_buf = Vec::<f32>::new();
    let mut previous_output_tail = Vec::<f32>::new();
    let mut processed_blocks: u64 = 0;
    let mut silence_skips: u64 = 0;
    let mut last_heartbeat = Instant::now();
    let block_budget = Duration::from_secs_f64(io_block_size as f64 / model_sample_rate as f64);
    let block_duration_sec = io_block_size as f32 / model_sample_rate.max(1) as f32;
    let silence_hold_blocks = ((fade_out_ms as f32 / 1000.0) / block_duration_sec)
        .ceil()
        .max(1.0) as usize;
    let mut silence_hold_remaining = 0usize;
    let mut gate = ReactionGate::new(
        response_threshold,
        fade_in_ms,
        fade_out_ms,
        model_sample_rate,
    );
    eprintln!(
        "[vc-audio] inference_block_size={} io_block_size={} target_buffer_samples={} block_budget_ms={:.2}",
        inference_block_size,
        io_block_size,
        target_buffer_samples,
        block_budget.as_secs_f64() * 1000.0
    );
    // Prime output queue to cover startup buffering and first inference warm-up.
    let prefill_model_samples = if model_sample_rate != output_rate {
        ((target_buffer_samples.saturating_add(io_block_size) as f64) * output_rate as f64
            / model_sample_rate as f64)
            .round()
            .max(1.0) as usize
    } else {
        target_buffer_samples.saturating_add(io_block_size)
    };
    let prefill_frames = prefill_model_samples * output_channels * 2;
    let _ = output_tx.try_send(vec![0.0; prefill_frames]);

    while running.load(Ordering::Relaxed) {
        let mut mono = match input_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                running.store(false, Ordering::Relaxed);
                break;
            }
        };
        gate.process_block(&mut mono);

        if input_rate != model_sample_rate {
            resample_linear_into(&mono, input_rate, model_sample_rate, &mut model_rate_buf);
            model_input_queue.extend(model_rate_buf.iter().copied());
        } else {
            model_input_queue.extend(mono.iter().copied());
        }

        while model_input_queue.len() >= target_buffer_samples {
            model_block.clear();
            model_block.extend(model_input_queue.iter().take(inference_block_size).copied());
            if model_block.len() < io_block_size {
                break;
            }
            let step_len = io_block_size.min(model_block.len());
            let step_input = &model_block[..step_len];

            let mut block_peak = 0.0_f32;
            let mut sum_sq = 0.0_f32;
            for s in step_input {
                let a = s.abs();
                block_peak = block_peak.max(a);
                sum_sq += s * s;
            }
            let block_rms = (sum_sq / step_input.len().max(1) as f32).sqrt();
            let quiet_block = response_threshold > 0.0
                && block_rms < response_threshold
                && block_peak < (response_threshold * 4.0);
            if quiet_block {
                if silence_hold_remaining > 0 {
                    silence_hold_remaining -= 1;
                } else {
                    // Keep inferring to avoid hard dropouts, but track sustained-quiet blocks.
                    silence_skips += 1;
                }
            } else if response_threshold > 0.0 {
                silence_hold_remaining = silence_hold_blocks;
            }

            let start = Instant::now();
            let model_out = match voice_changer.process_frame(&model_block) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[vc-audio] process_frame failed: {e}");
                    vec![0.0; step_len]
                }
            };

            let output_source: Vec<f32> = if model_out.len() >= step_len {
                model_out[..step_len].to_vec()
            } else {
                let mut padded = model_out;
                padded.resize(step_len, 0.0);
                padded
            };
            let output_mono: &[f32] = if model_sample_rate != output_rate {
                resample_linear_into(
                    &output_source,
                    model_sample_rate,
                    output_rate,
                    &mut output_rate_buf,
                );
                &output_rate_buf
            } else {
                &output_source
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
            if response_threshold > 0.0 {
                let gate_ratio = (block_rms / response_threshold).clamp(0.0, 1.0);
                if gate_ratio < 1.0 {
                    let gain = gate_ratio * gate_ratio;
                    for s in &mut smoothed_output {
                        *s *= gain;
                    }
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
            for _ in 0..io_block_size {
                let _ = model_input_queue.pop_front();
            }
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
                    "[vc-audio] heartbeat blocks={} queue={} rms={:.4} peak={:.4} silence_skips={}",
                    processed_blocks,
                    model_input_queue.len(),
                    level_meter.rms(),
                    level_meter.peak(),
                    silence_skips
                );
                last_heartbeat = Instant::now();
            }
        }
    }
    eprintln!("[vc-audio] worker loop stopped");
}

#[derive(Debug, Clone)]
struct ReactionGate {
    threshold: f32,
    attack_step: f32,
    release_step: f32,
    env: f32,
}

impl ReactionGate {
    fn new(threshold: f32, fade_in_ms: u32, fade_out_ms: u32, sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let attack_samples = ((fade_in_ms as f32 / 1000.0) * sr).max(1.0);
        let release_samples = ((fade_out_ms as f32 / 1000.0) * sr).max(1.0);
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            attack_step: (1.0 / attack_samples).clamp(0.0001, 1.0),
            release_step: (1.0 / release_samples).clamp(0.00001, 1.0),
            env: 0.0,
        }
    }

    fn process_block(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        let peak = samples
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, |acc, v| acc.max(v));
        let open = peak >= self.threshold;
        for s in samples {
            if open {
                self.env = (self.env + self.attack_step).min(1.0);
            } else {
                self.env = (self.env - self.release_step).max(0.0);
            }
            *s *= self.env;
        }
    }
}

fn normalize_response_threshold(raw: f32) -> f32 {
    if !raw.is_finite() {
        return 0.02;
    }
    if raw < 0.0 {
        // Support dBFS style input used by common VC tools (e.g. -50 dB).
        let db = raw.clamp(-90.0, 0.0);
        return 10.0_f32.powf(db / 20.0);
    }
    raw.clamp(0.0, 1.0)
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
