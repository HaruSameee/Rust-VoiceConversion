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
use vc_signal::{resample_hq_into, HqResampler};

const MIN_INFERENCE_BLOCK: usize = 8_192;
const MIN_OUTPUT_CROSSFADE_SAMPLES: usize = 256;
const MAX_OUTPUT_CROSSFADE_SAMPLES: usize = 1_024;
const SOLA_SEARCH_MS: f32 = 10.0;

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
    pub allow_process_window_grow: bool,
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
    let allow_process_window_grow = options.allow_process_window_grow;
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
    let sola_search_output = sola_search_samples(output_rate);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let inference_block_size = block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
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
        "[vc-audio] input_device='{}' output_device='{}' extra_inference_ms={} threshold={:.4} fade_in_ms={} fade_out_ms={} allow_window_grow={}",
        input_device.name().unwrap_or_else(|_| "unknown-input".to_string()),
        output_device.name().unwrap_or_else(|_| "unknown-output".to_string()),
        extra_inference_ms,
        response_threshold,
        fade_in_ms,
        fade_out_ms,
        allow_process_window_grow
    );
    eprintln!(
        "[vc-audio] queue_capacity input={} output={} (est_cb={} frames, infer_block={} target_buffer={})",
        input_queue_capacity,
        output_queue_capacity,
        est_callback_frames,
        inference_block_size,
        target_buffer_samples
    );
    // Warm up the model before starting audio streams to avoid first-block stalls
    // (notably large on GPU providers like DirectML during graph compilation).
    let warmup_input = vec![0.0_f32; inference_block_size.max(MIN_INFERENCE_BLOCK)];
    let warmup_start = Instant::now();
    match voice_changer.process_frame(&warmup_input) {
        Ok(out) => {
            eprintln!(
                "[vc-audio] warmup done: in={} out={} elapsed={:.2}ms",
                warmup_input.len(),
                out.len(),
                warmup_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Err(err) => {
            eprintln!(
                "[vc-audio] warmup failed (continuing): {} elapsed={:.2}ms",
                err,
                warmup_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

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
            allow_process_window_grow,
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
    allow_process_window_grow: bool,
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
    let output_step_samples =
        map_model_samples_to_output(io_block_size, model_sample_rate, output_rate);
    let crossfade_samples = default_output_crossfade_samples(output_rate).min(output_step_samples);
    let sola_search_output = sola_search_samples(output_rate);
    let edge_guard_samples = crossfade_samples.max(sola_search_output / 2);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let inference_block_size = io_block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    let target_buffer_samples = inference_block_size.saturating_add(extra_samples);
    let min_process_window_samples = inference_block_size;
    let process_window_cap = if allow_process_window_grow {
        target_buffer_samples
    } else {
        min_process_window_samples
    };
    let mut process_window_samples = min_process_window_samples;
    let mut smoothed_elapsed_ms = 0.0_f64;
    let mut model_rate_buf = Vec::<f32>::new();
    let mut model_input_queue = VecDeque::<f32>::new();
    let mut model_block = Vec::<f32>::with_capacity(inference_block_size);
    let mut output_rate_buf = Vec::<f32>::new();
    let mut previous_output_tail = Vec::<f32>::new();
    let input_to_model_resampler = HqResampler::new(input_rate, model_sample_rate);
    let model_to_output_resampler = HqResampler::new(model_sample_rate, output_rate);
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
        "[vc-audio] inference_block_size={} io_block_size={} target_buffer_samples={} process_window_samples={} block_budget_ms={:.2} sola_search_out={} edge_guard_out={}",
        inference_block_size,
        io_block_size,
        target_buffer_samples,
        process_window_samples,
        block_budget.as_secs_f64() * 1000.0,
        sola_search_output,
        edge_guard_samples
    );
    if target_buffer_samples > process_window_samples {
        if allow_process_window_grow {
            eprintln!(
                "[vc-audio] process_window adaptive start: requested={} start={} (auto-grow when realtime budget allows)",
                target_buffer_samples, process_window_samples
            );
        } else {
            eprintln!(
                "[vc-audio] process_window fixed: requested={} using={} (grow disabled for stability)",
                target_buffer_samples, process_window_samples
            );
        }
    }
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

        if !input_to_model_resampler.is_identity() {
            resample_hq_into(&input_to_model_resampler, &mono, &mut model_rate_buf);
            model_input_queue.extend(model_rate_buf.iter().copied());
        } else {
            model_input_queue.extend(mono.iter().copied());
        }

        while model_input_queue.len() >= process_window_samples {
            model_block.clear();
            model_block.extend(
                model_input_queue
                    .iter()
                    .take(process_window_samples)
                    .copied(),
            );
            if model_block.len() < io_block_size {
                break;
            }
            let step_len = io_block_size.min(model_block.len());
            let step_start = model_block.len().saturating_sub(step_len);
            let step_input = &model_block[step_start..];

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

            let output_source: Vec<f32> = if model_out.len() >= process_window_samples {
                model_out[..process_window_samples].to_vec()
            } else {
                let mut padded = model_out;
                padded.resize(process_window_samples, 0.0);
                padded
            };
            let output_mono: &[f32] = if model_sample_rate != output_rate {
                resample_hq_into(&model_to_output_resampler, &output_source, &mut output_rate_buf);
                &output_rate_buf
            } else {
                &output_source
            };
            let expected_len = output_step_samples.min(output_mono.len().max(1));
            // Avoid using the very end of each block where non-causal generators are often unstable.
            let max_guard = output_mono.len().saturating_sub(expected_len);
            let applied_guard = edge_guard_samples.min(max_guard);
            let anchor_end = output_mono.len().saturating_sub(applied_guard);
            let tail_span = expected_len.saturating_add(sola_search_output);
            let tail_start = anchor_end.saturating_sub(tail_span);
            let tail = &output_mono[tail_start..anchor_end];
            let search_limit = tail
                .len()
                .saturating_sub(expected_len)
                .min(sola_search_output);
            let crossfade_for_search = crossfade_samples
                .min(previous_output_tail.len())
                .min(tail.len());
            let mut offset = 0usize;
            if search_limit > 0 && crossfade_for_search >= 16 {
                offset = find_best_sola_offset(
                    &previous_output_tail,
                    tail,
                    crossfade_for_search,
                    search_limit,
                );
            }
            let mut smoothed_output =
                tail[offset..(offset + expected_len).min(tail.len())].to_vec();
            if smoothed_output.len() < expected_len {
                let pad = smoothed_output.last().copied().unwrap_or(0.0);
                smoothed_output.resize(expected_len, pad);
            }
            let crossfade = crossfade_samples
                .min(smoothed_output.len())
                .min(previous_output_tail.len());
            if crossfade > 0 {
                let prev_start = previous_output_tail.len() - crossfade;
                let inv = 1.0_f32 / (crossfade + 1) as f32;
                for i in 0..crossfade {
                    let a = previous_output_tail[prev_start + i];
                    let b = smoothed_output[i];
                    let t = (i + 1) as f32 * inv;
                    // Equal-power curve (sin/cos) avoids perceived dip at segment seams.
                    let theta = t * std::f32::consts::FRAC_PI_2;
                    let (w_b, w_a) = theta.sin_cos();
                    smoothed_output[i] = a * w_a + b * w_b;
                }
            }
            let keep = crossfade_samples.min(smoothed_output.len());
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
            let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
            let budget_ms = block_budget.as_secs_f64() * 1000.0;
            smoothed_elapsed_ms = if smoothed_elapsed_ms <= 0.0 {
                elapsed_ms
            } else {
                smoothed_elapsed_ms * 0.9 + elapsed_ms * 0.1
            };
            if process_window_samples > min_process_window_samples
                && smoothed_elapsed_ms > budget_ms * 1.03
            {
                let span = process_window_samples - min_process_window_samples;
                let shrink = ((span as f32) * 0.30).ceil().max(128.0) as usize;
                let next = process_window_samples
                    .saturating_sub(shrink)
                    .max(min_process_window_samples);
                if next != process_window_samples {
                    process_window_samples = next;
                    eprintln!(
                        "[vc-audio] process_window auto-shrink: now={} requested={} elapsed={:.2}ms budget={:.2}ms",
                        process_window_samples, target_buffer_samples, smoothed_elapsed_ms, budget_ms
                    );
                }
            } else if allow_process_window_grow
                && process_window_samples < process_window_cap
                && smoothed_elapsed_ms < budget_ms * 0.78
            {
                let span = process_window_cap - process_window_samples;
                let grow = ((span as f32) * 0.12).ceil().max(64.0) as usize;
                let next = process_window_samples
                    .saturating_add(grow)
                    .min(process_window_cap);
                if next != process_window_samples {
                    process_window_samples = next;
                    eprintln!(
                        "[vc-audio] process_window auto-grow: now={} cap={} elapsed={:.2}ms budget={:.2}ms",
                        process_window_samples, process_window_cap, smoothed_elapsed_ms, budget_ms
                    );
                }
            }
            if elapsed > block_budget {
                eprintln!(
                    "[vc-audio] slow block: elapsed={:.2}ms budget={:.2}ms queue={}",
                    elapsed_ms,
                    budget_ms,
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
    threshold_open: f32,
    threshold_close: f32,
    attack_step: f32,
    release_step: f32,
    hold_samples: usize,
    hold_remaining: usize,
    env: f32,
}

impl ReactionGate {
    fn new(threshold: f32, fade_in_ms: u32, fade_out_ms: u32, sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let attack_samples = ((fade_in_ms as f32 / 1000.0) * sr).max(1.0);
        let release_samples = ((fade_out_ms as f32 / 1000.0) * sr).max(1.0);
        let open = threshold.clamp(0.0, 1.0);
        let close = (open * 0.7).clamp(0.0, open);
        let hold_samples = ((fade_out_ms as f32 / 1000.0) * sr * 0.5).round().max(1.0) as usize;
        Self {
            threshold_open: open,
            threshold_close: close,
            attack_step: (1.0 / attack_samples).clamp(0.0001, 1.0),
            release_step: (1.0 / release_samples).clamp(0.00001, 1.0),
            hold_samples,
            hold_remaining: 0,
            env: 0.0,
        }
    }

    fn process_block(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        let mut sum_sq = 0.0_f32;
        let peak = samples
            .iter()
            .map(|v| {
                let a = v.abs();
                sum_sq += v * v;
                a
            })
            .fold(0.0_f32, |acc, v| acc.max(v));
        let rms = (sum_sq / samples.len() as f32).sqrt();
        let open_cond = rms >= self.threshold_open || peak >= self.threshold_open * 4.0;
        let hold_cond = rms >= self.threshold_close || peak >= self.threshold_close * 4.0;
        let open = if open_cond {
            self.hold_remaining = self.hold_samples;
            true
        } else if hold_cond {
            self.hold_remaining = self.hold_samples;
            true
        } else if self.hold_remaining > 0 {
            self.hold_remaining = self.hold_remaining.saturating_sub(samples.len());
            true
        } else {
            false
        };
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

fn map_model_samples_to_output(model_samples: usize, model_rate: u32, output_rate: u32) -> usize {
    if model_rate == 0 || output_rate == 0 {
        return model_samples.max(1);
    }
    ((model_samples as f64) * (output_rate as f64) / (model_rate as f64))
        .round()
        .max(1.0) as usize
}

fn map_output_samples_to_model(output_samples: usize, model_rate: u32, output_rate: u32) -> usize {
    if model_rate == 0 || output_rate == 0 {
        return output_samples.max(1);
    }
    ((output_samples as f64) * (model_rate as f64) / (output_rate as f64))
        .ceil()
        .max(1.0) as usize
}

fn sola_search_samples(output_rate: u32) -> usize {
    ((output_rate as f32) * (SOLA_SEARCH_MS / 1000.0))
        .round()
        .clamp(128.0, 1_024.0) as usize
}

fn default_output_crossfade_samples(output_rate: u32) -> usize {
    ((output_rate as f32) * 0.01).round().clamp(
        MIN_OUTPUT_CROSSFADE_SAMPLES as f32,
        MAX_OUTPUT_CROSSFADE_SAMPLES as f32,
    ) as usize
}

fn find_best_sola_offset(
    previous_tail: &[f32],
    current: &[f32],
    crossfade: usize,
    search_limit: usize,
) -> usize {
    if crossfade == 0 || previous_tail.len() < crossfade || current.len() < crossfade {
        return 0;
    }
    let prev = &previous_tail[previous_tail.len() - crossfade..];
    let mut prev_energy = 0.0_f32;
    for &v in prev {
        prev_energy += v * v;
    }
    if prev_energy <= 1.0e-12 {
        return 0;
    }

    let mut best_offset = 0usize;
    let mut best_score = f32::MIN;
    for offset in 0..=search_limit {
        if offset + crossfade > current.len() {
            break;
        }
        let seg = &current[offset..offset + crossfade];
        let mut dot = 0.0_f32;
        let mut seg_energy = 0.0_f32;
        for i in 0..crossfade {
            dot += prev[i] * seg[i];
            seg_energy += seg[i] * seg[i];
        }
        if seg_energy <= 1.0e-12 {
            continue;
        }
        let score = dot / (prev_energy * seg_energy).sqrt().max(1.0e-8);
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }
    best_offset
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
