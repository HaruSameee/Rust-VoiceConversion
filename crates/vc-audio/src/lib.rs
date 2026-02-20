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
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use vc_core::{InferenceEngine, VoiceChanger};
use vc_signal::{resample_hq_into, HqResampler};

const MIN_INFERENCE_BLOCK: usize = 8_192;
const MIN_OUTPUT_CROSSFADE_SAMPLES: usize = 256;
const MAX_OUTPUT_CROSSFADE_SAMPLES: usize = 1_024;
const SOLA_SEARCH_MS: f32 = 10.0;
const OLA_MIN_OVERLAP_RATIO: f32 = 0.25;
const OLA_MAX_OVERLAP_RATIO: f32 = 0.50;
const BLOCK_ALIGN_SAMPLES: usize = 256;
const WARMUP_BUDGET_HEADROOM: f64 = 1.25;
const MAX_AUTO_BLOCK_SIZE: usize = 96_000;
const MAX_WARMUP_BLOCK_CAP: usize = 16_384;
const HUBERT_WINDOW_SAMPLES: usize = 16_000;
const OUTPUT_SLICE_AUDIT_BLOCKS: u64 = 30;

/// Returns the minimum `extra_inference_ms` value that avoids buffer deadlock.
///
/// Formula: ceil((process_window - block_size) / (sample_rate / 1000))
///
/// Add the actual inference latency on top of this value for safe operation.
pub fn min_extra_inference_ms(process_window: usize, block_size: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    let samples_needed = process_window.saturating_sub(block_size);
    if samples_needed == 0 {
        return 0;
    }
    let ms_per_sample = 1000.0 / sample_rate as f64;
    (samples_needed as f64 * ms_per_sample).ceil() as u32
}

/// Validates that the audio buffer sizing is physically consistent.
///
/// # Panics
/// Panics if target_buffer < process_window, which would cause the engine
/// to deadlock waiting for samples that can never accumulate.
pub fn validate_buffer_contract(
    extra_inference_ms: u32,
    block_size: usize,
    process_window: usize,
    sample_rate: u32,
) {
    let target_buffer = (extra_inference_ms as usize)
        .saturating_mul(sample_rate as usize)
        .saturating_div(1000)
        .saturating_add(block_size);
    let min_extra_ms = min_extra_inference_ms(process_window, block_size, sample_rate) as f64;

    assert!(
        target_buffer >= process_window,
        "[vc-audio] FATAL: target_buffer({}) < process_window({}).\n\
         This causes a deadlock: the engine waits for {} samples that \
         can never accumulate.\n\
         Minimum extra_inference_ms required: {:.1}ms (recommended: {:.0}ms)",
        target_buffer,
        process_window,
        process_window.saturating_sub(target_buffer),
        min_extra_ms,
        min_extra_ms.ceil() + 80.0,
    );
}

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
    worker_control_tx: Option<SyncSender<WorkerControl>>,
    worker_shutdown_ack_rx: Option<Receiver<()>>,
}

#[derive(Debug, Clone)]
pub struct AudioStreamOptions {
    pub model_sample_rate: u32,
    pub block_size: usize,
    pub input_device_name: Option<String>,
    pub output_device_name: Option<String>,
    /// Enables adaptive process-window growth.
    ///
    /// Default runtime policy is `false` for strict geometry stability.
    pub allow_process_window_grow: bool,
    /// Milliseconds of audio buffered ahead of inference.
    ///
    /// Must satisfy:
    ///   extra_inference_ms >= ceil((process_window - block_size) / (sr / 1000))
    ///
    /// With process_window=16000, block_size=8192, sr=48000:
    ///   minimum = 163ms
    ///
    /// Recommended = minimum + actual_inference_ms + 20ms jitter margin.
    pub extra_inference_ms: u32,
    pub response_threshold: f32,
    pub fade_in_ms: u32,
    pub fade_out_ms: u32,
    pub output_tail_offset_ms: u32,
}

#[derive(Debug, Clone, Copy)]
enum WorkerControl {
    PrepareShutdown,
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

    fn prepare_inference_shutdown(&mut self) {
        if let Some(tx) = self.worker_control_tx.as_ref() {
            let _ = tx.try_send(WorkerControl::PrepareShutdown);
        }
        if let Some(rx) = self.worker_shutdown_ack_rx.as_ref() {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "[vc-audio] warning: timed out waiting for inference shutdown acknowledgement"
                    );
                }
            }
        }
        self.worker_control_tx = None;
        self.worker_shutdown_ack_rx = None;
    }

    fn join_threads(&mut self) {
        let current = thread::current().id();
        if let Some(handle) = self.stream_thread.take() {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
        if let Some(handle) = self.worker_thread.take() {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
    }

    pub fn stop_and_abort(mut self) {
        self.prepare_inference_shutdown();
        self.stop();
        self.join_threads();
    }
}

impl Drop for RealtimeAudioEngine {
    fn drop(&mut self) {
        self.prepare_inference_shutdown();
        self.stop();
        self.join_threads();
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
    let model_sample_rate = options.model_sample_rate.max(1);
    let requested_block_size = options.block_size.max(1);
    let allow_process_window_grow = options.allow_process_window_grow;
    let extra_inference_ms = options.extra_inference_ms;
    let response_threshold = normalize_response_threshold(options.response_threshold);
    let fade_in_ms = options.fade_in_ms;
    let fade_out_ms = options.fade_out_ms;
    let output_tail_offset_ms = options.output_tail_offset_ms;

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
    let sola_search_output = sola_search_samples(output_rate);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let warmup_inference_block_size = requested_block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    eprintln!(
        "[vc-audio] input_rate={} output_rate={} input_ch={} output_ch={} model_rate={} block_size_requested={}",
        input_rate,
        output_rate,
        input_channels,
        output_channels,
        model_sample_rate,
        requested_block_size
    );
    eprintln!(
        "[vc-audio] input_device='{}' output_device='{}' extra_inference_ms={} threshold={:.4} fade_in_ms={} fade_out_ms={} tail_offset_ms={} allow_window_grow={}",
        input_device.name().unwrap_or_else(|_| "unknown-input".to_string()),
        output_device.name().unwrap_or_else(|_| "unknown-output".to_string()),
        extra_inference_ms,
        response_threshold,
        fade_in_ms,
        fade_out_ms,
        output_tail_offset_ms,
        allow_process_window_grow
    );
    // Warm up the model before starting audio streams to avoid first-block stalls
    // (notably large on GPU providers like DirectML during graph compilation).
    let warmup_input = vec![0.0_f32; warmup_inference_block_size.max(MIN_INFERENCE_BLOCK)];
    let warmup_start = Instant::now();
    let warmup_elapsed_ms = match voice_changer.process_frame(&warmup_input) {
        Ok(out) => {
            let elapsed_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[vc-audio] warmup done: in={} out={} elapsed={:.2}ms",
                warmup_input.len(),
                out.len(),
                elapsed_ms
            );
            elapsed_ms
        }
        Err(err) => {
            let elapsed_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[vc-audio] warmup failed (continuing): {} elapsed={:.2}ms",
                err, elapsed_ms
            );
            elapsed_ms
        }
    };

    let block_size = recommended_block_size_from_warmup(
        requested_block_size,
        model_sample_rate,
        warmup_elapsed_ms,
    );
    if block_size != requested_block_size {
        eprintln!(
            "[vc-audio] block_size adjusted for realtime headroom: requested={} effective={} warmup_ms={:.2}",
            requested_block_size, block_size, warmup_elapsed_ms
        );
    }
    let extra_samples = (extra_inference_ms as usize)
        .saturating_mul(model_sample_rate as usize)
        .saturating_div(1000);
    let inference_block_size = block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    let target_buffer_samples = block_size.saturating_add(extra_samples);
    validate_buffer_contract(
        extra_inference_ms,
        block_size,
        HUBERT_WINDOW_SAMPLES,
        model_sample_rate,
    );
    let min_extra_ms = min_extra_inference_ms(HUBERT_WINDOW_SAMPLES, block_size, model_sample_rate);
    let headroom_samples = target_buffer_samples.saturating_sub(HUBERT_WINDOW_SAMPLES);
    let sr_per_ms = model_sample_rate as f64 / 1000.0;
    let headroom_ms = if sr_per_ms > 0.0 {
        headroom_samples as f64 / sr_per_ms
    } else {
        0.0
    };
    eprintln!(
        "[vc-audio] buffer_geometry: block={}smp process_window={}smp target_buffer={}smp extra_ms={}ms headroom={}smp ({:.1}ms) min_extra_ms={}ms",
        block_size,
        HUBERT_WINDOW_SAMPLES,
        target_buffer_samples,
        extra_inference_ms,
        headroom_samples,
        headroom_ms,
        min_extra_ms
    );
    let needed_input_frames = ((target_buffer_samples as f64) * (input_rate as f64)
        / (model_sample_rate as f64))
        .round()
        .max(1.0) as usize;
    let est_callback_frames = ((input_rate as f64) * 0.01).round().max(64.0) as usize;
    let callbacks_per_infer = needed_input_frames.div_ceil(est_callback_frames);
    let input_queue_capacity = callbacks_per_infer.saturating_mul(2).max(64);
    let output_queue_capacity = callbacks_per_infer.saturating_mul(2).max(64);
    eprintln!(
        "[vc-audio] queue_capacity input={} output={} (est_cb={} frames, infer_block={} target_buffer={})",
        input_queue_capacity,
        output_queue_capacity,
        est_callback_frames,
        inference_block_size,
        target_buffer_samples
    );
    let budget_ms = block_size as f64 / model_sample_rate as f64 * 1000.0;
    if warmup_elapsed_ms > budget_ms {
        eprintln!(
            "[vc-audio] latency advisory: warmup={:.2}ms exceeds budget={:.2}ms (effective_block={})",
            warmup_elapsed_ms, budget_ms, block_size
        );
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
    let (worker_control_tx, worker_control_rx) = sync_channel::<WorkerControl>(1);
    let (worker_shutdown_ack_tx, worker_shutdown_ack_rx) = sync_channel::<()>(1);
    let worker_thread = thread::spawn(move || {
        let mut level_meter = LevelMeter::new(0.92);
        worker_loop(
            &mut voice_changer,
            input_rx,
            output_tx,
            worker_control_rx,
            worker_shutdown_ack_tx,
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
            output_tail_offset_ms,
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
        worker_control_tx: Some(worker_control_tx),
        worker_shutdown_ack_rx: Some(worker_shutdown_ack_rx),
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

fn resolve_output_tail_offset_samples(
    output_rate: u32,
    edge_guard_samples: usize,
    output_tail_offset_ms: u32,
) -> usize {
    if output_tail_offset_ms == 0 {
        return edge_guard_samples;
    }
    let from_ms =
        ((output_rate as f64) * (output_tail_offset_ms as f64) / 1000.0).round().max(0.0) as usize;
    from_ms.max(edge_guard_samples)
}

#[allow(clippy::too_many_arguments)]
fn worker_loop<E>(
    voice_changer: &mut VoiceChanger<E>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: SyncSender<Vec<f32>>,
    worker_control_rx: Receiver<WorkerControl>,
    worker_shutdown_ack_tx: SyncSender<()>,
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
    output_tail_offset_ms: u32,
    running: &Arc<AtomicBool>,
    input_rms: &Arc<AtomicU32>,
    input_peak: &Arc<AtomicU32>,
    level_meter: &mut LevelMeter,
) where
    E: InferenceEngine,
{
    let mut shutdown_prepared = false;
    let extra_samples = (extra_inference_ms as usize)
        .saturating_mul(model_sample_rate as usize)
        .saturating_div(1000);
    let io_block_size = block_size.max(1);
    let output_step_samples =
        map_model_samples_to_output(io_block_size, model_sample_rate, output_rate);
    let crossfade_samples = default_output_crossfade_samples(output_rate).min(output_step_samples);
    let sola_search_output = sola_search_samples(output_rate);
    let edge_guard_samples = crossfade_samples.max(sola_search_output / 2);
    let output_tail_offset_samples =
        resolve_output_tail_offset_samples(output_rate, edge_guard_samples, output_tail_offset_ms);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let inference_block_size = io_block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    let target_buffer_samples = io_block_size.saturating_add(extra_samples);
    let min_process_window_samples = io_block_size;
    let process_window_cap = HUBERT_WINDOW_SAMPLES;
    let mut process_window_samples = if allow_process_window_grow {
        min_process_window_samples
    } else {
        process_window_cap
    };
    let mut smoothed_elapsed_ms = 0.0_f64;
    let mut model_rate_buf = Vec::<f32>::new();
    let mut model_input_queue = VecDeque::<f32>::new();
    let mut model_block = Vec::<f32>::with_capacity(process_window_cap.max(inference_block_size));
    let mut output_rate_buf = Vec::<f32>::new();
    let mut previous_output_tail = Vec::<f32>::new();
    let input_to_model_resampler = HqResampler::new(input_rate, model_sample_rate);
    let model_to_output_resampler = HqResampler::new(model_sample_rate, output_rate);
    let mut sola_correlator = SolaFftCorrelator::default();
    let mut processed_blocks: u64 = 0;
    let mut silence_skips: u64 = 0;
    let mut last_heartbeat = Instant::now();
    let mut last_low_queue_warn = Instant::now();
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
    eprintln!(
        "[vc-audio] output_tail_offset_out={} (~{:.2}ms) edge_guard_out={} (set output_tail_offset_ms in runtime config to tune)",
        output_tail_offset_samples,
        output_tail_offset_samples as f64 / output_rate.max(1) as f64 * 1000.0,
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
    let prefill_target_samples = target_buffer_samples.max(process_window_samples);
    let prefill_model_samples = if model_sample_rate != output_rate {
        ((prefill_target_samples.saturating_add(io_block_size) as f64) * output_rate as f64
            / model_sample_rate as f64)
            .round()
            .max(1.0) as usize
    } else {
        prefill_target_samples.saturating_add(io_block_size)
    };
    let prefill_frames = prefill_model_samples * output_channels * 2;
    let _ = output_tx.try_send(vec![0.0; prefill_frames]);

    while running.load(Ordering::Relaxed) {
        if !shutdown_prepared {
            match worker_control_rx.try_recv() {
                Ok(WorkerControl::PrepareShutdown) => {
                    if let Err(e) = voice_changer.prepare_shutdown() {
                        eprintln!("[vc-audio] warning: inference shutdown hook failed: {e}");
                    }
                    shutdown_prepared = true;
                    let _ = worker_shutdown_ack_tx.try_send(());
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {}
            }
        }
        let mut mono = match input_rx.recv_timeout(Duration::from_millis(5)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                running.store(false, Ordering::Relaxed);
                break;
            }
        };
        if response_threshold > 0.0 {
            gate.process_block(&mut mono);
        }

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

            let (output_source, source_slice_start, source_slice_end): (Vec<f32>, usize, usize) =
                if model_out.len() >= process_window_samples {
                    // Keep the newest region: sliding-window inference appends fresh content to the tail.
                    let end = model_out.len();
                    let start = end.saturating_sub(process_window_samples);
                    (model_out[start..end].to_vec(), start, end)
                } else {
                    // If model output is shorter than the process window, align it to the tail and
                    // pad the head with zeros so downstream tail slicing keeps recent audio.
                    let mut padded = vec![0.0_f32; process_window_samples];
                    let copy_len = model_out.len();
                    if copy_len > 0 {
                        let dst_start = process_window_samples.saturating_sub(copy_len);
                        padded[dst_start..].copy_from_slice(&model_out);
                    }
                    (padded, 0, model_out.len())
                };
            let output_mono: &[f32] = if model_sample_rate != output_rate {
                resample_hq_into(
                    &model_to_output_resampler,
                    &output_source,
                    &mut output_rate_buf,
                );
                &output_rate_buf
            } else {
                &output_source
            };
            let expected_len = output_step_samples.min(output_mono.len().max(1));
            let ola_overlap_samples =
                compute_ola_overlap_samples(expected_len, crossfade_samples, extra_inference_ms);
            // Avoid using the very end of each block where non-causal generators are often unstable.
            let max_guard = output_mono.len().saturating_sub(expected_len);
            let applied_guard = output_tail_offset_samples.min(max_guard);
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
                offset = sola_correlator.find_best_offset(
                    &previous_output_tail,
                    tail,
                    crossfade_for_search,
                    search_limit,
                );
            }
            if processed_blocks < OUTPUT_SLICE_AUDIT_BLOCKS {
                eprintln!(
                    "[vc-audio] output-slice-audit block={} model_out_len={} source_slice=[{}..{}) output_len={} expected_len={} tail_offset={} anchor_end={} tail=[{}..{}) search_limit={}",
                    processed_blocks + 1,
                    model_out.len(),
                    source_slice_start,
                    source_slice_end,
                    output_mono.len(),
                    expected_len,
                    applied_guard,
                    anchor_end,
                    tail_start,
                    anchor_end,
                    search_limit
                );
            }
            let mut smoothed_output =
                tail[offset..(offset + expected_len).min(tail.len())].to_vec();
            if smoothed_output.len() < expected_len {
                let pad = smoothed_output.last().copied().unwrap_or(0.0);
                smoothed_output.resize(expected_len, pad);
            }
            let overlap = ola_overlap_samples
                .min(smoothed_output.len())
                .min(previous_output_tail.len());
            apply_raised_cosine_overlap(&previous_output_tail, &mut smoothed_output, overlap);
            let keep = ola_overlap_samples.min(smoothed_output.len());
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
            if allow_process_window_grow
                && process_window_samples > min_process_window_samples
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
                let next_window = process_window_samples.saturating_add(grow);
                let clamped = next_window.min(process_window_cap);
                if clamped >= process_window_cap && process_window_samples < process_window_cap {
                    eprintln!(
                        "[vc-audio] process_window converged: now={} (target_buffer={})",
                        clamped, target_buffer_samples
                    );
                }
                if clamped != process_window_samples {
                    process_window_samples = clamped;
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
            let queue_ms =
                model_input_queue.len() as f64 / model_sample_rate.max(1) as f64 * 1000.0;
            let expected_steady_queue = process_window_samples.saturating_sub(io_block_size);
            if model_input_queue.len() + 64 < expected_steady_queue
                && last_low_queue_warn.elapsed() >= Duration::from_secs(1)
            {
                eprintln!(
                    "[vc-audio] low input queue headroom: queue={} (~{:.2}ms) expected_steady={} (~{:.2}ms)",
                    model_input_queue.len(),
                    queue_ms,
                    expected_steady_queue,
                    expected_steady_queue as f64 / model_sample_rate.max(1) as f64 * 1000.0
                );
                last_low_queue_warn = Instant::now();
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "[vc-audio] heartbeat blocks={} queue={} queue_ms={:.2} budget_ms={:.2} rms={:.4} peak={:.4} silence_skips={}",
                    processed_blocks,
                    model_input_queue.len(),
                    queue_ms,
                    budget_ms,
                    level_meter.rms(),
                    level_meter.peak(),
                    silence_skips
                );
                last_heartbeat = Instant::now();
            }
        }
    }
    if !shutdown_prepared {
        if let Err(e) = voice_changer.prepare_shutdown() {
            eprintln!("[vc-audio] warning: inference shutdown hook failed: {e}");
        }
        let _ = worker_shutdown_ack_tx.try_send(());
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
        return 0.0;
    }
    if raw == 0.0 {
        // 0.0 means gate disabled (fully open).
        return 0.0;
    }
    if raw < 0.0 {
        // Legacy/UI path: negative values are dBFS thresholds (e.g. -50 dB).
        let db = raw.clamp(-120.0, 0.0);
        return (10.0_f32.powf(db / 20.0)).clamp(0.0, 1.0);
    }
    raw.clamp(0.0, 1.0)
}

fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        return value.max(1);
    }
    value.div_ceil(align) * align
}

fn recommended_block_size_from_warmup(
    requested_block_size: usize,
    model_sample_rate: u32,
    warmup_elapsed_ms: f64,
) -> usize {
    let requested = requested_block_size.max(1);
    if !warmup_elapsed_ms.is_finite() || warmup_elapsed_ms <= 0.0 || model_sample_rate == 0 {
        return requested;
    }
    let target_budget_ms = warmup_elapsed_ms * WARMUP_BUDGET_HEADROOM;
    let required = ((target_budget_ms / 1000.0) * model_sample_rate as f64).ceil() as usize;
    let recommended = align_up(required.max(requested), BLOCK_ALIGN_SAMPLES);
    let cap = requested
        .max(requested.saturating_mul(2).min(MAX_WARMUP_BLOCK_CAP))
        .min(MAX_AUTO_BLOCK_SIZE);
    recommended.min(cap)
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

fn compute_ola_overlap_samples(
    expected_len: usize,
    base_crossfade: usize,
    extra_inference_ms: u32,
) -> usize {
    if expected_len == 0 {
        return 0;
    }
    let floor = base_crossfade.min(expected_len).max(1);
    let min_overlap = ((expected_len as f32) * OLA_MIN_OVERLAP_RATIO)
        .round()
        .max(floor as f32) as usize;
    let max_overlap = (((expected_len as f32) * OLA_MAX_OVERLAP_RATIO)
        .round()
        .max(min_overlap as f32) as usize)
        .min(expected_len);
    let t = (extra_inference_ms as f32 / 300.0).clamp(0.0, 1.0);
    min_overlap + (((max_overlap - min_overlap) as f32) * t).round() as usize
}

fn apply_raised_cosine_overlap(previous_tail: &[f32], current: &mut [f32], overlap: usize) {
    if overlap == 0 || previous_tail.len() < overlap || current.len() < overlap {
        return;
    }
    let prev_start = previous_tail.len() - overlap;
    let inv = 1.0_f32 / (overlap + 1) as f32;
    for i in 0..overlap {
        let t = (i + 1) as f32 * inv;
        // Raised-cosine overlap-add where weights always sum to 1.0.
        let w_curr = 0.5_f32 - 0.5_f32 * (std::f32::consts::PI * t).cos();
        let w_prev = 1.0_f32 - w_curr;
        current[i] = previous_tail[prev_start + i] * w_prev + current[i] * w_curr;
    }
}

#[derive(Default)]
struct SolaFftCorrelator {
    fft_len: usize,
    forward_fft: Option<Arc<dyn Fft<f32>>>,
    inverse_fft: Option<Arc<dyn Fft<f32>>>,
    fft_a: Vec<Complex32>,
    fft_b: Vec<Complex32>,
    fft_tmp: Vec<Complex32>,
    energy_prefix: Vec<f32>,
}

impl SolaFftCorrelator {
    fn ensure_plan(&mut self, fft_len: usize) {
        if self.fft_len == fft_len {
            return;
        }
        let mut planner = FftPlanner::<f32>::new();
        self.forward_fft = Some(planner.plan_fft_forward(fft_len));
        self.inverse_fft = Some(planner.plan_fft_inverse(fft_len));
        self.fft_a = vec![Complex32::new(0.0, 0.0); fft_len];
        self.fft_b = vec![Complex32::new(0.0, 0.0); fft_len];
        self.fft_tmp = vec![Complex32::new(0.0, 0.0); fft_len];
        self.fft_len = fft_len;
    }

    fn find_best_offset(
        &mut self,
        previous_tail: &[f32],
        current: &[f32],
        crossfade: usize,
        search_limit: usize,
    ) -> usize {
        if crossfade == 0 || previous_tail.len() < crossfade || current.len() < crossfade {
            return 0;
        }
        let prev = &previous_tail[previous_tail.len() - crossfade..];
        let prev_energy = prev.iter().map(|v| v * v).sum::<f32>();
        if prev_energy <= 1.0e-12 {
            return 0;
        }
        let max_offset = search_limit.min(current.len().saturating_sub(crossfade));
        let corr_input_len = crossfade + max_offset;
        if corr_input_len == 0 {
            return 0;
        }

        let conv_len = corr_input_len + crossfade - 1;
        let fft_len = conv_len.next_power_of_two().max(1);
        self.ensure_plan(fft_len);

        self.fft_a.fill(Complex32::new(0.0, 0.0));
        self.fft_b.fill(Complex32::new(0.0, 0.0));
        self.fft_tmp.fill(Complex32::new(0.0, 0.0));

        for i in 0..corr_input_len {
            self.fft_a[i].re = current[i];
        }
        for i in 0..crossfade {
            self.fft_b[i].re = prev[crossfade - 1 - i];
        }

        let Some(forward) = self.forward_fft.as_ref() else {
            return find_best_sola_offset_time_domain(
                previous_tail,
                current,
                crossfade,
                max_offset,
            );
        };
        forward.process(&mut self.fft_a);
        forward.process(&mut self.fft_b);
        for i in 0..self.fft_len {
            self.fft_tmp[i] = self.fft_a[i] * self.fft_b[i];
        }
        let Some(inverse) = self.inverse_fft.as_ref() else {
            return find_best_sola_offset_time_domain(
                previous_tail,
                current,
                crossfade,
                max_offset,
            );
        };
        inverse.process(&mut self.fft_tmp);
        let inv_fft_len = 1.0_f32 / self.fft_len as f32;

        if self.energy_prefix.len() < corr_input_len + 1 {
            self.energy_prefix.resize(corr_input_len + 1, 0.0);
        }
        self.energy_prefix[0] = 0.0;
        for i in 0..corr_input_len {
            self.energy_prefix[i + 1] = self.energy_prefix[i] + current[i] * current[i];
        }

        let mut best_offset = 0usize;
        let mut best_score = f32::MIN;
        for offset in 0..=max_offset {
            let corr_idx = offset + crossfade - 1;
            let nom = self.fft_tmp[corr_idx].re * inv_fft_len;
            let seg_energy = self.energy_prefix[offset + crossfade] - self.energy_prefix[offset];
            if seg_energy <= 1.0e-12 {
                continue;
            }
            let score = nom / (prev_energy * seg_energy).sqrt().max(1.0e-8);
            if score.is_finite() && score > best_score {
                best_score = score;
                best_offset = offset;
            }
        }

        if best_score == f32::MIN {
            find_best_sola_offset_time_domain(previous_tail, current, crossfade, max_offset)
        } else {
            best_offset
        }
    }
}

fn find_best_sola_offset_time_domain(
    previous_tail: &[f32],
    current: &[f32],
    crossfade: usize,
    search_limit: usize,
) -> usize {
    if crossfade == 0 || previous_tail.len() < crossfade || current.len() < crossfade {
        return 0;
    }
    let max_offset = search_limit.min(current.len().saturating_sub(crossfade));
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
    for offset in 0..=max_offset {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_buffer_contract_ok() {
        validate_buffer_contract(250, 8_192, 16_000, 48_000);
    }

    #[test]
    #[should_panic(expected = "FATAL: target_buffer")]
    fn test_validate_buffer_contract_deadlock() {
        validate_buffer_contract(80, 8_192, 16_000, 48_000);
    }

    #[test]
    fn test_min_extra_inference_ms() {
        assert_eq!(min_extra_inference_ms(16_000, 8_192, 48_000), 163);
    }

    #[test]
    fn test_min_extra_inference_ms_exact() {
        assert_eq!(min_extra_inference_ms(8_192, 8_192, 48_000), 0);
    }

    #[test]
    fn test_process_window_clamp_at_hubert_window() {
        const HUBERT_WINDOW: usize = 16_000;
        let mut w = 8_192usize;
        for _ in 0..1000 {
            w = (w + 500).min(HUBERT_WINDOW);
        }
        assert_eq!(w, HUBERT_WINDOW);
    }

    #[test]
    fn response_threshold_accepts_db_values() {
        let t = normalize_response_threshold(-50.0);
        assert!((t - 0.003_162_277_6).abs() < 1.0e-5);
    }

    #[test]
    fn response_threshold_zero_disables_gate() {
        assert_eq!(normalize_response_threshold(0.0), 0.0);
    }

    #[test]
    fn response_threshold_keeps_linear_values() {
        assert_eq!(normalize_response_threshold(0.2), 0.2);
    }

    #[test]
    fn sola_fft_matches_time_domain_offset() {
        let crossfade = 256usize;
        let search_limit = 64usize;
        let known_offset = 23usize;

        let mut previous_tail = vec![0.0_f32; 16];
        for i in 0..crossfade {
            let t = i as f32;
            let s = (2.0 * std::f32::consts::PI * t / 37.0).sin() * 0.7
                + (2.0 * std::f32::consts::PI * t / 19.0).cos() * 0.3;
            previous_tail.push(s);
        }

        let mut current = vec![0.0_f32; crossfade + search_limit];
        // Deterministic tiny noise floor
        let mut x = 0x1234_5678_u32;
        for v in &mut current {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((x >> 9) as f32 / (u32::MAX >> 9) as f32) - 0.5;
            *v = n * 0.02;
        }

        let prev = &previous_tail[previous_tail.len() - crossfade..];
        for i in 0..crossfade {
            current[known_offset + i] += prev[i];
        }

        let td =
            find_best_sola_offset_time_domain(&previous_tail, &current, crossfade, search_limit);
        let mut fft = SolaFftCorrelator::default();
        let fd = fft.find_best_offset(&previous_tail, &current, crossfade, search_limit);

        assert_eq!(td, known_offset);
        assert_eq!(fd, td);
    }
}
