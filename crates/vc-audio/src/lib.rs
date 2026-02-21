use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
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
const OLA_MIN_OVERLAP_RATIO: f32 = 0.25;
const OLA_MAX_OVERLAP_RATIO: f32 = 0.50;
const OUTPUT_SLICE_AUDIT_BLOCKS: u64 = 30;
const FRAME_GRID_SEC: f64 = 0.02; // 20ms per frame
const INPUT_DC_BLOCK_COEFF: f32 = 0.995;
const DEFAULT_DECODER_NEW_AUDIO_OFFSET_SAMPLES: usize = 31_680;
const FIXED_PROCESS_WINDOW_SAMPLES: usize = 48_000;
const DEBUG_DUMP_MAX_SECONDS: usize = 20;

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

/// Converts `target_buffer_ms` into samples at a given sample-rate.
pub fn target_buffer_samples_from_ms(target_buffer_ms: u32, sample_rate: u32) -> usize {
    if sample_rate == 0 {
        return 0;
    }
    (target_buffer_ms as usize)
        .saturating_mul(sample_rate as usize)
        .saturating_div(1000)
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
    /// Legacy compatibility flag.
    ///
    /// The backend now allocates the requested process window directly.
    /// Dynamic auto-grow/shrink behavior is no longer used.
    pub allow_process_window_grow: bool,
    /// Legacy inference overlap tuning (milliseconds).
    ///
    /// This value is used for overlap/fade heuristics in post-processing.
    /// Playback queue sizing is controlled by `target_buffer_ms`.
    pub extra_inference_ms: u32,
    /// Playback/output queue target size in milliseconds.
    ///
    /// Larger values increase stability and latency.
    /// This is independent from the model context window (`process_window`).
    pub target_buffer_ms: u32,
    pub response_threshold: f32,
    pub fade_in_ms: u32,
    pub fade_out_ms: u32,
    pub sola_search_ms: u32,
    pub output_tail_offset_ms: u32,
    pub output_slice_offset_samples: usize,
    pub record_dump: bool,
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
    let _allow_process_window_grow = options.allow_process_window_grow;
    let extra_inference_ms = options.extra_inference_ms;
    let target_buffer_ms = options.target_buffer_ms.max(1);
    let response_threshold = normalize_response_threshold(options.response_threshold);
    let fade_in_ms = options.fade_in_ms;
    let fade_out_ms = options.fade_out_ms;
    let sola_search_ms = options.sola_search_ms.max(1);
    let output_tail_offset_ms = options.output_tail_offset_ms;
    let output_slice_offset_samples = if options.output_slice_offset_samples == 0 {
        DEFAULT_DECODER_NEW_AUDIO_OFFSET_SAMPLES
    } else {
        options.output_slice_offset_samples
    };
    let record_dump = options.record_dump;

    let input_device = find_input_device(&host, options.input_device_name.as_deref())?;
    let output_device = find_output_device(&host, options.output_device_name.as_deref())?;

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    if output_config.sample_format() != cpal::SampleFormat::F32 {
        return Err(anyhow!(
            "only f32 output sample format is supported in this prototype"
        ));
    }
    let input_sample_format = input_config.sample_format();

    let input_stream_config: cpal::StreamConfig = input_config.clone().into();
    let output_stream_config: cpal::StreamConfig = output_config.clone().into();
    let input_rate = input_stream_config.sample_rate.0;
    let output_rate = output_stream_config.sample_rate.0;
    let input_channels = input_stream_config.channels as usize;
    let output_channels = output_stream_config.channels as usize;
    let sola_search_output = sola_search_samples(output_rate, sola_search_ms);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let warmup_inference_block_size = requested_block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    eprintln!(
        "[vc-audio] input_rate={} output_rate={} input_ch={} output_ch={} input_fmt={:?} output_fmt={:?} model_rate={} block_size_requested={}",
        input_rate,
        output_rate,
        input_channels,
        output_channels,
        input_sample_format,
        output_config.sample_format(),
        model_sample_rate,
        requested_block_size
    );
    eprintln!(
        "[vc-audio] input_device='{}' output_device='{}' extra_inference_ms={} target_buffer_ms={} threshold={:.4} fade_in_ms={} fade_out_ms={} sola_search_ms={} tail_offset_ms={} slice_offset_samples={} record_dump={}",
        input_device.name().unwrap_or_else(|_| "unknown-input".to_string()),
        output_device.name().unwrap_or_else(|_| "unknown-output".to_string()),
        extra_inference_ms,
        target_buffer_ms,
        response_threshold,
        fade_in_ms,
        fade_out_ms,
        sola_search_ms,
        output_tail_offset_ms,
        output_slice_offset_samples,
        record_dump
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

    let frame_grid = frame_grid_samples(model_sample_rate);
    let mut block_size = align_down(requested_block_size.max(1), frame_grid);
    block_size = block_size.max(1);
    if block_size != requested_block_size {
        eprintln!(
            "[vc-audio] block_size aligned to frame grid: pre_align={} aligned={} frame_grid={}samples",
            requested_block_size, block_size, frame_grid
        );
    }
    voice_changer.set_effective_block_size(block_size);
    eprintln!(
        "[vc-audio] synced effective block_size to inference config: {} samples",
        block_size
    );
    let inference_block_size = block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    let process_window_samples = FIXED_PROCESS_WINDOW_SAMPLES;
    if inference_block_size > process_window_samples {
        return Err(anyhow!(
            "invalid realtime geometry: inference_block_size({}) > fixed_process_window({})",
            inference_block_size,
            process_window_samples
        ));
    }
    let requested_target_buffer_out_samples =
        target_buffer_samples_from_ms(target_buffer_ms, output_rate).max(1);
    let requested_target_buffer_model_samples = map_output_samples_to_model(
        requested_target_buffer_out_samples,
        model_sample_rate,
        output_rate,
    );
    let min_target_buffer_model_samples = process_window_samples.saturating_add(block_size);
    let target_buffer_samples =
        requested_target_buffer_model_samples.max(min_target_buffer_model_samples);
    let target_buffer_out_samples =
        map_model_samples_to_output(target_buffer_samples, model_sample_rate, output_rate);
    let target_buffer_effective_ms = if output_rate > 0 {
        (target_buffer_out_samples as f64) * 1000.0 / output_rate as f64
    } else {
        0.0
    };
    let headroom_samples =
        target_buffer_samples.saturating_sub(process_window_samples.saturating_add(block_size));
    let sr_per_ms = model_sample_rate.max(1) as f64 / 1000.0;
    let headroom_ms = if sr_per_ms > 0.0 {
        headroom_samples as f64 / sr_per_ms
    } else {
        0.0
    };
    eprintln!(
        "[vc-audio] buffer_geometry: block={}smp process_window={}smp target_buffer={}smp (~{:.1}ms) independent requested_target_ms={} floor={}smp headroom={}smp ({:.1}ms)",
        block_size,
        process_window_samples,
        target_buffer_samples,
        target_buffer_effective_ms,
        target_buffer_ms,
        min_target_buffer_model_samples,
        headroom_samples,
        headroom_ms
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
    let playback_ready = Arc::new(AtomicBool::new(false));
    let queued_output_samples = Arc::new(AtomicUsize::new(0));

    let output_data_fn = build_output_callback(
        Arc::clone(&running),
        output_rx,
        Arc::clone(&playback_ready),
        Arc::clone(&queued_output_samples),
        output_channels,
    );
    let input_stream = match input_sample_format {
        cpal::SampleFormat::F32 => input_device.build_input_stream(
            &input_stream_config,
            build_input_callback_f32(Arc::clone(&running), input_channels, input_tx.clone()),
            |err| eprintln!("input stream error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I16 => input_device.build_input_stream(
            &input_stream_config,
            build_input_callback_i16(Arc::clone(&running), input_channels, input_tx.clone()),
            |err| eprintln!("input stream error: {err}"),
            None,
        )?,
        cpal::SampleFormat::U16 => input_device.build_input_stream(
            &input_stream_config,
            build_input_callback_u16(Arc::clone(&running), input_channels, input_tx.clone()),
            |err| eprintln!("input stream error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I32 => input_device.build_input_stream(
            &input_stream_config,
            build_input_callback_i32(Arc::clone(&running), input_channels, input_tx.clone()),
            |err| eprintln!("input stream error: {err}"),
            None,
        )?,
        cpal::SampleFormat::U32 => input_device.build_input_stream(
            &input_stream_config,
            build_input_callback_u32(Arc::clone(&running), input_channels, input_tx.clone()),
            |err| eprintln!("input stream error: {err}"),
            None,
        )?,
        other => {
            return Err(anyhow!(
                "unsupported input sample format: {other:?}. Supported: f32/i16/u16/i32/u32"
            ))
        }
    };
    let output_stream = output_device.build_output_stream(
        &output_stream_config,
        output_data_fn,
        |err| eprintln!("output stream error: {err}"),
        None,
    )?;

    let running_worker = Arc::clone(&running);
    let input_rms_worker = Arc::clone(&input_rms);
    let input_peak_worker = Arc::clone(&input_peak);
    let playback_ready_worker = Arc::clone(&playback_ready);
    let queued_output_samples_worker = Arc::clone(&queued_output_samples);
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
            extra_inference_ms,
            target_buffer_samples,
            response_threshold,
            fade_in_ms,
            fade_out_ms,
            sola_search_ms,
            output_tail_offset_ms,
            output_slice_offset_samples,
            record_dump,
            process_window_samples,
            &running_worker,
            &input_rms_worker,
            &input_peak_worker,
            &playback_ready_worker,
            &queued_output_samples_worker,
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

fn build_input_callback_f32(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[f32], &cpal::InputCallbackInfo) + Send + 'static {
    build_input_callback_with_convert(running, input_channels, input_tx, sample_f32_to_f32)
}

fn build_input_callback_i16(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[i16], &cpal::InputCallbackInfo) + Send + 'static {
    build_input_callback_with_convert(running, input_channels, input_tx, sample_i16_to_f32)
}

fn build_input_callback_u16(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[u16], &cpal::InputCallbackInfo) + Send + 'static {
    build_input_callback_with_convert(running, input_channels, input_tx, sample_u16_to_f32)
}

fn build_input_callback_i32(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[i32], &cpal::InputCallbackInfo) + Send + 'static {
    build_input_callback_with_convert(running, input_channels, input_tx, sample_i32_to_f32)
}

fn build_input_callback_u32(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
) -> impl FnMut(&[u32], &cpal::InputCallbackInfo) + Send + 'static {
    build_input_callback_with_convert(running, input_channels, input_tx, sample_u32_to_f32)
}

fn build_input_callback_with_convert<T, F>(
    running: Arc<AtomicBool>,
    input_channels: usize,
    input_tx: SyncSender<Vec<f32>>,
    sample_to_f32: F,
) -> impl FnMut(&[T], &cpal::InputCallbackInfo) + Send + 'static
where
    T: Copy + Send + 'static,
    F: Fn(T) -> f32 + Send + Copy + 'static,
{
    let mut dropped = 0usize;
    let mut dc_blocker = DcBlocker::new(INPUT_DC_BLOCK_COEFF);
    move |data: &[T], _: &cpal::InputCallbackInfo| {
        if !running.load(Ordering::Relaxed) {
            return;
        }
        let mut mono = downmix_to_mono_with_convert(data, input_channels, sample_to_f32);
        dc_blocker.process_in_place(&mut mono);
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
    playback_ready: Arc<AtomicBool>,
    queued_output_samples: Arc<AtomicUsize>,
    output_channels: usize,
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

        if !playback_ready.load(Ordering::Acquire) {
            data.fill(0.0);
            return;
        }

        let channels = output_channels.max(1);
        let mut consumed_interleaved = 0usize;
        let mut callback_underrun = false;
        for sample in data {
            if let Some(v) = pending.pop_front() {
                *sample = v;
                last_sample = v;
                consumed_interleaved += 1;
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
        let consumed_output_samples = consumed_interleaved / channels;
        atomic_saturating_sub(&queued_output_samples, consumed_output_samples);
    }
}

fn atomic_saturating_sub(cell: &AtomicUsize, by: usize) {
    if by == 0 {
        return;
    }
    let _ = cell.fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
        Some(cur.saturating_sub(by))
    });
}

fn append_samples_limited(dst: &mut Vec<f32>, src: &[f32], limit: usize) {
    if src.is_empty() || dst.len() >= limit {
        return;
    }
    let remaining = limit.saturating_sub(dst.len());
    let take = remaining.min(src.len());
    dst.extend_from_slice(&src[..take]);
}

fn write_debug_wav_i16(path: &str, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate.max(1),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn resolve_output_tail_offset_samples(
    output_rate: u32,
    edge_guard_samples: usize,
    output_tail_offset_ms: u32,
) -> usize {
    if output_tail_offset_ms == 0 {
        return 0;
    }
    let from_ms = ((output_rate as f64) * (output_tail_offset_ms as f64) / 1000.0)
        .round()
        .max(0.0) as usize;
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
    extra_inference_ms: u32,
    configured_target_buffer_samples: usize,
    response_threshold: f32,
    fade_in_ms: u32,
    fade_out_ms: u32,
    sola_search_ms: u32,
    output_tail_offset_ms: u32,
    output_slice_offset_samples: usize,
    record_dump: bool,
    configured_process_window_samples: usize,
    running: &Arc<AtomicBool>,
    input_rms: &Arc<AtomicU32>,
    input_peak: &Arc<AtomicU32>,
    playback_ready: &Arc<AtomicBool>,
    queued_output_samples: &Arc<AtomicUsize>,
    level_meter: &mut LevelMeter,
) where
    E: InferenceEngine,
{
    let mut shutdown_prepared = false;
    let io_block_size = block_size.max(1);
    let output_step_samples =
        map_model_samples_to_output(io_block_size, model_sample_rate, output_rate);
    let crossfade_samples = default_output_crossfade_samples(output_rate).min(output_step_samples);
    let sola_search_output = sola_search_samples(output_rate, sola_search_ms);
    let edge_guard_samples = crossfade_samples.max(sola_search_output / 2);
    let output_tail_offset_samples =
        resolve_output_tail_offset_samples(output_rate, edge_guard_samples, output_tail_offset_ms);
    let sola_search_model =
        map_output_samples_to_model(sola_search_output, model_sample_rate, output_rate);
    let inference_block_size = io_block_size
        .max(MIN_INFERENCE_BLOCK)
        .saturating_add(sola_search_model);
    let process_window_samples = configured_process_window_samples;
    let target_buffer_samples = configured_target_buffer_samples.max(process_window_samples);
    let prefill_target_samples =
        map_model_samples_to_output(target_buffer_samples, model_sample_rate, output_rate)
            .max(output_step_samples);
    let process_window_cap = process_window_samples;
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
    let max_recorded_in_samples = (input_rate as usize).saturating_mul(DEBUG_DUMP_MAX_SECONDS);
    let max_recorded_out_samples = (output_rate as usize).saturating_mul(DEBUG_DUMP_MAX_SECONDS);
    let mut recorded_in = record_dump.then(Vec::new);
    let mut recorded_out = record_dump.then(Vec::new);
    playback_ready.store(false, Ordering::Release);
    queued_output_samples.store(0, Ordering::Release);
    eprintln!(
        "[vc-audio] startup: target_buffer={}smp ({:.0}ms) prefill={}smp ({:.0}ms) block={}smp",
        target_buffer_samples,
        target_buffer_samples as f64 / model_sample_rate.max(1) as f64 * 1000.0,
        prefill_target_samples,
        prefill_target_samples as f64 / output_rate.max(1) as f64 * 1000.0,
        io_block_size
    );
    eprintln!(
        "[vc-audio] pre-fill: waiting for queue={}smp ({}ms) before playback",
        prefill_target_samples,
        (prefill_target_samples as u64).saturating_mul(1000) / output_rate.max(1) as u64
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
        "[vc-audio] output_tail_offset_out={} (~{:.2}ms) edge_guard_out={} slice_offset_samples={} (set output_tail_offset_ms/output_slice_offset_samples in runtime config to tune)",
        output_tail_offset_samples,
        output_tail_offset_samples as f64 / output_rate.max(1) as f64 * 1000.0,
        edge_guard_samples,
        output_slice_offset_samples
    );
    if process_window_samples != target_buffer_samples {
        eprintln!(
            "[vc-audio] target_buffer independent: process_window={} target_buffer={}",
            process_window_samples, target_buffer_samples
        );
    }
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
        if let Some(samples) = recorded_in.as_mut() {
            append_samples_limited(samples, &mono, max_recorded_in_samples);
        }
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
            let queue_len = model_input_queue.len();
            let window_start = queue_len.saturating_sub(process_window_samples);
            model_block.clear();
            model_block.extend(
                model_input_queue
                    .iter()
                    .skip(window_start)
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

            let output_mono: &[f32] = if model_sample_rate != output_rate {
                resample_hq_into(&model_to_output_resampler, &model_out, &mut output_rate_buf);
                &output_rate_buf
            } else {
                &model_out
            };
            // Keep callback pacing deterministic: always produce one output block per inference step.
            let expected_len = output_step_samples.max(1);
            let ola_overlap_samples =
                compute_ola_overlap_samples(expected_len, crossfade_samples, extra_inference_ms);
            // Compute output slice from the latest region dynamically.
            // For aligned decoder models this directly maps to the newest hop-sized chunk.
            let max_guard = output_mono.len().saturating_sub(1);
            let applied_guard = output_tail_offset_samples.min(max_guard);
            let guarded_end = output_mono.len().saturating_sub(applied_guard);
            let source_end = guarded_end;
            let source_start = source_end.saturating_sub(expected_len);
            let source_region = &output_mono[source_start..source_end];
            debug_assert_eq!(
                source_end.saturating_sub(source_start),
                expected_len,
                "slice must exactly match hop size - no padding expected"
            );
            let search_limit = source_region
                .len()
                .saturating_sub(expected_len)
                .min(sola_search_output);
            let crossfade_for_search = crossfade_samples
                .min(previous_output_tail.len())
                .min(source_region.len());
            let mut offset = 0usize;
            if search_limit > 0 && crossfade_for_search >= 16 {
                offset = sola_correlator.find_best_offset(
                    &previous_output_tail,
                    source_region,
                    crossfade_for_search,
                    search_limit,
                );
            }
            let final_start = source_start.saturating_add(offset);
            let final_end = (final_start + expected_len).min(source_end);
            if processed_blocks < OUTPUT_SLICE_AUDIT_BLOCKS {
                eprintln!(
                    "[vc-audio] output-slice-audit block={} model_out_len={} source_slice=[{}..{}) source_len={} expected_len={} tail_offset={} configured_start={} dynamic_start={} final_start={} final_end={} search_limit={}",
                    processed_blocks + 1,
                    model_out.len(),
                    source_start,
                    source_end,
                    source_region.len(),
                    expected_len,
                    applied_guard,
                    output_slice_offset_samples,
                    source_start,
                    final_start,
                    final_end,
                    search_limit
                );
            }
            let mut smoothed_output = output_mono[final_start..final_end].to_vec();
            if smoothed_output.len() < expected_len {
                let pad = smoothed_output.last().copied().unwrap_or(0.0);
                smoothed_output.resize(expected_len, pad);
            } else if smoothed_output.len() > expected_len {
                smoothed_output.truncate(expected_len);
            }
            if processed_blocks < OUTPUT_SLICE_AUDIT_BLOCKS {
                eprintln!(
                    "[vc-audio] output-slice-final block={} final_len={} expected_len={}",
                    processed_blocks + 1,
                    smoothed_output.len(),
                    expected_len
                );
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
            if let Some(samples) = recorded_out.as_mut() {
                append_samples_limited(samples, &smoothed_output, max_recorded_out_samples);
            }

            level_meter.push_block(&smoothed_output);
            input_rms.store(level_meter.rms().to_bits(), Ordering::Relaxed);
            input_peak.store(level_meter.peak().to_bits(), Ordering::Relaxed);

            let output_interleaved = upmix_from_mono(&smoothed_output, output_channels);
            let produced_output_samples = smoothed_output.len();
            match output_tx.try_send(output_interleaved) {
                Ok(()) => {
                    queued_output_samples.fetch_add(produced_output_samples, Ordering::AcqRel);
                }
                Err(TrySendError::Full(_)) => {
                    eprintln!("[vc-audio] output queue full");
                }
                Err(TrySendError::Disconnected(_)) => {
                    eprintln!("[vc-audio] output queue disconnected");
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }
            if !playback_ready.load(Ordering::Acquire) {
                let queued = queued_output_samples.load(Ordering::Acquire);
                eprintln!(
                    "[vc-audio] pre-fill progress: queue={}smp / {}smp",
                    queued.min(prefill_target_samples),
                    prefill_target_samples
                );
                if queued >= prefill_target_samples {
                    playback_ready.store(true, Ordering::Release);
                    eprintln!(
                        "[vc-audio] pre-fill complete: queue={}smp playback starting",
                        queued
                    );
                }
            }

            processed_blocks += 1;
            let keep_samples = process_window_samples.saturating_sub(io_block_size);
            while model_input_queue.len() > keep_samples {
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
            if elapsed > block_budget {
                eprintln!(
                    "[vc-audio] slow block: elapsed={:.2}ms budget={:.2}ms queue={}",
                    elapsed_ms,
                    budget_ms,
                    model_input_queue.len()
                );
            }
            let input_queue_ms =
                model_input_queue.len() as f64 / model_sample_rate.max(1) as f64 * 1000.0;
            let expected_steady_queue = process_window_samples.saturating_sub(io_block_size);
            if model_input_queue.len() + 64 < expected_steady_queue
                && last_low_queue_warn.elapsed() >= Duration::from_secs(1)
            {
                eprintln!(
                    "[vc-audio] low input queue headroom: queue={} (~{:.2}ms) expected_steady={} (~{:.2}ms)",
                    model_input_queue.len(),
                    input_queue_ms,
                    expected_steady_queue,
                    expected_steady_queue as f64 / model_sample_rate.max(1) as f64 * 1000.0
                );
                last_low_queue_warn = Instant::now();
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(1) {
                let queued = queued_output_samples.load(Ordering::Relaxed);
                let queue_ms = queued as f64 / output_rate.max(1) as f64 * 1000.0;
                let prefill_now = queued.min(prefill_target_samples);
                eprintln!(
                    "[vc-audio] heartbeat blocks={} queue={} queue_ms={:.2} budget_ms={:.2} prefill={}/{} rms={:.4} peak={:.4} silence_skips={}",
                    processed_blocks,
                    queued,
                    queue_ms,
                    budget_ms,
                    prefill_now,
                    prefill_target_samples,
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
    if record_dump {
        let input_len = recorded_in.as_ref().map_or(0, |v| v.len());
        let output_len = recorded_out.as_ref().map_or(0, |v| v.len());
        eprintln!(
            "[vc-audio] dump summary: input_samples={} output_samples={} max_seconds={}",
            input_len, output_len, DEBUG_DUMP_MAX_SECONDS
        );
        if let Some(samples) = recorded_in.as_ref() {
            if let Err(err) = write_debug_wav_i16("debug_input.wav", input_rate, samples) {
                eprintln!("[vc-audio] warning: failed to write debug_input.wav: {err}");
            } else {
                eprintln!(
                    "[vc-audio] wrote debug_input.wav ({} samples)",
                    samples.len()
                );
            }
        }
        if let Some(samples) = recorded_out.as_ref() {
            if let Err(err) = write_debug_wav_i16("debug_output.wav", output_rate, samples) {
                eprintln!("[vc-audio] warning: failed to write debug_output.wav: {err}");
            } else {
                eprintln!(
                    "[vc-audio] wrote debug_output.wav ({} samples)",
                    samples.len()
                );
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

fn align_down(value: usize, align: usize) -> usize {
    if align <= 1 {
        return value.max(1);
    }
    if value < align {
        return value.max(1);
    }
    (value / align) * align
}

fn frame_grid_samples(model_sample_rate: u32) -> usize {
    if model_sample_rate == 0 {
        return 1;
    }
    ((model_sample_rate as f64) * FRAME_GRID_SEC)
        .round()
        .max(1.0) as usize
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

fn sola_search_samples(output_rate: u32, sola_search_ms: u32) -> usize {
    let ms = sola_search_ms.max(1);
    ((output_rate as f32) * (ms as f32 / 1000.0))
        .round()
        .clamp(128.0, 4_096.0) as usize
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

#[inline]
fn sample_f32_to_f32(sample: f32) -> f32 {
    sample
}

#[inline]
fn sample_i16_to_f32(sample: i16) -> f32 {
    (sample as f32 / 32_768.0).clamp(-1.0, 1.0)
}

#[inline]
fn sample_u16_to_f32(sample: u16) -> f32 {
    ((sample as f32 / 65_535.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
}

#[inline]
fn sample_i32_to_f32(sample: i32) -> f32 {
    (sample as f32 / 2_147_483_648.0).clamp(-1.0, 1.0)
}

#[inline]
fn sample_u32_to_f32(sample: u32) -> f32 {
    ((sample as f32 / 4_294_967_295.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
}

#[cfg_attr(not(test), allow(dead_code))]
fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    downmix_to_mono_with_convert(data, channels, sample_f32_to_f32)
}

fn downmix_to_mono_with_convert<T, F>(data: &[T], channels: usize, sample_to_f32: F) -> Vec<f32>
where
    T: Copy,
    F: Fn(T) -> f32 + Copy,
{
    if channels <= 1 {
        let mut out = Vec::with_capacity(data.len());
        for &sample in data {
            out.push(sample_to_f32(sample));
        }
        return out;
    }

    let frames = data.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for frame_idx in 0..frames {
        let start = frame_idx * channels;
        let mut sum = 0.0_f32;
        for ch in 0..channels {
            sum += sample_to_f32(data[start + ch]);
        }
        out.push(sum / channels as f32);
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct DcBlocker {
    coeff: f32,
    prev_input: f32,
    prev_output: f32,
}

impl DcBlocker {
    fn new(coeff: f32) -> Self {
        Self {
            coeff: coeff.clamp(0.0, 0.9999),
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    fn process_in_place(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let input = *sample;
            let output = input - self.prev_input + self.coeff * self.prev_output;
            self.prev_input = input;
            self.prev_output = output;
            *sample = output;
        }
    }
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
    fn test_min_extra_inference_ms_for_1s_context() {
        assert_eq!(min_extra_inference_ms(48_000, 8_192, 48_000), 830);
    }

    #[test]
    fn test_min_extra_inference_ms_exact() {
        assert_eq!(min_extra_inference_ms(8_192, 8_192, 48_000), 0);
    }

    #[test]
    fn test_target_buffer_geometry_from_runtime_request() {
        assert_eq!(target_buffer_samples_from_ms(2_000, 48_000), 96_000);
        assert_eq!(target_buffer_samples_from_ms(500, 48_000), 24_000);
    }

    #[test]
    fn test_target_buffer_floor_from_process_window_plus_block() {
        let process_window = 48_000usize;
        let block = 24_000usize;
        let requested = target_buffer_samples_from_ms(1_000, 48_000); // 48_000
        let effective = requested.max(process_window + block);
        assert_eq!(effective, 72_000);
    }

    #[test]
    fn test_frame_grid_samples_48k_is_960() {
        assert_eq!(frame_grid_samples(48_000), 960);
    }

    #[test]
    fn test_align_down_to_frame_grid() {
        let grid = frame_grid_samples(48_000);
        assert_eq!(align_down(8_192, grid), 7_680);
        assert_eq!(align_down(15_520, grid), 15_360);
    }

    #[test]
    fn test_downmix_to_mono_stereo_interleaved() {
        let input = [1.0_f32, -1.0_f32, 0.5_f32, 0.5_f32, -0.25_f32, 0.25_f32];
        let mono = downmix_to_mono(&input, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.0).abs() < 1.0e-6);
        assert!((mono[1] - 0.5).abs() < 1.0e-6);
        assert!((mono[2] - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_sample_i16_to_f32_range() {
        assert!((sample_i16_to_f32(i16::MAX) - 0.9999695).abs() < 1.0e-4);
        assert_eq!(sample_i16_to_f32(i16::MIN), -1.0);
    }

    #[test]
    fn test_dc_blocker_reduces_constant_offset() {
        let mut dc = DcBlocker::new(INPUT_DC_BLOCK_COEFF);
        let mut samples = vec![0.1_f32; 4096];
        dc.process_in_place(&mut samples);
        let tail_avg = samples[3500..].iter().copied().sum::<f32>() / 596.0;
        assert!(tail_avg.abs() < 1.0e-3);
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
