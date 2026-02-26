#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    any::Any,
    fs,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use tauri::State;
use vc_audio::{
    list_input_devices, list_output_devices, spawn_voice_changer_stream, AudioStreamOptions,
    RealtimeAudioEngine,
};
use vc_core::{ModelConfig, RuntimeConfig, VoiceChanger};
use vc_inference::RvcOrtEngine;

fn log_debug(message: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    eprintln!("[tauri:{ts}] {message}");
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_string()
}

fn run_panic_safe<T>(cmd_name: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let details = panic_payload_to_string(payload);
            let msg = format!("{cmd_name} panicked: {details}");
            log_debug(&msg);
            Err(msg)
        }
    }
}

struct AppState {
    inner: Mutex<RuntimeState>,
}

impl AppState {
    fn lock_runtime(&self) -> MutexGuard<'_, RuntimeState> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log_debug("state lock poisoned; recovering");
                poisoned.into_inner()
            }
        }
    }
}

struct RuntimeState {
    config: RuntimeConfig,
    model: ModelConfig,
    running: bool,
    input_level_rms: f32,
    input_level_peak: f32,
    engine_task: Option<RealtimeAudioEngine>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(RuntimeState {
                config: default_runtime_config(),
                model: default_model_config(),
                running: false,
                input_level_rms: 0.0,
                input_level_peak: 0.0,
                engine_task: None,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct AudioDevicesPayload {
    input_devices: Vec<String>,
    output_devices: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EngineStatus {
    running: bool,
    input_level_rms: f32,
    input_level_peak: f32,
}

#[tauri::command]
fn list_audio_devices_cmd() -> Result<AudioDevicesPayload, String> {
    log_debug("list_audio_devices_cmd");
    let input = list_input_devices().map_err(|e| e.to_string())?;
    let output = list_output_devices().map_err(|e| e.to_string())?;
    Ok(AudioDevicesPayload {
        input_devices: input,
        output_devices: output,
    })
}

#[tauri::command]
fn get_runtime_config_cmd(state: State<'_, AppState>) -> Result<RuntimeConfig, String> {
    run_panic_safe("get_runtime_config_cmd", || {
        log_debug("get_runtime_config_cmd");
        let guard = state.lock_runtime();
        Ok(guard.config.clone())
    })
}

#[tauri::command]
fn set_runtime_config_cmd(config: RuntimeConfig, state: State<'_, AppState>) -> Result<(), String> {
    let mut config = config;
    if !config.pitch_shift_semitones.is_finite() {
        config.pitch_shift_semitones = 0.0;
    }
    let clamped_pitch_shift = config.pitch_shift_semitones.clamp(-24.0, 24.0);
    if (clamped_pitch_shift - config.pitch_shift_semitones).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd pitch_shift_semitones clamped: {} -> {}",
            config.pitch_shift_semitones, clamped_pitch_shift
        ));
        config.pitch_shift_semitones = clamped_pitch_shift;
    }
    if config.pitch_shift_semitones.abs() >= 6.0 {
        log_debug(&format!(
            "warning: large pitch_shift_semitones={:.2} may sound unnatural/high-pitched",
            config.pitch_shift_semitones
        ));
    }
    let max_threads = max_runtime_threads();
    let clamped_intra = clamp_intra_threads(config.intra_threads, max_threads);
    if clamped_intra != config.intra_threads {
        log_debug(&format!(
            "set_runtime_config_cmd intra_threads clamped: {} -> {} (max_threads={})",
            config.intra_threads, clamped_intra, max_threads
        ));
        config.intra_threads = clamped_intra;
    }
    let clamped_inter = clamp_inter_threads(config.inter_threads, max_threads);
    if clamped_inter != config.inter_threads {
        log_debug(&format!(
            "set_runtime_config_cmd inter_threads clamped: {} -> {} (max_threads={})",
            config.inter_threads, clamped_inter, max_threads
        ));
        config.inter_threads = clamped_inter;
    }
    let clamped_rows = config.index_search_rows.max(1);
    if clamped_rows != config.index_search_rows {
        log_debug(&format!(
            "set_runtime_config_cmd index_search_rows clamped: {} -> {}",
            config.index_search_rows, clamped_rows
        ));
        config.index_search_rows = clamped_rows;
    }
    if config.index_search_rows > 10_000 {
        log_debug(&format!(
            "warning: large index_search_rows={} may increase CPU time significantly",
            config.index_search_rows
        ));
    }
    let clamped_top_k = config.index_top_k.max(1).min(config.index_search_rows);
    if clamped_top_k != config.index_top_k {
        log_debug(&format!(
            "set_runtime_config_cmd index_top_k clamped: {} -> {} (rows={})",
            config.index_top_k, clamped_top_k, config.index_search_rows
        ));
        config.index_top_k = clamped_top_k;
    }
    if !config.vad_on_threshold.is_finite() {
        config.vad_on_threshold = config.response_threshold;
    }
    if config.vad_on_threshold == 0.0 && config.response_threshold != 0.0 {
        config.vad_on_threshold = config.response_threshold;
    }
    if !config.vad_off_threshold.is_finite() {
        config.vad_off_threshold = derive_vad_off_threshold(config.vad_on_threshold);
    }
    if config.vad_off_threshold == 0.0 && config.vad_on_threshold != 0.0 {
        config.vad_off_threshold = derive_vad_off_threshold(config.vad_on_threshold);
    }
    // Keep legacy field aligned for older subsystems and saved presets.
    config.response_threshold = config.vad_on_threshold;
    log_debug(&format!(
        "set_runtime_config_cmd sample_rate={} block_size={} in_dev={:?} out_dev={:?} extra_ms={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} fade_in_ms={} fade_out_ms={} sola_search_ms={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={} pitch_shift={:.2} index_rate={} index_smooth={:.2} top_k={} rows={} protect={:.2} rmvpe_th={:.3} pitch_smooth={:.2} rms_mix={:.2} post_filter={:.3} f0_med_r={} ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={}",
        config.sample_rate,
        config.block_size,
        config.input_device_name,
        config.output_device_name,
        config.extra_inference_ms,
        config.target_buffer_ms,
        config.response_threshold,
        config.vad_on_threshold,
        config.vad_off_threshold,
        config.fade_in_ms,
        config.fade_out_ms,
        config.sola_search_ms,
        config.output_tail_offset_ms,
        config.output_slice_offset_samples,
        config.bypass_slicing,
        config.record_dump,
        config.pitch_shift_semitones,
        config.index_rate,
        config.index_smooth_alpha,
        config.index_top_k,
        config.index_search_rows,
        config.protect,
        config.rmvpe_threshold,
        config.pitch_smooth_alpha,
        config.rms_mix_rate,
        config.post_filter_alpha,
        config.f0_median_filter_radius,
        config.ort_provider,
        config.ort_device_id,
        config.ort_gpu_mem_limit_mb,
        config.intra_threads,
        config.inter_threads,
        config.ort_parallel_execution,
        config.hubert_context_samples_16k,
        config.hubert_output_layer,
        config.hubert_upsample_factor,
        config.cuda_conv_algo,
        config.cuda_ws,
        config.cuda_conv1d_pad_to_nc1d,
        config.cuda_tf32,
        config.index_bin_dim,
        config.index_max_vectors
    ));
    let mut guard = state.lock_runtime();
    let thread_changed = guard.config.intra_threads != config.intra_threads
        || guard.config.inter_threads != config.inter_threads;
    if guard.running && thread_changed {
        return Err(
            "changing intra_threads/inter_threads requires model reload; stop_engine_cmd first, then call set_runtime_config_cmd again.".to_string(),
        );
    }
    let running = guard.running;
    let index_rows = config.index_search_rows;
    let index_top_k = config.index_top_k;
    guard.config = config;
    if running {
        if let Some(task) = guard.engine_task.as_ref() {
            task.update_index_search_params(index_rows, index_top_k)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[tauri::command]
fn get_model_config_cmd(state: State<'_, AppState>) -> Result<ModelConfig, String> {
    run_panic_safe("get_model_config_cmd", || {
        log_debug("get_model_config_cmd");
        let guard = state.lock_runtime();
        Ok(guard.model.clone())
    })
}

#[tauri::command]
fn set_model_config_cmd(model: ModelConfig, state: State<'_, AppState>) -> Result<(), String> {
    log_debug(&format!(
        "set_model_config_cmd model={} hubert={:?} rmvpe={:?} index={:?}",
        model.model_path, model.hubert_path, model.pitch_extractor_path, model.index_path
    ));
    let mut guard = state.lock_runtime();
    guard.model = model;
    Ok(())
}

#[tauri::command]
fn start_engine_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("start_engine_cmd", || {
        log_debug("start_engine_cmd");
        let _ = ensure_ort_dylib_path();
        let (mut runtime_config, model_raw) = {
            let mut guard = state.lock_runtime();
            if guard.running {
                log_debug("engine already running");
                sync_levels_from_engine(&mut guard);
                return Ok(EngineStatus {
                    running: guard.running,
                    input_level_rms: guard.input_level_rms,
                    input_level_peak: guard.input_level_peak,
                });
            }
            (guard.config.clone(), guard.model.clone())
        };
        let mut model = resolve_model_config_for_start(model_raw)?;
        apply_mode_selection(&mut model, &mut runtime_config);
        log_debug(&format!(
            "ORT_DYLIB_PATH={:?}",
            std::env::var("ORT_DYLIB_PATH").ok()
        ));
        log_debug(&format!(
            "start with model={} hubert={:?} rmvpe={:?} index={:?} sr={} block={} in_dev={:?} out_dev={:?} extra_ms={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} fade_in_ms={} fade_out_ms={} sola_search_ms={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={} pitch_shift={:.2} index_rate={} index_smooth={:.2} top_k={} rows={} protect={:.2} rmvpe_th={:.3} pitch_smooth={:.2} rms_mix={:.2} post_filter={:.3} f0_med_r={} ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={}",
            model.model_path,
            model.hubert_path,
            model.pitch_extractor_path,
            model.index_path,
            runtime_config.sample_rate,
            runtime_config.block_size,
            runtime_config.input_device_name,
            runtime_config.output_device_name,
            runtime_config.extra_inference_ms,
            runtime_config.target_buffer_ms,
            runtime_config.response_threshold,
            runtime_config.vad_on_threshold,
            runtime_config.vad_off_threshold,
            runtime_config.fade_in_ms,
            runtime_config.fade_out_ms,
            runtime_config.sola_search_ms,
            runtime_config.output_tail_offset_ms,
            runtime_config.output_slice_offset_samples,
            runtime_config.bypass_slicing,
            runtime_config.record_dump,
            runtime_config.pitch_shift_semitones,
            runtime_config.index_rate,
            runtime_config.index_smooth_alpha,
            runtime_config.index_top_k,
            runtime_config.index_search_rows,
            runtime_config.protect,
            runtime_config.rmvpe_threshold,
            runtime_config.pitch_smooth_alpha,
            runtime_config.rms_mix_rate,
            runtime_config.post_filter_alpha,
            runtime_config.f0_median_filter_radius,
            runtime_config.ort_provider,
            runtime_config.ort_device_id,
            runtime_config.ort_gpu_mem_limit_mb,
            runtime_config.intra_threads,
            runtime_config.inter_threads,
            runtime_config.ort_parallel_execution,
            runtime_config.hubert_context_samples_16k,
            runtime_config.hubert_output_layer,
            runtime_config.hubert_upsample_factor,
            runtime_config.cuda_conv_algo,
            runtime_config.cuda_ws,
            runtime_config.cuda_conv1d_pad_to_nc1d,
            runtime_config.cuda_tf32,
            runtime_config.index_bin_dim,
            runtime_config.index_max_vectors
        ));
        let infer_engine = RvcOrtEngine::new(model, &runtime_config).map_err(|e| e.to_string())?;
        let voice_changer = VoiceChanger::new(infer_engine, runtime_config.clone());
        // Compatibility plumbing only; vc-audio currently runs with fixed process_window.
        let allow_process_window_grow = false;
        let audio_engine = spawn_voice_changer_stream(
            voice_changer,
            AudioStreamOptions {
                model_sample_rate: runtime_config.sample_rate,
                block_size: runtime_config.block_size,
                input_device_name: runtime_config.input_device_name.clone(),
                output_device_name: runtime_config.output_device_name.clone(),
                allow_process_window_grow,
                extra_inference_ms: runtime_config.extra_inference_ms,
                target_buffer_ms: runtime_config.target_buffer_ms,
                response_threshold: runtime_config.response_threshold,
                vad_on_threshold: runtime_config.vad_on_threshold,
                vad_off_threshold: runtime_config.vad_off_threshold,
                fade_in_ms: runtime_config.fade_in_ms,
                fade_out_ms: runtime_config.fade_out_ms,
                sola_search_ms: runtime_config.sola_search_ms,
                output_tail_offset_ms: runtime_config.output_tail_offset_ms,
                output_slice_offset_samples: runtime_config.output_slice_offset_samples,
                bypass_slicing: runtime_config.bypass_slicing,
                record_dump: runtime_config.record_dump,
            },
        )
        .map_err(|e| e.to_string())?;

        let mut guard = state.lock_runtime();
        if guard.running {
            log_debug("engine already running");
            sync_levels_from_engine(&mut guard);
            audio_engine.stop_and_abort();
            return Ok(EngineStatus {
                running: guard.running,
                input_level_rms: guard.input_level_rms,
                input_level_peak: guard.input_level_peak,
            });
        }

        guard.running = true;
        guard.config = runtime_config;
        guard.engine_task = Some(audio_engine);
        sync_levels_from_engine(&mut guard);
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
        })
    })
}

#[tauri::command]
fn stop_engine_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("stop_engine_cmd", || {
        log_debug("stop_engine_cmd");
        let mut guard = state.lock_runtime();
        if let Some(task) = guard.engine_task.take() {
            task.stop_and_abort();
        }
        guard.running = false;
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
        })
    })
}

#[tauri::command]
fn get_engine_status_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("get_engine_status_cmd", || {
        let mut guard = state.lock_runtime();
        sync_levels_from_engine(&mut guard);
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
        })
    })
}

fn sync_levels_from_engine(state: &mut RuntimeState) {
    if let Some(task) = &state.engine_task {
        let (rms, peak) = task.levels();
        state.input_level_rms = rms;
        state.input_level_peak = peak;
        state.running = task.is_running();
    }
}

fn default_model_config() -> ModelConfig {
    let pitch_extractor_path = std::env::var("RUST_VC_RMVPE_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/rmvpe_strict.onnx"))
        .or_else(|| resolve_existing_path("model/rmvpe.onnx"));
    let hubert_path = std::env::var("RUST_VC_HUBERT_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/hubert_pad80.onnx"))
        .or_else(|| resolve_existing_path("model/hubert_strict.onnx"))
        .or_else(|| resolve_existing_path("model/hubert.onnx"));
    let index_path = std::env::var("RUST_VC_INDEX_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/model_vectors.bin"))
        .or_else(find_first_bin_in_model_dir)
        .or_else(|| resolve_existing_path("model/model.index"))
        .or_else(|| resolve_existing_path("model/feature.index"));
    let model_path = std::env::var("RUST_VC_MODEL_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/model.onnx"))
        .or_else(find_first_onnx_in_model_dir)
        .unwrap_or_else(|| "model/model.onnx".to_string());

    let cfg = ModelConfig {
        model_path,
        index_path,
        pitch_extractor_path,
        hubert_path,
    };
    log_debug(&format!(
        "default_model_config model={} hubert={:?} rmvpe={:?} index={:?}",
        cfg.model_path, cfg.hubert_path, cfg.pitch_extractor_path, cfg.index_path
    ));
    cfg
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.trim().parse::<usize>().ok()
}

fn env_i64(key: &str) -> Option<i64> {
    std::env::var(key).ok()?.trim().parse::<i64>().ok()
}

fn env_i32(key: &str) -> Option<i32> {
    std::env::var(key).ok()?.trim().parse::<i32>().ok()
}

fn env_u32(key: &str) -> Option<u32> {
    std::env::var(key).ok()?.trim().parse::<u32>().ok()
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok()?.trim().parse::<f32>().ok()
}

fn env_bool(key: &str) -> Option<bool> {
    let raw = std::env::var(key).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn max_runtime_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1)
}

fn clamp_intra_threads(value: u32, max_threads: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    value.clamp(1, max_threads.max(1))
}

fn clamp_inter_threads(value: u32, max_threads: u32) -> u32 {
    value.clamp(1, max_threads.max(1))
}

fn default_runtime_config() -> RuntimeConfig {
    let mut cfg = RuntimeConfig::default();
    let max_threads = max_runtime_threads();

    if let Some(v) = std::env::var("RUST_VC_ORT_PROVIDER").ok() {
        let lc = v.trim().to_ascii_lowercase();
        if !lc.is_empty() {
            cfg.ort_provider = lc;
        }
    }
    if let Some(v) = env_i32("RUST_VC_ORT_DEVICE_ID") {
        cfg.ort_device_id = v.max(0);
    }
    if let Some(v) = env_u32("RUST_VC_ORT_GPU_MEM_LIMIT_MB") {
        cfg.ort_gpu_mem_limit_mb = v;
    }
    if let Some(v) = env_u32("RUST_VC_ORT_INTRA_THREADS") {
        cfg.intra_threads = clamp_intra_threads(v, max_threads);
    }
    if let Some(v) = env_u32("RUST_VC_ORT_INTER_THREADS") {
        cfg.inter_threads = clamp_inter_threads(v, max_threads);
    }
    if let Some(v) = env_bool("RUST_VC_ORT_PARALLEL") {
        cfg.ort_parallel_execution = v;
    }
    if let Some(v) = env_usize("RUST_VC_HUBERT_CONTEXT_16K") {
        cfg.hubert_context_samples_16k = v.max(1_600);
    }
    if let Some(v) = env_i64("RUST_VC_HUBERT_OUTPUT_LAYER") {
        cfg.hubert_output_layer = v;
    }
    if let Some(v) = env_usize("RUST_VC_HUBERT_UPSAMPLE_FACTOR") {
        cfg.hubert_upsample_factor = v.clamp(1, 4);
    }
    if let Some(v) = std::env::var("RUST_VC_CUDA_CONV_ALGO").ok() {
        let lc = v.trim().to_ascii_lowercase();
        if !lc.is_empty() {
            cfg.cuda_conv_algo = lc;
        }
    }
    if let Some(v) = env_bool("RUST_VC_CUDA_WS") {
        cfg.cuda_ws = v;
    } else if let Some(v) = env_bool("RUST_VC_CUDA_CONV_MAX_WORKSPACE") {
        cfg.cuda_ws = v;
    }
    if let Some(v) = env_bool("RUST_VC_CUDA_CONV1D_PAD_TO_NC1D") {
        cfg.cuda_conv1d_pad_to_nc1d = v;
    }
    if let Some(v) = env_bool("RUST_VC_CUDA_TF32") {
        cfg.cuda_tf32 = v;
    }
    if let Some(v) = env_usize("RUST_VC_INDEX_BIN_DIM") {
        cfg.index_bin_dim = v.max(1);
    }
    if let Some(v) = env_usize("RUST_VC_INDEX_MAX_VECTORS") {
        cfg.index_max_vectors = v;
    }
    if let Some(v) = env_f32("RUST_VC_PITCH_SMOOTH_ALPHA") {
        cfg.pitch_smooth_alpha = v.max(0.0);
    }
    if let Some(v) = env_u32("RUST_VC_TARGET_BUFFER_MS") {
        cfg.target_buffer_ms = v.max(1);
    }
    if let Some(v) = env_f32("RUST_VC_RESPONSE_THRESHOLD") {
        cfg.response_threshold = v;
    }
    if let Some(v) = env_f32("RUST_VC_VAD_ON_THRESHOLD") {
        cfg.vad_on_threshold = v;
        cfg.response_threshold = v;
    } else {
        cfg.vad_on_threshold = cfg.response_threshold;
    }
    if let Some(v) = env_f32("RUST_VC_VAD_OFF_THRESHOLD") {
        cfg.vad_off_threshold = v;
    } else if cfg.vad_on_threshold != 0.0 {
        cfg.vad_off_threshold = derive_vad_off_threshold(cfg.vad_on_threshold);
    } else {
        cfg.vad_off_threshold = 0.0;
    }
    if let Some(v) = env_u32("RUST_VC_SOLA_SEARCH_MS") {
        cfg.sola_search_ms = v.max(1);
    }
    if let Some(v) = env_u32("RUST_VC_OUTPUT_TAIL_OFFSET_MS") {
        cfg.output_tail_offset_ms = v;
    }
    if let Some(v) = env_usize("RUST_VC_OUTPUT_SLICE_OFFSET_SAMPLES") {
        cfg.output_slice_offset_samples = v;
    }
    if let Some(v) = env_bool("RUST_VC_BYPASS_SLICING") {
        cfg.bypass_slicing = v;
    }
    log_debug(&format!(
        "default_runtime_config ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} sola_search_ms={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={}",
        cfg.ort_provider,
        cfg.ort_device_id,
        cfg.ort_gpu_mem_limit_mb,
        cfg.intra_threads,
        cfg.inter_threads,
        cfg.ort_parallel_execution,
        cfg.hubert_context_samples_16k,
        cfg.hubert_output_layer,
        cfg.hubert_upsample_factor,
        cfg.cuda_conv_algo,
        cfg.cuda_ws,
        cfg.cuda_conv1d_pad_to_nc1d,
        cfg.cuda_tf32,
        cfg.index_bin_dim,
        cfg.index_max_vectors,
        cfg.target_buffer_ms,
        cfg.response_threshold,
        cfg.vad_on_threshold,
        cfg.vad_off_threshold,
        cfg.sola_search_ms,
        cfg.output_tail_offset_ms,
        cfg.output_slice_offset_samples,
        cfg.bypass_slicing,
        cfg.record_dump
    ));

    cfg
}

fn derive_vad_off_threshold(vad_on_threshold: f32) -> f32 {
    if !vad_on_threshold.is_finite() || vad_on_threshold == 0.0 {
        return 0.0;
    }
    if vad_on_threshold < 0.0 {
        (vad_on_threshold - 15.0).max(-120.0)
    } else {
        (vad_on_threshold * 0.177_827_94).max(0.0)
    }
}

fn read_mode_block_size(model_dir: &Path) -> usize {
    let mode_path = model_dir.join("mode.txt");
    match fs::read_to_string(&mode_path) {
        Ok(contents) => {
            let trimmed = contents.trim();
            match trimmed.parse::<usize>() {
                Ok(bs) if matches!(bs, 12_000 | 24_000 | 48_000) => {
                    log_debug(&format!("mode.txt: block_size={}", bs));
                    bs
                }
                _ => {
                    log_debug(&format!(
                        "warning: mode.txt: invalid value '{}', defaulting to 24000",
                        trimmed
                    ));
                    24_000
                }
            }
        }
        Err(_) => {
            log_debug("mode.txt not found, defaulting to block_size=24000");
            24_000
        }
    }
}

fn apply_mode_selection(model: &mut ModelConfig, runtime_config: &mut RuntimeConfig) {
    let model_path = Path::new(&model.model_path);
    let model_dir = model_path.parent().unwrap_or(Path::new("."));
    let mode_block_size = read_mode_block_size(model_dir);

    if runtime_config.block_size != mode_block_size {
        log_debug(&format!(
            "mode.txt: runtime block_size overridden {} -> {}",
            runtime_config.block_size, mode_block_size
        ));
    }
    runtime_config.block_size = mode_block_size;

    let hubert_path = model_dir.join(format!("hubert_b{}.onnx", mode_block_size));
    if hubert_path.is_file() {
        model.hubert_path = Some(hubert_path.to_string_lossy().to_string());
    } else {
        log_debug(&format!(
            "warning: auto-selected hubert not found: {} (keeping existing {:?})",
            hubert_path.display(),
            model.hubert_path
        ));
    }
    log_debug(&format!("auto-selected hubert: {:?}", model.hubert_path));

    let rmvpe_path = model_dir.join(format!("rmvpe_b{}.onnx", mode_block_size));
    if rmvpe_path.is_file() {
        model.pitch_extractor_path = Some(rmvpe_path.to_string_lossy().to_string());
    } else {
        log_debug(&format!(
            "warning: auto-selected rmvpe not found: {} (keeping existing {:?})",
            rmvpe_path.display(),
            model.pitch_extractor_path
        ));
    }
    log_debug(&format!("auto-selected rmvpe: {:?}", model.pitch_extractor_path));
}

fn resolve_model_config_for_start(mut model: ModelConfig) -> Result<ModelConfig, String> {
    if let Some(resolved) = resolve_existing_path(&model.model_path) {
        model.model_path = resolved;
    } else if let Some(found) = find_first_onnx_in_model_dir() {
        log_debug(&format!(
            "model_path '{}' not found; fallback to '{}'",
            model.model_path, found
        ));
        model.model_path = found;
    } else {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        return Err(format!(
            "configuration error: model file not found: {} (cwd: {cwd})",
            model.model_path
        ));
    }

    model.pitch_extractor_path = model
        .pitch_extractor_path
        .and_then(|p| resolve_existing_path(&p).or(Some(p)));
    model.hubert_path = model
        .hubert_path
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .map(|p| prefer_padded_hubert_path(&p));
    if model.hubert_path.is_none() {
        model.hubert_path = resolve_existing_path("model/hubert_pad80.onnx")
            .or_else(|| resolve_existing_path("model/hubert_strict.onnx"))
            .or_else(|| resolve_existing_path("model/hubert.onnx"));
    }
    model.index_path = model
        .index_path
        .and_then(|p| resolve_existing_path(&p).or(Some(p)));
    Ok(model)
}

fn prefer_padded_hubert_path(path: &str) -> String {
    let p = Path::new(path);
    let Some(file_name) = p.file_name().and_then(|s| s.to_str()) else {
        return path.to_string();
    };
    let name_lc = file_name.to_ascii_lowercase();
    if name_lc != "hubert_strict.onnx" && name_lc != "hubert.onnx" {
        return path.to_string();
    }
    let Some(parent) = p.parent() else {
        return path.to_string();
    };
    let padded = parent.join("hubert_pad80.onnx");
    if padded.exists() {
        let swapped = padded.to_string_lossy().to_string();
        if swapped != path {
            log_debug(&format!(
                "hubert_path '{}' overridden to '{}' (prefer pad80 contract)",
                path, swapped
            ));
        }
        return swapped;
    }
    path.to_string()
}

fn resolve_existing_path(path: &str) -> Option<String> {
    let p = Path::new(path);
    if p.is_absolute() && p.exists() {
        return Some(p.to_string_lossy().to_string());
    }

    for base in candidate_bases() {
        let cand = base.join(path);
        if cand.exists() {
            return Some(cand.to_string_lossy().to_string());
        }
    }
    None
}

fn find_first_onnx_in_model_dir() -> Option<String> {
    for base in candidate_bases() {
        let dir = base.join("model");
        if !dir.is_dir() {
            continue;
        }
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("onnx"))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        if let Some(first) = files.first() {
            return Some(first.to_string_lossy().to_string());
        }
    }
    None
}

fn find_first_bin_in_model_dir() -> Option<String> {
    for base in candidate_bases() {
        let dir = base.join("model");
        if !dir.is_dir() {
            continue;
        }
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("bin"))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        if let Some(first) = files.first() {
            return Some(first.to_string_lossy().to_string());
        }
    }
    None
}

fn candidate_bases() -> Vec<PathBuf> {
    let mut out = Vec::<PathBuf>::new();
    if let Ok(mut p) = std::env::current_dir() {
        out.push(p.clone());
        for _ in 0..4 {
            if let Some(parent) = p.parent() {
                p = parent.to_path_buf();
                out.push(p.clone());
            } else {
                break;
            }
        }
    }
    out
}

fn prepend_path_dir(dir: &Path) {
    let dir_str = dir.to_string_lossy().to_string();
    if dir_str.is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    let current_s = current.to_string_lossy().to_string();
    let exists = current_s
        .split(';')
        .any(|p| p.eq_ignore_ascii_case(&dir_str));
    if exists {
        return;
    }
    let new_path = if current_s.is_empty() {
        dir_str.clone()
    } else {
        format!("{dir_str};{current_s}")
    };
    std::env::set_var("PATH", &new_path);
    log_debug(&format!("prepended DLL search path: {}", dir_str));
}

fn set_ort_dylib_path(path: &str, source_label: &str) {
    std::env::set_var("ORT_DYLIB_PATH", path);
    if let Some(parent) = Path::new(path).parent() {
        prepend_path_dir(parent);
    }
    log_debug(&format!(
        "ORT_DYLIB_PATH auto-set from {}: {}",
        source_label, path
    ));
}

fn ort_bundle_has_provider_shared(onnxruntime_dll_path: &str) -> bool {
    let dll_path = Path::new(onnxruntime_dll_path);
    let Some(dir) = dll_path.parent() else {
        return false;
    };
    dir.join("onnxruntime_providers_shared.dll").exists()
}

fn ensure_ort_dylib_path() -> Option<String> {
    let mut incomplete_candidate: Option<String> = None;

    if let Ok(current) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(&current);
        if p.exists() {
            if ort_bundle_has_provider_shared(&current) {
                return Some(current);
            }
            log_debug(&format!(
                "ORT_DYLIB_PATH points to '{}' but onnxruntime_providers_shared.dll is missing nearby; trying fallback candidates",
                current
            ));
            incomplete_candidate = Some(current);
        } else {
            log_debug(&format!(
                "ORT_DYLIB_PATH is set but not found on disk: {}",
                current
            ));
        }
    }

    if let Some(path) = resolve_existing_path("model/onnxruntime.dll") {
        if ort_bundle_has_provider_shared(&path) {
            set_ort_dylib_path(&path, "model directory");
            return Some(path);
        }
        log_debug(&format!(
            "model/onnxruntime.dll found but onnxruntime_providers_shared.dll is missing; skipping model bundle candidate: {}",
            path
        ));
        incomplete_candidate.get_or_insert(path);
    }

    if let Some(path) = find_onnxruntime_dll_via_python() {
        set_ort_dylib_path(&path, "python onnxruntime");
        return Some(path);
    }

    if let Some(path) = incomplete_candidate {
        set_ort_dylib_path(&path, "incomplete fallback candidate");
        log_debug(&format!(
            "using incomplete ORT candidate (fallback): {}. provider startup may fail until runtime DLL bundle is fixed",
            path
        ));
        return Some(path);
    }

    None
}

fn find_onnxruntime_dll_via_python() -> Option<String> {
    let py = r#"import os
import onnxruntime as ort
print(os.path.join(os.path.dirname(ort.__file__), "capi", "onnxruntime.dll"))
print(",".join(ort.get_available_providers()))"#;

    let output = Command::new("python").args(["-c", py]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let candidate = lines.next().unwrap_or("").trim().to_string();
    let providers_line = lines.next().unwrap_or("").trim().to_string();
    if candidate.is_empty() {
        return None;
    }
    if !providers_line.is_empty() {
        let providers: Vec<String> = providers_line
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let has_gpu = providers.iter().any(|p| {
            p.eq_ignore_ascii_case("CUDAExecutionProvider")
                || p.eq_ignore_ascii_case("DmlExecutionProvider")
        });
        log_debug(&format!(
            "python onnxruntime providers={:?} gpu_available={}",
            providers, has_gpu
        ));
    }
    if Path::new(&candidate).exists() {
        Some(candidate)
    } else {
        None
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices_cmd,
            get_runtime_config_cmd,
            set_runtime_config_cmd,
            get_model_config_cmd,
            set_model_config_cmd,
            start_engine_cmd,
            stop_engine_cmd,
            get_engine_status_cmd
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
