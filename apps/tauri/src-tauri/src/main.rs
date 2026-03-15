#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    any::Any,
    ffi::{c_char, CStr},
    fs,
    io::{self, Read, Write},
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use libloading::{Library, Symbol};
use reqwest::blocking::Client;
use serde::Serialize;
use tauri::{Emitter, State};
use vc_audio::{
    list_input_devices, list_output_devices, spawn_voice_changer_stream, AudioStreamOptions,
    RealtimeAudioEngine,
};
use vc_core::{set_log_sink, ModelConfig, RuntimeConfig, VoiceChanger};
use vc_inference::RvcOrtEngine;
use windows::{
    core::PCWSTR,
    Win32::System::LibraryLoader::{
        AddDllDirectory, SetDefaultDllDirectories, SetDllDirectoryW, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LOAD_LIBRARY_SEARCH_USER_DIRS,
    },
};

fn log_debug(message: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    eprintln!("[tauri:{ts}] {message}");
}

#[derive(Debug, Clone, Serialize)]
struct UiLogEvent {
    level: String,
    message: String,
    ts: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OrtSetupProgress {
    status: String,
    progress: f32,
    message: String,
}

#[derive(Debug, Clone)]
struct OrtSetupState {
    status: String,
    progress: f32,
    message: String,
    ready: bool,
}

impl Default for OrtSetupState {
    fn default() -> Self {
        Self {
            status: "checking".to_string(),
            progress: 0.0,
            message: "ONNX Runtime setup pending".to_string(),
            ready: false,
        }
    }
}

const ORT_VERSION: &str = "1.23.0";
const ORT_CUDA_DOWNLOAD_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-win-x64-gpu-1.23.0.zip";
const ORT_DML_DOWNLOAD_URL: &str =
    "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.0";
const DIRECTML_RUNTIME_DOWNLOAD_URL: &str =
    "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4";
const ORT_CUDA_REQUIRED_DLLS: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
];
const ORT_CUDA_OPTIONAL_REMOVE_DLLS: &[&str] = &["onnxruntime_providers_tensorrt.dll"];
const ORT_DML_REQUIRED_DLLS: &[&str] = &["onnxruntime.dll", "onnxruntime_providers_shared.dll"];
const DIRECTML_RUNTIME_REQUIRED_DLLS: &[&str] = &["DirectML.dll"];
const CUDA_REQUIRED_DLLS: &[&str] = &[
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    // cuDNN 9.x runtime in this bundle may still lazily request CUDA 11 cublas/nvrtc names.
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "nvrtc64_112_0.dll",
    "nvrtc-builtins64_118.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
];
const CUDNN_REQUIRED_DLLS: &[&str] = &[
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
];

struct RuntimeDllBundle {
    label: &'static str,
    url: &'static str,
    download_status: &'static str,
    extract_status: &'static str,
    dlls: &'static [&'static str],
}

const CUDA_REDIST_BUNDLES: &[RuntimeDllBundle] = &[
    RuntimeDllBundle {
        label: "CUDA Runtime",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cudart64_12.dll"],
    },
    RuntimeDllBundle {
        label: "cuBLAS",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cublas64_12.dll", "cublasLt64_12.dll"],
    },
    RuntimeDllBundle {
        label: "cuBLAS (compat CUDA11 names)",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-11.11.3.6-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cublas64_11.dll", "cublasLt64_11.dll"],
    },
    RuntimeDllBundle {
        label: "NVRTC (compat CUDA11 names)",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-11.8.89-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["nvrtc64_112_0.dll", "nvrtc-builtins64_118.dll"],
    },
    RuntimeDllBundle {
        label: "cuFFT",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.2.1.3-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cufft64_11.dll"],
    },
    RuntimeDllBundle {
        label: "cuRAND",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcurand/windows-x86_64/libcurand-windows-x86_64-10.3.5.147-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["curand64_10.dll"],
    },
    RuntimeDllBundle {
        label: "cuSOLVER",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcusolver/windows-x86_64/libcusolver-windows-x86_64-11.6.1.9-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cusolver64_11.dll"],
    },
    RuntimeDllBundle {
        label: "cuSPARSE",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-12.3.1.170-archive.zip",
        download_status: "downloading_cuda",
        extract_status: "extracting_cuda",
        dlls: &["cusparse64_12.dll"],
    },
];

const CUDNN_REDIST_BUNDLES: &[RuntimeDllBundle] = &[RuntimeDllBundle {
    label: "cuDNN 9.1",
    url: "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.1.1.17_cuda12-archive.zip",
    download_status: "downloading_cudnn",
    extract_status: "extracting_cudnn",
    dlls: CUDNN_REQUIRED_DLLS,
}];
static ORT_SETUP_STATE: OnceLock<Mutex<OrtSetupState>> = OnceLock::new();

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn emit_log(app: &tauri::AppHandle, level: &str, message: &str) {
    let payload = UiLogEvent {
        level: level.to_string(),
        message: message.to_string(),
        ts: now_millis(),
    };
    let _ = app.emit("vc-log", payload);
}

fn ort_setup_state() -> &'static Mutex<OrtSetupState> {
    ORT_SETUP_STATE.get_or_init(|| Mutex::new(OrtSetupState::default()))
}

fn snapshot_ort_setup_state() -> OrtSetupState {
    match ort_setup_state().lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

fn update_ort_setup_state(status: &str, progress: f32, message: String, ready: bool) {
    match ort_setup_state().lock() {
        Ok(mut guard) => {
            guard.status = status.to_string();
            guard.progress = progress.clamp(0.0, 1.0);
            guard.message = message;
            guard.ready = ready;
        }
        Err(mut poisoned) => {
            let guard = poisoned.get_mut();
            guard.status = status.to_string();
            guard.progress = progress.clamp(0.0, 1.0);
            guard.message = message;
            guard.ready = ready;
        }
    }
}

fn emit_ort_setup_progress(
    app: &tauri::AppHandle,
    status: &str,
    progress: f32,
    message: impl Into<String>,
) {
    let message = message.into();
    update_ort_setup_state(status, progress, message.clone(), status == "done");
    let payload = OrtSetupProgress {
        status: status.to_string(),
        progress: progress.clamp(0.0, 1.0),
        message,
    };
    let _ = app.emit("ort-setup-progress", payload);
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
    ort_ready: bool,
    ort_setup_status: String,
    ort_setup_message: String,
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
    let clamped_nprobe = config.index_nprobe.max(1).min(128);
    if clamped_nprobe != config.index_nprobe {
        log_debug(&format!(
            "set_runtime_config_cmd index_nprobe clamped: {} -> {}",
            config.index_nprobe, clamped_nprobe
        ));
        config.index_nprobe = clamped_nprobe;
    }
    let normalized_index_provider = match config.index_provider.trim().to_ascii_lowercase().as_str()
    {
        "gpu" => "gpu",
        _ => "cpu",
    };
    if config.index_provider != normalized_index_provider {
        log_debug(&format!(
            "set_runtime_config_cmd index_provider normalized: '{}' -> '{}'",
            config.index_provider, normalized_index_provider
        ));
        config.index_provider = normalized_index_provider.to_string();
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
    if !config.vad_hysteresis.is_finite() {
        config.vad_hysteresis = 0.5;
    }
    let clamped_vad_hysteresis = config.vad_hysteresis.clamp(0.1, 1.0);
    if (clamped_vad_hysteresis - config.vad_hysteresis).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd vad_hysteresis clamped: {} -> {}",
            config.vad_hysteresis, clamped_vad_hysteresis
        ));
        config.vad_hysteresis = clamped_vad_hysteresis;
    }
    if !config.noise_suppress_db.is_finite() {
        config.noise_suppress_db = -12.0;
    }
    let clamped_noise_suppress_db = config.noise_suppress_db.clamp(-30.0, 0.0);
    if (clamped_noise_suppress_db - config.noise_suppress_db).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd noise_suppress_db clamped: {} -> {}",
            config.noise_suppress_db, clamped_noise_suppress_db
        ));
        config.noise_suppress_db = clamped_noise_suppress_db;
    }
    if !config.noise_suppress_learn_sec.is_finite() {
        config.noise_suppress_learn_sec = 1.5;
    }
    let clamped_noise_learn = config.noise_suppress_learn_sec.clamp(0.1, 5.0);
    if (clamped_noise_learn - config.noise_suppress_learn_sec).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd noise_suppress_learn_sec clamped: {} -> {}",
            config.noise_suppress_learn_sec, clamped_noise_learn
        ));
        config.noise_suppress_learn_sec = clamped_noise_learn;
    }
    if !config.frame_gate_db.is_finite() {
        config.frame_gate_db = -60.0;
    }
    let clamped_frame_gate_db = config.frame_gate_db.clamp(-80.0, -30.0);
    if (clamped_frame_gate_db - config.frame_gate_db).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd frame_gate_db clamped: {} -> {}",
            config.frame_gate_db, clamped_frame_gate_db
        ));
        config.frame_gate_db = clamped_frame_gate_db;
    }
    if !config.hubert_context_sec.is_finite() {
        config.hubert_context_sec = 0.5;
    }
    let clamped_hubert_context_sec = config.hubert_context_sec.clamp(0.25, 2.0);
    if (clamped_hubert_context_sec - config.hubert_context_sec).abs() > f32::EPSILON {
        log_debug(&format!(
            "set_runtime_config_cmd hubert_context_sec clamped: {} -> {}",
            config.hubert_context_sec, clamped_hubert_context_sec
        ));
        config.hubert_context_sec = clamped_hubert_context_sec;
    }
    config.hubert_context_samples_16k =
        (config.hubert_context_sec * 16_000.0).round().max(1.0) as usize;
    config.sola_reset_threshold_samples = config
        .sola_reset_threshold_samples
        .clamp(1, config.block_size.max(1));
    // Keep legacy field aligned for older subsystems and saved presets.
    config.response_threshold = config.vad_on_threshold;
    log_debug(&format!(
        "set_runtime_config_cmd sample_rate={} block_size={} in_dev={:?} out_dev={:?} extra_ms={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} vad_hysteresis={:.2} noise_suppress={} noise_suppress_db={:.1} noise_learn_sec={:.1} frame_gate_db={:.1} fade_in_ms={} fade_out_ms={} sola_search_ms={} sola_reset_threshold={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={} pitch_shift={:.2} index_rate={} index_smooth={:.2} top_k={} rows={} nprobe={} index_provider={} protect={:.2} rmvpe_th={:.3} pitch_smooth={:.2} rms_mix={:.2} post_filter={:.3} f0_med_r={} ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_sec={:.2} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={}",
        config.sample_rate,
        config.block_size,
        config.input_device_name,
        config.output_device_name,
        config.extra_inference_ms,
        config.target_buffer_ms,
        config.response_threshold,
        config.vad_on_threshold,
        config.vad_off_threshold,
        config.vad_hysteresis,
        config.noise_suppress,
        config.noise_suppress_db,
        config.noise_suppress_learn_sec,
        config.frame_gate_db,
        config.fade_in_ms,
        config.fade_out_ms,
        config.sola_search_ms,
        config.sola_reset_threshold_samples,
        config.output_tail_offset_ms,
        config.output_slice_offset_samples,
        config.bypass_slicing,
        config.record_dump,
        config.pitch_shift_semitones,
        config.index_rate,
        config.index_smooth_alpha,
        config.index_top_k,
        config.index_search_rows,
        config.index_nprobe,
        config.index_provider,
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
        config.hubert_context_sec,
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
    let index_nprobe = config.index_nprobe;
    let index_provider = config.index_provider.clone();
    guard.config = config;
    if running {
        if let Some(task) = guard.engine_task.as_ref() {
            task.update_index_search_params(index_rows, index_top_k, index_nprobe, &index_provider)
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
fn start_engine_cmd(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("start_engine_cmd", || {
        log_debug("start_engine_cmd");
        emit_log(&app, "info", "start_engine_cmd");
        let (mut runtime_config, model_raw) = {
            let mut guard = state.lock_runtime();
            if guard.running {
                log_debug("engine already running");
                sync_levels_from_engine(&mut guard);
                let ort = snapshot_ort_setup_state();
                return Ok(EngineStatus {
                    running: guard.running,
                    input_level_rms: guard.input_level_rms,
                    input_level_peak: guard.input_level_peak,
                    ort_ready: ort.ready,
                    ort_setup_status: ort.status,
                    ort_setup_message: ort.message,
                });
            }
            (guard.config.clone(), guard.model.clone())
        };
        let ort = snapshot_ort_setup_state();
        if !ort.ready {
            return Err(format!(
                "ONNX Runtime setup is not ready: {} ({})",
                ort.message, ort.status
            ));
        }
        let model = resolve_model_config_for_start(model_raw)?;
        apply_mode_selection(&model, &mut runtime_config);
        let model_dir = resolve_model_dir();
        let ort_dylib = ensure_ort_dylib_path_for_provider(&model_dir, &runtime_config.ort_provider)?;
        log_debug(&format!(
            "ORT_DYLIB_PATH={:?} provider_bundle={}",
            std::env::var("ORT_DYLIB_PATH").ok(),
            ort_dylib
        ));
        log_debug(&format!(
            "start with model={} hubert={:?} rmvpe={:?} index={:?} sr={} block={} in_dev={:?} out_dev={:?} extra_ms={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} vad_hysteresis={:.2} noise_suppress={} noise_suppress_db={:.1} noise_learn_sec={:.1} frame_gate_db={:.1} fade_in_ms={} fade_out_ms={} sola_search_ms={} sola_reset_threshold={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={} pitch_shift={:.2} index_rate={} index_smooth={:.2} top_k={} rows={} nprobe={} index_provider={} protect={:.2} rmvpe_th={:.3} pitch_smooth={:.2} rms_mix={:.2} post_filter={:.3} f0_med_r={} ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_sec={:.2} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={}",
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
            runtime_config.vad_hysteresis,
            runtime_config.noise_suppress,
            runtime_config.noise_suppress_db,
            runtime_config.noise_suppress_learn_sec,
            runtime_config.frame_gate_db,
            runtime_config.fade_in_ms,
            runtime_config.fade_out_ms,
            runtime_config.sola_search_ms,
            runtime_config.sola_reset_threshold_samples,
            runtime_config.output_tail_offset_ms,
            runtime_config.output_slice_offset_samples,
            runtime_config.bypass_slicing,
            runtime_config.record_dump,
            runtime_config.pitch_shift_semitones,
            runtime_config.index_rate,
            runtime_config.index_smooth_alpha,
            runtime_config.index_top_k,
            runtime_config.index_search_rows,
            runtime_config.index_nprobe,
            runtime_config.index_provider,
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
            runtime_config.hubert_context_sec,
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
        let stateful_generator_model = infer_engine
            .model_path()
            .to_ascii_lowercase()
            .contains("stateful");
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
                vad_hysteresis: runtime_config.vad_hysteresis,
                noise_suppress: runtime_config.noise_suppress,
                noise_suppress_db: runtime_config.noise_suppress_db,
                noise_suppress_learn_sec: runtime_config.noise_suppress_learn_sec,
                frame_gate_db: runtime_config.frame_gate_db,
                fade_in_ms: runtime_config.fade_in_ms,
                fade_out_ms: runtime_config.fade_out_ms,
                sola_search_ms: runtime_config.sola_search_ms,
                sola_reset_threshold_samples: runtime_config.sola_reset_threshold_samples,
                output_tail_offset_ms: runtime_config.output_tail_offset_ms,
                output_slice_offset_samples: runtime_config.output_slice_offset_samples,
                bypass_slicing: runtime_config.bypass_slicing,
                stateful_generator_model,
                record_dump: runtime_config.record_dump,
            },
        )
        .map_err(|e| e.to_string())?;

        let mut guard = state.lock_runtime();
        if guard.running {
            log_debug("engine already running");
            sync_levels_from_engine(&mut guard);
            audio_engine.stop_and_abort();
            let ort = snapshot_ort_setup_state();
            return Ok(EngineStatus {
                running: guard.running,
                input_level_rms: guard.input_level_rms,
                input_level_peak: guard.input_level_peak,
                ort_ready: ort.ready,
                ort_setup_status: ort.status,
                ort_setup_message: ort.message,
            });
        }

        guard.running = true;
        guard.config = runtime_config;
        guard.engine_task = Some(audio_engine);
        sync_levels_from_engine(&mut guard);
        emit_log(&app, "info", "start_engine_cmd done");
        let ort = snapshot_ort_setup_state();
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
            ort_ready: ort.ready,
            ort_setup_status: ort.status,
            ort_setup_message: ort.message,
        })
    })
}

#[tauri::command]
fn stop_engine_cmd(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("stop_engine_cmd", || {
        log_debug("stop_engine_cmd");
        emit_log(&app, "info", "stop_engine_cmd");
        let mut guard = state.lock_runtime();
        if let Some(task) = guard.engine_task.take() {
            task.stop_and_abort();
        }
        guard.running = false;
        emit_log(&app, "info", "stop_engine_cmd done");
        let ort = snapshot_ort_setup_state();
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
            ort_ready: ort.ready,
            ort_setup_status: ort.status,
            ort_setup_message: ort.message,
        })
    })
}

#[tauri::command]
fn get_engine_status_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    run_panic_safe("get_engine_status_cmd", || {
        let mut guard = state.lock_runtime();
        sync_levels_from_engine(&mut guard);
        let ort = snapshot_ort_setup_state();
        Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
            ort_ready: ort.ready,
            ort_setup_status: ort.status,
            ort_setup_message: ort.message,
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
    let model_path = std::env::var("RUST_VC_MODEL_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/model.onnx"))
        .or_else(find_first_onnx_in_model_dir)
        .unwrap_or_else(|| "model/model.onnx".to_string());
    let model_dir = Path::new(&model_path).parent().unwrap_or(Path::new("."));

    let pitch_extractor_path = std::env::var("RUST_VC_RMVPE_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_sibling_if_exists(model_dir, "rmvpe_strict.onnx"));
    let hubert_path = std::env::var("RUST_VC_HUBERT_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_sibling_if_exists(model_dir, "hubert_pad80.onnx"));
    let index_path = std::env::var("RUST_VC_INDEX_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/model_vectors.bin"))
        .or_else(find_first_bin_in_model_dir)
        .or_else(|| resolve_existing_path("model/model.index"))
        .or_else(|| resolve_existing_path("model/feature.index"));

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
    cfg.output_slice_offset_samples = 10_800;
    cfg.bypass_slicing = true;
    cfg.hubert_context_samples_16k = (cfg.hubert_context_sec * 16_000.0).round().max(1.0) as usize;
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
        cfg.hubert_context_sec =
            (cfg.hubert_context_samples_16k as f32 / 16_000.0).clamp(0.25, 2.0);
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
    if let Some(v) = env_u32("RUST_VC_INDEX_NPROBE") {
        cfg.index_nprobe = v.max(1).min(128);
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
    if let Some(v) = env_f32("RUST_VC_VAD_HYSTERESIS") {
        cfg.vad_hysteresis = v.clamp(0.1, 1.0);
    }
    if let Some(v) = env_bool("RUST_VC_NOISE_SUPPRESS") {
        cfg.noise_suppress = v;
    }
    if let Some(v) = env_f32("RUST_VC_NOISE_SUPPRESS_DB") {
        cfg.noise_suppress_db = v.clamp(-30.0, 0.0);
    }
    if let Some(v) = env_f32("RUST_VC_NOISE_SUPPRESS_LEARN_SEC") {
        cfg.noise_suppress_learn_sec = v.clamp(0.1, 5.0);
    }
    if let Some(v) = env_f32("RUST_VC_FRAME_GATE_DB") {
        cfg.frame_gate_db = v.clamp(-80.0, -30.0);
    }
    if let Some(v) = env_usize("RUST_VC_SOLA_RESET_THRESHOLD") {
        cfg.sola_reset_threshold_samples = v.max(1);
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
    if let Some(v) = env_f32("RUST_VC_HUBERT_CONTEXT_SEC") {
        cfg.hubert_context_sec = v.clamp(0.25, 2.0);
        cfg.hubert_context_samples_16k = (cfg.hubert_context_sec * 16_000.0)
            .round()
            .max(1.0) as usize;
    }
    log_debug(&format!(
        "default_runtime_config ort_provider={} ort_dev={} ort_vram_mb={} ort_threads={}/{} ort_parallel={} hubert_ctx_sec={:.2} hubert_ctx_16k={} hubert_layer={} hubert_up={} cuda_conv_algo={} cuda_ws={} cuda_pad_nc1d={} cuda_tf32={} index_bin_dim={} index_max_vectors={} index_nprobe={} index_provider={} target_buffer_ms={} threshold={:.4} vad_on={:.4} vad_off={:.4} vad_hysteresis={:.2} noise_suppress={} noise_suppress_db={:.1} noise_learn_sec={:.1} frame_gate_db={:.1} sola_search_ms={} sola_reset_threshold={} tail_offset_ms={} slice_offset_samples={} bypass_slicing={} record_dump={}",
        cfg.ort_provider,
        cfg.ort_device_id,
        cfg.ort_gpu_mem_limit_mb,
        cfg.intra_threads,
        cfg.inter_threads,
        cfg.ort_parallel_execution,
        cfg.hubert_context_sec,
        cfg.hubert_context_samples_16k,
        cfg.hubert_output_layer,
        cfg.hubert_upsample_factor,
        cfg.cuda_conv_algo,
        cfg.cuda_ws,
        cfg.cuda_conv1d_pad_to_nc1d,
        cfg.cuda_tf32,
        cfg.index_bin_dim,
        cfg.index_max_vectors,
        cfg.index_nprobe,
        cfg.index_provider,
        cfg.target_buffer_ms,
        cfg.response_threshold,
        cfg.vad_on_threshold,
        cfg.vad_off_threshold,
        cfg.vad_hysteresis,
        cfg.noise_suppress,
        cfg.noise_suppress_db,
        cfg.noise_suppress_learn_sec,
        cfg.frame_gate_db,
        cfg.sola_search_ms,
        cfg.sola_reset_threshold_samples,
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
                Ok(bs) if matches!(bs, 12_000 | 24_000 | 48_000) => bs,
                _ => 24_000,
            }
        }
        Err(_) => 24_000,
    }
}

fn apply_mode_selection(model: &ModelConfig, runtime_config: &mut RuntimeConfig) {
    let model_path = Path::new(&model.model_path);
    let model_dir = model_path.parent().unwrap_or(Path::new("."));
    let mode_block_size = read_mode_block_size(model_dir);
    let configured_block_size = runtime_config.block_size;
    if matches!(configured_block_size, 12_000 | 24_000 | 48_000) {
        if configured_block_size == mode_block_size {
            log_debug(&format!("mode.txt: block_size={mode_block_size}"));
        } else {
            log_debug(&format!(
                "mode.txt ignored: block_size={} (runtime block_size={})",
                mode_block_size, configured_block_size
            ));
        }
        return;
    }

    runtime_config.block_size = mode_block_size;
    log_debug(&format!(
        "mode.txt applied: block_size={} (runtime block_size was {})",
        mode_block_size, configured_block_size
    ));
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
        .and_then(|p| resolve_existing_path(&p).or(Some(p)));
    model.index_path = model
        .index_path
        .and_then(|p| resolve_existing_path(&p).or(Some(p)));
    Ok(model)
}

fn resolve_sibling_if_exists(model_dir: &Path, file_name: &str) -> Option<String> {
    let candidate = model_dir.join(file_name);
    if candidate.is_file() {
        Some(candidate.to_string_lossy().to_string())
    } else {
        None
    }
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

fn resolve_model_dir() -> PathBuf {
    for base in candidate_bases() {
        let model_dir = base.join("model");
        if model_dir.is_dir() {
            return model_dir;
        }
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("model")
}

fn utf16_null(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn prepare_ort_dependency_search_paths(onnxruntime_dll_path: &str) {
    let Some(model_dir) = Path::new(onnxruntime_dll_path).parent() else {
        return;
    };
    let wide = utf16_null(&model_dir.to_string_lossy());
    unsafe {
        let _ = SetDllDirectoryW(PCWSTR::null());
        let _ = SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS);
        let _ = AddDllDirectory(PCWSTR(wide.as_ptr()));
    }
    log_debug(&format!(
        "ORT DLL search restricted to model directory: {}",
        model_dir.display()
    ));
}

#[repr(C)]
struct OrtApiBase {
    _get_api: unsafe extern "system" fn(u32) -> *const std::ffi::c_void,
    get_version_string: unsafe extern "system" fn() -> *const c_char,
}

type OrtGetApiBaseFn = unsafe extern "system" fn() -> *const OrtApiBase;

fn get_onnxruntime_version_from_dll(path: &Path) -> Result<String, String> {
    unsafe {
        let lib = Library::new(path)
            .map_err(|e| format!("failed to load '{}': {e}", path.display()))?;
        let get_api_base: Symbol<'_, OrtGetApiBaseFn> = lib
            .get(b"OrtGetApiBase\0")
            .map_err(|e| format!("OrtGetApiBase not found in '{}': {e}", path.display()))?;
        let api_base = get_api_base();
        if api_base.is_null() {
            return Err(format!("OrtGetApiBase returned null for '{}'", path.display()));
        }
        let version_ptr = ((*api_base).get_version_string)();
        if version_ptr.is_null() {
            return Err(format!("GetVersionString returned null for '{}'", path.display()));
        }
        Ok(CStr::from_ptr(version_ptr).to_string_lossy().trim().to_string())
    }
}

fn parse_version_tuple(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_raw = parts.next().unwrap_or("0");
    let patch_digits: String = patch_raw
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let patch = if patch_digits.is_empty() {
        0
    } else {
        patch_digits.parse().ok()?
    };
    Some((major, minor, patch))
}

fn is_supported_ort_version(raw: &str) -> bool {
    parse_version_tuple(raw)
        .map(|(major, minor, _)| major > 1 || (major == 1 && minor >= 23))
        .unwrap_or(false)
}

fn ort_cuda_bundle_dir(model_dir: &Path) -> PathBuf {
    model_dir.join("ort_cuda")
}

fn ort_dml_bundle_dir(model_dir: &Path) -> PathBuf {
    model_dir.join("ort_dml")
}

fn resolve_ort_bundle(model_dir: &Path, provider: &str) -> PathBuf {
    match provider.trim().to_ascii_lowercase().as_str() {
        "directml" => ort_dml_bundle_dir(model_dir),
        _ => ort_cuda_bundle_dir(model_dir),
    }
}

fn ort_bundle_dll_path(bundle_dir: &Path) -> PathBuf {
    bundle_dir.join("onnxruntime.dll")
}

fn check_ort_version(bundle_dir: &Path) -> Result<String, String> {
    get_onnxruntime_version_from_dll(&ort_bundle_dll_path(bundle_dir))
}

fn set_ort_dylib_path(path: &str, source_label: &str) {
    std::env::set_var("ORT_DYLIB_PATH", path);
    prepare_ort_dependency_search_paths(path);
    log_debug(&format!(
        "ORT_DYLIB_PATH auto-set from {}: {}",
        source_label, path
    ));
}

fn bundle_missing_dlls(model_dir: &Path, dlls: &[&str]) -> Vec<String> {
    dlls.iter()
        .filter(|dll| !model_dir.join(dll).exists())
        .map(|dll| (*dll).to_string())
        .collect()
}

fn ort_bundle_has_required_dlls(bundle_dir: &Path, required: &[&str]) -> bool {
    required.iter().all(|name| bundle_dir.join(name).exists())
}

fn ensure_ort_dylib_path_for_provider(model_dir: &Path, provider: &str) -> Result<String, String> {
    let bundle_dir = resolve_ort_bundle(model_dir, provider);
    let required: Vec<&str> = if provider.trim().eq_ignore_ascii_case("directml") {
        ORT_DML_REQUIRED_DLLS
            .iter()
            .chain(DIRECTML_RUNTIME_REQUIRED_DLLS.iter())
            .copied()
            .collect()
    } else {
        ORT_CUDA_REQUIRED_DLLS.to_vec()
    };
    if !ort_bundle_has_required_dlls(&bundle_dir, &required) {
        return Err(format!(
            "ORT bundle is incomplete for provider '{}': {}",
            provider,
            bundle_dir.display()
        ));
    }
    let version = check_ort_version(&bundle_dir)?;
    if !is_supported_ort_version(&version) {
        return Err(format!(
            "ORT bundle version {} is older than required {} in {}",
            version,
            ORT_VERSION,
            bundle_dir.display()
        ));
    }
    let dll_path = ort_bundle_dll_path(&bundle_dir);
    let dll = dll_path.to_string_lossy().to_string();
    set_ort_dylib_path(&dll, &format!("{} provider bundle", provider));
    Ok(dll)
}

fn remove_existing_ort_bundle(bundle_dir: &Path, names: &[&str]) -> Result<(), String> {
    for name in names {
        let path = bundle_dir.join(name);
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| format!("failed to remove '{}': {e}", path.display()))?;
        }
    }
    Ok(())
}

fn move_file_if_present(src: &Path, dst: &Path) -> Result<(), String> {
    if !src.exists() || dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create '{}': {e}", parent.display()))?;
    }
    match fs::rename(src, dst) {
        Ok(_) => Ok(()),
        Err(_) => {
            fs::copy(src, dst).map_err(|e| {
                format!(
                    "failed to copy '{}' -> '{}': {e}",
                    src.display(),
                    dst.display()
                )
            })?;
            fs::remove_file(src)
                .map_err(|e| format!("failed to remove '{}': {e}", src.display()))?;
            Ok(())
        }
    }
}

fn migrate_legacy_cuda_bundle(model_dir: &Path, bundle_dir: &Path) -> Result<(), String> {
    let names: Vec<&str> = ORT_CUDA_REQUIRED_DLLS
        .iter()
        .chain(CUDA_REQUIRED_DLLS.iter())
        .chain(CUDNN_REQUIRED_DLLS.iter())
        .copied()
        .collect();
    for name in names {
        move_file_if_present(&model_dir.join(name), &bundle_dir.join(name))?;
    }
    Ok(())
}

fn download_zip_with_progress(
    app: &tauri::AppHandle,
    client: &Client,
    url: &str,
    zip_path: &Path,
    status: &str,
    progress_start: f32,
    progress_span: f32,
    label: &str,
) -> Result<(), String> {
    let mut response = client
        .get(url)
        .send()
        .map_err(|e| format!("failed to request {label}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "failed to download {label}: HTTP {}",
            response.status()
        ));
    }

    let total_size = response.content_length();
    let mut out = fs::File::create(zip_path)
        .map_err(|e| format!("failed to create '{}': {e}", zip_path.display()))?;
    let mut downloaded = 0u64;
    let mut last_bucket = 0u64;
    let mut buf = vec![0u8; 64 * 1024];
    emit_ort_setup_progress(
        app,
        status,
        progress_start,
        format!("Downloading {label}"),
    );
    loop {
        let read = response
            .read(&mut buf)
            .map_err(|e| format!("failed while downloading {label}: {e}"))?;
        if read == 0 {
            break;
        }
        out.write_all(&buf[..read])
            .map_err(|e| format!("failed to write '{}': {e}", zip_path.display()))?;
        downloaded += read as u64;
        if let Some(total) = total_size {
            if total > 0 {
                let bucket = (downloaded.saturating_mul(100) / total).min(100);
                if bucket != last_bucket {
                    last_bucket = bucket;
                    emit_ort_setup_progress(
                        app,
                        status,
                        progress_start + (bucket as f32 / 100.0) * progress_span,
                        format!("Downloading {label}... {}%", bucket),
                    );
                }
            }
        }
    }
    Ok(())
}

fn extract_selected_dlls_from_zip(
    app: &tauri::AppHandle,
    zip_path: &Path,
    model_dir: &Path,
    required_dlls: &[&str],
    preferred_roots: &[&str],
    status: &str,
    progress_start: f32,
    progress_span: f32,
    label: &str,
) -> Result<(), String> {
    emit_ort_setup_progress(
        app,
        status,
        progress_start,
        format!("Extracting {label}"),
    );
    let file = fs::File::open(zip_path)
        .map_err(|e| format!("failed to open '{}': {e}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("failed to open zip '{}': {e}", zip_path.display()))?;
    let total_required = required_dlls.len().max(1) as f32;
    let mut extracted = Vec::<String>::new();
    let normalized_roots: Vec<String> = preferred_roots
        .iter()
        .map(|root| root.replace('\\', "/").trim_end_matches('/').to_string() + "/")
        .collect();

    for idx in 0..archive.len() {
        let mut entry = archive
            .by_index(idx)
            .map_err(|e| format!("failed to read zip entry #{idx}: {e}"))?;
        if entry.is_dir() {
            continue;
        }
        let entry_name = entry.name().replace('\\', "/");
        if !normalized_roots.is_empty()
            && !normalized_roots
                .iter()
                .any(|root| entry_name.starts_with(root))
        {
            continue;
        }
        let Some(file_name) = Path::new(&entry_name).file_name() else {
            continue;
        };
        let file_name = file_name.to_string_lossy().to_string();
        if !required_dlls
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&file_name))
        {
            continue;
        }
        let out_path = model_dir.join(&file_name);
        let mut out = fs::File::create(&out_path)
            .map_err(|e| format!("failed to create '{}': {e}", out_path.display()))?;
        io::copy(&mut entry, &mut out)
            .map_err(|e| format!("failed to extract '{}': {e}", out_path.display()))?;
        if !extracted.iter().any(|name| name.eq_ignore_ascii_case(&file_name)) {
            extracted.push(file_name.clone());
        }
        let progress = progress_start + (extracted.len() as f32 / total_required) * progress_span;
        emit_ort_setup_progress(
            app,
            status,
            progress,
            format!("Extracting {label}: {}", file_name),
        );
    }

    for required in required_dlls {
        if !model_dir.join(required).exists() {
            return Err(format!(
                "required DLL '{}' was not found in downloaded archive",
                required
            ));
        }
    }
    Ok(())
}

fn ensure_redist_bundles(
    app: &tauri::AppHandle,
    client: &Client,
    temp_root: &Path,
    model_dir: &Path,
    bundles: &[RuntimeDllBundle],
    status_base_progress: f32,
    status_span: f32,
) -> Result<(), String> {
    let total = bundles.len().max(1) as f32;
    for (index, bundle) in bundles.iter().enumerate() {
        if bundle_missing_dlls(model_dir, bundle.dlls).is_empty() {
            continue;
        }

        let step_start = status_base_progress + (index as f32 / total) * status_span;
        let step_span = status_span / total;
        let download_span = step_span * 0.7;
        let extract_span = step_span * 0.3;

        let file_name = Path::new(bundle.url)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("failed to derive zip filename from '{}'", bundle.url))?;
        let zip_path = temp_root.join(file_name);

        download_zip_with_progress(
            app,
            client,
            bundle.url,
            &zip_path,
            bundle.download_status,
            step_start,
            download_span,
            bundle.label,
        )?;
        extract_selected_dlls_from_zip(
            app,
            &zip_path,
            model_dir,
            bundle.dlls,
            &[],
            bundle.extract_status,
            step_start + download_span,
            extract_span,
            bundle.label,
        )?;

        let _ = fs::remove_file(&zip_path);
    }
    Ok(())
}

fn ensure_ort_cuda_bundle(
    app: &tauri::AppHandle,
    client: &Client,
    temp_root: &Path,
    model_dir: &Path,
) -> Result<String, String> {
    let bundle_dir = ort_cuda_bundle_dir(model_dir);
    fs::create_dir_all(&bundle_dir)
        .map_err(|e| format!("failed to create '{}': {e}", bundle_dir.display()))?;
    migrate_legacy_cuda_bundle(model_dir, &bundle_dir)?;
    for name in ORT_CUDA_OPTIONAL_REMOVE_DLLS {
        let stale = bundle_dir.join(name);
        if stale.exists() {
            let _ = fs::remove_file(&stale);
            log_debug(&format!(
                "removed optional CUDA provider dll '{}'",
                stale.display()
            ));
        }
    }

    let dll_path = ort_bundle_dll_path(&bundle_dir);
    let mut existing_version: Option<String> = None;
    if dll_path.exists() {
        if !ort_bundle_has_required_dlls(&bundle_dir, ORT_CUDA_REQUIRED_DLLS) {
            log_debug(&format!(
                "ORT CUDA bundle is incomplete; refreshing '{}'",
                bundle_dir.display()
            ));
            remove_existing_ort_bundle(&bundle_dir, ORT_CUDA_REQUIRED_DLLS)?;
        } else {
            match check_ort_version(&bundle_dir) {
                Ok(version) if is_supported_ort_version(&version) => {
                    existing_version = Some(version);
                }
                Ok(version) => {
                    log_debug(&format!(
                        "ORT CUDA bundle version {} is older than {}; refreshing '{}'",
                        version,
                        ORT_VERSION,
                        bundle_dir.display()
                    ));
                    remove_existing_ort_bundle(&bundle_dir, ORT_CUDA_REQUIRED_DLLS)?;
                }
                Err(err) => {
                    log_debug(&format!(
                        "failed to inspect ORT CUDA bundle '{}': {}; refreshing",
                        bundle_dir.display(),
                        err
                    ));
                    remove_existing_ort_bundle(&bundle_dir, ORT_CUDA_REQUIRED_DLLS)?;
                }
            }
        }
    }

    let zip_path = temp_root.join("onnxruntime-win-x64-gpu-1.23.0.zip");
    if ort_bundle_has_required_dlls(&bundle_dir, ORT_CUDA_REQUIRED_DLLS) && dll_path.exists() {
        emit_ort_setup_progress(
            app,
            "checking",
            0.10,
            format!("ORT CUDA bundle already present ({})", bundle_dir.display()),
        );
    } else {
        download_zip_with_progress(
            app,
            client,
            ORT_CUDA_DOWNLOAD_URL,
            &zip_path,
            "downloading",
            0.0,
            0.40,
            &format!("ONNX Runtime CUDA {}", ORT_VERSION),
        )?;
        extract_selected_dlls_from_zip(
            app,
            &zip_path,
            &bundle_dir,
            ORT_CUDA_REQUIRED_DLLS,
            &[],
            "extracting",
            0.40,
            0.10,
            "ONNX Runtime CUDA DLLs",
        )?;
    }

    ensure_redist_bundles(app, client, temp_root, &bundle_dir, CUDA_REDIST_BUNDLES, 0.50, 0.22)?;
    ensure_redist_bundles(app, client, temp_root, &bundle_dir, CUDNN_REDIST_BUNDLES, 0.72, 0.10)?;

    let missing_cuda = bundle_missing_dlls(&bundle_dir, CUDA_REQUIRED_DLLS);
    if !missing_cuda.is_empty() {
        return Err(format!(
            "missing CUDA runtime DLLs after setup in '{}': {}",
            bundle_dir.display(),
            missing_cuda.join(", ")
        ));
    }
    let missing_cudnn = bundle_missing_dlls(&bundle_dir, CUDNN_REQUIRED_DLLS);
    if !missing_cudnn.is_empty() {
        return Err(format!(
            "missing cuDNN DLLs after setup in '{}': {}",
            bundle_dir.display(),
            missing_cudnn.join(", ")
        ));
    }
    let version = check_ort_version(&bundle_dir)?;
    if !is_supported_ort_version(&version) {
        return Err(format!(
            "ORT CUDA bundle version is unsupported after refresh: {} ({})",
            version,
            bundle_dir.display()
        ));
    }
    Ok(existing_version.unwrap_or(version))
}

fn ensure_ort_dml_bundle(
    app: &tauri::AppHandle,
    client: &Client,
    temp_root: &Path,
    model_dir: &Path,
) -> Result<String, String> {
    let bundle_dir = ort_dml_bundle_dir(model_dir);
    fs::create_dir_all(&bundle_dir)
        .map_err(|e| format!("failed to create '{}': {e}", bundle_dir.display()))?;

    let dll_path = ort_bundle_dll_path(&bundle_dir);
    let mut existing_version: Option<String> = None;
    if dll_path.exists() {
        if !ort_bundle_has_required_dlls(&bundle_dir, ORT_DML_REQUIRED_DLLS) {
            log_debug(&format!(
                "ORT DirectML bundle is incomplete; refreshing '{}'",
                bundle_dir.display()
            ));
            remove_existing_ort_bundle(&bundle_dir, ORT_DML_REQUIRED_DLLS)?;
        } else {
            match check_ort_version(&bundle_dir) {
                Ok(version) if is_supported_ort_version(&version) => {
                    existing_version = Some(version);
                }
                Ok(version) => {
                    log_debug(&format!(
                        "ORT DirectML bundle version {} is older than {}; refreshing '{}'",
                        version,
                        ORT_VERSION,
                        bundle_dir.display()
                    ));
                    remove_existing_ort_bundle(&bundle_dir, ORT_DML_REQUIRED_DLLS)?;
                }
                Err(err) => {
                    log_debug(&format!(
                        "failed to inspect ORT DirectML bundle '{}': {}; refreshing",
                        bundle_dir.display(),
                        err
                    ));
                    remove_existing_ort_bundle(&bundle_dir, ORT_DML_REQUIRED_DLLS)?;
                }
            }
        }
    }

    let ort_dml_zip_path = temp_root.join("microsoft.ml.onnxruntime.directml.1.23.0.nupkg");
    if ort_bundle_has_required_dlls(&bundle_dir, ORT_DML_REQUIRED_DLLS) && dll_path.exists() {
        emit_ort_setup_progress(
            app,
            "checking",
            0.85,
            format!("ORT DirectML bundle already present ({})", bundle_dir.display()),
        );
    } else {
        download_zip_with_progress(
            app,
            client,
            ORT_DML_DOWNLOAD_URL,
            &ort_dml_zip_path,
            "downloading_directml",
            0.82,
            0.08,
            &format!("ONNX Runtime DirectML {}", ORT_VERSION),
        )?;
        extract_selected_dlls_from_zip(
            app,
            &ort_dml_zip_path,
            &bundle_dir,
            ORT_DML_REQUIRED_DLLS,
            &["runtimes/win-x64/native"],
            "extracting_directml",
            0.90,
            0.04,
            "ORT DirectML DLLs",
        )?;
    }

    if !ort_bundle_has_required_dlls(&bundle_dir, DIRECTML_RUNTIME_REQUIRED_DLLS) {
        let dml_runtime_zip = temp_root.join("microsoft.ai.directml.1.15.4.nupkg");
        download_zip_with_progress(
            app,
            client,
            DIRECTML_RUNTIME_DOWNLOAD_URL,
            &dml_runtime_zip,
            "downloading_directml",
            0.94,
            0.04,
            "DirectML runtime 1.15.4",
        )?;
        extract_selected_dlls_from_zip(
            app,
            &dml_runtime_zip,
            &bundle_dir,
            DIRECTML_RUNTIME_REQUIRED_DLLS,
            &["bin/x64-win"],
            "extracting_directml",
            0.98,
            0.01,
            "DirectML runtime DLLs",
        )?;
    }

    let missing_ort = bundle_missing_dlls(&bundle_dir, ORT_DML_REQUIRED_DLLS);
    if !missing_ort.is_empty() {
        return Err(format!(
            "missing DirectML ORT DLLs after setup in '{}': {}",
            bundle_dir.display(),
            missing_ort.join(", ")
        ));
    }
    let missing_runtime = bundle_missing_dlls(&bundle_dir, DIRECTML_RUNTIME_REQUIRED_DLLS);
    if !missing_runtime.is_empty() {
        return Err(format!(
            "missing DirectML runtime DLLs after setup in '{}': {}",
            bundle_dir.display(),
            missing_runtime.join(", ")
        ));
    }
    let version = check_ort_version(&bundle_dir)?;
    if !is_supported_ort_version(&version) {
        return Err(format!(
            "ORT DirectML bundle version is unsupported after refresh: {} ({})",
            version,
            bundle_dir.display()
        ));
    }
    Ok(existing_version.unwrap_or(version))
}

fn ensure_ort_runtime_bundle(app: &tauri::AppHandle) -> Result<(), String> {
    let model_dir = resolve_model_dir();
    emit_ort_setup_progress(
        app,
        "checking",
        0.0,
        format!("Checking ONNX Runtime {} bundles", ORT_VERSION),
    );
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("failed to create model directory '{}': {e}", model_dir.display()))?;

    let temp_root = std::env::temp_dir().join(format!("rust-vc-ort-{}", now_millis()));
    fs::create_dir_all(&temp_root)
        .map_err(|e| format!("failed to create temp directory '{}': {e}", temp_root.display()))?;
    let client = Client::builder()
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    let result = (|| -> Result<(), String> {
        let cuda_version = ensure_ort_cuda_bundle(app, &client, &temp_root, &model_dir)?;
        let dml_version = ensure_ort_dml_bundle(app, &client, &temp_root, &model_dir)?;

        emit_ort_setup_progress(
            app,
            "done",
            1.0,
            format!(
                "ORT bundles ready: cuda={} dml={}",
                cuda_version, dml_version
            ),
        );
        Ok(())
    })();

    let _ = fs::remove_dir_all(&temp_root);
    result
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            set_log_sink(Some(Arc::new(move |level, message| {
                emit_log(&handle, level, message);
            })));
            let ort_handle = app.handle().clone();
            thread::spawn(move || {
                if let Err(err) = ensure_ort_runtime_bundle(&ort_handle) {
                    let message = format!("ONNX Runtime setup failed: {err}");
                    log_debug(&message);
                    emit_ort_setup_progress(&ort_handle, "error", 1.0, message.clone());
                    emit_log(&ort_handle, "error", &message);
                }
            });
            Ok(())
        })
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
