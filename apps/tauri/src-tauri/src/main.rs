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
    list_input_devices, list_output_devices, spawn_voice_changer_stream, RealtimeAudioEngine,
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
                config: RuntimeConfig::default(),
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
    log_debug(&format!(
        "set_runtime_config_cmd sample_rate={} block_size={} index_rate={} top_k={} rows={} protect={:.2} rmvpe_th={:.3}",
        config.sample_rate,
        config.block_size,
        config.index_rate,
        config.index_top_k,
        config.index_search_rows,
        config.protect,
        config.rmvpe_threshold
    ));
    let mut guard = state.lock_runtime();
    guard.config = config;
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
        if runtime_config.block_size < 8_192 {
            log_debug(&format!(
                "block_size={} is too small for current pipeline; auto-raise to 8192",
                runtime_config.block_size
            ));
            runtime_config.block_size = 8_192;
        }

        let model = resolve_model_config_for_start(model_raw)?;
        log_debug(&format!(
            "ORT_DYLIB_PATH={:?}",
            std::env::var("ORT_DYLIB_PATH").ok()
        ));
        log_debug(&format!(
            "start with model={} hubert={:?} rmvpe={:?} index={:?} sr={} block={} index_rate={} top_k={} rows={} protect={:.2} rmvpe_th={:.3}",
            model.model_path,
            model.hubert_path,
            model.pitch_extractor_path,
            model.index_path,
            runtime_config.sample_rate,
            runtime_config.block_size,
            runtime_config.index_rate,
            runtime_config.index_top_k,
            runtime_config.index_search_rows,
            runtime_config.protect,
            runtime_config.rmvpe_threshold
        ));
        let infer_engine = RvcOrtEngine::new(model).map_err(|e| e.to_string())?;
        let voice_changer = VoiceChanger::new(infer_engine, runtime_config.clone());
        let audio_engine = spawn_voice_changer_stream(
            voice_changer,
            runtime_config.sample_rate,
            runtime_config.block_size,
        )
        .map_err(|e| e.to_string())?;

        let mut guard = state.lock_runtime();
        if guard.running {
            log_debug("engine already running");
            sync_levels_from_engine(&mut guard);
            drop(audio_engine);
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
        .or_else(|| resolve_existing_path("model/rmvpe.onnx"));
    let hubert_path = std::env::var("RUST_VC_HUBERT_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
        .or_else(|| resolve_existing_path("model/hubert.onnx"));
    let index_path = std::env::var("RUST_VC_INDEX_PATH")
        .ok()
        .and_then(|p| resolve_existing_path(&p).or(Some(p)))
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

fn ensure_ort_dylib_path() -> Option<String> {
    if let Ok(current) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(&current);
        if p.exists() {
            return Some(current);
        }
        log_debug(&format!(
            "ORT_DYLIB_PATH is set but not found on disk: {}",
            current
        ));
    }

    if let Some(path) = resolve_existing_path("model/onnxruntime.dll") {
        std::env::set_var("ORT_DYLIB_PATH", &path);
        log_debug(&format!(
            "ORT_DYLIB_PATH auto-set from model directory: {}",
            path
        ));
        return Some(path);
    }

    if let Some(path) = find_onnxruntime_dll_via_python() {
        std::env::set_var("ORT_DYLIB_PATH", &path);
        log_debug(&format!(
            "ORT_DYLIB_PATH auto-set from python onnxruntime: {}",
            path
        ));
        return Some(path);
    }

    None
}

fn find_onnxruntime_dll_via_python() -> Option<String> {
    let py = r#"import os
import onnxruntime as ort
print(os.path.join(os.path.dirname(ort.__file__), "capi", "onnxruntime.dll"))"#;

    let output = Command::new("python").args(["-c", py]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let candidate = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if candidate.is_empty() {
        return None;
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
