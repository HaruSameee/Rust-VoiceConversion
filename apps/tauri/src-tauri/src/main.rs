#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use serde::Serialize;
use tauri::State;
use vc_audio::{
    list_input_devices, list_output_devices, spawn_voice_changer_stream, RealtimeAudioEngine,
};
use vc_core::{ModelConfig, RuntimeConfig, VoiceChanger};
use vc_inference::RvcOrtEngine;

struct AppState {
    inner: Mutex<RuntimeState>,
}

struct RuntimeState {
    config: RuntimeConfig,
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
    let input = list_input_devices().map_err(|e| e.to_string())?;
    let output = list_output_devices().map_err(|e| e.to_string())?;
    Ok(AudioDevicesPayload {
        input_devices: input,
        output_devices: output,
    })
}

#[tauri::command]
fn get_runtime_config_cmd(state: State<'_, AppState>) -> RuntimeConfig {
    let guard = state.inner.lock().expect("state lock poisoned");
    guard.config.clone()
}

#[tauri::command]
fn set_runtime_config_cmd(config: RuntimeConfig, state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state
        .inner
        .lock()
        .map_err(|_| "state lock poisoned".to_string())?;
    guard.config = config;
    Ok(())
}

#[tauri::command]
fn start_engine_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    let mut guard = state
        .inner
        .lock()
        .map_err(|_| "state lock poisoned".to_string())?;
    if guard.running {
        sync_levels_from_engine(&mut guard);
        return Ok(EngineStatus {
            running: guard.running,
            input_level_rms: guard.input_level_rms,
            input_level_peak: guard.input_level_peak,
        });
    }

    let runtime_config = guard.config.clone();
    let model = ModelConfig {
        model_path: std::env::var("RUST_VC_MODEL_PATH")
            .unwrap_or_else(|_| "model/model.onnx".to_string()),
        index_path: None,
        pitch_extractor_path: None,
    };
    let infer_engine = RvcOrtEngine::new(model).map_err(|e| e.to_string())?;
    let voice_changer = VoiceChanger::new(infer_engine, runtime_config.clone());
    let audio_engine = spawn_voice_changer_stream(
        voice_changer,
        runtime_config.sample_rate,
        runtime_config.block_size,
    )
    .map_err(|e| e.to_string())?;

    guard.running = true;
    guard.engine_task = Some(audio_engine);
    sync_levels_from_engine(&mut guard);
    Ok(EngineStatus {
        running: guard.running,
        input_level_rms: guard.input_level_rms,
        input_level_peak: guard.input_level_peak,
    })
}

#[tauri::command]
fn stop_engine_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    let mut guard = state
        .inner
        .lock()
        .map_err(|_| "state lock poisoned".to_string())?;
    if let Some(task) = guard.engine_task.take() {
        task.stop_and_abort();
    }
    guard.running = false;
    Ok(EngineStatus {
        running: guard.running,
        input_level_rms: guard.input_level_rms,
        input_level_peak: guard.input_level_peak,
    })
}

#[tauri::command]
fn get_engine_status_cmd(state: State<'_, AppState>) -> Result<EngineStatus, String> {
    let mut guard = state
        .inner
        .lock()
        .map_err(|_| "state lock poisoned".to_string())?;
    sync_levels_from_engine(&mut guard);
    Ok(EngineStatus {
        running: guard.running,
        input_level_rms: guard.input_level_rms,
        input_level_peak: guard.input_level_peak,
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

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_devices_cmd,
            get_runtime_config_cmd,
            set_runtime_config_cmd,
            start_engine_cmd,
            stop_engine_cmd,
            get_engine_status_cmd
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
