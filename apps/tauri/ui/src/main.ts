import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type NullableString = string | null;

interface ModelConfig {
  model_path: string;
  index_path: NullableString;
  pitch_extractor_path: NullableString;
  hubert_path: NullableString;
}

interface RuntimeConfig {
  input_gain: number;
  output_gain: number;
  pitch_shift_semitones: number;
  index_rate: number;
  speaker_id: number;
  sample_rate: number;
  block_size: number;
}

interface EngineStatus {
  running: boolean;
  input_level_rms: number;
  input_level_peak: number;
}

type LogLevel = "INFO" | "WARN" | "ERROR";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`missing element: ${id}`);
  }
  return el as T;
};

const ui = {
  modelPath: $<HTMLInputElement>("modelPath"),
  hubertPath: $<HTMLInputElement>("hubertPath"),
  rmvpePath: $<HTMLInputElement>("rmvpePath"),
  indexPath: $<HTMLInputElement>("indexPath"),
  inputGain: $<HTMLInputElement>("inputGain"),
  outputGain: $<HTMLInputElement>("outputGain"),
  pitchShift: $<HTMLInputElement>("pitchShift"),
  indexRate: $<HTMLInputElement>("indexRate"),
  speakerId: $<HTMLInputElement>("speakerId"),
  sampleRate: $<HTMLInputElement>("sampleRate"),
  blockSize: $<HTMLInputElement>("blockSize"),
  reloadBtn: $<HTMLButtonElement>("reloadBtn"),
  saveBtn: $<HTMLButtonElement>("saveBtn"),
  startBtn: $<HTMLButtonElement>("startBtn"),
  stopBtn: $<HTMLButtonElement>("stopBtn"),
  clearLogBtn: $<HTMLButtonElement>("clearLogBtn"),
  statusLine: $<HTMLParagraphElement>("statusLine"),
  levelLine: $<HTMLParagraphElement>("levelLine"),
  messageLine: $<HTMLParagraphElement>("messageLine"),
  logBox: $<HTMLPreElement>("logBox")
};

let statusTimer: number | null = null;
let lastStatusLogAt = 0;
let statusPollErrorStreak = 0;

function now(): string {
  return new Date().toISOString();
}

function log(level: LogLevel, message: string, detail?: unknown): void {
  const line = `[${now()}] [${level}] ${message}${detail !== undefined ? ` ${JSON.stringify(detail)}` : ""}`;
  if (level === "ERROR") {
    console.error(line);
  } else if (level === "WARN") {
    console.warn(line);
  } else {
    console.log(line);
  }
  ui.logBox.textContent = `${line}\n${ui.logBox.textContent}`.slice(0, 12000);
}

function setMessage(msg: string): void {
  ui.messageLine.textContent = msg;
}

function normalizeOptional(value: string): NullableString {
  const v = value.trim();
  return v.length > 0 ? v : null;
}

function modelFromInputs(): ModelConfig {
  return {
    model_path: ui.modelPath.value.trim(),
    hubert_path: normalizeOptional(ui.hubertPath.value),
    pitch_extractor_path: normalizeOptional(ui.rmvpePath.value),
    index_path: normalizeOptional(ui.indexPath.value)
  };
}

function runtimeFromInputs(): RuntimeConfig {
  return {
    input_gain: Number(ui.inputGain.value),
    output_gain: Number(ui.outputGain.value),
    pitch_shift_semitones: Number(ui.pitchShift.value),
    index_rate: Number(ui.indexRate.value),
    speaker_id: Number(ui.speakerId.value),
    sample_rate: Number(ui.sampleRate.value),
    block_size: Number(ui.blockSize.value)
  };
}

function applyModel(model: ModelConfig): void {
  ui.modelPath.value = model.model_path ?? "";
  ui.hubertPath.value = model.hubert_path ?? "";
  ui.rmvpePath.value = model.pitch_extractor_path ?? "";
  ui.indexPath.value = model.index_path ?? "";
}

function applyRuntime(config: RuntimeConfig): void {
  ui.inputGain.value = String(config.input_gain);
  ui.outputGain.value = String(config.output_gain);
  ui.pitchShift.value = String(config.pitch_shift_semitones);
  ui.indexRate.value = String(config.index_rate);
  ui.speakerId.value = String(config.speaker_id);
  ui.sampleRate.value = String(config.sample_rate);
  ui.blockSize.value = String(config.block_size);
}

function applyStatus(status: EngineStatus): void {
  ui.statusLine.textContent = `status: ${status.running ? "running" : "stopped"}`;
  ui.levelLine.textContent = `input level: rms ${status.input_level_rms.toFixed(4)} / peak ${status.input_level_peak.toFixed(4)}`;
}

async function loadAll(): Promise<void> {
  log("INFO", "loadAll begin");
  const [model, runtime, status] = await Promise.all([
    invoke<ModelConfig>("get_model_config_cmd"),
    invoke<RuntimeConfig>("get_runtime_config_cmd"),
    invoke<EngineStatus>("get_engine_status_cmd")
  ]);
  applyModel(model);
  applyRuntime(runtime);
  applyStatus(status);
  log("INFO", "loadAll done", { running: status.running });
}

async function saveAll(): Promise<void> {
  const model = modelFromInputs();
  if (!model.model_path) {
    throw new Error("model_path is required");
  }
  if (!model.hubert_path) {
    throw new Error("hubert_path is required for RVC models");
  }
  if (!model.pitch_extractor_path) {
    throw new Error("pitch_extractor_path (RMVPE) is required for RVC models");
  }
  const runtime = runtimeFromInputs();
  log("INFO", "saveAll begin", { model, runtime });
  await invoke("set_model_config_cmd", { model });
  await invoke("set_runtime_config_cmd", { config: runtime });
  log("INFO", "saveAll done");
}

async function startEngine(): Promise<void> {
  await saveAll();
  log("INFO", "start_engine_cmd");
  const status = await invoke<EngineStatus>("start_engine_cmd");
  applyStatus(status);
  log("INFO", "start_engine_cmd done", status);
}

async function stopEngine(): Promise<void> {
  log("INFO", "stop_engine_cmd");
  const status = await invoke<EngineStatus>("stop_engine_cmd");
  applyStatus(status);
  log("INFO", "stop_engine_cmd done", status);
}

async function pollStatus(): Promise<void> {
  const status = await invoke<EngineStatus>("get_engine_status_cmd");
  applyStatus(status);
  const t = Date.now();
  if (t - lastStatusLogAt > 2000) {
    log("INFO", "status", status);
    lastStatusLogAt = t;
  }
}

async function runAction(name: string, action: () => Promise<void>): Promise<void> {
  try {
    setMessage("");
    await action();
    if (name === "status_poll") {
      statusPollErrorStreak = 0;
    }
  } catch (error) {
    const msg = error instanceof Error ? `${name}: ${error.message}` : `${name}: ${String(error)}`;
    setMessage(msg);
    if (name === "status_poll") {
      statusPollErrorStreak += 1;
      if (statusPollErrorStreak === 1 || statusPollErrorStreak % 10 === 0) {
        log("ERROR", `${msg} (x${statusPollErrorStreak})`);
      }
      if (statusPollErrorStreak >= 40 && statusTimer !== null) {
        clearInterval(statusTimer);
        statusTimer = null;
        log("WARN", "status polling stopped after repeated errors");
      }
    } else {
      log("ERROR", msg);
    }
  }
}

ui.reloadBtn.addEventListener("click", () => {
  void runAction("reload", loadAll);
});
ui.saveBtn.addEventListener("click", () => {
  void runAction("save", saveAll);
});
ui.startBtn.addEventListener("click", () => {
  void runAction("start", startEngine);
});
ui.stopBtn.addEventListener("click", () => {
  void runAction("stop", stopEngine);
});
ui.clearLogBtn.addEventListener("click", () => {
  ui.logBox.textContent = "";
});

void runAction("init", loadAll);
statusTimer = window.setInterval(() => {
  void runAction("status_poll", pollStatus);
}, 250);

if (statusTimer === null) {
  log("WARN", "status timer not started");
}
