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
  input_device_name: NullableString;
  output_device_name: NullableString;
  pitch_shift_semitones: number;
  index_rate: number;
  index_smooth_alpha: number;
  index_top_k: number;
  index_search_rows: number;
  protect: number;
  rmvpe_threshold: number;
  pitch_smooth_alpha: number;
  rms_mix_rate: number;
  f0_median_filter_radius: number;
  extra_inference_ms: number;
  response_threshold: number;
  fade_in_ms: number;
  fade_out_ms: number;
  output_tail_offset_ms: number;
  output_slice_offset_samples: number;
  speaker_id: number;
  sample_rate: number;
  block_size: number;
  ort_provider: string;
  ort_device_id: number;
  ort_gpu_mem_limit_mb: number;
  ort_intra_threads: number;
  ort_inter_threads: number;
  ort_parallel_execution: boolean;
  hubert_context_samples_16k: number;
  hubert_output_layer: number;
  hubert_upsample_factor: number;
  cuda_conv_algo: string;
  cuda_conv_max_workspace: boolean;
  cuda_conv1d_pad_to_nc1d: boolean;
  cuda_tf32: boolean;
  index_bin_dim: number;
  index_max_vectors: number;
}

type SetRuntimeConfigArgs = {
  config: RuntimeConfig;
};

interface PresetEntry {
  model: ModelConfig;
  runtime: RuntimeConfig;
  created_at: string;
  updated_at: string;
}

type PresetStore = Record<string, PresetEntry>;

interface AudioDevicesPayload {
  input_devices: string[];
  output_devices: string[];
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
  modelPath: $("modelPath") as HTMLInputElement,
  hubertPath: $("hubertPath") as HTMLInputElement,
  rmvpePath: $("rmvpePath") as HTMLInputElement,
  indexPath: $("indexPath") as HTMLInputElement,
  presetSelect: $("presetSelect") as HTMLSelectElement,
  presetName: $("presetName") as HTMLInputElement,
  presetSaveBtn: $("presetSaveBtn") as HTMLButtonElement,
  presetApplyBtn: $("presetApplyBtn") as HTMLButtonElement,
  presetDeleteBtn: $("presetDeleteBtn") as HTMLButtonElement,
  inputDevice: $("inputDevice") as HTMLSelectElement,
  outputDevice: $("outputDevice") as HTMLSelectElement,
  inputGain: $("inputGain") as HTMLInputElement,
  outputGain: $("outputGain") as HTMLInputElement,
  pitchShift: $("pitchShift") as HTMLInputElement,
  indexRate: $("indexRate") as HTMLInputElement,
  indexSmoothAlpha: $("indexSmoothAlpha") as HTMLInputElement,
  indexTopK: $("indexTopK") as HTMLInputElement,
  indexSearchRows: $("indexSearchRows") as HTMLInputElement,
  protect: $("protect") as HTMLInputElement,
  rmvpeThreshold: $("rmvpeThreshold") as HTMLInputElement,
  pitchSmoothAlpha: $("pitchSmoothAlpha") as HTMLInputElement,
  rmsMixRate: $("rmsMixRate") as HTMLInputElement,
  f0MedianFilterRadius: $("f0MedianFilterRadius") as HTMLInputElement,
  extraInferenceMs: $("extraInferenceMs") as HTMLInputElement,
  responseThreshold: $("responseThreshold") as HTMLInputElement,
  fadeInMs: $("fadeInMs") as HTMLInputElement,
  fadeOutMs: $("fadeOutMs") as HTMLInputElement,
  outputTailOffsetMs: $("outputTailOffsetMs") as HTMLInputElement,
  speakerId: $("speakerId") as HTMLInputElement,
  sampleRate: $("sampleRate") as HTMLInputElement,
  blockSize: $("blockSize") as HTMLInputElement,
  ortProvider: $("ortProvider") as HTMLSelectElement,
  ortDeviceId: $("ortDeviceId") as HTMLInputElement,
  ortGpuMemLimitMb: $("ortGpuMemLimitMb") as HTMLInputElement,
  ortIntraThreads: $("ortIntraThreads") as HTMLInputElement,
  ortInterThreads: $("ortInterThreads") as HTMLInputElement,
  reloadBtn: $("reloadBtn") as HTMLButtonElement,
  saveBtn: $("saveBtn") as HTMLButtonElement,
  startBtn: $("startBtn") as HTMLButtonElement,
  stopBtn: $("stopBtn") as HTMLButtonElement,
  clearLogBtn: $("clearLogBtn") as HTMLButtonElement,
  statusLine: $("statusLine") as HTMLParagraphElement,
  levelLine: $("levelLine") as HTMLParagraphElement,
  messageLine: $("messageLine") as HTMLParagraphElement,
  logBox: $("logBox") as HTMLPreElement,
  tabButtons: Array.from(document.querySelectorAll<HTMLButtonElement>(".tab-btn")),
  tabPanels: Array.from(document.querySelectorAll<HTMLElement>(".tab-panel"))
};

let statusTimer: number | null = null;
let lastStatusLogAt = 0;
let statusPollErrorStreak = 0;
let runtimeCache: RuntimeConfig | null = null;
const PRESET_STORAGE_KEY = "rust_vc_presets_v1";
let presetStore: PresetStore = {};
const MAX_ORT_THREADS = Math.max(1, Math.round(Number(navigator.hardwareConcurrency) || 4));

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

function switchTab(tabName: string): void {
  if (ui.tabButtons.length === 0 || ui.tabPanels.length === 0) {
    return;
  }
  for (const btn of ui.tabButtons) {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const panel of ui.tabPanels) {
    const active = panel.dataset.tabPanel === tabName;
    panel.classList.toggle("active", active);
  }
}

function normalizeOptional(value: string): NullableString {
  const v = value.trim();
  return v.length > 0 ? v : null;
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function atLeast(value: number, min: number, fallback: number): number {
  const v = finiteOr(value, fallback);
  return v < min ? min : v;
}

function intAtLeast(value: number, min: number, fallback: number): number {
  return Math.round(atLeast(value, min, fallback));
}

function clamp(value: number, min: number, max: number, fallback: number): number {
  const v = finiteOr(value, fallback);
  return Math.min(max, Math.max(min, v));
}

function selectedDevice(select: HTMLSelectElement): NullableString {
  const value = select.value.trim();
  return value.length > 0 ? value : null;
}

function renderDeviceSelect(select: HTMLSelectElement, devices: string[], selected: NullableString): void {
  const value = selected ?? "";
  select.innerHTML = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "(既定デバイス)";
  select.appendChild(defaultOption);

  for (const name of devices) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  select.value = value;
}

function normalizePresetName(value: string): string {
  return value.trim();
}

function readPresetStore(): PresetStore {
  try {
    const raw = localStorage.getItem(PRESET_STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    return parsed as PresetStore;
  } catch (error) {
    log("WARN", "failed to read preset store; reset", error);
    return {};
  }
}

function writePresetStore(store: PresetStore): void {
  localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(store));
}

function sortedPresetNames(store: PresetStore): string[] {
  return Object.keys(store).sort((a, b) => a.localeCompare(b, "ja"));
}

function renderPresetSelect(selected: string | null = null): void {
  const names = sortedPresetNames(presetStore);
  ui.presetSelect.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "(preset)";
  ui.presetSelect.appendChild(empty);
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    ui.presetSelect.appendChild(option);
  }
  const target = selected && presetStore[selected] ? selected : "";
  ui.presetSelect.value = target;
}

function savePresetLocal(): void {
  const name = normalizePresetName(ui.presetName.value || ui.presetSelect.value);
  if (!name) {
    throw new Error("preset name is required");
  }
  const timestamp = now();
  const prev = presetStore[name];
  presetStore[name] = {
    model: modelFromInputs(),
    runtime: sanitizeRuntime(runtimeFromInputs()),
    created_at: prev?.created_at ?? timestamp,
    updated_at: timestamp
  };
  writePresetStore(presetStore);
  ui.presetName.value = name;
  renderPresetSelect(name);
  log("INFO", "preset saved", { name, updated_at: timestamp });
}

function applyPresetLocal(): void {
  const name = normalizePresetName(ui.presetSelect.value || ui.presetName.value);
  if (!name) {
    throw new Error("preset selection is required");
  }
  const preset = presetStore[name];
  if (!preset) {
    throw new Error(`preset not found: ${name}`);
  }
  applyModel(preset.model);
  applyRuntime(sanitizeRuntime(preset.runtime));
  ui.presetName.value = name;
  ui.presetSelect.value = name;
  log("INFO", "preset applied", { name, updated_at: preset.updated_at });
}

function deletePresetLocal(): void {
  const name = normalizePresetName(ui.presetSelect.value || ui.presetName.value);
  if (!name) {
    throw new Error("preset selection is required");
  }
  if (!presetStore[name]) {
    throw new Error(`preset not found: ${name}`);
  }
  delete presetStore[name];
  writePresetStore(presetStore);
  ui.presetName.value = "";
  renderPresetSelect();
  log("INFO", "preset deleted", { name });
}

function initPresetUi(): void {
  presetStore = readPresetStore();
  renderPresetSelect();
}

function sanitizeRuntime(raw: RuntimeConfig): RuntimeConfig {
  return {
    input_gain: atLeast(raw.input_gain, 0.0, 1.0),
    output_gain: atLeast(raw.output_gain, 0.0, 1.0),
    input_device_name: raw.input_device_name,
    output_device_name: raw.output_device_name,
    pitch_shift_semitones: clamp(raw.pitch_shift_semitones, -24.0, 24.0, 0.0),
    index_rate: atLeast(raw.index_rate, 0.0, 0.3),
    index_smooth_alpha: atLeast(raw.index_smooth_alpha, 0.0, 0.85),
    index_top_k: intAtLeast(raw.index_top_k, 1, 8),
    index_search_rows: intAtLeast(raw.index_search_rows, 0, 2048),
    protect: atLeast(raw.protect, 0.0, 0.33),
    rmvpe_threshold: atLeast(raw.rmvpe_threshold, 0.0, 0.01),
    pitch_smooth_alpha: atLeast(raw.pitch_smooth_alpha, 0.0, 0.12),
    // 0.0 = strongest input envelope inheritance, 1.0 = almost disabled.
    rms_mix_rate: clamp(raw.rms_mix_rate, 0.0, 1.0, 0.2),
    f0_median_filter_radius: intAtLeast(raw.f0_median_filter_radius, 0, 3),
    extra_inference_ms: intAtLeast(raw.extra_inference_ms, 0, 0),
    response_threshold: finiteOr(raw.response_threshold, -50.0),
    fade_in_ms: intAtLeast(raw.fade_in_ms, 0, 12),
    fade_out_ms: intAtLeast(raw.fade_out_ms, 0, 120),
    output_tail_offset_ms: intAtLeast(raw.output_tail_offset_ms, 0, 0),
    output_slice_offset_samples: intAtLeast(raw.output_slice_offset_samples, 0, 31680),
    speaker_id: Math.round(finiteOr(raw.speaker_id, 0)),
    sample_rate: intAtLeast(raw.sample_rate, 1, 48000),
    block_size: intAtLeast(raw.block_size, 1, 8192),
    ort_provider: ["auto", "cpu", "cuda", "directml"].includes((raw.ort_provider ?? "").toLowerCase())
      ? raw.ort_provider.toLowerCase()
      : "auto",
    ort_device_id: intAtLeast(raw.ort_device_id, 0, 0),
    ort_gpu_mem_limit_mb: intAtLeast(raw.ort_gpu_mem_limit_mb, 0, 0),
    ort_intra_threads: Math.round(clamp(raw.ort_intra_threads, 0, MAX_ORT_THREADS, 0)),
    ort_inter_threads: Math.round(clamp(raw.ort_inter_threads, 1, MAX_ORT_THREADS, 1)),
    ort_parallel_execution: Boolean(raw.ort_parallel_execution),
    hubert_context_samples_16k: intAtLeast(raw.hubert_context_samples_16k, 1600, 16000),
    hubert_output_layer: Math.round(finiteOr(raw.hubert_output_layer, 12)),
    hubert_upsample_factor: intAtLeast(raw.hubert_upsample_factor, 1, 2),
    cuda_conv_algo: ["default", "heuristic", "exhaustive"].includes((raw.cuda_conv_algo ?? "").toLowerCase())
      ? raw.cuda_conv_algo.toLowerCase()
      : "default",
    cuda_conv_max_workspace: Boolean(raw.cuda_conv_max_workspace),
    cuda_conv1d_pad_to_nc1d: Boolean(raw.cuda_conv1d_pad_to_nc1d),
    cuda_tf32: Boolean(raw.cuda_tf32),
    index_bin_dim: intAtLeast(raw.index_bin_dim, 1, 768),
    index_max_vectors: intAtLeast(raw.index_max_vectors, 0, 0)
  };
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
  const base = runtimeCache;
  return {
    input_gain: Number(ui.inputGain.value),
    output_gain: Number(ui.outputGain.value),
    input_device_name: selectedDevice(ui.inputDevice),
    output_device_name: selectedDevice(ui.outputDevice),
    pitch_shift_semitones: Number(ui.pitchShift.value),
    index_rate: Number(ui.indexRate.value),
    index_smooth_alpha: Number(ui.indexSmoothAlpha.value),
    index_top_k: Number(ui.indexTopK.value),
    index_search_rows: Number(ui.indexSearchRows.value),
    protect: Number(ui.protect.value),
    rmvpe_threshold: Number(ui.rmvpeThreshold.value),
    pitch_smooth_alpha: Number(ui.pitchSmoothAlpha.value),
    rms_mix_rate: Number(ui.rmsMixRate.value),
    f0_median_filter_radius: Number(ui.f0MedianFilterRadius.value),
    extra_inference_ms: Number(ui.extraInferenceMs.value),
    response_threshold: Number(ui.responseThreshold.value),
    fade_in_ms: Number(ui.fadeInMs.value),
    fade_out_ms: Number(ui.fadeOutMs.value),
    output_tail_offset_ms: Number(ui.outputTailOffsetMs.value),
    output_slice_offset_samples: base?.output_slice_offset_samples ?? 31680,
    speaker_id: Number(ui.speakerId.value),
    sample_rate: Number(ui.sampleRate.value),
    block_size: Number(ui.blockSize.value),
    ort_provider: ui.ortProvider.value,
    ort_device_id: Number(ui.ortDeviceId.value),
    ort_gpu_mem_limit_mb: Number(ui.ortGpuMemLimitMb.value),
    ort_intra_threads: Number(ui.ortIntraThreads.value),
    ort_inter_threads: Number(ui.ortInterThreads.value),
    ort_parallel_execution: base?.ort_parallel_execution ?? false,
    hubert_context_samples_16k: base?.hubert_context_samples_16k ?? 16000,
    hubert_output_layer: base?.hubert_output_layer ?? 12,
    hubert_upsample_factor: base?.hubert_upsample_factor ?? 2,
    cuda_conv_algo: base?.cuda_conv_algo ?? "default",
    cuda_conv_max_workspace: base?.cuda_conv_max_workspace ?? false,
    cuda_conv1d_pad_to_nc1d: base?.cuda_conv1d_pad_to_nc1d ?? false,
    cuda_tf32: base?.cuda_tf32 ?? false,
    index_bin_dim: base?.index_bin_dim ?? 768,
    index_max_vectors: base?.index_max_vectors ?? 0
  };
}

function applyModel(model: ModelConfig): void {
  ui.modelPath.value = model.model_path ?? "";
  ui.hubertPath.value = model.hubert_path ?? "";
  ui.rmvpePath.value = model.pitch_extractor_path ?? "";
  ui.indexPath.value = model.index_path ?? "";
}

function applyRuntime(config: RuntimeConfig): void {
  runtimeCache = config;
  ui.inputGain.value = String(config.input_gain);
  ui.outputGain.value = String(config.output_gain);
  ui.inputDevice.value = config.input_device_name ?? "";
  ui.outputDevice.value = config.output_device_name ?? "";
  ui.pitchShift.value = String(config.pitch_shift_semitones);
  ui.indexRate.value = String(config.index_rate);
  ui.indexSmoothAlpha.value = String(config.index_smooth_alpha);
  ui.indexTopK.value = String(config.index_top_k);
  ui.indexSearchRows.value = String(config.index_search_rows);
  ui.protect.value = String(config.protect);
  ui.rmvpeThreshold.value = String(config.rmvpe_threshold);
  ui.pitchSmoothAlpha.value = String(config.pitch_smooth_alpha);
  ui.rmsMixRate.value = String(config.rms_mix_rate);
  ui.f0MedianFilterRadius.value = String(config.f0_median_filter_radius);
  ui.extraInferenceMs.value = String(config.extra_inference_ms);
  ui.responseThreshold.value = String(config.response_threshold);
  ui.fadeInMs.value = String(config.fade_in_ms);
  ui.fadeOutMs.value = String(config.fade_out_ms);
  ui.outputTailOffsetMs.value = String(config.output_tail_offset_ms);
  ui.speakerId.value = String(config.speaker_id);
  ui.sampleRate.value = String(config.sample_rate);
  ui.blockSize.value = String(config.block_size);
  ui.ortProvider.value = config.ort_provider;
  ui.ortDeviceId.value = String(config.ort_device_id);
  ui.ortGpuMemLimitMb.value = String(config.ort_gpu_mem_limit_mb);
  ui.ortIntraThreads.value = String(config.ort_intra_threads);
  ui.ortInterThreads.value = String(config.ort_inter_threads);
}

function applyStatus(status: EngineStatus): void {
  ui.statusLine.textContent = `status: ${status.running ? "running" : "stopped"}`;
  ui.levelLine.textContent = `input level: rms ${status.input_level_rms.toFixed(4)} / peak ${status.input_level_peak.toFixed(4)}`;
}

async function loadAll(): Promise<void> {
  log("INFO", "loadAll begin");
  const [devices, model, runtime, status] = await Promise.all([
    invoke<AudioDevicesPayload>("list_audio_devices_cmd"),
    invoke<ModelConfig>("get_model_config_cmd"),
    invoke<RuntimeConfig>("get_runtime_config_cmd"),
    invoke<EngineStatus>("get_engine_status_cmd")
  ]);
  renderDeviceSelect(ui.inputDevice, devices.input_devices, runtime.input_device_name);
  renderDeviceSelect(ui.outputDevice, devices.output_devices, runtime.output_device_name);
  applyModel(model);
  applyRuntime(sanitizeRuntime(runtime));
  applyStatus(status);
  log("INFO", "loadAll done", {
    running: status.running,
    input_devices: devices.input_devices.length,
    output_devices: devices.output_devices.length
  });
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
  const runtimeRaw = runtimeFromInputs();
  const runtime = sanitizeRuntime(runtimeRaw);
  if (JSON.stringify(runtimeRaw) !== JSON.stringify(runtime)) {
    log("WARN", "runtime values adjusted", { before: runtimeRaw, after: runtime });
    applyRuntime(runtime);
  }
  if (runtime.ort_provider === "cpu") {
    log("WARN", "CPU provider is not recommended for realtime RVC. Prefer CUDA or DirectML when available.");
  }
  log("INFO", "saveAll begin", { model, runtime });
  await invoke("set_model_config_cmd", { model });
  const runtimeArgs: SetRuntimeConfigArgs = { config: runtime };
  await invoke("set_runtime_config_cmd", runtimeArgs);
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
ui.presetSaveBtn.addEventListener("click", () => {
  void runAction("preset_save", async () => {
    savePresetLocal();
  });
});
ui.presetApplyBtn.addEventListener("click", () => {
  void runAction("preset_apply", async () => {
    applyPresetLocal();
  });
});
ui.presetDeleteBtn.addEventListener("click", () => {
  void runAction("preset_delete", async () => {
    deletePresetLocal();
  });
});
ui.presetSelect.addEventListener("change", () => {
  const selected = normalizePresetName(ui.presetSelect.value);
  if (selected) {
    ui.presetName.value = selected;
  }
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
for (const btn of ui.tabButtons) {
  btn.addEventListener("click", () => {
    switchTab(btn.dataset.tab ?? "model");
  });
}
switchTab(ui.tabButtons[0]?.dataset.tab ?? "model");
initPresetUi();

void runAction("init", loadAll);
statusTimer = window.setInterval(() => {
  void runAction("status_poll", pollStatus);
}, 250);

if (statusTimer === null) {
  log("WARN", "status timer not started");
}
