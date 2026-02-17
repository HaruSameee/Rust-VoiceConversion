const $ = (id) => document.getElementById(id);

const fields = {
  modelPath: $("modelPath"),
  hubertPath: $("hubertPath"),
  rmvpePath: $("rmvpePath"),
  indexPath: $("indexPath"),
  inputGain: $("inputGain"),
  outputGain: $("outputGain"),
  pitchShift: $("pitchShift"),
  indexRate: $("indexRate"),
  speakerId: $("speakerId"),
  sampleRate: $("sampleRate"),
  blockSize: $("blockSize"),
  reloadBtn: $("reloadBtn"),
  saveBtn: $("saveBtn"),
  startBtn: $("startBtn"),
  stopBtn: $("stopBtn"),
  statusLine: $("statusLine"),
  levelLine: $("levelLine"),
  messageLine: $("messageLine"),
};

let statusTimer = null;

function getInvoke() {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) {
    throw new Error("Tauri invoke API not found. Run inside Tauri app.");
  }
  return invoke;
}

function modelFromInputs() {
  const normalizeOptional = (v) => {
    const s = v.trim();
    return s.length > 0 ? s : null;
  };
  return {
    model_path: fields.modelPath.value.trim(),
    hubert_path: normalizeOptional(fields.hubertPath.value),
    pitch_extractor_path: normalizeOptional(fields.rmvpePath.value),
    index_path: normalizeOptional(fields.indexPath.value),
  };
}

function runtimeFromInputs() {
  return {
    input_gain: Number(fields.inputGain.value),
    output_gain: Number(fields.outputGain.value),
    pitch_shift_semitones: Number(fields.pitchShift.value),
    index_rate: Number(fields.indexRate.value),
    speaker_id: Number(fields.speakerId.value),
    sample_rate: Number(fields.sampleRate.value),
    block_size: Number(fields.blockSize.value),
  };
}

function setMessage(msg) {
  fields.messageLine.textContent = msg || "";
}

function applyModel(model) {
  fields.modelPath.value = model.model_path ?? "";
  fields.hubertPath.value = model.hubert_path ?? "";
  fields.rmvpePath.value = model.pitch_extractor_path ?? "";
  fields.indexPath.value = model.index_path ?? "";
}

function applyRuntime(cfg) {
  fields.inputGain.value = String(cfg.input_gain);
  fields.outputGain.value = String(cfg.output_gain);
  fields.pitchShift.value = String(cfg.pitch_shift_semitones);
  fields.indexRate.value = String(cfg.index_rate);
  fields.speakerId.value = String(cfg.speaker_id);
  fields.sampleRate.value = String(cfg.sample_rate);
  fields.blockSize.value = String(cfg.block_size);
}

function applyStatus(status) {
  fields.statusLine.textContent = `status: ${status.running ? "running" : "stopped"}`;
  fields.levelLine.textContent = `input level: rms ${status.input_level_rms.toFixed(4)} / peak ${status.input_level_peak.toFixed(4)}`;
}

async function loadAll() {
  const invoke = getInvoke();
  const [model, runtime, status] = await Promise.all([
    invoke("get_model_config_cmd"),
    invoke("get_runtime_config_cmd"),
    invoke("get_engine_status_cmd"),
  ]);
  applyModel(model);
  applyRuntime(runtime);
  applyStatus(status);
}

async function saveAll() {
  const invoke = getInvoke();
  const model = modelFromInputs();
  if (!model.model_path) {
    throw new Error("model_path is required");
  }
  await invoke("set_model_config_cmd", { model });
  await invoke("set_runtime_config_cmd", { config: runtimeFromInputs() });
}

async function startEngine() {
  const invoke = getInvoke();
  await saveAll();
  const status = await invoke("start_engine_cmd");
  applyStatus(status);
}

async function stopEngine() {
  const invoke = getInvoke();
  const status = await invoke("stop_engine_cmd");
  applyStatus(status);
}

async function pollStatus() {
  try {
    const invoke = getInvoke();
    const status = await invoke("get_engine_status_cmd");
    applyStatus(status);
  } catch {
    clearInterval(statusTimer);
    statusTimer = null;
  }
}

async function runAction(action) {
  try {
    setMessage("");
    await action();
  } catch (err) {
    setMessage(String(err));
  }
}

fields.reloadBtn.addEventListener("click", () => runAction(loadAll));
fields.saveBtn.addEventListener("click", () => runAction(saveAll));
fields.startBtn.addEventListener("click", () => runAction(startEngine));
fields.stopBtn.addEventListener("click", () => runAction(stopEngine));

runAction(loadAll);
statusTimer = setInterval(pollStatus, 250);
