use std::{
    any::Any,
    collections::{HashMap, HashSet},
    fs, io,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use ndarray::{Array1, Array2, Array3};
use ort::{
    ep::{self, ExecutionProvider, ExecutionProviderDispatch},
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    tensor::TensorElementType,
    value::Tensor,
};
use vc_core::{InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};
use vc_signal::{
    apply_rms_mix, coarse_pitch_from_f0, median_filter_pitch_track_inplace, normalize_for_onnx_input,
    pad_for_rmvpe, postprocess_generated_audio, resample_hq_into,
    resize_pitch_to_frames, rmvpe_mel_from_audio, RMVPE_SAMPLE_RATE, RVC_HOP_LENGTH,
};

const DEFAULT_HUBERT_CONTEXT_SAMPLES_16K: usize = 4_000;
const MAX_HUBERT_CONTEXT_SAMPLES_16K: usize = 64_000;
const DEFAULT_HUBERT_OUTPUT_LAYER: i64 = 12;
const DEFAULT_RMVPE_THRESHOLD: f32 = 0.03;
const DEFAULT_BIN_INDEX_DIM: usize = 768;
static ORT_INIT_FAILED: OnceLock<String> = OnceLock::new();
static HUBERT_LEN_FIX_CACHE: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
static ORT_PROVIDER_AVAIL_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_CUDA_MISSING_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_DML_MISSING_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_AUTO_CPU_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_CUDA_TUNE_LOGGED: OnceLock<()> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
enum OrtProvider {
    Auto,
    Cpu,
    Cuda,
    DirectMl,
}

#[derive(Debug, Clone, Copy)]
struct OrtExecutionConfig {
    provider: OrtProvider,
    device_id: i32,
    gpu_mem_limit_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct OrtProviderAvailability {
    cuda: bool,
    dml: bool,
    cpu: bool,
}

#[derive(Debug, Clone)]
struct FeatureIndex {
    vectors: Array2<f32>,
}

pub struct RvcOrtEngine {
    model: ModelConfig,
    rvc_session: Session,
    rmvpe_session: Option<Session>,
    hubert_session: Option<Session>,
    feature_index: Option<FeatureIndex>,
    phone_feature_dim: usize,
    hubert_source_history_16k: Vec<f32>,
    hubert_source_len_fixes: HashMap<usize, usize>,
    hubert_context_samples_16k: usize,
    hubert_output_layer: i64,
    hubert_upsample_factor: usize,
    hubert_runtime_cpu_fallback_tried: bool,
    rvc_runtime_cpu_fallback_tried: bool,
    default_pitch_smooth_alpha: f32,
    index_prev_vector: Vec<f32>,
    last_pitch_hz: f32,
    rnd_state: u64,
}

impl RvcOrtEngine {
    pub fn new(model: ModelConfig, runtime_config: &RuntimeConfig) -> Result<Self> {
        eprintln!(
            "[vc-inference] init model={} hubert={:?} rmvpe={:?} index={:?}",
            model.model_path, model.hubert_path, model.pitch_extractor_path, model.index_path
        );
        if let Some(msg) = ORT_INIT_FAILED.get() {
            return Err(VcError::Config(msg.clone()));
        }
        if !Path::new(&model.model_path).exists() {
            return Err(VcError::Config(format!(
                "model file not found: {}",
                model.model_path
            )));
        }

        let build_res = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Self> {
            ort::init().commit();
            let ort_intra_threads = read_env_usize("RUST_VC_ORT_INTRA_THREADS")
                .unwrap_or_else(default_ort_intra_threads)
                .max(1);
            let ort_inter_threads = read_env_usize("RUST_VC_ORT_INTER_THREADS")
                .unwrap_or(1)
                .max(1);
            let ort_parallel_execution = read_env_bool("RUST_VC_ORT_PARALLEL").unwrap_or(false);
            let ort_ep = resolve_ort_execution_config(runtime_config);
            let hubert_context_samples_16k = read_env_usize("RUST_VC_HUBERT_CONTEXT_16K")
                .unwrap_or(DEFAULT_HUBERT_CONTEXT_SAMPLES_16K)
                .max(1_600);
            let hubert_output_layer =
                read_env_i64("RUST_VC_HUBERT_OUTPUT_LAYER").unwrap_or(DEFAULT_HUBERT_OUTPUT_LAYER);
            let hubert_upsample_factor = read_env_usize("RUST_VC_HUBERT_UPSAMPLE_FACTOR")
                .unwrap_or(2)
                .clamp(1, 4);
            let pitch_smooth_alpha = if runtime_config.pitch_smooth_alpha.is_finite() {
                runtime_config.pitch_smooth_alpha
            } else {
                read_env_f32("RUST_VC_PITCH_SMOOTH_ALPHA").unwrap_or(0.12)
            }
            .max(0.0);
            eprintln!(
                "[vc-inference] ort intra_threads={} inter_threads={} parallel={} provider={:?} dev={} vram_limit_mb={} hubert_ctx_16k={} hubert_layer={} hubert_up={} pitch_smooth_alpha={:.2}",
                ort_intra_threads,
                ort_inter_threads,
                ort_parallel_execution,
                ort_ep.provider,
                ort_ep.device_id,
                ort_ep
                    .gpu_mem_limit_bytes
                    .map(|v| v / (1024 * 1024))
                    .unwrap_or(0),
                hubert_context_samples_16k,
                hubert_output_layer,
                hubert_upsample_factor,
                pitch_smooth_alpha
            );

            let rvc_session = build_ort_session(
                &model.model_path,
                "rvc",
                ort_intra_threads,
                ort_inter_threads,
                ort_parallel_execution,
                &ort_ep,
            )?;

            let rmvpe_session = if let Some(path) = model.pitch_extractor_path.as_deref() {
                if !Path::new(path).exists() {
                    return Err(VcError::Config(format!(
                        "pitch extractor model file not found: {}",
                        path
                    )));
                }
                Some(build_aux_session_with_cpu_fallback(
                    path,
                    "rmvpe",
                    ort_intra_threads,
                    ort_inter_threads,
                    ort_parallel_execution,
                    &ort_ep,
                )?)
            } else {
                None
            };

            let hubert_session = if let Some(path) = model.hubert_path.as_deref() {
                if !Path::new(path).exists() {
                    return Err(VcError::Config(format!(
                        "hubert model file not found: {}",
                        path
                    )));
                }
                Some(build_aux_session_with_cpu_fallback(
                    path,
                    "hubert",
                    ort_intra_threads,
                    ort_inter_threads,
                    ort_parallel_execution,
                    &ort_ep,
                )?)
            } else {
                None
            };

            let phone_feature_dim = infer_phone_feature_dim_from_session(&rvc_session);
            eprintln!("[vc-inference] rvc phone_feature_dim={phone_feature_dim}");
            let requirements = infer_rvc_requirements(&rvc_session);
            if hubert_session.is_none() {
                return Err(VcError::Config(
                    "RVC requires HuBERT/ContentVec features. set `hubert_path` (e.g. model/hubert.onnx)"
                        .to_string(),
                ));
            }
            if requirements.needs_pitch && rmvpe_session.is_none() {
                return Err(VcError::Config(
                    "this RVC model requires pitch inputs. set `pitch_extractor_path` (e.g. model/rmvpe.onnx)"
                        .to_string(),
                ));
            }
            if let Some(session) = hubert_session.as_ref() {
                if let Some(hubert_dim) = infer_hubert_output_dim_from_session(session) {
                    eprintln!("[vc-inference] hubert output_dim={hubert_dim}");
                    if hubert_dim != phone_feature_dim {
                        return Err(VcError::Config(format!(
                            "hubert output dim mismatch: hubert={hubert_dim}, rvc_phone={phone_feature_dim}"
                        )));
                    }
                }
            }
            let feature_index = match model.index_path.as_deref() {
                Some(path) => {
                    if !Path::new(path).exists() {
                        eprintln!("[vc-inference] index file not found, disable index: {path}");
                        None
                    } else {
                        match load_feature_index(path, phone_feature_dim) {
                            Ok(index) => Some(index),
                            Err(e) => {
                                eprintln!("index disabled: {e}");
                                None
                            }
                        }
                    }
                }
                None => None,
            };

            Ok(Self {
                model,
                rvc_session,
                rmvpe_session,
                hubert_session,
                feature_index,
                phone_feature_dim,
                hubert_source_history_16k: Vec::with_capacity(hubert_context_samples_16k),
                hubert_source_len_fixes: HashMap::new(),
                hubert_context_samples_16k,
                hubert_output_layer,
                hubert_upsample_factor,
                hubert_runtime_cpu_fallback_tried: false,
                rvc_runtime_cpu_fallback_tried: false,
                default_pitch_smooth_alpha: pitch_smooth_alpha,
                index_prev_vector: Vec::new(),
                last_pitch_hz: 0.0,
                rnd_state: 0x9E37_79B9_7F4A_7C15,
            })
        }));

        match build_res {
            Ok(result) => result,
            Err(payload) => {
                let details = panic_payload_to_string(payload);
                let msg = format!(
                    "failed to initialize ONNX Runtime (likely DLL/version mismatch). \
Set ORT_DYLIB_PATH to a compatible onnxruntime.dll (>= 1.23.x). details: {details}"
                );
                let _ = ORT_INIT_FAILED.set(msg.clone());
                Err(VcError::Config(msg))
            }
        }
    }

    pub fn model_path(&self) -> &str {
        &self.model.model_path
    }

    fn estimate_pitch(
        &mut self,
        padded_input: &[f32],
        input_sample_rate: u32,
        fallback_frames: usize,
        threshold: f32,
    ) -> Result<Vec<f32>> {
        let Some(session) = self.rmvpe_session.as_mut() else {
            return Ok(vec![0.0; fallback_frames]);
        };

        let inlet = session
            .inputs()
            .first()
            .ok_or_else(|| VcError::Inference("rmvpe model has no inputs".to_string()))?;
        let input_name = inlet.name().to_string();
        let input_shape = inlet
            .dtype()
            .tensor_shape()
            .map_or_else(Vec::new, |s| s.to_vec());
        let rank = input_shape.len();
        let expects_mel = rank == 3 && input_shape.get(1).map(|v| *v == 128).unwrap_or(false);
        let (input_tensor, valid_frames) = if expects_mel {
            let mel_input = rmvpe_mel_from_audio(padded_input, input_sample_rate);
            let tensor = Tensor::from_array(mel_input.mel.clone()).map_err(|e| {
                VcError::Inference(format!("failed to create rmvpe mel tensor [1,128,T]: {e}"))
            })?;
            (tensor, mel_input.valid_frames)
        } else {
            (
                tensor_from_audio_rank_owned(rank.max(1), padded_input.to_vec())?,
                fallback_frames,
            )
        };

        let outputs = session
            .run(vec![(input_name, SessionInputValue::from(input_tensor))])
            .map_err(|e| VcError::Inference(format!("rmvpe inference failed: {e}")))?;
        if outputs.len() == 0 {
            return Err(VcError::Inference(
                "rmvpe model returned no outputs".to_string(),
            ));
        }

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            VcError::Inference(format!("failed to extract rmvpe output tensor<f32>: {e}"))
        })?;
        let mut f0 = decode_rmvpe_output(shape, data, threshold)?;
        if valid_frames > 0 && f0.len() > valid_frames {
            f0.truncate(valid_frames);
        }
        Ok(f0)
    }

    fn extract_phone_features(
        &mut self,
        normalized: &[f32],
        input_sample_rate: u32,
        target_phone_frames: usize,
    ) -> Result<Array3<f32>> {
        if self.hubert_session.is_none() {
            return Err(VcError::Config(
                "hubert session is unavailable. RVC requires HuBERT/ContentVec phone features; set `hubert_path` to a valid ONNX model."
                    .to_string(),
            ));
        }

        let mut chunk_16k = Vec::<f32>::new();
        if input_sample_rate == RMVPE_SAMPLE_RATE {
            chunk_16k.extend_from_slice(normalized);
        } else {
            resample_hq_into(
                normalized,
                input_sample_rate,
                RMVPE_SAMPLE_RATE,
                &mut chunk_16k,
            );
        }
        if chunk_16k.is_empty() {
            chunk_16k.push(0.0);
        }

        let chunk_len_16k = chunk_16k.len();
        self.hubert_source_history_16k.extend(chunk_16k);
        let dynamic_ctx = self
            .hubert_context_samples_16k
            .max(chunk_len_16k)
            .min(MAX_HUBERT_CONTEXT_SAMPLES_16K);
        if self.hubert_source_history_16k.len() > dynamic_ctx {
            let drop_n = self.hubert_source_history_16k.len() - dynamic_ctx;
            self.hubert_source_history_16k.drain(0..drop_n);
        }

        let mut source = vec![0.0_f32; dynamic_ctx];
        let hist_len = self.hubert_source_history_16k.len().min(dynamic_ctx);
        let pad_left = dynamic_ctx - hist_len;
        source[pad_left..].copy_from_slice(
            &self.hubert_source_history_16k[self.hubert_source_history_16k.len() - hist_len..],
        );

        let phone = match run_hubert_session_with_len_fallback(
            self.hubert_session
                .as_mut()
                .expect("hubert_session checked above"),
            &source,
            pad_left,
            self.phone_feature_dim,
            self.hubert_output_layer,
            &mut self.hubert_source_len_fixes,
        ) {
            Ok(phone) => phone,
            Err(err) => {
                if !self.try_hubert_runtime_cpu_fallback(&err)? {
                    return Err(err);
                }
                run_hubert_session_with_len_fallback(
                    self.hubert_session
                        .as_mut()
                        .expect("hubert_session fallback should exist"),
                    &source,
                    pad_left,
                    self.phone_feature_dim,
                    self.hubert_output_layer,
                    &mut self.hubert_source_len_fixes,
                )?
            }
        };
        let tail = tail_phone_frames(&phone, target_phone_frames.max(1));
        Ok(tail)
    }

    fn try_hubert_runtime_cpu_fallback(&mut self, cause: &VcError) -> Result<bool> {
        if self.hubert_runtime_cpu_fallback_tried {
            return Ok(false);
        }
        if !is_hubert_cuda_runtime_failure(cause) {
            return Ok(false);
        }
        let Some(path) = self.model.hubert_path.as_deref() else {
            return Ok(false);
        };
        self.hubert_runtime_cpu_fallback_tried = true;

        eprintln!(
            "[vc-inference] warning: hubert runtime failed on GPU EP; switching hubert session to CPU EP. cause={}",
            cause
        );

        let ort_intra_threads = read_env_usize("RUST_VC_ORT_INTRA_THREADS")
            .unwrap_or_else(default_ort_intra_threads)
            .max(1);
        let ort_inter_threads = read_env_usize("RUST_VC_ORT_INTER_THREADS")
            .unwrap_or(1)
            .max(1);
        let ort_parallel_execution = read_env_bool("RUST_VC_ORT_PARALLEL").unwrap_or(false);
        let cpu_execution = OrtExecutionConfig {
            provider: OrtProvider::Cpu,
            device_id: 0,
            gpu_mem_limit_bytes: None,
        };

        let cpu_session = build_ort_session(
            path,
            "hubert(cpu-fallback)",
            ort_intra_threads,
            ort_inter_threads,
            ort_parallel_execution,
            &cpu_execution,
        )
        .map_err(|e| {
            VcError::Inference(format!(
                "hubert runtime GPU failure and CPU fallback session init failed: {e}"
            ))
        })?;
        self.hubert_session = Some(cpu_session);
        self.hubert_source_len_fixes.clear();
        eprintln!("[vc-inference] hubert session fallback: using CPU EP");
        Ok(true)
    }

    fn blend_phone_with_index(&mut self, phone: &mut Array3<f32>, config: &RuntimeConfig) {
        let Some(index) = &self.feature_index else {
            self.index_prev_vector.clear();
            return;
        };
        if index.vectors.nrows() == 0 || index.vectors.ncols() == 0 {
            self.index_prev_vector.clear();
            return;
        }
        let rate = if config.index_rate.is_finite() {
            config.index_rate.max(0.0)
        } else {
            0.0
        };
        if rate <= f32::EPSILON {
            self.index_prev_vector.clear();
            return;
        }
        let index_smooth = if config.index_smooth_alpha.is_finite() {
            config.index_smooth_alpha.max(0.0)
        } else {
            0.0
        };
        let top_k = config.index_top_k.max(1);

        let frames = phone.shape()[1];
        let dims = phone.shape()[2];
        let rows = index.vectors.nrows();
        let mut search_rows = if config.index_search_rows == 0 {
            rows
        } else {
            config.index_search_rows.min(rows)
        }
        .max(top_k);
        // Lower index influence should also lower search cost.
        if rate < 1.0 {
            let scaled = ((search_rows as f32) * rate.max(0.15)).round() as usize;
            search_rows = scaled.max(top_k).min(rows);
        }
        let frame_search_stride = if search_rows >= 4_096 {
            3
        } else if search_rows >= 2_048 {
            2
        } else {
            1
        };
        let dist_dim_step = if search_rows >= 4_096 {
            8
        } else if search_rows >= 2_048 {
            4
        } else if search_rows >= 1_024 {
            2
        } else {
            1
        };
        let stride = (rows / search_rows.max(1)).max(1);
        let mut retrieved = vec![0.0_f32; dims];
        let mut prev_vec = if self.index_prev_vector.len() == dims {
            self.index_prev_vector.clone()
        } else {
            vec![0.0_f32; dims]
        };
        let mut has_prev = self.index_prev_vector.len() == dims;

        for t in 0..frames {
            if frame_search_stride > 1 && has_prev && t % frame_search_stride != 0 {
                for c in 0..dims {
                    let from_index = prev_vec[c];
                    let base = phone[(0, t, c)];
                    phone[(0, t, c)] = base * (1.0 - rate) + from_index * rate;
                }
                continue;
            }
            let mut best = Vec::<(f32, usize)>::with_capacity(top_k);
            let mut scanned = 0usize;
            let mut row = 0usize;
            while row < rows && scanned < search_rows {
                let mut l2 = 0.0_f32;
                for c in (0..dims).step_by(dist_dim_step) {
                    let diff = phone[(0, t, c)] - index.vectors[(row, c)];
                    l2 += diff * diff;
                }
                push_top_k(&mut best, (l2, row), top_k);
                row += stride;
                scanned += 1;
            }
            if best.is_empty() {
                continue;
            }

            retrieved.fill(0.0);
            let mut weight_sum = 0.0_f32;
            for (dist, row) in best {
                let d = dist.max(1e-12);
                let w = (1.0 / d) * (1.0 / d);
                weight_sum += w;
                for c in 0..dims {
                    retrieved[c] += index.vectors[(row, c)] * w;
                }
            }
            if weight_sum <= f32::EPSILON {
                continue;
            }

            for c in 0..dims {
                let mut from_index = retrieved[c] / weight_sum;
                if has_prev {
                    from_index = prev_vec[c] * index_smooth + from_index * (1.0 - index_smooth);
                }
                prev_vec[c] = from_index;
                let current = phone[(0, t, c)];
                phone[(0, t, c)] = current * (1.0 - rate) + from_index * rate;
            }
            has_prev = true;
        }
        self.index_prev_vector = prev_vec;
    }

    fn run_rvc(
        &mut self,
        phone: &Array3<f32>,
        pitchf: &[f32],
        pitch: &[i64],
        speaker_id: i64,
        frame_len: i64,
        rnd: &Array3<f32>,
        waveform: &Array3<f32>,
    ) -> Result<Vec<f32>> {
        let mut input_map: Vec<(String, SessionInputValue<'static>)> = Vec::new();
        let fallback_type =
            |rank: usize, ty: Option<TensorElementType>, name: &str| -> RvcInputKind {
                if rank == 3 && ty == Some(TensorElementType::Float32) {
                    return RvcInputKind::Phone;
                }
                if rank == 2 && ty == Some(TensorElementType::Float32) {
                    return RvcInputKind::PitchF;
                }
                if rank == 2 && ty == Some(TensorElementType::Int64) {
                    return RvcInputKind::Pitch;
                }
                if rank == 1 && ty == Some(TensorElementType::Int64) {
                    if name.contains("len") {
                        return RvcInputKind::Length;
                    }
                    return RvcInputKind::Sid;
                }
                RvcInputKind::Wave
            };

        for input in self.rvc_session.inputs() {
            let name = input.name().to_string();
            let lname = name.to_lowercase();
            let rank = input.dtype().tensor_shape().map_or(1, |shape| shape.len());
            let ty = input.dtype().tensor_type();
            let kind =
                classify_rvc_input(&lname).unwrap_or_else(|| fallback_type(rank, ty, &lname));

            let value = match kind {
                RvcInputKind::Phone => {
                    SessionInputValue::from(Tensor::from_array(phone.clone()).map_err(|e| {
                        VcError::Inference(format!("failed to create phone tensor: {e}"))
                    })?)
                }
                RvcInputKind::PitchF => {
                    let arr = Array2::from_shape_vec((1, pitchf.len()), pitchf.to_vec()).map_err(
                        |e| VcError::Inference(format!("failed to shape pitchf as [1, T]: {e}")),
                    )?;
                    SessionInputValue::from(Tensor::from_array(arr).map_err(|e| {
                        VcError::Inference(format!("failed to create pitchf tensor: {e}"))
                    })?)
                }
                RvcInputKind::Pitch => {
                    let arr =
                        Array2::from_shape_vec((1, pitch.len()), pitch.to_vec()).map_err(|e| {
                            VcError::Inference(format!("failed to shape pitch as [1, T]: {e}"))
                        })?;
                    SessionInputValue::from(Tensor::from_array(arr).map_err(|e| {
                        VcError::Inference(format!("failed to create pitch tensor: {e}"))
                    })?)
                }
                RvcInputKind::Sid => {
                    let arr = Array1::from_vec(vec![speaker_id]);
                    SessionInputValue::from(Tensor::from_array(arr).map_err(|e| {
                        VcError::Inference(format!("failed to create sid tensor: {e}"))
                    })?)
                }
                RvcInputKind::Length => {
                    let arr = Array1::from_vec(vec![frame_len]);
                    SessionInputValue::from(Tensor::from_array(arr).map_err(|e| {
                        VcError::Inference(format!("failed to create length tensor: {e}"))
                    })?)
                }
                RvcInputKind::Wave => {
                    SessionInputValue::from(Tensor::from_array(waveform.clone()).map_err(|e| {
                        VcError::Inference(format!("failed to create wave tensor: {e}"))
                    })?)
                }
                RvcInputKind::Rnd => {
                    SessionInputValue::from(Tensor::from_array(rnd.clone()).map_err(|e| {
                        VcError::Inference(format!("failed to create rnd tensor: {e}"))
                    })?)
                }
            };
            input_map.push((name, value));
        }

        let outputs = self
            .rvc_session
            .run(input_map)
            .map_err(|e| VcError::Inference(format!("rvc inference failed: {e}")))?;
        if outputs.len() == 0 {
            return Err(VcError::Inference(
                "rvc model returned no outputs".to_string(),
            ));
        }

        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            VcError::Inference(format!("failed to extract rvc output tensor<f32>: {e}"))
        })?;
        Ok(data.to_vec())
    }

    fn make_rnd_tensor(&mut self, frame_count: usize) -> Array3<f32> {
        let t = frame_count.max(1);
        let mut out = Array3::<f32>::zeros((1, 192, t));
        for c in 0..192 {
            for i in 0..t {
                out[(0, c, i)] = self.next_standard_normal();
            }
        }
        out
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64* (fast, deterministic, no extra dependency).
        let mut x = self.rnd_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rnd_state = x;
        x.wrapping_mul(2685821657736338717)
    }

    fn next_unit(&mut self) -> f32 {
        let x = self.next_u64();
        (((x >> 40) as u32) as f32 + 0.5) / (1u32 << 24) as f32
    }

    fn next_standard_normal(&mut self) -> f32 {
        // Box-Muller transform (matches numpy.random.randn intent: N(0,1)).
        let u1 = self.next_unit().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.next_unit();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    fn try_rvc_runtime_cpu_fallback(&mut self, cause: &VcError) -> Result<bool> {
        if self.rvc_runtime_cpu_fallback_tried {
            return Ok(false);
        }
        if !is_rvc_cuda_runtime_failure(cause) {
            return Ok(false);
        }
        self.rvc_runtime_cpu_fallback_tried = true;

        eprintln!(
            "[vc-inference] warning: rvc runtime failed on GPU EP; switching rvc session to CPU EP. cause={}",
            cause
        );

        let ort_intra_threads = read_env_usize("RUST_VC_ORT_INTRA_THREADS")
            .unwrap_or_else(default_ort_intra_threads)
            .max(1);
        let ort_inter_threads = read_env_usize("RUST_VC_ORT_INTER_THREADS")
            .unwrap_or(1)
            .max(1);
        let ort_parallel_execution = read_env_bool("RUST_VC_ORT_PARALLEL").unwrap_or(false);
        let cpu_execution = OrtExecutionConfig {
            provider: OrtProvider::Cpu,
            device_id: 0,
            gpu_mem_limit_bytes: None,
        };

        let cpu_session = build_ort_session(
            &self.model.model_path,
            "rvc(cpu-fallback)",
            ort_intra_threads,
            ort_inter_threads,
            ort_parallel_execution,
            &cpu_execution,
        )
        .map_err(|e| {
            VcError::Inference(format!(
                "rvc runtime GPU failure and CPU fallback session init failed: {e}"
            ))
        })?;
        self.rvc_session = cpu_session;
        eprintln!("[vc-inference] rvc session fallback: using CPU EP");
        Ok(true)
    }
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], config: &RuntimeConfig) -> Result<Vec<f32>> {
        let mut infer = || -> Result<Vec<f32>> {
            let normalized = normalize_for_onnx_input(frame, 0.95);
            let rmvpe_pad = pad_for_rmvpe(&normalized, RVC_HOP_LENGTH);
            let fallback_frames =
                (normalized.len() as u64).div_ceil(RVC_HOP_LENGTH as u64) as usize;
            let fallback_frames = fallback_frames.max(1);
            let hubert_target_frames =
                if self.hubert_upsample_factor > 1 && self.hubert_session.is_some() {
                    fallback_frames.div_ceil(self.hubert_upsample_factor)
                } else {
                    fallback_frames
                }
                .max(1);
            let rmvpe_threshold = if config.rmvpe_threshold.is_finite() {
                config.rmvpe_threshold
            } else {
                DEFAULT_RMVPE_THRESHOLD
            };
            let f0 = self.estimate_pitch(
                &rmvpe_pad.padded,
                config.sample_rate,
                fallback_frames,
                rmvpe_threshold,
            )?;

            let mut phone =
                self.extract_phone_features(&normalized, config.sample_rate, hubert_target_frames)?;
            let mut phone_raw = None;
            if config.protect < 0.5 {
                phone_raw = Some(phone.clone());
            }
            self.blend_phone_with_index(&mut phone, config);
            if self.hubert_upsample_factor > 1 && self.hubert_session.is_some() {
                phone = upsample_phone_frames(&phone, self.hubert_upsample_factor);
                if let Some(raw) = phone_raw.as_mut() {
                    *raw = upsample_phone_frames(raw, self.hubert_upsample_factor);
                }
            }
            let frame_count = fallback_frames.max(1);
            phone = align_phone_frames_to_target(&phone, frame_count);
            if let Some(raw) = phone_raw.take() {
                let raw = align_phone_frames_to_target(&raw, frame_count);
                apply_unvoiced_protect(&mut phone, &raw, &f0, config.protect);
            }

            let mut pitchf = align_pitch_frames_to_target(&f0, frame_count);
            apply_pitch_shift_inplace(&mut pitchf, config.pitch_shift_semitones);
            let f0_median_radius = config.f0_median_filter_radius;
            if f0_median_radius > 0 {
                median_filter_pitch_track_inplace(&mut pitchf, f0_median_radius);
            }
            let pitch_smooth_alpha = if config.pitch_smooth_alpha.is_finite() {
                config.pitch_smooth_alpha.max(0.0)
            } else {
                self.default_pitch_smooth_alpha
            };
            if pitch_smooth_alpha > 0.0 {
                smooth_pitch_track(&mut pitchf, &mut self.last_pitch_hz, pitch_smooth_alpha);
            }
            let pitch = coarse_pitch_from_f0(&pitchf);
            let rnd = self.make_rnd_tensor(frame_count);
            let wave = Array3::from_shape_vec((1, 1, normalized.len()), normalized.clone())
                .map_err(|e| {
                    VcError::Inference(format!("failed to shape waveform as [1,1,T]: {e}"))
                })?;

            let out = match self.run_rvc(
                &phone,
                &pitchf,
                &pitch,
                config.speaker_id,
                frame_count as i64,
                &rnd,
                &wave,
            ) {
                Ok(out) => out,
                Err(err) => {
                    if !self.try_rvc_runtime_cpu_fallback(&err)? {
                        return Err(err);
                    }
                    self.run_rvc(
                        &phone,
                        &pitchf,
                        &pitch,
                        config.speaker_id,
                        frame_count as i64,
                        &rnd,
                        &wave,
                    )?
                }
            };
            let mixed = apply_rms_mix(&normalized, &out, config.rms_mix_rate);
            let processed = postprocess_generated_audio(&mixed);
            Ok(match_output_length(&processed, frame.len()))
        };

        match infer() {
            Ok(out) => Ok(out),
            Err(e) => {
                eprintln!("inference failed, outputting silence: {e}");
                Ok(vec![0.0; frame.len()])
            }
        }
    }
}

fn match_output_length(samples: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if samples.is_empty() {
        return vec![0.0; target_len];
    }
    if samples.len() == target_len {
        return samples.to_vec();
    }
    if samples.len() == 1 {
        return vec![samples[0]; target_len];
    }

    let mut out = Vec::with_capacity(target_len);
    let scale = (samples.len() - 1) as f64 / (target_len - 1).max(1) as f64;
    for i in 0..target_len {
        let src = i as f64 * scale;
        let left = src.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (src - left as f64) as f32;
        out.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }
    out
}

#[derive(Debug, Clone, Copy)]
enum RvcInputKind {
    Phone,
    Pitch,
    PitchF,
    Sid,
    Length,
    Wave,
    Rnd,
}

fn classify_rvc_input(name: &str) -> Option<RvcInputKind> {
    if name.contains("len") || name.contains("p_len") {
        return Some(RvcInputKind::Length);
    }
    if name == "ds" || name.contains("sid") || name.contains("spk") || name.contains("speaker") {
        return Some(RvcInputKind::Sid);
    }
    if name.contains("pitchf") || name.contains("nsff0") || name.contains("f0") {
        return Some(RvcInputKind::PitchF);
    }
    if name == "pitch" || name.contains("pitch_") {
        return Some(RvcInputKind::Pitch);
    }
    if name.contains("rnd") || name.contains("noise") {
        return Some(RvcInputKind::Rnd);
    }
    if name.contains("phone") || name.contains("hubert") || name.contains("feat") {
        return Some(RvcInputKind::Phone);
    }
    if name.contains("wav") || name.contains("audio") {
        return Some(RvcInputKind::Wave);
    }
    None
}

fn fallback_rvc_input_kind(
    rank: usize,
    ty: Option<TensorElementType>,
    name: &str,
) -> RvcInputKind {
    if rank == 3 && ty == Some(TensorElementType::Float32) {
        return RvcInputKind::Phone;
    }
    if rank == 2 && ty == Some(TensorElementType::Float32) {
        return RvcInputKind::PitchF;
    }
    if rank == 2 && ty == Some(TensorElementType::Int64) {
        return RvcInputKind::Pitch;
    }
    if rank == 1 && ty == Some(TensorElementType::Int64) {
        if name.contains("len") {
            return RvcInputKind::Length;
        }
        return RvcInputKind::Sid;
    }
    RvcInputKind::Wave
}

fn classify_rvc_input_with_fallback(
    name: &str,
    rank: usize,
    ty: Option<TensorElementType>,
) -> RvcInputKind {
    classify_rvc_input(name).unwrap_or_else(|| fallback_rvc_input_kind(rank, ty, name))
}

fn tensor_from_audio_rank(rank: usize, samples: &[f32]) -> Result<Tensor<f32>> {
    tensor_from_audio_rank_owned(rank, samples.to_vec())
}

fn tensor_from_audio_rank_owned(rank: usize, samples: Vec<f32>) -> Result<Tensor<f32>> {
    match rank {
        1 => Tensor::from_array(Array1::from_vec(samples))
            .map_err(|e| VcError::Inference(format!("failed to create rank1 audio tensor: {e}"))),
        2 => {
            Tensor::from_array(Array2::from_shape_vec((1, samples.len()), samples).map_err(
                |e| VcError::Inference(format!("failed to shape rank2 audio tensor: {e}")),
            )?)
            .map_err(|e| VcError::Inference(format!("failed to create rank2 audio tensor: {e}")))
        }
        _ => Tensor::from_array(
            Array3::from_shape_vec((1, 1, samples.len()), samples).map_err(|e| {
                VcError::Inference(format!("failed to shape rank3 audio tensor: {e}"))
            })?,
        )
        .map_err(|e| VcError::Inference(format!("failed to create rank3 audio tensor: {e}"))),
    }
}

fn infer_phone_feature_dim_from_session(session: &Session) -> usize {
    for input in session.inputs() {
        let name = input.name().to_lowercase();
        if !(name.contains("phone") || name.contains("hubert") || name.contains("feat")) {
            continue;
        }
        if let Some(shape) = input.dtype().tensor_shape() {
            if shape.len() >= 3 && shape[2] > 0 {
                return shape[2] as usize;
            }
        }
    }
    256
}

fn infer_hubert_output_dim_from_session(session: &Session) -> Option<usize> {
    for output in session.outputs() {
        if let Some(shape) = output.dtype().tensor_shape() {
            if shape.len() >= 3 && shape[2] > 0 {
                return Some(shape[2] as usize);
            }
            if shape.len() >= 2 && shape[1] > 0 {
                return Some(shape[1] as usize);
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Default)]
struct RvcRequirements {
    needs_phone_features: bool,
    needs_pitch: bool,
}

fn infer_rvc_requirements(session: &Session) -> RvcRequirements {
    let mut req = RvcRequirements::default();
    for input in session.inputs() {
        let lname = input.name().to_lowercase();
        let rank = input.dtype().tensor_shape().map_or(1, |shape| shape.len());
        let ty = input.dtype().tensor_type();
        match classify_rvc_input_with_fallback(&lname, rank, ty) {
            RvcInputKind::Phone => req.needs_phone_features = true,
            RvcInputKind::Pitch | RvcInputKind::PitchF => req.needs_pitch = true,
            _ => {}
        }
    }
    req
}

fn phone_from_hubert_tensor(
    shape: &[i64],
    hubert: &[f32],
    feature_dim: usize,
) -> Result<Array3<f32>> {
    if hubert.is_empty() {
        return Err(VcError::Inference(
            "hubert output tensor is empty".to_string(),
        ));
    }

    let dims = shape
        .iter()
        .map(|&v| if v > 0 { v as usize } else { 0 })
        .collect::<Vec<usize>>();
    let out = match dims.len() {
        3 => {
            let batch = dims[0];
            let frames = dims[1].max(1);
            let channels = dims[2].max(1);
            if batch == 0 {
                return Err(VcError::Inference(
                    "hubert output has zero batch".to_string(),
                ));
            }
            if hubert.len() < batch * frames * channels {
                return Err(VcError::Inference(
                    "hubert output tensor data is smaller than declared shape".to_string(),
                ));
            }
            let mut arr = Array3::<f32>::zeros((1, frames, channels));
            for t in 0..frames {
                for c in 0..channels {
                    let idx = t * channels + c;
                    arr[(0, t, c)] = hubert[idx];
                }
            }
            arr
        }
        2 => {
            let frames = dims[0].max(1);
            let channels = dims[1].max(1);
            if hubert.len() < frames * channels {
                return Err(VcError::Inference(
                    "hubert output tensor data is smaller than declared shape".to_string(),
                ));
            }
            let mut arr = Array3::<f32>::zeros((1, frames, channels));
            for t in 0..frames {
                for c in 0..channels {
                    arr[(0, t, c)] = hubert[t * channels + c];
                }
            }
            arr
        }
        1 => {
            let frames = dims[0].max(1);
            if hubert.len() < frames {
                return Err(VcError::Inference(
                    "hubert output tensor data is smaller than declared shape".to_string(),
                ));
            }
            let mut arr = Array3::<f32>::zeros((1, frames, 1));
            for t in 0..frames {
                arr[(0, t, 0)] = hubert[t];
            }
            arr
        }
        _ => {
            return Err(VcError::Inference(format!(
                "unsupported hubert output rank: {}",
                dims.len()
            )));
        }
    };

    if out.shape()[2] != feature_dim {
        return Err(VcError::Inference(format!(
            "hubert output dim mismatch at runtime: hubert={}, rvc_phone={feature_dim}",
            out.shape()[2]
        )));
    }
    Ok(out)
}

fn tail_phone_frames(phone: &Array3<f32>, frames: usize) -> Array3<f32> {
    let want = frames.max(1);
    let total = phone.shape()[1];
    let dims = phone.shape()[2];
    if total <= want {
        return phone.clone();
    }
    let start = total - want;
    let mut out = Array3::<f32>::zeros((1, want, dims));
    for t in 0..want {
        for c in 0..dims {
            out[(0, t, c)] = phone[(0, start + t, c)];
        }
    }
    out
}

fn upsample_phone_frames(phone: &Array3<f32>, factor: usize) -> Array3<f32> {
    if factor <= 1 {
        return phone.clone();
    }
    let src_frames = phone.shape()[1].max(1);
    let dims = phone.shape()[2];
    let target_frames = (src_frames * factor).max(1);
    let mut out = Array3::<f32>::zeros((1, target_frames, dims));
    for t in 0..target_frames {
        let src_t = (t / factor).min(src_frames - 1);
        for c in 0..dims {
            out[(0, t, c)] = phone[(0, src_t, c)];
        }
    }
    out
}

fn align_phone_frames_to_target(phone: &Array3<f32>, target_frames: usize) -> Array3<f32> {
    let frames = phone.shape()[1].max(1);
    let target = target_frames.max(1);
    if frames == target {
        return phone.clone();
    }
    if frames > target {
        return tail_phone_frames(phone, target);
    }

    // Keep temporal alignment by extending the tail frame, instead of time-warp interpolation.
    let dims = phone.shape()[2];
    let mut out = Array3::<f32>::zeros((1, target, dims));
    for t in 0..target {
        let src_t = t.min(frames - 1);
        for c in 0..dims {
            out[(0, t, c)] = phone[(0, src_t, c)];
        }
    }
    out
}

fn align_pitch_frames_to_target(pitch: &[f32], target_frames: usize) -> Vec<f32> {
    let target = target_frames.max(1);
    if pitch.is_empty() {
        return vec![0.0; target];
    }
    if pitch.len() == target {
        return pitch.to_vec();
    }
    if pitch.len() > target {
        return pitch[pitch.len() - target..].to_vec();
    }

    let mut out = pitch.to_vec();
    let fill = out.last().copied().unwrap_or(0.0);
    out.resize(target, fill);
    out
}

fn run_hubert_session(
    session: &mut Session,
    source: &[f32],
    pad_left: usize,
    phone_feature_dim: usize,
    hubert_output_layer: i64,
) -> Result<Array3<f32>> {
    let mut inputs: Vec<(String, SessionInputValue<'static>)> = Vec::new();
    let source_len = source.len().max(1);
    for input in session.inputs() {
        let name = input.name().to_string();
        let lname = name.to_lowercase();
        let rank = input.dtype().tensor_shape().map_or(2, |s| s.len());
        let ty = input.dtype().tensor_type();
        let value = if lname.contains("source") || ty == Some(TensorElementType::Float32) {
            let tensor = tensor_from_audio_rank(rank.max(2), source)?;
            SessionInputValue::from(tensor)
        } else if lname.contains("padding") || ty == Some(TensorElementType::Bool) {
            let mut mask = vec![false; source_len];
            let pad = pad_left.min(source_len);
            for m in &mut mask[..pad] {
                *m = true;
            }
            if rank <= 1 {
                SessionInputValue::from(
                    Tensor::from_array(Array1::from_vec(mask.clone())).map_err(|e| {
                        VcError::Inference(format!(
                            "failed to create hubert padding mask rank1: {e}"
                        ))
                    })?,
                )
            } else {
                SessionInputValue::from(
                    Tensor::from_array(Array2::from_shape_vec((1, source_len), mask).map_err(
                        |e| {
                            VcError::Inference(format!(
                                "failed to shape hubert padding mask as [1,T]: {e}"
                            ))
                        },
                    )?)
                    .map_err(|e| {
                        VcError::Inference(format!(
                            "failed to create hubert padding mask rank2: {e}"
                        ))
                    })?,
                )
            }
        } else if ty == Some(TensorElementType::Int64) {
            let scalar = if lname.contains("output_layer") || lname.contains("layer") {
                hubert_output_layer
            } else if lname.contains("sample_rate") || lname == "sr" {
                RMVPE_SAMPLE_RATE as i64
            } else {
                source_len as i64
            };
            SessionInputValue::from(Tensor::from_array(Array1::from_vec(vec![scalar])).map_err(
                |e| {
                    VcError::Inference(format!(
                        "failed to create hubert int64 side-input tensor: {e}"
                    ))
                },
            )?)
        } else {
            return Err(VcError::Inference(format!(
                "unsupported hubert input '{}' type={ty:?} rank={rank}",
                name
            )));
        };
        inputs.push((name, value));
    }

    let outputs = session.run(inputs).map_err(|e| {
        VcError::Inference(format!(
            "hubert inference failed (source_len={source_len}, pad_left={pad_left}): {e}"
        ))
    })?;
    if outputs.len() == 0 {
        return Err(VcError::Inference(
            "hubert model returned no outputs".to_string(),
        ));
    }
    let mut best: Option<(usize, Vec<i64>, Vec<f32>)> = None;
    for idx in 0..outputs.len() {
        let Ok((shape, data)) = outputs[idx].try_extract_tensor::<f32>() else {
            continue;
        };
        let shape_vec = shape.to_vec();
        let feature_axis = shape_vec.last().copied().unwrap_or_default();
        let has_expected_dim = feature_axis == phone_feature_dim as i64;
        let rank_score = shape_vec.len();
        let score = (if has_expected_dim { 100 } else { 0 }) + rank_score;
        let replace = best.as_ref().map(|(s, _, _)| score > *s).unwrap_or(true);
        if replace {
            best = Some((score, shape_vec, data.to_vec()));
        }
    }
    let Some((_, shape_vec, data_vec)) = best else {
        return Err(VcError::Inference(
            "failed to extract any hubert output tensor<f32>".to_string(),
        ));
    };
    phone_from_hubert_tensor(&shape_vec, &data_vec, phone_feature_dim)
}

fn run_hubert_session_with_len_fallback(
    session: &mut Session,
    source: &[f32],
    pad_left: usize,
    phone_feature_dim: usize,
    hubert_output_layer: i64,
    len_fixes: &mut HashMap<usize, usize>,
) -> Result<Array3<f32>> {
    let requested_len = source.len().max(1);
    if !len_fixes.contains_key(&requested_len) {
        if let Some(fixed_len) = get_global_hubert_len_fix(requested_len) {
            len_fixes.insert(requested_len, fixed_len);
        }
    }

    if let Some(&fixed_len) = len_fixes.get(&requested_len) {
        let (patched_source, patched_pad_left) =
            remap_hubert_source_len(source, pad_left, fixed_len.max(1));
        match run_hubert_session(
            session,
            &patched_source,
            patched_pad_left,
            phone_feature_dim,
            hubert_output_layer,
        ) {
            Ok(phone) => return Ok(phone),
            Err(err) => {
                if !is_hubert_length_compat_error(&err) {
                    return Err(err);
                }
                len_fixes.remove(&requested_len);
                remove_global_hubert_len_fix(requested_len);
            }
        }
    }

    let candidates = hubert_fallback_candidates(requested_len);
    let mut tried = HashSet::<usize>::new();
    let precheck_candidates = requested_len >= 6_000;
    if precheck_candidates {
        for &candidate_len in &candidates {
            if candidate_len == requested_len {
                continue;
            }
            tried.insert(candidate_len);
            let (patched_source, patched_pad_left) =
                remap_hubert_source_len(source, pad_left, candidate_len);
            match run_hubert_session(
                session,
                &patched_source,
                patched_pad_left,
                phone_feature_dim,
                hubert_output_layer,
            ) {
                Ok(phone) => {
                    len_fixes.insert(requested_len, candidate_len);
                    set_global_hubert_len_fix(requested_len, candidate_len);
                    eprintln!(
                        "[vc-inference] hubert source_len adjusted (precheck): requested={} -> {} (pad_left={} -> {})",
                        requested_len, candidate_len, pad_left, patched_pad_left
                    );
                    return Ok(phone);
                }
                Err(err) => {
                    if !is_hubert_length_compat_error(&err) {
                        return Err(err);
                    }
                }
            }
        }
    }

    let mut last_err = match run_hubert_session(
        session,
        source,
        pad_left,
        phone_feature_dim,
        hubert_output_layer,
    ) {
        Ok(phone) => return Ok(phone),
        Err(err) if !is_hubert_length_compat_error(&err) => return Err(err),
        Err(err) => Some(err),
    };

    for candidate_len in candidates {
        if candidate_len == requested_len || tried.contains(&candidate_len) {
            continue;
        }
        let (patched_source, patched_pad_left) =
            remap_hubert_source_len(source, pad_left, candidate_len);
        match run_hubert_session(
            session,
            &patched_source,
            patched_pad_left,
            phone_feature_dim,
            hubert_output_layer,
        ) {
            Ok(phone) => {
                len_fixes.insert(requested_len, candidate_len);
                set_global_hubert_len_fix(requested_len, candidate_len);
                eprintln!(
                    "[vc-inference] hubert source_len adjusted: requested={} -> {} (pad_left={} -> {})",
                    requested_len, candidate_len, pad_left, patched_pad_left
                );
                return Ok(phone);
            }
            Err(err) => {
                if !is_hubert_length_compat_error(&err) {
                    return Err(err);
                }
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        VcError::Inference(format!(
            "hubert inference failed and no compatible source length found (requested_len={requested_len})"
        ))
    }))
}

fn hubert_fallback_candidates(requested_len: usize) -> Vec<usize> {
    let mut candidates = Vec::<usize>::new();
    let positive_offsets: &[usize] = if requested_len >= 6_000 {
        &[
            320, 640, 160, 80, 256, 128, 384, 512, 768, 1024, 64, 32, 16, 8, 1, 96, 192,
        ]
    } else {
        &[
            80, 320, 160, 64, 128, 256, 384, 512, 768, 1024, 32, 16, 8, 1, 96, 192,
        ]
    };
    for &off in positive_offsets {
        if let Some(v) = requested_len.checked_add(off) {
            candidates.push(v);
        }
    }
    for off in [80usize, 160, 64, 32, 16, 8, 1, 96, 128, 192] {
        if requested_len > off {
            candidates.push(requested_len - off);
        }
    }
    candidates.push(DEFAULT_HUBERT_CONTEXT_SAMPLES_16K);
    candidates.push(4_096);
    candidates.push(5_201);
    candidates
        .into_iter()
        .filter(|&v| v >= 1_600 && v <= MAX_HUBERT_CONTEXT_SAMPLES_16K)
        .collect()
}

fn get_global_hubert_len_fix(requested_len: usize) -> Option<usize> {
    let cache = HUBERT_LEN_FIX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache.lock().ok()?.get(&requested_len).copied()
}

fn set_global_hubert_len_fix(requested_len: usize, fixed_len: usize) {
    let cache = HUBERT_LEN_FIX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(requested_len, fixed_len.max(1));
    }
}

fn remove_global_hubert_len_fix(requested_len: usize) {
    let cache = HUBERT_LEN_FIX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut guard) = cache.lock() {
        guard.remove(&requested_len);
    }
}

fn remap_hubert_source_len(
    source: &[f32],
    pad_left: usize,
    target_len: usize,
) -> (Vec<f32>, usize) {
    let target_len = target_len.max(1);
    let source_len = source.len().max(1);
    if target_len == source_len {
        return (source.to_vec(), pad_left.min(target_len));
    }
    if target_len > source_len {
        let add = target_len - source_len;
        let mut out = vec![0.0_f32; target_len];
        out[add..].copy_from_slice(source);
        return (out, (pad_left + add).min(target_len));
    }

    let drop = source_len - target_len;
    let mut out = vec![0.0_f32; target_len];
    out.copy_from_slice(&source[drop..]);
    (out, pad_left.saturating_sub(drop))
}

fn is_hubert_length_compat_error(err: &VcError) -> bool {
    let VcError::Inference(msg) = err else {
        return false;
    };
    msg.contains("hubert inference failed")
        && (msg.contains("Where node")
            || msg.contains("Broadcast")
            || msg.contains("Invalid input shape"))
}

fn is_hubert_cuda_runtime_failure(err: &VcError) -> bool {
    let VcError::Inference(msg) = err else {
        return false;
    };
    let m = msg.to_ascii_lowercase();
    if !m.contains("hubert") {
        return false;
    }
    is_cuda_kernel_compat_failure_text(&m)
        || m.contains("onnxruntime_providers_cuda.dll")
        || (m.contains("cuda") && m.contains("failed"))
}

fn is_rvc_cuda_runtime_failure(err: &VcError) -> bool {
    let VcError::Inference(msg) = err else {
        return false;
    };
    let m = msg.to_ascii_lowercase();
    if !m.contains("rvc inference failed") {
        return false;
    }
    is_cuda_kernel_compat_failure_text(&m)
        || m.contains("onnxruntime_providers_cuda.dll")
        || (m.contains("cuda") && m.contains("failed"))
}

fn is_cuda_kernel_compat_failure_text(message_lc: &str) -> bool {
    message_lc.contains("no kernel image is available for execution on the device")
        || message_lc.contains("cudnn_fe failure")
        || message_lc.contains("cudnn_status_execution_failed_cudart")
        || message_lc.contains("cudnn failure 5003")
}

fn decode_rmvpe_output(shape: &[i64], data: &[f32], threshold: f32) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if shape.len() >= 3 {
        let batch = shape[0].max(1) as usize;
        let time = if shape[1] > 0 {
            shape[1] as usize
        } else {
            data.len() / 360
        };
        let bins = if shape[2] > 0 { shape[2] as usize } else { 360 };
        if bins == 360 && data.len() >= batch * time * bins {
            let mut out = Vec::<f32>::with_capacity(time);
            for t in 0..time {
                let base = t * bins;
                out.push(decode_rmvpe_salience_row(
                    &data[base..base + bins],
                    threshold,
                ));
            }
            return Ok(out);
        }
    }
    if shape.len() == 2 && shape[1] == 360 {
        let time = shape[0].max(1) as usize;
        if data.len() >= time * 360 {
            let mut out = Vec::<f32>::with_capacity(time);
            for t in 0..time {
                let base = t * 360;
                out.push(decode_rmvpe_salience_row(
                    &data[base..base + 360],
                    threshold,
                ));
            }
            return Ok(out);
        }
    }
    Ok(data.to_vec())
}

fn decode_rmvpe_salience_row(row: &[f32], threshold: f32) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    if best_val <= threshold.clamp(0.0, 1.0) {
        return 0.0;
    }

    let start = best_idx.saturating_sub(4);
    let end = (best_idx + 5).min(row.len());
    let mut cents_num = 0.0_f32;
    let mut cents_den = 0.0_f32;
    for (offset, &score) in row[start..end].iter().enumerate() {
        let bin = start + offset;
        let w = score.max(1e-12);
        let cents = 1997.3794_f32 + 20.0_f32 * bin as f32;
        cents_num += cents * w;
        cents_den += w;
    }
    if cents_den <= f32::EPSILON {
        return 0.0;
    }
    let cents = cents_num / cents_den;
    10.0_f32 * 2.0_f32.powf(cents / 1200.0_f32)
}

fn load_feature_index(path: &str, target_dim: usize) -> Result<FeatureIndex> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "index" {
        let cache_path = index_cache_path(path);
        if is_cache_fresh(Path::new(path), &cache_path) {
            match load_feature_index_cache(&cache_path, target_dim) {
                Ok(index) => {
                    eprintln!(
                        "[vc-inference] index loaded from cache: {}",
                        cache_path.display()
                    );
                    return Ok(index);
                }
                Err(e) => {
                    eprintln!("[vc-inference] index cache read failed (will regenerate): {e}");
                }
            }
        }

        let (rows, dims, flat) = extract_faiss_vectors_with_python(path)?;
        let index = feature_index_from_flat(rows, dims, flat.clone(), path, target_dim)?;
        if let Err(e) = write_feature_index_cache(&cache_path, rows, dims, &flat) {
            eprintln!(
                "[vc-inference] failed to write index cache {}: {e}",
                cache_path.display()
            );
        } else {
            eprintln!(
                "[vc-inference] index cache written: {}",
                cache_path.display()
            );
        }
        return Ok(index);
    }
    if ext == "bin" || ext == "f32" {
        return load_feature_index_from_bin(path, target_dim);
    }

    load_feature_index_from_text(path, target_dim)
}

fn load_feature_index_from_bin(path: &str, target_dim: usize) -> Result<FeatureIndex> {
    let raw = fs::read(path)
        .map_err(|e| VcError::Config(format!("failed to read binary index file ({path}): {e}")))?;
    let flat = decode_f32_le(&raw).map_err(|e| {
        VcError::Config(format!(
            "failed to decode binary index f32 payload ({path}): {e}"
        ))
    })?;
    if flat.is_empty() {
        return Err(VcError::Config(format!(
            "binary index file has no vectors: {path}"
        )));
    }
    let dim = if target_dim > 0 {
        target_dim
    } else {
        read_env_usize("RUST_VC_INDEX_BIN_DIM").unwrap_or(DEFAULT_BIN_INDEX_DIM)
    };
    if dim == 0 {
        return Err(VcError::Config(
            "binary index dimension is 0; check RUST_VC_INDEX_BIN_DIM".to_string(),
        ));
    }
    if flat.len() % dim != 0 {
        return Err(VcError::Config(format!(
            "binary index size mismatch: values={} is not divisible by dim={} ({path})",
            flat.len(),
            dim
        )));
    }
    let rows = flat.len() / dim;
    eprintln!(
        "[vc-inference] binary index loaded: rows={} dims={} path={}",
        rows, dim, path
    );
    feature_index_from_flat(rows, dim, flat, path, target_dim)
}

fn load_feature_index_from_text(path: &str, target_dim: usize) -> Result<FeatureIndex> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| VcError::Config(format!("failed to read index file as text ({path}): {e}")))?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut dims = 0usize;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let values: Vec<f32> = trimmed
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|v| !v.is_empty())
            .map(|v| {
                v.parse::<f32>().map_err(|e| {
                    VcError::Config(format!("invalid float in index file {path}: {e}"))
                })
            })
            .collect::<Result<Vec<f32>>>()?;
        if values.is_empty() {
            continue;
        }
        if dims == 0 {
            dims = values.len();
        }
        if values.len() != dims {
            continue;
        }
        rows.push(values);
    }

    if rows.is_empty() || dims == 0 {
        return Err(VcError::Config(format!(
            "index file has no usable vectors: {path}"
        )));
    }

    let mut flat = Vec::<f32>::with_capacity(rows.len() * dims);
    for row in rows {
        flat.extend(row);
    }
    feature_index_from_flat(flat.len() / dims, dims, flat, path, target_dim)
}

fn feature_index_from_flat(
    rows: usize,
    dims: usize,
    flat: Vec<f32>,
    source: &str,
    target_dim: usize,
) -> Result<FeatureIndex> {
    if rows == 0 || dims == 0 || flat.is_empty() {
        return Err(VcError::Config(format!(
            "index has no usable vectors: {source}"
        )));
    }
    if flat.len() != rows * dims {
        return Err(VcError::Config(format!(
            "index vector size mismatch: rows={rows} dims={dims} values={}",
            flat.len()
        )));
    }
    if target_dim > 0 && dims != target_dim {
        return Err(VcError::Config(format!(
            "index dim mismatch: index={dims}, rvc_phone={target_dim}"
        )));
    }
    let vectors = Array2::from_shape_vec((rows, dims), flat)
        .map_err(|e| VcError::Config(format!("failed to build index matrix from {source}: {e}")))?;
    Ok(FeatureIndex { vectors })
}

fn index_cache_path(index_path: &str) -> PathBuf {
    PathBuf::from(format!("{index_path}.rustvc.cache"))
}

fn is_cache_fresh(index_path: &Path, cache_path: &Path) -> bool {
    let Ok(index_meta) = fs::metadata(index_path) else {
        return false;
    };
    let Ok(cache_meta) = fs::metadata(cache_path) else {
        return false;
    };
    let Ok(index_mtime) = index_meta.modified() else {
        return false;
    };
    let Ok(cache_mtime) = cache_meta.modified() else {
        return false;
    };
    cache_mtime >= index_mtime
}

fn load_feature_index_cache(cache_path: &Path, target_dim: usize) -> Result<FeatureIndex> {
    let raw = fs::read(cache_path).map_err(|e| {
        VcError::Config(format!(
            "failed to read index cache {}: {e}",
            cache_path.display()
        ))
    })?;
    if raw.len() < 8 {
        return Err(VcError::Config(format!(
            "index cache is too small: {}",
            cache_path.display()
        )));
    }
    let rows = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    let dims = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let payload = &raw[8..];
    let flat = decode_f32_le(payload).map_err(|e| {
        VcError::Config(format!(
            "failed to decode index cache {}: {e}",
            cache_path.display()
        ))
    })?;
    feature_index_from_flat(
        rows,
        dims,
        flat,
        &cache_path.display().to_string(),
        target_dim,
    )
}

fn write_feature_index_cache(
    cache_path: &Path,
    rows: usize,
    dims: usize,
    flat: &[f32],
) -> io::Result<()> {
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "index rows exceed u32"))?;
    let dims_u32 = u32::try_from(dims)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "index dims exceed u32"))?;
    let mut out = Vec::<u8>::with_capacity(8 + flat.len() * 4);
    out.extend_from_slice(&rows_u32.to_le_bytes());
    out.extend_from_slice(&dims_u32.to_le_bytes());
    for v in flat {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(cache_path, out)
}

fn decode_f32_le(raw: &[u8]) -> io::Result<Vec<f32>> {
    if raw.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload byte length is not divisible by 4",
        ));
    }
    let mut out = Vec::<f32>::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn extract_faiss_vectors_with_python(index_path: &str) -> Result<(usize, usize, Vec<f32>)> {
    let max_vectors = read_env_usize("RUST_VC_INDEX_MAX_VECTORS").unwrap_or(0);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let dump_path = std::env::temp_dir().join(format!(
        "rust_vc_index_dump_{ts}_{}.f32",
        std::process::id()
    ));
    let dump_path_str = dump_path.to_string_lossy().to_string();
    let py = r#"
import faiss
import numpy as np
import sys

index_path = sys.argv[1]
dump_path = sys.argv[2]
max_vectors = int(sys.argv[3])

index = faiss.read_index(index_path)
count = int(index.ntotal)
if max_vectors > 0 and count > max_vectors:
    count = max_vectors

vecs = index.reconstruct_n(0, count).astype(np.float32, copy=False)
vecs.tofile(dump_path)
print(f"{vecs.shape[0]} {vecs.shape[1]}")
"#;
    let output = Command::new("python")
        .args(["-c", py, index_path, &dump_path_str, &max_vectors.to_string()])
        .output()
        .map_err(|e| {
            VcError::Config(format!(
                "failed to run python for .index load. install python + faiss-cpu or clear index_path: {e}"
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(VcError::Config(format!(
            "python/faiss failed to load index {index_path}: {stderr}"
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (rows, dims) = parse_rows_dims(stdout.trim()).ok_or_else(|| {
        VcError::Config(format!(
            "failed to parse python/faiss output while loading {index_path}: '{}'",
            stdout.trim()
        ))
    })?;
    let raw = fs::read(&dump_path).map_err(|e| {
        VcError::Config(format!(
            "failed to read extracted index vectors {}: {e}",
            dump_path.display()
        ))
    })?;
    let _ = fs::remove_file(&dump_path);
    let flat = decode_f32_le(&raw).map_err(|e| {
        VcError::Config(format!(
            "failed to decode extracted vectors {}: {e}",
            dump_path.display()
        ))
    })?;
    let expected = rows.saturating_mul(dims);
    if expected == 0 || flat.len() < expected {
        return Err(VcError::Config(format!(
            "faiss extracted vectors are too small: rows={rows} dims={dims} values={}",
            flat.len()
        )));
    }
    let mut flat = flat;
    if flat.len() > expected {
        flat.truncate(expected);
    }
    eprintln!(
        "[vc-inference] loaded faiss index vectors: rows={} dims={} max_vectors={}",
        rows, dims, max_vectors
    );
    Ok((rows, dims, flat))
}

fn parse_rows_dims(s: &str) -> Option<(usize, usize)> {
    let nums: Vec<usize> = s
        .split_whitespace()
        .filter_map(|tok| tok.parse::<usize>().ok())
        .collect();
    if nums.len() < 2 {
        return None;
    }
    Some((nums[0], nums[1]))
}

fn push_top_k(best: &mut Vec<(f32, usize)>, cand: (f32, usize), k: usize) {
    let mut idx = 0usize;
    while idx < best.len() && best[idx].0 <= cand.0 {
        idx += 1;
    }
    if idx < k {
        best.insert(idx, cand);
        if best.len() > k {
            best.pop();
        }
    } else if best.len() < k {
        best.push(cand);
    }
}

fn apply_unvoiced_protect(
    phone: &mut Array3<f32>,
    phone_raw: &Array3<f32>,
    f0_hz: &[f32],
    protect: f32,
) {
    let protect = protect.clamp(0.0, 0.5);
    if protect >= 0.5 {
        return;
    }
    let frames = phone.shape()[1];
    let dims = phone.shape()[2];
    if phone_raw.shape() != phone.shape() {
        return;
    }
    if frames == 0 || dims == 0 {
        return;
    }
    let pitchf = resize_pitch_to_frames(f0_hz, frames);
    for t in 0..frames {
        if pitchf[t] >= 1.0 {
            continue;
        }
        for c in 0..dims {
            let transformed = phone[(0, t, c)];
            let raw = phone_raw[(0, t, c)];
            phone[(0, t, c)] = transformed * protect + raw * (1.0 - protect);
        }
    }
}

fn apply_pitch_shift_inplace(f0: &mut [f32], semitones: f32) {
    if semitones.abs() <= f32::EPSILON {
        return;
    }
    let ratio = 2.0_f32.powf(semitones / 12.0);
    for v in f0 {
        if *v > 0.0 {
            *v *= ratio;
        }
    }
}

fn smooth_pitch_track(pitchf: &mut [f32], last_pitch_hz: &mut f32, alpha: f32) {
    let a = if alpha.is_finite() && alpha > 0.0 {
        alpha / (1.0 + alpha)
    } else {
        0.0
    };
    let mut last = *last_pitch_hz;
    for p in pitchf {
        if *p > 0.0 {
            if last > 0.0 {
                *p = last * a + *p * (1.0 - a);
            }
            last = *p;
        } else {
            last *= 0.95;
            if last < 1.0 {
                last = 0.0;
            }
        }
    }
    *last_pitch_hz = last;
}

fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.trim().parse::<usize>().ok()
}

fn read_env_i64(key: &str) -> Option<i64> {
    std::env::var(key).ok()?.trim().parse::<i64>().ok()
}

fn read_env_i32(key: &str) -> Option<i32> {
    std::env::var(key).ok()?.trim().parse::<i32>().ok()
}

fn read_env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok()?.trim().parse::<f32>().ok()
}

fn read_env_bool(key: &str) -> Option<bool> {
    let raw = std::env::var(key).ok()?;
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn default_ort_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
        .max(1)
}

fn parse_ort_provider(raw: &str) -> OrtProvider {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => OrtProvider::Cpu,
        "cuda" => OrtProvider::Cuda,
        "directml" | "dml" => OrtProvider::DirectMl,
        "auto" | "" => OrtProvider::Auto,
        _ => OrtProvider::Auto,
    }
}

fn resolve_ort_execution_config(runtime: &RuntimeConfig) -> OrtExecutionConfig {
    let provider_raw = std::env::var("RUST_VC_ORT_PROVIDER")
        .ok()
        .unwrap_or_else(|| runtime.ort_provider.clone());
    let provider = parse_ort_provider(&provider_raw);
    let device_id = read_env_i32("RUST_VC_ORT_DEVICE_ID").unwrap_or(runtime.ort_device_id);
    let gpu_mem_limit_mb = read_env_usize("RUST_VC_ORT_GPU_MEM_LIMIT_MB")
        .unwrap_or(runtime.ort_gpu_mem_limit_mb as usize);
    let gpu_mem_limit_bytes = if gpu_mem_limit_mb == 0 {
        None
    } else {
        Some(gpu_mem_limit_mb.saturating_mul(1024 * 1024))
    };
    OrtExecutionConfig {
        provider,
        device_id,
        gpu_mem_limit_bytes,
    }
}

fn build_execution_providers(config: &OrtExecutionConfig) -> Vec<ExecutionProviderDispatch> {
    let cuda_ep = || {
        let mut cuda = ep::CUDA::default().with_device_id(config.device_id);
        if let Some(limit) = config.gpu_mem_limit_bytes {
            cuda = cuda.with_memory_limit(limit);
        }
        let conv_algo = resolve_cuda_conv_algorithm();
        let conv_max_workspace = read_env_bool("RUST_VC_CUDA_CONV_MAX_WORKSPACE").unwrap_or(false);
        let conv1d_pad_to_nc1d = read_env_bool("RUST_VC_CUDA_CONV1D_PAD_TO_NC1D").unwrap_or(false);
        let tf32 = read_env_bool("RUST_VC_CUDA_TF32").unwrap_or(false);
        if ORT_CUDA_TUNE_LOGGED.set(()).is_ok() {
            eprintln!(
                "[vc-inference] cuda ep tuning: conv_algo={:?} conv_max_workspace={} conv1d_pad_to_nc1d={} tf32={} (env: RUST_VC_CUDA_CONV_ALGO / RUST_VC_CUDA_CONV_MAX_WORKSPACE / RUST_VC_CUDA_CONV1D_PAD_TO_NC1D / RUST_VC_CUDA_TF32)",
                conv_algo, conv_max_workspace, conv1d_pad_to_nc1d, tf32
            );
        }
        cuda = cuda
            .with_conv_algorithm_search(conv_algo)
            .with_conv_max_workspace(conv_max_workspace)
            .with_conv1d_pad_to_nc1d(conv1d_pad_to_nc1d)
            .with_tf32(tf32);
        cuda.build()
    };
    let dml_ep = || {
        ep::DirectML::default()
            .with_device_id(config.device_id)
            .build()
    };

    let availability = detect_ort_provider_availability();
    if ORT_PROVIDER_AVAIL_LOGGED.set(()).is_ok() {
        eprintln!(
            "[vc-inference] ort providers available: cuda={} dml={} cpu={}",
            availability.cuda, availability.dml, availability.cpu
        );
    }

    match config.provider {
        OrtProvider::Cpu => vec![ep::CPU::default().build().error_on_failure()],
        OrtProvider::Cuda => {
            if !availability.cuda {
                if ORT_CUDA_MISSING_WARN_LOGGED.set(()).is_ok() {
                    eprintln!(
                        "[vc-inference] warning: CUDA provider requested but unavailable in current onnxruntime.dll; startup may fail unless GPU-enabled ORT is installed"
                    );
                }
            }
            vec![cuda_ep().error_on_failure()]
        }
        OrtProvider::DirectMl => {
            if !availability.dml {
                if ORT_DML_MISSING_WARN_LOGGED.set(()).is_ok() {
                    eprintln!(
                        "[vc-inference] warning: DirectML provider requested but unavailable in current onnxruntime.dll; startup may fail unless DirectML-enabled ORT is installed"
                    );
                }
            }
            vec![dml_ep().error_on_failure()]
        }
        OrtProvider::Auto => {
            let mut eps = Vec::<ExecutionProviderDispatch>::new();
            if availability.cuda {
                eps.push(cuda_ep().error_on_failure());
            }
            if availability.dml {
                eps.push(dml_ep().error_on_failure());
            }
            if eps.is_empty() {
                if ORT_AUTO_CPU_WARN_LOGGED.set(()).is_ok() {
                    eprintln!(
                        "[vc-inference] warning: Auto provider resolved to CPU only (no CUDA/DML available). For typical RVC realtime, install a GPU-enabled onnxruntime.dll."
                    );
                }
                eps.push(ep::CPU::default().build().error_on_failure());
            }
            eps
        }
    }
}

fn detect_ort_provider_availability() -> OrtProviderAvailability {
    let cuda = ep::CUDA::default().is_available().unwrap_or(false);
    let dml = ep::DirectML::default().is_available().unwrap_or(false);
    let cpu = ep::CPU::default().is_available().unwrap_or(true);
    OrtProviderAvailability { cuda, dml, cpu }
}

fn resolve_cuda_conv_algorithm() -> ep::cuda::ConvAlgorithmSearch {
    let raw = std::env::var("RUST_VC_CUDA_CONV_ALGO")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "default".to_string());
    match raw.as_str() {
        "exhaustive" | "exh" => ep::cuda::ConvAlgorithmSearch::Exhaustive,
        "heuristic" | "heu" => ep::cuda::ConvAlgorithmSearch::Heuristic,
        "default" | "def" | "" => ep::cuda::ConvAlgorithmSearch::Default,
        _ => ep::cuda::ConvAlgorithmSearch::Default,
    }
}

fn build_ort_session(
    path: &str,
    label: &str,
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
    execution: &OrtExecutionConfig,
) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| VcError::Inference(format!("failed to create {label} session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VcError::Inference(format!("failed to set {label} optimization level: {e}")))?
        .with_intra_threads(intra_threads)
        .map_err(|e| VcError::Inference(format!("failed to set {label} intra threads: {e}")))?
        .with_inter_threads(inter_threads)
        .map_err(|e| VcError::Inference(format!("failed to set {label} inter threads: {e}")))?
        .with_parallel_execution(parallel_execution)
        .map_err(|e| VcError::Inference(format!("failed to set {label} execution mode: {e}")))?;
    let eps = build_execution_providers(execution);
    builder = builder.with_execution_providers(eps).map_err(|e| {
        let raw = e.to_string();
        let raw_lc = raw.to_ascii_lowercase();
        if raw_lc.contains("cublaslt64_12.dll")
            || (raw_lc.contains("onnxruntime_providers_cuda.dll") && raw_lc.contains("missing"))
        {
            let missing = extract_missing_dep_dll_name(&raw)
                .unwrap_or_else(|| "CUDA runtime DLL".to_string());
            VcError::Inference(format!(
                "failed to set {label} execution providers: {raw}. \
Missing dependency: {missing}. \
Run scripts\\install_onnxruntime_provider.bat cuda (CUDA12) or cuda11 (legacy CUDA11), then retry. \
If CUDA is unavailable on this machine, switch ort_provider to directml."
            ))
        } else if matches!(execution.provider, OrtProvider::Cuda)
            && is_cuda_kernel_compat_failure_text(&raw_lc)
        {
            VcError::Inference(format!(
                "failed to set {label} execution providers: {raw}. \
CUDA kernel compatibility issue detected. \
Try scripts\\install_onnxruntime_provider.bat cuda11 (legacy CUDA11 feed, recommended for GTX 10xx and older). \
If your GPU supports CUDA12 + cuDNN9, keep cuda and update NVIDIA driver/runtime."
            ))
        } else if raw.contains("onnxruntime_providers_shared.dll") {
            VcError::Inference(format!(
                "failed to set {label} execution providers: {raw}. \
ONNX Runtime DLL bundle is incomplete. \
Place onnxruntime.dll and onnxruntime*.dll together (onnxruntime_providers_shared.dll is required). \
On Windows, run scripts\\install_onnxruntime_provider.bat cuda or directml."
            ))
        } else {
            VcError::Inference(format!("failed to set {label} execution providers: {raw}"))
        }
    })?;
    builder.commit_from_file(path).map_err(|e| {
        let raw = e.to_string();
        let raw_lc = raw.to_ascii_lowercase();
        if matches!(execution.provider, OrtProvider::Cuda)
            && is_cuda_kernel_compat_failure_text(&raw_lc)
        {
            VcError::Inference(format!(
                "failed to load {label} onnx model {path}: {raw}. \
CUDA kernel compatibility issue detected. \
Try scripts\\install_onnxruntime_provider.bat cuda11 (legacy CUDA11 feed, recommended for GTX 10xx and older). \
If your GPU supports CUDA12 + cuDNN9, keep cuda and update NVIDIA driver/runtime."
            ))
        } else {
            VcError::Inference(format!("failed to load {label} onnx model {path}: {raw}"))
        }
    })
}

fn build_aux_session_with_cpu_fallback(
    path: &str,
    label: &str,
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
    execution: &OrtExecutionConfig,
) -> Result<Session> {
    match build_ort_session(
        path,
        label,
        intra_threads,
        inter_threads,
        parallel_execution,
        execution,
    ) {
        Ok(session) => Ok(session),
        Err(primary_err) => {
            let primary_text = primary_err.to_string();
            if !should_try_aux_cpu_fallback(execution.provider, &primary_text) {
                return Err(primary_err);
            }

            eprintln!(
                "[vc-inference] warning: {label} session init failed on {:?}; retrying with CPU EP. cause={}",
                execution.provider, primary_text
            );

            let mut cpu_execution = *execution;
            cpu_execution.provider = OrtProvider::Cpu;
            cpu_execution.gpu_mem_limit_bytes = None;

            match build_ort_session(
                path,
                label,
                intra_threads,
                inter_threads,
                parallel_execution,
                &cpu_execution,
            ) {
                Ok(session) => {
                    eprintln!("[vc-inference] {label} session fallback: using CPU EP");
                    Ok(session)
                }
                Err(cpu_err) => Err(VcError::Inference(format!(
                    "{label} session init failed on {:?}: {primary_text}; CPU fallback also failed: {cpu_err}",
                    execution.provider
                ))),
            }
        }
    }
}

fn should_try_aux_cpu_fallback(provider: OrtProvider, error_text: &str) -> bool {
    if matches!(provider, OrtProvider::Cpu) {
        return false;
    }
    let msg = error_text.to_ascii_lowercase();
    msg.contains("onnxruntime_providers_cuda.dll")
        || msg.contains("onnxruntime_providers_dml.dll")
        || msg.contains("cuda")
        || msg.contains("cudnn")
        || msg.contains("directml")
        || msg.contains("providerlibrary::get")
        || msg.contains("failed to set")
}

fn extract_missing_dep_dll_name(message: &str) -> Option<String> {
    let marker = "depends on \"";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('"')?;
    let dll = rest[..end].trim();
    if dll.is_empty() {
        None
    } else {
        Some(dll.to_string())
    }
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
