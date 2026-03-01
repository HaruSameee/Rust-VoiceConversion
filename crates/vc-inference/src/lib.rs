use std::{
    any::Any,
    collections::VecDeque,
    env, fs, io,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use ndarray::{s, Array1, Array2, Array3};
use ort::{
    ep::{self, ExecutionProvider, ExecutionProviderDispatch},
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    tensor::TensorElementType,
    value::Tensor,
};
use vc_core::{emit_log as emit_backend_log, InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};
use vc_signal::{
    apply_rms_mix, coarse_pitch_from_f0, median_filter_pitch_track_inplace,
    normalize_for_onnx_input, postprocess_generated_audio, resample_hq_into,
    resize_pitch_to_frames, rmvpe_mel_from_audio, HqResampler, RMVPE_SAMPLE_RATE,
};

pub mod audio_pipeline;
#[cfg(feature = "faiss-native")]
pub mod faiss_index;
pub mod ivf_index;
pub mod zero_copy_engine;
#[cfg(feature = "faiss-native")]
use faiss_index::FaissIndex;
use ivf_index::{IvfIndex, IVF_MAGIC};

const STRICT_ONNX_INPUT_SAMPLES_16K: usize = 16_000;
const STRICT_HUBERT_OUTPUT_FRAMES: usize = 50;
const DEFAULT_RMVPE_THRESHOLD: f32 = 0.01;
const TIMING_LOG_EVERY_BLOCKS: u64 = 10;
// scipy.signal.butter(N=5, Wn=48, btype="high", fs=16000, output="sos")
const HP48_BUTTER5_SOS_16K: [[f32; 6]; 3] = [
    [0.96996063, -0.96996063, 0.0, 1.0, -0.98132586, 0.0],
    [1.0, -2.0, 1.0, 1.0, -1.9696107, 0.96996063],
    [1.0, -2.0, 1.0, 1.0, -1.9880652, 0.98841846],
];
const HP48_FILTFILT_PADLEN_MIN: usize = 32;
const HP48_FILTFILT_PADLEN_MAX: usize = 512;
static ORT_INIT_FAILED: OnceLock<String> = OnceLock::new();
static ORT_PROVIDER_AVAIL_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_CUDA_MISSING_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_DML_MISSING_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_AUTO_CPU_WARN_LOGGED: OnceLock<()> = OnceLock::new();
static ORT_CUDA_TUNE_LOGGED: OnceLock<()> = OnceLock::new();
static TENSOR_STATS_DEBUG: OnceLock<bool> = OnceLock::new();

fn emit_info(message: &str) {
    emit_backend_log("info", message);
}

fn emit_warn(message: &str) {
    emit_backend_log("warn", message);
}

fn emit_timing(message: &str) {
    emit_backend_log("timing", message);
}

#[derive(Debug, Clone, Copy)]
struct TensorStats {
    total_count: usize,
    finite_count: usize,
    nan_count: usize,
    inf_count: usize,
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct TensorStatsAccum {
    total_count: usize,
    finite_count: usize,
    nan_count: usize,
    inf_count: usize,
    min: f32,
    max: f32,
    sum: f64,
    sum_sq: f64,
}

impl TensorStatsAccum {
    fn push(&mut self, v: f32) {
        self.total_count += 1;
        if v.is_nan() {
            self.nan_count += 1;
            return;
        }
        if v.is_infinite() {
            self.inf_count += 1;
            return;
        }
        if self.finite_count == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.finite_count += 1;
        let vf = v as f64;
        self.sum += vf;
        self.sum_sq += vf * vf;
    }

    fn finish(self) -> TensorStats {
        if self.finite_count == 0 {
            return TensorStats {
                total_count: self.total_count,
                finite_count: 0,
                nan_count: self.nan_count,
                inf_count: self.inf_count,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                rms: 0.0,
            };
        }
        let denom = self.finite_count as f64;
        TensorStats {
            total_count: self.total_count,
            finite_count: self.finite_count,
            nan_count: self.nan_count,
            inf_count: self.inf_count,
            min: self.min,
            max: self.max,
            mean: (self.sum / denom) as f32,
            rms: (self.sum_sq / denom).sqrt() as f32,
        }
    }
}

fn tensor_stats_enabled() -> bool {
    *TENSOR_STATS_DEBUG.get_or_init(|| {
        env::var("RUST_VC_DEBUG_TENSOR_STATS")
            .map(|v| {
                let s = v.trim().to_ascii_lowercase();
                s == "1" || s == "true" || s == "yes" || s == "on"
            })
            .unwrap_or(false)
    })
}

fn tensor_stats_from_iter<I>(iter: I) -> TensorStats
where
    I: IntoIterator<Item = f32>,
{
    let mut acc = TensorStatsAccum::default();
    for v in iter {
        acc.push(v);
    }
    acc.finish()
}

fn maybe_log_tensor_stats(label: &str, stats: TensorStats) {
    if !tensor_stats_enabled() {
        return;
    }
    eprintln!(
        "[vc-inference] stats {label}: total={} finite={} nan={} inf={} min={:.6e} max={:.6e} mean={:.6e} rms={:.6e}",
        stats.total_count,
        stats.finite_count,
        stats.nan_count,
        stats.inf_count,
        stats.min,
        stats.max,
        stats.mean,
        stats.rms
    );
}

fn ensure_tensor_finite(label: &str, stats: TensorStats) -> Result<()> {
    if stats.nan_count > 0 || stats.inf_count > 0 {
        return Err(VcError::Inference(format!(
            "{label} contains non-finite values: nan={} inf={} total={}",
            stats.nan_count, stats.inf_count, stats.total_count
        )));
    }
    Ok(())
}

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
struct OrtCudaTuning {
    conv_algo: ep::cuda::ConvAlgorithmSearch,
    conv_max_workspace: bool,
    conv1d_pad_to_nc1d: bool,
    tf32: bool,
}

#[derive(Debug, Clone)]
struct FeatureIndex {
    vectors: Array2<f32>,
    vectors_t: Array2<f32>,
}

#[derive(Debug)]
enum IndexBackend {
    Binary(FeatureIndex),
    #[cfg(feature = "faiss-native")]
    Faiss {
        search: FaissIndex,
        vectors: FeatureIndex,
    },
    Ivf(IvfIndex),
}

#[derive(Debug, Clone)]
struct PitchEstimate {
    f0_hz: Vec<f32>,
    confidence: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct IndexBlendStats {
    rows: usize,
    top_k: usize,
    provider: &'static str,
    nprobe: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexProvider {
    Cpu,
    Gpu,
}

pub struct RvcOrtEngine {
    model: ModelConfig,
    rvc_session: Session,
    rmvpe_session: Option<Session>,
    hubert_session: Option<Session>,
    feature_index: Option<IndexBackend>,
    phone_feature_dim: usize,
    input_to_16k_resampler: HqResampler,
    ort_intra_threads: usize,
    ort_inter_threads: usize,
    ort_parallel_execution: bool,
    ort_cuda_tuning: OrtCudaTuning,
    zero_copy_engine: Option<Arc<Mutex<zero_copy_engine::ZeroCopyInferenceEngine>>>,
    rvc_runtime_cpu_fallback_tried: bool,
    default_pitch_smooth_alpha: f32,
    hubert_upsample_factor: usize,
    index_prev_vector: Vec<f32>,
    last_pitch_hz: f32,
    source_16k_context: VecDeque<f32>,
    rmvpe_16k_context: VecDeque<f32>,
    hubert_context_tail_16k: Vec<f32>,
    configured_context_samples_16k: usize,
    configured_hop_samples_16k: usize,
    hop_samples_16k: usize,
    rnd_state: u64,
    infer_block_count: u64,
    post_filter_prev_sample: f32,
    post_filter_prev_valid: bool,
    post_filter_sample_rate: u32,
    index_gpu_fallback_warned: bool,
}

impl RvcOrtEngine {
    pub fn new(model: ModelConfig, runtime_config: &RuntimeConfig) -> Result<Self> {
        eprintln!(
            "[vc-inference] init model={} hubert={:?} rmvpe={:?} index={:?}",
            model.model_path, model.hubert_path, model.pitch_extractor_path, model.index_path
        );
        let mut model = model;
        let requested_model_path = PathBuf::from(&model.model_path);
        let resolved_model_path =
            resolve_model_path(&requested_model_path, runtime_config.block_size.max(1));
        if resolved_model_path != requested_model_path {
            eprintln!(
                "[vc-inference] model path resolved: {} (requested={} block_size={})",
                resolved_model_path.display(),
                requested_model_path.display(),
                runtime_config.block_size
            );
        } else {
            eprintln!(
                "[vc-inference] model path fallback: {} (no block-specific variant)",
                requested_model_path.display()
            );
        }
        model.model_path = resolved_model_path.to_string_lossy().into_owned();
        if let Some(msg) = ORT_INIT_FAILED.get() {
            return Err(VcError::Config(msg.clone()));
        }
        if !Path::new(&model.model_path).exists() {
            return Err(VcError::Config(format!(
                "model file not found: {}",
                model.model_path
            )));
        }
        eprintln!(
            "[vc-inference] loaded model: {} for block_size={}",
            model.model_path, runtime_config.block_size
        );

        let build_res = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Self> {
            ort::init().commit();
            let ort_intra_threads = runtime_config.intra_threads as usize;
            let ort_inter_threads = runtime_config.inter_threads as usize;
            let ort_parallel_execution = runtime_config.ort_parallel_execution;
            let hubert_upsample_factor = runtime_config.hubert_upsample_factor.max(1);
            let decoder_frame_count = STRICT_HUBERT_OUTPUT_FRAMES
                .saturating_mul(hubert_upsample_factor)
                .max(1);
            let ort_ep = resolve_ort_execution_config(runtime_config);
            let ort_cuda_tuning = resolve_ort_cuda_tuning(runtime_config);
            let pitch_smooth_alpha = if runtime_config.pitch_smooth_alpha.is_finite() {
                runtime_config.pitch_smooth_alpha
            } else {
                0.12
            }
            .max(0.0);
            eprintln!(
                "[vc-inference] ort intra_threads={} inter_threads={} parallel={} provider={:?} dev={} vram_limit_mb={} strict_input_16k={} strict_hubert_frames={} hubert_up={} decoder_frames={} pitch_smooth_alpha={:.2}",
                if ort_intra_threads == 0 {
                    "auto".to_string()
                } else {
                    ort_intra_threads.to_string()
                },
                if ort_inter_threads == 0 {
                    "auto".to_string()
                } else {
                    ort_inter_threads.to_string()
                },
                ort_parallel_execution,
                ort_ep.provider,
                ort_ep.device_id,
                ort_ep
                    .gpu_mem_limit_bytes
                    .map(|v| v / (1024 * 1024))
                    .unwrap_or(0),
                STRICT_ONNX_INPUT_SAMPLES_16K,
                STRICT_HUBERT_OUTPUT_FRAMES,
                hubert_upsample_factor,
                decoder_frame_count,
                pitch_smooth_alpha
            );

            let rvc_session = build_ort_session(
                &model.model_path,
                "rvc",
                ort_intra_threads,
                ort_inter_threads,
                ort_parallel_execution,
                &ort_ep,
                &ort_cuda_tuning,
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
                    &ort_cuda_tuning,
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
                    &ort_cuda_tuning,
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
                        let msg = format!("[vc-inference] index file not found, disable index: {path}");
                        eprintln!("{msg}");
                        emit_warn(&msg);
                        None
                    } else {
                        match load_index_backend(
                            path,
                            phone_feature_dim,
                            runtime_config.index_bin_dim.max(1),
                            runtime_config.index_max_vectors,
                            runtime_config.index_nprobe.max(1),
                        ) {
                            Ok(index) => Some(index),
                            Err(e) => {
                                let msg = format!("index disabled: {e}");
                                eprintln!("{msg}");
                                emit_warn(&msg);
                                None
                            }
                        }
                    }
                }
                None => None,
            };
            let block_samples_16k = estimate_hop_samples_16k(
                runtime_config.block_size.max(1),
                runtime_config.sample_rate.max(1),
            );
            let hop_samples_16k = block_samples_16k.min(STRICT_ONNX_INPUT_SAMPLES_16K).max(1);
            let requested_context_samples = (runtime_config.hubert_context_sec.clamp(0.25, 2.0)
                * RMVPE_SAMPLE_RATE as f32)
                .round() as usize;
            let context_samples = STRICT_ONNX_INPUT_SAMPLES_16K.saturating_sub(hop_samples_16k);
            let hops_per_block = 1usize;
            let frames_per_hop = STRICT_HUBERT_OUTPUT_FRAMES;
            let total_frames = STRICT_HUBERT_OUTPUT_FRAMES;
            eprintln!(
                "[vc-inference] sliding_window: single-window window={}smp hop={}smp context={}smp ({:.1}sec) @ 16kHz requested_context={}smp ({:.1}sec) hops_per_block={} frames_per_hop={} total_frames={}",
                STRICT_ONNX_INPUT_SAMPLES_16K,
                hop_samples_16k,
                context_samples,
                context_samples as f32 / RMVPE_SAMPLE_RATE as f32,
                requested_context_samples,
                requested_context_samples as f32 / RMVPE_SAMPLE_RATE as f32,
                hops_per_block,
                frames_per_hop,
                total_frames
            );
            eprintln!(
                "[vc-inference] context_window initialized: {} zeros (context={}smp hop={}smp)",
                context_samples, context_samples, hop_samples_16k
            );
            let post_filter_alpha = if runtime_config.post_filter_alpha.is_finite() {
                runtime_config.post_filter_alpha.clamp(0.0, 0.999)
            } else {
                0.96
            };
            eprintln!("[vc-inference] post_filter: alpha={:.2}", post_filter_alpha);
            eprintln!(
                "[vc-inference] confidence_gate: rmvpe_th={:.3} applied to index blend",
                runtime_config.rmvpe_threshold.max(0.0)
            );
            eprintln!(
                "[vc-inference] index provider preference: {}",
                runtime_config.index_provider
            );
            eprintln!(
                "[vc-inference] preprocessing: 48Hz high-pass (Butterworth-5, filtfilt-like) enabled for HuBERT/RMVPE inputs"
            );
            eprintln!("[vc-inference] execution path: standard copy (zero-copy disabled)");
            let zero_copy_engine: Option<Arc<Mutex<zero_copy_engine::ZeroCopyInferenceEngine>>> =
                None;

            Ok(Self {
                model,
                rvc_session,
                rmvpe_session,
                hubert_session,
                feature_index,
                phone_feature_dim,
                input_to_16k_resampler: HqResampler::new(
                    runtime_config.sample_rate.max(1),
                    RMVPE_SAMPLE_RATE,
                ),
                ort_intra_threads,
                ort_inter_threads,
                ort_parallel_execution,
                ort_cuda_tuning,
                zero_copy_engine,
                rvc_runtime_cpu_fallback_tried: false,
                default_pitch_smooth_alpha: pitch_smooth_alpha,
                hubert_upsample_factor,
                index_prev_vector: Vec::new(),
                last_pitch_hz: 0.0,
                source_16k_context: {
                    let mut buf = VecDeque::with_capacity(STRICT_ONNX_INPUT_SAMPLES_16K);
                    buf.resize(STRICT_ONNX_INPUT_SAMPLES_16K, 0.0);
                    buf
                },
                rmvpe_16k_context: {
                    let mut buf = VecDeque::with_capacity(STRICT_ONNX_INPUT_SAMPLES_16K);
                    buf.resize(STRICT_ONNX_INPUT_SAMPLES_16K, 0.0);
                    buf
                },
                hubert_context_tail_16k: vec![0.0; context_samples],
                configured_context_samples_16k: context_samples,
                configured_hop_samples_16k: hop_samples_16k,
                hop_samples_16k,
                rnd_state: 0x9E37_79B9_7F4A_7C15,
                infer_block_count: 0,
                post_filter_prev_sample: 0.0,
                post_filter_prev_valid: false,
                post_filter_sample_rate: runtime_config.sample_rate.max(1),
                index_gpu_fallback_warned: false,
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
        source_16k: &[f32],
        threshold: f32,
        target_decoder_frames: usize,
    ) -> Result<PitchEstimate> {
        let Some(session) = self.rmvpe_session.as_mut() else {
            // Models that do not require RMVPE keep legacy index behavior.
            return Ok(PitchEstimate {
                f0_hz: vec![0.0; target_decoder_frames.max(1)],
                confidence: vec![1.0; target_decoder_frames.max(1)],
            });
        };
        if source_16k.is_empty() {
            return Ok(PitchEstimate {
                f0_hz: vec![0.0; target_decoder_frames.max(1)],
                confidence: vec![0.0; target_decoder_frames.max(1)],
            });
        }

        let inputs = session.inputs();
        if inputs.len() != 1 {
            return Err(VcError::Inference(format!(
                "strict rmvpe expects exactly 1 input, got {}",
                inputs.len()
            )));
        }
        let inlet = &inputs[0];
        let input_name = inlet.name().to_string();
        let rank = inlet.dtype().tensor_shape().map_or(2, |shape| shape.len());
        let input_shape = inlet.dtype().tensor_shape();
        let input_name_lc = input_name.to_ascii_lowercase();
        let expects_mel = input_name_lc.contains("mel")
            || input_shape.as_ref().is_some_and(|shape| {
                shape.len() == 3 && shape.get(1).is_some_and(|&d| d <= 0 || d as usize == 128)
            });
        let input_tensor = if expects_mel {
            let mel = rmvpe_mel_from_audio(source_16k, RMVPE_SAMPLE_RATE);
            Tensor::from_array(mel.mel).map_err(|e| {
                VcError::Inference(format!("failed to create RMVPE mel tensor: {e}"))
            })?
        } else {
            let audio_samples = fixed_audio_input_len(input_shape.as_ref().map(|shape| shape.as_ref()))
                .filter(|&target_len| target_len != source_16k.len())
                .map_or_else(|| source_16k.to_vec(), |target_len| {
                    fit_waveform_to_len(source_16k, target_len)
                });
            tensor_from_audio_rank_owned(rank.max(2), audio_samples)?
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
        let rmvpe_raw_stats = tensor_stats_from_iter(data.iter().copied());
        maybe_log_tensor_stats("rmvpe_raw", rmvpe_raw_stats);
        ensure_tensor_finite("rmvpe_raw", rmvpe_raw_stats)?;
        let decoded = decode_rmvpe_output(shape, data, threshold)?;
        let f0 = resize_pitch_to_frames(&decoded.f0_hz, target_decoder_frames.max(1));
        let confidence = resize_pitch_to_frames(&decoded.confidence, target_decoder_frames.max(1))
            .into_iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect::<Vec<f32>>();
        let f0_stats = tensor_stats_from_iter(f0.iter().copied());
        maybe_log_tensor_stats("rmvpe_f0", f0_stats);
        ensure_tensor_finite("rmvpe_f0", f0_stats)?;
        Ok(PitchEstimate {
            f0_hz: f0,
            confidence,
        })
    }

    fn extract_phone_features(
        &mut self,
        source_16k: &[f32],
        target_decoder_frames: usize,
    ) -> Result<Array3<f32>> {
        if source_16k.is_empty() {
            return Ok(Array3::<f32>::zeros((
                1,
                target_decoder_frames.max(1),
                self.phone_feature_dim,
            )));
        }
        let Some(session) = self.hubert_session.as_mut() else {
            return Err(VcError::Config(
                "hubert session is unavailable. strict mode requires HuBERT ONNX.".to_string(),
            ));
        };
        let window_16k = STRICT_ONNX_INPUT_SAMPLES_16K;
        let hop_16k = self.hop_samples_16k.max(1).min(window_16k);
        let context_16k = self
            .configured_context_samples_16k
            .min(window_16k.saturating_sub(hop_16k));
        let upsample = self.hubert_upsample_factor.max(1);
        let phone_feature_dim = self.phone_feature_dim;
        let target_phone_frames = target_decoder_frames.max(1);
        let target_hubert_frames = target_phone_frames.div_ceil(upsample).max(1);
        let inputs = session.inputs();
        if inputs.len() != 1 {
            return Err(VcError::Inference(format!(
                "strict hubert expects exactly 1 input, got {}",
                inputs.len()
            )));
        }
        let inlet = &inputs[0];
        let input_name = inlet.name().to_string();
        let rank = inlet.dtype().tensor_shape().map_or(2, |shape| shape.len());
        let mut run_hubert_window = |window: &[f32]| -> Result<Array3<f32>> {
            if window.len() != window_16k {
                return Err(VcError::Inference(format!(
                    "hubert sliding window length mismatch: got={} expected={}",
                    window.len(),
                    window_16k
                )));
            }
            let input_tensor = tensor_from_audio_rank_owned(rank.max(2), window.to_vec())?;
            let outputs = session
                .run(vec![(
                    input_name.clone(),
                    SessionInputValue::from(input_tensor),
                )])
                .map_err(|e| VcError::Inference(format!("hubert inference failed: {e}")))?;
            if outputs.len() == 0 {
                return Err(VcError::Inference(
                    "hubert output tensor is empty".to_string(),
                ));
            }
            let mut selected = None::<Array3<f32>>;
            for (_, output) in &outputs {
                let Ok((shape, data)) = output.try_extract_tensor::<f32>() else {
                    continue;
                };
                if data.is_empty() {
                    continue;
                }
                let hubert_stats = tensor_stats_from_iter(data.iter().copied());
                maybe_log_tensor_stats("hubert_raw", hubert_stats);
                ensure_tensor_finite("hubert_raw", hubert_stats)?;
                if let Ok(phone3d) = phone_from_hubert_tensor(shape, data, phone_feature_dim) {
                    selected = Some(phone3d);
                    break;
                }
            }
            let Some(hubert_phone) = selected else {
                return Err(VcError::Inference(format!(
                    "hubert output tensor is empty ({} outputs, no non-empty float tensor)",
                    outputs.len()
                )));
            };
            Ok(force_frame_count_3d_tail_pad(
                hubert_phone,
                STRICT_HUBERT_OUTPUT_FRAMES,
            ))
        };
        let window = if source_16k.len() == window_16k {
            source_16k.to_vec()
        } else {
            fit_waveform_to_len(source_16k, window_16k)
        };
        let phone_hubert = run_hubert_window(&window)?;
        self.hubert_context_tail_16k = if context_16k == 0 {
            Vec::new()
        } else {
            window[window_16k - context_16k..].to_vec()
        };
        let phone_hubert = force_frame_count_3d_tail_pad(phone_hubert, target_hubert_frames);
        let phone = upsample_phone_frames_repeat(phone_hubert, target_phone_frames);
        let phone_stats = tensor_stats_from_iter(phone.iter().copied());
        maybe_log_tensor_stats("hubert_phone", phone_stats);
        ensure_tensor_finite("hubert_phone", phone_stats)?;
        Ok(phone)
    }

    fn prepare_inference_inputs_16k(
        &mut self,
        frame: &[f32],
        input_sample_rate: u32,
        hop_samples_16k: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let source_16k = self.resample_block_to_16k(frame, input_sample_rate);
        if source_16k.is_empty() {
            return Err(VcError::Inference(format!(
                "resample produced empty source: input_len={} sr={}",
                frame.len(),
                input_sample_rate
            )));
        }
        let effective_hop = hop_samples_16k.max(1).min(STRICT_ONNX_INPUT_SAMPLES_16K);
        // Advance feature contexts by the newest hop (tail region).
        let hop_advance_16k: Vec<f32> = if source_16k.len() >= effective_hop {
            source_16k[source_16k.len() - effective_hop..].to_vec()
        } else {
            // Right-align short warmup/edge blocks.
            let mut padded = vec![0.0_f32; effective_hop];
            let offset = effective_hop.saturating_sub(source_16k.len());
            padded[offset..offset + source_16k.len()].copy_from_slice(&source_16k);
            padded
        };
        debug_assert_eq!(
            hop_advance_16k.len(),
            effective_hop,
            "resample_hop mismatch: got={} expected={} (input_len={} sr={})",
            hop_advance_16k.len(),
            effective_hop,
            frame.len(),
            input_sample_rate
        );
        self.hop_samples_16k = effective_hop;
        let hubert_context_len = self
            .configured_context_samples_16k
            .min(STRICT_ONNX_INPUT_SAMPLES_16K.saturating_sub(effective_hop));
        if self.hubert_context_tail_16k.len() != hubert_context_len {
            let mut next = vec![0.0_f32; hubert_context_len];
            let keep = self.hubert_context_tail_16k.len().min(hubert_context_len);
            if keep > 0 {
                let src_start = self.hubert_context_tail_16k.len() - keep;
                let dst_start = hubert_context_len - keep;
                next[dst_start..].copy_from_slice(&self.hubert_context_tail_16k[src_start..]);
            }
            self.hubert_context_tail_16k = next;
        }
        slide_context_window(
            &mut self.source_16k_context,
            &hop_advance_16k,
            STRICT_ONNX_INPUT_SAMPLES_16K,
        );
        debug_assert_eq!(self.source_16k_context.len(), STRICT_ONNX_INPUT_SAMPLES_16K);
        slide_context_window(
            &mut self.rmvpe_16k_context,
            &hop_advance_16k,
            STRICT_ONNX_INPUT_SAMPLES_16K,
        );
        debug_assert_eq!(self.rmvpe_16k_context.len(), STRICT_ONNX_INPUT_SAMPLES_16K);
        let hubert_input = self.source_16k_context.iter().copied().collect::<Vec<f32>>();
        let rmvpe_input = self.rmvpe_16k_context.iter().copied().collect::<Vec<f32>>();
        Ok((hubert_input, rmvpe_input, 1))
    }

    fn resample_block_to_16k(&self, frame: &[f32], input_sample_rate: u32) -> Vec<f32> {
        let mut source_16k = Vec::<f32>::new();
        if input_sample_rate == RMVPE_SAMPLE_RATE {
            source_16k.extend_from_slice(frame);
        } else if input_sample_rate == self.input_to_16k_resampler.src_rate()
            && self.input_to_16k_resampler.dst_rate() == RMVPE_SAMPLE_RATE
        {
            resample_hq_into(&self.input_to_16k_resampler, frame, &mut source_16k);
        } else {
            let local = HqResampler::new(input_sample_rate.max(1), RMVPE_SAMPLE_RATE);
            resample_hq_into(&local, frame, &mut source_16k);
        }
        source_16k
    }

    fn blend_phone_with_index(
        &mut self,
        phone: &mut Array3<f32>,
        config: &RuntimeConfig,
        confidence: &[f32],
        rmvpe_threshold: f32,
    ) -> IndexBlendStats {
        let index_provider = parse_index_provider(&config.index_provider);
        if matches!(index_provider, IndexProvider::Gpu) && !self.index_gpu_fallback_warned {
            eprintln!(
                "[vc-inference] warning: index_provider=gpu requested, but GPU index backend is not available yet; falling back to CPU"
            );
            self.index_gpu_fallback_warned = true;
        }
        let requested_rows = config.index_search_rows.max(1);
        let requested_top_k = config.index_top_k.max(1).min(requested_rows);
        let requested_nprobe = config.index_nprobe.max(1);
        let Some(index_backend) = self.feature_index.as_mut() else {
            self.index_prev_vector.clear();
            return IndexBlendStats {
                rows: requested_rows,
                top_k: requested_top_k,
                provider: "none",
                nprobe: requested_nprobe,
            };
        };
        let rate = if config.index_rate.is_finite() {
            config.index_rate.max(0.0)
        } else {
            0.0
        };
        let index_smooth = if config.index_smooth_alpha.is_finite() {
            config.index_smooth_alpha.max(0.0)
        } else {
            0.0
        };
        let frames = phone.shape()[1];
        let dims = phone.shape()[2];
        let phone_matrix = phone.slice(s![0, .., ..]).to_owned();
        let mut prev_vec = if self.index_prev_vector.len() == dims {
            self.index_prev_vector.clone()
        } else {
            vec![0.0_f32; dims]
        };
        let mut has_prev = self.index_prev_vector.len() == dims;
        let conf_threshold = rmvpe_threshold.clamp(0.0, 1.0);
        let conf_denom = (1.0 - conf_threshold).max(1.0e-6);

        let (search_rows, top_k, provider, used_nprobe) = match index_backend {
            IndexBackend::Binary(index) => {
                if index.vectors.nrows() == 0 || index.vectors.ncols() == 0 {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "binary_cpu",
                        nprobe: requested_nprobe,
                    };
                }
                if phone_matrix.ncols() != index.vectors_t.nrows() {
                    eprintln!(
                        "[vc-inference] index dim mismatch at runtime: phone={} index={}",
                        phone_matrix.ncols(),
                        index.vectors_t.nrows()
                    );
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "binary_cpu",
                        nprobe: requested_nprobe,
                    };
                }

                if rate <= f32::EPSILON {
                    self.index_prev_vector.clear();
                    let rows = requested_rows.min(index.vectors.nrows());
                    let top_k = requested_top_k.min(index.vectors.nrows().max(1));
                    return IndexBlendStats {
                        rows,
                        top_k,
                        provider: "binary_cpu",
                        nprobe: requested_nprobe,
                    };
                }

                let search_rows = requested_rows.min(index.vectors.nrows()).max(1);
                let top_k = requested_top_k.min(search_rows.max(1));
                let frame_search_stride = if search_rows >= 4_096 {
                    3
                } else if search_rows >= 2_048 {
                    2
                } else {
                    1
                };
                let vectors_t_sub = index.vectors_t.slice(s![.., ..search_rows]);
                // Batched score computation:
                // phone [frames, dims] x index_t_sub [dims, search_rows] -> [frames, search_rows].
                let scores = phone_matrix.dot(&vectors_t_sub);
                let mut retrieved = vec![0.0_f32; dims];
                let mut best = Vec::<(f32, usize)>::with_capacity(top_k);

                for t in 0..frames {
                    let conf = confidence.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    // Confidence-gated rate:
                    // - below threshold: no index blend
                    // - threshold..1.0: remap to 0..1, then scale by index_rate
                    // This keeps full `index_rate` available at conf=1.0.
                    let frame_rate = if conf > conf_threshold {
                        let conf_norm = ((conf - conf_threshold) / conf_denom).clamp(0.0, 1.0);
                        rate * conf_norm
                    } else {
                        0.0
                    };
                    if frame_rate <= f32::EPSILON {
                        continue;
                    }
                    if frame_search_stride > 1 && has_prev && t % frame_search_stride != 0 {
                        for c in 0..dims {
                            let from_index = prev_vec[c];
                            let base = phone[(0, t, c)];
                            phone[(0, t, c)] = base * (1.0 - frame_rate) + from_index * frame_rate;
                        }
                        continue;
                    }

                    best.clear();
                    for row in 0..search_rows {
                        let score = scores[(t, row)];
                        push_top_k(&mut best, (score, row), top_k);
                    }
                    if best.is_empty() {
                        continue;
                    }

                    retrieved.fill(0.0);
                    let mut weight_sum = 0.0_f32;
                    let min_score = best
                        .iter()
                        .map(|(score, _)| *score)
                        .fold(f32::INFINITY, f32::min);
                    for &(score, row) in &best {
                        let w = (score - min_score + 1e-6).max(1e-6);
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
                            from_index =
                                prev_vec[c] * index_smooth + from_index * (1.0 - index_smooth);
                        }
                        prev_vec[c] = from_index;
                        let current = phone[(0, t, c)];
                        phone[(0, t, c)] = current * (1.0 - frame_rate) + from_index * frame_rate;
                    }
                    has_prev = true;
                }
                (search_rows, top_k, "binary_cpu", requested_nprobe)
            }
            #[cfg(feature = "faiss-native")]
            IndexBackend::Faiss { search, vectors } => {
                if search.dims() == 0 || search.ntotal() == 0 {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "faiss_native",
                        nprobe: requested_nprobe,
                    };
                }
                if vectors.vectors.nrows() == 0 || vectors.vectors.ncols() == 0 {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "faiss_native",
                        nprobe: requested_nprobe,
                    };
                }
                if phone_matrix.ncols() != search.dims() || phone_matrix.ncols() != vectors.vectors.ncols()
                {
                    eprintln!(
                        "[vc-inference] faiss index dim mismatch at runtime: phone={} faiss={} vectors={}",
                        phone_matrix.ncols(),
                        search.dims(),
                        vectors.vectors.ncols()
                    );
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "faiss_native",
                        nprobe: requested_nprobe,
                    };
                }

                let search_rows = requested_rows.min(search.ntotal().max(1)).max(1);
                let top_k = requested_top_k.min(search_rows.max(1));
                let clamped_nprobe = match search.set_nprobe(requested_nprobe as usize) {
                    Ok(nprobe) => nprobe,
                    Err(e) => {
                        eprintln!("[vc-inference] failed to set faiss nprobe: {e}");
                        self.index_prev_vector.clear();
                        return IndexBlendStats {
                            rows: 0,
                            top_k: 0,
                            provider: "faiss_native",
                            nprobe: requested_nprobe,
                        };
                    }
                };

                if rate <= f32::EPSILON {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k,
                        provider: "faiss_native",
                        nprobe: clamped_nprobe,
                    };
                }

                let mut retrieved = vec![0.0_f32; dims];
                for t in 0..frames {
                    let conf = confidence.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let frame_rate = if conf > conf_threshold {
                        let conf_norm = ((conf - conf_threshold) / conf_denom).clamp(0.0, 1.0);
                        rate * conf_norm
                    } else {
                        0.0
                    };
                    if frame_rate <= f32::EPSILON {
                        continue;
                    }

                    let row_view = phone_matrix.row(t);
                    let query_buf;
                    let query_vec = if let Some(slice) = row_view.as_slice() {
                        slice
                    } else {
                        query_buf = row_view.iter().copied().collect::<Vec<f32>>();
                        &query_buf
                    };
                    let neighbors = match search.search(query_vec, search_rows) {
                        Ok(ids) => ids,
                        Err(e) => {
                            eprintln!("[vc-inference] faiss search failed at runtime: {e}");
                            self.index_prev_vector.clear();
                            return IndexBlendStats {
                                rows: 0,
                                top_k: 0,
                                provider: "faiss_native",
                                nprobe: clamped_nprobe,
                            };
                        }
                    };
                    if neighbors.is_empty() {
                        continue;
                    }

                    retrieved.fill(0.0);
                    let mut weight_sum = 0.0_f32;
                    for &id in neighbors.iter().take(top_k) {
                        let row = id as usize;
                        if row >= vectors.vectors.nrows() {
                            continue;
                        }
                        let neighbor = vectors.vectors.row(row);
                        let mut distance = 0.0_f32;
                        for c in 0..dims {
                            let d = query_vec[c] - neighbor[c];
                            distance += d * d;
                        }
                        let w = 1.0 / (distance + 1.0e-6).powi(2);
                        weight_sum += w;
                        for c in 0..dims {
                            retrieved[c] += neighbor[c] * w;
                        }
                    }
                    if weight_sum <= f32::EPSILON {
                        continue;
                    }

                    for c in 0..dims {
                        let mut from_index = retrieved[c] / weight_sum;
                        if has_prev {
                            from_index =
                                prev_vec[c] * index_smooth + from_index * (1.0 - index_smooth);
                        }
                        prev_vec[c] = from_index;
                        let current = phone[(0, t, c)];
                        phone[(0, t, c)] = current * (1.0 - frame_rate) + from_index * frame_rate;
                    }
                    has_prev = true;
                }

                (search_rows, top_k, "faiss_native", clamped_nprobe)
            }
            IndexBackend::Ivf(index) => {
                if index.dims() == 0 || index.ntotal() == 0 {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "ivf_native",
                        nprobe: requested_nprobe,
                    };
                }
                if phone_matrix.ncols() != index.dims() {
                    eprintln!(
                        "[vc-inference] ivf index dim mismatch at runtime: phone={} index={}",
                        phone_matrix.ncols(),
                        index.dims()
                    );
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k: 0,
                        provider: "ivf_native",
                        nprobe: requested_nprobe,
                    };
                }

                let clamped_nprobe = requested_nprobe
                    .max(1)
                    .min(index.nlist().max(1) as u32);
                let max_rows = requested_rows.max(1);
                let top_k = requested_top_k.min(max_rows.max(1));
                if rate <= f32::EPSILON {
                    self.index_prev_vector.clear();
                    return IndexBlendStats {
                        rows: 0,
                        top_k,
                        provider: "ivf_native",
                        nprobe: clamped_nprobe,
                    };
                }

                let frame_search_stride = 1usize;

                let mut retrieved = vec![0.0_f32; dims];
                let mut scanned_rows_total = 0usize;
                let mut searched_frames = 0usize;

                for t in 0..frames {
                    let conf = confidence.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let frame_rate = if conf > conf_threshold {
                        let conf_norm = ((conf - conf_threshold) / conf_denom).clamp(0.0, 1.0);
                        rate * conf_norm
                    } else {
                        0.0
                    };
                    if frame_rate <= f32::EPSILON {
                        continue;
                    }
                    if frame_search_stride > 1 && has_prev && t % frame_search_stride != 0 {
                        for c in 0..dims {
                            let from_index = prev_vec[c];
                            let base = phone[(0, t, c)];
                            phone[(0, t, c)] = base * (1.0 - frame_rate) + from_index * frame_rate;
                        }
                        continue;
                    }

                    let row_view = phone_matrix.row(t);
                    let query_buf;
                    let query_vec = if let Some(slice) = row_view.as_slice() {
                        slice
                    } else {
                        query_buf = row_view.iter().copied().collect::<Vec<f32>>();
                        &query_buf
                    };
                    let (candidates, scanned_rows) = index.search_with_stats(
                        query_vec,
                        top_k,
                        clamped_nprobe as usize,
                    );
                    scanned_rows_total = scanned_rows_total.saturating_add(scanned_rows);
                    searched_frames = searched_frames.saturating_add(1);
                    if candidates.is_empty() {
                        continue;
                    }

                    retrieved.fill(0.0);
                    let mut weight_sum = 0.0_f32;
                    for &(distance, id) in &candidates {
                        let Some(neighbor) = index.vector(id) else {
                            continue;
                        };
                        let w = 1.0 / (distance + 1.0e-6).powi(2);
                        weight_sum += w;
                        for c in 0..dims {
                            retrieved[c] += neighbor[c] * w;
                        }
                    }
                    if weight_sum <= f32::EPSILON {
                        continue;
                    }

                    for c in 0..dims {
                        let mut from_index = retrieved[c] / weight_sum;
                        if has_prev {
                            from_index =
                                prev_vec[c] * index_smooth + from_index * (1.0 - index_smooth);
                        }
                        prev_vec[c] = from_index;
                        let current = phone[(0, t, c)];
                        phone[(0, t, c)] = current * (1.0 - frame_rate) + from_index * frame_rate;
                    }
                    has_prev = true;
                }

                let effective_rows = if searched_frames > 0 {
                    scanned_rows_total / searched_frames
                } else {
                    0
                };
                (effective_rows, top_k, "ivf_native", clamped_nprobe)
            }
        };

        self.index_prev_vector = prev_vec;
        IndexBlendStats {
            rows: search_rows,
            top_k,
            provider,
            nprobe: used_nprobe,
        }
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

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            VcError::Inference(format!("failed to extract rvc output tensor<f32>: {e}"))
        })?;
        let audio = extract_rvc_audio_from_runtime_shape(shape, data)?;
        let rvc_out_stats = tensor_stats_from_iter(audio.iter().copied());
        maybe_log_tensor_stats("rvc_out", rvc_out_stats);
        ensure_tensor_finite("rvc_out", rvc_out_stats)?;
        Ok(audio)
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

        let cpu_execution = OrtExecutionConfig {
            provider: OrtProvider::Cpu,
            device_id: 0,
            gpu_mem_limit_bytes: None,
        };

        let cpu_session = build_ort_session(
            &self.model.model_path,
            "rvc(cpu-fallback)",
            self.ort_intra_threads,
            self.ort_inter_threads,
            self.ort_parallel_execution,
            &cpu_execution,
            &self.ort_cuda_tuning,
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

    fn drop_zero_copy_engine_explicit(&mut self) {
        if let Some(engine) = self.zero_copy_engine.take() {
            match engine.lock() {
                Ok(mut guard) => guard.prepare_for_drop(),
                Err(_) => eprintln!(
                    "[vc-inference] warning: zero-copy engine mutex poisoned during explicit drop"
                ),
            }
            drop(engine);
        }
    }
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], config: &RuntimeConfig) -> Result<Vec<f32>> {
        let mut infer = || -> Result<Vec<f32>> {
            let mut infer_hop = |hop_input: &[f32]| -> Result<Vec<f32>> {
                let current_sr = config.sample_rate.max(1);
                if self.post_filter_sample_rate != current_sr {
                    // Post-filter is stateful; reset when sample-rate changes so
                    // previous-block state is never reused across different SR domains.
                    self.post_filter_prev_sample = 0.0;
                    self.post_filter_prev_valid = false;
                    self.post_filter_sample_rate = current_sr;
                }
                self.infer_block_count = self.infer_block_count.saturating_add(1);
                let infer_block_count = self.infer_block_count;
                let should_emit_timing = infer_block_count % TIMING_LOG_EVERY_BLOCKS == 1;
                let t_start = Instant::now();
                let waveform = hop_input.to_vec();
                // Keep explicit configured hop when it is smaller than window (e.g. 8000/16000).
                // Only perform runtime hop-sync from input length when configured hop is window-sized.
                let target_hop_samples_16k = resolve_runtime_hop_samples_16k(
                    self.configured_hop_samples_16k,
                    hop_input.len().max(1),
                    config.sample_rate.max(1),
                );
                if target_hop_samples_16k != self.hop_samples_16k {
                    let context_samples =
                        STRICT_ONNX_INPUT_SAMPLES_16K.saturating_sub(target_hop_samples_16k);
                    eprintln!(
                        "[vc-inference] sliding_window hop sync: hop={}smp context={}smp @ 16kHz (input_len={} sr={})",
                        target_hop_samples_16k,
                        context_samples,
                        hop_input.len(),
                        config.sample_rate
                    );
                }
                let (mut hubert_source_16k, mut rmvpe_source_16k, n_hops) = self
                    .prepare_inference_inputs_16k(
                        hop_input,
                        config.sample_rate,
                        target_hop_samples_16k,
                    )?;
                let t_resample_in_us = t_start.elapsed().as_micros();
                let frames_per_hop = STRICT_HUBERT_OUTPUT_FRAMES
                    .saturating_mul(self.hubert_upsample_factor.max(1));
                let effective_hops = n_hops.max(1);
                let total_phone_frames = effective_hops.saturating_mul(frames_per_hop).max(1);
                let expected_output_samples = total_phone_frames
                    .saturating_mul(config.sample_rate.max(1) as usize)
                    .saturating_div(100);
                eprintln!(
                    "[vc-inference] geometry: block={} process_window={} input_16k={} hops={} frames={} expected_out={}",
                    config.block_size,
                    hop_input.len(),
                    hubert_source_16k.len(),
                    effective_hops,
                    total_phone_frames,
                    expected_output_samples
                );
                eprintln!(
                    "[vc-inference] dynamic scaling: input_16k={} hops={} phone_frames={} expected_output={}",
                    hubert_source_16k.len(),
                    effective_hops,
                    total_phone_frames,
                    expected_output_samples
                );
                apply_highpass_48hz_filtfilt_inplace(&mut hubert_source_16k);
                if rmvpe_source_16k.len() == hubert_source_16k.len() {
                    rmvpe_source_16k.copy_from_slice(&hubert_source_16k);
                } else {
                    apply_highpass_48hz_filtfilt_inplace(&mut rmvpe_source_16k);
                }
                let hubert_norm_stats = normalize_for_onnx_input(&mut hubert_source_16k);
                let rmvpe_norm_stats = normalize_for_onnx_input(&mut rmvpe_source_16k);
                if tensor_stats_enabled() {
                    eprintln!(
                        "[vc-inference] stats source_16k_norm_hubert: rms_before={:.6e} peak_before={:.6e} rms_after={:.6e} peak_after={:.6e} gain={:.6e}",
                        hubert_norm_stats.rms_before,
                        hubert_norm_stats.peak_before,
                        hubert_norm_stats.rms_after,
                        hubert_norm_stats.peak_after,
                        hubert_norm_stats.gain_applied
                    );
                    eprintln!(
                        "[vc-inference] stats source_16k_norm_rmvpe: rms_before={:.6e} peak_before={:.6e} rms_after={:.6e} peak_after={:.6e} gain={:.6e}",
                        rmvpe_norm_stats.rms_before,
                        rmvpe_norm_stats.peak_before,
                        rmvpe_norm_stats.rms_after,
                        rmvpe_norm_stats.peak_after,
                        rmvpe_norm_stats.gain_applied
                    );
                }
                let hubert_source_stats = tensor_stats_from_iter(hubert_source_16k.iter().copied());
                maybe_log_tensor_stats("source_16k_hubert", hubert_source_stats);
                ensure_tensor_finite("source_16k_hubert", hubert_source_stats)?;
                let rmvpe_source_stats = tensor_stats_from_iter(rmvpe_source_16k.iter().copied());
                maybe_log_tensor_stats("source_16k_rmvpe", rmvpe_source_stats);
                ensure_tensor_finite("source_16k_rmvpe", rmvpe_source_stats)?;
                let t_preproc_us = t_start.elapsed().as_micros();
                let rmvpe_threshold = if config.rmvpe_threshold.is_finite() {
                    config.rmvpe_threshold
                } else {
                    DEFAULT_RMVPE_THRESHOLD
                };
                let requested_index_rows = config.index_search_rows.max(1);
                let requested_index_top_k = config.index_top_k.max(1).min(requested_index_rows);
                let requested_index_nprobe = config.index_nprobe.max(1);
                let timing_index_provider = match self.feature_index.as_ref() {
                    Some(IndexBackend::Ivf(_)) => "ivf_native",
                    #[cfg(feature = "faiss-native")]
                    Some(IndexBackend::Faiss { .. }) => "faiss_native",
                    Some(IndexBackend::Binary(_)) => "binary_cpu",
                    None => "none",
                };
                let post_filter_alpha = if config.post_filter_alpha.is_finite() {
                    config.post_filter_alpha.clamp(0.0, 0.999)
                } else {
                    0.96
                };
                let mut post_filter_prev_sample = self.post_filter_prev_sample;
                let mut post_filter_prev_valid = self.post_filter_prev_valid;
                let mut finalize_output = |out: &[f32]| -> Result<Vec<f32>> {
                    let mut filtered_storage = Vec::<f32>::new();
                    let out_ref: &[f32] = if post_filter_alpha > 0.0 {
                        filtered_storage.extend_from_slice(out);
                        apply_post_filter_output(
                            &mut filtered_storage,
                            post_filter_alpha,
                            &mut post_filter_prev_sample,
                            &mut post_filter_prev_valid,
                        );
                        &filtered_storage
                    } else {
                        out
                    };
                    let out_stats = tensor_stats_from_iter(out_ref.iter().copied());
                    maybe_log_tensor_stats("rvc_wave", out_stats);
                    ensure_tensor_finite("rvc_wave", out_stats)?;
                    let mixed =
                        apply_rms_mix(&waveform, out_ref, config.rms_mix_rate, config.sample_rate);
                    let mixed_stats = tensor_stats_from_iter(mixed.iter().copied());
                    maybe_log_tensor_stats("mixed_wave", mixed_stats);
                    ensure_tensor_finite("mixed_wave", mixed_stats)?;
                    let processed = postprocess_generated_audio(&mixed);
                    let post_stats = tensor_stats_from_iter(processed.iter().copied());
                    maybe_log_tensor_stats("post_wave", post_stats);
                    ensure_tensor_finite("post_wave", post_stats)?;
                    // Return decoder output as-is; vc-audio slices the latest region.
                    Ok(processed)
                };
                let zero_copy_result = if let Some(engine) = self.zero_copy_engine.as_ref() {
                    let run = engine
                        .lock()
                        .map_err(|_| {
                            VcError::Inference("zero-copy engine mutex poisoned".to_string())
                        })
                        .and_then(|mut guard| {
                            guard.run(
                                &hubert_source_16k,
                                &rmvpe_source_16k,
                                config.speaker_id,
                                rmvpe_threshold,
                            )
                        });
                    Some(run)
                } else {
                    None
                };
                if let Some(result) = zero_copy_result {
                    match result {
                        Ok(out) => {
                            let t_finalize0 = Instant::now();
                            let finalized = finalize_output(&out)?;
                            self.post_filter_prev_sample = post_filter_prev_sample;
                            self.post_filter_prev_valid = post_filter_prev_valid;
                            let finalize_us = t_finalize0.elapsed().as_micros();
                            if should_emit_timing {
                                let total_us = t_start.elapsed().as_micros();
                                let resample_in_ms = t_resample_in_us as f64 / 1000.0;
                                let preproc_ms =
                                    t_preproc_us.saturating_sub(t_resample_in_us) as f64 / 1000.0;
                                let finalize_ms = finalize_us as f64 / 1000.0;
                                let total_ms = total_us as f64 / 1000.0;
                                let msg = format!(
                                    "[vc-inference] timing detail block={}: resample_in={:.2}ms preproc={:.2}ms hubert=0.00ms index=0.00ms rows={} top_k={} nprobe={} index_provider={} rmvpe=0.00ms feature_post=0.00ms generator=0.00ms finalize={:.2}ms total={:.2}ms zero_copy=true",
                                    infer_block_count,
                                    resample_in_ms,
                                    preproc_ms,
                                    requested_index_rows,
                                    requested_index_top_k,
                                    requested_index_nprobe,
                                    timing_index_provider,
                                    finalize_ms,
                                    total_ms
                                );
                                eprintln!("{msg}");
                                emit_timing(&msg);
                            }
                            return Ok(finalized);
                        }
                        Err(err) => {
                            eprintln!(
                                "[vc-inference] warning: zero-copy path failed; disabling it for this session. cause={err}"
                            );
                            self.zero_copy_engine = None;
                        }
                    }
                }
                let t_hubert0 = Instant::now();
                let mut phone =
                    self.extract_phone_features(&hubert_source_16k, total_phone_frames)?;
                let hubert_us = t_hubert0.elapsed().as_micros();
                let t_rmvpe0 = Instant::now();
                let pitch_estimate =
                    self.estimate_pitch(&rmvpe_source_16k, rmvpe_threshold, total_phone_frames)?;
                let rmvpe_us = t_rmvpe0.elapsed().as_micros();
                let t_index0 = Instant::now();
                let mut phone_raw = None;
                if config.protect < 0.5 {
                    phone_raw = Some(phone.clone());
                }
                let index_blend_stats = self.blend_phone_with_index(
                    &mut phone,
                    config,
                    &pitch_estimate.confidence,
                    rmvpe_threshold,
                );
                let frame_count = phone.shape()[1].max(1);
                if pitch_estimate.f0_hz.len() != frame_count {
                    return Err(VcError::Inference(format!(
                        "strict frame contract mismatch: hubert_frames={} rmvpe_frames={}",
                        frame_count,
                        pitch_estimate.f0_hz.len()
                    )));
                }
                if pitch_estimate.confidence.len() != frame_count {
                    return Err(VcError::Inference(format!(
                        "strict frame contract mismatch: hubert_frames={} rmvpe_confidence_frames={}",
                        frame_count,
                        pitch_estimate.confidence.len()
                    )));
                }
                if let Some(raw) = phone_raw.take() {
                    apply_unvoiced_protect(&mut phone, &raw, &pitch_estimate.f0_hz, config.protect);
                }
                let index_us = t_index0.elapsed().as_micros();

                let t_feature_post0 = Instant::now();
                let mut pitchf = pitch_estimate.f0_hz;
                stabilize_sparse_pitch_track(&mut pitchf);
                audio_pipeline::smooth_pitch_track_gaussian_inplace(&mut pitchf, 2);
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
                let wave = Array3::from_shape_vec((1, 1, waveform.len()), waveform.clone())
                    .map_err(|e| {
                        VcError::Inference(format!("failed to shape waveform as [1,1,T]: {e}"))
                    })?;
                let feature_post_us = t_feature_post0.elapsed().as_micros();

                let t_generator0 = Instant::now();
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
                let generator_us = t_generator0.elapsed().as_micros();
                let out = if out.len() != expected_output_samples {
                    eprintln!(
                        "[vc-inference] warning: generator output length mismatch: got={} expected={} (dynamic axes may be disabled)",
                        out.len(),
                        expected_output_samples
                    );
                    fit_generated_output_len(&out, expected_output_samples)
                } else {
                    out
                };
                let t_finalize0 = Instant::now();
                let finalized = finalize_output(&out)?;
                self.post_filter_prev_sample = post_filter_prev_sample;
                self.post_filter_prev_valid = post_filter_prev_valid;
                let finalize_us = t_finalize0.elapsed().as_micros();
                if should_emit_timing {
                    let total_us = t_start.elapsed().as_micros();
                    let resample_in_ms = t_resample_in_us as f64 / 1000.0;
                    let preproc_ms = t_preproc_us.saturating_sub(t_resample_in_us) as f64 / 1000.0;
                    let hubert_ms = hubert_us as f64 / 1000.0;
                    let index_ms = index_us as f64 / 1000.0;
                    let rmvpe_ms = rmvpe_us as f64 / 1000.0;
                    let feature_post_ms = feature_post_us as f64 / 1000.0;
                    let generator_ms = generator_us as f64 / 1000.0;
                    let finalize_ms = finalize_us as f64 / 1000.0;
                    let total_ms = total_us as f64 / 1000.0;
                    let accounted_us = t_resample_in_us
                        + t_preproc_us.saturating_sub(t_resample_in_us)
                        + hubert_us
                        + index_us
                        + rmvpe_us
                        + feature_post_us
                        + generator_us
                        + finalize_us;
                    let other_ms = total_us.saturating_sub(accounted_us) as f64 / 1000.0;
                    let msg = format!(
                        "[vc-inference] timing detail block={}: resample_in={:.2}ms preproc={:.2}ms hubert={:.2}ms index={:.2}ms rows={} top_k={} nprobe={} index_provider={} rmvpe={:.2}ms feature_post={:.2}ms generator={:.2}ms finalize={:.2}ms other={:.2}ms total={:.2}ms",
                        infer_block_count,
                        resample_in_ms,
                        preproc_ms,
                        hubert_ms,
                        index_ms,
                        index_blend_stats.rows,
                        index_blend_stats.top_k,
                        index_blend_stats.nprobe,
                        index_blend_stats.provider,
                        rmvpe_ms,
                        feature_post_ms,
                        generator_ms,
                        finalize_ms,
                        other_ms,
                        total_ms
                    );
                    eprintln!("{msg}");
                    emit_timing(&msg);
                }
                Ok(finalized)
            };

            if frame.is_empty() {
                return Ok(Vec::new());
            }
            infer_hop(frame)
        };
        infer()
    }

    fn prepare_for_shutdown(&mut self) -> Result<()> {
        self.last_pitch_hz = 0.0;
        self.post_filter_prev_sample = 0.0;
        self.post_filter_prev_valid = false;
        self.drop_zero_copy_engine_explicit();
        Ok(())
    }
}

impl Drop for RvcOrtEngine {
    fn drop(&mut self) {
        self.drop_zero_copy_engine_explicit();
    }
}

fn fit_waveform_to_len(samples: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if samples.is_empty() {
        return vec![0.0; target_len];
    }
    if samples.len() >= target_len {
        return samples[samples.len() - target_len..].to_vec();
    }
    if samples.len() == 1 {
        return vec![samples[0]; target_len];
    }

    let pad = target_len - samples.len();
    let mut out = vec![0.0_f32; target_len];
    for i in 0..pad {
        let dist = pad - i;
        out[i] = samples[reflect_left_pad_index(dist, samples.len())];
    }
    out[pad..].copy_from_slice(samples);
    out
}

fn reflect_left_pad_index(dist: usize, len: usize) -> usize {
    let period = 2 * (len - 1);
    let mut m = dist % period;
    if m == 0 {
        m = period;
    }
    if m <= (len - 1) {
        m
    } else {
        period - m
    }
}

fn estimate_hop_samples_16k(input_samples: usize, input_sample_rate: u32) -> usize {
    if input_samples == 0 || input_sample_rate == 0 {
        return 1;
    }
    let hop = ((input_samples as f64) * (RMVPE_SAMPLE_RATE as f64) / (input_sample_rate as f64))
        .round() as usize;
    hop.max(1).min(STRICT_ONNX_INPUT_SAMPLES_16K)
}

fn fixed_audio_input_len(input_shape: Option<&[i64]>) -> Option<usize> {
    let shape = input_shape?;
    shape
        .iter()
        .rev()
        .copied()
        .find(|&dim| dim > 0)
        .map(|dim| dim as usize)
}

fn resolve_runtime_hop_samples_16k(
    configured_hop_samples_16k: usize,
    input_samples: usize,
    input_sample_rate: u32,
) -> usize {
    let configured = configured_hop_samples_16k
        .max(1)
        .min(STRICT_ONNX_INPUT_SAMPLES_16K);
    if configured < STRICT_ONNX_INPUT_SAMPLES_16K {
        configured
    } else {
        estimate_hop_samples_16k(input_samples, input_sample_rate)
    }
}

fn slide_context_window(context: &mut VecDeque<f32>, new_samples: &[f32], window_samples: usize) {
    if window_samples == 0 {
        context.clear();
        return;
    }
    if context.len() != window_samples {
        context.clear();
        context.resize(window_samples, 0.0);
    }
    if new_samples.len() >= window_samples {
        context.clear();
        context.extend(
            new_samples[new_samples.len() - window_samples..]
                .iter()
                .copied(),
        );
        return;
    }
    for _ in 0..new_samples.len() {
        let _ = context.pop_front();
    }
    context.extend(new_samples.iter().copied());
}

fn fit_generated_output_len(samples: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if samples.len() == target_len {
        return samples.to_vec();
    }
    if samples.len() > target_len {
        return samples[samples.len() - target_len..].to_vec();
    }
    let mut out = vec![0.0_f32; target_len];
    let copy_len = samples.len();
    if copy_len > 0 {
        out[..copy_len].copy_from_slice(samples);
    }
    out
}

fn force_frame_count_3d_tail_pad(frames: Array3<f32>, target_frames: usize) -> Array3<f32> {
    let (batch, frame_len, channels) = frames.dim();
    if frame_len == target_frames {
        return frames;
    }
    if target_frames == 0 {
        return Array3::<f32>::zeros((batch, 0, channels));
    }
    if frame_len == 0 {
        return Array3::<f32>::zeros((batch, target_frames, channels));
    }
    if frame_len == 1 || target_frames == 1 {
        let mut out = Array3::<f32>::zeros((batch, target_frames, channels));
        for t in 0..target_frames {
            out.slice_mut(s![.., t, ..])
                .assign(&frames.slice(s![.., 0, ..]));
        }
        return out;
    }

    let scale = (frame_len - 1) as f32 / (target_frames - 1) as f32;
    let mut out = Array3::<f32>::zeros((batch, target_frames, channels));
    for b in 0..batch {
        for t in 0..target_frames {
            let src = t as f32 * scale;
            let left = src.floor() as usize;
            let right = (left + 1).min(frame_len - 1);
            let frac = src - left as f32;
            let w_left = 1.0 - frac;
            for c in 0..channels {
                let a = frames[(b, left, c)];
                let bval = frames[(b, right, c)];
                out[(b, t, c)] = a * w_left + bval * frac;
            }
        }
    }
    out
}

fn upsample_phone_frames_repeat(frames: Array3<f32>, target_frames: usize) -> Array3<f32> {
    let (batch, src_frames, channels) = frames.dim();
    if target_frames == src_frames {
        return frames;
    }
    if target_frames == 0 {
        return Array3::<f32>::zeros((batch, 0, channels));
    }
    if src_frames == 0 {
        return Array3::<f32>::zeros((batch, target_frames, channels));
    }

    let mut out = Array3::<f32>::zeros((batch, target_frames, channels));
    for b in 0..batch {
        for t in 0..target_frames {
            let mut src_t = t.saturating_mul(src_frames) / target_frames;
            if src_t >= src_frames {
                src_t = src_frames - 1;
            }
            out.slice_mut(s![b, t, ..])
                .assign(&frames.slice(s![b, src_t, ..]));
        }
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

fn fallback_rvc_input_kind(rank: usize, ty: Option<TensorElementType>, name: &str) -> RvcInputKind {
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

fn extract_rvc_audio_from_runtime_shape(shape: &[i64], data: &[f32]) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let dims = shape
        .iter()
        .map(|&d| if d > 0 { d as usize } else { 0 })
        .collect::<Vec<usize>>();
    match dims.as_slice() {
        [samples] if *samples > 0 => Ok(data[..(*samples).min(data.len())].to_vec()),
        [batch, samples] if *batch > 0 && *samples > 0 => {
            let total = batch.saturating_mul(*samples);
            if data.len() < total {
                return Err(VcError::Inference(format!(
                    "rvc output tensor data is smaller than declared shape: len={} required={} shape={:?}",
                    data.len(),
                    total,
                    shape
                )));
            }
            Ok(data[..*samples].to_vec())
        }
        [batch, channels, samples] if *batch > 0 && *channels > 0 && *samples > 0 => {
            let total = batch.saturating_mul(*channels).saturating_mul(*samples);
            if data.len() < total {
                return Err(VcError::Inference(format!(
                    "rvc output tensor data is smaller than declared shape: len={} required={} shape={:?}",
                    data.len(),
                    total,
                    shape
                )));
            }
            Ok(data[..*samples].to_vec())
        }
        [batch, samples, channels] if *batch > 0 && *samples > 0 && *channels > 0 => {
            let total = batch.saturating_mul(*samples).saturating_mul(*channels);
            if data.len() < total {
                return Err(VcError::Inference(format!(
                    "rvc output tensor data is smaller than declared shape: len={} required={} shape={:?}",
                    data.len(),
                    total,
                    shape
                )));
            }
            if *channels == 1 {
                let mut out = Vec::with_capacity(*samples);
                for i in 0..*samples {
                    out.push(data[i]);
                }
                Ok(out)
            } else {
                // Fallback: flatten first batch if channel layout is non-standard.
                Ok(data[..((*samples).saturating_mul(*channels)).min(data.len())].to_vec())
            }
        }
        _ => Ok(data.to_vec()),
    }
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

fn decode_rmvpe_output(shape: &[i64], data: &[f32], threshold: f32) -> Result<PitchEstimate> {
    if data.is_empty() {
        return Ok(PitchEstimate {
            f0_hz: Vec::new(),
            confidence: Vec::new(),
        });
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
            let mut f0_hz = Vec::<f32>::with_capacity(time);
            let mut confidence = Vec::<f32>::with_capacity(time);
            for t in 0..time {
                let base = t * bins;
                let (f0, conf) = decode_rmvpe_salience_row(&data[base..base + bins], threshold);
                f0_hz.push(f0);
                confidence.push(conf);
            }
            return Ok(PitchEstimate { f0_hz, confidence });
        }
    }
    if shape.len() == 2 && shape[1] == 360 {
        let time = shape[0].max(1) as usize;
        if data.len() >= time * 360 {
            let mut f0_hz = Vec::<f32>::with_capacity(time);
            let mut confidence = Vec::<f32>::with_capacity(time);
            for t in 0..time {
                let base = t * 360;
                let (f0, conf) = decode_rmvpe_salience_row(&data[base..base + 360], threshold);
                f0_hz.push(f0);
                confidence.push(conf);
            }
            return Ok(PitchEstimate { f0_hz, confidence });
        }
    }
    // Fallback path for non-salience RMVPE outputs.
    // Use binary confidence from voiced/unvoiced state.
    let f0_hz = data.to_vec();
    let confidence = f0_hz
        .iter()
        .map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    Ok(PitchEstimate { f0_hz, confidence })
}

fn decode_rmvpe_salience_row(row: &[f32], threshold: f32) -> (f32, f32) {
    if row.is_empty() {
        return (0.0, 0.0);
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    let confidence = best_val.clamp(0.0, 1.0);
    if confidence <= threshold.clamp(0.0, 1.0) {
        return (0.0, confidence);
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
        return (0.0, confidence);
    }
    let cents = cents_num / cents_den;
    let f0 = 10.0_f32 * 2.0_f32.powf(cents / 1200.0_f32);
    (f0, confidence)
}

fn has_ivf_magic(path: &Path) -> bool {
    let Ok(raw) = fs::read(path) else {
        return false;
    };
    raw.get(..IVF_MAGIC.len()) == Some(IVF_MAGIC)
}

fn resolve_ivf_index_path(index_path: &str) -> Option<PathBuf> {
    let configured = Path::new(index_path);
    if configured.exists() && has_ivf_magic(configured) {
        return Some(configured.to_path_buf());
    }

    let parent = configured.parent().unwrap_or_else(|| Path::new("."));
    let stem = configured
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model_vectors");
    let candidate_stem = parent.join(format!("{stem}_ivf.bin"));
    if candidate_stem.exists() && has_ivf_magic(&candidate_stem) {
        return Some(candidate_stem);
    }

    let canonical = parent.join("model_vectors_ivf.bin");
    if canonical.exists() && has_ivf_magic(&canonical) {
        return Some(canonical);
    }
    None
}

fn load_index_backend(
    path: &str,
    target_dim: usize,
    bin_index_dim: usize,
    index_max_vectors: usize,
    requested_nprobe: u32,
) -> Result<IndexBackend> {
    #[cfg(feature = "faiss-native")]
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    #[cfg(feature = "faiss-native")]
    if ext == "index" {
        let search = FaissIndex::load(path, requested_nprobe as usize)?;
        if target_dim > 0 && search.dims() != target_dim {
            return Err(VcError::Config(format!(
                "faiss dim mismatch: index={} rvc_phone={} ({})",
                search.dims(),
                target_dim,
                path
            )));
        }
        let vectors = load_feature_index(path, target_dim, bin_index_dim, index_max_vectors)?;
        let nprobe = requested_nprobe
            .max(1)
            .min(search.nlist().max(1) as u32);
        let msg = format!(
            "[vc-inference] index backend: faiss_native nprobe={} total={}",
            nprobe,
            search.ntotal()
        );
        eprintln!("{msg}");
        emit_info(&msg);
        return Ok(IndexBackend::Faiss { search, vectors });
    }
    if let Some(ivf_path) = resolve_ivf_index_path(path) {
        match IvfIndex::load(&ivf_path) {
            Ok(index) => {
                if target_dim > 0 && index.dims() != target_dim {
                    return Err(VcError::Config(format!(
                        "ivf dim mismatch: index={} rvc_phone={} ({})",
                        index.dims(),
                        target_dim,
                        ivf_path.display()
                    )));
                }
                let nprobe = requested_nprobe
                    .max(1)
                    .min(index.nlist().max(1) as u32);
                let msg = format!(
                    "[vc-inference] index backend: ivf_native nlist={} nprobe={} total={}",
                    index.nlist(),
                    nprobe,
                    index.ntotal()
                );
                eprintln!("{msg}");
                emit_info(&msg);
                return Ok(IndexBackend::Ivf(index));
            }
            Err(e) => {
                let msg = format!(
                    "[vc-inference] warning: failed to load native IVF index ({}), falling back to binary/text backend: {}",
                    ivf_path.display(),
                    e
                );
                eprintln!("{msg}");
                emit_warn(&msg);
            }
        }
    }
    let binary = load_feature_index(path, target_dim, bin_index_dim, index_max_vectors)?;
    let msg = format!(
        "[vc-inference] index backend: binary rows={} dims={}",
        binary.vectors.nrows(),
        binary.vectors.ncols()
    );
    eprintln!("{msg}");
    emit_info(&msg);
    Ok(IndexBackend::Binary(binary))
}

fn load_feature_index(
    path: &str,
    target_dim: usize,
    bin_index_dim: usize,
    index_max_vectors: usize,
) -> Result<FeatureIndex> {
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

        let (rows, dims, flat) = extract_faiss_vectors_with_python(path, index_max_vectors)?;
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
        return load_feature_index_from_bin(path, target_dim, bin_index_dim);
    }

    load_feature_index_from_text(path, target_dim)
}

fn load_feature_index_from_bin(
    path: &str,
    target_dim: usize,
    bin_index_dim: usize,
) -> Result<FeatureIndex> {
    let raw = fs::read(path)
        .map_err(|e| VcError::Config(format!("failed to read binary index file ({path}): {e}")))?;
    if raw.get(..IVF_MAGIC.len()) == Some(IVF_MAGIC) {
        return Err(VcError::Config(format!(
            "native IVF index must be loaded via the IVF backend path: {path}"
        )));
    }
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
        bin_index_dim.max(1)
    };
    if dim == 0 {
        return Err(VcError::Config(
            "binary index dimension is 0; check runtime config `index_bin_dim`".to_string(),
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
    let vectors_t = vectors.t().to_owned();
    Ok(FeatureIndex { vectors, vectors_t })
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

fn extract_faiss_vectors_with_python(
    index_path: &str,
    max_vectors: usize,
) -> Result<(usize, usize, Vec<f32>)> {
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
    if k == 0 {
        return;
    }
    if best.len() < k {
        best.push(cand);
        return;
    }
    // Keep a fixed-size max-score pool without full sort.
    let mut worst_idx = 0usize;
    let mut worst_score = best[0].0;
    for (i, &(score, _)) in best.iter().enumerate().skip(1) {
        if score < worst_score {
            worst_score = score;
            worst_idx = i;
        }
    }
    if cand.0 > worst_score {
        best[worst_idx] = cand;
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
    assert_eq!(
        f0_hz.len(),
        frames,
        "strict frame contract mismatch in unvoiced protect: f0={} phone={}",
        f0_hz.len(),
        frames
    );
    for t in 0..frames {
        if f0_hz[t] >= 1.0 {
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

fn apply_highpass_48hz_filtfilt_inplace(samples: &mut [f32]) {
    if samples.len() < 3 {
        return;
    }
    let pad = samples
        .len()
        .saturating_sub(1)
        .min(HP48_FILTFILT_PADLEN_MAX)
        .max(HP48_FILTFILT_PADLEN_MIN);
    if pad == 0 || pad >= samples.len() {
        return;
    }

    let mut work = Vec::<f32>::with_capacity(samples.len() + pad * 2);
    // Reflect-pad both edges to suppress IIR boundary transients.
    for i in (1..=pad).rev() {
        work.push(samples[i]);
    }
    work.extend_from_slice(samples);
    let last = samples.len() - 1;
    for i in 1..=pad {
        work.push(samples[last - i]);
    }

    apply_sos_filter_inplace(&mut work, &HP48_BUTTER5_SOS_16K);
    work.reverse();
    apply_sos_filter_inplace(&mut work, &HP48_BUTTER5_SOS_16K);
    work.reverse();

    let start = pad;
    let end = start + samples.len();
    samples.copy_from_slice(&work[start..end]);
}

fn apply_sos_filter_inplace(samples: &mut [f32], sos: &[[f32; 6]]) {
    for sec in sos {
        let b0 = sec[0];
        let b1 = sec[1];
        let b2 = sec[2];
        let a0 = sec[3];
        let a1 = sec[4];
        let a2 = sec[5];
        let inv_a0 = if a0 != 0.0 { 1.0 / a0 } else { 1.0 };

        let mut z1 = 0.0_f32;
        let mut z2 = 0.0_f32;
        for x in samples.iter_mut() {
            let xn = *x;
            let y = (b0 * xn + z1) * inv_a0;
            z1 = b1 * xn + z2 - a1 * y;
            z2 = b2 * xn - a2 * y;
            *x = y;
        }
    }
}

fn apply_post_filter_output(
    output: &mut [f32],
    alpha: f32,
    prev_sample: &mut f32,
    prev_valid: &mut bool,
) {
    let alpha = if alpha.is_finite() {
        alpha.clamp(0.0, 0.999)
    } else {
        0.0
    };
    if alpha <= 0.0 || output.is_empty() {
        return;
    }
    // `alpha` is the current-sample weight:
    // y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    // alpha≈1.0 => near pass-through, alpha→0 => stronger smoothing.
    let mut prev = if *prev_valid { *prev_sample } else { output[0] };
    for sample in output.iter_mut() {
        let filtered = alpha * *sample + (1.0 - alpha) * prev;
        *sample = filtered;
        prev = filtered;
    }
    *prev_sample = prev;
    *prev_valid = true;
}

fn stabilize_sparse_pitch_track(pitchf: &mut [f32]) {
    if pitchf.len() < 3 {
        return;
    }

    // Remove isolated 1-frame voiced spikes that tend to produce metallic artifacts.
    let mut i = 0usize;
    while i < pitchf.len() {
        if pitchf[i] <= 0.0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < pitchf.len() && pitchf[i] > 0.0 {
            i += 1;
        }
        let end = i;
        let run_len = end - start;
        if run_len <= 1 {
            let left_unvoiced = start == 0 || pitchf[start - 1] <= 0.0;
            let right_unvoiced = end >= pitchf.len() || pitchf[end] <= 0.0;
            if left_unvoiced && right_unvoiced {
                pitchf[start] = 0.0;
            }
        }
    }

    // Interpolate tiny unvoiced holes between voiced frames to reduce crackling.
    let mut j = 0usize;
    while j < pitchf.len() {
        if pitchf[j] > 0.0 {
            j += 1;
            continue;
        }
        let gap_start = j;
        while j < pitchf.len() && pitchf[j] <= 0.0 {
            j += 1;
        }
        let gap_end = j;
        if gap_start == 0 || gap_end >= pitchf.len() {
            continue;
        }
        let gap_len = gap_end - gap_start;
        if gap_len == 0 || gap_len > 2 {
            continue;
        }
        let left = pitchf[gap_start - 1];
        let right = pitchf[gap_end];
        if left <= 0.0 || right <= 0.0 {
            continue;
        }
        let ratio = if left > right {
            left / right.max(1.0e-6)
        } else {
            right / left.max(1.0e-6)
        };
        if ratio > 2.2 {
            continue;
        }
        let denom = (gap_len + 1) as f32;
        for k in 0..gap_len {
            let t = (k + 1) as f32 / denom;
            pitchf[gap_start + k] = left * (1.0 - t) + right * t;
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

fn parse_ort_provider(raw: &str) -> OrtProvider {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => OrtProvider::Cpu,
        "cuda" => OrtProvider::Cuda,
        "directml" | "dml" => OrtProvider::DirectMl,
        "auto" | "" => OrtProvider::Auto,
        _ => OrtProvider::Auto,
    }
}

fn parse_index_provider(raw: &str) -> IndexProvider {
    match raw.trim().to_ascii_lowercase().as_str() {
        "gpu" => IndexProvider::Gpu,
        _ => IndexProvider::Cpu,
    }
}

fn resolve_ort_execution_config(runtime: &RuntimeConfig) -> OrtExecutionConfig {
    let provider = parse_ort_provider(&runtime.ort_provider);
    let device_id = runtime.ort_device_id.max(0);
    let gpu_mem_limit_mb = runtime.ort_gpu_mem_limit_mb as usize;
    let gpu_mem_limit_bytes = if runtime.cuda_ws || gpu_mem_limit_mb == 0 {
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

fn resolve_cuda_conv_algorithm(raw: &str) -> ep::cuda::ConvAlgorithmSearch {
    match raw.trim().to_ascii_lowercase().as_str() {
        "exhaustive" | "exh" => ep::cuda::ConvAlgorithmSearch::Exhaustive,
        "heuristic" | "heu" => ep::cuda::ConvAlgorithmSearch::Heuristic,
        "default" | "def" | "" => ep::cuda::ConvAlgorithmSearch::Default,
        _ => ep::cuda::ConvAlgorithmSearch::Default,
    }
}

fn resolve_ort_cuda_tuning(runtime: &RuntimeConfig) -> OrtCudaTuning {
    OrtCudaTuning {
        conv_algo: resolve_cuda_conv_algorithm(&runtime.cuda_conv_algo),
        conv_max_workspace: runtime.cuda_ws,
        conv1d_pad_to_nc1d: runtime.cuda_conv1d_pad_to_nc1d,
        tf32: runtime.cuda_tf32,
    }
}

fn build_execution_providers(
    config: &OrtExecutionConfig,
    cuda_tuning: &OrtCudaTuning,
) -> Vec<ExecutionProviderDispatch> {
    let cuda_ep = || {
        let mut cuda = ep::CUDA::default().with_device_id(config.device_id);
        if let Some(limit) = config.gpu_mem_limit_bytes {
            cuda = cuda.with_memory_limit(limit);
        }
        if ORT_CUDA_TUNE_LOGGED.set(()).is_ok() {
            eprintln!(
                "[vc-inference] cuda ep tuning: conv_algo={:?} cuda_ws={} conv1d_pad_to_nc1d={} tf32={} (from RuntimeConfig)",
                cuda_tuning.conv_algo,
                cuda_tuning.conv_max_workspace,
                cuda_tuning.conv1d_pad_to_nc1d,
                cuda_tuning.tf32
            );
        }
        cuda = cuda
            .with_conv_algorithm_search(cuda_tuning.conv_algo.clone())
            .with_conv_max_workspace(cuda_tuning.conv_max_workspace)
            .with_conv1d_pad_to_nc1d(cuda_tuning.conv1d_pad_to_nc1d)
            .with_tf32(cuda_tuning.tf32);
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

fn build_ort_session(
    path: &str,
    label: &str,
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
    execution: &OrtExecutionConfig,
    cuda_tuning: &OrtCudaTuning,
) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| VcError::Inference(format!("failed to create {label} session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| {
            VcError::Inference(format!("failed to set {label} optimization level: {e}"))
        })?;
    if intra_threads > 0 {
        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| VcError::Inference(format!("failed to set {label} intra threads: {e}")))?;
    }
    if inter_threads > 0 {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(|e| VcError::Inference(format!("failed to set {label} inter threads: {e}")))?;
    }
    builder = builder
        .with_parallel_execution(parallel_execution)
        .map_err(|e| VcError::Inference(format!("failed to set {label} execution mode: {e}")))?;
    let eps = build_execution_providers(execution, cuda_tuning);
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
    cuda_tuning: &OrtCudaTuning,
) -> Result<Session> {
    match build_ort_session(
        path,
        label,
        intra_threads,
        inter_threads,
        parallel_execution,
        execution,
        cuda_tuning,
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
                cuda_tuning,
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

fn normalize_model_stem(stem: &str) -> String {
    let mut clean = stem.to_string();
    if let Some(stripped) = clean.strip_suffix("_dynamic") {
        clean = stripped.to_string();
    }
    if let Some((prefix, suffix)) = clean.rsplit_once("_b") {
        if !prefix.is_empty() && !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            clean = prefix.to_string();
        }
    }
    clean
}

fn has_block_suffix(stem: &str) -> bool {
    stem.rsplit_once("_b")
        .map(|(prefix, suffix)| {
            !prefix.is_empty() && !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit())
        })
        .unwrap_or(false)
}

fn resolve_model_path(base_path: &Path, block_size: usize) -> PathBuf {
    let Some(stem_os) = base_path.file_stem() else {
        return base_path.to_path_buf();
    };
    let stem = stem_os.to_string_lossy();
    if has_block_suffix(&stem) {
        // Respect explicit block-specific paths configured by user.
        return base_path.to_path_buf();
    }
    let clean_stem = normalize_model_stem(&stem);
    if clean_stem.is_empty() {
        return base_path.to_path_buf();
    }
    let specific_name = match base_path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if !ext.is_empty() => format!("{clean_stem}_b{block_size}.{ext}"),
        _ => format!("{clean_stem}_b{block_size}"),
    };
    let specific = base_path.with_file_name(specific_name);
    if specific.exists() {
        specific
    } else {
        base_path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_hop_samples_16k, fit_waveform_to_len, has_block_suffix, normalize_model_stem,
        resolve_runtime_hop_samples_16k, slide_context_window, upsample_phone_frames_repeat,
        STRICT_ONNX_INPUT_SAMPLES_16K,
    };
    use ndarray::Array3;
    use std::collections::VecDeque;

    #[test]
    fn strict_input_fits_to_16k() {
        let src = vec![0.1_f32; 8000];
        let fitted = fit_waveform_to_len(&src, STRICT_ONNX_INPUT_SAMPLES_16K);
        assert_eq!(fitted.len(), STRICT_ONNX_INPUT_SAMPLES_16K);
    }

    #[test]
    fn strict_input_keeps_latest_tail_when_longer() {
        let src: Vec<f32> = (0..20_000).map(|i| i as f32).collect();
        let fitted = fit_waveform_to_len(&src, STRICT_ONNX_INPUT_SAMPLES_16K);
        assert_eq!(fitted.len(), STRICT_ONNX_INPUT_SAMPLES_16K);
        assert_eq!(fitted[0], 4_000.0);
        assert_eq!(fitted[STRICT_ONNX_INPUT_SAMPLES_16K - 1], 19_999.0);
    }

    #[test]
    fn sliding_window_advances_correctly() {
        const WINDOW: usize = STRICT_ONNX_INPUT_SAMPLES_16K;
        const HOP: usize = 2730;
        let mut buf: VecDeque<f32> = (0..WINDOW).map(|_| 0.0f32).collect();
        let new_block = vec![1.0f32; HOP];
        slide_context_window(&mut buf, &new_block, WINDOW);

        assert_eq!(buf.len(), WINDOW);
        let tail: Vec<f32> = buf.iter().skip(WINDOW - HOP).copied().collect();
        assert!(tail.iter().all(|&x| x == 1.0));
        let head: Vec<f32> = buf.iter().take(WINDOW - HOP).copied().collect();
        assert!(head.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hop_size_calculation_48k_block_8192() {
        let hop = estimate_hop_samples_16k(8192, 48_000);
        assert_eq!(hop, 2731);
        assert!((hop * 3).abs_diff(8192) <= 2);
    }

    #[test]
    fn hop_size_calculation_48k_block_16320() {
        let hop = estimate_hop_samples_16k(16_320, 48_000);
        assert_eq!(hop, 5_440);
        assert_eq!(hop * 3, 16_320);
    }

    #[test]
    fn runtime_hop_prefers_configured_subwindow_hop() {
        // Explicit configured hop (8000) must not be overridden by input_len-based sync.
        let hop = resolve_runtime_hop_samples_16k(8_000, 48_000, 48_000);
        assert_eq!(hop, 8_000);
    }

    #[test]
    fn runtime_hop_syncs_only_when_configured_hop_is_window() {
        // Window-sized configured hop permits runtime sync from input length.
        let hop = resolve_runtime_hop_samples_16k(16_000, 24_000, 48_000);
        assert_eq!(hop, 8_000);
    }

    #[test]
    fn normalize_model_stem_strips_dynamic_and_block_suffix() {
        assert_eq!(normalize_model_stem("model_dynamic"), "model");
        assert_eq!(normalize_model_stem("model_b24000"), "model");
        assert_eq!(normalize_model_stem("voice_model"), "voice_model");
    }

    #[test]
    fn block_suffix_detection() {
        assert!(has_block_suffix("model_b8000"));
        assert!(!has_block_suffix("model_dynamic"));
        assert!(!has_block_suffix("model_bx"));
    }

    #[test]
    fn rmvpe_context_independent_from_hubert() {
        const WINDOW: usize = STRICT_ONNX_INPUT_SAMPLES_16K;
        const HOP: usize = 2731;

        let mut hubert_ctx: VecDeque<f32> = (0..WINDOW).map(|_| 0.0).collect();
        let mut rmvpe_ctx: VecDeque<f32> = (0..WINDOW).map(|_| 0.0).collect();
        let block = vec![1.0_f32; HOP];

        slide_context_window(&mut hubert_ctx, &block, WINDOW);
        slide_context_window(&mut rmvpe_ctx, &block, WINDOW);

        assert_eq!(hubert_ctx.len(), WINDOW);
        assert_eq!(rmvpe_ctx.len(), WINDOW);
        let h: Vec<f32> = hubert_ctx.iter().copied().collect();
        let r: Vec<f32> = rmvpe_ctx.iter().copied().collect();
        assert_eq!(h, r);
    }

    #[test]
    fn hubert_repeat_upsample_50_to_100() {
        let mut src = Array3::<f32>::zeros((1, 50, 1));
        for t in 0..50 {
            src[(0, t, 0)] = t as f32;
        }
        let out = upsample_phone_frames_repeat(src, 100);
        assert_eq!(out.shape(), &[1, 100, 1]);
        for t in 0..50 {
            let a = out[(0, t * 2, 0)];
            let b = out[(0, t * 2 + 1, 0)];
            assert_eq!(a, t as f32);
            assert_eq!(b, t as f32);
        }
    }
}
