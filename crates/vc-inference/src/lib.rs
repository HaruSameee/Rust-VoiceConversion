use std::{
    any::Any,
    fs,
    io,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use ndarray::{Array1, Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    tensor::TensorElementType,
    value::Tensor,
};
use vc_core::{InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};
use vc_signal::{
    coarse_pitch_from_f0, frame_features_for_rvc, normalize_for_onnx_input, pad_for_rmvpe,
    postprocess_generated_audio, resize_pitch_to_frames, rmvpe_mel_from_audio, resample_linear_into,
    RVC_HOP_LENGTH, RMVPE_SAMPLE_RATE,
};

const DEFAULT_HUBERT_CONTEXT_SAMPLES_16K: usize = 4_000;
const DEFAULT_HUBERT_OUTPUT_LAYER: i64 = 12;
const DEFAULT_RMVPE_THRESHOLD: f32 = 0.03;
static ORT_INIT_FAILED: OnceLock<String> = OnceLock::new();

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
    hubert_context_samples_16k: usize,
    hubert_output_layer: i64,
    hubert_upsample_factor: usize,
    pitch_smooth_alpha: f32,
    last_pitch_hz: f32,
    rnd_state: u64,
}

impl RvcOrtEngine {
    pub fn new(model: ModelConfig) -> Result<Self> {
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
            let hubert_context_samples_16k = read_env_usize("RUST_VC_HUBERT_CONTEXT_16K")
                .unwrap_or(DEFAULT_HUBERT_CONTEXT_SAMPLES_16K)
                .max(1_600);
            let hubert_output_layer =
                read_env_i64("RUST_VC_HUBERT_OUTPUT_LAYER").unwrap_or(DEFAULT_HUBERT_OUTPUT_LAYER);
            let hubert_upsample_factor = read_env_usize("RUST_VC_HUBERT_UPSAMPLE_FACTOR")
                .unwrap_or(2)
                .clamp(1, 4);
            let pitch_smooth_alpha = read_env_f32("RUST_VC_PITCH_SMOOTH_ALPHA")
                .unwrap_or(0.0)
                .clamp(0.0, 0.98);
            eprintln!(
                "[vc-inference] ort intra_threads={} inter_threads={} parallel={} hubert_ctx_16k={} hubert_layer={} hubert_up={} pitch_smooth_alpha={:.2}",
                ort_intra_threads,
                ort_inter_threads,
                ort_parallel_execution,
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
            )?;

            let rmvpe_session = if let Some(path) = model.pitch_extractor_path.as_deref() {
                if !Path::new(path).exists() {
                    return Err(VcError::Config(format!(
                        "pitch extractor model file not found: {}",
                        path
                    )));
                }
                Some(
                    build_ort_session(
                        path,
                        "rmvpe",
                        ort_intra_threads,
                        ort_inter_threads,
                        ort_parallel_execution,
                    )?,
                )
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
                Some(
                    build_ort_session(
                        path,
                        "hubert",
                        ort_intra_threads,
                        ort_inter_threads,
                        ort_parallel_execution,
                    )?,
                )
            } else {
                None
            };

            let phone_feature_dim = infer_phone_feature_dim_from_session(&rvc_session);
            eprintln!("[vc-inference] rvc phone_feature_dim={phone_feature_dim}");
            let requirements = infer_rvc_requirements(&rvc_session);
            if requirements.needs_phone_features && hubert_session.is_none() {
                return Err(VcError::Config(
                    "this RVC model requires HuBERT features. set `hubert_path` (e.g. model/hubert.onnx)"
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
                hubert_context_samples_16k,
                hubert_output_layer,
                hubert_upsample_factor,
                pitch_smooth_alpha,
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
        let input_shape = inlet.dtype().tensor_shape().map_or_else(Vec::new, |s| s.to_vec());
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
        fallback_frames: usize,
    ) -> Result<Array3<f32>> {
        if let Some(session) = self.hubert_session.as_mut() {
            let mut chunk_16k = Vec::<f32>::new();
            if input_sample_rate == RMVPE_SAMPLE_RATE {
                chunk_16k.extend_from_slice(normalized);
            } else {
                resample_linear_into(
                    normalized,
                    input_sample_rate,
                    RMVPE_SAMPLE_RATE,
                    &mut chunk_16k,
                );
            }
            if chunk_16k.is_empty() {
                chunk_16k.push(0.0);
            }

            self.hubert_source_history_16k.extend(chunk_16k);
            if self.hubert_source_history_16k.len() > self.hubert_context_samples_16k {
                let drop_n =
                    self.hubert_source_history_16k.len() - self.hubert_context_samples_16k;
                self.hubert_source_history_16k.drain(0..drop_n);
            }

            let mut source = vec![0.0_f32; self.hubert_context_samples_16k];
            let hist_len = self
                .hubert_source_history_16k
                .len()
                .min(self.hubert_context_samples_16k);
            let pad_left = self.hubert_context_samples_16k - hist_len;
            source[pad_left..].copy_from_slice(
                &self.hubert_source_history_16k
                    [self.hubert_source_history_16k.len() - hist_len..],
            );

            let phone = run_hubert_session(
                session,
                &source,
                pad_left,
                self.phone_feature_dim,
                self.hubert_output_layer,
            )?;
            let tail = tail_phone_frames(&phone, fallback_frames.max(1));
            return Ok(upsample_phone_frames(&tail, self.hubert_upsample_factor));
        }

        Ok(frame_features_for_rvc(
            normalized,
            RVC_HOP_LENGTH,
            self.phone_feature_dim.max(1),
        ))
    }

    fn blend_phone_with_index(&self, phone: &mut Array3<f32>, config: &RuntimeConfig) {
        let Some(index) = &self.feature_index else {
            return;
        };
        if index.vectors.nrows() == 0 || index.vectors.ncols() == 0 {
            return;
        }
        let rate = config.index_rate.clamp(0.0, 1.0);
        if rate <= f32::EPSILON {
            return;
        }
        let top_k = config.index_top_k.clamp(1, 64);

        let frames = phone.shape()[1];
        let dims = phone.shape()[2];
        let rows = index.vectors.nrows();
        let search_rows = if config.index_search_rows == 0 {
            rows
        } else {
            config.index_search_rows.min(rows)
        };
        let stride = (rows / search_rows.max(1)).max(1);
        let mut retrieved = vec![0.0_f32; dims];

        for t in 0..frames {
            let mut best = Vec::<(f32, usize)>::with_capacity(top_k);
            let mut scanned = 0usize;
            let mut row = 0usize;
            while row < rows && scanned < search_rows {
                let mut l2 = 0.0_f32;
                for c in 0..dims {
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
                let from_index = retrieved[c] / weight_sum;
                let current = phone[(0, t, c)];
                phone[(0, t, c)] = current * (1.0 - rate) + from_index * rate;
            }
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
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], config: &RuntimeConfig) -> Result<Vec<f32>> {
        let mut infer = || -> Result<Vec<f32>> {
            let normalized = normalize_for_onnx_input(frame, 0.95);
            let rmvpe_pad = pad_for_rmvpe(&normalized, RVC_HOP_LENGTH);
            let fallback_frames = normalized.len().div_ceil(RVC_HOP_LENGTH).max(1);
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
                self.extract_phone_features(&normalized, config.sample_rate, fallback_frames)?;
            let mut phone_raw = None;
            if config.protect < 0.5 {
                phone_raw = Some(phone.clone());
            }
            self.blend_phone_with_index(&mut phone, config);
            let phone_frames = phone.shape()[1].max(1);
            let pitch_frames = f0.len().max(1);
            let frame_count = phone_frames.min(pitch_frames).max(1);
            if phone_frames != frame_count {
                phone = resize_phone_frames(&phone, frame_count);
            }
            if let Some(raw) = phone_raw.take() {
                let raw = if raw.shape()[1] == frame_count {
                    raw
                } else {
                    resize_phone_frames(&raw, frame_count)
                };
                apply_unvoiced_protect(&mut phone, &raw, &f0, config.protect);
            }

            let mut pitchf = resize_pitch_to_frames(&f0, frame_count);
            apply_pitch_shift_inplace(&mut pitchf, config.pitch_shift_semitones);
            if self.pitch_smooth_alpha > 0.0 {
                smooth_pitch_track(
                    &mut pitchf,
                    &mut self.last_pitch_hz,
                    self.pitch_smooth_alpha,
                );
            }
            let pitch = coarse_pitch_from_f0(&pitchf);
            let rnd = self.make_rnd_tensor(frame_count);
            let wave =
                Array3::from_shape_vec((1, 1, normalized.len()), normalized).map_err(|e| {
                    VcError::Inference(format!("failed to shape waveform as [1,1,T]: {e}"))
                })?;

            let out = self.run_rvc(
                &phone,
                &pitchf,
                &pitch,
                config.speaker_id,
                frame_count as i64,
                &rnd,
                &wave,
            )?;
            let processed = postprocess_generated_audio(&out);
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
        match classify_rvc_input(&lname) {
            Some(RvcInputKind::Phone) => req.needs_phone_features = true,
            Some(RvcInputKind::Pitch) | Some(RvcInputKind::PitchF) => req.needs_pitch = true,
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

fn resize_phone_frames(phone: &Array3<f32>, target_frames: usize) -> Array3<f32> {
    let frames = phone.shape()[1].max(1);
    let target = target_frames.max(1);
    let dims = phone.shape()[2];
    if target == frames {
        return phone.clone();
    }

    let mut out = Array3::<f32>::zeros((1, target, dims));
    if frames == 1 {
        for t in 0..target {
            for c in 0..dims {
                out[(0, t, c)] = phone[(0, 0, c)];
            }
        }
        return out;
    }

    let scale = (frames - 1) as f64 / (target - 1).max(1) as f64;
    for t in 0..target {
        let src = t as f64 * scale;
        let left = src.floor() as usize;
        let right = (left + 1).min(frames - 1);
        let frac = (src - left as f64) as f32;
        for c in 0..dims {
            let a = phone[(0, left, c)];
            let b = phone[(0, right, c)];
            out[(0, t, c)] = a * (1.0 - frac) + b * frac;
        }
    }
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
                SessionInputValue::from(Tensor::from_array(Array1::from_vec(mask.clone())).map_err(
                    |e| VcError::Inference(format!("failed to create hubert padding mask rank1: {e}")),
                )?)
            } else {
                SessionInputValue::from(
                    Tensor::from_array(Array2::from_shape_vec((1, source_len), mask).map_err(|e| {
                        VcError::Inference(format!(
                            "failed to shape hubert padding mask as [1,T]: {e}"
                        ))
                    })?).map_err(|e| {
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
            SessionInputValue::from(
                Tensor::from_array(Array1::from_vec(vec![scalar])).map_err(|e| {
                    VcError::Inference(format!(
                        "failed to create hubert int64 side-input tensor: {e}"
                    ))
                })?,
            )
        } else {
            return Err(VcError::Inference(format!(
                "unsupported hubert input '{}' type={ty:?} rank={rank}",
                name
            )));
        };
        inputs.push((name, value));
    }

    let outputs = session
        .run(inputs)
        .map_err(|e| {
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
        let bins = if shape[2] > 0 {
            shape[2] as usize
        } else {
            360
        };
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
                    eprintln!(
                        "[vc-inference] index cache read failed (will regenerate): {e}"
                    );
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

    load_feature_index_from_text(path, target_dim)
}

fn load_feature_index_from_text(path: &str, target_dim: usize) -> Result<FeatureIndex> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        VcError::Config(format!(
            "failed to read index file as text ({path}): {e}"
        ))
    })?;
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
    let vectors = Array2::from_shape_vec((rows, dims), flat).map_err(|e| {
        VcError::Config(format!("failed to build index matrix from {source}: {e}"))
    })?;
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
    feature_index_from_flat(rows, dims, flat, &cache_path.display().to_string(), target_dim)
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
    let dump_path =
        std::env::temp_dir().join(format!("rust_vc_index_dump_{ts}_{}.f32", std::process::id()));
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
    let a = alpha.clamp(0.0, 0.98);
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

fn build_ort_session(
    path: &str,
    label: &str,
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
) -> Result<Session> {
    Session::builder()
        .map_err(|e| VcError::Inference(format!("failed to create {label} session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VcError::Inference(format!("failed to set {label} optimization level: {e}")))?
        .with_intra_threads(intra_threads)
        .map_err(|e| VcError::Inference(format!("failed to set {label} intra threads: {e}")))?
        .with_inter_threads(inter_threads)
        .map_err(|e| VcError::Inference(format!("failed to set {label} inter threads: {e}")))?
        .with_parallel_execution(parallel_execution)
        .map_err(|e| VcError::Inference(format!("failed to set {label} execution mode: {e}")))?
        .commit_from_file(path)
        .map_err(|e| VcError::Inference(format!("failed to load {label} onnx model {path}: {e}")))
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
