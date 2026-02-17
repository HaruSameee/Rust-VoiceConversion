use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use ort::{
    session::{Session, SessionInputValue},
    tensor::TensorElementType,
    value::Tensor,
};
use vc_core::{InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};
use vc_signal::{
    coarse_pitch_from_f0, frame_features_for_rvc, normalize_for_onnx_input, pad_for_rmvpe,
    postprocess_generated_audio, resize_pitch_to_frames, RVC_HOP_LENGTH,
};

const INDEX_TOP_K: usize = 8;

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
}

impl RvcOrtEngine {
    pub fn new(model: ModelConfig) -> Result<Self> {
        if !Path::new(&model.model_path).exists() {
            return Err(VcError::Config(format!(
                "model file not found: {}",
                model.model_path
            )));
        }

        ort::init().commit();

        let rvc_session = Session::builder()
            .map_err(|e| VcError::Inference(format!("failed to create rvc session builder: {e}")))?
            .commit_from_file(&model.model_path)
            .map_err(|e| {
                VcError::Inference(format!(
                    "failed to load rvc onnx model {}: {e}",
                    model.model_path
                ))
            })?;

        let rmvpe_session = if let Some(path) = model.pitch_extractor_path.as_deref() {
            if !Path::new(path).exists() {
                return Err(VcError::Config(format!(
                    "pitch extractor model file not found: {}",
                    path
                )));
            }
            Some(
                Session::builder()
                    .map_err(|e| {
                        VcError::Inference(format!("failed to create rmvpe session builder: {e}"))
                    })?
                    .commit_from_file(path)
                    .map_err(|e| {
                        VcError::Inference(format!("failed to load rmvpe onnx model {path}: {e}"))
                    })?,
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
                Session::builder()
                    .map_err(|e| {
                        VcError::Inference(format!("failed to create hubert session builder: {e}"))
                    })?
                    .commit_from_file(path)
                    .map_err(|e| {
                        VcError::Inference(format!("failed to load hubert onnx model {path}: {e}"))
                    })?,
            )
        } else {
            None
        };

        let phone_feature_dim = infer_phone_feature_dim_from_session(&rvc_session);
        if let Some(session) = hubert_session.as_ref() {
            if let Some(hubert_dim) = infer_hubert_output_dim_from_session(session) {
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
                    return Err(VcError::Config(format!("index file not found: {path}")));
                }
                match load_feature_index(path, phone_feature_dim) {
                    Ok(index) => Some(index),
                    Err(e) => {
                        eprintln!("index disabled: {e}");
                        None
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
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model.model_path
    }

    fn estimate_pitch(
        &mut self,
        padded_input: Vec<f32>,
        fallback_frames: usize,
    ) -> Result<Vec<f32>> {
        let Some(session) = self.rmvpe_session.as_mut() else {
            return Ok(vec![0.0; fallback_frames]);
        };

        let inlet = session
            .inputs()
            .first()
            .ok_or_else(|| VcError::Inference("rmvpe model has no inputs".to_string()))?;
        let input_name = inlet.name().to_string();
        let rank = inlet.dtype().tensor_shape().map_or(3, |s| s.len());
        let input_tensor = tensor_from_audio_rank_owned(rank, padded_input)?;

        let outputs = session
            .run(vec![(input_name, SessionInputValue::from(input_tensor))])
            .map_err(|e| VcError::Inference(format!("rmvpe inference failed: {e}")))?;
        if outputs.len() == 0 {
            return Err(VcError::Inference(
                "rmvpe model returned no outputs".to_string(),
            ));
        }

        let (_, f0) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            VcError::Inference(format!("failed to extract rmvpe output tensor<f32>: {e}"))
        })?;
        Ok(f0.to_vec())
    }

    fn extract_phone_features(
        &mut self,
        normalized: &[f32],
        _fallback_frames: usize,
    ) -> Result<Array3<f32>> {
        if let Some(session) = self.hubert_session.as_mut() {
            let inlet = session
                .inputs()
                .first()
                .ok_or_else(|| VcError::Inference("hubert model has no inputs".to_string()))?;
            let input_name = inlet.name().to_string();
            let rank = inlet.dtype().tensor_shape().map_or(2, |s| s.len());
            let input_tensor = tensor_from_audio_rank(rank, normalized)?;
            let outputs = session
                .run(vec![(input_name, SessionInputValue::from(input_tensor))])
                .map_err(|e| VcError::Inference(format!("hubert inference failed: {e}")))?;
            if outputs.len() == 0 {
                return Err(VcError::Inference(
                    "hubert model returned no outputs".to_string(),
                ));
            }
            let (shape, hubert) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                VcError::Inference(format!("failed to extract hubert output tensor<f32>: {e}"))
            })?;
            return phone_from_hubert_tensor(shape, hubert, self.phone_feature_dim);
        }

        Ok(frame_features_for_rvc(
            normalized,
            RVC_HOP_LENGTH,
            self.phone_feature_dim.max(1),
        ))
    }

    fn blend_phone_with_index(&self, phone: &mut Array3<f32>, index_rate: f32) {
        let Some(index) = &self.feature_index else {
            return;
        };
        if index.vectors.nrows() == 0 || index.vectors.ncols() == 0 {
            return;
        }
        let rate = index_rate.clamp(0.0, 1.0);
        if rate <= f32::EPSILON {
            return;
        }

        let frames = phone.shape()[1];
        let dims = phone.shape()[2];
        let rows = index.vectors.nrows();
        let mut retrieved = vec![0.0_f32; dims];

        for t in 0..frames {
            let mut best = Vec::<(f32, usize)>::with_capacity(INDEX_TOP_K);
            for row in 0..rows {
                let mut l2 = 0.0_f32;
                for c in 0..dims {
                    let diff = phone[(0, t, c)] - index.vectors[(row, c)];
                    l2 += diff * diff;
                }
                push_top_k(&mut best, (l2, row), INDEX_TOP_K);
            }
            if best.is_empty() {
                continue;
            }

            retrieved.fill(0.0);
            let mut weight_sum = 0.0_f32;
            for (dist, row) in best {
                let w = 1.0 / (dist.sqrt() + 1e-6);
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
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], config: &RuntimeConfig) -> Result<Vec<f32>> {
        let mut infer = || -> Result<Vec<f32>> {
            let normalized = normalize_for_onnx_input(frame, 0.95);
            let rmvpe_pad = pad_for_rmvpe(&normalized, RVC_HOP_LENGTH);
            let fallback_frames = normalized.len().div_ceil(RVC_HOP_LENGTH).max(1);
            let f0 = self.estimate_pitch(rmvpe_pad.padded, fallback_frames)?;

            let mut phone = self.extract_phone_features(&normalized, fallback_frames)?;
            self.blend_phone_with_index(&mut phone, config.index_rate);
            let frame_count = phone.shape()[1].max(1);
            let mut pitchf = resize_pitch_to_frames(&f0, frame_count);
            apply_pitch_shift_inplace(&mut pitchf, config.pitch_shift_semitones);
            let pitch = coarse_pitch_from_f0(&pitchf);
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
                &wave,
            )?;
            Ok(postprocess_generated_audio(&out))
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

#[derive(Debug, Clone, Copy)]
enum RvcInputKind {
    Phone,
    Pitch,
    PitchF,
    Sid,
    Length,
    Wave,
}

fn classify_rvc_input(name: &str) -> Option<RvcInputKind> {
    if name.contains("phone") || name.contains("hubert") || name.contains("feat") {
        return Some(RvcInputKind::Phone);
    }
    if name.contains("pitchf") || name.contains("nsff0") || name.contains("f0") {
        return Some(RvcInputKind::PitchF);
    }
    if name.contains("pitch") {
        return Some(RvcInputKind::Pitch);
    }
    if name.contains("sid") || name.contains("spk") || name.contains("speaker") {
        return Some(RvcInputKind::Sid);
    }
    if name.contains("len") || name.contains("p_len") {
        return Some(RvcInputKind::Length);
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

fn load_feature_index(path: &str, target_dim: usize) -> Result<FeatureIndex> {
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
    let vectors = Array2::from_shape_vec((flat.len() / dims, dims), flat)
        .map_err(|e| VcError::Config(format!("failed to build index matrix from {path}: {e}")))?;
    if target_dim > 0 && vectors.ncols() != target_dim {
        return Err(VcError::Config(format!(
            "index dim mismatch: index={}, rvc_phone={target_dim}",
            vectors.ncols()
        )));
    }
    Ok(FeatureIndex { vectors })
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
