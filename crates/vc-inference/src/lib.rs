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

pub struct RvcOrtEngine {
    model: ModelConfig,
    rvc_session: Session,
    rmvpe_session: Option<Session>,
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

        Ok(Self {
            model,
            rvc_session,
            rmvpe_session,
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model.model_path
    }

    fn infer_phone_feature_dim(&self) -> usize {
        for input in self.rvc_session.inputs() {
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

    fn estimate_pitch(&mut self, padded_input: &[f32], fallback_frames: usize) -> Result<Vec<f32>> {
        let Some(session) = self.rmvpe_session.as_mut() else {
            return Ok(vec![0.0; fallback_frames]);
        };

        let inlet = session
            .inputs()
            .first()
            .ok_or_else(|| VcError::Inference("rmvpe model has no inputs".to_string()))?;
        let input_name = inlet.name().to_string();
        let rank = inlet.dtype().tensor_shape().map_or(3, |s| s.len());
        let input_tensor = tensor_from_audio_rank(rank, padded_input)?;

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
        let normalized = normalize_for_onnx_input(frame, 0.95);
        let rmvpe_pad = pad_for_rmvpe(&normalized, RVC_HOP_LENGTH);
        let fallback_frames = normalized.len().div_ceil(RVC_HOP_LENGTH).max(1);
        let f0 = self.estimate_pitch(&rmvpe_pad.padded, fallback_frames)?;

        let feature_dim = self.infer_phone_feature_dim();
        let phone = frame_features_for_rvc(&normalized, RVC_HOP_LENGTH, feature_dim);
        let frame_count = phone.shape()[1];
        let pitchf = resize_pitch_to_frames(&f0, frame_count);
        let pitch = coarse_pitch_from_f0(&pitchf);
        let wave = Array3::from_shape_vec((1, 1, normalized.len()), normalized.clone())
            .map_err(|e| VcError::Inference(format!("failed to shape waveform as [1,1,T]: {e}")))?;

        let out = self.run_rvc(
            &phone,
            &pitchf,
            &pitch,
            config.speaker_id,
            frame_count as i64,
            &wave,
        )?;
        Ok(postprocess_generated_audio(&out))
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
    match rank {
        1 => Tensor::from_array(Array1::from_vec(samples.to_vec()))
            .map_err(|e| VcError::Inference(format!("failed to create rank1 audio tensor: {e}"))),
        2 => Tensor::from_array(
            Array2::from_shape_vec((1, samples.len()), samples.to_vec()).map_err(|e| {
                VcError::Inference(format!("failed to shape rank2 audio tensor: {e}"))
            })?,
        )
        .map_err(|e| VcError::Inference(format!("failed to create rank2 audio tensor: {e}"))),
        _ => Tensor::from_array(
            Array3::from_shape_vec((1, 1, samples.len()), samples.to_vec()).map_err(|e| {
                VcError::Inference(format!("failed to shape rank3 audio tensor: {e}"))
            })?,
        )
        .map_err(|e| VcError::Inference(format!("failed to create rank3 audio tensor: {e}"))),
    }
}
