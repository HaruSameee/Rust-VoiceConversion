use std::path::Path;

use ndarray::Array3;
use ort::{session::Session, value::TensorRef};
use vc_core::{InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};

/// RVC用のONNX推論エンジン。
///
/// `Session` を内部に保持し、`infer_frame` で
/// `frame -> tensor -> ONNX実行 -> 出力ベクタ` の流れを担います。
pub struct RvcOrtEngine {
    model: ModelConfig,
    session: Session,
}

impl RvcOrtEngine {
    /// ONNXモデルをロードして推論器を作る。
    pub fn new(model: ModelConfig) -> Result<Self> {
        if !Path::new(&model.model_path).exists() {
            return Err(VcError::Config(format!(
                "model file not found: {}",
                model.model_path
            )));
        }

        // 明示初期化しておくと、ロード時の失敗原因が追いやすい。
        // すでに初期化済みでも安全に呼べます。
        ort::init().commit();

        let session = Session::builder()
            .map_err(|e| VcError::Inference(format!("failed to create ort session builder: {e}")))?
            .commit_from_file(&model.model_path)
            .map_err(|e| {
                VcError::Inference(format!(
                    "failed to load onnx model {}: {e}",
                    model.model_path
                ))
            })?;

        Ok(Self { model, session })
    }

    pub fn model_path(&self) -> &str {
        &self.model.model_path
    }
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], _config: &RuntimeConfig) -> Result<Vec<f32>> {
        // いまのプロトタイプでは [1, 1, length] を入力形状として扱う。
        let input = Array3::from_shape_vec((1, 1, frame.len()), frame.to_vec()).map_err(|e| {
            VcError::Inference(format!("failed to shape input frame as [1,1,length]: {e}"))
        })?;

        let input_tensor = TensorRef::from_array_view(input.view())
            .map_err(|e| VcError::Inference(format!("failed to create input tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .map_err(|e| VcError::Inference(format!("onnx inference failed: {e}")))?;

        if outputs.len() == 0 {
            return Err(VcError::Inference(
                "onnx model returned no outputs".to_string(),
            ));
        }

        // 先頭出力を音声波形として取り出す想定。
        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            VcError::Inference(format!("failed to extract output tensor<f32>: {e}"))
        })?;

        Ok(data.to_vec())
    }
}
