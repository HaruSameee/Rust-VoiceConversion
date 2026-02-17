use std::path::Path;

use vc_core::{InferenceEngine, ModelConfig, Result, RuntimeConfig, VcError};

use ort as _;

pub struct RvcOrtEngine {
    model: ModelConfig,
}

impl RvcOrtEngine {
    pub fn new(model: ModelConfig) -> Result<Self> {
        if !Path::new(&model.model_path).exists() {
            return Err(VcError::Config(format!(
                "model file not found: {}",
                model.model_path
            )));
        }
        Ok(Self { model })
    }

    pub fn model_path(&self) -> &str {
        &self.model.model_path
    }
}

impl InferenceEngine for RvcOrtEngine {
    fn infer_frame(&mut self, frame: &[f32], _config: &RuntimeConfig) -> Result<Vec<f32>> {
        Ok(frame.to_vec())
    }
}
