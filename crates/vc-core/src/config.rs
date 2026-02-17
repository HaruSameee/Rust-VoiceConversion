use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub index_path: Option<String>,
    pub pitch_extractor_path: Option<String>,
    pub hubert_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub input_gain: f32,
    pub output_gain: f32,
    pub pitch_shift_semitones: f32,
    pub index_rate: f32,
    pub speaker_id: i64,
    pub sample_rate: u32,
    pub block_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            input_gain: 1.0,
            output_gain: 1.0,
            pitch_shift_semitones: 0.0,
            index_rate: 0.5,
            speaker_id: 0,
            sample_rate: 48_000,
            block_size: 480,
        }
    }
}
