use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub index_path: Option<String>,
    pub pitch_extractor_path: Option<String>,
    pub hubert_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub input_gain: f32,
    pub output_gain: f32,
    pub input_device_name: Option<String>,
    pub output_device_name: Option<String>,
    pub pitch_shift_semitones: f32,
    pub index_rate: f32,
    pub index_smooth_alpha: f32,
    pub index_top_k: usize,
    pub index_search_rows: usize,
    pub protect: f32,
    pub rmvpe_threshold: f32,
    pub pitch_smooth_alpha: f32,
    pub rms_mix_rate: f32,
    pub f0_median_filter_radius: usize,
    pub extra_inference_ms: u32,
    pub response_threshold: f32,
    pub fade_in_ms: u32,
    pub fade_out_ms: u32,
    pub speaker_id: i64,
    pub sample_rate: u32,
    pub block_size: usize,
    pub ort_provider: String,
    pub ort_device_id: i32,
    pub ort_gpu_mem_limit_mb: u32,
    pub ort_intra_threads: usize,
    pub ort_inter_threads: usize,
    pub ort_parallel_execution: bool,
    pub hubert_context_samples_16k: usize,
    pub hubert_output_layer: i64,
    pub hubert_upsample_factor: usize,
    pub cuda_conv_algo: String,
    pub cuda_conv_max_workspace: bool,
    pub cuda_conv1d_pad_to_nc1d: bool,
    pub cuda_tf32: bool,
    pub index_bin_dim: usize,
    pub index_max_vectors: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            input_gain: 1.0,
            output_gain: 1.0,
            input_device_name: None,
            output_device_name: None,
            pitch_shift_semitones: 0.0,
            index_rate: 0.3,
            index_smooth_alpha: 0.85,
            index_top_k: 8,
            index_search_rows: 2_048,
            protect: 0.33,
            rmvpe_threshold: 0.03,
            pitch_smooth_alpha: 0.12,
            rms_mix_rate: 0.2,
            f0_median_filter_radius: 3,
            extra_inference_ms: 0,
            response_threshold: 0.0,
            fade_in_ms: 15,
            fade_out_ms: 80,
            speaker_id: 0,
            sample_rate: 48_000,
            block_size: 8_192,
            ort_provider: "auto".to_string(),
            ort_device_id: 0,
            ort_gpu_mem_limit_mb: 0,
            ort_intra_threads: default_ort_intra_threads(),
            ort_inter_threads: 1,
            ort_parallel_execution: false,
            hubert_context_samples_16k: 4_000,
            hubert_output_layer: 12,
            hubert_upsample_factor: 2,
            cuda_conv_algo: "default".to_string(),
            cuda_conv_max_workspace: false,
            cuda_conv1d_pad_to_nc1d: false,
            cuda_tf32: false,
            index_bin_dim: 768,
            index_max_vectors: 0,
        }
    }
}

fn default_ort_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(4)
        .max(1)
}
