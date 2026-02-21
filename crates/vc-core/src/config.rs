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
    /// Overlap/fade tuning hint in milliseconds.
    ///
    /// This value is used by `vc-audio` post-processing heuristics
    /// (e.g. OLA overlap shaping). It is not the playback queue depth.
    /// Queue depth is configured by `target_buffer_ms`.
    pub extra_inference_ms: u32,
    /// Playback/output queue target size in milliseconds.
    ///
    /// Larger values improve underrun resistance at the cost of latency.
    /// This is independent from `process_window` (model context window).
    /// Runtime applies a safety floor so the effective queue is never smaller
    /// than the model context window plus one hop.
    pub target_buffer_ms: u32,
    pub response_threshold: f32,
    pub fade_in_ms: u32,
    pub fade_out_ms: u32,
    /// SOLA search range in milliseconds (output sample-rate domain).
    ///
    /// 48kHz reference:
    /// - 10ms = 480 samples
    /// - 20ms = 960 samples
    /// - 40ms = 1920 samples
    pub sola_search_ms: u32,
    /// Extra right-edge safety offset (ms) when slicing decoder output tails.
    ///
    /// 0 means "disable right-edge trimming".
    pub output_tail_offset_ms: u32,
    /// Decoder output slice start offset in samples (output domain).
    ///
    /// For 48kHz/100-frame decoder outputs, a practical initial value is 31680.
    /// 0 means "use engine default".
    pub output_slice_offset_samples: usize,
    /// Enables debug WAV dump output from `vc-audio`.
    ///
    /// Toggled from runtime config/UI. When true, the engine writes
    /// synchronized input/output WAV files (`debug_input.wav`,
    /// `debug_output.wav`) on shutdown.
    pub record_dump: bool,
    pub speaker_id: i64,
    pub sample_rate: u32,
    pub block_size: usize,
    pub ort_provider: String,
    pub ort_device_id: i32,
    pub ort_gpu_mem_limit_mb: u32,
    #[serde(rename = "ort_intra_threads", alias = "intra_threads")]
    pub intra_threads: u32,
    #[serde(rename = "ort_inter_threads", alias = "inter_threads")]
    pub inter_threads: u32,
    pub ort_parallel_execution: bool,
    pub hubert_context_samples_16k: usize,
    pub hubert_output_layer: i64,
    pub hubert_upsample_factor: usize,
    pub cuda_conv_algo: String,
    /// Enables CUDA EP max-workspace conv path.
    ///
    /// When true, inference code treats GPU memory limit as unlimited (0).
    #[serde(alias = "cuda_conv_max_workspace")]
    pub cuda_ws: bool,
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
            rmvpe_threshold: 0.01,
            pitch_smooth_alpha: 0.12,
            rms_mix_rate: 0.2,
            f0_median_filter_radius: 3,
            extra_inference_ms: 250,
            target_buffer_ms: 2_000,
            response_threshold: 0.0,
            fade_in_ms: 12,
            fade_out_ms: 120,
            sola_search_ms: 10,
            output_tail_offset_ms: 0,
            output_slice_offset_samples: 31_680,
            record_dump: false,
            speaker_id: 0,
            sample_rate: 48_000,
            block_size: 8_192,
            ort_provider: "auto".to_string(),
            ort_device_id: 0,
            ort_gpu_mem_limit_mb: 0,
            intra_threads: default_intra_threads(),
            inter_threads: 1,
            ort_parallel_execution: false,
            hubert_context_samples_16k: 16_000,
            hubert_output_layer: 12,
            hubert_upsample_factor: 2,
            cuda_conv_algo: "default".to_string(),
            cuda_ws: false,
            cuda_conv1d_pad_to_nc1d: false,
            cuda_tf32: false,
            index_bin_dim: 768,
            index_max_vectors: 0,
        }
    }
}

fn default_intra_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
        .max(1)
}
