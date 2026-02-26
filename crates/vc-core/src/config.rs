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
    /// Index blend execution target (`cpu` or `gpu`).
    ///
    /// Current stable implementation is CPU.
    /// `gpu` is accepted for compatibility and future extensions.
    pub index_provider: String,
    pub protect: f32,
    pub rmvpe_threshold: f32,
    pub pitch_smooth_alpha: f32,
    pub rms_mix_rate: f32,
    /// Mild generator-output smoothing filter strength.
    ///
    /// Note: this is the *current-sample* weight in
    /// `y[n] = alpha*x[n] + (1-alpha)*y[n-1]` (not classic EMA naming).
    /// Range:
    /// - `0.0` disables post-filter
    /// - `0.9..0.999` keeps near pass-through with mild smoothing
    pub post_filter_alpha: f32,
    pub f0_median_filter_radius: usize,
    /// Legacy compatibility field (milliseconds).
    ///
    /// Runtime queue sizing is controlled by `target_buffer_ms`.
    /// This value is retained for backward compatibility and diagnostics.
    pub extra_inference_ms: u32,
    /// Playback/output queue target size in milliseconds.
    ///
    /// Larger values improve underrun resistance at the cost of latency.
    /// This is independent from `process_window` (model context window).
    /// Runtime applies a safety floor so the effective queue is never smaller
    /// than the model context window plus one hop.
    pub target_buffer_ms: u32,
    /// Legacy VAD threshold field.
    ///
    /// Backward compatibility alias for `vad_on_threshold`.
    /// Negative values are interpreted as dBFS (e.g. `-40.0`), and
    /// positive values are interpreted as linear amplitude.
    pub response_threshold: f32,
    /// VAD open threshold.
    ///
    /// Negative values are interpreted as dBFS (e.g. `-40.0`), and
    /// positive values are interpreted as linear amplitude.
    pub vad_on_threshold: f32,
    /// VAD close threshold.
    ///
    /// Negative values are interpreted as dBFS (e.g. `-55.0`), and
    /// positive values are interpreted as linear amplitude.
    pub vad_off_threshold: f32,
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
    /// For 3-block center-window slicing, default is one block (24_000 @ 48kHz block=24_000),
    /// i.e. start of the center block.
    /// 0 means "use engine default".
    #[serde(alias = "slice_offset_samples")]
    pub output_slice_offset_samples: usize,
    /// Bypass output slicing and emit full decoder output as-is.
    ///
    /// When true, `output_slice_offset_samples` and strict hop-length slicing
    /// are skipped in `vc-audio`.
    pub bypass_slicing: bool,
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
            index_provider: "cpu".to_string(),
            protect: 0.33,
            rmvpe_threshold: 0.01,
            pitch_smooth_alpha: 0.12,
            rms_mix_rate: 0.2,
            post_filter_alpha: 0.96,
            f0_median_filter_radius: 3,
            extra_inference_ms: 250,
            target_buffer_ms: 2_000,
            response_threshold: -40.0,
            vad_on_threshold: -40.0,
            vad_off_threshold: -55.0,
            fade_in_ms: 12,
            fade_out_ms: 120,
            sola_search_ms: 40,
            output_tail_offset_ms: 0,
            output_slice_offset_samples: 24_000,
            bypass_slicing: false,
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
