use std::{
    env, fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use ndarray::{Array3, ArrayView2};
use ort::{
    ep::{self, ArbitrarilyConfigurableExecutionProvider, ExecutionProviderDispatch},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{DynTensorValueType, Tensor},
};
use vc_core::{ModelConfig, Result, RuntimeConfig, VcError};
use vc_signal::coarse_pitch_from_f0;

const DEFAULT_INPUT_SAMPLES: usize = 16_000;
const HUBERT_BASE_FRAMES: usize = 50;
const DEFAULT_WARMUP_RUNS: usize = 2;
const RND_CHANNELS: usize = 192;
const RMVPE_EXPECTED_FRAMES: usize = 101;
const RMVPE_SALIENCE_BINS: usize = 360;
const RMVPE_CENTS_BASE: f32 = 1997.3794;
const RMVPE_CENTS_STEP: f32 = 20.0;
const RMVPE_DECODE_RADIUS: usize = 4;
const DEFAULT_RMS_GATE_THRESHOLD: f32 = 0.0032;

#[derive(Debug, Clone, Default)]
pub struct ZeroCopyIoOverrides {
    pub hubert_input: Option<String>,
    pub hubert_output: Option<String>,
    pub rmvpe_input: Option<String>,
    pub rmvpe_output: Option<String>,
    pub decoder_phone_input: Option<String>,
    pub decoder_phone_lengths_input: Option<String>,
    pub decoder_pitch_input: Option<String>,
    pub decoder_pitchf_input: Option<String>,
    pub decoder_ds_input: Option<String>,
    pub decoder_rnd_input: Option<String>,
    pub decoder_output: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ZeroCopyIoNames {
    pub hubert_input: String,
    pub hubert_output: String,
    pub rmvpe_input: String,
    pub rmvpe_output: String,
    pub decoder_phone_input: String,
    pub decoder_phone_lengths_input: String,
    pub decoder_pitch_input: String,
    pub decoder_pitchf_input: String,
    pub decoder_ds_input: String,
    pub decoder_rnd_input: String,
    pub decoder_output: String,
}

#[derive(Debug, Clone)]
pub struct ZeroCopyEngineConfig {
    pub hubert_model: PathBuf,
    pub rmvpe_model: PathBuf,
    pub decoder_model: PathBuf,
    pub cache_dir: PathBuf,
    pub device_id: i32,
    pub input_samples: usize,
    pub frame_count: usize,
    pub warmup_runs: usize,
    pub rmvpe_threshold: f32,
    pub rms_gate_threshold: f32,
    pub io_overrides: ZeroCopyIoOverrides,
}

pub struct ZeroCopyInferenceEngine {
    io: ZeroCopyIoNames,
    input_samples: usize,
    hubert_input_samples: usize,
    frame_count: usize,
    phone_feature_dim: usize,
    rmvpe_threshold: f32,
    rms_gate_threshold: f32,
    host_waveform: Tensor<f32>,
    hubert_host_waveform: Tensor<f32>,
    hubert_waveform_gpu: Tensor<f32>,
    rmvpe_waveform_gpu: Tensor<f32>,
    features_workspace: Array3<f32>,
    features_buf_host: Tensor<f32>,
    features_buf: Tensor<f32>,
    decoder_phone_lengths_host: Tensor<i64>,
    decoder_phone_lengths_gpu: Tensor<i64>,
    decoder_pitch_host: Tensor<i64>,
    decoder_pitch_gpu: Tensor<i64>,
    decoder_pitchf_host: Tensor<f32>,
    decoder_pitchf_gpu: Tensor<f32>,
    decoder_ds_host: Tensor<i64>,
    decoder_ds_gpu: Tensor<i64>,
    decoder_rnd_host: Tensor<f32>,
    decoder_rnd_gpu: Tensor<f32>,
    rnd_state: u64,
    block_count: usize,
    audit_enabled: bool,
    hubert_frame_mismatch_warned: bool,
    drop_synced: bool,
    // Keep sessions last so tensor resources are dropped first.
    hubert: Session,
    rmvpe: Session,
    decoder: Session,
}

// SAFETY: The engine is always accessed under a mutex in `RvcOrtEngine`.
unsafe impl Send for ZeroCopyInferenceEngine {}

impl ZeroCopyInferenceEngine {
    pub fn new(config: ZeroCopyEngineConfig) -> Result<Self> {
        let eps = build_cuda_execution_providers(config.device_id);
        let hubert =
            build_session_with_cache(&config.hubert_model, &config.cache_dir, "hubert", &eps)?;
        let rmvpe =
            build_session_with_cache(&config.rmvpe_model, &config.cache_dir, "rmvpe", &eps)?;
        let decoder =
            build_session_with_cache(&config.decoder_model, &config.cache_dir, "decoder", &eps)?;
        let io = resolve_io_names(&hubert, &rmvpe, &decoder, &config.io_overrides)?;
        let hubert_input_samples =
            resolve_hubert_input_samples(&hubert, &io.hubert_input, config.input_samples);
        let phone_feature_dim =
            infer_decoder_phone_feature_dim(&decoder, &io.decoder_phone_input).max(1);

        let cpu_alloc = Allocator::default();
        let host_waveform = map_ort(
            "failed to allocate host waveform tensor",
            Tensor::<f32>::new(&cpu_alloc, [1_usize, config.input_samples]),
        )?;
        let hubert_host_waveform = map_ort(
            "failed to allocate HuBERT host waveform tensor",
            Tensor::<f32>::new(&cpu_alloc, [1_usize, hubert_input_samples]),
        )?;

        let hubert_cuda_alloc = make_cuda_allocator(&hubert, config.device_id)?;
        let rmvpe_cuda_alloc = make_cuda_allocator(&rmvpe, config.device_id)?;
        let decoder_cuda_alloc = make_cuda_allocator(&decoder, config.device_id)?;
        let hubert_waveform_gpu = map_ort(
            "failed to allocate HuBERT CUDA input tensor",
            Tensor::<f32>::new(&hubert_cuda_alloc, [1_usize, hubert_input_samples]),
        )?;
        let rmvpe_waveform_gpu = map_ort(
            "failed to allocate RMVPE CUDA input tensor",
            Tensor::<f32>::new(&rmvpe_cuda_alloc, [1_usize, config.input_samples]),
        )?;
        let features_workspace = Array3::<f32>::zeros((1, config.frame_count, phone_feature_dim));
        let features_buf_host = map_ort(
            "failed to allocate features host tensor",
            Tensor::<f32>::new(&cpu_alloc, [1_usize, config.frame_count, phone_feature_dim]),
        )?;
        let features_buf = map_ort(
            "failed to allocate features CUDA tensor",
            Tensor::<f32>::new(
                &decoder_cuda_alloc,
                [1_usize, config.frame_count, phone_feature_dim],
            ),
        )?;

        let decoder_phone_lengths_host = map_ort(
            "failed to allocate decoder phone_lengths host tensor",
            Tensor::<i64>::new(&cpu_alloc, [1_usize]),
        )?;
        let decoder_phone_lengths_gpu = map_ort(
            "failed to allocate decoder phone_lengths CUDA tensor",
            Tensor::<i64>::new(&decoder_cuda_alloc, [1_usize]),
        )?;
        let decoder_pitch_host = map_ort(
            "failed to allocate decoder pitch host tensor",
            Tensor::<i64>::new(&cpu_alloc, [1_usize, config.frame_count]),
        )?;
        let decoder_pitch_gpu = map_ort(
            "failed to allocate decoder pitch CUDA tensor",
            Tensor::<i64>::new(&decoder_cuda_alloc, [1_usize, config.frame_count]),
        )?;
        let decoder_pitchf_host = map_ort(
            "failed to allocate decoder pitchf host tensor",
            Tensor::<f32>::new(&cpu_alloc, [1_usize, config.frame_count]),
        )?;
        let decoder_pitchf_gpu = map_ort(
            "failed to allocate decoder pitchf CUDA tensor",
            Tensor::<f32>::new(&decoder_cuda_alloc, [1_usize, config.frame_count]),
        )?;
        let decoder_ds_host = map_ort(
            "failed to allocate decoder ds host tensor",
            Tensor::<i64>::new(&cpu_alloc, [1_usize]),
        )?;
        let decoder_ds_gpu = map_ort(
            "failed to allocate decoder ds CUDA tensor",
            Tensor::<i64>::new(&decoder_cuda_alloc, [1_usize]),
        )?;
        let decoder_rnd_host = map_ort(
            "failed to allocate decoder rnd host tensor",
            Tensor::<f32>::new(&cpu_alloc, [1_usize, RND_CHANNELS, config.frame_count]),
        )?;
        let decoder_rnd_gpu = map_ort(
            "failed to allocate decoder rnd CUDA tensor",
            Tensor::<f32>::new(
                &decoder_cuda_alloc,
                [1_usize, RND_CHANNELS, config.frame_count],
            ),
        )?;

        let mut engine = Self {
            io,
            input_samples: config.input_samples,
            hubert_input_samples,
            frame_count: config.frame_count,
            phone_feature_dim,
            rmvpe_threshold: config.rmvpe_threshold,
            rms_gate_threshold: resolve_rms_gate_threshold(config.rms_gate_threshold),
            host_waveform,
            hubert_host_waveform,
            hubert_waveform_gpu,
            rmvpe_waveform_gpu,
            features_workspace,
            features_buf_host,
            features_buf,
            decoder_phone_lengths_host,
            decoder_phone_lengths_gpu,
            decoder_pitch_host,
            decoder_pitch_gpu,
            decoder_pitchf_host,
            decoder_pitchf_gpu,
            decoder_ds_host,
            decoder_ds_gpu,
            decoder_rnd_host,
            decoder_rnd_gpu,
            rnd_state: 0x9E37_79B9_7F4A_7C15,
            block_count: 0,
            audit_enabled: false,
            hubert_frame_mismatch_warned: false,
            drop_synced: false,
            hubert,
            rmvpe,
            decoder,
        };

        {
            let (_, lengths) = engine.decoder_phone_lengths_host.extract_tensor_mut();
            lengths[0] = engine.frame_count as i64;
        }
        map_ort(
            "failed to copy decoder phone_lengths host->CUDA",
            engine
                .decoder_phone_lengths_host
                .copy_into(&mut engine.decoder_phone_lengths_gpu),
        )?;

        let warmup_runs = config.warmup_runs.max(DEFAULT_WARMUP_RUNS);
        let warmup_input = vec![0.0_f32; engine.input_samples];
        for _ in 0..warmup_runs {
            let _ = engine.run(&warmup_input, &warmup_input, 0, config.rmvpe_threshold)?;
        }
        engine.block_count = 0;
        engine.audit_enabled = true;
        Ok(engine)
    }

    pub fn run(
        &mut self,
        hubert_waveform_16k: &[f32],
        rmvpe_waveform_16k: &[f32],
        speaker_id: i64,
        rmvpe_threshold: f32,
    ) -> Result<Vec<f32>> {
        if hubert_waveform_16k.len() != self.hubert_input_samples {
            return Err(VcError::Inference(format!(
                "HuBERT input length mismatch: got={}, expected={}",
                hubert_waveform_16k.len(),
                self.hubert_input_samples
            )));
        }
        if rmvpe_waveform_16k.len() != self.input_samples {
            return Err(VcError::Inference(format!(
                "RMVPE input length mismatch: got={}, expected={}",
                rmvpe_waveform_16k.len(),
                self.input_samples
            )));
        }
        {
            let (_, host) = self.host_waveform.extract_tensor_mut();
            host.copy_from_slice(rmvpe_waveform_16k);
        }
        {
            let (_, hubert_host) = self.hubert_host_waveform.extract_tensor_mut();
            hubert_host.copy_from_slice(hubert_waveform_16k);
        }
        map_ort(
            "failed to copy waveform host->HuBERT CUDA tensor",
            self.hubert_host_waveform
                .copy_into(&mut self.hubert_waveform_gpu),
        )?;
        map_ort(
            "failed to copy waveform host->RMVPE CUDA tensor",
            self.host_waveform.copy_into(&mut self.rmvpe_waveform_gpu),
        )?;

        let hubert_output_mem = make_cpu_output_memory_info()?;
        let mut hubert_binding = map_ort(
            "failed to create HuBERT IoBinding",
            self.hubert.create_binding(),
        )?;
        map_ort(
            "failed to bind HuBERT input",
            hubert_binding.bind_input(&self.io.hubert_input, &self.hubert_waveform_gpu),
        )?;
        map_ort(
            "failed to bind HuBERT output",
            hubert_binding.bind_output_to_device(&self.io.hubert_output, &hubert_output_mem),
        )?;
        let hubert_features = {
            let mut hubert_outputs = map_ort(
                "HuBERT run_binding failed",
                self.hubert.run_binding(&hubert_binding),
            )?;
            hubert_outputs
                .remove(&self.io.hubert_output)
                .ok_or_else(|| {
                    VcError::Inference(format!(
                        "HuBERT output '{}' was not found",
                        self.io.hubert_output
                    ))
                })?
        };
        let hubert_features = map_ort(
            "HuBERT output is not a tensor",
            hubert_features.downcast::<DynTensorValueType>(),
        )?;
        let (hubert_shape, hubert_data) = map_ort(
            "failed to extract HuBERT output tensor<f32>",
            hubert_features.try_extract_tensor::<f32>(),
        )?;
        if hubert_shape.len() != 3 {
            return Err(VcError::Inference(format!(
                "unexpected HuBERT output rank on zero-copy path: shape={:?}",
                hubert_shape
            )));
        }
        let batch = hubert_shape[0].max(0) as usize;
        let src_frames = hubert_shape[1].max(0) as usize;
        let src_dim = hubert_shape[2].max(0) as usize;
        if batch != 1 {
            return Err(VcError::Inference(format!(
                "unexpected HuBERT batch on zero-copy path: got={} expected=1",
                batch
            )));
        }
        if src_frames == 0 {
            return Err(VcError::Inference(
                "HuBERT returned zero frames on zero-copy path".to_string(),
            ));
        }
        if src_frames != self.frame_count && !self.hubert_frame_mismatch_warned {
            self.hubert_frame_mismatch_warned = true;
            eprintln!(
                "[vc-inference] warning: HuBERT returned {} frames; host will resample to decoder_frames={}",
                src_frames, self.frame_count
            );
        }
        let copy_dims = src_dim.min(self.phone_feature_dim);
        {
            self.features_workspace.fill(0.0);
            if src_frames > 0 && copy_dims > 0 {
                let expected = src_frames.saturating_mul(src_dim).max(1);
                if hubert_data.len() < expected {
                    return Err(VcError::Inference(format!(
                        "HuBERT output buffer too small: len={} required={}",
                        hubert_data.len(),
                        expected
                    )));
                }
                for t in 0..self.frame_count {
                    let mut src_t = t.saturating_mul(src_frames) / self.frame_count.max(1);
                    if src_t >= src_frames {
                        src_t = src_frames - 1;
                    }
                    let src_base = src_t.saturating_mul(src_dim);
                    for c in 0..copy_dims {
                        self.features_workspace[(0, t, c)] = hubert_data[src_base + c];
                    }
                }
            }
        }
        self.update_features_buf_from_workspace()?;

        let rmvpe_output_mem = make_cpu_output_memory_info()?;
        let mut rmvpe_binding = map_ort(
            "failed to create RMVPE IoBinding",
            self.rmvpe.create_binding(),
        )?;
        map_ort(
            "failed to bind RMVPE input",
            rmvpe_binding.bind_input(&self.io.rmvpe_input, &self.rmvpe_waveform_gpu),
        )?;
        map_ort(
            "failed to bind RMVPE output",
            rmvpe_binding.bind_output_to_device(&self.io.rmvpe_output, &rmvpe_output_mem),
        )?;
        let rmvpe_salience = {
            let mut rmvpe_outputs = map_ort(
                "RMVPE run_binding failed",
                self.rmvpe.run_binding(&rmvpe_binding),
            )?;
            rmvpe_outputs.remove(&self.io.rmvpe_output).ok_or_else(|| {
                VcError::Inference(format!(
                    "RMVPE output '{}' was not found",
                    self.io.rmvpe_output
                ))
            })?
        };
        let rmvpe_salience = map_ort(
            "RMVPE output is not a tensor",
            rmvpe_salience.downcast::<DynTensorValueType>(),
        )?;
        let (rmvpe_shape, rmvpe_data) = map_ort(
            "failed to extract RMVPE output tensor<f32>",
            rmvpe_salience.try_extract_tensor::<f32>(),
        )?;

        let threshold = if rmvpe_threshold.is_finite() {
            rmvpe_threshold
        } else {
            self.rmvpe_threshold
        };
        let rmvpe_shape_vec = rmvpe_shape.to_vec();
        let rmvpe_shape = rmvpe_shape_vec.as_slice();
        let (rmvpe_frames, rmvpe_bins, rmvpe_required) = match rmvpe_shape {
            [batch, frames, bins] => {
                let batch = (*batch).max(0) as usize;
                let frames = (*frames).max(0) as usize;
                let bins = (*bins).max(0) as usize;
                if batch != 1 {
                    return Err(VcError::Inference(format!(
                        "unexpected RMVPE batch on zero-copy path: got={} expected=1 shape={:?}",
                        batch, rmvpe_shape
                    )));
                }
                let required = frames.saturating_mul(bins);
                (frames, bins, required)
            }
            [frames, bins] => {
                let frames = (*frames).max(0) as usize;
                let bins = (*bins).max(0) as usize;
                let required = frames.saturating_mul(bins);
                (frames, bins, required)
            }
            _ => {
                return Err(VcError::Inference(format!(
                    "unexpected RMVPE output rank on zero-copy path: shape={:?}",
                    rmvpe_shape
                )))
            }
        };
        if rmvpe_bins != RMVPE_SALIENCE_BINS {
            return Err(VcError::Inference(format!(
                "unexpected RMVPE salience bins on zero-copy path: got={} expected={} shape={:?}",
                rmvpe_bins, RMVPE_SALIENCE_BINS, rmvpe_shape
            )));
        }
        if rmvpe_data.len() < rmvpe_required {
            return Err(VcError::Inference(format!(
                "RMVPE output buffer too small: len={} required={} shape={:?}",
                rmvpe_data.len(),
                rmvpe_required,
                rmvpe_shape
            )));
        }
        debug_assert_eq!(
            rmvpe_frames, RMVPE_EXPECTED_FRAMES,
            "RMVPE strict model should output {} frames before host resample",
            RMVPE_EXPECTED_FRAMES
        );
        let salience =
            ArrayView2::from_shape((rmvpe_frames, rmvpe_bins), &rmvpe_data[..rmvpe_required])
                .map_err(|e| {
                    VcError::Inference(format!("failed to view RMVPE salience as 2D: {e}"))
                })?;
        let f0_101 = decode_f0_from_salience(&salience, threshold);
        let f0_101_voiced_pre = f0_101.iter().filter(|&&x| x > 0.0).count();
        let mut pitchf = resample_f0_linear(&f0_101, self.frame_count);
        debug_assert_eq!(
            pitchf.len(),
            self.frame_count,
            "RMVPE resampled f0 must be exactly {} frames",
            self.frame_count
        );
        if pitchf.len() != self.frame_count {
            return Err(VcError::Inference(format!(
                "unexpected RMVPE resampled frame count on zero-copy path: got={} expected={}",
                pitchf.len(),
                self.frame_count
            )));
        }
        let input_rms = compute_rms(rmvpe_waveform_16k);
        let gated = self.rms_gate_threshold > 0.0 && input_rms < self.rms_gate_threshold;
        if gated {
            pitchf.fill(0.0);
        } else {
            filter_f0_octave_errors(&mut pitchf, 70.0, 800.0);
        }
        if self.audit_enabled {
            self.block_count = self.block_count.saturating_add(1);
            if self.block_count <= 30 {
                let f0_voiced_post = pitchf.iter().filter(|&&x| x > 0.0).count();
                let sal_max =
                    salience
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, |a, b| if a > b { a } else { b });
                let sal_mean = if salience.is_empty() {
                    0.0
                } else {
                    salience.iter().copied().sum::<f32>() / salience.len() as f32
                };
                let mut min_hz = f32::INFINITY;
                let mut max_hz = 0.0_f32;
                for &v in &pitchf {
                    if v > 0.0 {
                        if v < min_hz {
                            min_hz = v;
                        }
                        if v > max_hz {
                            max_hz = v;
                        }
                    }
                }
                if !min_hz.is_finite() {
                    min_hz = 0.0;
                }
                eprintln!(
                    "[rmvpe-audit] block={} salience_shape={:?} sal_max={:.4} sal_mean={:.6} f0_voiced_pre={}/{} f0_voiced_post={}/{} f0_range={:.1}..{:.1}Hz rms={:.4} gated={} rmvpe_th={:.4} rms_gate_th={:.4}",
                    self.block_count,
                    rmvpe_shape,
                    sal_max,
                    sal_mean,
                    f0_101_voiced_pre,
                    rmvpe_frames,
                    f0_voiced_post,
                    self.frame_count,
                    min_hz,
                    max_hz,
                    input_rms,
                    gated,
                    threshold,
                    self.rms_gate_threshold
                );
            }
        }
        smooth_pitch_track_gaussian_inplace(&mut pitchf, 2);
        let pitch = coarse_pitch_from_f0(&pitchf);
        if pitch.len() != self.frame_count {
            return Err(VcError::Inference(format!(
                "pitch frame mismatch on zero-copy path: pitch={} expected={}",
                pitch.len(),
                self.frame_count
            )));
        }

        {
            let (_, buf) = self.decoder_pitch_host.extract_tensor_mut();
            buf.copy_from_slice(&pitch);
        }
        {
            let (_, buf) = self.decoder_pitchf_host.extract_tensor_mut();
            buf.copy_from_slice(&pitchf);
        }
        {
            let (_, buf) = self.decoder_ds_host.extract_tensor_mut();
            buf[0] = speaker_id;
        }
        self.fill_rnd_host();

        map_ort(
            "failed to copy decoder pitch host->CUDA",
            self.decoder_pitch_host
                .copy_into(&mut self.decoder_pitch_gpu),
        )?;
        map_ort(
            "failed to copy decoder pitchf host->CUDA",
            self.decoder_pitchf_host
                .copy_into(&mut self.decoder_pitchf_gpu),
        )?;
        map_ort(
            "failed to copy decoder ds host->CUDA",
            self.decoder_ds_host.copy_into(&mut self.decoder_ds_gpu),
        )?;
        map_ort(
            "failed to copy decoder rnd host->CUDA",
            self.decoder_rnd_host.copy_into(&mut self.decoder_rnd_gpu),
        )?;

        let decoder_output_mem = make_cpu_output_memory_info()?;
        let mut decoder_binding = map_ort(
            "failed to create decoder IoBinding",
            self.decoder.create_binding(),
        )?;
        map_ort(
            "failed to bind decoder output",
            decoder_binding.bind_output_to_device(&self.io.decoder_output, &decoder_output_mem),
        )?;
        map_ort(
            "failed to bind decoder phone input",
            decoder_binding.bind_input(&self.io.decoder_phone_input, &self.features_buf),
        )?;
        map_ort(
            "failed to bind decoder phone_lengths input",
            decoder_binding.bind_input(
                &self.io.decoder_phone_lengths_input,
                &self.decoder_phone_lengths_gpu,
            ),
        )?;
        map_ort(
            "failed to bind decoder pitch input",
            decoder_binding.bind_input(&self.io.decoder_pitch_input, &self.decoder_pitch_gpu),
        )?;
        map_ort(
            "failed to bind decoder pitchf input",
            decoder_binding.bind_input(&self.io.decoder_pitchf_input, &self.decoder_pitchf_gpu),
        )?;
        map_ort(
            "failed to bind decoder ds input",
            decoder_binding.bind_input(&self.io.decoder_ds_input, &self.decoder_ds_gpu),
        )?;
        map_ort(
            "failed to bind decoder rnd input",
            decoder_binding.bind_input(&self.io.decoder_rnd_input, &self.decoder_rnd_gpu),
        )?;

        let mut decoder_outputs = map_ort(
            "decoder run_binding failed",
            self.decoder.run_binding(&decoder_binding),
        )?;
        let decoder_output = decoder_outputs
            .remove(&self.io.decoder_output)
            .ok_or_else(|| {
                VcError::Inference(format!(
                    "decoder output '{}' was not found",
                    self.io.decoder_output
                ))
            })?;
        let decoder_output = map_ort(
            "decoder output is not a tensor",
            decoder_output.downcast::<DynTensorValueType>(),
        )?;
        let (_, audio) = map_ort(
            "failed to extract decoder output tensor<f32>",
            decoder_output.try_extract_tensor::<f32>(),
        )?;
        Ok(audio.to_vec())
    }

    pub fn try_new_from_env(model: &ModelConfig, runtime: &RuntimeConfig) -> Result<Option<Self>> {
        if !env_bool_default_true("RUST_VC_ZERO_COPY_ENABLE") {
            return Ok(None);
        }
        let hubert_model = model.hubert_path.clone().ok_or_else(|| {
            VcError::Config("zero-copy requires hubert_path in ModelConfig".to_string())
        })?;
        let rmvpe_model = model.pitch_extractor_path.clone().ok_or_else(|| {
            VcError::Config("zero-copy requires pitch_extractor_path in ModelConfig".to_string())
        })?;

        let cache_dir = env::var("RUST_VC_ZERO_COPY_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_zero_copy_cache_dir());
        let input_samples =
            env_parse_usize("RUST_VC_ZERO_COPY_INPUT_SAMPLES").unwrap_or(DEFAULT_INPUT_SAMPLES);
        let default_frame_count = HUBERT_BASE_FRAMES
            .saturating_mul(runtime.hubert_upsample_factor.max(1))
            .max(1);
        let frame_count = env_parse_usize("RUST_VC_ZERO_COPY_FRAMES")
            .unwrap_or(default_frame_count)
            .max(1);
        let warmup_runs =
            env_parse_usize("RUST_VC_ZERO_COPY_WARMUP").unwrap_or(DEFAULT_WARMUP_RUNS);
        let io_overrides = ZeroCopyIoOverrides {
            hubert_input: optional_env("RUST_VC_ZERO_COPY_HUBERT_INPUT"),
            hubert_output: optional_env("RUST_VC_ZERO_COPY_HUBERT_OUTPUT"),
            rmvpe_input: optional_env("RUST_VC_ZERO_COPY_RMVPE_INPUT"),
            rmvpe_output: optional_env("RUST_VC_ZERO_COPY_RMVPE_OUTPUT"),
            decoder_phone_input: optional_env("RUST_VC_ZERO_COPY_DECODER_PHONE_INPUT"),
            decoder_phone_lengths_input: optional_env(
                "RUST_VC_ZERO_COPY_DECODER_PHONE_LENGTHS_INPUT",
            ),
            decoder_pitch_input: optional_env("RUST_VC_ZERO_COPY_DECODER_PITCH_INPUT"),
            decoder_pitchf_input: optional_env("RUST_VC_ZERO_COPY_DECODER_PITCHF_INPUT"),
            decoder_ds_input: optional_env("RUST_VC_ZERO_COPY_DECODER_DS_INPUT"),
            decoder_rnd_input: optional_env("RUST_VC_ZERO_COPY_DECODER_RND_INPUT"),
            decoder_output: optional_env("RUST_VC_ZERO_COPY_DECODER_OUTPUT"),
        };
        let config = ZeroCopyEngineConfig {
            hubert_model: PathBuf::from(hubert_model),
            rmvpe_model: PathBuf::from(rmvpe_model),
            decoder_model: PathBuf::from(model.model_path.clone()),
            cache_dir,
            device_id: runtime.ort_device_id,
            input_samples,
            frame_count,
            warmup_runs,
            rmvpe_threshold: runtime.rmvpe_threshold,
            rms_gate_threshold: runtime.response_threshold,
            io_overrides,
        };
        Self::new(config).map(Some)
    }

    fn update_features_buf_from_workspace(&mut self) -> Result<()> {
        if self.features_workspace.shape() != [1, self.frame_count, self.phone_feature_dim] {
            return Err(VcError::Inference(format!(
                "features workspace shape mismatch: got={:?} expected=[1,{},{}]",
                self.features_workspace.shape(),
                self.frame_count,
                self.phone_feature_dim
            )));
        }
        update_features_buf_from_ndarray_impl(
            &self.features_workspace,
            &mut self.features_buf_host,
            &mut self.features_buf,
        )
    }

    fn fill_rnd_host(&mut self) {
        let mut rnd_state = self.rnd_state;
        let (_, rnd) = self.decoder_rnd_host.extract_tensor_mut();
        for v in rnd.iter_mut() {
            *v = next_standard_normal(&mut rnd_state);
        }
        self.rnd_state = rnd_state;
    }

    fn synchronize_gpu_before_drop(&mut self) -> Result<()> {
        map_ort(
            "failed to synchronize HuBERT CUDA buffer before drop",
            self.hubert_waveform_gpu
                .copy_into(&mut self.hubert_host_waveform),
        )?;
        map_ort(
            "failed to synchronize RMVPE CUDA buffer before drop",
            self.rmvpe_waveform_gpu.copy_into(&mut self.host_waveform),
        )?;
        map_ort(
            "failed to synchronize decoder phone CUDA buffer before drop",
            self.features_buf.copy_into(&mut self.features_buf_host),
        )?;
        map_ort(
            "failed to synchronize decoder phone_lengths CUDA buffer before drop",
            self.decoder_phone_lengths_gpu
                .copy_into(&mut self.decoder_phone_lengths_host),
        )?;
        map_ort(
            "failed to synchronize decoder pitch CUDA buffer before drop",
            self.decoder_pitch_gpu
                .copy_into(&mut self.decoder_pitch_host),
        )?;
        map_ort(
            "failed to synchronize decoder pitchf CUDA buffer before drop",
            self.decoder_pitchf_gpu
                .copy_into(&mut self.decoder_pitchf_host),
        )?;
        map_ort(
            "failed to synchronize decoder ds CUDA buffer before drop",
            self.decoder_ds_gpu.copy_into(&mut self.decoder_ds_host),
        )?;
        map_ort(
            "failed to synchronize decoder rnd CUDA buffer before drop",
            self.decoder_rnd_gpu.copy_into(&mut self.decoder_rnd_host),
        )?;
        Ok(())
    }

    pub(crate) fn prepare_for_drop(&mut self) {
        if self.drop_synced {
            return;
        }
        self.drop_synced = true;
        if let Err(e) = self.synchronize_gpu_before_drop() {
            eprintln!("[vc-inference] warning: zero-copy pre-drop sync failed: {e}");
        }
    }
}

impl Drop for ZeroCopyInferenceEngine {
    fn drop(&mut self) {
        self.prepare_for_drop();
    }
}

fn update_features_buf_from_ndarray_impl(
    shaped_ndarray: &Array3<f32>,
    features_buf_host: &mut Tensor<f32>,
    features_buf: &mut Tensor<f32>,
) -> Result<()> {
    let shaped = shaped_ndarray
        .as_slice()
        .ok_or_else(|| VcError::Inference("features ndarray is not contiguous".to_string()))?;
    {
        let (_, host) = features_buf_host.extract_tensor_mut();
        host.copy_from_slice(shaped);
    }
    map_ort(
        "failed to copy features host->CUDA",
        features_buf_host.copy_into(features_buf),
    )
}

fn smooth_pitch_track_gaussian_inplace(pitchf: &mut [f32], radius: usize) {
    if radius == 0 || pitchf.len() < 3 {
        return;
    }
    let src = pitchf.to_vec();
    for i in 0..pitchf.len() {
        if src[i] <= 0.0 {
            continue;
        }
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(src.len());
        for j in lo..hi {
            let v = src[j];
            if v <= 0.0 {
                continue;
            }
            let d = i.abs_diff(j) as f32;
            let sigma = radius.max(1) as f32 * 0.75;
            let w = (-0.5 * (d / sigma).powi(2)).exp();
            num += v * w;
            den += w;
        }
        if den > 1.0e-8 {
            pitchf[i] = num / den;
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(2685821657736338717)
}

fn next_unit(state: &mut u64) -> f32 {
    let x = next_u64(state);
    (((x >> 40) as u32) as f32 + 0.5) / (1u32 << 24) as f32
}

fn next_standard_normal(state: &mut u64) -> f32 {
    let u1 = next_unit(state).clamp(1e-7, 1.0 - 1e-7);
    let u2 = next_unit(state);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    r * theta.cos()
}

fn build_cuda_execution_providers(device_id: i32) -> Vec<ExecutionProviderDispatch> {
    let cuda = ep::CUDA::default()
        .with_device_id(device_id)
        .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
        .with_conv_max_workspace(true)
        .with_arbitrary_config("do_copy_in_default_stream", "1")
        .build()
        .error_on_failure();
    vec![cuda]
}

fn build_session_with_cache(
    model_path: &Path,
    cache_dir: &Path,
    cache_tag: &str,
    eps: &[ExecutionProviderDispatch],
) -> Result<Session> {
    fs::create_dir_all(cache_dir).map_err(|e| {
        VcError::Inference(format!(
            "failed to create cache dir '{}': {e}",
            cache_dir.display()
        ))
    })?;
    let optimized = optimized_cache_path(cache_dir, cache_tag, model_path);
    let load_path = if optimized.exists() {
        optimized.as_path()
    } else {
        model_path
    };

    map_ort(
        "failed to build session",
        Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_optimized_model_path(&optimized))
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.with_parallel_execution(false))
            .and_then(|b| b.with_execution_providers(eps))
            .and_then(|b| b.commit_from_file(load_path)),
    )
}

fn optimized_cache_path(cache_dir: &Path, cache_tag: &str, model_path: &Path) -> PathBuf {
    let (size, mtime_ns) = fs::metadata(model_path)
        .ok()
        .map(|m| {
            let size = m.len();
            let mtime_ns = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            (size, mtime_ns)
        })
        .unwrap_or((0, 0));
    cache_dir.join(format!("{cache_tag}.{size}.{mtime_ns}.opt.onnx"))
}

fn default_zero_copy_cache_dir() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let root = find_workspace_root(&cwd).unwrap_or(cwd);
    root.join("ort_cache")
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    for dir in start.ancestors() {
        let cargo_toml = dir.join("Cargo.toml");
        if !cargo_toml.exists() {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&cargo_toml) {
            if text.contains("[workspace]") {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}

fn resolve_hubert_input_samples(
    hubert: &Session,
    hubert_input_name: &str,
    base_input_samples: usize,
) -> usize {
    let desired = base_input_samples.max(1);
    let fixed = hubert
        .inputs()
        .iter()
        .find(|v| v.name() == hubert_input_name)
        .and_then(|v| v.dtype().tensor_shape().map(|shape| shape.to_vec()))
        .and_then(|shape| {
            if shape.len() >= 2 && shape[1] > 0 {
                Some(shape[1] as usize)
            } else {
                None
            }
        });
    match fixed {
        Some(v) if v != desired => {
            eprintln!(
                "[vc-inference] warning: HuBERT input shape is fixed at {} samples; requested {}",
                v, desired
            );
            v.max(1)
        }
        Some(v) => v.max(1),
        None => desired.max(1),
    }
}

fn infer_decoder_phone_feature_dim(decoder: &Session, phone_input_name: &str) -> usize {
    decoder
        .inputs()
        .iter()
        .find(|v| v.name() == phone_input_name)
        .and_then(|v| v.dtype().tensor_shape().map(|shape| shape.to_vec()))
        .and_then(|shape| {
            if shape.len() >= 3 && shape[2] > 0 {
                Some(shape[2] as usize)
            } else {
                None
            }
        })
        .unwrap_or(768)
}

fn make_cuda_allocator(session: &Session, device_id: i32) -> Result<Allocator> {
    let mem_info = make_cuda_memory_info(device_id)?;
    map_ort(
        "failed to create CUDA allocator",
        Allocator::new(session, mem_info),
    )
}

fn make_cuda_memory_info(device_id: i32) -> Result<MemoryInfo> {
    map_ort(
        "failed to create CUDA MemoryInfo",
        MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        ),
    )
}

fn make_cpu_output_memory_info() -> Result<MemoryInfo> {
    map_ort(
        "failed to create CPU output MemoryInfo",
        MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        ),
    )
}

fn resolve_io_names(
    hubert: &Session,
    rmvpe: &Session,
    decoder: &Session,
    overrides: &ZeroCopyIoOverrides,
) -> Result<ZeroCopyIoNames> {
    Ok(ZeroCopyIoNames {
        hubert_input: resolve_single_io_name(
            hubert
                .inputs()
                .iter()
                .map(|v| v.name().to_string())
                .collect(),
            overrides.hubert_input.as_ref(),
            &["audio", "source", "wave", "input"],
            "hubert input",
        )?,
        hubert_output: resolve_single_io_name(
            hubert
                .outputs()
                .iter()
                .map(|v| v.name().to_string())
                .collect(),
            overrides.hubert_output.as_ref(),
            &["phone", "feat", "hubert", "output"],
            "hubert output",
        )?,
        rmvpe_input: resolve_single_io_name(
            rmvpe
                .inputs()
                .iter()
                .map(|v| v.name().to_string())
                .collect(),
            overrides.rmvpe_input.as_ref(),
            &["audio", "source", "wave", "input"],
            "rmvpe input",
        )?,
        rmvpe_output: resolve_single_io_name(
            rmvpe
                .outputs()
                .iter()
                .map(|v| v.name().to_string())
                .collect(),
            overrides.rmvpe_output.as_ref(),
            &["salience", "pitch", "f0", "output"],
            "rmvpe output",
        )?,
        decoder_phone_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_phone_input.as_ref(),
            "decoder phone input",
            |name, shape, ty| {
                if name.contains("phone") && !name.contains("length") {
                    return true;
                }
                is_float_type(ty) && shape.len() >= 3 && shape.last().copied().unwrap_or(0) == 768
            },
        )?,
        decoder_phone_lengths_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_phone_lengths_input.as_ref(),
            "decoder phone_lengths input",
            |name, shape, ty| {
                name.contains("length")
                    || (is_integer_type(ty) && shape.len() <= 2 && name.contains("phone"))
            },
        )?,
        decoder_pitch_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_pitch_input.as_ref(),
            "decoder pitch input",
            |name, shape, ty| {
                (name == "pitch" || (name.contains("pitch") && !name.contains("pitchf")))
                    || (is_integer_type(ty) && shape.len() == 2 && name.contains("pitch"))
            },
        )?,
        decoder_pitchf_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_pitchf_input.as_ref(),
            "decoder pitchf input",
            |name, shape, ty| {
                name.contains("pitchf")
                    || name.contains("f0")
                    || (is_float_type(ty) && shape.len() == 2 && name.contains("pitch"))
            },
        )?,
        decoder_ds_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_ds_input.as_ref(),
            "decoder ds input",
            |name, shape, ty| {
                name == "ds"
                    || name.contains("speaker")
                    || name.contains("sid")
                    || (is_integer_type(ty)
                        && shape.len() <= 1
                        && (name.contains("ds") || name.contains("spk")))
            },
        )?,
        decoder_rnd_input: infer_decoder_input_name(
            decoder,
            overrides.decoder_rnd_input.as_ref(),
            "decoder rnd input",
            |name, shape, ty| {
                name.contains("rnd")
                    || name.contains("noise")
                    || (is_float_type(ty)
                        && shape.len() == 3
                        && shape.get(1).copied().unwrap_or(0) == RND_CHANNELS as i64)
            },
        )?,
        decoder_output: resolve_single_io_name(
            decoder
                .outputs()
                .iter()
                .map(|v| v.name().to_string())
                .collect(),
            overrides.decoder_output.as_ref(),
            &["audio", "output", "wave"],
            "decoder output",
        )?,
    })
}

fn infer_decoder_input_name<F>(
    session: &Session,
    override_name: Option<&String>,
    label: &str,
    predicate: F,
) -> Result<String>
where
    F: Fn(&str, &[i64], Option<TensorElementType>) -> bool,
{
    if let Some(name) = override_name {
        if session.inputs().iter().any(|v| v.name() == name.as_str()) {
            return Ok(name.clone());
        }
        return Err(VcError::Config(format!(
            "{} override '{}' does not exist in decoder inputs",
            label, name
        )));
    }

    for input in session.inputs() {
        let lname = input.name().to_ascii_lowercase();
        let shape_vec = input
            .dtype()
            .tensor_shape()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let ty = input.dtype().tensor_type();
        if predicate(&lname, &shape_vec, ty) {
            return Ok(input.name().to_string());
        }
    }

    let known: Vec<String> = session
        .inputs()
        .iter()
        .map(|v| v.name().to_string())
        .collect();
    Err(VcError::Config(format!(
        "failed to infer {}; available decoder inputs: {:?}",
        label, known
    )))
}

fn resolve_single_io_name(
    available: Vec<String>,
    override_name: Option<&String>,
    keywords: &[&str],
    label: &str,
) -> Result<String> {
    if let Some(name) = override_name {
        if available.iter().any(|v| v == name) {
            return Ok(name.clone());
        }
        return Err(VcError::Config(format!(
            "{} override '{}' was not found in {:?}",
            label, name, available
        )));
    }

    for key in keywords {
        if let Some(found) = available
            .iter()
            .find(|name| name.to_ascii_lowercase().contains(key))
        {
            return Ok(found.clone());
        }
    }

    available.first().cloned().ok_or_else(|| {
        VcError::Config(format!(
            "failed to infer {}; model has no matching inputs/outputs",
            label
        ))
    })
}

fn is_float_type(ty: Option<TensorElementType>) -> bool {
    matches!(
        ty,
        Some(TensorElementType::Float16)
            | Some(TensorElementType::Float32)
            | Some(TensorElementType::Float64)
            | Some(TensorElementType::Bfloat16)
    )
}

fn is_integer_type(ty: Option<TensorElementType>) -> bool {
    matches!(
        ty,
        Some(TensorElementType::Int8)
            | Some(TensorElementType::Int16)
            | Some(TensorElementType::Int32)
            | Some(TensorElementType::Int64)
            | Some(TensorElementType::Uint8)
            | Some(TensorElementType::Uint16)
            | Some(TensorElementType::Uint32)
            | Some(TensorElementType::Uint64)
    )
}

fn map_ort<T>(context: &str, result: ort::Result<T>) -> Result<T> {
    result.map_err(|e| VcError::Inference(format!("{context}: {e}")))
}

fn env_parse_usize(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

fn optional_env(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_bool_default_true(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "no" || s == "off")
        })
        .unwrap_or(true)
}

/// Decodes RMVPE salience map to F0 Hz values.
///
/// `salience` is shaped as `[frames, bins]` where `bins` is typically 360.
/// Values may be logits or already normalized probabilities.
/// Returns `0.0` for unvoiced frames.
fn decode_f0_from_salience(salience: &ArrayView2<'_, f32>, threshold: f32) -> Vec<f32> {
    let n_frames = salience.shape()[0];
    let n_bins = salience.shape()[1];
    if n_frames == 0 || n_bins == 0 {
        return Vec::new();
    }
    let threshold = threshold.max(0.0);
    let mut f0 = vec![0.0_f32; n_frames];

    for t in 0..n_frames {
        let row = salience.row(t);
        let mut peak_bin = 0usize;
        let mut peak_val = f32::NEG_INFINITY;
        for (idx, &v) in row.iter().enumerate() {
            if v > peak_val {
                peak_val = v;
                peak_bin = idx;
            }
        }
        if !peak_val.is_finite() || peak_val <= threshold {
            continue;
        }

        let start = peak_bin.saturating_sub(RMVPE_DECODE_RADIUS);
        let end = (peak_bin + RMVPE_DECODE_RADIUS + 1).min(n_bins);
        if start >= end {
            continue;
        }

        let mut local_max = f32::NEG_INFINITY;
        for b in start..end {
            let v = row[b];
            if v > local_max {
                local_max = v;
            }
        }
        if !local_max.is_finite() {
            continue;
        }

        let mut cents_num = 0.0_f32;
        let mut cents_den = 0.0_f32;
        for b in start..end {
            let w = (row[b] - local_max).exp();
            let cents = RMVPE_CENTS_BASE + RMVPE_CENTS_STEP * b as f32;
            cents_num += cents * w;
            cents_den += w;
        }
        if cents_den > f32::EPSILON {
            let cents = cents_num / cents_den;
            f0[t] = 10.0_f32 * 2.0_f32.powf(cents / 1200.0_f32);
        }
    }
    f0
}

/// Resamples f0 array from `src.len()` to `dst_len` with linear interpolation.
///
/// Unvoiced frames (`0.0`) are treated as gaps and not linearly blended across
/// voiced/unvoiced boundaries.
fn resample_f0_linear(src: &[f32], dst_len: usize) -> Vec<f32> {
    if dst_len == 0 {
        return Vec::new();
    }
    if src.is_empty() {
        return vec![0.0_f32; dst_len];
    }
    if src.len() == 1 {
        return vec![src[0]; dst_len];
    }
    if dst_len == 1 {
        return vec![src[0]];
    }

    let src_len = src.len();
    let mut dst = vec![0.0_f32; dst_len];
    for (i, out) in dst.iter_mut().enumerate() {
        let src_pos = i as f32 * (src_len as f32 - 1.0) / (dst_len as f32 - 1.0);
        let lo = src_pos.floor() as usize;
        let hi = (lo + 1).min(src_len - 1);
        let t = src_pos - lo as f32;

        let a = src[lo];
        let b = src[hi];
        *out = match (a > 0.0, b > 0.0) {
            (true, true) => a * (1.0 - t) + b * t,
            (true, false) => a,
            (false, true) => b,
            (false, false) => 0.0,
        };
    }
    dst
}

#[inline]
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[inline]
fn resolve_rms_gate_threshold(raw: f32) -> f32 {
    if !raw.is_finite() {
        return DEFAULT_RMS_GATE_THRESHOLD;
    }
    if raw > 0.0 {
        return raw.clamp(0.0, 1.0);
    }
    if raw < -1.0 {
        // Treat negative values as dBFS (e.g. -50 dB -> ~0.00316 linear).
        return 10.0_f32.powf(raw / 20.0).clamp(0.0, 1.0);
    }
    // `0.0` (or tiny around zero) should not disable gate silently.
    DEFAULT_RMS_GATE_THRESHOLD
}

fn filter_f0_octave_errors(f0: &mut Vec<f32>, f0_min_hz: f32, f0_max_hz: f32) {
    let mut voiced: Vec<f32> = f0
        .iter()
        .copied()
        .filter(|&x| x >= f0_min_hz * 0.5 && x <= f0_max_hz * 2.0)
        .collect();
    if voiced.is_empty() {
        return;
    }
    voiced.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = voiced[voiced.len() / 2];
    for v in f0.iter_mut() {
        if *v <= 0.0 {
            continue;
        }
        let mut x = *v;
        while x > 0.0 && x < f0_min_hz {
            x *= 2.0;
            if x > f0_max_hz * 2.0 {
                x = 0.0;
                break;
            }
        }
        while x > f0_max_hz {
            x *= 0.5;
            if x < f0_min_hz * 0.5 {
                x = 0.0;
                break;
            }
        }
        if x <= 0.0 {
            *v = 0.0;
            continue;
        }

        // Conservative octave fix around local median.
        if x < median * 0.55 {
            let up = x * 2.0;
            if up >= f0_min_hz && up <= f0_max_hz {
                x = up;
            }
        } else if x > median * 1.95 {
            let down = x * 0.5;
            if down >= f0_min_hz && down <= f0_max_hz {
                x = down;
            }
        }

        if x < f0_min_hz || x > f0_max_hz {
            // Keep as unvoiced only when it's still out of valid range.
            *v = 0.0;
        } else {
            *v = x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_rms, decode_f0_from_salience, filter_f0_octave_errors, resample_f0_linear,
        resolve_rms_gate_threshold,
    };
    use ndarray::Array2;

    #[test]
    fn test_decode_f0_voiced() {
        let mut sal = Array2::<f32>::zeros((1, 360));
        sal[[0, 100]] = 10.0;
        let f0 = decode_f0_from_salience(&sal.view(), 0.006);
        assert!(f0[0] > 0.0, "should be voiced");
    }

    #[test]
    fn test_decode_f0_unvoiced() {
        let sal = Array2::<f32>::ones((1, 360));
        let f0 = decode_f0_from_salience(&sal.view(), 2.0);
        assert_eq!(f0[0], 0.0, "flat salience should be unvoiced");
    }

    #[test]
    fn test_decode_f0_bin0_base_frequency() {
        let mut sal = Array2::<f32>::zeros((1, 360));
        sal[[0, 0]] = 10.0;
        let f0 = decode_f0_from_salience(&sal.view(), 0.01);
        assert!(
            (30.0..35.0).contains(&f0[0]),
            "bin0 should decode near C1 (~32.7Hz), got {}",
            f0[0]
        );
    }

    #[test]
    fn test_resample_f0_shape() {
        let src: Vec<f32> = (0..101)
            .map(|i| if i % 3 == 0 { 0.0 } else { 440.0 })
            .collect();
        let dst = resample_f0_linear(&src, 50);
        assert_eq!(dst.len(), 50);
    }

    #[test]
    fn test_resample_f0_no_voiced_unvoiced_bleed() {
        let mut src = vec![440.0f32; 101];
        for v in src.iter_mut().skip(50) {
            *v = 0.0;
        }
        let dst = resample_f0_linear(&src, 50);
        assert!(
            dst[49] == 0.0 || dst[48] == 0.0,
            "unvoiced region must not be filled by interpolation"
        );
    }

    #[test]
    fn test_filter_f0_octave_errors_removes_subharmonic() {
        let mut f0 = vec![55.0, 110.0, 112.0, 108.0, 54.0, 111.0];
        filter_f0_octave_errors(&mut f0, 70.0, 800.0);
        assert!(f0[0] > 70.0 || f0[0] == 0.0);
        assert!(f0[4] > 70.0 || f0[4] == 0.0);
        assert!(f0[1] > 100.0);
    }

    #[test]
    fn test_filter_f0_removes_out_of_range() {
        let mut f0 = vec![40.0, 110.0, 900.0, 115.0];
        filter_f0_octave_errors(&mut f0, 70.0, 800.0);
        assert!(f0[0] >= 70.0 || f0[0] == 0.0);
        assert!((70.0..=800.0).contains(&f0[2]) || f0[2] == 0.0);
        assert!(f0[1] > 0.0);
    }

    #[test]
    fn test_compute_rms_silence() {
        let samples = vec![0.0f32; 1024];
        assert_eq!(compute_rms(&samples), 0.0);
    }

    #[test]
    fn test_compute_rms_sine() {
        let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let rms = compute_rms(&samples);
        assert!((rms - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_resolve_rms_gate_threshold_from_db() {
        let th = resolve_rms_gate_threshold(-50.0);
        assert!((th - 0.00316).abs() < 0.0005);
    }

    #[test]
    fn test_resolve_rms_gate_threshold_zero_uses_default() {
        let th = resolve_rms_gate_threshold(0.0);
        assert!(th > 0.0);
    }
}
