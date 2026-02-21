use crate::{Result, RuntimeConfig};

pub trait InferenceEngine: Send + Sync + 'static {
    fn infer_frame(&mut self, frame: &[f32], config: &RuntimeConfig) -> Result<Vec<f32>>;

    fn prepare_for_shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct VoiceChanger<E: InferenceEngine> {
    engine: E,
    config: RuntimeConfig,
}

impl<E: InferenceEngine> VoiceChanger<E> {
    pub fn new(engine: E, config: RuntimeConfig) -> Self {
        Self { engine, config }
    }

    pub fn update_config(&mut self, config: RuntimeConfig) {
        self.config = config;
    }

    pub fn set_effective_block_size(&mut self, block_size: usize) {
        self.config.block_size = block_size.max(1);
    }

    pub fn process_frame(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut pre = Vec::with_capacity(input.len());
        for sample in input {
            pre.push(sample * self.config.input_gain);
        }

        let mut out = self.engine.infer_frame(&pre, &self.config)?;
        for sample in &mut out {
            *sample *= self.config.output_gain;
        }

        Ok(out)
    }

    pub fn prepare_shutdown(&mut self) -> Result<()> {
        self.engine.prepare_for_shutdown()
    }
}
