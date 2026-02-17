use thiserror::Error;

pub type Result<T> = std::result::Result<T, VcError>;

#[derive(Debug, Error)]
pub enum VcError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("audio backend error: {0}")]
    Audio(String),
    #[error("inference engine error: {0}")]
    Inference(String),
}
