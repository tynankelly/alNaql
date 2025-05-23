use thiserror::Error;
use std::io;
use std::time::SystemTimeError;
use tokio::task::JoinError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Text processing error: {0}")]
    TextProcessing(String),
    
    #[error("Invalid ngram size: {0}")]
    InvalidNgramSize(String),
    
    #[error("Memory limit exceeded: {0}")]
    Memory(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Database error: {0}")]
    Database(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Thread join error: {0}")]
    ThreadJoin(#[from] JoinError),
    
    // Add a new variant for general async/thread errors
    #[error("Async operation error: {0}")]
    AsyncError(String),
    
    #[error("Metrics error: {0}")]
    Metrics(String),
    
    #[error("Processing boundary error: {0}")]
    Boundary(String),

    #[error("Sequence validation error: {0}")]
    SequenceValidation(String),
}

// Type alias for Result
pub type Result<T> = std::result::Result<T, Error>;

// Helper functions for common error conversions
impl Error {
    pub fn storage<S: Into<String>>(msg: S) -> Self {
        Error::Storage(msg.into())
    }

    pub fn config<S: Into<String>>(msg: S) -> Self {
        Error::Config(msg.into())
    }

    pub fn text<S: Into<String>>(msg: S) -> Self {
        Error::TextProcessing(msg.into())
    }

    pub fn async_err<S: Into<String>>(msg: S) -> Self {
        Error::AsyncError(msg.into())
    }
    pub fn sequence_validation<S: Into<String>>(msg: S) -> Self {
        Error::SequenceValidation(msg.into())
    }
}

impl From<crate::parser::ParserError> for Error {
    fn from(err: crate::parser::ParserError) -> Self {
        Error::TextProcessing(err.to_string())
    }
}
impl From<SystemTimeError> for Error {
    fn from(err: SystemTimeError) -> Self {
        Error::TextProcessing(format!("System time error: {}", err))
    }
}

impl From<crate::types::ProcessingError> for Error {
    fn from(err: crate::types::ProcessingError) -> Self {
        match err {
            crate::types::ProcessingError::IoError(e) => Error::Io(e),
            crate::types::ProcessingError::ConfigError(e) => Error::Config(e),
            crate::types::ProcessingError::StorageError(e) => Error::Storage(e.to_string()),
            crate::types::ProcessingError::ParsingError(e) => Error::TextProcessing(e),
            crate::types::ProcessingError::CleaningError(e) => Error::TextProcessing(e),
            crate::types::ProcessingError::InvalidNgramSize(e) => Error::InvalidNgramSize(e),
            crate::types::ProcessingError::MemoryError(e) => Error::Memory(e),
            crate::types::ProcessingError::ThreadError(e) => Error::AsyncError(e),
            crate::types::ProcessingError::TextProcessing(e) => Error::TextProcessing(e),
        }
    }
}

// Implement From for common external error types
impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<sys_info::Error> for Error {
    fn from(err: sys_info::Error) -> Self {
        Error::Storage(err.to_string())
    }
}

impl From<rayon::ThreadPoolBuildError> for Error {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        Error::AsyncError(format!("Thread pool build failed: {}", err))
    }
}