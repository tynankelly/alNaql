use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NGramEntry {
    pub sequence_number: u64,
    pub file_path: String,
    pub text_position: TextPosition,
    pub text_content: NGramText,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextPosition {
    pub start_byte: usize,
    pub end_byte: usize,
    pub line_number: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NGramText {
    pub original_slice: String,
    pub cleaned_text: String,
}

#[derive(Debug, Error)]
pub enum ProcessingError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),
    
    #[error("Parser error: {0}")]
    ParsingError(String),
    
    #[error("Text cleaning error: {0}")]
    CleaningError(String),
    
    #[error("Invalid ngram size: {0}")]
    InvalidNgramSize(String),
    
    #[error("Memory limit exceeded: {0}")]
    MemoryError(String),
    
    #[error("Thread error: {0}")]
    ThreadError(String),

    #[error("Text processing error: {0}")]
    TextProcessing(String),
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Database operation failed: {0}")]
    Database(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Memory limit exceeded: {0}")]
    Memory(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, ProcessingError>;
