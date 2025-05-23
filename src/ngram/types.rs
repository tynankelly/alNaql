use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NGramEntry {
    pub sequence_number: u64,
    pub file_path: String,
    pub text_position: TextPosition,
    pub text_content: NGramText,
}

#[derive(Debug, Clone)]
pub struct NGramConfig {
    pub ngram_size: usize,
}

impl Default for NGramConfig {
    fn default() -> Self {
        Self {
            ngram_size: 5
        }
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    #[serde(default = "default_remove_numbers")]
    pub remove_numbers: bool,
    #[serde(default = "default_remove_diacritics")]
    pub remove_diacritics: bool,
    #[serde(default = "default_remove_tatweel")]
    pub remove_tatweel: bool,
    #[serde(default = "default_normalize_arabic")]
    pub normalize_arabic: bool,
    #[serde(default = "default_preserve_punctuation")]
    pub preserve_punctuation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_min_match_length")]
    pub min_match_length: usize,
    #[serde(default = "default_max_gap")]
    pub max_gap: usize,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    
    #[serde(default = "default_processing_mode")]
    pub processing_mode: String,
    #[serde(default = "default_max_concurrent_files")]
    pub max_concurrent_files: usize,
    #[serde(default = "default_force_sequential_memory_threshold")]
    pub force_sequential_memory_threshold: f64,
    #[serde(default = "default_target_memory_percent")]
    pub target_memory_percent: f64,

    #[serde(default)]
    pub text_processing: TextProcessingConfig,
}

// Text processing defaults
fn default_remove_numbers() -> bool { true }
fn default_remove_diacritics() -> bool { true }
fn default_remove_tatweel() -> bool { true }
fn default_normalize_arabic() -> bool { true }
fn default_preserve_punctuation() -> bool { false }

// Processing defaults
fn default_ngram_size() -> usize { 5 }
fn default_min_match_length() -> usize { 5 }
fn default_max_gap() -> usize { 10 }
fn default_similarity_threshold() -> f64 { 0.9 }
fn default_batch_size() -> usize { 1000 }
fn default_processing_mode() -> String { "sequential".to_string() }
fn default_max_concurrent_files() -> usize { 0 }
fn default_force_sequential_memory_threshold() -> f64 { 80.0 }
fn default_target_memory_percent() -> f64 { 70.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    pub source_dir: PathBuf,
    pub target_dir: PathBuf,
    pub output_dir: PathBuf,
    pub temp_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub rocksdb_path: PathBuf,
    pub compression: bool,
    pub max_open_files: i32,
    pub cache_size_mb: usize,
    pub batch_size: usize,
    pub write_buffer_size: usize,
    pub max_write_buffer_number: i32,
    pub target_file_size_mb: usize,
    pub parallel_threads: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            rocksdb_path: PathBuf::from("db/ngrams"),
            compression: true,
            max_open_files: 2000,
            cache_size_mb: 4096,
            batch_size: 5000,
            write_buffer_size: 256,
            max_write_buffer_number: 5,
            target_file_size_mb: 256,
            parallel_threads: 4,
        }
    }
}

impl Default for TextProcessingConfig {
    fn default() -> Self {
        Self {
            remove_numbers: default_remove_numbers(),
            remove_diacritics: default_remove_diacritics(),
            remove_tatweel: default_remove_tatweel(),
            normalize_arabic: default_normalize_arabic(),
            preserve_punctuation: default_preserve_punctuation(),
        }
    }
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

impl ProcessingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.ngram_size < 1 {
            return Err(ProcessingError::InvalidNgramSize(
                "Ngram size must be at least 1".to_string()
            ));
        }
        if self.min_match_length > self.ngram_size {
            return Err(ProcessingError::InvalidNgramSize(
                "Minimum match length cannot be greater than ngram size".to_string()
            ));
        }
        if self.similarity_threshold < 0.0 || self.similarity_threshold > 1.0 {
            return Err(ProcessingError::ConfigError(
                "Similarity threshold must be between 0 and 1".to_string()
            ));
        }
        Ok(())
    }
}

impl FileConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.source_dir.exists() {
            return Err(ProcessingError::ConfigError(
                format!("Source directory does not exist: {:?}", self.source_dir)
            ));
        }
        if !self.target_dir.exists() {
            return Err(ProcessingError::ConfigError(
                format!("Target directory does not exist: {:?}", self.target_dir)
            ));
        }
        // Create output and temp directories if they don't exist
        std::fs::create_dir_all(&self.output_dir).map_err(ProcessingError::IoError)?;
        std::fs::create_dir_all(&self.temp_dir).map_err(ProcessingError::IoError)?;
        Ok(())
    }
} 