//! kuttab is a library for Arabic text comparison and matching using n-grams.
//! It provides functionality for generating, storing, and comparing n-grams
//! with special handling for Arabic text characteristics.

// Module declarations
pub mod error;
pub mod parser;
pub mod storage;
pub mod ngram;
pub mod matcher;
pub mod utils;
pub mod config;
pub mod types;

// Re-exports
pub use error::{Error, Result};
pub use matcher::{ClusteringMatcher, SimilarityCalculator};
pub use utils::{
    checkpoint::CheckpointManager,
    tempfile::TempFileManager,
    processing::ProcessingManager,
};


// Re-export the config from config module
pub use config::KuttabConfig;