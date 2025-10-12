// src/ngram/mod.rs

pub mod types;
pub mod arabic_char_mapping;
pub mod generator;
pub mod character;
pub mod word;
pub mod sequential;
pub mod parallel;
pub mod utils;
pub mod features;
pub mod chunking;

// PUBLIC EXPORTS - This is what bin/generate_ngrams.rs needs
pub use generator::NGramGenerator;
pub use types::{PositionKey, GenerationResult};

// Re-export for backward compatibility if needed elsewhere
pub use generator::GeneratorStatistics;