// src/ngram/generator/mod.rs

mod core;
mod char_ngrams;
mod word_ngrams;
mod chunking;
mod parallel;
mod position;

// Re-export the main types and structs
pub use self::core::NGramGenerator;
pub use self::core::PositionKey;
pub use self::position::GenerationResult;

// Any other necessary re-exports to maintain the public API