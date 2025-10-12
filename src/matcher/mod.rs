pub mod buffered_writer;
pub mod clusters;
pub mod grid_orchestrator;
pub mod ngrams_matcher;
pub mod orchestrator;
pub mod similarity_algorithms;
pub mod types;

// Add public exports
pub use orchestrator::{MatcherOrchestrator, MatcherSharedState};

use crate::storage::LMDBStorage;
// The is_traced_ngram function that sequential_ngrams imports
#[inline]
/// Check if a sequence contains the traced phrase
pub fn is_traced_sequence(
    seq: u64, 
    storage: Option<&LMDBStorage>, 
    traced_phrase: &str
) -> bool {
    if traced_phrase.is_empty() {
        return false;
    }
    
    if let Some(storage) = storage {
        if let Ok(text_map) = storage.get_text_bytes_batch(&[seq]) {
            if let Some(bytes) = text_map.get(&seq) {
                if let Ok(text) = std::str::from_utf8(bytes) {
                    return text.contains(traced_phrase);
                }
            }
        }
    }
    false
}