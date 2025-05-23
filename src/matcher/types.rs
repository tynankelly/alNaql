// types.rs
use serde::{Serialize, Deserialize};
use std::ops::Range;
use crate::types::NGramEntry;
pub use crate::config::subsystems::matcher::SimilarityMetric;

#[derive(Debug, Clone)]
pub struct NGramMatch {
    pub source_position: usize,
    pub target_position: usize,
    pub ngram: NGramEntry,
    pub similarity: f64,
}

#[derive(Debug, Clone)]
pub struct MatchCandidate {
    pub source_range: Range<usize>,
    pub target_range: Range<usize>,
    pub matches: Vec<NGramMatch>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub source_text: String,
    pub target_text: String,
    pub similarity: f64,
    pub source_range: Range<usize>,
    pub target_range: Range<usize>,
    pub similarity_metric: SimilarityMetric,
    pub is_isnad: bool,
    pub source_seq_range: Range<usize>,
    pub target_seq_range: Range<usize>, 
}

// For serializing to JSONL output
impl From<MatchResult> for SimpleMatchResult {
    fn from(result: MatchResult) -> Self {
        Self {
            source_text: result.source_text,
            target_text: result.target_text,
            similarity: result.similarity,
            source_start: result.source_range.start,
            source_end: result.source_range.end,
            target_start: result.target_range.start,
            target_end: result.target_range.end,
            similarity_metric: result.similarity_metric.as_str().to_string(),
            is_isnad: result.is_isnad
        }
    }
}

// For JSON output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMatchResult {
    pub source_text: String,
    pub source_start: usize,
    pub source_end: usize,
    pub target_text: String,
    pub target_start: usize,
    pub target_end: usize,
    pub similarity: f64,
    pub similarity_metric: String,
    pub is_isnad: bool,
}

// Helper type for internal use in matcher
#[derive(Debug, Clone)]
pub struct TextSpan {
    pub original_text: String,
    pub ngrams: Vec<NGramEntry>,
}