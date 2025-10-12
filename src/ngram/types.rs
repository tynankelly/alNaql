// ngram/types.rs

use ahash::{AHashMap, AHashSet};
use crate::types::NGramEntry;

//==============================================================================
// TYPE ALIASES
//==============================================================================

/// Represents a byte position range in the original text
/// (start_byte, end_byte) - used for tracking processed positions
pub type PositionKey = (usize, usize);

//==============================================================================
// CONSTANTS
//==============================================================================

/// Minimum text size (in bytes) to trigger parallel processing
pub const MIN_PARALLEL_THRESHOLD: usize = 500_000;

//==============================================================================
// TEXT PROCESSING STRUCTURES
//==============================================================================

/// Represents a window of text being processed, with both relative and absolute positions
#[derive(Debug, Clone)]
pub struct TextWindow {
   
    // Relative positions within the window
    pub main_content_start: usize,
    pub main_content_end: usize,
    pub window_start: usize,
    pub window_end: usize,
    
    // Absolute positions in the original file
    pub abs_window_start: usize,
    pub abs_window_end: usize,
    pub abs_content_start: usize,
    pub abs_content_end: usize,
    
    // Line tracking
    pub start_line: usize,
}

/// Defines boundaries for a chunk of text to be processed
#[derive(Debug, Clone)]
pub struct ChunkBoundary {
    /// Start position of the chunk
    pub start: usize,
    /// End position of the chunk           
    pub end: usize,
    /// Start of overlap region with previous chunk             
    pub overlap_start: usize,
    /// End of overlap region   
    pub overlap_end: usize,
    /// Starting line number     
    pub start_line: usize,
}

/// Information gathered from pre-scanning text for safe chunking
#[derive(Debug)]
pub struct TextScanInfo {
    /// Positions of XML tags (start, end) to avoid splitting
    pub tag_positions: Vec<(usize, usize)>,
    /// Byte positions where splitting is safe
    pub safe_points: Vec<usize>,
    /// Count of Arabic characters for density calculation
    pub arabic_char_count: usize,
    /// Total character count
    pub total_char_count: usize,
    /// Byte positions of line breaks for line number tracking
    pub line_breaks: Vec<usize>,
    /// Byte positions of word boundaries for word-based ngrams
    pub word_boundaries: Vec<usize>,
}

//==============================================================================
// GENERATION RESULTS
//==============================================================================

/// Result of ngram generation, containing both ngrams and tracking information
pub struct GenerationResult {
    /// The generated ngrams
    pub ngrams: Vec<NGramEntry>,
    /// Set of processed positions to avoid duplicates
    pub positions: AHashSet<PositionKey>,
    pub frequencies: AHashMap<u64, u32>,
}
