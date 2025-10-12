// matcher/types.rs
use ahash::AHashMap;
use std::path::PathBuf;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use crate::types::CompactCharFrequencies;

// ============================================================================
// STAGE 1: SIMILARITY CALCULATION TYPES
// ============================================================================

/// Output of NGRAM similarity calculation - just the match fact, no score
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SequenceMatch {
    pub source_sequence: u64,
    pub target_sequence: u64,
    // No similarity score - we only care that it passed threshold
}

// ============================================================================
// STAGE 2: CLUSTERING/DENSE REGIONS TYPES
// ============================================================================

/// A cluster of matching sequences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceCluster {
    pub source_start: u64,
    pub source_end: u64,
    pub target_start: u64,
    pub target_end: u64,
    pub match_count: usize,  // Number of matches in this cluster
}

// ============================================================================
// STAGE 3: TEXT RECONSTRUCTION TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructedSegment {
    pub source_text: String,
    pub target_text: String,
    pub source_start_seq: u64,
    pub source_end_seq: u64,
    pub target_start_seq: u64,
    pub target_end_seq: u64,
}

// ============================================================================
// STAGE 4: VALIDATION & MERGING TYPES
// ============================================================================

/// Text segment after validation (banality removal, isnad detection)
#[derive(Debug, Clone)]
pub struct ValidatedSegment {
    pub source_text: String,
    pub target_text: String,
    pub source_sequences: (u64, u64),  // (start, end)
    pub target_sequences: (u64, u64),  // (start, end)
    pub is_isnad: bool,
    pub similarity_score: f32,  // Final similarity after validation
}

// ============================================================================
// STAGE 6: FINAL OUTPUT TYPES
// ============================================================================


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalMatchResult {
    pub source_text: String,
    pub target_text: String,
    pub source_byte_start: usize,
    pub source_byte_end: usize,
    pub target_byte_start: usize,
    pub target_byte_end: usize,
    pub similarity_score: f32,
    pub is_isnad: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Jaccard,
    Cosine,
    Levenshtein,
    VSA,
    Exact,
}

// ============================================================================
// BATCH PROCESSING TYPES
// ============================================================================

/// Batch of sequences with their column data for efficient processing
#[derive(Debug)]
pub struct ColumnBatch {
    pub sequences: Vec<u64>,
    pub text_bytes: Option<AHashMap<u64, Vec<u8>>>,
    pub char_freqs: Option<AHashMap<u64, CompactCharFrequencies>>,
    pub prefixes: Option<AHashMap<u64, Vec<u8>>>,
}

impl ColumnBatch {
    /// Create empty batch with just sequences
    pub fn new(sequences: Vec<u64>) -> Self {
        Self {
            sequences,
            text_bytes: None,
            char_freqs: None,
            prefixes: None,
        }
    }
    
    /// Load specific columns from storage
    pub fn load_columns(&mut self, 
        storage: &LMDBStorage, 
        columns: &[ColumnType]
    ) -> Result<()> {
        for column in columns {
            match column {
                ColumnType::TextBytes => {
                    self.text_bytes = Some(storage.get_text_bytes_batch(&self.sequences)?);
                }
                ColumnType::CharFrequencies => {
                    self.char_freqs = Some(storage.get_char_frequencies_batch(&self.sequences)?);
                }
                ColumnType::Prefixes => {
                    self.prefixes = Some(storage.get_prefix_bytes_batch(&self.sequences)?);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ColumnType {
    TextBytes,
    CharFrequencies,
    Prefixes,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

use crate::storage::LMDBStorage;
use crate::error::{Error, Result};

/// Column-specific errors
#[derive(Debug)]
pub enum ColumnError {
    MissingColumn(String),
    SequenceNotFound(u64),
    InconsistentData(String),
}

impl From<ColumnError> for Error {
    fn from(err: ColumnError) -> Self {
        match err {
            ColumnError::MissingColumn(col) => 
                Error::Storage(format!("Missing column: {}", col)),
            ColumnError::SequenceNotFound(seq) => 
                Error::Storage(format!("Sequence {} not found", seq)),
            ColumnError::InconsistentData(msg) => 
                Error::Storage(format!("Data inconsistency: {}", msg)),
        }
    }
}

// ============================================================================
// CHANNEL MESSAGE
// ============================================================================

/// Messages sent from chunk processors to writer thread
#[derive(Debug)]
pub enum BufferedChunkMessage {
    /// Matches from a processed chunk
    Matches {
        chunk_index: usize,
        matches: Vec<ReconstructedSegment>,
    },
    
    /// All chunks done, shut down
    Shutdown,
}

// ============================================================================
// WRITER STATE
// ============================================================================

/// State of the writer thread (just the essentials)
pub struct WriterState {
    /// Current buffer of matches
    pub buffer: Vec<ReconstructedSegment>,
    
    /// Paths to temp files created
    pub temp_file_paths: Vec<PathBuf>,
    
    /// Time of last flush
    pub last_flush: Instant,
    
    /// Total matches written so far
    pub total_written: usize,
}

impl WriterState {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            temp_file_paths: Vec::new(),
            last_flush: Instant::now(),
            total_written: 0,
        }
    }
}

// ============================================================================
// RESULT TYPE
// ============================================================================

/// What the writer thread returns when done
#[derive(Debug)]
pub struct WriterResult {
    pub temp_file_paths: Vec<PathBuf>,
    pub total_matches_written: usize,
}

// In src/matcher/types.rs (or wherever your types are)

/// Generic representation of data that can be either in memory or on disk
/// Used to pass data between pipeline stages
#[derive(Debug, Clone)]
pub enum PipelineData<T> {
    InMemory(Vec<T>),
    OnDisk { 
        path: PathBuf, 
        count: usize 
    },
}

impl<T> PipelineData<T> {
    /// Get the count of items regardless of storage location
    pub fn count(&self) -> usize {
        match self {
            PipelineData::InMemory(data) => data.len(),
            PipelineData::OnDisk { count, .. } => *count,
        }
    }
    
    /// Check if data is in memory
    pub fn is_in_memory(&self) -> bool {
        matches!(self, PipelineData::InMemory(_))
    }
    
    /// Check if data is on disk
    pub fn is_on_disk(&self) -> bool {
        matches!(self, PipelineData::OnDisk { .. })
    }
    
    /// Get the path if on disk
    pub fn disk_path(&self) -> Option<&PathBuf> {
        match self {
            PipelineData::OnDisk { path, .. } => Some(path),
            PipelineData::InMemory(_) => None,
        }
    }
}