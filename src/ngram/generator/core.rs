// src/ngram/generator/core.rs

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashSet;
use log::debug;

use crate::parser::TextParser;
use crate::config::subsystems::generator::GeneratorConfig;

// Type for tracking processed byte positions
pub type PositionKey = (usize, usize); // (start_byte, end_byte)

// Constants for performance tuning
pub const MIN_PARALLEL_THRESHOLD: usize = 500_000;

pub struct NGramGenerator<P: TextParser> {
   pub(crate) parser: P,
   pub(crate) config: GeneratorConfig,
   pub(crate) current_sequence: Arc<AtomicUsize>,
   pub(crate) file_path: Option<String>,
   pub(crate) processed_positions: Option<HashSet<PositionKey>>,
}

// Structures to handle our parallel processing
#[derive(Debug, Clone)]
pub(crate) struct ChunkBoundary {
    pub start: usize,           
    pub end: usize,             
    pub overlap_start: usize,   
    pub overlap_end: usize,     
    pub start_line: usize,
    pub end_line: usize,
}

pub(crate) struct TextWindow<'a> {
    pub content: &'a str,
    pub main_content_start: usize,
    pub main_content_end: usize,
    pub window_start: usize,
    pub window_end: usize,
    pub abs_window_start: usize,
    pub abs_window_end: usize,
    pub abs_content_start: usize,
    pub abs_content_end: usize,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug)]
pub(crate) struct TextScanInfo {
    // Track XML tag positions
    pub tag_positions: Vec<(usize, usize)>,  // (start, end) byte positions of XML tags
    
    // Track probable "safe" splitting points
    pub safe_points: Vec<usize>,  // Byte positions where splitting is safe
    
    // Track text density information
    pub arabic_char_count: usize,
    pub total_char_count: usize,
    
    // Track line numbers for positioning
    pub line_breaks: Vec<usize>,  // Byte positions of line breaks
    
    // Track word boundaries for word-based ngrams
    pub word_boundaries: Vec<usize>, // Byte positions of word boundaries
}

impl<P: TextParser + Sync + Send> NGramGenerator<P> {
    pub fn new(parser: P, config: GeneratorConfig) -> Self {
        Self {
            parser,
            config,
            current_sequence: Arc::new(AtomicUsize::new(1)),
            file_path: None,
            processed_positions: None,
        }
    }

    // New method to set processed positions from previous chunks
    pub fn set_processed_positions(&mut self, positions: HashSet<PositionKey>) {
        self.processed_positions = Some(positions);
    }

    // New method to get processed positions
    pub fn get_processed_positions(&self) -> Option<&HashSet<PositionKey>> {
        self.processed_positions.as_ref()
    }

    pub fn set_file_path<T: AsRef<Path>>(&mut self, path: T) {
        self.file_path = Some(path.as_ref().to_string_lossy().to_string());
    }

    pub fn reset_sequence(&mut self) {
        self.current_sequence.store(1, Ordering::Relaxed);
    }

    /// Set the next sequence number to use for new ngrams
    pub fn set_next_sequence_number(&mut self, value: u64) {
        self.current_sequence.store(value as usize, Ordering::Relaxed);
    }
    
    pub fn ensure_minimum_sequence(&self, proposed_sequence: u64) -> u64 {
        let current_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        if proposed_sequence <= current_sequence {
            // If the proposed sequence is less than or equal to current max, use the next number
            debug!("Sequence safety check: proposed {} <= current {}. Using next available: {}",
                   proposed_sequence, current_sequence, current_sequence + 1);
            current_sequence + 1
        } else {
            proposed_sequence
        }
    }
}