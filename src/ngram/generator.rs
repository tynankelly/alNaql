// src/ngram/generator.rs
//! Main NGram generator struct and orchestration logic
//! This module provides the primary interface for ngram generation while
//! maintaining backward compatibility with existing bin/ code.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::path::Path;
use ahash::AHashMap;
use log::{info, debug};

use crate::parser::{TextMapping, TextParser};
use crate::config::NGramType;
use crate::config::generation::GenerationConfig;
use crate::error::Result;

use super::types::{GenerationResult, MIN_PARALLEL_THRESHOLD};
use super::{sequential, parallel};

/// The main NGram generator that orchestrates all generation strategies
pub struct NGramGenerator<P: TextParser> {
    pub(super) parser: P,
    pub(super) config: GenerationConfig,
    pub(super) current_sequence: Arc<AtomicUsize>,
    pub(super) file_path: Option<String>,
    pub(super) ngram_frequencies: Option<AHashMap<u64, u32>>,
}

impl<P: TextParser + Sync + Send> NGramGenerator<P> {
    /// Create a new NGram generator with the given parser and configuration
    pub fn new(parser: P, config: GenerationConfig) -> Self {
               
        Self {
            parser,
            config: config.clone(),
            current_sequence: Arc::new(AtomicUsize::new(0)),
            file_path: None,
            ngram_frequencies: None,
        }
    }
    
    /// Set the file path for the current generation session
    /// Accepts &Path which is what bin/generate_ngrams.rs provides
    pub fn set_file_path(&mut self, path: &Path) {  // Just use Path directly
        let path_str = path.to_str().unwrap_or("unknown");
        self.file_path = Some(path_str.to_string());
        debug!("Set file path for NGramGenerator: {}", path_str);
    }
    
    /// Get the current sequence number
    pub fn current_sequence_number(&self) -> u64 {
        self.current_sequence.load(Ordering::Relaxed) as u64
    }
    
    /// Set the next sequence number (used when resuming from checkpoint)
    /// This method is required by bin/generate_ngrams.rs for checkpoint recovery
    pub fn set_next_sequence_number(&self, sequence: u64) {
        self.current_sequence.store(sequence as usize, Ordering::Relaxed);
        debug!("Set next sequence number to: {}", sequence);
    }
    
    /// Reset the sequence counter (useful for testing or new sessions)
    pub fn reset_sequence(&self) {
        self.current_sequence.store(0, Ordering::Relaxed);
    }
    
    // ===========================================================================
    // PUBLIC API - Maintains backward compatibility with bin/generate_ngrams.rs
    // ===========================================================================
    
    /// Main entry point for ngram generation - called by bin/generate_ngrams.rs
    /// 
    /// This method maintains the exact same signature expected by the binary,
    /// including the storage parameter (which we no longer use internally).
    pub fn generate_ngrams_with_position(
        &mut self, 
        text: &str, 
        absolute_position: usize,
        overlap_bytes: usize,
    ) -> Result<GenerationResult> {
        info!("\n=== Starting generate_ngrams_with_position ===");
        info!("Text length: {}, absolute position: {}, overlap bytes: {}", 
            text.len(), absolute_position, overlap_bytes);
        
        // Verify that we're starting from the right sequence
        let starting_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        info!("Starting ngram generation from sequence number: {}", starting_sequence);
        
        info!("NGram type: {}, Ngram size: {}", 
            self.config.ngram_type.as_str(),
            self.config.ngram_size);
        
        
        // Create text mapping - this was done here in the original position.rs
        let initial_mapping = self.parser.clean_text_with_mapping_absolute(text, absolute_position)?;
        
        // FIXED: Pass the mapping to generate_internal so it doesn't double-clean
        let result = self.generate_internal(
            &initial_mapping.cleaned_text, 
            absolute_position, 
            Some(&initial_mapping)
        )?;
        
        // Enhanced logging for the result (from original position.rs)
        match result.ngrams.len() {
            0 => info!("No ngrams generated from this chunk"),
            n => {
                info!("Successfully generated {} ngrams with absolute positions", n);
                
                // Log the first few ngrams with absolute positions
                info!("First 5 ngrams with absolute positions:");
                for (i, ngram) in result.ngrams.iter().take(5).enumerate() {
                    info!("  {}. seq={}, text='{}', pos={}..{}", 
                        i + 1,
                        ngram.sequence_number,
                        ngram.text_content.cleaned_text,
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte);
                }
            }
        }
        
        Ok(result)
    }
    // ===========================================================================
    // INTERNAL CLEAN ARCHITECTURE
    // ===========================================================================
    
    /// Internal generation method with clean architecture
    /// This is where the sequential/parallel decision is made
    fn generate_internal(
        &mut self,
        text: &str,
        absolute_position: usize,
        provided_mapping: Option<&TextMapping>
    ) -> Result<GenerationResult> {
        // Decision logic: Sequential vs Parallel
        // This was previously in parallel.rs::generate_ngrams_parallel
        let use_parallel = self.should_use_parallel(text);
        
        info!("Generation strategy decision: text_length={}, use_parallel={}, config.use_parallel={}", 
            text.len(), use_parallel, self.config.use_parallel);
        
        if use_parallel {
            info!("Using PARALLEL processing (text length {} >= threshold {})",
                text.len(), MIN_PARALLEL_THRESHOLD);
            
            // FIXED: Pass the provided_mapping to parallel::execute
            parallel::execute(self, text, absolute_position, provided_mapping)
        } else {
            info!("Using SEQUENTIAL processing (text length {} < threshold {} or disabled)",
                text.len(), MIN_PARALLEL_THRESHOLD);
            
            // FIXED: Pass the provided_mapping to sequential::execute
            sequential::execute(self, text, absolute_position, provided_mapping)
        }
    }

    
    /// Determine whether to use parallel processing
    /// Extracted from the decision logic in parallel.rs
    fn should_use_parallel(&self, text: &str) -> bool {
        // Check if parallel is enabled in config
        if !self.config.use_parallel {
            debug!("Parallel processing disabled in configuration");
            return false;
        }
        
        // Check if text is large enough for parallel processing
        let text_length = text.len();
        if text_length < MIN_PARALLEL_THRESHOLD {
            debug!("Text length {} below parallel threshold {}", 
                text_length, MIN_PARALLEL_THRESHOLD);
            return false;
        }
        
        // Check thread count configuration
        if self.config.thread_count < 2 {
            debug!("Thread count {} too low for parallel processing", 
                self.config.thread_count);
            return false;
        }
        
        true
    }
    
    /// Initialize frequency tracking for a new document
    pub fn start_document_frequency_tracking(&mut self) {
        self.ngram_frequencies = Some(AHashMap::new());
        debug!("Initialized frequency tracking for new document");
    }

    /// Get and clear frequency data (called when document is complete)
    pub fn take_document_frequencies(&mut self) -> Option<AHashMap<u64, u32>> {
        self.ngram_frequencies.take()
    }

    /// Get current frequency count (for debugging/monitoring)
    pub fn get_frequency_count(&self) -> usize {
        self.ngram_frequencies.as_ref().map(|f| f.len()).unwrap_or(0)
    }


}

/// Statistics about the generator's current state
#[derive(Debug, Clone)]
pub struct GeneratorStatistics {
    pub current_sequence: u64,
    pub file_path: Option<String>,
    pub ngram_type: NGramType,
    pub ngram_size: usize,
    pub features_enabled: bool,
}

// Make NGramGenerator thread-safe (required for parallel processing)
unsafe impl<P: TextParser + Sync + Send> Sync for NGramGenerator<P> {}
unsafe impl<P: TextParser + Sync + Send> Send for NGramGenerator<P> {}