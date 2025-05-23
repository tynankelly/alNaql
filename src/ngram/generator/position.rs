// src/ngram/generator/position.rs

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use log::{info, debug, trace};

use crate::error::Result;
use crate::storage::StorageBackend;
use crate::types::NGramEntry;

use super::core::{NGramGenerator, PositionKey};
use crate::parser::TextParser;

// Enhanced result type that returns both ngrams and processed positions
pub struct GenerationResult {
    pub ngrams: Vec<NGramEntry>,
    pub positions: HashSet<PositionKey>,
}

impl<P: TextParser + Sync + Send> NGramGenerator<P> {
    pub fn generate_ngrams_with_position(
        &mut self, 
        text: &str, 
        absolute_position: usize,  // Absolute position in the file
        overlap_bytes: usize,
        storage: &mut dyn StorageBackend
    ) -> Result<GenerationResult> {
        info!("\n=== Starting generate_ngrams_with_position ===");
        info!("Text length: {}, absolute position: {}, overlap bytes: {}", 
            text.len(), absolute_position, overlap_bytes);
        
        // Enhanced debugging for the overlap region
        if overlap_bytes > 0 {
            let overlap_text = if text.len() >= overlap_bytes {
                &text[..overlap_bytes.min(text.len())]
            } else {
                text
            };
            
            info!("Overlap region details:");
            info!("  Size: {} bytes", overlap_bytes);
            info!("  Text (first 40 chars): \"{}\"", 
                overlap_text.chars().take(40).collect::<String>());
            info!("  Absolute file position: {}..{}", 
                absolute_position, absolute_position + overlap_bytes);
        }
        
        // Call text mapping
        let initial_mapping = self.parser.clean_text_with_mapping_absolute(text, absolute_position)?;
        
        debug!("POSITION_GEN_DEBUG: Generated mapping with {} chars and {} mappings", 
            initial_mapping.cleaned_text.len(), initial_mapping.char_map.len());
        debug!("POSITION_GEN_DEBUG: Cleaned text sample: \"{}\"", 
            initial_mapping.cleaned_text.chars().take(50).collect::<String>());
        // Call generate_ngrams_parallel with the mapping that contains absolute positions
        let result = self.generate_ngrams_parallel(&initial_mapping, storage)?;
        
        // Enhanced logging for the result
        match result.ngrams.len() {
            0 => info!("No ngrams generated from this chunk"),
            n => {
                info!("Successfully generated {} ngrams with absolute positions", n);
                
                // Verify positions of some ngrams
                // Log the first few ngrams with absolute positions
                info!("First 5 ngrams with absolute positions:");
                for (i, ngram) in result.ngrams.iter().take(5).enumerate() {
                    info!("  {}. Text: '{}', Absolute position: {}..{}", 
                        i+1,
                        ngram.text_content.cleaned_text,
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte);
                }
                
                // Log the last few ngrams with absolute positions
                if n > 5 {
                    info!("Last 5 ngrams with absolute positions:");
                    for (i, ngram) in result.ngrams.iter().skip(n - 5).enumerate() {
                        info!("  {}. Text: '{}', Absolute position: {}..{}", 
                            n - 5 + i + 1,
                            ngram.text_content.cleaned_text,
                            ngram.text_position.start_byte,
                            ngram.text_position.end_byte);
                    }
                }
            }
        }
        
        info!("=== Completed generate_ngrams_with_position ===\n");
        
        Ok(result)
    }

    pub(crate) fn resequence_ngrams_with_positions(
        &self, 
        mut ngrams: Vec<NGramEntry>, 
        processed_positions: &HashSet<PositionKey>
    ) -> Result<GenerationResult> {
        debug!("\n=== Starting ngram resequencing with absolute positions ===");
        let input_count = ngrams.len();
        
        // Get the current sequence number as our starting point
        let start_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        debug!("Resequencing starting from sequence number: {}", start_sequence);
        
        // 1. Sort by position in original text (positions should now all be absolute)
        ngrams.sort_by_key(|ng| ng.text_position.start_byte);
        
        let mut resequenced = Vec::with_capacity(ngrams.len());
        let mut new_sequence = start_sequence; // Start from our configured sequence
        
        // Track seen positions in a HashSet, including previously processed ones
        // Start with a copy of already processed positions
        let mut seen_positions = processed_positions.clone();
        
        // Count existing positions for logging
        let existing_positions_count = seen_positions.len();
        debug!("Starting with {} previously processed positions", existing_positions_count);
        
        // Track statistics about duplicates
        let mut new_duplicates = 0;
        let mut existing_duplicates = 0;
        
        // Debug: examine some positions from the processed set
        if !processed_positions.is_empty() && log::log_enabled!(log::Level::Debug) {
            let sample_count = 5.min(processed_positions.len());
            debug!("Sample of {} previously processed positions:", sample_count);
            for (i, pos) in processed_positions.iter().take(sample_count).enumerate() {
                debug!("  {}: {}..{}", i+1, pos.0, pos.1);
            }
        }
        
        // Resequence all ngrams, skipping any that are duplicates
        for mut ngram in ngrams {
            let current_position = (ngram.text_position.start_byte, ngram.text_position.end_byte);
            
            // Check if this position was already in the processed set
            if processed_positions.contains(&current_position) {
                existing_duplicates += 1;
                trace!("Skipping duplicate ngram '{}' at position {}..{} (already processed in previous chunk)", 
                    ngram.text_content.cleaned_text,
                    current_position.0, current_position.1);
                continue;
            }
            
            // Check if we've seen this position before in this batch
            if seen_positions.contains(&current_position) {
                new_duplicates += 1;
                trace!("Skipping duplicate ngram '{}' at position {}..{} (duplicate position within this chunk)", 
                    ngram.text_content.cleaned_text,
                    current_position.0, current_position.1);
                continue;
            }
            
            // Debug logging for a few ngrams to verify absolute positions
            // Must do this BEFORE we move text_position out of ngram
            if resequenced.len() <= 3 || resequenced.len() % 10000 == 0 {
                debug!("Resequencing ngram {} - '{}' at absolute position {}..{}", 
                    new_sequence,
                    ngram.text_content.cleaned_text,
                    ngram.text_position.start_byte,
                    ngram.text_position.end_byte);
            }
            
            // Create new ngram with correct sequence number
            // Just update the sequence number in place rather than creating a new struct
            ngram.sequence_number = new_sequence;
        
            resequenced.push(ngram);
            new_sequence += 1;
            seen_positions.insert(current_position);
        }
    
        // CRITICAL: Update atomic sequence counter to the next available sequence
        // This ensures the next chunk starts with the correct sequence number
        self.current_sequence.store(new_sequence as usize, Ordering::Relaxed);
        
        // Log detailed statistics
        debug!("Resequencing details:");
        debug!("  Start sequence number: {}", start_sequence);
        debug!("  Next sequence number: {}", new_sequence);
        debug!("  Sequence range used: {}", new_sequence - start_sequence);
        
        info!("Resequencing complete:");
        info!("  Input ngrams: {}", input_count);
        info!("  Output ngrams: {}", resequenced.len());
        info!("  Duplicates skipped (previously processed): {}", existing_duplicates);
        info!("  Duplicates removed (within this chunk): {}", new_duplicates);
        info!("  Total duplicates: {}", existing_duplicates + new_duplicates);
        info!("  NGram type: {}", self.config.ngram_type.as_str());
        info!("  Next sequence number: {}", new_sequence);
        
        if !resequenced.is_empty() {
            // Log first and last ngram for verification of coordinate system
            let first = &resequenced[0];
            let last = resequenced.last().unwrap();
            
            debug!("First ngram after resequencing: '{}' at position {}..{} (seq: {})",
                first.text_content.cleaned_text,
                first.text_position.start_byte,
                first.text_position.end_byte,
                first.sequence_number);
                
            debug!("Last ngram after resequencing: '{}' at position {}..{} (seq: {})",
                last.text_content.cleaned_text,
                last.text_position.start_byte,
                last.text_position.end_byte,
                last.sequence_number);
        }
        
        // Return both resequenced ngrams and updated positions
        Ok(GenerationResult {
            ngrams: resequenced,
            positions: seen_positions,
        })
    }
}