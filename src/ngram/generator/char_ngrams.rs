// src/ngram/generator/char_ngrams.rs

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use log::{info, debug, trace, warn};

use crate::error::Result;
use crate::types::{NGramEntry, TextPosition, NGramText};
use crate::parser::TextMapping;
use crate::config::subsystems::generator::NGramType;

use super::core::{NGramGenerator, TextWindow, PositionKey};
use super::position::GenerationResult;

impl<P: crate::parser::TextParser + Sync + Send> NGramGenerator<P> {
    pub(crate) fn generate_ngrams_with_deduplication(
        &self, 
        mapping: &TextMapping,
        processed_positions: &HashSet<PositionKey>
    ) -> Result<GenerationResult> {
        // If we're using word-based ngrams, call the word-specific function
        if let NGramType::Word = self.config.ngram_type {
            // CRITICAL CHANGE: Pass the mapping, not just the cleaned text
            return self.generate_word_ngrams_with_deduplication(mapping, processed_positions);
        }
        
        // Original character-based implementation modified for position tracking and sequence continuity
        trace!("\n=== Starting Character-based NGram Generation ===");
        trace!("NGram size: {} characters", self.config.ngram_size);
        
        let chars: Vec<char> = mapping.cleaned_text.chars().collect();
        let char_count = chars.len();
        trace!("Character count: {}", char_count);
        trace!("Character map length: {}", mapping.char_map.len());
        
        if char_count < self.config.ngram_size {
            trace!("Text too short for ngram size {} (length: {})",
                self.config.ngram_size, char_count);
            return Ok(GenerationResult {
                ngrams: Vec::new(),
                positions: HashSet::new(),
            });
        }
        
        let mut ngrams = Vec::new();
        let mut new_positions = HashSet::new();
        let total_windows = char_count.saturating_sub(self.config.ngram_size - 1);
        trace!("Will generate {} character-based ngrams", total_windows);
        
        // Pre-allocate vectors for better performance
        ngrams.reserve(total_windows);
        let mut text_buffer = String::with_capacity(self.config.ngram_size * 4); // Pre-allocate string buffer
        
        // Get the base sequence number from the atomic counter
        let base_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        let mut local_offset = 0; // Use as offset from base sequence
        
        for window_start in (0..total_windows).step_by(self.config.stride) {
            let window_end = window_start + self.config.ngram_size;
        
            // Boundary check
            if window_end > mapping.char_map.len() {
                warn!("End of ngram exceeds character map. Breaking loop.");
                break;
            }
        
            // Get the byte positions in original text - NOW ALREADY ABSOLUTE POSITIONS
            let start_byte = mapping.char_map[window_start].0;
            let end_byte = mapping.char_map[window_end - 1].1;
        
            // Check if this position has already been processed
            let position_key = (start_byte, end_byte);
            if processed_positions.contains(&position_key) {
                trace!("Skipping duplicate ngram at position {}..{} (already processed)", 
                    start_byte, end_byte);
                continue;
            }
        
            // Reuse text buffer for better memory efficiency
            text_buffer.clear();
            text_buffer.extend(chars[window_start..window_end].iter());
            
            let line = mapping.line_map[window_start];
        
            // Use global base sequence + local offset for continuous numbering
            let sequence_number = base_sequence + local_offset;
            local_offset += 1;
        
            let ngram = NGramEntry {
                sequence_number,
                text_position: TextPosition {
                    start_byte,
                    end_byte,
                    line_number: line,
                },
                text_content: NGramText {
                    original_slice: text_buffer.clone(),
                    cleaned_text: text_buffer.clone(),
                },
                file_path: self.file_path.clone().unwrap_or_default(),
            };
        
            ngrams.push(ngram);
            new_positions.insert(position_key);
        }
        
        // Update the sequence counter for next use
        if local_offset > 0 {
            self.current_sequence.store((base_sequence + local_offset) as usize, Ordering::Relaxed);
        }
        
        info!("\n=== Character-based NGram Generation Complete ===");
        info!("Generated {} character-based ngrams", ngrams.len());
        
        Ok(GenerationResult {
            ngrams,
            positions: new_positions,
        })
    }

    pub(crate) fn generate_ngrams_for_window_with_temp_sequence(
        &self,
        window: &TextWindow,
        mapping: &TextMapping,
        processed_positions: &HashSet<PositionKey>,
        start_sequence: u64,
    ) -> Result<Vec<NGramEntry>> {
        trace!("\n=== Starting character-based ngram generation for window with temp sequences ===");
        debug!("Window relative bounds: {}..{}", window.window_start, window.window_end);
        debug!("Window absolute bounds: {}..{}", window.abs_window_start, window.abs_window_end);
        trace!("Main content: {}..{}", window.main_content_start, window.main_content_end);
    
        let chars: Vec<char> = mapping.cleaned_text.chars().collect();
        let char_count = chars.len();
    
        if char_count < self.config.ngram_size {
            warn!("Window text too short for ngrams ({} < {})", 
                char_count, self.config.ngram_size);
            return Ok(Vec::new());
        }
    
        let mut ngrams = Vec::new();
        // Find starting position in mapping that corresponds to window start
        let window_start_idx = mapping.char_map.iter()
            .position(|(start, _)| *start >= window.abs_window_start)
            .unwrap_or(0);
    
        // Only process up to where adding ngram_size more chars would exceed window end
        let mut pos = window_start_idx;
        let mut local_offset = 0;  // Track position for temporary sequence
        
        while pos + self.config.ngram_size <= char_count {
            let window_end = pos + self.config.ngram_size;
            // Check if this ngram would exceed window bounds
            let end_pos = mapping.char_map[window_end - 1].1;
            if end_pos > window.window_end {
                break;
            }
    
            let start_in_window = mapping.char_map[pos].0;
            let end_in_window = mapping.char_map[window_end - 1].1;
    
            // Check if this position has already been processed
            let position_key = (start_in_window, end_in_window);
            if processed_positions.contains(&position_key) {
                trace!("Skipping duplicate ngram at position {}..{} (already processed in previous chunk)", 
                    start_in_window, end_in_window);
                pos += self.config.stride;
                continue;
            }
    
            let text: String = chars[pos..window_end].iter().collect();
    
            // Use temporary sequence number - no need to update global counter
            let sequence_number = start_sequence + local_offset;
            local_offset += 1;
    
            let ngram = NGramEntry {
                sequence_number,
                text_position: TextPosition {
                    start_byte: start_in_window,
                    end_byte: end_in_window,
                    line_number: window.start_line + mapping.line_map[pos],
                },
                text_content: NGramText {
                    original_slice: text.clone(),
                    cleaned_text: text,
                },
                file_path: self.file_path.clone().unwrap_or_default(),
            };
    
            ngrams.push(ngram);
            pos += self.config.stride;
        }
    
        debug!("Generated {} character-based ngrams from window", ngrams.len());
        Ok(ngrams)
    }
}