// src/ngram/generator/word_ngrams.rs

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use log::{info, debug, trace};

use crate::error::Result;
use crate::types::{NGramEntry, TextPosition, NGramText};
use crate::parser::TextMapping;

use super::core::{NGramGenerator, TextWindow, PositionKey};
use super::position::GenerationResult;

impl<P: crate::parser::TextParser + Sync + Send> NGramGenerator<P> {
    pub(crate) fn generate_word_ngrams_with_deduplication(
        &self, 
        mapping: &TextMapping,  // Change from text: &str
        processed_positions: &HashSet<PositionKey>
    ) -> Result<GenerationResult> {
        trace!("\n=== Starting Word-based NGram Generation ===");
        trace!("NGram size: {} words", self.config.ngram_size);
    
        // Use tokenize_text_with_mapping to get tokens with absolute positions
        let tokens = self.parser.tokenize_text_with_mapping(mapping)?;
        let token_count = tokens.len();
        trace!("Text tokenized into {} words with absolute positions", token_count);
    
        if token_count < self.config.ngram_size {
            trace!("Text too short for ngram size {} (only {} words)",
                self.config.ngram_size, token_count);
            return Ok(GenerationResult {
                ngrams: Vec::new(),
                positions: HashSet::new(),
            });
        }
    
        let mut ngrams = Vec::new();
        let mut new_positions = HashSet::new();
        let total_windows = token_count.saturating_sub(self.config.ngram_size - 1);
        trace!("Will generate {} word-based ngrams", total_windows);
    
        // Pre-allocate vectors for better performance
        ngrams.reserve(total_windows);
        
        // Get the base sequence number from the atomic counter
        let base_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        let mut local_offset = 0; // Use as offset from base sequence
        
        // For each window position, create an ngram
        for window_start in (0..total_windows).step_by(self.config.stride) {
            let window_end = window_start + self.config.ngram_size;
            
            // Boundary check
            if window_end > tokens.len() {
                trace!("End of ngram exceeds token count. Breaking loop.");
                break;
            }
            
            // Get byte positions from tokens - ALREADY ABSOLUTE from tokenize_text_with_mapping
            let start_byte = tokens[window_start].start_byte;
            let end_byte = tokens[window_end - 1].end_byte;

            debug!("WORD_NGRAM_DEBUG: Token[{}] '{}' pos={}, Token[{}] '{}' pos={}", 
                window_start, tokens[window_start].cleaned_text, start_byte,
                window_end-1, tokens[window_end-1].cleaned_text, end_byte);
                       
            // Check if this position has already been processed
            let position_key = (start_byte, end_byte);
            if processed_positions.contains(&position_key) {
                trace!("Skipping duplicate ngram at position {}..{} (already processed)", 
                    start_byte, end_byte);
                continue;
            }
            
            // Build the original and cleaned text from tokens
            let mut original_text = String::new();
            let mut cleaned_text = String::new();
            
            for i in window_start..window_end {
                if i > window_start {
                    original_text.push(' ');
                    cleaned_text.push(' ');
                }
                original_text.push_str(&tokens[i].original_text);
                cleaned_text.push_str(&tokens[i].cleaned_text);
            }
            
            let line = tokens[window_start].line_number;
            
            // Use global base sequence + local offset for continuous numbering
            let sequence_number = base_sequence + local_offset;
            local_offset += 1;
            debug!("NGRAM CREATION (word): '{}', token_idx={}..{}, abs_pos={}..{}", 
            cleaned_text, window_start, window_end, start_byte, end_byte);
            
            let ngram = NGramEntry {
                sequence_number,
                text_position: TextPosition {
                    start_byte,
                    end_byte,
                    line_number: line,
                },
                text_content: NGramText {
                    original_slice: original_text,
                    cleaned_text: cleaned_text,
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
    
        info!("\n=== Word-based NGram Generation Complete ===");
        info!("Generated {} word-based ngrams", ngrams.len());
        
        if !ngrams.is_empty() {
            trace!("First ngram: '{}' at absolute position {}..{}",
                ngrams[0].text_content.cleaned_text,
                ngrams[0].text_position.start_byte,
                ngrams[0].text_position.end_byte);
    
            trace!("Last ngram: '{}' at absolute position {}..{}",
                ngrams.last().unwrap().text_content.cleaned_text,
                ngrams.last().unwrap().text_position.start_byte,
                ngrams.last().unwrap().text_position.end_byte);
        }
        
        Ok(GenerationResult {
            ngrams,
            positions: new_positions,
        })
    }
    
    pub(crate) fn generate_word_ngrams_for_window_with_temp_sequence(
        &self,
        window: &TextWindow,
        mapping: &TextMapping,
        processed_positions: &HashSet<PositionKey>,
        start_sequence: u64,
    ) -> Result<Vec<NGramEntry>> {
        trace!("\n=== Starting word-based ngram generation for window with position checking ===");
        
        // Get the relevant text for this window
        let window_text = &mapping.cleaned_text;
        
        // Tokenize the window text
        let tokens = self.parser.tokenize_text(window_text)?;
        trace!("Window text tokenized into {} words", tokens.len());
        
        // Filter tokens that are within this window's boundaries
        let window_tokens: Vec<_> = tokens.iter()
            .filter(|token| {
                token.start_byte >= window.window_start && 
                token.end_byte <= window.window_end
            })
            .collect();
        
        trace!("After filtering, {} tokens are within window bounds", window_tokens.len());
        
        if window_tokens.len() < self.config.ngram_size {
            trace!("Not enough tokens in window for ngram size {}", self.config.ngram_size);
            return Ok(Vec::new());
        }
        
        let mut ngrams = Vec::new();
        let total_windows = window_tokens.len().saturating_sub(self.config.ngram_size - 1);
        
        // Generate ngrams from token windows using temporary sequence numbers
        for window_start in (0..total_windows).step_by(self.config.stride) {
            let window_end = window_start + self.config.ngram_size;
            
            // Get byte positions from tokens with position offset applied
            let start_byte = window_tokens[window_start].start_byte;
            let end_byte = window_tokens[window_end - 1].end_byte;
            
            // Check if this position has already been processed
            let position_key = (start_byte, end_byte);
            if processed_positions.contains(&position_key) {
                trace!("Skipping duplicate ngram at position {}..{} (already processed in previous chunk)", 
                    start_byte, end_byte);
                continue;
            }
            
            // Build ngram text from tokens
            let mut original_text = String::new();
            let mut cleaned_text = String::new();
            
            for i in window_start..window_end {
                if i > window_start {
                    original_text.push(' ');
                    cleaned_text.push(' ');
                }
                original_text.push_str(&window_tokens[i].original_text);
                cleaned_text.push_str(&window_tokens[i].cleaned_text);
            }
            
            // Use temporary sequence number - no need to update global counter
            let sequence_number = start_sequence + window_start as u64;
            
            // Create ngram with correct position mapping
            let ngram = NGramEntry {
                sequence_number,
                text_position: TextPosition {
                    start_byte: start_byte,
                    end_byte: end_byte,
                    line_number: window_tokens[window_start].line_number,
                },
                text_content: NGramText {
                    original_slice: original_text,
                    cleaned_text: cleaned_text,
                },
                file_path: self.file_path.clone().unwrap_or_default(),
            };
            
            ngrams.push(ngram);
        }
        
        info!("Generated {} word-based ngrams from window", ngrams.len());
        Ok(ngrams)
    }
}