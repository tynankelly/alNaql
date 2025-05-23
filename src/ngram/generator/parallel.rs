// src/ngram/generator/parallel.rs

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use log::{info, debug, warn};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::storage::StorageBackend;
use crate::types::NGramEntry;
use crate::parser::TextMapping;
use crate::config::subsystems::generator::NGramType;

use super::core::{NGramGenerator, MIN_PARALLEL_THRESHOLD};
use super::position::GenerationResult;

impl<P: crate::parser::TextParser + Sync + Send> NGramGenerator<P> {
    pub fn generate_ngrams_parallel(&mut self, mapping: &TextMapping, _storage: &mut dyn StorageBackend) -> Result<GenerationResult> {
        info!("\nStarting ngram generation for text of length {}", mapping.cleaned_text.len());
        
        // Verify that we're starting from the right sequence
        let starting_sequence = self.current_sequence.load(Ordering::Relaxed) as u64;
        info!("Starting ngram generation from sequence number: {}", starting_sequence);
            
        info!("Ngram type: {}, Ngram size: {}", 
            self.config.ngram_type.as_str(),
            self.config.ngram_size);
    
        // Initialize processed positions set if not already set
        let mut positions = self.processed_positions.take().unwrap_or_else(HashSet::new);
        
        let ngrams = if !self.config.use_parallel {
            // Handle sequential processing case - just generate ngrams, no prefix handling
            debug!("Using sequential processing as configured in default.ini");
            let result = self.generate_ngrams_with_deduplication(mapping, &positions)?;
            
            // Add the newly generated positions to our tracking set
            positions.extend(result.positions);
            
            // Add sampling debug output
            info!("Sampling 10 ngrams for debugging:");
            let sample_size = 10.min(result.ngrams.len());
            for i in 0..sample_size {
                let ngram = &result.ngrams[i];
                info!("  Ngram {}: seq={}, text='{}', position={}..{}", 
                    i+1, 
                    ngram.sequence_number,
                    ngram.text_content.cleaned_text,
                    ngram.text_position.start_byte,
                    ngram.text_position.end_byte);
                }
            
            result.ngrams
        } else {
            // Parallel processing path
            if mapping.cleaned_text.len() >= MIN_PARALLEL_THRESHOLD {
                info!("Text length {} meets parallel threshold {}. Using parallel processing.", 
                    mapping.cleaned_text.len(), MIN_PARALLEL_THRESHOLD);
    
                // Analyze text structure for safe splitting
                let scan_info = self.pre_scan_text(&mapping.cleaned_text)?;
                debug!("Completed text pre-scan analysis");
    
                // Create chunk boundaries
                let boundaries = self.calculate_chunk_boundaries(mapping.cleaned_text.len(), &scan_info)?;
                if boundaries.is_empty() {
                    return Err(Error::text("Failed to create valid chunk boundaries"));
                }
                info!("Created {} chunk boundaries", boundaries.len());
    
                // Create windows
                let windows = self.create_overlapping_windows(&mapping.cleaned_text, &boundaries, &scan_info, mapping)?;
                if windows.is_empty() {
                    return Err(Error::text("Failed to create valid text windows"));
                }
                info!("Created {} windows for parallel processing", windows.len());
    
                // Get a thread-safe reference to processed positions for parallel processing
                let positions_ref = Arc::new(positions);
                
                info!("BEFORE windows processing, sequence is: {}", 
                    self.current_sequence.load(Ordering::Relaxed));
    
                // Process windows in parallel with position checking
                let raw_ngram_results: Vec<Result<Vec<NGramEntry>>> = windows.par_iter()
                    .map(|window| {
                        // Use process_window that checks against processed positions
                        let positions_clone = Arc::clone(&positions_ref);
                        let result = self.process_window_with_deduplication(
                            window, 
                            mapping, 
                            &mapping.cleaned_text,
                            &positions_clone
                        );
                        
                        match result {
                            Ok(window_ngrams) => {
                                // Validate each window's ngrams
                                match self.validate_ngram_integrity(&window_ngrams, mapping) {
                                    Ok(_) => {
                                        info!("Window {}..{} generated {} valid ngrams",
                                            window.window_start,
                                            window.window_end,
                                            window_ngrams.len());
                                        Ok(window_ngrams)
                                    },
                                    Err(e) => {
                                        warn!("Window ngram validation failed: {}", e);
                                        Ok(Vec::new())
                                    }
                                }
                            },
                            Err(e) => {
                                warn!("Error processing window: {}", e);
                                Ok(Vec::new())
                            }
                        }
                    })
                    .collect();
                info!("AFTER windows processing, sequence is: {}", 
                    self.current_sequence.load(Ordering::Relaxed));   
                
                // Convert results to ngrams, handling any errors
                let raw_ngrams: Vec<NGramEntry> = raw_ngram_results.into_iter()
                    .filter_map(|r| r.ok())
                    .flatten()
                    .collect();
    
                info!("Parallel processing complete, {} total raw ngrams", raw_ngrams.len());
    
                // Add sampling debug output after flattening
                info!("Sampling 10 raw ngrams from parallel processing:");
                let sample_size = 10.min(raw_ngrams.len());
                for i in 0..sample_size {
                    let ngram = &raw_ngrams[i];
                    info!("  Raw Ngram {}: seq={}, text='{}', position={}..{}", 
                        i+1, 
                        ngram.sequence_number,
                        ngram.text_content.cleaned_text,
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte);
                }
    
                // After collecting all raw_ngrams and before resequencing
                info!("BEFORE resequencing, sequence is: {}", 
                    self.current_sequence.load(Ordering::Relaxed));
    
                // Reset sequence counter to the correct starting value
                self.current_sequence.store(starting_sequence as usize, Ordering::Relaxed);
                
                
                // Sort and resequence ngrams, taking into account processed positions from Arc
                let resequenced = self.resequence_ngrams_with_positions(raw_ngrams, positions_ref.as_ref())?;
                
                // Update positions set with newly processed positions
                positions = resequenced.positions;
                
                info!("Resequencing complete, {} ngrams after deduplication", resequenced.ngrams.len());
    
                // Final validation
                self.validate_ngram_integrity(&resequenced.ngrams, mapping)?;
                debug!("Final validation passed");
    
                resequenced.ngrams
            } else {
                // Handle short text case
                info!("Text length {} below parallel threshold {}. Using sequential processing.", 
                    mapping.cleaned_text.len(), MIN_PARALLEL_THRESHOLD);
                let result = self.generate_ngrams_with_deduplication(mapping, &positions)?;
                
                // Add the newly generated positions to our tracking set
                positions.extend(result.positions);
                
                // Add sampling debug output
                info!("Sampling 10 ngrams for debugging (short text):");
                let sample_size = 10.min(result.ngrams.len());
                for i in 0..sample_size {
                    let ngram = &result.ngrams[i];
                    info!("  Ngram {}: seq={}, text='{}', position={}..{}", 
                        i+1, 
                        ngram.sequence_number,
                        ngram.text_content.cleaned_text,
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte);
                }
                
                result.ngrams
            }
        };
    
        // Log summary information
        if !ngrams.is_empty() {
            info!("First ngram: \"{}\" (sequence: {}, position: {}..{})",
                ngrams[0].text_content.cleaned_text,
                ngrams[0].sequence_number,
                ngrams[0].text_position.start_byte,
                ngrams[0].text_position.end_byte);
    
            info!("Last ngram: \"{}\" (sequence: {}, position: {}..{})",
                ngrams.last().unwrap().text_content.cleaned_text,
                ngrams.last().unwrap().sequence_number,
                ngrams.last().unwrap().text_position.start_byte,
                ngrams.last().unwrap().text_position.end_byte);
        }
    
        if !ngrams.is_empty() {
            debug!("\nFinal ngram details:");
            for ngram in &ngrams {
                debug!(
                    "Text: '{}', Sequence: {}, Bytes: {}..{}", 
                    ngram.text_content.cleaned_text,
                    ngram.sequence_number,
                    ngram.text_position.start_byte,
                    ngram.text_position.end_byte
                );
            }
        }
        
        // Store the updated processed positions back into the struct
        self.processed_positions = Some(positions.clone());
        
        info!("Generated {} total ngrams", ngrams.len());
        
        // Return both the ngrams and the updated positions
        Ok(GenerationResult {
            ngrams,
            positions,
        })
    }

    pub(crate) fn validate_ngram_integrity(&self, ngrams: &[NGramEntry], mapping: &TextMapping) -> Result<()> {
        if ngrams.is_empty() {
            return Ok(());
        }
    
        // Find the maximum byte position in the mapping
        let max_byte_position = mapping.char_map.iter()
            .map(|(_, end)| *end)
            .max()
            .unwrap_or(0);
    
        let mut prev_sequence = None;
    
        for ngram in ngrams {
            // Check bounds against the actual byte positions
            if ngram.text_position.end_byte > max_byte_position {
                // For word-based ngrams, the end position might be beyond the last char mapping
                // because word tokens can extend to whitespace not included in character mapping
                if let NGramType::Word = self.config.ngram_type {
                    // For word mode, just warn but don't fail validation
                    debug!(
                        "Word-based NGram position {}..{} extends beyond mapped characters (max: {})",
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte,
                        max_byte_position
                    );
                } else {
                    // For character mode, this is a serious error
                    return Err(Error::text(format!(
                        "NGram position {}..{} exceeds text bounds (max: {})", 
                        ngram.text_position.start_byte,
                        ngram.text_position.end_byte,
                        max_byte_position
                    )));
                }
            }
    
            // Sequence continuity check remains the same regardless of ngram type
            if let Some(prev_seq) = prev_sequence {
                if ngram.sequence_number != prev_seq + 1 {
                    return Err(Error::text(format!(
                        "Sequence discontinuity: {} -> {}", 
                        prev_seq,
                        ngram.sequence_number
                    )));
                }
            }
            prev_sequence = Some(ngram.sequence_number);
        }
    
        Ok(())
    }

    pub(crate) fn pre_scan_text(&self, text: &str) -> Result<super::core::TextScanInfo> {
        let mut info = super::core::TextScanInfo {
            tag_positions: Vec::new(),
            safe_points: Vec::new(),
            arabic_char_count: 0,
            total_char_count: 0,
            line_breaks: Vec::new(),
            word_boundaries: Vec::new(), // Add this new field to track word boundaries
        };
        
        let mut in_tag = false;
        let mut tag_start = 0;
        let mut last_safe_point = 0;
        
        // Add word boundary detection if we're using word-based ngrams
        if let NGramType::Word = self.config.ngram_type {
            if let Ok(tokens) = self.parser.tokenize_text(text) {
                for token in &tokens {
                    // Add token boundaries to safe points
                    info.word_boundaries.push(token.start_byte);
                    info.word_boundaries.push(token.end_byte);
                }
                debug!("Found {} word boundaries", info.word_boundaries.len());
            }
        }
        
        // Use char_indices() to get both byte positions and characters
        for (byte_pos, c) in text.char_indices() {
            info.total_char_count += 1;
            
            match c {
                '<' => {
                    in_tag = true;
                    tag_start = byte_pos;
                    
                    // Before a tag is often a safe splitting point
                    if byte_pos > last_safe_point + self.config.ngram_size {
                        info.safe_points.push(byte_pos);
                        last_safe_point = byte_pos;
                    }
                },
                '>' if in_tag => {
                    in_tag = false;
                    info.tag_positions.push((tag_start, byte_pos + 1));
                },
                '\n' => {
                    info.line_breaks.push(byte_pos);
                },
                _ if !in_tag => {
                    // Count Arabic characters using existing parser logic
                    if self.parser.is_countable_char(c) {
                        info.arabic_char_count += 1;
                    }
                    
                    // Consider space after Arabic text as a safe point
                    if c.is_whitespace() && 
                       byte_pos > last_safe_point + self.config.ngram_size {
                        info.safe_points.push(byte_pos);
                        last_safe_point = byte_pos;
                    }
                },
                _ => {}
            }
        }
        
        // For word mode, add word boundaries to safe points
        if let NGramType::Word = self.config.ngram_type {
            info.safe_points.extend(&info.word_boundaries);
            // Sort and deduplicate safe points
            info.safe_points.sort_unstable();
            info.safe_points.dedup();
        }
        
        debug!("Pre-scan complete:");
        debug!("Found {} XML tags", info.tag_positions.len());
        debug!("Identified {} safe split points", info.safe_points.len());
        debug!("Arabic character density: {:.2}%", 
            (info.arabic_char_count as f64 / info.total_char_count as f64) * 100.0);
            
        Ok(info)
    }

    pub(crate) fn find_nearest_safe_point(&self, 
        desired_pos: usize, 
        scan_info: &super::core::TextScanInfo, 
        text: &str) -> usize {
       
        // Early return for end of text
        if desired_pos >= text.len() {
            return text.len();
        }
    
        // Check if already a valid char boundary
        if text.is_char_boundary(desired_pos) {
            // Even if it's a valid boundary, try to find safe point
        }
    
        // First check if we're in the middle of an XML tag
        for (tag_start, tag_end) in &scan_info.tag_positions {
            if desired_pos > *tag_start && desired_pos < *tag_end {
                return *tag_end;  // Skip past the tag
            }
        }
        
        // For word-based ngrams, try to find a nearby word boundary
        if let NGramType::Word = self.config.ngram_type {
            // Look for word boundaries near desired position
            if !scan_info.word_boundaries.is_empty() {
                // Try to find a word boundary after the desired position
                if let Some(boundary) = scan_info.word_boundaries
                    .iter()
                    .find(|&&pos| pos >= desired_pos) {
                    return *boundary;
                }
                
                // If no boundary after, use the last boundary before
                if let Some(boundary) = scan_info.word_boundaries
                    .iter()
                    .rev()
                    .find(|&&pos| pos < desired_pos) {
                    return *boundary;
                }
            }
        }
        
        // Try to find safe point
        if let Some(safe_point) = scan_info.safe_points.iter()
            .find(|&&pos| pos >= desired_pos) {
            return super::chunking::find_char_boundary(text, *safe_point);
        }
    
        // If no safe point found, find next valid UTF-8 boundary
        let mut pos = desired_pos;
        while pos < text.len() {
            if text.is_char_boundary(pos) {
                return pos;
            }
            pos += 1;
        }
        
        // If we reach here, return last valid UTF-8 boundary
        let mut pos = desired_pos;
        while pos > 0 {
            if text.is_char_boundary(pos) {
                return pos;
            }
            pos -= 1;
        }
        
        // If all else fails, return 0
        0
    }
}