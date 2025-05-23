// src/ngram/generator/chunking.rs

use std::collections::HashSet;
use log::{info, debug};

use crate::error::{Error, Result};
use crate::types::NGramEntry;
use crate::parser::TextMapping;
use crate::config::subsystems::generator::NGramType;
use crate::ngram::generator::PositionKey;


use super::core::{NGramGenerator, ChunkBoundary, TextWindow, TextScanInfo};

impl<P: crate::parser::TextParser + Sync + Send> NGramGenerator<P> {
    pub(crate) fn calculate_chunk_boundaries(&self, total_length: usize, scan_info: &TextScanInfo) -> Result<Vec<ChunkBoundary>> {
        let chunk_size = self.calculate_optimal_chunk_size(
            total_length,
            scan_info.arabic_char_count as f64 / scan_info.total_char_count as f64
        )?;
         
        debug!("Calculated optimal chunk size: {} bytes", chunk_size);
         
        let mut boundaries: Vec<ChunkBoundary> = Vec::new();
        let mut current_pos = 0;
        
        // Calculate overlap size based on ngram type - use larger values for safety
        let overlap_size = match self.config.ngram_type {
            NGramType::Character => self.config.ngram_size * 4,  // 4x ngram size for character mode
            NGramType::Word => {
                // For word-based ngrams, ensure overlap covers complete word sequences
                // Use larger overlap to ensure no word sequences are split
                self.config.ngram_size * 30  // Significantly larger for word mode
            }
        };
        
        debug!("Using overlap size of {} for {} ngrams", 
              overlap_size, self.config.ngram_type.as_str());
         
        while current_pos < total_length {
            let desired_end = (current_pos + chunk_size).min(total_length);
            let mut actual_end = desired_end;
            
            // Check if we're in the middle of an XML tag
            for (tag_start, tag_end) in &scan_info.tag_positions {
                if actual_end > *tag_start && actual_end < *tag_end {
                    // Move end point to before the tag
                    actual_end = *tag_start;
                    debug!("Adjusted end point to avoid splitting XML tag: {} -> {}", 
                        desired_end, actual_end);
                    break;
                }
            }
            
            // For word-based ngrams, ensure we split at word boundaries
            if let NGramType::Word = self.config.ngram_type {
                // Find the nearest word boundary before our desired end point
                if let Some(&boundary) = scan_info.word_boundaries.iter()
                    .rev()
                    .find(|&&pos| pos <= actual_end) {
                    actual_end = boundary;
                    debug!("Adjusted end point to word boundary: {}", actual_end);
                }
            }
    
            if actual_end == 0 || actual_end <= current_pos {
                debug!("No more valid splitting points found");
                if let Some(last_boundary) = boundaries.last_mut() {
                    last_boundary.end = total_length;
                    last_boundary.overlap_end = total_length;
                }
                break;
            }
         
            // Determine if this is the final chunk
            let is_final_chunk = actual_end >= total_length;
            
            // Calculate overlap regions
            let overlap_start = if current_pos == 0 {
                0  // First window starts at 0
            } else if is_final_chunk {
                // For final chunk, force backward search for overlap
                debug!("Final chunk - searching backward for overlap start");
                if let Some(&boundary) = scan_info.word_boundaries.iter()
                    .rev()
                    .find(|&&pos| pos < current_pos) {
                    debug!("Found backward safe point at {}", boundary);
                    boundary
                } else {
                    debug!("No backward safe point found, using fallback");
                    current_pos.saturating_sub(overlap_size)
                }
            } else {
                let proposed_start = current_pos.saturating_sub(overlap_size);
                
                // Find nearest word boundary for overlap start if using word ngrams
                if let NGramType::Word = self.config.ngram_type {
                    if let Some(&boundary) = scan_info.word_boundaries.iter()
                        .rev()
                        .find(|&&pos| pos <= proposed_start) {
                        boundary
                    } else {
                        proposed_start
                    }
                } else {
                    proposed_start
                }
            };
         
            let overlap_end = if is_final_chunk {
                total_length  // Final chunk ends at text end
            } else {
                let proposed_end = actual_end + overlap_size;
                
                // Find nearest word boundary for overlap end if using word ngrams
                if let NGramType::Word = self.config.ngram_type {
                    if let Some(&boundary) = scan_info.word_boundaries.iter()
                        .find(|&&pos| pos >= proposed_end) {
                        boundary
                    } else {
                        proposed_end.min(total_length)
                    }
                } else {
                    proposed_end.min(total_length)
                }
            };
         
            let start_line = scan_info.line_breaks.partition_point(|&pos| pos < current_pos);
            let end_line = scan_info.line_breaks.partition_point(|&pos| pos < actual_end);
         
            debug!("Creating chunk boundary: {}..{} (overlap: {}..{})", 
                current_pos, actual_end, overlap_start, overlap_end);
         
            boundaries.push(ChunkBoundary {
                start: current_pos,
                end: actual_end,
                overlap_start,
                overlap_end,
                start_line,
                end_line,
            });
         
            if actual_end >= total_length { break; }
            current_pos = actual_end;
        }
         
        // Special handling to ensure sufficient overlap for all boundaries
        if boundaries.len() > 1 {
            debug!("\nChecking and adjusting boundary overlaps");
            
            // Calculate minimum required overlap based on ngram type
            let min_overlap = match self.config.ngram_type {
                NGramType::Character => self.config.ngram_size * 2,  // Double the ngram size for safety
                NGramType::Word => self.config.ngram_size * 4,       // Quadruple for word-based ngrams
            };
            
            // Iterate through boundaries and check overlaps
            for i in 1..boundaries.len() {
                // Use split_at_mut to get two non-overlapping slices
                let (left, right) = boundaries.split_at_mut(i);
                let prev = &left[i-1];
                let current = &mut right[0];  // First element of right is boundaries[i]
                
                // Calculate actual overlap
                let actual_overlap = if current.overlap_start <= prev.end {
                    prev.end - current.overlap_start
                } else {
                    0
                };
                
                debug!("Boundary {} overlap check: current={}, minimum={}", 
                    i, actual_overlap, min_overlap);
                
                // If overlap is insufficient, adjust
                if actual_overlap < min_overlap {
                    let old_start = current.overlap_start;
                    current.overlap_start = prev.end.saturating_sub(min_overlap);
                    
                    debug!("Adjusted boundary {} overlap_start: {} → {} (overlap now: {})", 
                        i, old_start, current.overlap_start, 
                        prev.end - current.overlap_start);
                }
            }
        }
        debug!("Created {} chunk boundaries", boundaries.len());
        
        Ok(boundaries)
    }
    
    
    pub(crate) fn calculate_optimal_chunk_size(&self, total_length: usize, content_density: f64) -> Result<usize> {
        const MIN_CHUNK_SIZE: usize = 64 * 1024;          // 64KB minimum for multi-chunk
        const MAX_CHUNK_SIZE: usize = 1024 * 1024;        // 1MB maximum
        const DENSITY_SCALING_FACTOR: f64 = 0.7;
    
        debug!("Chunk size calculation:");
        debug!("  Total length: {} bytes", total_length);
        debug!("  Content density: {:.2}%", content_density * 100.0);
    
        // If single chunk and not using parallel processing use the length, otherwise split
        if total_length < MIN_CHUNK_SIZE {
            debug!("  File smaller than {} bytes", MIN_CHUNK_SIZE);
            debug!("  Using file length as chunk size");
            return Ok(total_length);
        }
    
        // For larger files, calculate optimal chunk size
        let base_chunk_size = total_length / 8;  // Target 8 chunks for large files
        let density_factor = 1.0 - (content_density * DENSITY_SCALING_FACTOR);
        let adjusted_size = (base_chunk_size as f64 * density_factor) as usize;
        let chunk_size = adjusted_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
    
        debug!("  Base size: {} bytes", base_chunk_size);
        debug!("  Density adjustment: {:.2}x", density_factor);
        debug!("  Final size: {} bytes", chunk_size);
        debug!("  Estimated chunks: {}", (total_length + chunk_size - 1) / chunk_size);
    
        Ok(chunk_size)
    }
    
    pub(crate) fn create_overlapping_windows<'a>(
        &self,
        text: &'a str,
        boundaries: &[ChunkBoundary],
        scan_info: &TextScanInfo,
        mapping: &TextMapping  // NEW: Add mapping parameter
    ) -> Result<Vec<TextWindow<'a>>> {
        let mut windows = Vec::with_capacity(boundaries.len());
        
        debug!("\n=== Creating Windows ===");
        debug!("Total text length: {}", text.len());
        debug!("Number of boundaries: {}", boundaries.len());
        debug!("NGram type: {}", self.config.ngram_type.as_str());
    
        // Single window case - skip overlap checks
        if boundaries.len() == 1 || !self.is_suitable_for_multi_window(text.len()) {
            debug!("Single window mode - text size ({} bytes) or boundary count ({})", 
                   text.len(), boundaries.len());
            let boundary = &boundaries[0];
            
            // Validate XML tag boundaries
            for (tag_start, tag_end) in &scan_info.tag_positions {
                if (boundary.start > *tag_start && boundary.start < *tag_end) ||
                   (boundary.end > *tag_start && boundary.end < *tag_end) {
                    return Err(Error::text(format!(
                        "Window boundary {}..{} splits XML tag {}..{}", 
                        boundary.start, boundary.end,
                        tag_start, tag_end
                    )));
                }
            }
    
            // Find absolute positions for the entire text
            let abs_start = self.find_absolute_position(0, mapping);
            let abs_end = if !mapping.char_map.is_empty() {
                mapping.char_map.last().map_or(0, |(_, end)| *end)
            } else {
                0
            };
            
            // Create single window covering entire text
            debug!("Created single window:");
            debug!("  Relative content range: 0..{}", text.len());
            debug!("  Absolute content range: {}..{}", abs_start, abs_end);
            debug!("  Lines: {}..{}", boundary.start_line, boundary.end_line);
            
            let window = TextWindow {
                content: text,
                main_content_start: 0,
                main_content_end: text.len(),
                window_start: 0,
                window_end: text.len(),
                abs_window_start: abs_start,        // NEW
                abs_window_end: abs_end,            // NEW
                abs_content_start: abs_start,       // NEW
                abs_content_end: abs_end,           // NEW
                start_line: boundary.start_line,
                end_line: boundary.end_line,
            };
            windows.push(window);
            return Ok(windows);
        }
    
        // Multi-window case - first adjust boundaries to ensure sufficient overlap
        let adjusted_boundaries = if boundaries.len() > 1 {
            let mut boundaries_copy = boundaries.to_vec();
            
            // Calculate minimum required overlap based on ngram type
            let min_overlap = match self.config.ngram_type {
                NGramType::Character => self.config.ngram_size * 2, // Double minimum for safety
                NGramType::Word => self.config.ngram_size * 3,      // Triple for word ngrams
            };
            
            debug!("Minimum required overlap: {} units", min_overlap);
            
            // For each pair of adjacent boundaries, ensure adequate overlap
            for i in 0..boundaries_copy.len()-1 {
                // Use split_at_mut to get two non-overlapping slices
                let (left, right) = boundaries_copy.split_at_mut(i+1);
                let curr = &left[i];
                let next = &mut right[0];  // First element of right is boundaries_copy[i+1]
                
                // Calculate current overlap
                let current_overlap = if next.overlap_start <= curr.end {
                    curr.end - next.overlap_start
                } else {
                    0
                };
                
                if current_overlap < min_overlap {
                    debug!("Insufficient overlap between boundaries {} and {}: {} < {}", 
                           i, i+1, current_overlap, min_overlap);
                    
                    // Adjust overlap_start of next boundary to ensure minimum overlap
                    next.overlap_start = curr.end.saturating_sub(min_overlap);
                    debug!("Adjusted overlap_start to {} (new overlap: {})", 
                           next.overlap_start, curr.end - next.overlap_start);
                }
            }
            
            boundaries_copy
        } else {
            boundaries.to_vec()
        };
        
        debug!("Multi-window mode - creating overlapping windows");
        for boundary in &adjusted_boundaries {
            debug!("\nProcessing boundary:");
            debug!("  Boundary range: {}..{}", boundary.start, boundary.end);
            debug!("  Overlap range: {}..{}", boundary.overlap_start, boundary.overlap_end);
            debug!("  Lines: {}..{}", boundary.start_line, boundary.end_line);
    
            // Validate XML tag boundaries
            for (tag_start, tag_end) in &scan_info.tag_positions {
                if (boundary.start > *tag_start && boundary.start < *tag_end) ||
                   (boundary.end > *tag_start && boundary.end < *tag_end) {
                    return Err(Error::text(format!(
                        "Window boundary {}..{} splits XML tag {}..{}", 
                        boundary.start, boundary.end,
                        tag_start, tag_end
                    )));
                }
            }
            
            // For word-based ngrams, check if boundaries split words
            if let NGramType::Word = self.config.ngram_type {
                // Check if any boundary point is in the middle of a word
                for point in [boundary.start, boundary.end, 
                              boundary.overlap_start, boundary.overlap_end] {
                    // Skip text start/end
                    if point == 0 || point == text.len() {
                        continue;
                    }
                    
                    // Check if this point is a word boundary
                    let is_word_boundary = scan_info.word_boundaries
                        .binary_search(&point)
                        .is_ok();
                        
                    if !is_word_boundary && point > 0 && point < text.len() {
                        // Look for nearest word boundary
                        let nearest = self.find_nearest_safe_point(point, scan_info, text);
                        debug!("Point {} is not a word boundary, nearest is {}", point, nearest);
                    }
                }
            }
    
            // Find character-aligned boundaries
            let window_start = find_char_boundary(text, boundary.overlap_start);
            let window_end = find_char_boundary(text, boundary.overlap_end);
            let content_start = find_char_boundary(text, boundary.start);
            let content_end = find_char_boundary(text, boundary.end);
    
            debug!("  After char boundary alignment:");
            debug!("    Window: {}..{}", window_start, window_end);
            debug!("    Content: {}..{}", content_start, content_end);
    
            // NEW: Find corresponding absolute positions
            let abs_window_start = self.find_absolute_position(window_start, mapping);
            let abs_window_end = self.find_absolute_position(window_end, mapping);
            let abs_content_start = self.find_absolute_position(content_start, mapping);
            let abs_content_end = self.find_absolute_position(content_end, mapping);
            
            debug!("  Corresponding absolute positions:");
            debug!("    Window: {}..{}", abs_window_start, abs_window_end);
            debug!("    Content: {}..{}", abs_content_start, abs_content_end);
    
            // Extract window content
            let window_content = &text[window_start..window_end];
                
            // Verify overlap size
            let start_overlap = content_start.saturating_sub(window_start);
            let end_overlap = window_end.saturating_sub(content_end);
            let total_overlap = start_overlap + end_overlap;
    
            debug!("  Overlap verification:");
            debug!("    Start overlap: {}", start_overlap);
            debug!("    End overlap: {}", end_overlap);
            debug!("    Total overlap: {}", total_overlap);
    
            // Calculate minimum required overlap based on ngram type
            let min_overlap = match self.config.ngram_type {
                NGramType::Character => self.config.ngram_size,
                NGramType::Word => {
                    // For word-based ngrams, require more overlap
                    self.config.ngram_size * 2
                }
            };
    
            // Skip overlap validation for start of first window and end of last window
            if window_start == 0 {
                // First window - only check end overlap
                if end_overlap < min_overlap {
                    return Err(Error::text(format!(
                        "Insufficient end overlap size {} for first window {}..{} (need {})", 
                        end_overlap, window_start, window_end, min_overlap
                    )));
                }
            } else if window_end == text.len() {
                // Last window - only check start overlap
                // Add more detailed diagnostics for this critical error
                if start_overlap < min_overlap {
                    debug!("ERROR: Final window overlap check failed:");
                    debug!("  Window bounds: {}..{}", window_start, window_end);
                    debug!("  Content bounds: {}..{}", content_start, content_end);
                    debug!("  Start overlap: {} (need: {})", start_overlap, min_overlap);
                    debug!("  Boundary overlap_start: {}", boundary.overlap_start);
                    debug!("  Text length: {}", text.len());
                    
                    return Err(Error::text(format!(
                        "Insufficient start overlap size {} for final window {}..{} (need {})", 
                        start_overlap, window_start, window_end, min_overlap
                    )));
                }
            } else {
                // Middle windows - check total overlap
                if total_overlap < min_overlap {
                    return Err(Error::text(format!(
                        "Insufficient overlap size {} for window {}..{} (need {})", 
                        total_overlap, window_start, window_end, min_overlap
                    )));
                }
            }
    
            let window = TextWindow {
                content: window_content,
                main_content_start: content_start,
                main_content_end: content_end,
                window_start,
                window_end,
                abs_window_start,     // NEW
                abs_window_end,       // NEW
                abs_content_start,    // NEW
                abs_content_end,      // NEW
                start_line: boundary.start_line,
                end_line: boundary.end_line,
            };
    
            info!("  Created window:");
            info!("    Relative range: {}..{}", window.window_start, window.window_end);
            info!("    Absolute range: {}..{}", window.abs_window_start, window.abs_window_end);
            info!("    Main content (relative): {}..{}", window.main_content_start, window.main_content_end);
            info!("    Main content (absolute): {}..{}", window.abs_content_start, window.abs_content_end);
            info!("    Content size: {}", window.content.len());
            
            windows.push(window);
        }
    
        // Validate window set
        if !windows.is_empty() {
            debug!("\nValidating complete window set:");
            
            let first = &windows[0];
            let last = windows.last().unwrap();
            
            debug!("  First window starts at: {} (abs: {})", first.window_start, first.abs_window_start);
            debug!("  Last window ends at: {} (abs: {})", last.window_end, last.abs_window_end);
            debug!("  Total text length: {}", text.len());
            
            if first.window_start != 0 {
                return Err(Error::text(format!(
                    "First window doesn't start at beginning of text: {}", 
                    first.window_start
                )));
            }
            
            if last.window_end != text.len() {
                return Err(Error::text(format!(
                    "Last window doesn't reach end of text: {} != {}", 
                    last.window_end, 
                    text.len()
                )));
            }
    
            // Verify window connectivity and overlaps
            for (i, pair) in windows.windows(2).enumerate() {
                let current = &pair[0];
                let next = &pair[1];
                
                debug!("\n  Checking window connection {}/{}", i + 1, windows.len() - 1);
                debug!("    Current window end: {} (abs: {})", current.main_content_end, current.abs_content_end);
                debug!("    Next window start: {} (abs: {})", next.main_content_start, next.abs_content_start);
                
                if current.main_content_end != next.main_content_start {
                    return Err(Error::text(format!(
                        "Gap between window content: {} to {}", 
                        current.main_content_end, 
                        next.main_content_start
                    )));
                }
                
                // Calculate minimum required overlap based on ngram type
                let min_overlap = match self.config.ngram_type {
                    NGramType::Character => self.config.ngram_size,
                    NGramType::Word => self.config.ngram_size * 2,
                };
                
                let overlap_size = current.window_end.saturating_sub(next.window_start);
                
                debug!("    Window overlap size: {}", overlap_size);
                debug!("    Minimum required overlap: {}", min_overlap);
                
                if overlap_size < min_overlap {
                    return Err(Error::text(format!(
                        "Insufficient overlap between windows: {} < {}", 
                        overlap_size, 
                        min_overlap
                    )));
                }
            }
        }
    
        Ok(windows)
    }
    
    // Helper function to find absolute position from relative position in the cleaned text
    fn find_absolute_position(&self, relative_pos: usize, mapping: &TextMapping) -> usize {
        // Case 1: Position is at the beginning of text
        if relative_pos == 0 {
            let result = mapping.char_map.first().map_or(0, |(start, _)| *start);
            debug!("POSITION MAPPING: relative {} -> absolute {}", relative_pos, result);
            return result;
        }
        
        // Case 2: Position is beyond all mapped characters
        if relative_pos >= mapping.char_map.len() {
            let result = mapping.char_map.last().map_or(0, |(_, end)| *end);
            debug!("POSITION MAPPING: relative {} -> absolute {}", relative_pos, result);
            return result;
        }
        
        // Case 3: Position is directly at a mapped character
        if relative_pos < mapping.char_map.len() {
            let result = mapping.char_map[relative_pos].0;
            debug!("POSITION MAPPING: relative {} -> absolute {}", relative_pos, result);
            return result;
        }   
        
        // Case 4: Position is between mapped characters (shouldn't typically happen with proper alignment)
        // Find the nearest mapped position
        let mut closest_idx = 0;
        let mut closest_dist = usize::MAX;
        
        for (idx, (start, _)) in mapping.char_map.iter().enumerate() {
            let dist = if *start <= relative_pos {
                relative_pos - *start
            } else {
                *start - relative_pos
            };
            
            if dist < closest_dist {
                closest_dist = dist;
                closest_idx = idx;
            }
        }
        
        // Return the start position of the closest character
        let result = mapping.char_map[closest_idx].0;
        debug!("POSITION MAPPING (closest): relative {} -> absolute {}", relative_pos, result);
        result
    }
    
    // Helper function to check if text size is suitable for multi-window processing
    pub(crate) fn is_suitable_for_multi_window(&self, text_len: usize) -> bool {
        let min_size = match self.config.ngram_type {
            NGramType::Character => self.config.ngram_size * 5,
            NGramType::Word => self.config.ngram_size * 30, // Words need more space
        };
        
        text_len >= min_size
    }

    pub(crate) fn process_window_with_deduplication(
        &self,
        window: &TextWindow,
        mapping: &TextMapping,
        original_text: &str,
        processed_positions: &HashSet<PositionKey>,
    ) -> Result<Vec<NGramEntry>> {
        debug!("\n=== Starting window processing with position checking ===");
        debug!("Window bounds: {}..{}", window.window_start, window.window_end);
        debug!("Main content: {}..{}", window.main_content_start, window.main_content_end);
        debug!("Lines: {}..{}", window.start_line, window.end_line);
        
        // Initial validation - ensure window content matches original text
        let expected_content = &original_text[window.window_start..window.window_end];
        if window.content != expected_content {
            debug!("Window content mismatch!");
            debug!("Expected content length: {}", expected_content.len());
            debug!("Actual content length: {}", window.content.len());
            debug!("Expected content prefix: {:?}", &expected_content[..64.min(expected_content.len())]);
            debug!("Actual content prefix: {:?}", &window.content[..64.min(window.content.len())]);
            return Err(Error::text(format!(
                "Window content mismatch at position {}..{}", 
                window.window_start, 
                window.window_end
            )));
        }
        
        debug!("Window content validation passed");
        info!("CHUNK-DEBUG: Processing window {}..{} (main content: {}..{})",
            window.window_start, window.window_end,
            window.main_content_start, window.main_content_end);
        
        // Use temporary sequence numbers based on window position to avoid conflicts
        let temp_start_seq = (window.window_start as u64) * 1000000;
        
        // Generate ngrams for window - this will be different depending on ngram type
        let window_ngrams = if let NGramType::Word = self.config.ngram_type {
            self.generate_word_ngrams_for_window_with_temp_sequence(
                window, mapping, processed_positions, temp_start_seq)?
        } else {
            self.generate_ngrams_for_window_with_temp_sequence(
                window, mapping, processed_positions, temp_start_seq)?
        };
        
        info!("Generated {} initial ngrams", window_ngrams.len());
        
        // Validate positions
        let mut valid_ngrams = Vec::new();
        
        for (i, ngram) in window_ngrams.iter().enumerate() {
            // Get the original position from the ngram - ALREADY ABSOLUTE
            let start_byte = ngram.text_position.start_byte;
            let end_byte = ngram.text_position.end_byte;
            
            // Create a new ngram with the already properly positioned values
            let adjusted_ngram = NGramEntry {
                sequence_number: ngram.sequence_number,
                text_position: crate::types::TextPosition {
                    start_byte,
                    end_byte,
                    line_number: ngram.text_position.line_number,
                },
                text_content: ngram.text_content.clone(),
                file_path: ngram.file_path.clone(),
            };
            
            // Validate positions against window's ABSOLUTE bounds
            if adjusted_ngram.text_position.start_byte < window.abs_window_start ||
            adjusted_ngram.text_position.end_byte > window.abs_window_end {
            info!("ERROR: NGram position outside window bounds:");
            info!("  NGram absolute: {}..{}", 
                adjusted_ngram.text_position.start_byte,
                adjusted_ngram.text_position.end_byte);
            info!("  Window absolute: {}..{}", 
                window.abs_window_start, 
                window.abs_window_end);
            info!("  NGram index: {}", i);
            info!("  Sequence: {}", adjusted_ngram.sequence_number);
            continue;
            }
            
            valid_ngrams.push(adjusted_ngram);
        }
        
        info!("CHUNK-DEBUG: Window {}..{} generated {} ngrams",
            window.window_start, window.window_end, valid_ngrams.len());
        Ok(valid_ngrams)
    }
}

// Static helper function moved to module level
pub(crate) fn find_char_boundary(text: &str, byte_pos: usize) -> usize {
    // First try the exact position
    if text.is_char_boundary(byte_pos) {
        return byte_pos;
    }
    
    // Search forward for next boundary
    let mut pos = byte_pos;
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    
    // If we couldn't find forward boundary, search backward
    if pos >= text.len() {
        pos = byte_pos;
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
    }
    
    pos
}