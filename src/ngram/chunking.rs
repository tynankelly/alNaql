// src/ngram/chunking.rs
//! Text chunking and boundary detection for parallel processing

use log::{info, debug};

use crate::error::{Error, Result};
use crate::parser::TextParser;
use crate::config::NGramType;

use super::types::{ChunkBoundary, TextScanInfo, TextWindow};
use super::utils;

/// Scan text to identify safe splitting points and characteristics
pub fn scan_text<P: TextParser>(
    text: &str,
    parser: &P,
    ngram_type: NGramType,
) -> Result<TextScanInfo> {
    let mut info = TextScanInfo {
        tag_positions: Vec::new(),
        safe_points: Vec::new(),
        arabic_char_count: 0,
        total_char_count: 0,
        line_breaks: Vec::new(),
        word_boundaries: Vec::new(),
    };
    
    let mut in_tag = false;
    let mut tag_start = 0;
    
    // Scan for word boundaries if using word-based ngrams
    if let NGramType::Word = ngram_type {
        if let Ok(tokens) = parser.tokenize_text(text) {
            for token in &tokens {
                info.word_boundaries.push(token.start_byte);
                info.word_boundaries.push(token.end_byte);
            }
            debug!("Found {} word boundaries", info.word_boundaries.len());
        }
    }
    
    // Scan character by character for other features
    for (byte_pos, ch) in text.char_indices() {
        info.total_char_count += 1;
        
        // Count Arabic characters
        if ('\u{0600}'..='\u{06FF}').contains(&ch) || 
           ('\u{0750}'..='\u{077F}').contains(&ch) {
            info.arabic_char_count += 1;
        }
        
        // Track line breaks
        if ch == '\n' {
            info.line_breaks.push(byte_pos);
        }
        
        // Track XML tags
        if ch == '<' {
            in_tag = true;
            tag_start = byte_pos;
        } else if ch == '>' && in_tag {
            info.tag_positions.push((tag_start, byte_pos + 1));
            in_tag = false;
            
            // Mark position after tag as safe
            if byte_pos + 1 < text.len() {
                info.safe_points.push(byte_pos + 1);
            }
        }
        
        // Mark certain punctuation as safe split points
        if matches!(ch, '.' | '!' | '?' | '\n') && !in_tag {
            if byte_pos + 1 < text.len() {
                info.safe_points.push(byte_pos + 1);
            }
        }
    }
    
    Ok(info)
}

/// Calculate chunk boundaries for parallel processing
// src/ngram/chunking.rs - Around line 80-120

pub fn calculate_boundaries(
    total_length: usize,
    scan_info: &TextScanInfo,
    ngram_type: NGramType,
    ngram_size: usize,
) -> Result<Vec<ChunkBoundary>> {
    // Calculate optimal chunk size
    let arabic_density = scan_info.arabic_char_count as f64 / scan_info.total_char_count.max(1) as f64;
    let chunk_size = utils::calculate_optimal_chunk_size(total_length, arabic_density);
    
    debug!("Calculated optimal chunk size: {} bytes", chunk_size);
    
    // Calculate overlap size
    let overlap_size = utils::calculate_overlap_size(ngram_type, ngram_size);
    debug!("Using overlap size of {} bytes", overlap_size);
    
    let mut boundaries = Vec::new();
    let mut current_pos = 0;
    
    while current_pos < total_length {
        let desired_end = (current_pos + chunk_size).min(total_length);
        let mut actual_end = desired_end;
        
        // Avoid splitting XML tags
        for (tag_start, tag_end) in &scan_info.tag_positions {
            if actual_end > *tag_start && actual_end < *tag_end {
                actual_end = *tag_start;
                debug!("Adjusted end to avoid XML tag: {} -> {}", desired_end, actual_end);
                break;
            }
        }
        
        // For word ngrams, align to word boundaries
        if let NGramType::Word = ngram_type {
            if let Some(&boundary) = scan_info.word_boundaries.iter()
                .rev()
                .find(|&&pos| pos <= actual_end && pos > current_pos) {  // Added check: pos > current_pos
                actual_end = boundary;
                debug!("Aligned to word boundary: {}", actual_end);
            } else {
                // No word boundary found in range, try to find one forward
                if let Some(&boundary) = scan_info.word_boundaries.iter()
                    .find(|&&pos| pos > current_pos && pos <= actual_end + 1000) {  // Look ahead up to 1000 bytes
                    actual_end = boundary;
                    debug!("Aligned to forward word boundary: {}", actual_end);
                } else {
                    // No word boundary available, stick with character boundary
                    debug!("No word boundary found, using character boundary at {}", actual_end);
                }
            }
        }
        
        // Find line numbers for this chunk
        let start_line = scan_info.line_breaks.iter()
            .take_while(|&&pos| pos < current_pos)
            .count();
        
        // Calculate overlap region
        let overlap_start = current_pos.saturating_sub(overlap_size);
        let overlap_end = (actual_end + overlap_size).min(total_length);
        
        boundaries.push(ChunkBoundary {
            start: current_pos,
            end: actual_end,
            overlap_start,
            overlap_end,
            start_line,
        });
        
        // CRITICAL FIX: Move to next chunk position
        current_pos = actual_end;
        
        // FIXED: Only break if we've reached the end of text or made no progress
        if current_pos >= total_length {
            break;  // We've covered all the text
        }
        
        if actual_end == 0 || actual_end <= boundaries.last().unwrap().start {
            // No progress made - this is an error condition
            return Err(Error::text(format!(
                "Failed to make progress in chunking at position {}", current_pos
            )));
        }
    }
    
    info!("Created {} chunk boundaries for {} bytes of text", boundaries.len(), total_length);
    Ok(boundaries)
}

/// Create a text window from a chunk boundary
pub fn create_window(
    text: &str,
    boundary: &ChunkBoundary,
    absolute_offset: usize,
) -> TextWindow {
    // Extract the window text with overlap
    let window_start = utils::find_char_boundary(text, boundary.overlap_start);
    let window_end = utils::find_char_boundary(text, boundary.overlap_end);
    
    // Main content boundaries (without overlap)
    let main_start = utils::find_char_boundary(text, boundary.start);
    let main_end = utils::find_char_boundary(text, boundary.end);
    
    TextWindow {
        main_content_start: main_start - window_start,
        main_content_end: main_end - window_start,
        window_start,
        window_end,
        abs_window_start: absolute_offset + window_start,
        abs_window_end: absolute_offset + window_end,
        abs_content_start: absolute_offset + main_start,
        abs_content_end: absolute_offset + main_end,
        start_line: boundary.start_line,
    }
}