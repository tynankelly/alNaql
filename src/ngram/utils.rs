// src/ngram/utils.rs
//! Utility functions for ngram generation

use crate::config::NGramType;

/// Find the nearest valid UTF-8 character boundary
pub fn find_char_boundary(text: &str, byte_pos: usize) -> usize {
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

/// Calculate required overlap size based on ngram configuration
pub fn calculate_overlap_size(ngram_type: NGramType, ngram_size: usize) -> usize {
    match ngram_type {
        NGramType::Character => ngram_size * 4,
        NGramType::Word => ngram_size * 30,
    }
}
/// Calculate the optimal chunk size based on text characteristics

pub fn calculate_optimal_chunk_size(
    total_length: usize,
    arabic_density: f64,
) -> usize {
    // For parallel processing within a file chunk, use reasonable sizes
    // These are chunks WITHIN the already-chunked file pieces
    let base_size = if total_length < 100_000 {
        50_000  // 50KB chunks for small texts
    } else if total_length < 1_000_000 {
        200_000  // 200KB chunks for medium texts (INCREASED)
    } else {
        400_000  // 400KB chunks for large texts (INCREASED)
    };
    
    // Adjust for Arabic text density (Arabic text needs smaller chunks)
    let adjusted_size = if arabic_density > 0.7 {
        (base_size as f64 * 0.75) as usize  // 75% of base instead of 50%
    } else if arabic_density > 0.3 {
        (base_size as f64 * 0.85) as usize  // 85% of base
    } else {
        base_size
    };
    
    // Ensure minimum chunk size
    adjusted_size.max(50_000)  // At least 50KB
}