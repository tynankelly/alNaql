// ngram/features.rs
//! Feature generation for SIMD-optimized Arabic text matching.
//! 
//! This module generates the character frequency arrays and metadata needed
//! for high-performance similarity comparisons using SIMD instructions.

use ahash::AHasher;
use std::hash::{Hash, Hasher};
use log::{trace, debug};

use crate::types::{CompactCharFrequencies, NGramMetadata};
use crate::error::{Error, Result};

// Import the character mapping
use super::arabic_char_mapping::{char_to_index, is_valid_arabic_char};

// ===========================================================================
// MAIN FEATURE GENERATION
// ===========================================================================

/// Generate SIMD-optimized character frequencies from cleaned Arabic text.
/// 
/// This function creates the fixed-size frequency array that can be processed
/// efficiently using SIMD instructions for similarity comparisons.
/// 
/// # Arguments
/// * `text` - Cleaned Arabic text (already normalized by the parser)
/// 
/// # Returns
/// A `CompactCharFrequencies` structure with:
/// - Character counts in a 64-byte aligned array
/// - Pre-computed metadata for fast filtering
/// - Magnitude squared for cosine similarity
pub fn generate_arabic_char_frequencies(text: &str) -> CompactCharFrequencies {
    let mut freqs = CompactCharFrequencies::new();
    
    // Early return for empty text
    if text.is_empty() {
        return freqs;
    }
    
    // Track which indices have been seen for unique count
    let mut seen_indices = [false; 64];
    
    // First pass: Count character occurrences
    for ch in text.chars() {
        // Get the index for this character
        if let Some(index) = char_to_index(ch) {
            let idx = index as usize;
            
            // Track unique characters
            if !seen_indices[idx] {
                seen_indices[idx] = true;
                freqs.unique_chars += 1;
            }
            
            // Increment count with saturation to prevent overflow
            freqs.counts[idx] = freqs.counts[idx].saturating_add(1);
            freqs.total_chars += 1;
            
            trace!("Character '{}' mapped to index {} (count: {})", 
                   ch, idx, freqs.counts[idx]);
        } else {
            // This shouldn't happen with properly cleaned Arabic text
            debug!("Warning: Character '{}' (U+{:04X}) not in Arabic character set", 
                   ch, ch as u32);
        }
    }
    
    // Second pass: Compute magnitude squared AND bitmap together
    let mut magnitude_sq = 0u64;
    let mut bitmap = 0u64;
    
    for i in 0..64 {
        let count = freqs.counts[i];
        if count > 0 {
            // Calculate magnitude component
            let count_u64 = count as u64;
            magnitude_sq += count_u64 * count_u64;
            
            // Set bit for this index
            bitmap |= 1u64 << i;
        }
    }
    
    // Store computed values
    freqs.magnitude_squared = magnitude_sq as f32;
    freqs.char_bitmap = bitmap;
    
    trace!("Generated frequencies: {} total chars, {} unique chars, magnitude²: {}, bitmap: {:064b}", 
           freqs.total_chars, freqs.unique_chars, freqs.magnitude_squared, freqs.char_bitmap);
    
    freqs
}

/// Generate metadata for n-gram storage and filtering.
/// 
/// This creates the metadata that will be stored in ngram_metadata_db
/// for quick filtering and exact match detection.
/// 
/// # Arguments
/// * `sequence` - The sequence number for this n-gram
/// * `text` - The cleaned text
/// 
/// # Returns
/// An `NGramMetadata` structure containing hash, length, and prefix bytes
pub fn generate_ngram_metadata(sequence: u64, text: &str) -> NGramMetadata {
    // Generate a hash for exact match detection
    let mut hasher = AHasher::default();
    text.hash(&mut hasher);
    let text_hash = hasher.finish();
    
    // Get the first 8 bytes for quick prefix comparison
    let text_bytes = text.as_bytes();
    let mut first_chars = [0u8; 8];
    let copy_len = text_bytes.len().min(8);
    first_chars[..copy_len].copy_from_slice(&text_bytes[..copy_len]);
    
    NGramMetadata {
        sequence_number: sequence,
        text_length: text.len() as u64,
        text_hash,
    }
}

/// Extract prefix bytes for indexing.
/// 
/// This extracts the prefix that will be stored in prefix_bytes_db
/// and used as a key in prefix_index_db for fast filtering.
/// 
/// # Arguments
/// * `text` - The cleaned text
/// * `depth` - Number of characters to extract (from config.prefix_index_depth)
/// 
/// # Returns
/// UTF-8 bytes of the prefix
pub fn extract_prefix_bytes(text: &str, depth: usize) -> Vec<u8> {
    // Take the first `depth` characters and convert to bytes
    let prefix: String = text.chars().take(depth).collect();
    prefix.into_bytes()
}

/// Generate all features needed for n-gram storage.
/// 
/// This is a convenience function that generates all the data needed
/// to store an n-gram in the columnar storage system.
/// 
/// # Arguments
/// * `sequence` - The sequence number
/// * `text` - The cleaned text
/// * `prefix_depth` - Prefix depth from configuration
/// 
/// # Returns
/// A tuple of (character_frequencies, metadata, prefix_bytes)
pub fn generate_all_features(
    sequence: u64,
    text: &str,
    prefix_depth: usize,
) -> (CompactCharFrequencies, NGramMetadata, Vec<u8>) {
    let char_frequencies = generate_arabic_char_frequencies(text);
    let metadata = generate_ngram_metadata(sequence, text);
    let prefix_bytes = extract_prefix_bytes(text, prefix_depth);
    
    (char_frequencies, metadata, prefix_bytes)
}

// ===========================================================================
// VALIDATION AND STATISTICS
// ===========================================================================

/// Validate that text only contains valid Arabic characters.
/// 
/// This is useful for debugging to ensure the parser is working correctly.
/// 
/// # Arguments
/// * `text` - Text to validate
/// 
/// # Returns
/// * `Ok(())` if all characters are valid
/// * `Err` with details about invalid characters
pub fn validate_arabic_text(text: &str) -> Result<()> {
    let mut invalid_chars = Vec::new();
    
    for ch in text.chars() {
        if !is_valid_arabic_char(ch) {
            invalid_chars.push(ch);
        }
    }
    
    if invalid_chars.is_empty() {
        Ok(())
    } else {
        Err(Error::TextProcessing(format!(
            "Text contains {} invalid characters: {:?} (first few shown)",
            invalid_chars.len(),
            &invalid_chars[..invalid_chars.len().min(5)]
        )))
    }
}

/// Calculate statistics about character distribution.
/// 
/// Useful for debugging and understanding the text characteristics.
pub fn calculate_char_stats(frequencies: &CompactCharFrequencies) -> CharStats {
    let mut max_count = 0u8;
    let mut most_common_index = 0usize;
    let mut total_squared = 0u64;
    
    for (idx, &count) in frequencies.counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            most_common_index = idx;
        }
        if count > 0 {
            total_squared += (count as u64) * (count as u64);
        }
    }
    
    CharStats {
        total_chars: frequencies.total_chars,
        unique_chars: frequencies.unique_chars,
        max_frequency: max_count,
        most_common_index: most_common_index as u8,
        entropy: calculate_entropy(&frequencies.counts, frequencies.total_chars),
        magnitude_squared: total_squared as f32,
    }
}

/// Calculate Shannon entropy of character distribution.
/// 
/// Higher entropy means more evenly distributed characters.
fn calculate_entropy(counts: &[u8; 64], total: u32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    
    let total_f32 = total as f32;
    let mut entropy = 0.0f32;
    
    for &count in counts {
        if count > 0 {
            let probability = (count as f32) / total_f32;
            entropy -= probability * probability.log2();
        }
    }
    
    entropy
}

/// Statistics about character distribution in text
#[derive(Debug, Clone)]
pub struct CharStats {
    pub total_chars: u32,
    pub unique_chars: u16,
    pub max_frequency: u8,
    pub most_common_index: u8,
    pub entropy: f32,
    pub magnitude_squared: f32,
}

// ===========================================================================
// TESTING UTILITIES
// ===========================================================================

#[cfg(test)]
pub mod test_utils {
    use super::*;
    
    /// Create a sample frequency array for testing
    pub fn create_test_frequencies(text: &str) -> CompactCharFrequencies {
        generate_arabic_char_frequencies(text)
    }
    
    /// Verify that two frequency arrays are equivalent
    pub fn frequencies_equal(a: &CompactCharFrequencies, b: &CompactCharFrequencies) -> bool {
        a.counts == b.counts &&
        a.total_chars == b.total_chars &&
        a.unique_chars == b.unique_chars &&
        (a.magnitude_squared - b.magnitude_squared).abs() < 0.001
    }
}