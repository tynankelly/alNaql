// src/ngram/character.rs
//! Character-based ngram generation logic extracted from char_ngrams.rs
//! This maintains the exact same logic, just reorganized for cleaner architecture

use ahash::{AHashSet, AHashMap};
use std::hash::Hasher;
use std::sync::atomic::Ordering;
use log::{info, debug, trace, warn};

use crate::error::Result;
use crate::types::{NGramEntry, TextPosition, NGramText};
use crate::parser::{TextParser, TextMapping};

use super::types::{GenerationResult, TextWindow};
use super::generator::NGramGenerator;
use crate::ngram::features::generate_arabic_char_frequencies;

/// Generate character ngrams using sequential processing
pub fn generate_sequential<P: TextParser + Sync + Send>(
    generator: &NGramGenerator<P>,
    mapping: &TextMapping,
) -> Result<GenerationResult> {
    // Original character-based implementation modified for position tracking and sequence continuity
    trace!("\n=== Starting Character-based NGram Generation ===");
    trace!("NGram size: {} characters", generator.config.ngram_size);

    let mut local_frequencies: AHashMap<u64, u32> = AHashMap::new();
    let chars: Vec<char> = mapping.cleaned_text.chars().collect();
    let char_count = chars.len();
    trace!("Character count: {}", char_count);
    trace!("Character map length: {}", mapping.char_map.len());
    
    if char_count < generator.config.ngram_size {
        trace!("Text too short for ngram size {} (length: {})",
            generator.config.ngram_size, char_count);
        return Ok(GenerationResult {
            ngrams: Vec::new(),
            positions: AHashSet::new(),
            frequencies: AHashMap::new(),
        });
    }
    
    let mut ngrams = Vec::new();
    let total_windows = char_count.saturating_sub(generator.config.ngram_size - 1);
    trace!("Will generate {} character-based ngrams", total_windows);
    
    // Pre-allocate vectors for better performance
    ngrams.reserve(total_windows);
    let mut text_buffer = String::with_capacity(generator.config.ngram_size * 4); // Pre-allocate string buffer
    
    // Get the base sequence number from the atomic counter
    let base_sequence = generator.current_sequence.load(Ordering::Relaxed) as u64;
    let mut local_offset = 0; // Use as offset from base sequence
    
    for window_start in (0..total_windows).step_by(generator.config.stride) {
        let window_end = window_start + generator.config.ngram_size;
        
        // Boundary check
        if window_end > mapping.char_map.len() {
            warn!("End of ngram exceeds character map. Breaking loop.");
            break;
        }
        
        // Get the byte positions in original text - NOW ALREADY ABSOLUTE POSITIONS
        let start_byte = mapping.char_map[window_start].0;
        let end_byte = mapping.char_map[window_end - 1].1;
                
        // Reuse text buffer for better memory efficiency
        text_buffer.clear();
        text_buffer.extend(chars[window_start..window_end].iter());
        
        let line = mapping.line_map[window_start];
        
        // Use global base sequence + local offset for continuous numbering
        let sequence_number = base_sequence + local_offset;
        local_offset += 1;
        
        // After building the ngram text string
        let ngram_text: String = chars[window_start..window_end].iter().collect();
        
        // Compute similarity features
        let char_frequencies = generate_arabic_char_frequencies(&ngram_text);
        
        let ngram = NGramEntry {
            sequence_number,
            text_position: TextPosition {
                start_byte,
                end_byte,
                line_number: line,
            },
            text_content: NGramText {
                original_slice: ngram_text.clone(),
                cleaned_text: ngram_text.clone(),
            },
            file_path: generator.file_path.clone().unwrap_or_default(),
            char_frequencies,
        };
        
        ngrams.push(ngram);
        // Track frequency locally
        {
            use std::hash::Hash;
            let mut hasher = ahash::AHasher::default();
            ngram_text.hash(&mut hasher);  // Use Hash trait instead of write
            let text_hash = hasher.finish();
            
            *local_frequencies.entry(text_hash).or_insert(0) += 1;
        }
    }
    
    // Update the sequence counter for next use
    if local_offset > 0 {
        generator.current_sequence.store((base_sequence + local_offset) as usize, Ordering::Relaxed);
    }
    
    info!("\n=== Character-based NGram Generation Complete ===");
    info!("Generated {} character-based ngrams", ngrams.len());
    
    Ok(GenerationResult {
        ngrams,
        positions: AHashSet::new(),
        frequencies: local_frequencies,
    })
}

/// Generate character ngrams from a window for parallel processing
/// Extracted from char_ngrams.rs::char_ngrams_parallel
pub fn generate_from_window<P: TextParser + Sync + Send>(
    generator: &NGramGenerator<P>,
    window: &TextWindow,
    mapping: &TextMapping,
    start_sequence: u64,
) -> Result<Vec<NGramEntry>> {
    trace!("\n=== Starting character-based ngram generation for window with temp sequences ===");
    debug!("Window relative bounds: {}..{}", window.window_start, window.window_end);
    debug!("Window absolute bounds: {}..{}", window.abs_window_start, window.abs_window_end);
    trace!("Main content: {}..{}", window.main_content_start, window.main_content_end);
    
    let chars: Vec<char> = mapping.cleaned_text.chars().collect();
    let char_count = chars.len();
    
    if char_count < generator.config.ngram_size {
        warn!("Window text too short for ngrams ({} < {})", 
            char_count, generator.config.ngram_size);
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
    
    while pos + generator.config.ngram_size <= char_count {
        let window_end = pos + generator.config.ngram_size;
        // Check if this ngram would exceed window bounds
        let end_pos = mapping.char_map[window_end - 1].1;
        if end_pos > window.window_end {
            break;
        }
        
        let start_in_window = mapping.char_map[pos].0;
        let end_in_window = mapping.char_map[window_end - 1].1;
                
        let text: String = chars[pos..window_end].iter().collect();
        
        // Use temporary sequence number - no need to update global counter
        let sequence_number = start_sequence + local_offset;
        local_offset += 1;
        
        // Compute similarity features before creating the ngram
        let char_frequencies = generate_arabic_char_frequencies(&text);
        
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
            file_path: generator.file_path.clone().unwrap_or_default(),
            char_frequencies,
        };
        
        
        ngrams.push(ngram);
        pos += generator.config.stride;
    }
    
    debug!("Generated {} character-based ngrams from window", ngrams.len());
    Ok(ngrams)
}