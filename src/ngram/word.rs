// src/ngram/word.rs
//! Word-based ngram generation logic extracted from word_ngrams.rs
//! This maintains the exact same logic, just reorganized for cleaner architecture

use ahash::{AHashSet, AHashMap};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use log::{info, debug, trace};

use crate::error::Result;
use crate::types::{NGramEntry, TextPosition, NGramText};
use crate::parser::{TextParser, TextMapping};

use super::types::{GenerationResult, TextWindow};
use super::generator::NGramGenerator;
use crate::ngram::features::generate_arabic_char_frequencies;

/// Generate word ngrams using sequential processing
/// Extracted from word_ngrams.rs::word_ngrams_sequential (lines 18-108)
pub fn generate_sequential<P: TextParser + Sync + Send>(
    generator: &NGramGenerator<P>,
    mapping: &TextMapping,
) -> Result<GenerationResult> {
    trace!("\n=== Starting Word-based NGram Generation ===");
    trace!("NGram size: {} words", generator.config.ngram_size);
    
    let mut local_frequencies: AHashMap<u64, u32> = AHashMap::new();

    // Use tokenize_text_with_mapping to get tokens with absolute positions
    let tokens = generator.parser.tokenize_text_with_mapping(mapping)?;
    let token_count = tokens.len();
    trace!("Text tokenized into {} words with absolute positions", token_count);
    
    if token_count < generator.config.ngram_size {
        trace!("Text too short for ngram size {} (only {} words)",
            generator.config.ngram_size, token_count);
        return Ok(GenerationResult {
            ngrams: Vec::new(),
            positions: AHashSet::new(),
            frequencies: AHashMap::new(),
        });
    }
    
    let mut ngrams = Vec::new();
    let total_windows = token_count.saturating_sub(generator.config.ngram_size - 1);
    trace!("Will generate {} word-based ngrams", total_windows);
    
    // Pre-allocate vectors for better performance
    ngrams.reserve(total_windows);
    
    // Get the base sequence number from the atomic counter
    let base_sequence = generator.current_sequence.load(Ordering::Relaxed) as u64;
    let mut local_offset = 0; // Use as offset from base sequence
    
    // For each window position, create an ngram
    for window_start in (0..total_windows).step_by(generator.config.stride) {
        let window_end = window_start + generator.config.ngram_size;
        
        // Boundary check
        if window_end > tokens.len() {
            trace!("End of ngram exceeds token count. Breaking loop.");
            break;
        }
        
        // Get byte positions from tokens - ALREADY ABSOLUTE from tokenize_text_with_mapping
        let start_byte = tokens[window_start].start_byte;
        let end_byte = tokens[window_end - 1].end_byte;
        
        // Build ngram text from tokens - exactly as original
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
        
        let char_frequencies = generate_arabic_char_frequencies(&cleaned_text);
        
        let ngram = NGramEntry {
            sequence_number,
            text_position: TextPosition {
                start_byte,
                end_byte,
                line_number: line,
            },
            text_content: NGramText {
                original_slice: original_text,
                cleaned_text: cleaned_text.clone(),
            },
            file_path: generator.file_path.clone().unwrap_or_default(),
            char_frequencies,
        };
        info!("NGRAM: seq={}, start={}, end={}, span={}, cleaned='{}'", 
            sequence_number, 
            start_byte, 
            end_byte, 
            end_byte - start_byte,
            cleaned_text
        );
          
        ngrams.push(ngram);
        {
            let mut hasher = ahash::AHasher::default();  // This IS AHash!
            cleaned_text.hash(&mut hasher);  // Using Hash trait with AHash hasher
            let text_hash = hasher.finish();
            
            *local_frequencies.entry(text_hash).or_insert(0) += 1;
        }
    }
    
    // Update the sequence counter for next use
    if local_offset > 0 {
        generator.current_sequence.store((base_sequence + local_offset) as usize, Ordering::Relaxed);
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
        positions: AHashSet::new(),
        frequencies: local_frequencies,
    })
}

/// Generate word ngrams from a window for parallel processing
/// Extracted from word_ngrams.rs::word_ngrams_parallel (lines 110-205)
pub fn generate_from_window<P: TextParser + Sync + Send>(
    generator: &NGramGenerator<P>,
    window: &TextWindow,
    mapping: &TextMapping,
    start_sequence: u64,
) -> Result<Vec<NGramEntry>> {
    trace!("\n=== Starting word-based ngram generation for window with position checking ===");
    
    // Get the relevant text for this window
    // This is what it SHOULD be doing
    let window_text = &mapping.cleaned_text[window.window_start..window.window_end];
    
    // Tokenize the window text
    let tokens = generator.parser.tokenize_text(window_text)?;
    trace!("Window tokenized into {} words", tokens.len());
    
    if tokens.len() < generator.config.ngram_size {
        trace!("Window has insufficient tokens ({} < {})", 
            tokens.len(), generator.config.ngram_size);
        return Ok(Vec::new());
    }
    
    // Filter tokens to only those within the main content area
    let window_tokens: Vec<_> = tokens.clone().into_iter()
        .filter(|token| {
            let abs_start = window.abs_window_start + token.start_byte;
            let abs_end = window.abs_window_start + token.end_byte;
            abs_start >= window.abs_content_start && abs_end <= window.abs_content_end
        })
        .collect();
    // ADD THESE DEBUG LINES:
    info!("DEBUG: Window has {} tokens total, {} after filtering for main content", 
        tokens.len(), window_tokens.len());
    info!("  Window bounds: abs={}..{}, main_content={}..{}", 
        window.abs_window_start, window.abs_window_end,
        window.abs_content_start, window.abs_content_end);
    
    if window_tokens.len() < generator.config.ngram_size {
        trace!("Filtered tokens insufficient for ngrams");
        return Ok(Vec::new());
    }
    
    let mut ngrams = Vec::new();
    let total_windows = window_tokens.len().saturating_sub(generator.config.ngram_size - 1);
    
    for window_start in (0..total_windows).step_by(generator.config.stride) {
        let window_end = window_start + generator.config.ngram_size;
        
        if window_end > window_tokens.len() {
            break;
        }
        
        // Calculate absolute byte positions
        let start_byte = window.abs_window_start + window_tokens[window_start].start_byte;
        let end_byte = window.abs_window_start + window_tokens[window_end - 1].end_byte;
        
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
        
        let char_frequencies = generate_arabic_char_frequencies(&cleaned_text);
        
        // Create ngram with correct position mapping
        let ngram = NGramEntry {
            sequence_number,
            text_position: TextPosition {
                start_byte,
                end_byte,
                line_number: window_tokens[window_start].line_number,
            },
            text_content: NGramText {
                original_slice: original_text,
                cleaned_text: cleaned_text.clone(),
            },
            file_path: generator.file_path.clone().unwrap_or_default(),
            char_frequencies,
        };
        info!("NGRAM: seq={}, start={}, end={}, span={}, leaned='{}'",
    sequence_number,
    start_byte,
    end_byte,
    end_byte - start_byte,
    cleaned_text
);
        
        ngrams.push(ngram);
    }
    
    info!("Generated {} word-based ngrams from window", ngrams.len());
    Ok(ngrams)
}