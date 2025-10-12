// src/ngram/sequential.rs
//! Sequential execution engine for ngram generation
//! 
//! This module orchestrates single-threaded ngram generation by coordinating
//! between text parsing, strategy selection, and the actual generation functions.

use log::{info, debug};

use crate::error::Result;
use crate::parser::TextParser;
use crate::config::NGramType;
use crate::parser::TextMapping;

use super::types::GenerationResult;
use super::generator::NGramGenerator;
use super::{character, word};

/// Execute ngram generation using sequential processing
/// 
/// This is the main orchestration function for sequential processing.
/// It handles text mapping, strategy routing, and state management.
pub fn execute<P: TextParser + Sync + Send>(
    generator: &mut NGramGenerator<P>,
    text: &str,
    absolute_position: usize,
    provided_mapping: Option<&TextMapping>,
) -> Result<GenerationResult> {
    info!("\n=== Starting Sequential NGram Generation ===");
    info!("Text length: {}, absolute position: {}", text.len(), absolute_position);
    info!("NGram type: {}, size: {}", 
        generator.config.ngram_type.as_str(),
        generator.config.ngram_size);
    
    // Step 1: Create text mapping with absolute positions
    // This is extracted from position.rs::generate_ngrams_with_position
    debug!("Creating text mapping with absolute positions");
    let mapping = if let Some(provided) = provided_mapping {
        provided.clone()  // Use the provided mapping
    } else {
        // Fallback: This won't work correctly if text is already cleaned!
        // This path should eventually be removed
        generator.parser.clean_text_with_mapping_absolute(text, absolute_position)?
    };    
    debug!("Mapping created: {} chars, {} char mappings", 
        mapping.cleaned_text.len(), 
        mapping.char_map.len());
    
    // Step 3: Route to appropriate generation strategy
    // This routing logic is currently buried in char_ngrams_sequential but belongs here
    let result = match generator.config.ngram_type {
        NGramType::Character => {
            debug!("Routing to character-based generation");
            character::generate_sequential(generator, &mapping)?
        },
        NGramType::Word => {
            debug!("Routing to word-based generation");
            word::generate_sequential(generator, &mapping)?
        }
    };
        
    debug!("Generation complete: {} ngrams generated",
        result.ngrams.len());
    
    // Step 6: Log summary information
    if !result.ngrams.is_empty() {
        info!("Sequential generation complete: {} ngrams", result.ngrams.len());
        
        let first = &result.ngrams[0];
        let last = result.ngrams.last().unwrap();
        
        debug!("First ngram: seq={}, text='{}', pos={}..{}", 
            first.sequence_number,
            first.text_content.cleaned_text,
            first.text_position.start_byte,
            first.text_position.end_byte);
            
        debug!("Last ngram: seq={}, text='{}', pos={}..{}",
            last.sequence_number,
            last.text_content.cleaned_text,
            last.text_position.start_byte,
            last.text_position.end_byte);
        
        // Verify sequence continuity
        let sequence_range = last.sequence_number - first.sequence_number + 1;
        if sequence_range != result.ngrams.len() as u64 {
            log::warn!("Sequence discontinuity detected: {} ngrams but sequence range is {}",
                result.ngrams.len(), sequence_range);
        }
    } else {
        info!("No ngrams generated from this text");
    }
    
    Ok(result)
}