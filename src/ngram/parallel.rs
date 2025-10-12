// src/ngram/parallel.rs
//! Parallel execution engine for ngram generation
//! 
//! This module orchestrates parallel ngram generation by dividing text into
//! chunks, pre-assigning sequence ranges, and coordinating parallel processing.

use ahash::{AHashSet, AHashMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use log::{info, debug, warn};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::parser::{TextMapping, TextParser};
use crate::config::NGramType;
use crate::config::generation::GenerationConfig;
use crate::types::NGramEntry;

use super::types::{
    GenerationResult, ChunkBoundary
};
use super::generator::NGramGenerator;
use super::{character, word, chunking, utils};

/// Information about a chunk's sequence range
#[derive(Debug, Clone)]
struct ChunkSequenceInfo {
    start_sequence: u64,
    estimated_count: usize,
}

/// Execute ngram generation using parallel processing
/// 
/// This function divides the text into chunks, pre-assigns sequence ranges,
/// and processes chunks in parallel while maintaining global ordering.
pub fn execute<P: TextParser + Sync + Send>(
    generator: &mut NGramGenerator<P>,
    text: &str,
    absolute_position: usize,
    provided_mapping: Option<&TextMapping>,  // NEW PARAMETER
) -> Result<GenerationResult> {
    info!("\n=== Starting Parallel NGram Generation ===");
    info!("Text length: {}, absolute position: {}", text.len(), absolute_position);
    info!("NGram type: {}, size: {}", 
        generator.config.ngram_type.as_str(),
        generator.config.ngram_size);
    
    // Step 1: Create text mapping with absolute positions
    // FIXED: Use provided mapping if available, otherwise create new one
    debug!("Creating text mapping for parallel processing");
    let mapping = if let Some(provided) = provided_mapping {
        provided.clone()  // Use the provided mapping
    } else {
        // Fallback: create new mapping (for legacy code paths)
        generator.parser.clean_text_with_mapping_absolute(text, absolute_position)?
    };
    
    // Step 2: Pre-scan text for safe splitting points
    // Extracted from current parallel.rs::pre_scan_text
    debug!("Pre-scanning text for chunk boundaries");
    let scan_info = chunking::scan_text(
        &mapping.cleaned_text, 
        &generator.parser, 
        generator.config.ngram_type
    )?;
    debug!("Scan complete: {} chars, {} Arabic chars, {} safe points", 
        scan_info.total_char_count, scan_info.arabic_char_count, scan_info.safe_points.len());
    
    // Step 3: Calculate chunk boundaries
    let boundaries = chunking::calculate_boundaries(
        mapping.cleaned_text.len(),
        &scan_info,
        generator.config.ngram_type,
        generator.config.ngram_size
    )?;
    // ADD THESE DEBUG LINES:
    info!("DEBUG: Text length: {}, Created {} chunks", mapping.cleaned_text.len(), boundaries.len());
    for (i, boundary) in boundaries.iter().enumerate() {
        info!("  Chunk {}: main={}..{} (size={}), overlap={}..{}", 
            i, boundary.start, boundary.end, boundary.end - boundary.start,
            boundary.overlap_start, boundary.overlap_end);
    }
    info!("Divided text into {} chunks for parallel processing", boundaries.len());
    
    // Step 4: Pre-calculate sequence ranges for each chunk (CRITICAL!)
    debug!("Pre-calculating sequence ranges for chunks");
    let chunk_sequences = pre_calculate_sequence_ranges(
        &boundaries,
        &mapping,
        &generator.parser,
        &generator.config,
        generator.current_sequence_number()
    )?;
        
    // Step 5: Create generator reference for parallel processing
    // We need to share the generator across threads safely
    let generator_ref = Arc::new(&*generator);
    
    // Step 6: Process chunks in parallel
    info!("Processing {} chunks in parallel", boundaries.len());
    let chunk_results: Vec<Result<Vec<NGramEntry>>> = boundaries
        .par_iter()
        .zip(chunk_sequences.par_iter())
        .map(|(boundary, seq_info)| {
            process_chunk(
                &generator_ref,
                boundary,
                seq_info,
                &mapping,
                absolute_position
            )
        })
        .collect();
    
    // Step 8: Collect results and handle errors
    let mut all_ngrams = Vec::new();
    let mut total_generated = 0;

    for (i, result) in chunk_results.into_iter().enumerate() {
        match result {
            Ok(chunk_ngrams) => {
                info!("DEBUG: Chunk {} returned {} ngrams", i, chunk_ngrams.len());
                debug!("Chunk {} generated {} ngrams", i, chunk_ngrams.len());
                total_generated += chunk_ngrams.len();
                all_ngrams.extend(chunk_ngrams);
            }
            Err(e) => {
                warn!("Chunk {} failed: {}", i, e);
                return Err(e);
            }
        }
    }

    info!("All chunks processed: {} total ngrams generated", total_generated);

    let mut aggregated_frequencies: AHashMap<u64, u32> = AHashMap::new();
    {
        use std::hash::{Hash, Hasher};
        for ngram in &all_ngrams {
            let mut hasher = ahash::AHasher::default();
            ngram.text_content.cleaned_text.hash(&mut hasher);
            let text_hash = hasher.finish();
            
            *aggregated_frequencies.entry(text_hash).or_insert(0) += 1;
        }
    }
    debug!("Frequency analysis: {} unique ngrams from {} total", 
        aggregated_frequencies.len(), all_ngrams.len());
    // Reassign sequence numbers to ensure continuity
    // Get the starting sequence number
    let base_sequence = chunk_sequences.first()
        .map(|seq| seq.start_sequence)
        .unwrap_or_else(|| generator.current_sequence.load(Ordering::Relaxed) as u64);

    // Reassign all sequence numbers sequentially
    for (i, ngram) in all_ngrams.iter_mut().enumerate() {
        ngram.sequence_number = base_sequence + i as u64;
    }

    info!("Reassigned sequence numbers: {} to {}", 
        base_sequence, 
        base_sequence + all_ngrams.len() as u64 - 1);

    // Step 9: Validate integrity (should pass now with continuous sequences)
    if !all_ngrams.is_empty() {
        validate_ngram_integrity(&all_ngrams, &mapping)?;
    }
    
    // Update the sequence counter to reflect all generated ngrams
    if let Some(last_ngram) = all_ngrams.last() {
        let new_sequence = (last_ngram.sequence_number + 1) as usize;
        generator.current_sequence.store(new_sequence, std::sync::atomic::Ordering::Relaxed);
        debug!("Updated sequence counter to {}", new_sequence);
    }
    
    info!("Parallel generation complete: {} ngrams", all_ngrams.len());
    
    Ok(GenerationResult {
        ngrams: all_ngrams,
        positions: AHashSet::new(),
        frequencies: aggregated_frequencies,
    })
}

/// Pre-calculate sequence ranges for each chunk
fn pre_calculate_sequence_ranges<P: TextParser>(
    boundaries: &[ChunkBoundary],
    mapping: &crate::parser::TextMapping,
    parser: &P,
    config: &GenerationConfig,
    start_sequence: u64,
) -> Result<Vec<ChunkSequenceInfo>> {
    let mut sequence_infos = Vec::new();
    let mut current_sequence = start_sequence;
    
    for (i, boundary) in boundaries.iter().enumerate() {
        // Extract the chunk text (main content only, not overlap)
        let chunk_start = utils::find_char_boundary(&mapping.cleaned_text, boundary.start);
        let chunk_end = utils::find_char_boundary(&mapping.cleaned_text, boundary.end);
        
        if chunk_end <= chunk_start {
            warn!("Invalid chunk boundaries: start={}, end={}", chunk_start, chunk_end);
            sequence_infos.push(ChunkSequenceInfo {
                start_sequence: current_sequence,
                estimated_count: 0,
            });
            continue;
        }
        
        let chunk_text = &mapping.cleaned_text[chunk_start..chunk_end];
        
        // Estimate ngram count for this chunk
        let estimated_count = match config.ngram_type {
            NGramType::Character => {
                // For character ngrams: (text_length - ngram_size + 1) / stride
                let char_count = chunk_text.chars().count();
                if char_count >= config.ngram_size {
                    (char_count - config.ngram_size + 1) / config.stride
                } else {
                    0
                }
            }
            NGramType::Word => {
                // For word ngrams, we need to tokenize to get accurate count
                if let Ok(tokens) = parser.tokenize_text(chunk_text) {
                    if tokens.len() >= config.ngram_size {
                        (tokens.len() - config.ngram_size + 1) / config.stride
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
        };
        // ADD THIS DEBUG LINE:
        info!("DEBUG: Chunk {}: start_seq={}, estimated_count={}, chunk_size={}", 
            i, current_sequence, estimated_count, chunk_text.len());
        //
        debug!("Chunk {}: start_seq={}, estimated_count={}", 
            i, current_sequence, estimated_count);
        
        sequence_infos.push(ChunkSequenceInfo {
            start_sequence: current_sequence,
            estimated_count,
        });
        
        // Advance sequence counter for next chunk
        current_sequence += estimated_count as u64;
    }
    
    info!("Pre-calculated sequences for {} chunks, total estimated ngrams: {}", 
        boundaries.len(), current_sequence - start_sequence);
    
    Ok(sequence_infos)
}

/// Process a single chunk in parallel
fn process_chunk<P: TextParser + Sync + Send>(
    generator: &Arc<&NGramGenerator<P>>,
    boundary: &ChunkBoundary,
    seq_info: &ChunkSequenceInfo,
    mapping: &crate::parser::TextMapping,
    absolute_offset: usize,
) -> Result<Vec<NGramEntry>> {
    debug!("Processing chunk with sequence range {}..{}", 
        seq_info.start_sequence, 
        seq_info.start_sequence + seq_info.estimated_count as u64);
    
    // Create text window for this chunk
    let window = chunking::create_window(&mapping.cleaned_text, boundary, absolute_offset);
    
    // Route to appropriate generation function based on ngram type
    let ngrams = match generator.config.ngram_type {
        NGramType::Character => {
            character::generate_from_window(
                generator,
                &window,
                mapping,
                seq_info.start_sequence
            )?
        }
        NGramType::Word => {
            word::generate_from_window(
                generator,
                &window,
                mapping,
                seq_info.start_sequence
            )?
        }
    };
    
    debug!("Chunk generated {} ngrams (estimated: {})", 
        ngrams.len(), seq_info.estimated_count);
    
    // Warn if actual count differs significantly from estimate
    if ngrams.len() != seq_info.estimated_count {
        let diff = (ngrams.len() as i32 - seq_info.estimated_count as i32).abs();
        if diff > 10 {
            warn!("Chunk sequence estimation was off by {}: estimated={}, actual={}", 
                diff, seq_info.estimated_count, ngrams.len());
        }
    }
    
    Ok(ngrams)
}

/// Validate the integrity of generated ngrams
/// Extracted from current parallel.rs::validate_ngram_integrity
fn validate_ngram_integrity(
    ngrams: &[NGramEntry],
    mapping: &crate::parser::TextMapping,
) -> Result<()> {
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
        // Check bounds
        if ngram.text_position.end_byte > max_byte_position {
            // Word ngrams might extend beyond character mapping
            debug!("NGram position {}..{} extends beyond mapped characters (max: {})",
                ngram.text_position.start_byte,
                ngram.text_position.end_byte,
                max_byte_position);
        }
        
        // Check sequence continuity
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
    
    info!("Validation passed: {} ngrams with continuous sequences", ngrams.len());
    Ok(())
}