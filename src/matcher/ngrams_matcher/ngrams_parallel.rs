// src/matcher/ngrams_parallel.rs
// Source Batching + Sequential Candidate Processing with configurable filtering pipeline

use rayon::prelude::*;
use log::{info, debug};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::Result;
use crate::matcher::types::SequenceMatch;
use crate::matcher::ngrams_matcher::ngrams_filter::apply_all_filters_with_batching;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::storage::LMDBStorage;
use crate::AlNaqlConfig;


/// Unified parallel processing function that replaces the complex fallback chain
/// of find_matches_parallel -> find_matches_fallback -> process_chunks_parallel
pub fn process_parallel(
    config: &AlNaqlConfig,
    source_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
) -> Result<Vec<SequenceMatch>> { 
    let start_time = Instant::now();
    info!("Starting unified parallel processing for {} source sequences", source_sequences.len());
    // =================================================================
    // STAGE 1: INITIALIZATION & CONFIGURATION
    // =================================================================
    
    // Hard-coded chunk sizes (no dynamic calculation)
    let source_chunk_size = config.matcher.parallel_source_chunk_size;
    let target_chunk_size = config.matcher.parallel_target_chunk_size;

    debug!("Using hard-coded chunk sizes: source={}, target={}", source_chunk_size, target_chunk_size);
    
    // Progress tracking
    let progress_counter = Arc::new(AtomicUsize::new(0));
    let total_ngrams = source_sequences.len();

    debug!("Parallel processing using {} threads from ambient pool", 
       rayon::current_num_threads());
    
    // =================================================================
    // STAGE 2: SOURCE BATCHING & PARALLEL PROCESSING
    // Threading Model: Source batches processed in parallel,
    // candidate processing within each batch is sequential
    // =================================================================
    
    let results = source_sequences.chunks(source_chunk_size)
    .collect::<Vec<_>>()
    .par_iter()
    .enumerate()
    .map(|(batch_idx, source_batch)| {
        process_source_batch(
            batch_idx,
            source_batch,
            source_storage,
            target_storage,
            config,
            target_chunk_size,
            &progress_counter,
            total_ngrams,
            &similarity_calc,
        )
    })
    .collect::<Result<Vec<_>>>()?;
    
    // =================================================================
    // STAGE 3: RESULT AGGREGATION & DEDUPLICATION
    // =================================================================
    
    // Combine all results efficiently
    let total_matches: usize = results.iter().map(|batch| batch.len()).sum();
    let mut all_matches = Vec::with_capacity(total_matches);
    for batch_matches in results {
        all_matches.extend(batch_matches);
    }
    
    // Deduplicate matches if needed (large datasets can produce duplicates)
    if !all_matches.is_empty() {
        info!("Deduplicating {} matches...", all_matches.len());
        let before_count = all_matches.len();
        
        // Sort first to enable efficient deduplication
        all_matches.sort_by(|a, b| {
            a.source_sequence.cmp(&b.source_sequence)
                .then(a.target_sequence.cmp(&b.target_sequence))
        });
        
        // Remove adjacent duplicates
        all_matches.dedup_by(|a, b| {
            a.source_sequence == b.source_sequence &&
            a.target_sequence == b.target_sequence
        });
        
        let removed = before_count - all_matches.len();
        if removed > 0 {
            debug!("Removed {} duplicate matches", removed);
        }
    }
    
    info!("Unified parallel processing complete: found {} matches in {:?}", 
         all_matches.len(), start_time.elapsed());
    
    Ok(all_matches)
}

/// Process a batch of source ngrams through the configurable filtering pipeline
fn process_source_batch(
    batch_idx: usize,
    source_batch: &[u64], 
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,   
    config: &AlNaqlConfig,
    target_chunk_size: usize,
    progress_counter: &Arc<AtomicUsize>,
    total_ngrams: usize,
    similarity_calc: &SimilarityCalculator,
) -> Result<Vec<SequenceMatch>> {
    let batch_start = Instant::now();
    let mut all_matches = Vec::new();
    
    for source_seq in source_batch {
        
        // =============================================================
        // STAGE 3: PER-SOURCE TARGET N-GRAM FILTERING PIPELINE
        // =============================================================
        
        // Start with either prefix-filtered candidates or all targets
        let target_candidates = apply_all_filters_with_batching(
            *source_seq,
            source_storage,
            target_storage,
            &config.matcher
        )?;
        
        // =============================================================
        // STAGE 4: FINAL SEQUENTIAL CHUNK PROCESSING
        // =============================================================
        
        if !target_candidates.is_empty() {
            let matches = parallel_ngram_matching(
                *source_seq, 
                &target_candidates,
                source_storage,
                target_storage,
                target_chunk_size,
                similarity_calc,
                &config,
            )?;
                       
            all_matches.extend(matches);
        }
    }
    
    // Update progress
    let processed = progress_counter.fetch_add(source_batch.len(), Ordering::Relaxed) + source_batch.len();
    if processed % 1000 == 0 || processed == total_ngrams {
        debug!("Processed batch {} - {}/{} source ngrams ({:.1}%) in {:?}", 
               batch_idx, processed, total_ngrams, 
               (processed as f64 / total_ngrams as f64) * 100.0,
               batch_start.elapsed());
    }
    
    Ok(all_matches)
}

/// Process filtered candidates for a single source ngram
fn parallel_ngram_matching(
    source_seq: u64,
    target_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    chunk_size: usize,
    similarity_calc: &SimilarityCalculator,
    config: &AlNaqlConfig,
) -> Result<Vec<SequenceMatch>> {
    if target_sequences.is_empty() {
        return Ok(Vec::new());
    }
    
    // Split target sequences into chunks and process each chunk
    let chunk_results: Result<Vec<Vec<SequenceMatch>>> = target_sequences
        .chunks(chunk_size)
        .map(|chunk| {
            // Use find_matching_ngrams which handles all similarity calculation,
            // threshold checking, and match creation
            similarity_calc.find_matching_ngrams(
                source_seq,
                chunk,
                source_storage,
                target_storage,
                config,
                None,
                None,
                None,
            )
        })
        .collect();
    
    // Merge chunk results
    let results = chunk_results?;
    let total_matches: usize = results.iter().map(|chunk| chunk.len()).sum();
    let mut all_matches = Vec::with_capacity(total_matches);
    
    for chunk_matches in results {
        all_matches.extend(chunk_matches);
    }
    
    Ok(all_matches)
}