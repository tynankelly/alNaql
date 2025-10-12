// src/matcher/ngrams_parallel_opt.rs

use rayon::prelude::*;
use log::{info, debug, trace};
use std::time::Instant;
use crate::error::Result;
use crate::matcher::types::SequenceMatch;
use crate::config::AlNaqlConfig;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::storage::LMDBStorage;
use crate::matcher::ngrams_matcher::ngrams_filter::apply_all_filters_with_batching;

/// Unified pipeline processing function optimized for memory efficiency and large datasets
/// Consolidates all pipeline logic into single function following ngrams_parallel.rs pattern
pub fn process_parallel_opt(
    config: &AlNaqlConfig,
    source_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
) -> Result<Vec<SequenceMatch>> {
    let start_time = Instant::now();
    info!("Starting unified pipeline processing for {} source ngrams", source_sequences.len());

    // =================================================================
    // STAGE 1: INITIALIZATION & CONFIGURATION
    // =================================================================
    
    // Hard-coded chunk sizes optimized for pipeline processing
    let source_chunk_size = config.matcher.parallel_source_chunk_size;
    let target_chunk_size = config.matcher.parallel_target_chunk_size;
    
    // Hard-coded pipeline-specific optimizations
    let storage_access_strategy = "shared";      // Most memory efficient
    let memory_limit_per_worker = 0;             // Disabled for now
    let enable_memory_checks = false;            // Disabled for now  
    let _adaptive_memory_scaling = false;         // Disabled for now
    let enable_chunk_prefetching = true;         // Pipeline optimization
    let pipeline_buffer_size = 1000;             // Pipeline-specific buffering
    debug!("Using hard-coded chunk sizes: source={}, target={}", source_chunk_size, target_chunk_size);
    debug!("Pipeline settings: storage={}, memory_limit={}, memory_checks={}, prefetch={}", 
           storage_access_strategy, memory_limit_per_worker, enable_memory_checks, enable_chunk_prefetching);
    
    debug!("Parallel processing using {} threads from ambient pool", 
       rayon::current_num_threads());
    
    // =================================================================
    // STAGE 2: SOURCE BATCHING
    // Threading Model: Source batches processed in parallel with memory-aware coordination
    // =================================================================
    
    // Process directly using ambient thread pool
    let results = source_sequences.chunks(source_chunk_size)  // Changed from source_ngrams
        .collect::<Vec<_>>()
        .par_iter()
        .enumerate()
        .map(|(batch_idx, source_batch)| {
            process_source_batch_parallel_opt(
                batch_idx,
                source_batch,
                source_storage,
                target_storage,
                config,
                similarity_calc,
                target_chunk_size,
                pipeline_buffer_size,
                enable_chunk_prefetching,
                storage_access_strategy,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    
    // =================================================================
    // STAGE 3: RESULT AGGREGATION & DEDUPLICATION  
    // =================================================================
    
    // Combine all results efficiently with pipeline-optimized buffering
    let total_matches: usize = results.iter().map(|batch| batch.len()).sum();
    let mut all_matches = Vec::with_capacity(total_matches);
    for batch_matches in results {
        all_matches.extend(batch_matches);
    }
    
    // Deduplicate matches if needed (same logic as parallel but with pipeline logging)
    if !all_matches.is_empty() {
        info!("Pipeline deduplicating {} matches...", all_matches.len());
        let before_count = all_matches.len();
        
        // Sort first to enable efficient deduplication
        all_matches.sort_by(|a, b| {
            a.source_sequence.cmp(&b.source_sequence)
                .then(a.target_sequence.cmp(&b.target_sequence))
        });

        all_matches.dedup_by(|a, b| {
            a.source_sequence == b.source_sequence && 
            a.target_sequence == b.target_sequence
        });
        
        let removed = before_count - all_matches.len();
        if removed > 0 {
            debug!("Pipeline removed {} duplicate matches", removed);
        }
    }
    
    info!("Unified pipeline processing complete: found {} matches in {:?}", 
         all_matches.len(), start_time.elapsed());
    
    Ok(all_matches)
}

/// Process a batch of source ngrams in parallel with memory-optimized coordination
fn process_source_batch_parallel_opt(
    batch_idx: usize,
    source_batch: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    config: &AlNaqlConfig,
    similarity_calc: &SimilarityCalculator,
    target_chunk_size: usize,
    pipeline_buffer_size: usize,
    enable_chunk_prefetching: bool,
    storage_access_strategy: &str,
) -> Result<Vec<SequenceMatch>>  {
    let batch_start = Instant::now();
    let mut all_matches = Vec::new();
    
    // Pipeline-specific: Pre-allocate result buffer based on estimated match rate
    let estimated_matches = source_batch.len() / 5; // Estimate 20% match rate
    all_matches.reserve(estimated_matches.max(pipeline_buffer_size));
    
    for &source_seq in source_batch.iter() {
        
        // =============================================================
        // PARALLEL OPTIMIZATION: Multi-stage filtering pipeline
        // =============================================================
        
        let target_candidates = apply_all_filters_with_batching(
            source_seq,
            source_storage,
            target_storage,
            &config.matcher
        )?;
        
        // =============================================================
        // PARALLEL PROCESSING: Memory-optimized candidate processing
        // =============================================================
        
        if !target_candidates.is_empty() {
            // Pipeline optimization: Process candidates in memory-friendly chunks
            let chunk_size = if enable_chunk_prefetching {
                target_chunk_size.min(pipeline_buffer_size)
            } else {
                target_chunk_size
            };
            
            // Use SimilarityCalculator for batch processing
            let matches = if target_candidates.len() > chunk_size {
                // Process in chunks for memory efficiency
                let chunk_results: Result<Vec<Vec<SequenceMatch>>> = target_candidates
                    .chunks(chunk_size)
                    .map(|chunk| {
                        if storage_access_strategy == "shared" {
                            trace!("Pipeline using shared storage access for {} candidates", chunk.len());
                        }
                        similarity_calc.find_matching_ngrams(
                            source_seq,
                            chunk,
                            source_storage,
                            target_storage,
                            &config,
                            None,
                            None,
                            None,
                        )
                    })
                    .collect();
                
                // Merge results
                let results = chunk_results?;
                let mut all_chunk_matches = Vec::with_capacity(
                    results.iter().map(|r| r.len()).sum()
                );
                for chunk_matches in results {
                    all_chunk_matches.extend(chunk_matches);
                }
                all_chunk_matches
            } else {
                // Small batch - process directly
                similarity_calc.find_matching_ngrams(
                    source_seq,
                    &target_candidates,
                    source_storage,
                    target_storage,
                    &config,
                    None,
                    None,
                    None,
                )?
            };
                        
            all_matches.extend(matches);
        }
    }
    
    debug!("Pipeline batch {} processed {} source ngrams -> {} matches in {:?}", 
           batch_idx, source_batch.len(), all_matches.len(), batch_start.elapsed());
    
    Ok(all_matches)
}