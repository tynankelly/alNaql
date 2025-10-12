// src/matcher/sequential_ngrams.rs
//
// Sequential n-gram matching module - processes source n-grams one by one
// against target n-grams using single-threaded processing.
//
// This module is designed to be self-contained and dependency-free, with all
// caching handled internally by the SimilarityCalculator.

use crate::error::Result;
use crate::matcher::types::SequenceMatch;
use crate::storage::LMDBStorage;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::matcher::ngrams_matcher::ngrams_filter::apply_all_filters_with_batching;
use crate::AlNaqlConfig;
use log::debug;

/// Process source n-grams sequentially against target n-grams to find matches.
pub fn process(
    config: &AlNaqlConfig,
    source_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
) -> Result<Vec<SequenceMatch>> {
    // Initialize results vector
    let mut all_matches = Vec::new();
   
    // Process each source sequence
    for &source_seq in source_sequences {
        
        // Apply all filters with lazy loading
        let target_candidates = apply_all_filters_with_batching(
            source_seq,
            source_storage,
            target_storage,
            &config.matcher
        )?;
        
        // Process only filtered candidates
        let matches = if !target_candidates.is_empty() {
            similarity_calc.find_matching_ngrams(
                source_seq,
                &target_candidates,
                source_storage,
                target_storage,
                config,
                None,
                None,
                None,
            )?
        } else {
            Vec::new()
        };
                
        // Add matches to results
        all_matches.extend(matches);
    }
    
    debug!("Sequential processing complete: found {} total matches from {} source sequences",
        all_matches.len(), source_sequences.len());
    
    // Sort and deduplicate matches
    if !all_matches.is_empty() {
        all_matches.sort_by(|a, b| {
            a.source_sequence.cmp(&b.source_sequence)
                .then(a.target_sequence.cmp(&b.target_sequence))
        });
        
        all_matches.dedup_by(|a, b| {
            a.source_sequence == b.source_sequence && 
            a.target_sequence == b.target_sequence
        });
    }
    
    Ok(all_matches)
}