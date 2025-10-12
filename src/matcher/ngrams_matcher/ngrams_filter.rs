
use log::{warn, debug, trace};

use crate::error::{Error, Result};
use crate::config::{SimilarityMetric, MatchingConfig};
use crate::storage::LMDBStorage;
// =================================================================
// FILTER FUNCTIONS
// =================================================================

/// Simple prefix filtering using existing LMDB prefix index
pub fn apply_prefix_filtering(
    source_seq: u64,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Result<Vec<u64>> {
    let start_time = std::time::Instant::now();
    
    // Step 1: Get source prefix from prefix_bytes_db
    let source_prefix = source_storage.get_prefix_bytes(source_seq)?;
    
    if source_prefix.is_empty() {
        trace!("No prefix found for source sequence {}", source_seq);
        return Ok(Vec::new());
    }
    
    // Step 2: Get all target sequences with matching prefix from prefix_index_db
    let matching_sequences = target_storage.get_sequences_by_prefix_bytes(&source_prefix)?;
    
    trace!("Prefix filter: {} bytes prefix matched {} target sequences in {:?}",
        source_prefix.len(), 
        matching_sequences.len(), 
        start_time.elapsed()
    );
    
    Ok(matching_sequences)
}

pub fn apply_metadata_filtering(
    source_seq: u64,
    target_sequences: Vec<u64>,  // Sequences to filter (from prefix or all)
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Result<Vec<u64>> {
    if target_sequences.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = std::time::Instant::now();
    
    // Step 1: Get source metadata using batch function with single element
    let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
    let source_metadata = source_metadata_vec.get(0)  // Index 0 since we passed one sequence
        .ok_or_else(|| Error::Storage(format!("Source sequence {} not found", source_seq)))?;
    
    // Step 2: Batch load target metadata
    let target_metadata_vec = target_storage.get_ngram_metadata_batch(&target_sequences)?;
    
    // Step 3: Create a map for efficient lookup (or work with parallel vecs)
    // Assuming the returned vec is in the same order as requested sequences
    let filtered: Vec<u64> = target_sequences
        .iter()
        .zip(target_metadata_vec.iter())
        .filter_map(|(&seq, target_meta)| {
            // Only keep sequences with matching hash
            if target_meta.text_hash == source_metadata.text_hash {
                Some(seq)
            } else {
                None
            }
        })
        .collect();
    
    trace!("Metadata filter (hash match): {} → {} sequences in {:?}",
        target_metadata_vec.len(), 
        filtered.len(), 
        start_time.elapsed()
    );
    
    Ok(filtered)
}

pub fn apply_quick_filtering(
    source_seq: u64,
    target_sequences: Vec<u64>,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Result<Vec<u64>> {
    if target_sequences.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = std::time::Instant::now();
    
    // Step 1: Get source character frequencies
    let source_char_freqs_map = source_storage.get_char_frequencies_batch(&[source_seq])?;
    let source_char_freqs = source_char_freqs_map.get(&source_seq)
        .ok_or_else(|| Error::Storage(format!("Source sequence {} char frequencies not found", source_seq)))?;
    
    // Step 2: Batch load target character frequencies
    let target_char_freqs = target_storage.get_char_frequencies_batch(&target_sequences)?;
    
    // Step 3: Filter using character overlap
    let source_bitmap = source_char_freqs.char_bitmap;  // Direct access!
    
    let filtered: Vec<u64> = target_sequences
        .into_iter()
        .filter(|&seq| {
            if let Some(target_freqs) = target_char_freqs.get(&seq) {
                // Single AND + POPCNT operation!
                let shared_chars = (source_bitmap & target_freqs.char_bitmap).count_ones();
                let overlap_ratio = shared_chars as f32 / source_char_freqs.unique_chars.max(1) as f32;
                overlap_ratio >= 0.5
            } else {
                false
            }
        })
        .collect();
    
    trace!("Quick filter (char overlap): {} → {} sequences in {:?}",
        target_char_freqs.len(), 
        filtered.len(), 
        start_time.elapsed()
    );
    
    Ok(filtered)
}

pub fn apply_length_filtering(
    source_seq: u64,
    target_sequences: Vec<u64>,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    config: &MatchingConfig,  // Need this for length ratio thresholds
) -> Result<Vec<u64>> {
    if target_sequences.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = std::time::Instant::now();
    
    // Step 1: Get source metadata (for text_length)
    let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
    let source_metadata = source_metadata_vec.get(0)
        .ok_or_else(|| Error::Storage(format!("Source sequence {} metadata not found", source_seq)))?;
    
    let source_length = source_metadata.text_length as f64;
    
    // Step 2: Batch load target metadata
    let target_metadata_vec = target_storage.get_ngram_metadata_batch(&target_sequences)?;
    
    // Step 3: Determine length ratio thresholds based on similarity metric
    let (length_ratio_min, length_ratio_max) = match config.text_similarity_metric {  // Fixed field name
        SimilarityMetric::Levenshtein => (0.8, 1.2),  // Stricter for edit distance
        SimilarityMetric::Jaccard => (0.7, 1.3),      // More lenient for Jaccard
        SimilarityMetric::Cosine => (0.7, 1.3),       // More lenient for Cosine
        SimilarityMetric::VSA => (0.75, 1.25),        // Moderate for VSA
        SimilarityMetric::Exact => (1.0, 1.0),        // Exact length match
    };
    
    // Step 4: Filter by length ratio
    let filtered: Vec<u64> = target_sequences
        .iter()
        .zip(target_metadata_vec.iter())
        .filter_map(|(&seq, target_meta)| {
            let length_ratio = target_meta.text_length as f64 / source_length;
            if length_ratio >= length_ratio_min && length_ratio <= length_ratio_max {
                Some(seq)
            } else {
                None
            }
        })
        .collect();
    
    trace!("Length filter: {} → {} sequences in {:?} (ratio: {:.2}-{:.2})",
        target_metadata_vec.len(), 
        filtered.len(), 
        start_time.elapsed(),
        length_ratio_min,
        length_ratio_max
    );
    
    Ok(filtered)
}

/// Orchestrator function that applies all enabled filters with adaptive batching
/// Batches are used when candidate count exceeds threshold to maintain memory efficiency
pub fn apply_all_filters_with_batching(
    source_seq: u64,  // Just the sequence ID
    source_storage: &LMDBStorage,  // To get source data when needed
    target_storage: &LMDBStorage,  // To query target sequences
    config: &MatchingConfig,
) -> Result<Vec<u64>> {
    const BATCH_THRESHOLD: usize = 1000;
    const BATCH_SIZE: usize = 1000;
        
    // =============================================================
    // STAGE 1: PREFIX FILTERING (Uses index, no batching needed)
    // =============================================================
    
    // Start with sequence IDs, not NGramEntry objects
    let mut candidate_sequences: Vec<u64>;

    if config.enable_prefix_filtering.unwrap_or(false) {
        match apply_prefix_filtering(source_seq, source_storage, target_storage) {
            Ok(prefix_sequences) => {
                candidate_sequences = prefix_sequences;
                trace!("Prefix filtering returned {} candidate sequences", candidate_sequences.len());
            }
            Err(e) => {
                warn!("Prefix filtering failed: {}, using all target sequences", e);
                // Get all target sequences
                let total_count = target_storage.get_db_info()?.total_ngrams as usize;
                candidate_sequences = (0..total_count as u64).collect();
            }
        }
    } else {
        // No prefix filtering - use all target sequences
        let total_count = target_storage.get_db_info()?.total_ngrams as usize;
        candidate_sequences = (0..total_count as u64).collect();
        trace!("Prefix filtering disabled, processing all {} sequences", total_count);
    }
    
    // =============================================================
    // STAGE 2: METADATA FILTERING
    // =============================================================
    
    if config.enable_metadata_filtering.unwrap_or(false) {
        // Metadata filtering works with sequences directly
        if candidate_sequences.len() <= BATCH_THRESHOLD {
            // Small enough to process directly
            candidate_sequences = apply_metadata_filtering(
                source_seq, 
                candidate_sequences, 
                source_storage, 
                target_storage
            )?;
        } else {
            // Need to batch process the candidates
            trace!("Metadata filter: batching {} candidate sequences", candidate_sequences.len());
            let mut filtered = Vec::new();
            
            for chunk in candidate_sequences.chunks(BATCH_SIZE) {
                let chunk_filtered = apply_metadata_filtering(
                    source_seq,
                    chunk.to_vec(),  // Convert slice to Vec
                    source_storage,
                    target_storage
                )?;
                filtered.extend(chunk_filtered);
            }
            candidate_sequences = filtered;
        }
        
        trace!("After metadata filtering: {} candidate sequences", candidate_sequences.len());
    }
    
    // =============================================================
    // STAGE 3: QUICK FILTERING
    // =============================================================
    
    if config.enable_quick_filtering.unwrap_or(false) {
    if !candidate_sequences.is_empty() {
        if candidate_sequences.len() <= BATCH_THRESHOLD {
            // Small enough to process directly
            candidate_sequences = apply_quick_filtering(
                source_seq,
                candidate_sequences,
                source_storage,
                target_storage
            )?;
        } else {
            // Batch process
            trace!("Quick filter: batching {} candidate sequences", candidate_sequences.len());
            let mut filtered = Vec::new();
            
            for chunk in candidate_sequences.chunks(BATCH_SIZE) {
                let chunk_filtered = apply_quick_filtering(
                    source_seq,
                    chunk.to_vec(),  // Convert slice to Vec
                    source_storage,
                    target_storage
                )?;
                filtered.extend(chunk_filtered);
            }
            candidate_sequences = filtered;
        }
        
        trace!("After quick filtering: {} candidate sequences", candidate_sequences.len());
    }
}
    
    // =============================================================
    // STAGE 4: LENGTH FILTERING
    // =============================================================
    
    if config.enable_length_filtering.unwrap_or(false) {
        if !candidate_sequences.is_empty() {
            if candidate_sequences.len() <= BATCH_THRESHOLD {
                // Small enough to process directly
                candidate_sequences = apply_length_filtering(
                    source_seq,
                    candidate_sequences,
                    source_storage,
                    target_storage,
                    config
                )?;
            } else {
                // Batch process
                trace!("Length filter: batching {} candidate sequences", candidate_sequences.len());
                let mut filtered = Vec::new();
                
                for chunk in candidate_sequences.chunks(BATCH_SIZE) {
                    let chunk_filtered = apply_length_filtering(
                        source_seq,
                        chunk.to_vec(),  // Convert slice to Vec
                        source_storage,
                        target_storage,
                        config
                    )?;
                    filtered.extend(chunk_filtered);
                }
                candidate_sequences = filtered;
            }
            
            trace!("After length filtering: {} candidate sequences", candidate_sequences.len());
        }
    }
    
    // =============================================================
    // FINAL: Return candidates or load all if no filters were applied
    // =============================================================
    
    Ok(candidate_sequences)
}

// =============================================================
// GRID FILTERING FUNCTIONS
// =============================================================
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// Load all target sequences that match any of the given prefixes AND fall within the range
/// Returns a HashMap mapping prefixes to their matching sequences
pub fn load_prefix_matches_for_range(
    source_sequences: &[u64],           // Changed from &[NGramEntry]
    source_storage: &LMDBStorage,       // Added to get source prefixes
    target_storage: &LMDBStorage,
    target_range: &Range<u64>,
    _prefix_length: usize,              // Not needed - using stored prefix depth
) -> Result<HashMap<Vec<u8>, Vec<u64>>> {  // Changed to bytes->sequences
    
    // Step 1: Collect all unique prefixes from source sequences
    let mut unique_prefixes = HashSet::new();
    
    // Batch load prefixes for all source sequences
    let source_prefixes = source_storage.get_prefix_bytes_batch(source_sequences)?;
    
    for (_seq, prefix_bytes) in source_prefixes {
        if !prefix_bytes.is_empty() {
            unique_prefixes.insert(prefix_bytes);
        }
    }
    
    if unique_prefixes.is_empty() {
        return Ok(HashMap::new());
    }
    
    debug!("Cell has {} unique prefixes to search", unique_prefixes.len());
    
    // Step 2: Get all sequence numbers for all prefixes
    let mut all_sequences = HashSet::new();
    
    for prefix_bytes in &unique_prefixes {
        // Use the prefix index to get sequences
        let sequences = target_storage.get_sequences_by_prefix_bytes(prefix_bytes)?;
        
        // Filter to only sequences in our range
        for seq in sequences {
            if seq >= target_range.start && seq < target_range.end {
                all_sequences.insert(seq);
            }
        }
    }
    
    if all_sequences.is_empty() {
        debug!("No sequences found in range {:?} for any prefix", target_range);
        return Ok(HashMap::new());
    }
    
    debug!("Found {} unique sequences in range for all prefixes", all_sequences.len());
    
    // Step 3: Build prefix -> sequences map
    // We need to load target prefixes to properly group them
    let sequences_vec: Vec<u64> = all_sequences.into_iter().collect();
    let target_prefixes = target_storage.get_prefix_bytes_batch(&sequences_vec)?;
    
    let mut prefix_map: HashMap<Vec<u8>, Vec<u64>> = HashMap::new();
    
    for (seq, prefix_bytes) in target_prefixes {
        // Only add if this prefix was in our search set
        if unique_prefixes.contains(&prefix_bytes) {
            prefix_map.entry(prefix_bytes).or_default().push(seq);
        }
    }
    
    debug!("Built prefix map with {} prefixes containing {} total sequences",
        prefix_map.len(),
        prefix_map.values().map(|v| v.len()).sum::<usize>()
    );
    
    Ok(prefix_map)
}