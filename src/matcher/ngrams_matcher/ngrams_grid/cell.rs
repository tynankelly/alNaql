// src/matcher/ngrams_matcher/ngrams_grid/cell.rs

use std::ops::Range;
use log::{info, debug, warn};
use ahash::{AHashMap, AHashSet};

use crate::error::Result;
use crate::config::AlNaqlConfig;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::storage::LMDBStorage;
use crate::matcher::types::SequenceMatch;
use crate::matcher::is_traced_sequence;
use crate::matcher::ngrams_matcher::ngrams_filter::load_prefix_matches_for_range;
use crate::matcher::ngrams_matcher::{apply_metadata_filtering, apply_quick_filtering, apply_length_filtering};

/// Represents a single cell in the processing grid
#[derive(Debug, Clone)]
pub struct GridCell {
    pub id: (usize, usize),  // (row, col) position in grid
    pub source_range: Range<usize>,  // Indices into source ngrams
    pub target_range: Range<u64>,    // Sequence numbers for target
}

pub fn process_cell(
    cell: &GridCell,
    source_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    config: &AlNaqlConfig,
    similarity_calc: &SimilarityCalculator,
) -> Result<Vec<SequenceMatch>> {
    
    debug!("Processing cell {:?}: source[{}..{}] × target[{}..{}]",
        cell.id, 
        cell.source_range.start, cell.source_range.end,
        cell.target_range.start, cell.target_range.end
    );
    
    // Get source ngrams for this cell
    let source_partition = &source_sequences[cell.source_range.clone()];
    
    // ========================================================================
    // ✅ OPTIMIZATION #1: PRE-LOAD ALL SOURCE PREFIXES IN ONE BATCH
    // ========================================================================
    let source_prefix_map = source_storage.get_prefix_bytes_batch(source_partition)?;
    
    debug!("Cell {:?}: Pre-loaded {} source prefixes in one batch",
        cell.id, source_prefix_map.len());
    
    // ========================================================================
    // PRE-LOAD ALL TARGET PREFIX MATCHES
    // ========================================================================
    let prefix_length = config.generator.prefix_index_depth;
    let prefix_map = load_prefix_matches_for_range(
        source_partition,
        source_storage,
        target_storage,
        &cell.target_range,
        prefix_length,
    )?;
    
    if prefix_map.is_empty() {
        debug!("No prefix matches found for cell {:?}", cell.id);
        return Ok(Vec::new());
    }
    
    // Track statistics
    let total_targets_loaded: usize = prefix_map.values()
        .map(|v| v.len())
        .sum();
    debug!("Cell {:?}: Loaded {} total targets across {} prefixes",
        cell.id, total_targets_loaded, prefix_map.len());

    // ========================================================================
    // ✅ OPTIMIZATION #2: PRE-LOAD ALL TARGET DATA FOR THE ENTIRE CELL
    // This eliminates redundant DB calls when multiple sources compare 
    // against the same targets
    // ========================================================================
    
    // Collect ALL unique targets that ANY source in this cell will compare against
    let all_target_seqs: AHashSet<u64> = prefix_map.values()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let all_target_seqs_vec: Vec<u64> = all_target_seqs.into_iter().collect();
    
    debug!("Cell {:?}: Pre-loading data for {} unique targets across all sources", 
        cell.id, all_target_seqs_vec.len());
    
    // Batch load ALL target metadata in one call
    let target_metadata_map = target_storage.get_ngram_metadata_batch(&all_target_seqs_vec)?
        .into_iter()
        .map(|m| (m.sequence_number, m))
        .collect::<AHashMap<_, _>>();
    
    // Batch load target frequencies if using precomputed features
    let target_freqs_map = if config.matcher.use_precomputed_similarity_features {
        Some(target_storage.get_char_frequencies_batch(&all_target_seqs_vec)?)
    } else {
        None
    };
    
    // Batch load target text if NOT using precomputed features
    let target_text_map = if !config.matcher.use_precomputed_similarity_features {
        Some(target_storage.get_text_bytes_batch(&all_target_seqs_vec)?)
    } else {
        None
    };
    
    let db_calls = 1 + if target_freqs_map.is_some() { 1 } else { 0 } + if target_text_map.is_some() { 1 } else { 0 };
    debug!("Cell {:?}: Pre-loaded all target data in {} batch calls (vs {} calls without optimization)", 
        cell.id, db_calls, source_partition.len() * 3);

    // Check ONCE if tracing is enabled at all
    let tracing_enabled = !config.debug.traced_phrase.is_empty() 
        && log::log_enabled!(log::Level::Debug);
    
    // ========================================================================
    // MAIN PROCESSING LOOP - Now with zero redundant DB calls!
    // ========================================================================
    let mut cell_matches = Vec::new();
    let mut sources_with_matches = 0;
    let mut traced_match_count = 0;
    
    for source_seq in source_partition {
                
        // Only check if tracing is enabled
        let is_traced = tracing_enabled 
            && is_traced_sequence(*source_seq, Some(source_storage), &config.debug.traced_phrase);
        
        // Get prefix from pre-loaded map (zero DB overhead)
        let source_prefix = match source_prefix_map.get(source_seq) {
            Some(prefix_bytes) if !prefix_bytes.is_empty() => prefix_bytes,
            _ => continue,
        };
        
        // Look up pre-loaded candidates for this prefix
        if let Some(prefix_candidates) = prefix_map.get(source_prefix) {
            // Apply remaining filters (metadata, quick, length) if enabled
            let filtered_candidates = if config.matcher.enable_metadata_filtering.unwrap_or(false)
                || config.matcher.enable_quick_filtering.unwrap_or(false)
                || config.matcher.enable_length_filtering.unwrap_or(false) {
                apply_additional_filters(
                    *source_seq,
                    prefix_candidates,
                    source_storage,
                    target_storage,
                    config
                )?
            } else {
                prefix_candidates.clone()
            };
            
            if !filtered_candidates.is_empty() {
                // ✅ Call find_matching_ngrams with pre-loaded data
                // (We'll modify this function signature in the next step)
                let matches = similarity_calc.find_matching_ngrams(
                    *source_seq,
                    &filtered_candidates,
                    source_storage,
                    target_storage,
                    &config,
                    Some(&target_metadata_map),
                    target_freqs_map.as_ref(),
                    target_text_map.as_ref(),
                )?;
                
                // Log traced matches
                if is_traced {
                    if !matches.is_empty() {
                        info!("TRACE [grid cell {:?}]: Source {} found {} matches", 
                            cell.id, source_seq, matches.len());
                        traced_match_count += matches.len();
                    } else {
                        warn!("TRACE [grid cell {:?}]: Source {} found NO matches after filtering!", 
                            cell.id, source_seq);
                    }
                }
                
                if !matches.is_empty() {
                    sources_with_matches += 1;
                    cell_matches.extend(matches);
                }
            }
        }
    }
    
    // Summary log only if tracing was enabled and we found traced matches
    if tracing_enabled && traced_match_count > 0 {
        info!("TRACE [grid cell {:?}]: Total {} traced matches found", 
            cell.id, traced_match_count);
    }

    // Deduplicate matches within this cell
    if !cell_matches.is_empty() {
        cell_matches.sort_by_key(|m| (m.source_sequence, m.target_sequence));
        cell_matches.dedup_by_key(|m| (m.source_sequence, m.target_sequence));
    }
    
    debug!("Cell {:?}: Found {} matches from {} sources ({:.1}% match rate)",
        cell.id, cell_matches.len(), sources_with_matches,
        if source_partition.len() > 0 {
            sources_with_matches as f64 / source_partition.len() as f64 * 100.0
        } else {
            0.0
        });
        
    Ok(cell_matches)
}

// Helper function to apply non-prefix filters
fn apply_additional_filters(
    source_seq: u64,
    candidates: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    config: &AlNaqlConfig,
) -> Result<Vec<u64>> {
    let mut filtered = candidates.to_vec();
    
    if config.matcher.enable_metadata_filtering.unwrap_or(false) {
        filtered = apply_metadata_filtering(
            source_seq,
            filtered,      // Note: not &filtered, the function takes ownership
            source_storage,
            target_storage,
        )?;
    }
    
    if config.matcher.enable_quick_filtering.unwrap_or(false) {
        filtered = apply_quick_filtering(
            source_seq,
            filtered,
            source_storage,
            target_storage,
        )?;
    }
    
    if config.matcher.enable_length_filtering.unwrap_or(false) {
        filtered = apply_length_filtering(
            source_seq,
            filtered,
            source_storage,
            target_storage,
            &config.matcher,
        )?;
    }
    
    Ok(filtered)
}