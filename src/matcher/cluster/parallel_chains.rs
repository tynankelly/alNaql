// src/matcher/cluster/parallel_chains.rs
//
// This file implements parallel processing for dense region finding
// using an adaptive chunking strategy that respects natural boundaries in the data.

use rayon::prelude::*;
use log::{info, debug, trace, warn};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::cmp::max;
use ahash::{AHashMap, AHashSet};

use crate::error::Result;
use crate::matcher::types::{NGramMatch, MatchCandidate};
use crate::config::subsystems::generator::NGramType;
use crate::storage::StorageBackend;
use super::matcher::ClusteringMatcher;

pub struct ParallelProcessingStats {
    pub candidates_found: std::sync::atomic::AtomicUsize,
    pub chains_processed: std::sync::atomic::AtomicUsize,
    pub chains_rejected_banal: std::sync::atomic::AtomicUsize,
    pub chains_rejected_isnad: std::sync::atomic::AtomicUsize,
}

impl ParallelProcessingStats {
    pub fn new() -> Self {
        Self {
            candidates_found: std::sync::atomic::AtomicUsize::new(0),
            chains_processed: std::sync::atomic::AtomicUsize::new(0),
            chains_rejected_banal: std::sync::atomic::AtomicUsize::new(0),
            chains_rejected_isnad: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    pub fn log_summary(&self) {
        info!("Parallel processing stats:");
        info!("  Total chains processed: {}", self.chains_processed.load(std::sync::atomic::Ordering::Relaxed));
        info!("  Chains rejected as banal: {}", self.chains_rejected_banal.load(std::sync::atomic::Ordering::Relaxed));
        info!("  Chains rejected as short isnads: {}", self.chains_rejected_isnad.load(std::sync::atomic::Ordering::Relaxed));
        info!("  Final candidates: {}", self.candidates_found.load(std::sync::atomic::Ordering::Relaxed));
    }
}
pub(super) fn find_dense_regions_parallel<S: StorageBackend>(
    matcher: &ClusteringMatcher<S>,
    matches: &[NGramMatch]
) -> Result<Vec<MatchCandidate>> {
    let start_time = Instant::now();
    
    // Log which matching method we're using
    if matcher.matcher_config.proximity_matching {
        info!("\nStarting parallel proximity-based matching with {} matches", matches.len());
        info!("Using proximity distance: {}", matcher.matcher_config.proximity_distance);
    } else {
        info!("\nStarting parallel sequential matching with {} matches", matches.len());
        info!("Using max gap: {}", matcher.matcher_config.max_gap);
    }
    
    // Create stats tracking object
    let stats = Arc::new(ParallelProcessingStats::new());
    
    // Early return for small match sets
    if matches.len() < matcher.processor_config.parallel_dense_threshold {
        debug!("Match count below threshold, using sequential processing");
        return matcher.find_dense_regions(matches);
    }
    
    // Get effective parameters based on matching type and ngram type
    let (effective_gap, boundary_type) = if matcher.matcher_config.proximity_matching {
        // For proximity matching, use proximity_distance parameter
        let gap = matcher.matcher_config.proximity_distance;
        (gap, "proximity distance")
    } else {
        // For sequential matching, use max_gap parameter adjusted for ngram type
        let gap = match matcher.ngram_type {
            NGramType::Character => matcher.matcher_config.max_gap,
            NGramType::Word => matcher.matcher_config.max_gap * 2,  // Double for word ngrams
        };
        (gap, "sequential gap")
    };
    
    debug!("Using effective {}: {} for ngram type: {:?}", 
           boundary_type, effective_gap, matcher.ngram_type);
    
    // Determine optimal chunk size parameters based on system resources
    let cpu_count = num_cpus::get();
    let target_chunk_count = cpu_count * 2; // Aim for 2 chunks per CPU for better load balancing
    
    let min_chunk_size = matcher.processor_config.min_dense_chunk_size.max(1000);
    let adaptive_chunk_size = max(min_chunk_size, matches.len() / target_chunk_count);
    let max_chunk_size = matcher.processor_config.max_dense_chunk_size.min(100_000);
    
    let chunk_size = adaptive_chunk_size.min(max_chunk_size);
    
    debug!("Using adaptive chunk size: {} (min: {}, max: {})", 
           chunk_size, min_chunk_size, max_chunk_size);
    
    // Find natural chunk boundaries - the boundary method depends on the matching type
    let boundaries = if matcher.matcher_config.proximity_matching {
        // For proximity matching, we use grid-based boundaries
        find_proximity_boundaries(matches, effective_gap, chunk_size)
    } else {
        // For sequential matching, we use source position gaps
        find_natural_boundaries(matches, effective_gap, chunk_size)
    };
    
    debug!("Split matches into {} chunks using {} boundaries", 
          boundaries.len(), if matcher.matcher_config.proximity_matching { "proximity" } else { "sequential" });
    
    // Track progress with atomic counter
    let progress_counter = Arc::new(AtomicUsize::new(0));
    let total_chunks = boundaries.len();
    
    // Clone the stats for sharing across threads
    let stats_clone = Arc::clone(&stats);
    
    // Process chunks in parallel using rayon
    let chunk_results: Vec<Vec<MatchCandidate>> = boundaries.par_iter()
        .map(|(start_idx, end_idx)| {
            let chunk_start_time = Instant::now();
            let chunk_matches = &matches[*start_idx..*end_idx];
            trace!("Processing chunk with {} matches (indices {}..{})", 
                  chunk_matches.len(), start_idx, end_idx);
            
            // Pass the stats counter to the function that handles the chunk
            let result = process_chunk_with_stats(matcher, chunk_matches, &stats_clone);
            
            // Update progress counter and log progress
            let processed = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if processed % 10 == 0 || processed == total_chunks {
                info!("Processed {}/{} chunks ({:.1}%) - Last chunk: {} matches in {:?}", 
                     processed, total_chunks, 
                     (processed as f64 / total_chunks as f64) * 100.0,
                     chunk_matches.len(), chunk_start_time.elapsed());
            } else {
                debug!("Processed chunk {}/{} with {} matches in {:?}", 
                     processed, total_chunks, chunk_matches.len(), chunk_start_time.elapsed());
            }
            
            result
        })
        .collect::<Result<Vec<Vec<MatchCandidate>>>>()?;
    
    // Merge results from all chunks
    let merged_candidates = merge_chunk_results(chunk_results);
    
    // Log the statistics summary
    stats.log_summary();
    
    info!("Parallel {} finding complete: found {} candidates in {:?}", 
         if matcher.matcher_config.proximity_matching { "proximity" } else { "sequential" },
         merged_candidates.len(), start_time.elapsed());
    
    Ok(merged_candidates)
}

fn process_chunk_with_stats<S: StorageBackend>(
    matcher: &ClusteringMatcher<S>,
    chunk_matches: &[NGramMatch],
    stats: &Arc<ParallelProcessingStats>
) -> Result<Vec<MatchCandidate>> {
    // Use the matching method specified in the configuration
    let candidates = matcher.find_dense_regions_chunk(chunk_matches, Some(stats))?;
    
    // Estimate chains processed based on typical chain-to-match ratio
    // This needs to be adjusted for proximity matching which might create different numbers of clusters
    let estimated_chains = if matcher.matcher_config.proximity_matching {
        // Proximity matching tends to create fewer but larger clusters
        (chunk_matches.len() as f64 * 0.10).ceil() as usize
    } else {
        // Sequential matching creates more smaller chains
        (chunk_matches.len() as f64 * 0.15).ceil() as usize
    };
    
    stats.chains_processed.fetch_add(estimated_chains, Ordering::Relaxed);

    Ok(candidates)
}
fn find_proximity_boundaries(
    matches: &[NGramMatch],
    proximity_distance: usize,
    target_chunk_size: usize
) -> Vec<(usize, usize)> {
    // Return early if matches is too small for chunking
    if matches.len() <= target_chunk_size {
        return vec![(0, matches.len())];
    }
    
    // Create a grid-based index of matches
    let mut grid: AHashMap<(usize, usize), Vec<usize>> = AHashMap::new();
    let grid_size = proximity_distance * 2;  // Use 2x proximity_distance as cell size
    
    // Place each match into a grid cell based on its source and target positions
    for (idx, m) in matches.iter().enumerate() {
        let cell_x = m.source_position / grid_size;
        let cell_y = m.target_position / grid_size;
        grid.entry((cell_x, cell_y)).or_insert_with(Vec::new).push(idx);
    }
    
    // Group cells into chunks
    let mut visited_cells = AHashSet::new();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;
    
    // Sort grid cells for deterministic processing
    let mut grid_cells: Vec<_> = grid.keys().collect();
    grid_cells.sort();
    
    for &cell in &grid_cells {
        if visited_cells.contains(&cell) {
            continue;
        }
        
        // Mark cell as visited
        visited_cells.insert(cell);
        
        // Fixed: Create empty vector before unwrap_or to avoid temporary value issue
        let empty_vec: Vec<usize> = Vec::new();
        let matches_in_cell = grid.get(&cell).unwrap_or(&empty_vec);
        current_chunk.extend(matches_in_cell);
        current_size += matches_in_cell.len();
        
        // If we've reached target size, finalize this chunk
        if current_size >= target_chunk_size {
            if !current_chunk.is_empty() {
                // Sort indices for consistent chunk boundaries
                current_chunk.sort();
                let start_idx = *current_chunk.first().unwrap();
                let end_idx = *current_chunk.last().unwrap() + 1;
                chunks.push((start_idx, end_idx));
            }
            
            current_chunk = Vec::new();
            current_size = 0;
        }
    }
    
    // Add any remaining matches as a final chunk
    if !current_chunk.is_empty() {
        current_chunk.sort();
        let start_idx = *current_chunk.first().unwrap();
        let end_idx = *current_chunk.last().unwrap() + 1;
        chunks.push((start_idx, end_idx));
    }
    
    // Sort boundaries by start index
    chunks.sort_by_key(|(start, _)| *start);
    
    // Ensure there are no gaps in coverage
    validate_and_fix_boundaries(chunks, matches.len())
}

fn validate_and_fix_boundaries(
    chunks: Vec<(usize, usize)>, 
    total_matches: usize
) -> Vec<(usize, usize)> {
    // Convert input chunks to sorted, contiguous chunks
    let mut result = Vec::new();
    let mut covered = AHashSet::new();
    
    // Add matches from each chunk to our coverage tracking
    for (start, end) in chunks {
        for idx in start..end {
            covered.insert(idx);
        }
        result.push((start, end));
    }
    
    // Check for any gaps in coverage
    if covered.len() < total_matches {
        warn!("Detected gaps in chunk coverage: {} out of {} indices covered", 
             covered.len(), total_matches);
        
        // Add a chunk covering all matches to ensure no gaps
        result.push((0, total_matches));
    }
    
    debug!("Created {} chunks with boundary validation", result.len());
    result
}
fn find_natural_boundaries(
    matches: &[NGramMatch],
    max_gap: usize,
    target_chunk_size: usize
) -> Vec<(usize, usize)> {
    // Return early if matches is too small for chunking
    if matches.len() <= target_chunk_size {
        return vec![(0, matches.len())];
    }
    
    // Create a mapping of matches by source position for efficient lookup
    let mut positions: Vec<(usize, usize)> = matches.iter()
        .enumerate()
        .map(|(idx, m)| (idx, m.source_position))
        .collect();
    
    // Sort by source position while maintaining original indices
    positions.sort_by_key(|(_idx, pos)| *pos);
    
    // Find natural gaps based on source positions
    let mut boundaries = Vec::new();
    let mut chunk_start_idx = 0;
    let mut last_pos = positions[0].1;
    
    for i in 1..positions.len() {
        let (_idx, pos) = positions[i];
        let gap = pos.saturating_sub(last_pos);
        
        // Consider a boundary if:
        // 1. We find a natural gap > max_gap, OR
        // 2. The current chunk size exceeds target size
        let current_chunk_size = i - chunk_start_idx;
        let is_natural_boundary = gap > max_gap;
        let is_size_boundary = current_chunk_size >= target_chunk_size;
        
        if is_natural_boundary || is_size_boundary {
            // Avoid creating very small chunks that would have processing overhead
            if current_chunk_size >= target_chunk_size / 4 {
                boundaries.push((chunk_start_idx, i));
                chunk_start_idx = i;
            }
        }
        
        last_pos = pos;
    }
    
    // Add the final chunk if not empty
    if chunk_start_idx < positions.len() {
        boundaries.push((chunk_start_idx, positions.len()));
    }
    
    // Convert sorted indices back to original indices
    let mut result: Vec<(usize, usize)> = boundaries.into_iter()
        .map(|(start, end)| {
            let start_idx = positions[start].0;
            let end_idx = if end < positions.len() { 
                positions[end].0 
            } else { 
                matches.len() 
            };
            (start_idx, end_idx)
        })
        .collect();
    
    // Sort boundaries by original index
    result.sort_by_key(|(start, _)| *start);
    
    // Validate boundaries to ensure complete coverage
    let mut validated = Vec::new();
    let mut covered_indices = AHashSet::new();
    
    for (start, end) in result {
        // Add all indices in this range to the covered set
        for idx in start..end {
            covered_indices.insert(idx);
        }
        validated.push((start, end));
    }
    
    // Check for any gaps in coverage
    if covered_indices.len() < matches.len() {
        warn!("Detected gaps in chunk coverage: {} out of {} indices covered", 
             covered_indices.len(), matches.len());
        
        // Add a chunk covering all matches to ensure no gaps
        validated.push((0, matches.len()));
    }
    
    debug!("Created {} chunks with natural boundaries", validated.len());
    trace!("Chunk boundaries: {:?}", validated);
    
    validated
}

/// Merge results from multiple chunks, handling potential duplicates.
/// This function combines the candidates from all chunks and deduplicates
/// any overlapping or duplicate candidates that might occur at chunk boundaries.
fn merge_chunk_results(chunk_results: Vec<Vec<MatchCandidate>>) -> Vec<MatchCandidate> {
    // Early return for single or empty chunk
    let total_chunks = chunk_results.len();
    if total_chunks == 0 {
        return Vec::new();
    } else if total_chunks == 1 {
        return chunk_results.into_iter().next().unwrap();
    }
    
    // Flatten all results
    let all_candidates_count: usize = chunk_results.iter().map(|c| c.len()).sum();
    let mut all_candidates = Vec::with_capacity(all_candidates_count);
    
    for chunk_candidates in chunk_results {
        all_candidates.extend(chunk_candidates);
    }
    
    debug!("Merging {} total candidates from {} chunks", all_candidates.len(), total_chunks);
    
    // Sort candidates for efficient deduplication
    all_candidates.sort_by(|a, b| {
        a.source_range.start.cmp(&b.source_range.start)
            .then_with(|| a.source_range.end.cmp(&b.source_range.end))
            .then_with(|| a.target_range.start.cmp(&b.target_range.start))
    });
    
    // Deduplicate by removing exact duplicates
    if all_candidates.len() > 1 {
        let original_count = all_candidates.len();
        let mut unique_candidates = Vec::with_capacity(original_count);
        
        let mut prev_candidate: Option<&MatchCandidate> = None;
        
        for candidate in all_candidates {
            let is_duplicate = prev_candidate.as_ref().map_or(false, |prev| {
                prev.source_range.start == candidate.source_range.start &&
                prev.source_range.end == candidate.source_range.end &&
                prev.target_range.start == candidate.target_range.start &&
                prev.target_range.end == candidate.target_range.end
            });
            
            if !is_duplicate {
                unique_candidates.push(candidate.clone());
                prev_candidate = Some(&unique_candidates.last().unwrap());
            }
        }
        
        let duplicates_removed = original_count - unique_candidates.len();
        if duplicates_removed > 0 {
            debug!("Removed {} duplicates from merged results ({} → {})", 
                 duplicates_removed, original_count, unique_candidates.len());
        }
        
        unique_candidates
    } else {
        all_candidates
    }
}