// matcher/clusters/chains.rs
// Unified chain building and dense region detection

use rayon::prelude::*;
use bit_vec::BitVec;
use std::time::Instant;
use indicatif::ProgressBar;
use log::{info, debug, warn, trace, error, log_enabled, Level};

use crate::error::{Result, Error};
use crate::matcher::is_traced_sequence;
use crate::matcher::types::{SequenceMatch, SequenceCluster};
use crate::config::ClusteringMode;
use crate::config::execution::ExecutionConfig;
use crate::storage::LMDBStorage;
use super::spatial::proximate_chains_spatial;
use crate::AlNaqlConfig;

// ============================================================================
// MAIN PUBLIC API
// ============================================================================

/// Single entry point for finding dense regions in matches
/// Delegates to parallel or sequential processing based on match count
pub fn find_dense_regions(
    matches: &[SequenceMatch],
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,
    progress_bar: Option<&ProgressBar>,
) -> Result<Vec<SequenceCluster>> {
    // Early return for empty matches
    if matches.is_empty() {
        debug!("No matches to process for dense region finding");
        return Ok(Vec::new());
    }

    let start_time = Instant::now();
            
    // Decide whether to use parallel processing
    let use_parallel = matches.len() >= config.execution.parallel_dense_threshold;
    
    // Process matches with appropriate strategy
    let candidates = if use_parallel {
        // Get current thread pool info for logging
        let thread_count = rayon::current_num_threads();
        debug!("Processing strategy: PARALLEL with {} threads", thread_count);
        
        // Just call process_parallel directly
        process_parallel(matches, config, source_storage)?
    } else {
        debug!("Processing strategy: SEQUENTIAL (no thread pool needed)");
        process_sequential(matches, &config, source_storage, progress_bar)?
    };
    
    // Log results and timing
    trace!("Found {} dense region candidates in {:?}", 
        candidates.len(), start_time.elapsed());
        
    // CHECK FOR TRACED N-GRAM IN FINAL CANDIDATES
    if !config.debug.traced_phrase.is_empty() && log_enabled!(Level::Info) {
        if let Some(storage) = source_storage {
            let candidates_with_traced: Vec<_> = candidates.iter()
                .enumerate()
                .filter(|(_, c)| {
                    // Check if any sequence in the cluster range contains traced phrase
                    (c.source_start..=c.source_end).any(|seq| {
                        is_traced_sequence(seq, Some(storage), &config.debug.traced_phrase)
                    })
                })
                .collect();
                
            if !candidates_with_traced.is_empty() {
                info!("TRACE [dense_regions FINAL]: {} of {} candidates contain traced n-gram '{}'",
                    candidates_with_traced.len(), candidates.len(), config.debug.traced_phrase);
                    
                // Log details about first few candidates with traced ngram
                for (cand_idx, candidate) in candidates_with_traced.iter().take(3) {
                    // Count how many sequences in range contain traced phrase
                    let traced_seq_count = (candidate.source_start..=candidate.source_end)
                        .filter(|&seq| is_traced_sequence(seq, Some(storage), &config.debug.traced_phrase))
                        .count();
                        
                    debug!("TRACE [dense_regions FINAL]: Candidate #{} has {} traced sequences (cluster has {} matches total): source={}..{}, target={}..{}",
                        cand_idx, traced_seq_count, candidate.match_count,
                        candidate.source_start, candidate.source_end,
                        candidate.target_start, candidate.target_end
                    );
                }
            } else {
                warn!("TRACE [dense_regions FINAL]: NO final candidates contain traced n-gram '{}' (checked {} candidates)",
                    config.debug.traced_phrase, candidates.len());
                    
                // Check if input had traced matches but none survived
                if matches.iter().any(|m| is_traced_sequence(m.source_sequence, Some(storage), &config.debug.traced_phrase)) {
                    error!("TRACE [dense_regions FINAL]: Input had traced matches but no candidates survived! Check density/filtering logic.");
                }
            }
        }
    }
    
    trace!("Found {} dense region candidates in {:?}", 
        candidates.len(), start_time.elapsed());
    
    Ok(candidates)
}

// ============================================================================
// PARALLEL PROCESSING
// ============================================================================

/// Process matches in parallel using smart chunking
fn process_parallel(
    matches: &[SequenceMatch],
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,
) -> Result<Vec<SequenceCluster>> {

    let start_time = Instant::now();
    
    // Calculate optimal chunk size
    let chunk_size = calculate_optimal_chunk_size(matches.len(), &config.execution);
    
    // Find chunk boundaries that respect natural gaps in the data
    let boundary_gap = match config.matcher.clustering_mode {
        ClusteringMode::Sequential => config.matcher.max_gap,
        ClusteringMode::Proximity => config.matcher.proximity_distance,
        ClusteringMode::Both => config.matcher.max_gap.max(config.matcher.proximity_distance),
    };

    // Use unified boundary finding with the appropriate gap
    let chunks = find_boundaries(matches, boundary_gap, chunk_size);
        
    trace!("Processing {} matches in {} parallel chunks", matches.len(), chunks.len());
    
    // Process chunks in parallel
    let chunk_results: Vec<Vec<SequenceCluster>> = chunks  
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, (start, end))| {
            let chunk_start_time = Instant::now();
            let chunk_matches = &matches[*start..*end];
            
            debug!("Processing chunk {}/{} with {} matches",
                chunk_idx + 1, chunks.len(), chunk_matches.len());
            
            // Process this chunk
            let result = process_chunk(chunk_matches, &config, source_storage);
            
            // Log timing for debugging
            debug!("Chunk {}/{} processed in {:?}", 
                chunk_idx + 1, chunks.len(), chunk_start_time.elapsed());
            
            result
        })
        .collect::<Result<Vec<_>>>()?;
    
    // Merge results from all chunks
    let merged_candidates = merge_chunk_results(chunk_results);
    
    trace!("Parallel processing complete: found {} candidates in {:?}", 
        merged_candidates.len(), start_time.elapsed());
    
    Ok(merged_candidates)
}

/// Calculate optimal chunk size based on data characteristics
fn calculate_optimal_chunk_size(
    total_matches: usize,
    config: &ExecutionConfig,
) -> usize {
    // Get configured bounds
    let min_size = config.min_dense_chunk_size;
    let max_size = config.max_dense_chunk_size;
    
    // Calculate ideal size (aim for ~8 chunks for good parallel utilization)
    let ideal_size = total_matches / 8;
    
    // Clamp to configured bounds
    let chunk_size = ideal_size.max(min_size).min(max_size);
    
    debug!("Calculated chunk size: {} (total: {}, min: {}, max: {})",
        chunk_size, total_matches, min_size, max_size);
    
    chunk_size
}

/// Find natural boundaries in sequential matches for optimal chunking
fn find_boundaries(
    matches: &[SequenceMatch],
    max_gap: usize,
    target_chunk_size: usize,
) -> Vec<(usize, usize)> {
    if matches.len() <= target_chunk_size {
        return vec![(0, matches.len())];
    }
    
    let mut boundaries = Vec::new();
    let mut current_start = 0;
    let mut last_position = matches[0].source_sequence;
    
    for (i, match_item) in matches.iter().enumerate().skip(1) {
        let gap = match_item.source_sequence.saturating_sub(last_position);
        let max_gap = max_gap as u64;
        // Check if we've found a natural boundary (large gap)
        if gap > max_gap * 2 {
            // Check if this chunk would be reasonably sized
            let chunk_size = i - current_start;
            if chunk_size >= target_chunk_size / 2 {
                boundaries.push((current_start, i));
                current_start = i;
            }
        }
        // Or if we've reached target chunk size and have any gap
        else if i - current_start >= target_chunk_size && gap > 1 {
            boundaries.push((current_start, i));
            current_start = i;
        }
        
        last_position = match_item.source_sequence;
    }
    
    // Add the final chunk
    if current_start < matches.len() {
        boundaries.push((current_start, matches.len()));
    }
    
    // If we ended up with too many small chunks, consolidate
    if boundaries.len() > 1 {
        let avg_size = matches.len() / boundaries.len();
        if avg_size < target_chunk_size / 2 {
            boundaries = consolidate_small_chunks(boundaries, target_chunk_size);
        }
    }
    
    debug!("Found {} natural boundaries for {} matches", 
        boundaries.len(), matches.len());
    
    boundaries
}

/// Consolidate small chunks into larger ones
fn consolidate_small_chunks(
    boundaries: Vec<(usize, usize)>,
    target_size: usize,
) -> Vec<(usize, usize)> {
    let mut consolidated = Vec::new();
    let mut current_start = boundaries[0].0;
    let mut current_end = boundaries[0].1;
    
    for (start, end) in boundaries.into_iter().skip(1) {
        if current_end - current_start < target_size {
            // Extend current chunk
            current_end = end;
        } else {
            // Save current chunk and start new one
            consolidated.push((current_start, current_end));
            current_start = start;
            current_end = end;
        }
    }
    
    // Add the final consolidated chunk
    consolidated.push((current_start, current_end));
    
    consolidated
}

/// Merge results from multiple chunks and remove duplicates
fn merge_chunk_results(chunks: Vec<Vec<SequenceCluster>>) -> Vec<SequenceCluster> {
    let total_size: usize = chunks.iter().map(|c| c.len()).sum();
    let mut merged = Vec::with_capacity(total_size);
    
    for chunk_candidates in chunks {
        merged.extend(chunk_candidates);
    }
    
    // Sort by source range start for potential deduplication
    merged.sort_by_key(|c| c.source_start);
    
    // Note: We could add deduplication logic here if needed
    // For now, chunks shouldn't produce overlapping candidates due to boundary detection
    
    debug!("Merged {} candidates from parallel chunks", merged.len());
    
    merged
}
// ============================================================================
// SEQUENTIAL PROCESSING
// ============================================================================

// Similarly for sequential processing
fn process_sequential(
    matches: &[SequenceMatch],
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,
    progress_bar: Option<&ProgressBar>,
) -> Result<Vec<SequenceCluster>> {
    debug!("Processing {} matches sequentially", matches.len());
    
    if let Some(pb) = progress_bar {
        pb.set_length(1);
        pb.set_position(0);
        pb.set_message("Processing sequentially (single chunk)");
    }
    
    let result = process_chunk(matches, &config, source_storage)?;
    
    if let Some(pb) = progress_bar {
        pb.set_position(1);
        pb.set_message(format!("Found {} candidates", result.len()));
    }
    
    Ok(result)
}

// ============================================================================
// CORE PROCESSING (shared by parallel and sequential)
// ============================================================================

/// Process a chunk of matches to find dense regions
fn process_chunk(
    matches: &[SequenceMatch],
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,
) -> Result<Vec<SequenceCluster>> {
    if matches.is_empty() {
        return Ok(Vec::new());
    }
    
    // Choose clustering algorithm based on clustering mode
    let clusters = match config.matcher.clustering_mode {
        ClusteringMode::Sequential => {
            sequential_chains(
                matches, 
                config.matcher.max_gap, 
                &config.debug.traced_phrase, 
                source_storage
            )
        },
        ClusteringMode::Proximity => {
            // Use spatial indexing
            debug!("Using spatial indexing for proximity clustering");
            proximate_chains_spatial(
                matches,
                config.matcher.proximity_distance,
                &config.debug.traced_phrase,
                source_storage,
            )
        },
        ClusteringMode::Both => {
            both_methods_clustering(matches, config, source_storage)
        }
    };
    
    debug!("Found {} clusters from {} matches", clusters.len(), matches.len());
    
    let mut candidates = Vec::new();
    
    for (i, cluster_indices) in clusters.iter().enumerate() {
        
        // Create candidate based on clustering mode
        let candidate_result = match config.matcher.clustering_mode {
            ClusteringMode::Sequential => {
                sequential_candidate(&cluster_indices, matches)
            },
            ClusteringMode::Proximity => {
                proximity_candidate(&cluster_indices, matches)
            },
            ClusteringMode::Both => {
                // Use proximity candidate creation - it works for any cluster type
                // (finds min/max positions rather than assuming order)
                proximity_candidate(&cluster_indices, matches)
            }
        };
        
        // Handle candidate creation errors like legacy code - log and skip
        let candidate = match candidate_result {
            Ok(candidate) => candidate,
            Err(e) => {
                trace!("Failed to create candidate for cluster #{}: {}", i, e);
                continue; // Skip this cluster and continue to the next
            }
        };
        
        // Check density
        let is_dense = check_density(&candidate, config, source_storage)?;
        
        if is_dense {
            candidates.push(candidate);
        } else {
            trace!("Cluster {} rejected for insufficient density", i);
        }
    }
    
    debug!("Chunk produced {} candidates", candidates.len());
    
    Ok(candidates)
}

/// Run both sequential and proximity clustering and combine results
fn both_methods_clustering(
    matches: &[SequenceMatch],
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,
) -> Vec<Vec<usize>> {
    // Run sequential clustering
    let sequential_clusters = sequential_chains(
        matches, 
        config.matcher.max_gap, 
        &config.debug.traced_phrase,
        source_storage,
    );
    
    // Run proximity clustering  
    let proximity_clusters = proximate_chains_spatial(
        matches,
        config.matcher.proximity_distance,
        &config.debug.traced_phrase,
        source_storage,
    );
    
    // Log statistics
    if log_enabled!(log::Level::Debug) {
        debug!(
            "Both-mode clustering: {} sequential clusters, {} proximity clusters, {} total",
            sequential_clusters.len(),
            proximity_clusters.len(),
            sequential_clusters.len() + proximity_clusters.len()
        );
        
    }
    
    // Combine both cluster lists
    let mut combined = sequential_clusters;
    combined.extend(proximity_clusters);
    
    combined
}

/// Unified density check for all candidates regardless of clustering method
fn check_density(
    candidate: &SequenceCluster,     // Changed from &MatchCandidate
    config: &AlNaqlConfig,
    source_storage: Option<&LMDBStorage>,  // Added for tracing
) -> Result<bool> {
    // Check if this candidate contains traced phrase for debugging
        let has_traced = if !config.debug.traced_phrase.is_empty() 
        && log::log_enabled!(log::Level::Debug) 
        && source_storage.is_some() 
    {
        let storage = source_storage.unwrap();
        (candidate.source_start..=candidate.source_end).any(|seq| 
            is_traced_sequence(seq, Some(storage), &config.debug.traced_phrase)
        )
    } else {
        false
    };
    
    // Minimum number of matches required
    if candidate.match_count < 2 {
        if has_traced {
            debug!("TRACE [density]: REJECTED - Too few matches: {} (min: 2)",
                candidate.match_count);
        }
        return Ok(false);
    }
    
    // Calculate range sizes
    let source_range = candidate.source_end - candidate.source_start + 1;  // +1 for inclusive
    let target_range = candidate.target_end - candidate.target_start + 1;  // +1 for inclusive
    
    // Avoid division by zero
    if source_range == 0 || target_range == 0 {
        if has_traced {
            debug!("TRACE [density]: REJECTED - Zero-length range detected");
        }
        return Ok(false);
    }
    
    // Calculate density as matches per unit length (using average of both ranges)
    let avg_range = (source_range + target_range) as f64 / 2.0;
    let density = candidate.match_count as f64 / avg_range;
    
    if has_traced {
        debug!("TRACE [density]: Density = {:.3} (matches={}, avg_range={:.1}, threshold={:.3})",
            density, candidate.match_count, avg_range, config.matcher.min_density);
    }
    
    // Single threshold for all candidates
    Ok(density >= config.matcher.min_density)
}
// ============================================================================
// CLUSTERING ALGORITHMS
// ============================================================================

/// Group matches into chains based on sequential proximity
fn sequential_chains(
    matches: &[SequenceMatch],
    max_gap: usize,
    traced_phrase: &str,
    source_storage: Option<&LMDBStorage>,
) -> Vec<Vec<usize>> {
    if matches.is_empty() {
        return Vec::new();
    }

    // Search for traced-ngram
    let traced_matches: Vec<(usize, u64)> = if log_enabled!(log::Level::Debug) && !traced_phrase.is_empty() {
        // Check if storage is provided
        if let Some(storage) = source_storage {
            let mut traced = Vec::new();
            
            for (idx, m) in matches.iter().enumerate() {
                // Load source text to check for traced phrase
                if let Ok(text_map) = storage.get_text_bytes_batch(&[m.source_sequence]) {
                    if let Some(bytes) = text_map.get(&m.source_sequence) {
                        if let Ok(text) = std::str::from_utf8(bytes) {
                            if text.contains(traced_phrase) {
                                traced.push((idx, m.source_sequence));
                            }
                        }
                    }
                }
            }
            traced
        } else {
            // No storage provided, can't trace
            Vec::new()
        }
    } else {
        Vec::new() // Empty vec if logging disabled or no traced phrase
    };
    
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut assigned = BitVec::from_elem(matches.len(), false);
    
    for i in 0..matches.len() {
        if assigned[i] {
            continue;
        }
        
        let mut current_chain = vec![i];
        assigned.set(i, true);
        
        let mut last_source = matches[i].source_sequence;
        let mut last_target = matches[i].target_sequence;
        
        // Check if starting a chain with traced ngram
        let chain_has_traced = if let Some(storage) = source_storage {
            if let Ok(text_map) = storage.get_text_bytes_batch(&[matches[i].source_sequence]) {
                if let Some(bytes) = text_map.get(&matches[i].source_sequence) {
                    if let Ok(text) = std::str::from_utf8(bytes) {
                        !traced_phrase.is_empty() && text.contains(traced_phrase)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if chain_has_traced && log_enabled!(log::Level::Debug) {
            debug!("TRACE [chains]: Starting new chain with traced match at index {}", i);
        }
        
        // Look for matches that continue this chain
        for j in (i + 1)..matches.len() {
            if assigned[j] {
                continue;
            }
            
            let source_gap = matches[j].source_sequence.saturating_sub(last_source);
            let target_gap = matches[j].target_sequence.saturating_sub(last_target);
            
            let max_gap = max_gap as u64;
            // Check if this match extends the chain
            if source_gap <= max_gap && target_gap <= max_gap {
                current_chain.push(j);
                assigned.set(j, true);
                last_source = matches[j].source_sequence;
                last_target = matches[j].target_sequence;
                
                // Check if the extended match is traced
                if chain_has_traced {
                    let match_has_traced = is_traced_sequence(
                        matches[j].source_sequence, 
                        source_storage, 
                        traced_phrase
                    );
                    
                    if match_has_traced {
                        trace!("TRACE [chains]: Extended traced chain with match at index {}", j);
                    }
                }
            } else if source_gap > max_gap * 2 {
                // Optimization: if the gap is too large, we can stop looking
                break;
            }
        }
                
        chains.push(current_chain);
    }
    
    // Log traced chain statistics if applicable
    if !traced_matches.is_empty() && log_enabled!(log::Level::Debug) {
        if let Some(storage) = source_storage {
            let traced_chains = chains.iter().filter(|chain| {
                // Check if any match in the chain contains the traced phrase
                chain.iter().any(|&idx| 
                    is_traced_sequence(matches[idx].source_sequence, Some(storage), traced_phrase)
                )
            }).count();
            
            info!("TRACE [chains]: Built {} chains, {} contain traced n-gram",
                chains.len(), traced_chains);
            
            for (chain_idx, chain) in chains.iter().enumerate() {
                let has_traced = chain.iter().any(|&idx| 
                    is_traced_sequence(matches[idx].source_sequence, Some(storage), traced_phrase)
                );
                
                if has_traced {
                    debug!("TRACE [chains]: Chain {} has {} matches, includes traced n-gram",
                        chain_idx, chain.len());
                }
            }
        }
    }
    
    debug!("Built {} chains from {} matches (max_gap: {})", 
        chains.len(), matches.len(), max_gap);
    
    chains
}

/// Group matches into clusters based on spatial proximity
#[allow(dead_code)]
fn proximate_chains(
    matches: &[SequenceMatch],
    proximity_distance: usize,
    traced_phrase: &str,
    source_storage: Option<&LMDBStorage>,
) -> Vec<Vec<usize>> {
    if matches.is_empty() {
        return Vec::new();
    }
    
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut processed = std::collections::HashSet::new();
    
    // Track clusters with our traced n-gram
    let mut traced_cluster_count = 0;
    
    // Process each match as a potential start of a new cluster
    for i in 0..matches.len() {
        // Skip if this match has already been assigned to a cluster
        if processed.contains(&i) {
            continue;
        }
        
        // Start a new cluster with this match index
        let mut cluster = vec![i];
        processed.insert(i);
        
        // Check if starting with our traced n-gram
        let cluster_starts_with_traced = if !traced_phrase.is_empty() && log_enabled!(log::Level::Info) {
            is_traced_sequence(matches[i].source_sequence, source_storage, traced_phrase)
        } else {
            false
        };

        if cluster_starts_with_traced {
            debug!("TRACE [proximity]: Starting new cluster with traced n-gram at source={}, target={}",
                matches[i].source_sequence, matches[i].target_sequence);  // Changed from source_position, target_position
        }
        
        // Keep expanding the cluster until no more matches can be added
        let mut added_more = true;
        
        while added_more {
            added_more = false;
            
            // Look for unprocessed matches that are close to any match in our current cluster
            for j in 0..matches.len() {
                // Skip if already processed
                if processed.contains(&j) {
                    continue;
                }
                
                let candidate = &matches[j];
                
                // At the beginning of the function or loop, convert proximity_distance once
                let proximity_distance = proximity_distance as u64;
                
                // Check distance to every match in the current cluster
                let mut is_close = false;
                for &cluster_idx in &cluster {
                    let cluster_match = &matches[cluster_idx];
                    
                    // Calculate distance in both source and target dimensions
                    let source_distance = if cluster_match.source_sequence > candidate.source_sequence {
                        cluster_match.source_sequence - candidate.source_sequence
                    } else {
                        candidate.source_sequence - cluster_match.source_sequence
                    };
                    
                    let target_distance = if cluster_match.target_sequence > candidate.target_sequence {
                        cluster_match.target_sequence - candidate.target_sequence
                    } else {
                        candidate.target_sequence - cluster_match.target_sequence
                    };
                    
                    // If both distances are within proximity_distance, this match is close enough
                    if source_distance <= proximity_distance && target_distance <= proximity_distance {
                        is_close = true;
                        break;  // No need to check against other cluster matches
                    }
                }
                
                // If this candidate is close to any match in the cluster, add it
                if is_close {
                    // Check if this match contains traced n-gram - but only if needed
                    let has_traced = if !traced_phrase.is_empty() && log_enabled!(log::Level::Info) {
                        is_traced_sequence(candidate.source_sequence, source_storage, traced_phrase)
                    } else {
                        false
                    };
                    
                    if has_traced && cluster_starts_with_traced {
                        debug!("TRACE [proximity]: Adding traced n-gram to existing traced cluster at position {}",
                            cluster.len());
                    } else if has_traced {
                        debug!("TRACE [proximity]: Adding traced n-gram to cluster at position {}",
                            cluster.len());
                    }
                    
                    cluster.push(j);
                    processed.insert(j);
                    added_more = true; // We added a match, so we need to do another pass
                }
            }
        }
        
        let cluster_has_traced = if !traced_phrase.is_empty() && log_enabled!(log::Level::Info) {
            cluster.iter()
                .any(|&idx| is_traced_sequence(matches[idx].source_sequence, source_storage, traced_phrase))
        } else {
            false
        };

        // Only keep clusters with at least 2 matches
        if cluster.len() >= 2 {
            if cluster_has_traced {
                traced_cluster_count += 1;
                debug!("TRACE [proximity]: Built cluster #{} containing traced n-gram with {} elements",
                    traced_cluster_count, cluster.len());
                
                // Show some specific matches in this cluster
                for (pos, &match_idx) in cluster.iter().enumerate().take(3) {
                    let m = &matches[match_idx];
                    debug!("TRACE [proximity]: Match #{} in traced cluster: source={}, target={}",
                        pos+1, m.source_sequence, m.target_sequence);
                }
            }
            clusters.push(cluster);
        } else if cluster_has_traced {
            // Log if a cluster with traced n-gram is being discarded due to length
            debug!("TRACE [proximity]: DISCARDED cluster containing traced n-gram with only {} elements",
                cluster.len());
        }
    }
    
    debug!("Built {} proximity clusters from {} matches (proximity_distance: {})", 
        clusters.len(), matches.len(), proximity_distance);
    
    clusters
}

/// Create a candidate from a sequential chain of matches
fn sequential_candidate(
    cluster_indices: &[usize],
    matches: &[SequenceMatch]
) -> Result<SequenceCluster> {
    if cluster_indices.is_empty() {
        return Err(Error::text("Cannot create candidate from empty cluster"));
    }
    
    // For sequential chains, first and last define the range
    let first_match = &matches[cluster_indices[0]];
    let last_match = &matches[cluster_indices[cluster_indices.len() - 1]];

    // Create ranges
    let source_range = first_match.source_sequence..(last_match.source_sequence + 1);
    let target_range = first_match.target_sequence..(last_match.target_sequence + 1);

    // Validate ranges
    let source_size = source_range.end - source_range.start;
    let target_size = target_range.end - target_range.start;

    if source_size == 0 || target_size == 0 {
        return Err(Error::text("Zero-length range detected"));
    }

    // Range ratio validation (from legacy)
    let range_ratio = source_size as f64 / target_size as f64;
    if range_ratio < 0.32 || range_ratio > 3.0 {
        return Err(Error::text(format!(
            "Range size mismatch: source={}, target={}, ratio={:.2}",
            source_size, target_size, range_ratio
        )));
    }

    // Create the SequenceCluster
    Ok(SequenceCluster {
        source_start: first_match.source_sequence,
        source_end: last_match.source_sequence,
        target_start: first_match.target_sequence,
        target_end: last_match.target_sequence,
        match_count: cluster_indices.len(),
    })
}

/// Create a candidate from a proximity cluster of matches
fn proximity_candidate(
    cluster_indices: &[usize],
    matches: &[SequenceMatch]      // Changed from &[NGramMatch]
) -> Result<SequenceCluster> {     // Changed from MatchCandidate
    if cluster_indices.is_empty() {
        return Err(Error::text("Cannot create candidate from empty cluster"));
    }
    
    // For proximity clusters, find min/max across all matches
    let mut source_min = u64::MAX;  // Changed from usize::MAX
    let mut source_max = 0u64;      // Changed type
    let mut target_min = u64::MAX;  // Changed from usize::MAX
    let mut target_max = 0u64;      // Changed type
    
    for &idx in cluster_indices {
        let m = &matches[idx];
        source_min = source_min.min(m.source_sequence);  // Changed field name
        source_max = source_max.max(m.source_sequence);  // Changed field name
        target_min = target_min.min(m.target_sequence);  // Changed field name
        target_max = target_max.max(m.target_sequence);  // Changed field name
    }
    
    // Create ranges that encompass entire cluster
    let source_range = source_min..(source_max + 1);
    let target_range = target_min..(target_max + 1);
    
    // Validate ranges
    let source_size = source_range.end - source_range.start;
    let target_size = target_range.end - target_range.start;
    
    if source_size == 0 || target_size == 0 {
        return Err(Error::text("Zero-length range detected"));
    }
    
    // Range ratio validation (from legacy)
    let range_ratio = source_size as f64 / target_size as f64;
    if range_ratio < 0.32 || range_ratio > 3.0 {
        return Err(Error::text(format!(
            "Range size mismatch: source={}, target={}, ratio={:.2}",
            source_size, target_size, range_ratio
        )));
    }
        
    Ok(SequenceCluster {
        source_start: source_min,
        source_end: source_max,      // Just use max directly
        target_start: target_min,
        target_end: target_max,      // Just use max directly
        match_count: cluster_indices.len(),
    })  
}