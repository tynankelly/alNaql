// src/matcher/clusters/validation.rs

use ahash::{AHashMap, AHashSet};
use std::path::Path;
use std::time::Instant;
use std::ops::{Range, RangeInclusive};
use rayon::prelude::*;
use log::{info, debug, warn, trace, error};
use std::fs::File;

use crate::error::{Result, Error};
use crate::utils::binary_io::{BinaryReader, BinaryWriter, DataType};
use crate::matcher::types::{SequenceCluster, ReconstructedSegment, ValidatedSegment, FinalMatchResult};
use crate::matcher::clusters::spatial;
use crate::matcher::clusters::text_classification::TextClassifier;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::utils::progress_manager::ProgressContext;
use crate::config::AlNaqlConfig;
use crate::config::NGramType;
use crate::storage::LMDBStorage;
use crate::matcher::is_traced_sequence;

// ============================================================================
// TRAITS
// ============================================================================

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        // Initially, each element is its own parent (its own group)
        // and has a rank of 0 (tree height of 0)
        UnionFind {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        // Find the root of the tree containing x
        // We use path compression here: as we traverse up to find the root,
        // we make every node point directly to the root, flattening the tree
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) {
        // Unite the groups containing x and y
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return; // Already in the same group
        }
        
        // Union by rank: attach the smaller tree under the root of the larger tree
        // This keeps the trees balanced and operations efficient
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            // If ranks are equal, make one root and increase its rank
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
    }
}
// ============================================================================
// MAIN VALIDATION FUNCTIONS
// ============================================================================

/// Main validation entry point - validates candidates without merging
pub fn validate_without_merging(
    candidates: &[SequenceCluster],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    config: &AlNaqlConfig,
    text_classifier: &TextClassifier,
    ngram_type: NGramType
) -> Result<Vec<ReconstructedSegment>> {
    let start_time = Instant::now();
    debug!("Validating {} candidates without merging", candidates.len());
        
    // Check for traced candidates only if trace logging is enabled
    let traced_candidates: Vec<&SequenceCluster> = if log::log_enabled!(log::Level::Trace) {
        candidates.iter()
            .filter(|c| {
                // Check if any sequence in the source or target range is traced
                // Check source sequences
                for seq in c.source_start..=c.source_end {
                    if is_traced_sequence(seq, Some(source_storage), &config.debug.traced_phrase) {
                        return true;
                    }
                }
                // Check target sequences
                for seq in c.target_start..=c.target_end {
                    if is_traced_sequence(seq, Some(target_storage), &config.debug.traced_phrase) {
                        return true;
                    }
                }
                false
            })
            .collect()
    } else {
        vec![]
    };
    
    // Process candidates in parallel
    let valid_matches: Vec<ReconstructedSegment> = candidates
    .par_iter()
    .filter_map(|candidate| {
        let has_traced = traced_candidates.contains(&candidate);
        // Reconstruct source text directly from sequence range
        let source_text = match reconstruct_text_optimized(
            candidate.source_start..=candidate.source_end,  // Pass sequence range
            source_storage,                                  // Pass storage reference
            config.generator.ngram_type,
            config.generator.stride,
            None,
        ) {
            Ok(text) => text,
            Err(_) => return None,  // Skip if reconstruction fails
        };
        
        // Reconstruct target text directly from sequence range
        let target_text = match reconstruct_text_optimized(
            candidate.target_start..=candidate.target_end,
            target_storage,
            config.generator.ngram_type,
            config.generator.stride,
            None,
        ) {
            Ok(text) => text,
            Err(_) => return None,  // Skip if reconstruction fails
        };
            
            // Calculate similarity
            let similarity = match similarity_calc.compare_texts(
                &source_text, 
                &target_text, 
                Some(text_classifier), 
                config, 
                ngram_type,
                Some((candidate.source_start, candidate.source_end)),
                Some(source_storage),
            ) {
                Ok(sim) => sim,
                Err(e) => {
                    if has_traced {
                        debug!("TRACE [validation]: Error calculating similarity for traced candidate: {}", e);
                    }
                    return None;
                }
            };
            
            if has_traced {
                debug!("TRACE [validation]: Similarity for traced candidate: {:.3} (threshold: {:.3})", 
                    similarity, config.matcher.initial_text_min_confidence);
            }
            
            // Check if similarity meets threshold
            if similarity < config.matcher.initial_text_min_confidence {
                if has_traced {
                    debug!("TRACE [validation]: Traced candidate rejected - similarity too low");
                }
                return None;
            }
                        
            // Create the validated match result
            Some(ReconstructedSegment {
                source_text,  // The source text you reconstructed
                target_text,  // The target text you reconstructed
                source_start_seq: candidate.source_start,
                source_end_seq: candidate.source_end,
                target_start_seq: candidate.target_start,
                target_end_seq: candidate.target_end,
            })
        })
        .collect();
    
    info!("Validated {} matches in {:?} (without merging)", 
        valid_matches.len(), start_time.elapsed());
    
    Ok(valid_matches)
}
use std::collections::HashSet;

/// Prefetch all text sequences needed for a batch of clusters
/// Returns maps of sequence -> text bytes for both source and target
fn prefetch_text_for_batch(
    clusters: &[SequenceCluster],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Result<(AHashMap<u64, Vec<u8>>, AHashMap<u64, Vec<u8>>)> {
    // Collect all unique sequences we'll need
    let mut source_seqs = HashSet::new();
    let mut target_seqs = HashSet::new();
    
    for cluster in clusters {
        // Add all sequences in the cluster's range
        for seq in cluster.source_start..=cluster.source_end {
            source_seqs.insert(seq);
        }
        for seq in cluster.target_start..=cluster.target_end {
            target_seqs.insert(seq);
        }
    }
    
    let source_seq_count = source_seqs.len();
    let target_seq_count = target_seqs.len();
    
    trace!("Prefetching {} source and {} target sequences for batch of {} clusters",
        source_seq_count, target_seq_count, clusters.len());
    
    // Convert to Vec for batch fetch
    let source_vec: Vec<u64> = source_seqs.into_iter().collect();
    let target_vec: Vec<u64> = target_seqs.into_iter().collect();
    
    // Single batch fetch per storage (one transaction each)
    let source_map = source_storage.get_text_bytes_batch(&source_vec)?;
    let target_map = target_storage.get_text_bytes_batch(&target_vec)?;
    
    trace!("Prefetched {}/{} source and {}/{} target sequences",
        source_map.len(), source_seq_count,
        target_map.len(), target_seq_count);
    
    Ok((source_map, target_map))
}
pub fn stream_validation(
    input_path: &Path,
    output_path: &Path,
    config: &AlNaqlConfig,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    text_classifier: &TextClassifier,
    progress_context: Option<&ProgressContext>,  // Keep this, we'll use it
) -> Result<usize> {
    use rayon::prelude::*;
    use std::time::Instant;
    
    let start_time = Instant::now();
    let batch_size = config.execution.validation_batch_size;
    
    info!("Starting parallel validation with batch size {} and compressed binary I/O", batch_size);
    
    // Open input file and create binary reader
    let input_file = File::open(input_path)?;
    let mut reader = BinaryReader::new(input_file)?;
    
    // Verify data type
    if reader.data_type() != DataType::SequenceCluster {
        return Err(Error::storage(
            format!("Expected SequenceCluster data in {}, found {:?}", 
                input_path.display(), reader.data_type())
        ));
    }
    
    // Create output file and binary writer
    let output_file = File::create(output_path)?;
    let mut writer = BinaryWriter::new(
        output_file,
        true,  // Always compress
        DataType::ReconstructedSegment
    )?;
    
    let mut total_read = 0;
    let mut total_written = 0;
    let mut batch_count = 0;
    
    // Batch buffer
    let mut batch = Vec::with_capacity(batch_size);
    
    // ========================================================================
    // Main loop: Read clusters into batches, process in parallel, write results
    // ========================================================================
    
    while let Some(cluster) = reader.read_item::<SequenceCluster>()? {
        batch.push(cluster);
        total_read += 1;
        
        // When batch is full, process it in parallel
        if batch.len() >= batch_size {
            batch_count += 1;
            
            // ================================================================
            // PREFETCH: Get all text sequences for this batch upfront
            // ================================================================
            let (source_cache, target_cache) = prefetch_text_for_batch(
                &batch,
                source_storage,
                target_storage,
            )?;
            
            trace!("Batch {}: Prefetched {} source and {} target sequences",
                batch_count, source_cache.len(), target_cache.len());
            
            // ================================================================
            // PARALLEL PROCESSING with prefetched cache
            // ================================================================
            let validated: Vec<ReconstructedSegment> = batch
                .par_iter()
                .filter_map(|cluster| {
                    match validate_candidate_streaming(
                        cluster.clone(),
                        config,
                        source_storage,
                        target_storage,
                        similarity_calc,
                        text_classifier,
                        Some((&source_cache, &target_cache)),  // <-- Pass cache!
                    ) {
                        Ok(Some(segment)) => Some(segment),
                        Ok(None) => None,  // Cluster didn't pass validation
                        Err(e) => {
                            // Preserve error logging
                            warn!("Validation error for cluster: {}", e);
                            None
                        }
                    }
                })
                .collect();
            
            // Write validated segments
            for segment in validated {
                writer.write_item(&segment)?;
                total_written += 1;
            }
            
            // Periodic flush (same as original, but per batch instead of every 1000)
            writer.flush()?;
            
            // Clear batch for next iteration
            batch.clear();
            
            // Progress updates (enhanced from original)
            if let Some(ctx) = progress_context {
                ctx.update_validation_progress(batch_count, 0, total_written);
            }
            
            // Periodic logging (same logic as original, every 5000 clusters)
            if total_read % 5000 == 0 {
                let acceptance_rate = (total_written as f64 / total_read as f64) * 100.0;
                let elapsed = start_time.elapsed().as_secs_f64();
                let throughput = total_read as f64 / elapsed;
                
                trace!("Validation: {} candidates processed, {} accepted ({:.1}% rate, {:.0} clusters/sec)",
                    total_read, total_written, acceptance_rate, throughput);
            }
        }
    }
    
    // ========================================================================
    // Process final partial batch (same as main loop, just for remainder)
    // ========================================================================
    
    if !batch.is_empty() {
        // Prefetch for final batch too
        let (source_cache, target_cache) = prefetch_text_for_batch(
            &batch,
            source_storage,
            target_storage,
        )?;
        
        trace!("Final batch: Prefetched {} source and {} target sequences",
            source_cache.len(), target_cache.len());
        
        let validated: Vec<ReconstructedSegment> = batch
            .par_iter()
            .filter_map(|cluster| {
                validate_candidate_streaming(
                    cluster.clone(),
                    config,
                    source_storage,
                    target_storage,
                    similarity_calc,
                    text_classifier,
                    Some((&source_cache, &target_cache)),  // <-- Pass cache!
                ).ok().flatten()
            })
            .collect();
        
        for segment in validated {
            writer.write_item(&segment)?;
            total_written += 1;
        }
    }
    
    writer.flush()?;
    
    // ========================================================================
    // Final statistics (same as original, with added throughput)
    // ========================================================================
    
    let acceptance_rate = if total_read > 0 {
        (total_written as f64 / total_read as f64) * 100.0
    } else {
        0.0
    };
    
    let duration = start_time.elapsed();
    let throughput = total_read as f64 / duration.as_secs_f64();
    
    trace!("Validation complete: {} candidates processed, {} accepted ({:.1}% acceptance rate)",
        total_read, total_written, acceptance_rate);
    info!("Throughput: {:.0} clusters/sec in {:.2}s", throughput, duration.as_secs_f64());
    
    // Log compression statistics (same as original)
    if let Ok(metadata) = std::fs::metadata(output_path) {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        info!("Validation output: {} validated matches in {:.2} MB (compressed binary)",
            total_written, size_mb);
    }
    
    Ok(total_written)
}

/// Validate a single cluster for streaming - fetches data and reconstructs text
fn validate_candidate_streaming(
    cluster: SequenceCluster,
    config: &AlNaqlConfig,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    text_classifier: &TextClassifier,
    text_cache: Option<(&AHashMap<u64, Vec<u8>>, &AHashMap<u64, Vec<u8>>)>,
) -> Result<Option<ReconstructedSegment>> {
    // Check if tracing is enabled
    let tracing_enabled = !config.debug.traced_phrase.is_empty() 
        && log::log_enabled!(log::Level::Debug);
    
    // Check if this cluster contains the traced phrase
    let has_traced = tracing_enabled && {
        (cluster.source_start..=cluster.source_end).any(|seq| 
            is_traced_sequence(seq, Some(source_storage), &config.debug.traced_phrase)
        )
    };
    
    if has_traced {
        debug!("TRACE [validate_streaming]: Processing traced cluster - Source[{}..{}] Target[{}..{}] with {} matches",
            cluster.source_start, cluster.source_end,
            cluster.target_start, cluster.target_end,
            cluster.match_count);
    }
    
    // Basic checks
    if cluster.match_count == 0 {
        if has_traced {
            warn!("TRACE [validate_streaming]: Traced cluster rejected - zero matches!");
        }
        return Ok(None);
    }
    
    // Get sequence ranges directly from the cluster
    let source_seq_start = cluster.source_start;
    let source_seq_end = cluster.source_end;
    let target_seq_start = cluster.target_start;
    let target_seq_end = cluster.target_end;
    
    if has_traced {
        debug!("TRACE [validate_streaming]: Reconstructing text for sequences Source[{}..{}] Target[{}..{}]",
            source_seq_start, source_seq_end-1,
            target_seq_start, target_seq_end-1);
    }
    
    // Extract cache references if available
    let source_cache = text_cache.map(|(s, _)| s);
    let target_cache = text_cache.map(|(_, t)| t);
    
    // Use the optimized text reconstruction with cache
    let source_text = reconstruct_text_optimized(
        source_seq_start..=source_seq_end,
        source_storage,
        config.generator.ngram_type,
        config.generator.stride,
        source_cache,  // <-- Pass source cache
    )?;
    
    let target_text = reconstruct_text_optimized(
        target_seq_start..=target_seq_end,
        target_storage,
        config.generator.ngram_type,
        config.generator.stride,
        target_cache,  // <-- Pass target cache
    )?;
    
    if has_traced {
        info!("TRACE [validate_streaming]: Reconstructed text:");
        info!("  Source: '{}'", source_text);
        info!("  Target: '{}'", target_text);
        
        // Check if the reconstructed text actually contains the traced phrase
        if !source_text.contains(&config.debug.traced_phrase) {
            error!("TRACE [validate_streaming]: WARNING - Reconstructed source text does NOT contain traced phrase!");
            error!("  Looking for: '{}'", config.debug.traced_phrase);
            error!("  Got: '{}'", source_text);
        }
    }
    
    // Calculate source and target similarity
    let similarity = similarity_calc.compare_texts(
        &source_text,
        &target_text,
        Some(text_classifier),
        config,
        config.generator.ngram_type,
        Some((cluster.source_start, cluster.source_end)),
        Some(source_storage),
    )?;
    
    if has_traced {
        info!("TRACE [validate_streaming]: Similarity score: {:.3} (threshold: {:.3})",
            similarity, config.matcher.ngram_min_confidence);
    }
    
    // Check similarity threshold
    if similarity < config.matcher.ngram_min_confidence {
        if has_traced {
            warn!("TRACE [validate_streaming]: Traced cluster REJECTED - similarity {:.3} below threshold {:.3}",
                similarity, config.matcher.ngram_min_confidence);
        }
        return Ok(None);
    }
    
    if has_traced {
        info!("TRACE [validate_streaming]: Traced cluster PASSED validation with similarity {:.3}",
            similarity);
    }
    
    // Create ReconstructedSegment
    let reconstructed_segment = ReconstructedSegment {
        source_text,
        target_text,
        source_start_seq: source_seq_start,
        source_end_seq: source_seq_end,
        target_start_seq: target_seq_start,
        target_end_seq: target_seq_end,
    };
    
    Ok(Some(reconstructed_segment))
}

// ============================================================================
// MERGE OVERLAPPING MATCHES
// ============================================================================

pub fn merge_overlapping_matches(
    matches: &mut Vec<ReconstructedSegment>,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    config: &AlNaqlConfig,
) -> Vec<ReconstructedSegment> {
    // Early return for empty or single match cases
    if matches.len() <= 1 {
        return matches.clone();
    }

    // Initialize our Union-Find structure to track which matches belong together
    // Each match starts as its own group
    let mut union_find = UnionFind::new(matches.len());
    
    // =================================================================
    // Phase 1: Group matches by overlapping source ranges (inline)
    // =================================================================
    // We'll create groups of matches that share overlapping or adjacent source ranges.
    // Since matches are pre-sorted by source_range.start, we can do this in a single pass.
    
    let mut source_groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = vec![0]; // Start with first match
    let mut group_source_seq_end = matches[0].source_end_seq;  // Changed from source_seq_range.end

    for i in 1..matches.len() {
        let current_match = &matches[i];
        
        // Check if this match's source range overlaps or is adjacent to the current group
        if current_match.source_start_seq <= group_source_seq_end + 1 {
            // This match belongs to the current source group
            current_group.push(i);
            // Extend the group's boundary if this match extends further
            if current_match.source_end_seq > group_source_seq_end {
                group_source_seq_end = current_match.source_end_seq;
            }
        } else {
            // This match starts a new source group
            source_groups.push(current_group);
            current_group = vec![i];
            group_source_seq_end = current_match.source_end_seq;
        }
    }

    // Don't forget to add the last group
    if !current_group.is_empty() {
        source_groups.push(current_group);
    }
    
    // =================================================================
    // Phase 2: Find target overlaps within each source group
    // =================================================================
    // For each source group, we need to find which matches also have 
    // overlapping or adjacent target ranges
    
    for source_group in &source_groups {
        if source_group.len() < 2 {
            // Single match groups can't have internal overlaps
            continue;
        }
        
        // Call our helper function to find all pairs of overlapping matches
        // within this source group
        let overlapping_pairs = find_target_overlaps_in_group(&source_group, &matches);
        
        // Each overlapping pair represents matches that should be merged
        for (idx1, idx2) in overlapping_pairs {
            union_find.union(idx1, idx2);
        }
    }
    
    // =================================================================
    // Phase 3: Build connected components (inline)
    // =================================================================
    // The Union-Find structure now contains our transitive merge groups.
    // We need to extract these connected components.
    
    let mut components: std::collections::HashMap<usize, Vec<usize>> = 
        std::collections::HashMap::new();
    
    for i in 0..matches.len() {
        let root = union_find.find(i);
        components.entry(root).or_insert_with(Vec::new).push(i);
    }
    
    // =================================================================
    // Phase 4: Consolidate merged groups (inline)
    // =================================================================
    // For each connected component, merge all matches into a single match
    let mut merged_matches = Vec::new();

    for (_root, component_indices) in components {
        if component_indices.len() == 1 {
            // Single match doesn't need merging, just clone it
            merged_matches.push(matches[component_indices[0]].clone());
        } else {
            // Multiple matches need to be merged into one
            // Sort component indices by source start sequence for ordered processing
            let mut sorted_indices = component_indices.clone();
            sorted_indices.sort_by_key(|&idx| matches[idx].source_start_seq);
            // DEBUG: Show what we're about to merge with FULL text
            trace!("=== MERGING {} OVERLAPPING MATCHES ===", sorted_indices.len());
            for &idx in &sorted_indices {
                let segment = &matches[idx];
                trace!("  Match {}: Source[{}..{}] Target[{}..{}]", 
                    idx, 
                    segment.source_start_seq, segment.source_end_seq,
                    segment.target_start_seq, segment.target_end_seq
                );
                trace!("    Source text: '{}'", segment.source_text);
                trace!("    Target text: '{}'", segment.target_text);
            }
            
            // Start with the first match as our base
            let base_idx = sorted_indices[0];
            let base_segment = &matches[base_idx];
            
            // Initialize with the base segment's text and ranges
            let mut merged_source_text = base_segment.source_text.clone();
            let mut merged_target_text = base_segment.target_text.clone();
            
            // Track the overall span of sequences
            let mut min_source_seq = base_segment.source_start_seq;
            let mut max_source_seq = base_segment.source_end_seq;
            let mut min_target_seq = base_segment.target_start_seq;
            let mut max_target_seq = base_segment.target_end_seq;
            
            // Track what we've already incorporated into our merged text
            let mut current_source_end = base_segment.source_end_seq;
            let mut current_target_end = base_segment.target_end_seq;
            
            // Process remaining matches in order to extend the merged text
            for &idx in &sorted_indices[1..] {
                let segment = &matches[idx];
                
                // Update the overall bounds
                min_source_seq = min_source_seq.min(segment.source_start_seq);
                min_target_seq = min_target_seq.min(segment.target_start_seq);
                
                // Handle source text extension
                if segment.source_end_seq > current_source_end {
                    // This segment extends beyond what we've already merged
                    // Fetch sequences from current_source_end + 1 to segment.source_end_seq
                    let start_seq = current_source_end + 1;
                    
                    match reconstruct_text_optimized(
                        start_seq..=segment.source_end_seq,
                        source_storage,
                        config.generator.ngram_type,
                        config.generator.stride,
                        None
                    ) {
                        Ok(additional_text) => {
                            if !additional_text.is_empty() {
                                // Add a space separator between segments
                                merged_source_text.push(' ');
                                merged_source_text.push_str(&additional_text);
                            }
                            // Update our tracking of what's been merged
                            current_source_end = segment.source_end_seq;
                            max_source_seq = segment.source_end_seq;
                        }
                        Err(e) => {
                            // If reconstruction fails, log and continue
                            // We'll keep what we have so far
                            warn!("Failed to reconstruct source extension {}-{}: {}", 
                                start_seq, segment.source_end_seq, e);
                        }
                    }
                }
                
                // Handle target text extension (same logic)
                if segment.target_end_seq > current_target_end {
                    let start_seq = current_target_end + 1;
                    
                    match reconstruct_text_optimized(
                        start_seq..=segment.target_end_seq,
                        target_storage,
                        config.generator.ngram_type,
                        config.generator.stride,
                        None
                    ) {
                        Ok(additional_text) => {
                            if !additional_text.is_empty() {
                                merged_target_text.push(' ');
                                merged_target_text.push_str(&additional_text);
                            }
                            current_target_end = segment.target_end_seq;
                            max_target_seq = segment.target_end_seq;
                        }
                        Err(e) => {
                            warn!("Failed to reconstruct target extension {}-{}: {}", 
                                start_seq, segment.target_end_seq, e);
                        }
                    }
                }
            }
            
            // DEBUG: Show the FULL result of merging
            trace!("=== MERGED RESULT ===");
            trace!("  Final: Source[{}..{}] Target[{}..{}]",
                min_source_seq, max_source_seq,
                min_target_seq, max_target_seq
            );
            trace!("    Merged source text: '{}'", merged_source_text);
            trace!("    Merged target text: '{}'", merged_target_text);
            trace!("  Text grew from {} to {} source chars, {} to {} target chars",
                base_segment.source_text.len(), merged_source_text.len(),
                base_segment.target_text.len(), merged_target_text.len()
            );

            // Create the merged segment with the accumulated text and ranges
            merged_matches.push(ReconstructedSegment {
                source_text: merged_source_text,
                target_text: merged_target_text,
                source_start_seq: min_source_seq,
                source_end_seq: max_source_seq,
                target_start_seq: min_target_seq,
                target_end_seq: max_target_seq,
            });
        }
    }

    merged_matches
    }

fn find_target_overlaps_in_group(
    source_group: &[usize], 
    matches: &[ReconstructedSegment]
) -> Vec<(usize, usize)> {
    let mut overlapping_pairs = Vec::new();
    const SORT_THRESHOLD: usize = 50;

    if source_group.len() < SORT_THRESHOLD {
        // Simple nested loop for small groups (unchanged)
        for i in 0..source_group.len() {
            for j in (i + 1)..source_group.len() {
                let idx1 = source_group[i];
                let idx2 = source_group[j];
                let match1 = &matches[idx1];
                let match2 = &matches[idx2];
                
                let target_seq_overlap = 
                    (match1.target_start_seq <= match2.target_end_seq &&
                    match2.target_start_seq <= match1.target_end_seq) ||
                    (match1.target_end_seq + 1 == match2.target_start_seq) ||
                    (match2.target_end_seq + 1 == match1.target_start_seq);
                
                if target_seq_overlap {
                    overlapping_pairs.push((idx1, idx2));
                }
            }
        }
    } else {
        // Sort by target range and use sweep-line algorithm
        let mut sorted_by_target: Vec<(usize, usize, usize)> = source_group
            .iter()
            .map(|&idx| {
                let m = &matches[idx];
                (m.target_start_seq as usize, m.target_end_seq as usize, idx)
            })
            .collect();
        
        sorted_by_target.sort_by_key(|&(start, _, _)| start);
        
        // Sweep-line logic
        for i in 0..sorted_by_target.len() {
            let (_current_start, current_end, current_idx) = sorted_by_target[i];
            // We prefix with underscore since we don't need current_start
            // after sorting (we know everything ahead has start >= current_start)
            
            for j in (i + 1)..sorted_by_target.len() {
                let (next_start, _next_end, next_idx) = sorted_by_target[j];
                
                // Early termination: if next starts too far after current ends
                if next_start > current_end + 1 {
                    break;  // No more possible overlaps with current
                }
                                
                let ranges_overlap = next_start <= current_end;
                let ranges_adjacent = next_start == current_end + 1;
                
                if ranges_overlap || ranges_adjacent {
                    overlapping_pairs.push((current_idx, next_idx));
                }

            }
        }
    }
    
    overlapping_pairs
}

// ============================================================================
// MERGE NEARBY MATCHES
// ============================================================================

/// Merges adjacent matches using R-tree spatial indexing
pub fn merge_nearby_matches(
    matches: &mut Vec<ReconstructedSegment>,
    config: &AlNaqlConfig,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    text_classifier: &TextClassifier,
) -> Result<()> {
    let adjacent_gap_ratio = config.matcher.adjacent_merge_ratio;
    
    info!("Starting adjacent match merging with merge gap ratio: {}", adjacent_gap_ratio);
    
    if matches.len() <= 1 {
        debug!("Skipping merge - only {} matches", matches.len());
        return Ok(());
    }
    
    let mut iteration = 0;
    let max_iterations = 10; // Prevent infinite loops
    
    // Keep merging until no more merges are possible
    // Build R-tree once before loop starts
    let mut current_rtree = spatial::build_rtree(matches);
    let mut previous_iteration_made_merges = false;

    loop {
        iteration += 1;
        if iteration > max_iterations {
            warn!("Reached maximum iterations ({}) for adjacent merging", max_iterations);
            break;
        }
        
        // Only rebuild R-tree if previous iteration made merges
        if previous_iteration_made_merges {
            debug!("Rebuilding R-tree after merges in iteration {}", iteration - 1);
            current_rtree = spatial::build_rtree(matches);
        }
        
        // Track which matches have been processed or merged
        let mut processed = AHashSet::new();
        let mut merged_results = Vec::new();
        let mut made_merges = false;
                
        // Process matches in order
        for (i, current) in matches.iter().enumerate() {
            // Skip if already processed
            if processed.contains(&i) {
                continue;
            }
            
            processed.insert(i);
            
            // Find adjacent candidates using R-tree
            let candidates = spatial::find_adjacent_candidates(&current_rtree, current, adjacent_gap_ratio);            
            // Try to find a match to merge with
            let mut found_merge = false;
            for &j in &candidates {
                if i == j || processed.contains(&j) {
                    continue;
                }
                
                let other = &matches[j];
                
                // Check if they should be merged
                // In merge_adjacent_matches_with_rtree
                if should_merge_nearby(current, other, adjacent_gap_ratio)? {
                    // ADD THIS DEBUG WITH FULL TEXT
                    trace!("=== ATTEMPTING TO MERGE MATCHES {} AND {} ===", i, j);
                    trace!("Match {} details:", i);
                    trace!("  Sequences: Source[{}..{}] Target[{}..{}]", 
                        current.source_start_seq, current.source_end_seq,
                        current.target_start_seq, current.target_end_seq
                    );
                    trace!("  Source text: '{}'", current.source_text);
                    trace!("  Target text: '{}'", current.target_text);
                    
                    trace!("Match {} details:", j);
                    trace!("  Sequences: Source[{}..{}] Target[{}..{}]",
                        other.source_start_seq, other.source_end_seq,
                        other.target_start_seq, other.target_end_seq
                    );
                    trace!("  Source text: '{}'", other.source_text);
                    trace!("  Target text: '{}'", other.target_text);
                    
                    trace!("Gap analysis:");
                    trace!("  Source gap: {} sequences", other.source_start_seq - current.source_end_seq);
                    trace!("  Target gap: {} sequences", other.target_start_seq - current.target_end_seq);
                    
                    // Try to merge - but it might return None if quality is too low
                    match merge_matches_with_gap_text(
                        current,
                        other,
                        source_storage,
                        target_storage,
                        similarity_calc,
                        config,
                        text_classifier,
                    )? {
                         Some(merged) => {
                            // ADD THIS DEBUG WITH COMPLETE MERGED TEXT
                            trace!("=== MERGE SUCCESSFUL ===");
                            trace!("Created merged match from {} and {}:", i, j);
                            trace!("  New sequences: Source[{}..{}] Target[{}..{}]",
                                merged.source_start_seq, merged.source_end_seq,
                                merged.target_start_seq, merged.target_end_seq
                            );
                            trace!("  MERGED SOURCE TEXT: '{}'", merged.source_text);
                            trace!("  MERGED TARGET TEXT: '{}'", merged.target_text);
                            trace!("  Text growth: {} -> {} source chars, {} -> {} target chars",
                                current.source_text.len() + other.source_text.len(),
                                merged.source_text.len(),
                                current.target_text.len() + other.target_text.len(),
                                merged.target_text.len()
                            );
                            
                            merged_results.push(merged);
                            processed.insert(j);
                            made_merges = true;
                            found_merge = true;
                            break;
                        }
                        None => {
                            trace!("=== MERGE REJECTED ===");
                            trace!("Quality check failed for merging {} and {}", i, j);
                            trace!("These matches will remain separate");
                            continue;
                        }
                    }
                }
            }
            
            // If no merge was found, keep the original match
            if !found_merge {
                merged_results.push(current.clone());
            }
        }
        
        // Update matches with merged results
        if made_merges {
            // Store the previous count before updating
            let previous_count = matches.len();
            *matches = merged_results;
            previous_iteration_made_merges = true;
            
            // Now we can use the previous count in our debug output
            debug!("=== END OF ITERATION {} - MATCHES WERE MERGED ===", iteration);
            debug!("Reduced from {} to {} matches", previous_count, matches.len());
            
            // Show full details for first few matches
            for (i, m) in matches.iter().enumerate().take(3) {
                debug!("Match {} after iteration {}:", i, iteration);
                debug!("  Sequences: Source[{}..{}] Target[{}..{}]",
                    m.source_start_seq, m.source_end_seq,
                    m.target_start_seq, m.target_end_seq
                );
                debug!("  Source text: '{}'", m.source_text);
                debug!("  Target text: '{}'", m.target_text);
            }
            
            if matches.len() > 3 {
                debug!("... and {} more matches (not shown for brevity)", matches.len() - 3);
            }
        } else {
            debug!("=== END OF ITERATION {} - NO MERGES ===", iteration);
            debug!("No adjacent matches found to merge, stopping");
            break;
        }
    }
    
    info!("Adjacent match merging complete after {} iterations, final count: {}", 
        iteration, matches.len());
    
    Ok(())
}

/// Determines if two non-overlapping matches are close enough to merge
pub fn should_merge_nearby(
    match1: &ReconstructedSegment,  // Changed from MatchResult
    match2: &ReconstructedSegment,  // Changed from MatchResult  
    adjacent_gap_ratio: f64,
) -> Result<bool> {

    
    // First check - source sequences must be adjacent and not overlapping
    // match2 should come after match1
    if match2.source_start_seq <= match1.source_end_seq {
        return Ok(false);  // Sequences overlap or are in wrong order
    }
    
    // Same check for target sequences
    if match2.target_start_seq <= match1.target_end_seq {
        return Ok(false);  // Sequences overlap or are in wrong order
    }
    
    // Calculate source sizes (in number of sequences)
    // This tells us how substantial each match is
    let source_size1 = match1.source_end_seq - match1.source_start_seq;
    let source_size2 = match2.source_end_seq - match2.source_start_seq;
    let combined_source_size = source_size1 + source_size2;
    
    // Calculate target sizes
    let target_size1 = match1.target_end_seq - match1.target_start_seq;
    let target_size2 = match2.target_end_seq - match2.target_start_seq;
    let combined_target_size = target_size1 + target_size2;
    
    // Calculate maximum allowed gaps based on combined sizes
    // The ratio determines how tolerant we are of gaps
    let max_source_gap = (combined_source_size as f64 * adjacent_gap_ratio).ceil() as u64;
    let max_target_gap = (combined_target_size as f64 * adjacent_gap_ratio).ceil() as u64;
    
    // Calculate actual gaps between the matches
    // This is the number of sequences between the end of match1 and start of match2
    let source_gap = match2.source_start_seq - match1.source_end_seq;
    let target_gap = match2.target_start_seq - match1.target_end_seq;
    
    // Check if gaps are within allowed limits
    let source_within_limit = source_gap <= max_source_gap;
    let target_within_limit = target_gap <= max_target_gap;
    
    if !source_within_limit || !target_within_limit {
        return Ok(false);
    }
    
    // Both gaps are within limits, so these matches can be merged
    trace!("Matches can be merged - source gap: {}/{}, target gap: {}/{}", 
        source_gap, max_source_gap, target_gap, max_target_gap);
    
    Ok(true)
}

/// Advanced merging that includes gap text
pub fn merge_matches_with_gap_text(
    match1: &ReconstructedSegment,
    match2: &ReconstructedSegment,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    config: &AlNaqlConfig,
    text_classifier: &TextClassifier,
) -> Result<Option<ReconstructedSegment>> {
    // Ensure match2 comes after match1 (they should be adjacent)
    if match2.source_start_seq <= match1.source_end_seq ||
       match2.target_start_seq <= match1.target_end_seq {
        return Err(Error::text(
            "Matches must be adjacent (non-overlapping) for gap text merging"
        ));
    }
    
    // Reconstruct the ENTIRE span as one continuous text
    // This ensures proper reshingling throughout without any boundary duplication
    
    // Reconstruct the complete source text from start of match1 to end of match2
    let complete_source_text = reconstruct_text_optimized(
        match1.source_start_seq..=match2.source_end_seq,
        source_storage,
        config.generator.ngram_type,
        config.generator.stride,
        None
    )?;
    
    // Reconstruct the complete target text from start of match1 to end of match2
    let complete_target_text = reconstruct_text_optimized(
        match1.target_start_seq..=match2.target_end_seq,
        target_storage,
        config.generator.ngram_type,
        config.generator.stride,
        None
    )?;
    
    // Now check if this merged text maintains sufficient similarity
    let similarity = similarity_calc.compare_texts(
        &complete_source_text,
        &complete_target_text,
        Some(text_classifier),
        config,
        config.generator.ngram_type,
        Some((match1.source_start_seq, match2.source_end_seq)),
        Some(source_storage),
    )?;

    // Check against the threshold for merged text
    let threshold = config.matcher.merged_text_min_confidence;

    if similarity < threshold {
        trace!(
            "Merge rejected: combined similarity {:.3} is below threshold {:.3}. \
            Keeping matches separate to maintain quality.",
            similarity, threshold
        );
        return Ok(None);
    }

    trace!(
        "Merge approved: combined similarity {:.3} meets or exceeds threshold {:.3}",
        similarity, threshold
    );
    
    // Create the merged segment with the properly reconstructed text
    let merged_segment = ReconstructedSegment {
        source_text: complete_source_text,
        target_text: complete_target_text,
        source_start_seq: match1.source_start_seq,
        source_end_seq: match2.source_end_seq,
        target_start_seq: match1.target_start_seq,
        target_end_seq: match2.target_end_seq,
    };

    Ok(Some(merged_segment))
}

// ============================================================================
// CLEANUP FUNCTIONS
// ============================================================================

pub fn remove_contained_matches(matches: &mut Vec<ReconstructedSegment>) {
    if matches.len() <= 1 {
        return; // Nothing to do with 0 or 1 matches
    }
    
    // Step 1: Sort matches by source start position, then by source range size (descending)
    matches.sort_by(|a, b| {
        a.source_start_seq.cmp(&b.source_start_seq)
            .then_with(|| (b.source_end_seq - b.source_start_seq)
                .cmp(&(a.source_end_seq - a.source_start_seq)))
    });
    
    // Step 2: Use a more efficient algorithm to identify contained matches
    let mut to_keep = vec![true; matches.len()];
    let mut keep_count = matches.len();
    
    for i in 0..matches.len() {
        if !to_keep[i] {
            continue;
        }
        
        let container = &matches[i];
        
        for j in (i+1)..matches.len() {
            if !to_keep[j] {
                continue;
            }
            
            let potential_containee = &matches[j];
            
            // Early exit optimization - if this match starts beyond the container's end
            if potential_containee.source_start_seq >= container.source_end_seq {  // Changed field names
                break;
            }
            
            // Check containment using the new field names
            if potential_containee.source_start_seq >= container.source_start_seq &&
               potential_containee.source_end_seq <= container.source_end_seq &&
               potential_containee.target_start_seq >= container.target_start_seq &&
               potential_containee.target_end_seq <= container.target_end_seq {
                to_keep[j] = false;
                keep_count -= 1;
            }
        }
    }
    
    // Step 3: Only rebuild the vector if we found matches to remove
    if keep_count < matches.len() {
        let retained = matches.len() - keep_count;
        debug!("Removing {} contained matches", retained);
        
        let filtered = matches
            .iter()
            .zip(to_keep.iter())
            .filter_map(|(m, &keep)| if keep { Some(m.clone()) } else { None })
            .collect::<Vec<ReconstructedSegment>>();  // Changed from Vec<MatchResult>
        
        *matches = filtered;
    }
}
// ============================================================================
// TEXT PROCESSING FUNCTIONS
// ============================================================================

/// Efficiently reconstructs readable text from ngram sequences with proper reshingling
/// Fetches sequence data directly from column storage and handles UTF-8 conversion
pub fn reconstruct_text_optimized(
    sequence_range: RangeInclusive<u64>,
    storage: &LMDBStorage,
    ngram_type: NGramType,
    stride: usize,
    cache: Option<&AHashMap<u64, Vec<u8>>>,
) -> Result<String> {

    // Fetch text bytes for all sequences in the range
    let sequences: Vec<u64> = sequence_range.clone().collect();
    if sequences.is_empty() {
        trace!("No sequences in range to reconstruct text from");
        return Ok(String::new());
    }

    // Load text bytes - use cache if available, otherwise fetch from storage
    let text_bytes_map = if let Some(cache) = cache {
        // Extract sequences we need from the prefetched cache
        let mut map = AHashMap::with_capacity(sequences.len());
        for &seq in &sequences {
            if let Some(bytes) = cache.get(&seq) {
                map.insert(seq, bytes.clone());
            }
        }
        map
    } else {
        // No cache provided - fetch from storage (original behavior)
        storage.get_text_bytes_batch(&sequences)?
    };
    
    if text_bytes_map.is_empty() {
        trace!("No text data found for sequences {:?}", sequences);
        return Ok(String::new());
    }
   
    // =========================================================================
    // Process sequences in order using the range
    // Since HashMap doesn't preserve order, we iterate through the range
    // and lookup each sequence to ensure correct ordering for reshingling
    // =========================================================================
    let sequence_count = sequence_range.end() - sequence_range.start() + 1;
    trace!("Reconstructing text from {} sequences", sequence_count);

    // Collect sequences that actually have data (for gap detection)
    let mut found_sequences = Vec::new();
    for seq in sequence_range.clone() {
        if text_bytes_map.contains_key(&seq) {
            found_sequences.push(seq);
        }
    }

    trace!("Found {}/{} sequences with text data", found_sequences.len(), sequence_count);
    if !found_sequences.is_empty() {
        trace!("Sequence range: {}..={}", found_sequences.first().unwrap(), found_sequences.last().unwrap());
    }
       
    // =========================================================================
    // Process sequences in order using the range
    // Since HashMap doesn't preserve order, we iterate through the range
    // and lookup each sequence to ensure correct ordering for reshingling
    // =========================================================================
    let sequence_count = sequence_range.end() - sequence_range.start() + 1;
    trace!("Reconstructing text from {} sequences", sequence_count);

    // Collect sequences that actually have data (for gap detection)
    let mut found_sequences = Vec::new();
    for seq in sequence_range.clone() {
        if text_bytes_map.contains_key(&seq) {
            found_sequences.push(seq);
        }
    }

    trace!("Found {}/{} sequences with text data", found_sequences.len(), sequence_count);
    if !found_sequences.is_empty() {
        trace!("Sequence range: {}..={}", found_sequences.first().unwrap(), found_sequences.last().unwrap());
    }
    
    // BRANCH BY NGRAM TYPE

    match ngram_type {
        NGramType::Word => reconstruct_word_ngrams(
            sequence_range.clone(),
            &text_bytes_map,
            stride,
        ),
        NGramType::Character => reconstruct_char_ngrams(
            sequence_range.clone(),
            &text_bytes_map,
            stride,
        ),
    }
}

fn reconstruct_word_ngrams(
    sequence_range: RangeInclusive<u64>,
    text_bytes_map: &AHashMap<u64, Vec<u8>>,
    stride: usize,
) -> Result<String> {
    // Convert all bytes to strings
    let mut texts = Vec::new();
    for seq in sequence_range {
        if let Some(bytes) = text_bytes_map.get(&seq) {
            let text = String::from_utf8(bytes.clone())
                .map_err(|e| Error::Serialization(format!("Invalid UTF-8 in sequence {}: {}", seq, e)))?;
            texts.push(text);
        } else {
            // Missing sequence in cluster - this is a data integrity issue
            warn!("Missing sequence {} in cluster", seq);
        }
    }
    
    if texts.is_empty() {
        return Ok(String::new());
    }
    
    // Start with first ngram
    let mut result = texts[0].clone();
    
    // For subsequent ngrams, only add the new words (reshingling)
    for text in texts.iter().skip(1) {
        let words: Vec<&str> = text.split_whitespace().collect();
        // With stride=1, only take the last word
        // With stride=2, take the last 2 words, etc.
        if words.len() >= stride {
            let new_words = &words[words.len() - stride..];
            for word in new_words {
                result.push(' ');
                result.push_str(word);
            }
        }
    }
    
    Ok(result.trim().to_string())
}

// Helper function for character-based ngram reconstruction
fn reconstruct_char_ngrams(
    sequence_range: RangeInclusive<u64>,
    text_bytes_map: &AHashMap<u64, Vec<u8>>,
    stride: usize,
) -> Result<String> {
    // Convert all bytes to strings
    let mut texts = Vec::new();
    for seq in sequence_range {
        if let Some(bytes) = text_bytes_map.get(&seq) {
            let text = String::from_utf8(bytes.clone())
                .map_err(|e| Error::Serialization(format!("Invalid UTF-8 in sequence {}: {}", seq, e)))?;
            texts.push(text);
        } else {
            warn!("Missing sequence {} in cluster", seq);
        }
    }
    
    if texts.is_empty() {
        return Ok(String::new());
    }
    
    // Start with first ngram
    let mut result = texts[0].clone();
    
    // For subsequent ngrams, only add the new characters (reshingling)
    for text in texts.iter().skip(1) {
        let chars: Vec<char> = text.chars().collect();
        
        // Example with character bigrams (2-char ngrams) and stride=1:
        // Previous: "ab"
        // Current:  "bc"
        // We only want 'c' (the last 1 character)
        
        if chars.len() >= stride {
            // Take only the last 'stride' characters (the new portion)
            let start_index = chars.len() - stride;
            
            trace!("RESHINGLING: From '{}' taking chars from index {}", 
                text, start_index);
            
            // Append only the new characters
            for c in &chars[start_index..] {
                result.push(*c);
            }
        }
    }
    
    Ok(result.trim().to_string())
}

// ============================================================================
// FILTER FUNCTIONS
// ============================================================================

pub fn passes_final_filters(
    segment: ReconstructedSegment,
    similarity_calc: &SimilarityCalculator,
    config: &AlNaqlConfig,
    text_classifier: &TextClassifier,
    ngram_type: NGramType,
    traced_phrase: Option<&str>,
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Option<ValidatedSegment> {
    
    // Calculate similarity for this final segment
    let similarity = match similarity_calc.compare_texts(
        &segment.source_text,
        &segment.target_text,
        Some(text_classifier),
        config,
        ngram_type,
        Some((segment.source_start_seq, segment.source_end_seq)),
        Some(source_storage),
    ) {
        Ok(sim) => sim,
        Err(e) => {
            trace!("Final filter: REJECTED - similarity calculation failed: {}", e);
            return None;
        }
    };
    
    // Check against the threshold for merged/final text
    if similarity < config.matcher.merged_text_min_confidence {
        trace!("Final filter: REJECTED - similarity {:.3} below threshold {:.3}",
            similarity, config.matcher.merged_text_min_confidence);
        return None;
    }
    
    // Length validation based on ngram type
    match ngram_type {
        NGramType::Word => {
            let source_word_count = segment.source_text.split_whitespace().count();
            let target_word_count = segment.target_text.split_whitespace().count();
            if source_word_count < config.matcher.min_word_length || 
               target_word_count < config.matcher.min_word_length {
                trace!("Final filter: REJECTED - word count too low (source: {}, target: {}, min: {})",
                    source_word_count, target_word_count, config.matcher.min_word_length);
                return None;
            }
        },
        NGramType::Character => {
            let source_char_count = segment.source_text.chars().count();
            let target_char_count = segment.target_text.chars().count();
            if source_char_count < config.matcher.min_chars_length || 
               target_char_count < config.matcher.min_chars_length {
                trace!("Final filter: REJECTED - character count too low (source: {}, target: {}, min: {})",
                    source_char_count, target_char_count, config.matcher.min_chars_length);
                return None;
            }
        }
    }
    
    // Banality check - SOURCE
    let (source_is_banal, _, _) = text_classifier.is_likely_banality(
        &segment.source_text,
        &config,
        traced_phrase.unwrap_or(""),
        Some((segment.source_start_seq, segment.source_end_seq)),  // ADD THIS
        Some(source_storage),  // ADD THIS
    );

    if source_is_banal {
        trace!("Final filter: REJECTED - source text is banal");
        return None;
    }

    // Banality check - TARGET  
    let (target_is_banal, _, _) = text_classifier.is_likely_banality(
        &segment.target_text,
        &config,
        traced_phrase.unwrap_or(""),
        Some((segment.target_start_seq, segment.target_end_seq)),  // ADD THIS
        Some(target_storage),  // ADD THIS
    );

    if target_is_banal {
        trace!("Final filter: REJECTED - target text is banal");
        return None;
    }
    
    // Isnad detection and classification
    let (source_is_isnad, source_is_short, _) = text_classifier.is_likely_isnad(
        &segment.source_text,
        &config.generator,
        traced_phrase.unwrap_or("")
    );
    
    let (target_is_isnad, target_is_short, _) = text_classifier.is_likely_isnad(
        &segment.target_text,
        &config.generator,
        traced_phrase.unwrap_or("")
    );
    
    // Reject short isnads
    if source_is_short || target_is_short {
        trace!("Final filter: REJECTED - short isnad");
        return None;
    }
    
    // Determine if this match is an isnad
    let is_isnad = source_is_isnad || target_is_isnad;
    if is_isnad {
        trace!("Final filter: KEEPING long isnad match");
    }
    
    // All filters passed - create the validated segment
    Some(ValidatedSegment {
        source_text: segment.source_text,
        target_text: segment.target_text,
        source_sequences: (segment.source_start_seq, segment.source_end_seq),  // Convert to tuple
        target_sequences: (segment.target_start_seq, segment.target_end_seq),  // Convert to tuple
        is_isnad,
        similarity_score: similarity as f32,  // Convert f64 to f32
    })
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Checks if two ranges overlap
pub fn ranges_overlap(r1: &Range<usize>, r2: &Range<usize>) -> bool {
    // Two ranges overlap if:
    // - The start of r1 is before the end of r2 AND
    // - The start of r2 is before the end of r1
    r1.start < r2.end && r2.start < r1.end
}

/// Converts validated segments to final output format with byte positions
/// This function batches all position lookups for efficiency
pub fn convert_to_final_results(
    validated_matches: &[ValidatedSegment],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
) -> Result<Vec<FinalMatchResult>> {
    // If there are no matches, return early to avoid unnecessary work
    if validated_matches.is_empty() {
        return Ok(Vec::new());
    }
    
    debug!("Converting {} validated matches to final format with byte positions", 
           validated_matches.len());
    
    // Step 1: Collect all the sequence numbers we need to look up
    // We need the start and end sequence for each match's source and target
    let mut source_sequences_needed = Vec::with_capacity(validated_matches.len() * 2);
    let mut target_sequences_needed = Vec::with_capacity(validated_matches.len() * 2);
    
    for validated_match in validated_matches {
        let (source_start, source_end) = validated_match.source_sequences;
        source_sequences_needed.push(source_start);
        // Only add the end if it's different from start (avoids duplicate lookups)
        if source_end != source_start {
            source_sequences_needed.push(source_end);
        }
        
        let (target_start, target_end) = validated_match.target_sequences;
        target_sequences_needed.push(target_start);
        if target_end != target_start {
            target_sequences_needed.push(target_end);
        }
    }
    
    // Step 2: Batch fetch all position data in just two database calls
    // This is much more efficient than individual lookups for each match
    debug!("Fetching {} source positions and {} target positions", 
           source_sequences_needed.len(), target_sequences_needed.len());
    
    let source_positions = source_storage.get_position_batch(&source_sequences_needed)?;
    let target_positions = target_storage.get_position_batch(&target_sequences_needed)?;
    
    // Step 3: Convert each validated match using the cached position data
    let mut final_results = Vec::with_capacity(validated_matches.len());
    
    for validated_match in validated_matches {
        let (source_start_seq, source_end_seq) = validated_match.source_sequences;
        let (target_start_seq, target_end_seq) = validated_match.target_sequences;
        
        // Look up the positions from our cached data
        // If any position is missing, log a warning and skip this match
        let source_start_pos = match source_positions.get(&source_start_seq) {
            Some(pos) => pos,
            None => {
                warn!("Missing position data for source sequence {}, skipping match", 
                      source_start_seq);
                continue;
            }
        };
        
        let source_end_pos = match source_positions.get(&source_end_seq) {
            Some(pos) => pos,
            None => {
                warn!("Missing position data for source sequence {}, skipping match", 
                      source_end_seq);
                continue;
            }
        };
        
        let target_start_pos = match target_positions.get(&target_start_seq) {
            Some(pos) => pos,
            None => {
                warn!("Missing position data for target sequence {}, skipping match", 
                      target_start_seq);
                continue;
            }
        };
        
        let target_end_pos = match target_positions.get(&target_end_seq) {
            Some(pos) => pos,
            None => {
                warn!("Missing position data for target sequence {}, skipping match", 
                      target_end_seq);
                continue;
            }
        };
        
        // Create the final result with byte positions
        // Note: we use start_byte from the first sequence and end_byte from the last sequence
        // This gives us the complete byte range for the entire matched text
        final_results.push(FinalMatchResult {
            source_text: validated_match.source_text.clone(),
            target_text: validated_match.target_text.clone(),
            source_byte_start: source_start_pos.start_byte,
            source_byte_end: source_end_pos.end_byte,  // End byte of the last sequence
            target_byte_start: target_start_pos.start_byte,
            target_byte_end: target_end_pos.end_byte,  // End byte of the last sequence
            similarity_score: validated_match.similarity_score,
            is_isnad: validated_match.is_isnad,
        });
    }
    
    debug!("Successfully converted {} matches to final format", final_results.len());
    Ok(final_results)
}
