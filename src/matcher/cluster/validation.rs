use ahash::{AHashMap, AHashSet};
use std::sync::Mutex;
use std::time::Instant;
use rayon::prelude::*;
use log::{info, debug, warn, trace};
use std::ops::Range;
use crate::error::Result;
use crate::types::NGramEntry;
use crate::matcher::types::{MatchCandidate, MatchResult};
use crate::matcher::cluster::spatial;
use crate::config::subsystems::generator::NGramType;
use crate::storage::StorageBackend;
use super::matcher::{ClusteringMatcher, TextContent};


impl<S: StorageBackend> ClusteringMatcher<S> {
    pub(super) fn validate_without_merging(
        &self,
        candidates: Vec<MatchCandidate>,
        source_ngrams: &[NGramEntry],
        target_ngrams: &[NGramEntry]
    ) -> Result<Vec<MatchResult>> {
        let start_time = Instant::now();
        debug!("Validating {} candidates without merging", candidates.len());
        
        // Check for traced candidates
        let traced_candidates: Vec<&MatchCandidate> = candidates.iter()
            .filter(|c| c.matches.iter().any(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text)))
            .collect();
                
        if !traced_candidates.is_empty() {
            info!("TRACE [validation]: Starting validation of {} candidates containing traced n-gram", 
                traced_candidates.len());
                
            // Show details of the first few traced candidates
            for (i, candidate) in traced_candidates.iter().enumerate().take(3) {
                info!("TRACE [validation]: Traced candidate #{} has {} matches, source={}..{}, target={}..{}", 
                    i+1, candidate.matches.len(),
                    candidate.source_range.start, candidate.source_range.end,
                    candidate.target_range.start, candidate.target_range.end);
            }
        }
        
        // Create lookup maps for faster access
        let source_map: AHashMap<usize, &NGramEntry> = source_ngrams.iter()
            .map(|ng| (ng.sequence_number as usize, ng))
            .collect();
        
        let target_map: AHashMap<usize, &NGramEntry> = target_ngrams.iter()
            .map(|ng| (ng.sequence_number as usize, ng))
            .collect();
        
        // Create caches for reconstructed text to avoid redundant processing
        let source_text_cache: Mutex<AHashMap<Range<usize>, String>> = Mutex::new(AHashMap::new());
        let target_text_cache: Mutex<AHashMap<Range<usize>, String>> = Mutex::new(AHashMap::new());
        
        // Process candidates in parallel
        let results: Vec<Result<Option<MatchResult>>> = candidates.par_iter()
            .map(|candidate| {
                // Check if this candidate contains our traced n-gram
                let has_traced = candidate.matches.iter()
                    .any(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text));
                
                // Skip if range is too large (anomalous data)
                if candidate.source_range.end - candidate.source_range.start > 1000 ||
                   candidate.target_range.end - candidate.target_range.start > 1000 {
                    if has_traced {
                        info!("TRACE [validation]: Skipping traced candidate with range too large: source={}..{}, target={}..{}", 
                            candidate.source_range.start, candidate.source_range.end,
                            candidate.target_range.start, candidate.target_range.end);
                    }
                    return Ok(None);
                }
                
                // Try to get source text from cache first
                let source_text = {
                    let mut cache = source_text_cache.lock().unwrap();
                    if let Some(text) = cache.get(&candidate.source_range) {
                        if has_traced {
                            info!("TRACE [validation]: Using cached source text for traced candidate");
                        }
                        text.clone()
                    } else {
                        // Extract source ngrams using map lookup instead of filtering
                        let source_range_ngrams: Vec<&NGramEntry> = (candidate.source_range.start..candidate.source_range.end)
                            .filter_map(|pos| source_map.get(&pos).copied())
                            .collect();
                        
                        if source_range_ngrams.is_empty() {
                            if has_traced {
                                info!("TRACE [validation]: No source ngrams found for traced candidate");
                            }
                            return Ok(None); // Skip if no ngrams found
                        }
                        
                        // Reconstruct text
                        let text = self.reconstruct_text_optimized(&source_range_ngrams);

                        if has_traced {
                            info!("TRACE [validation]: Reconstructed source text for traced candidate: '{}'", text);
                        }

                        
                        // Cache it for future use
                        cache.insert(candidate.source_range.clone(), text.clone());
                        text
                    }
                };
                
                // Similar approach for target text
                let target_text = {
                    let mut cache = target_text_cache.lock().unwrap();
                    if let Some(text) = cache.get(&candidate.target_range) {
                        text.clone()
                    } else {
                        let target_range_ngrams: Vec<&NGramEntry> = (candidate.target_range.start..candidate.target_range.end)
                            .filter_map(|pos| target_map.get(&pos).copied())
                            .collect();
                        
                        if target_range_ngrams.is_empty() {
                            if has_traced {
                                info!("TRACE [validation]: No target ngrams found for traced candidate");
                            }
                            return Ok(None); // Skip if no ngrams found
                        }
                        
                        let text = self.reconstruct_text_optimized(&target_range_ngrams);

                        if has_traced {
                            info!("TRACE [validation]: Reconstructed target text for traced candidate: '{}'", text);
                        }

                        
                        cache.insert(candidate.target_range.clone(), text.clone());
                        text
                    }
                };
                               
                // Calculate similarity using existing method
                let similarity = self.similarity_calc.compare_texts(&source_text, &target_text)?;
                
                if has_traced {
                    info!("TRACE [validation]: Similarity for traced candidate: {:.3} (threshold: {:.3})", 
                        similarity, self.matcher_config.initial_text_min_confidence);
                }
                
                if similarity >= self.matcher_config.initial_text_min_confidence {
                    // Map sequence number ranges to byte position ranges
                    // --------------------------------------------------
                    
                    // Find first and last source ngrams by sequence number
                    let first_source_seq = candidate.source_range.start;
                    let last_source_seq = candidate.source_range.end - 1;
                    
                    let first_source_ngram = match source_map.get(&first_source_seq) {
                        Some(ngram) => ngram,
                        None => {
                            warn!("Source ngram not found for sequence {}", first_source_seq);
                            if has_traced {
                                info!("TRACE [validation]: Source ngram not found for sequence {}", first_source_seq);
                            }
                            return Ok(None);
                        }
                    };
                    
                    let last_source_ngram = match source_map.get(&last_source_seq) {
                        Some(ngram) => ngram,
                        None => {
                            warn!("Source ngram not found for sequence {}", last_source_seq);
                            if has_traced {
                                info!("TRACE [validation]: Source ngram not found for sequence {}", last_source_seq);
                            }
                            return Ok(None);
                        }
                    };
                    
                    // Find first and last target ngrams by sequence number
                    let first_target_seq = candidate.target_range.start;
                    let last_target_seq = candidate.target_range.end - 1;
                    
                    let first_target_ngram = match target_map.get(&first_target_seq) {
                        Some(ngram) => ngram,
                        None => {
                            warn!("Target ngram not found for sequence {}", first_target_seq);
                            if has_traced {
                                info!("TRACE [validation]: Target ngram not found for sequence {}", first_target_seq);
                            }
                            return Ok(None);
                        }
                    };
                    
                    let last_target_ngram = match target_map.get(&last_target_seq) {
                        Some(ngram) => ngram,
                        None => {
                            warn!("Target ngram not found for sequence {}", last_target_seq);
                            if has_traced {
                                info!("TRACE [validation]: Target ngram not found for sequence {}", last_target_seq);
                            }
                            return Ok(None);
                        }
                    };
                    
                    // Create byte position ranges
                    let source_byte_range = first_source_ngram.text_position.start_byte..last_source_ngram.text_position.end_byte;
                    let target_byte_range = first_target_ngram.text_position.start_byte..last_target_ngram.text_position.end_byte;
                    
                    if has_traced {
                        info!("TRACE [validation]: Validated traced candidate with similarity {:.3}", similarity);
                        let truncated_text = if source_text.chars().count() > 50 {
                            source_text.chars().take(50).collect::<String>() + "..."
                        } else {
                            source_text.clone()
                        };
                        info!("TRACE [validation]: Source text (sample): '{}'", truncated_text);
                    }
                    
                    // Create MatchResult with byte position ranges
                    Ok(Some(MatchResult {
                        source_text,
                        target_text,
                        similarity,
                        source_range: source_byte_range,
                        target_range: target_byte_range,
                        similarity_metric: self.matcher_config.text_similarity_metric,
                        is_isnad: false,
                        source_seq_range: candidate.source_range.clone(),
                        target_seq_range: candidate.target_range.clone(),
                    }))
                } else {
                    if has_traced {
                        info!("TRACE [validation]: Traced candidate REJECTED - similarity {:.3} below threshold {:.3}", 
                            similarity, self.matcher_config.initial_text_min_confidence);
                    }
                    Ok(None)
                }
            })
            .collect();
        
        // Collect valid matches
        let mut valid_matches = Vec::new();
        let mut traced_matches_count = 0;
        
        for result in results {
            if let Ok(Some(match_result)) = result {
                // Check if this match contains our traced n-gram
                if self.is_traced_ngram(&match_result.source_text) {
                    traced_matches_count += 1;
                    info!("TRACE [validation]: Keeping traced match #{} with similarity {:.3}", 
                        traced_matches_count, match_result.similarity);
                }
                
                valid_matches.push(match_result);
            }
        }
        
        // Summary for traced matches
        if traced_candidates.is_empty() {
            info!("TRACE [validation]: No traced candidates found in this validation batch");
        } else if traced_matches_count > 0 {
            info!("TRACE [validation]: {} of {} traced candidates survived validation", 
                traced_matches_count, traced_candidates.len());
        } else {
            info!("TRACE [validation]: NO traced candidates survived validation (all rejected)");
        }
        
        debug!("Validated {} matches in {:?} (without merging)", valid_matches.len(), start_time.elapsed());
        
        // Return validated matches WITHOUT merging them
        Ok(valid_matches)
    }
    /// Legacy method replaced by stream_merge_from_temp_files
    /// Kept for reference purposes and backward compatibility
    #[allow(dead_code)]
    pub(super) fn perform_final_merging(&self, matches: Vec<MatchResult>) -> Result<Vec<MatchResult>> {
        let start_time = Instant::now();
        info!("Starting final merging process for {} matches using R-tree", matches.len());
        
        // Check for traced matches
        let traced_matches: Vec<&MatchResult> = matches.iter()
            .filter(|m| self.is_traced_ngram(&m.source_text))
            .collect();
                
        if !traced_matches.is_empty() {
            info!("TRACE [r-tree]: Found {} matches containing traced text before merging", 
                traced_matches.len());
                
            // Show first few traced matches
            for (i, m) in traced_matches.iter().enumerate().take(3) {
                info!("TRACE [r-tree]: Traced match #{} - source={}..{}, target={}..{}", 
                    i+1, m.source_range.start, m.source_range.end,
                    m.target_range.start, m.target_range.end);
                    
                let truncated_text = if m.source_text.chars().count() > 50 {
                    m.source_text.chars().take(50).collect::<String>() + "..."
                } else {
                    m.source_text.clone()
                };
                
                info!("TRACE [r-tree]: Traced match text (sample): '{}'", truncated_text);
            }
        }
        
        // Sort matches by source range start for consistent processing
        let mut sorted_matches = matches;
        sorted_matches.sort_by(|a, b| a.source_range.start.cmp(&b.source_range.start));
        
        // STEP 1: Build the R-tree for spatial indexing of matches
        info!("Building R-tree spatial index for {} matches...", sorted_matches.len());
        let rtree = spatial::build_rtree(&sorted_matches);
        
        // STEP 2: Merge overlapping matches using R-tree for efficient candidate finding
        let mut final_matches = Vec::new();
        let mut processed = AHashSet::new();
        
        for (i, match1) in sorted_matches.iter().enumerate() {
            // Skip if already processed
            if processed.contains(&i) {
                continue;
            }
        
            let mut current_match = match1.clone();
            processed.insert(i);
            
            // Check if this match contains our traced n-gram
            let has_traced = self.is_traced_ngram(&current_match.source_text);
            
            if has_traced {
                info!("TRACE [r-tree]: Processing traced match for overlapping merges");
            }
        
            // Find potential overlapping candidates efficiently using R-tree
            let candidates = spatial::find_overlapping_candidates(&rtree, &current_match);
            
            for &j in &candidates {
                if i == j || processed.contains(&j) {
                    continue;
                }
        
                let match2 = &sorted_matches[j];
                
                if self.should_merge_matches(&current_match, match2)? {
                    if has_traced {
                        info!("TRACE [r-tree]: Merging traced match with another match");
                        
                        // Check if the other match also contains our traced n-gram
                        let other_has_traced = self.is_traced_ngram(&match2.source_text);
                        if other_has_traced {
                            info!("TRACE [r-tree]: Both matches contain traced text");
                        }
                    }
                    
                    // Merge the matches
                    let merged_match = self.merge_matches(&current_match, match2)?;
                    current_match = merged_match;
                    processed.insert(j);
                }
            }
        
            final_matches.push(current_match);
        }
        
        info!("After overlapping merging: {} matches", final_matches.len());
        
        // STEP 3: Remove contained matches
        self.remove_contained_matches(&mut final_matches);
        info!("After removing contained matches: {} matches", final_matches.len());
        
        // STEP 4: Merge adjacent, non-overlapping matches using R-tree
        info!("Starting adjacent match merging with R-tree...");
        self.merge_adjacent_matches_with_rtree(&mut final_matches)?;
        info!("After merging adjacent matches: {} matches", final_matches.len());
        
        // STEP 5: Apply final filters and mark isnads
        let mut filtered_matches = Vec::new();
                    
        for match_result in final_matches {
            // Apply all final filters
            if !self.passes_final_filters(&match_result) {
                // If match doesn't pass filters, skip it
                if self.is_traced_ngram(&match_result.source_text) {
                    info!("TRACE [r-tree]: Traced match REJECTED by final filters");
                }
                continue;
            }
            
            // Check if either source or target is an isnad (MARKING only)
            let (source_is_isnad, _, _) = self.is_likely_isnad(&match_result.source_text);
            let (target_is_isnad, _, _) = self.is_likely_isnad(&match_result.target_text);
    
            // Mark as isnad if either source or target is an isnad
            let combined_is_isnad = source_is_isnad || target_is_isnad;
    
            // Create a new match result with is_isnad correctly set
            let mut marked_match = match_result.clone();
            marked_match.is_isnad = combined_is_isnad;
            
            // Add to filtered matches
            filtered_matches.push(marked_match);
            
            // Log if this was a traced match
            if self.is_traced_ngram(&match_result.source_text) {
                if combined_is_isnad {
                    info!("TRACE [r-tree]: Traced match KEPT and marked as isnad");
                } else {
                    info!("TRACE [r-tree]: Traced match KEPT as normal match");
                }
            }
        }
        
        // Final counts
        info!("Final match count after R-tree merging and filtering: {} matches", filtered_matches.len());
        info!("Total R-tree merging process took {:?}", start_time.elapsed());
        
        Ok(filtered_matches)
    }
    
    /// Merge adjacent matches using R-tree spatial index for efficient candidate finding
    pub(super) fn merge_adjacent_matches_with_rtree(
        &self, 
        matches: &mut Vec<MatchResult>
    ) -> Result<()> {
        // Get the merge_gap ratio from config
        let merge_gap_ratio = self.matcher_config.adjacent_merge_ratio;
        
        info!("Starting adjacent match merging with ratio {} using R-tree", merge_gap_ratio);
        
        // No need to continue if there are no matches or only one match
        if matches.len() <= 1 {
            return Ok(());
        }
        
        // Sort matches by source range start for consistent processing
        matches.sort_by(|a, b| a.source_range.start.cmp(&b.source_range.start));
        
        // Create a mutable copy of our matches and build an initial R-tree
        let mut current_matches = matches.clone();
        // Build our own R-tree rather than accepting one as a parameter
        let mut current_rtree = spatial::build_rtree(&current_matches);
        
        // Track whether we've made any merges (to determine if we need another pass)
        let mut made_merges = true;
        let mut iteration = 0;
        
        // Continue merging until no more merges are possible
        while made_merges && iteration < 10 {
            iteration += 1;
            made_merges = false;
            
            // Create a new vector to hold our merged results
            let mut merged_results = Vec::new();
            let mut processed = AHashSet::new();
            
            for i in 0..current_matches.len() {
                // Skip if already processed
                if processed.contains(&i) {
                    continue;
                }
                
                // Get current match
                let current = &current_matches[i];
                processed.insert(i);
                
                // Check if this match contains our traced n-gram
                let has_traced = self.is_traced_ngram(&current.source_text);
                
                // Find adjacent candidates efficiently using R-tree
                let candidates = spatial::find_adjacent_candidates(&current_rtree, current, merge_gap_ratio);
                
                // Find best candidate to merge with
                let mut best_candidate = None;
                let mut best_gap = usize::MAX;
                
                for &j in &candidates {
                    if i == j || processed.contains(&j) {
                        continue;
                    }
                    
                    let next = &current_matches[j];
                    
                    // Skip if positions are wrong (must be after current match)
                    if next.source_range.start <= current.source_range.end ||
                    next.target_range.start <= current.target_range.end {
                        continue;
                    }
                    
                    // Calculate gap
                    let source_gap = next.source_range.start - current.source_range.end;
                    
                    // Check if this is a valid merge and better than previous candidates
                    if self.should_merge_adjacent(current, next, merge_gap_ratio)? && source_gap < best_gap {
                        best_candidate = Some(j);
                        best_gap = source_gap;
                    }
                }
                
                // If we found a candidate, merge with it
                if let Some(j) = best_candidate {
                    if has_traced {
                        info!("TRACE [r-tree]: Merging traced match with adjacent match");
                    }
                    
                    // Create merged match
                    let merged = self.create_merged_adjacent(current, &current_matches[j])?;
                    merged_results.push(merged);
                    processed.insert(j);
                    made_merges = true;
                } else {
                    // No candidate found, keep original match
                    merged_results.push(current.clone());
                }
            }
            
            // Update matches and rebuild R-tree for next iteration
            if made_merges {
                current_matches = merged_results;
                current_rtree = spatial::build_rtree(&current_matches);
                
                info!("Iteration {}: merged adjacent matches, new count: {}", 
                    iteration, current_matches.len());
            }
        }
        
        // Update the original matches vector with our final results
        *matches = current_matches;
        
        Ok(())
    }
    #[allow(dead_code)]
    // Merge adjacent matches that are within the configured gap ratio
    pub(super) fn merge_adjacent_matches(&self, matches: &mut Vec<MatchResult>) -> Result<()> {
        // Get the merge_gap ratio from config
        let merge_gap_ratio = self.matcher_config.adjacent_merge_ratio;
        
        info!("Starting adjacent match merging with ratio {} ({}%)", 
             merge_gap_ratio, merge_gap_ratio * 100.0);
        
        // No need to continue if there are no matches or only one match
        if matches.len() <= 1 {
            return Ok(());
        }
        
        // Sort matches by source range start position
        matches.sort_by(|a, b| a.source_range.start.cmp(&b.source_range.start));
        
        // Log all match positions to verify the order
        info!("Match positions after sorting (showing first 20):");
        for (idx, m) in matches.iter().enumerate().take(20) {
            info!("  Match {}: source={}..{}, target={}..{}", 
                 idx, m.source_range.start, m.source_range.end, 
                 m.target_range.start, m.target_range.end);
        }
        
        // Find our matches of interest
        for (i, m) in matches.iter().enumerate() {
            if m.source_range.start == 46104 && m.source_range.end == 46270 {
                info!("Found our first match of interest at index {}", i);
                
                // Check if there's a next match
                if i + 1 < matches.len() {
                    let next = &matches[i+1];
                    info!("Next match: source={}..{}, target={}..{}", 
                         next.source_range.start, next.source_range.end,
                         next.target_range.start, next.target_range.end);
                    
                    // Calculate gaps and thresholds
                    let source_gap = next.source_range.start - m.source_range.end;
                    let source_size1 = m.source_range.end - m.source_range.start;
                    let source_size2 = next.source_range.end - next.source_range.start;
                    let combined_source_size = source_size1 + source_size2;
                    
                    let target_gap = next.target_range.start - m.target_range.end;
                    let target_size1 = m.target_range.end - m.target_range.start;
                    let target_size2 = next.target_range.end - next.target_range.start;
                    let combined_target_size = target_size1 + target_size2;
                    
                    let max_source_gap = (combined_source_size as f64 * merge_gap_ratio).ceil() as usize;
                    let max_target_gap = (combined_target_size as f64 * merge_gap_ratio).ceil() as usize;
                    
                    info!("Gap analysis: source_gap={}, max_source_gap={} ({}%)", 
                         source_gap, max_source_gap, 
                         (source_gap as f64 / combined_source_size as f64) * 100.0);
                         
                    info!("Gap analysis: target_gap={}, max_target_gap={} ({}%)", 
                         target_gap, max_target_gap, 
                         (target_gap as f64 / combined_target_size as f64) * 100.0);
                    
                    // Check merged similarity
                    let combined_source_text = format!("{} {}", m.source_text, next.source_text);
                    let combined_target_text = format!("{} {}", m.target_text, next.target_text);
                    if let Ok(merged_similarity) = self.similarity_calc.compare_texts(
                        &combined_source_text, &combined_target_text) {
                        
                        info!("Merged similarity: {:.3}", merged_similarity);
                        info!("Threshold: {:.3}", self.matcher_config.merged_text_min_confidence);
                    }
                } else {
                    info!("No next match found - this is the last match");
                }
            }
            
            // Also look for the second match of interest
            if m.source_range.start == 46744 && m.source_range.end == 46993 {
                info!("Found our second match of interest at index {}", i);
                
                // Log the previous match if it exists
                if i > 0 {
                    let prev = &matches[i-1];
                    info!("Previous match: source={}..{}, target={}..{}", 
                         prev.source_range.start, prev.source_range.end,
                         prev.target_range.start, prev.target_range.end);
                }
            }
        }
        
        // Track whether we've made any merges (to determine if we need another pass)
        let mut made_merges = true;
        let mut iteration = 0;
        
        // Continue merging until no more merges are possible
        while made_merges {
            iteration += 1;
            made_merges = false;
            
            // Create a new vector to hold our merged results
            let mut merged_results = Vec::new();
            let mut i = 0;
            
            while i < matches.len() {
                // If this is the last match, just add it and break
                if i == matches.len() - 1 {
                    merged_results.push(matches[i].clone());
                    break;
                }
                
                // Get current and next match
                let current = &matches[i];
                let next = &matches[i + 1];
                
                // Extra logging for our matches of interest
                if (current.source_range.start == 46104 && current.source_range.end == 46270) ||
                   (next.source_range.start == 46744 && next.source_range.end == 46993) {
                    info!("Attempting to merge our match of interest (iteration {}, index {})", 
                         iteration, i);
                    info!("Current: source={}..{}, target={}..{}", 
                         current.source_range.start, current.source_range.end,
                         current.target_range.start, current.target_range.end);
                    info!("Next: source={}..{}, target={}..{}", 
                         next.source_range.start, next.source_range.end,
                         next.target_range.start, next.target_range.end);
                }
                
                // Check if they should be merged
                if self.should_merge_adjacent(current, next, merge_gap_ratio)? {
                    // Merge them and add to results
                    let merged = self.create_merged_adjacent(current, next)?;
                    
                    // Extra logging for our matches of interest
                    if (current.source_range.start == 46104 && current.source_range.end == 46270) ||
                       (next.source_range.start == 46744 && next.source_range.end == 46993) {
                        info!("Successfully merged our matches of interest!");
                        info!("Merged result: source={}..{}, target={}..{}", 
                             merged.source_range.start, merged.source_range.end,
                             merged.target_range.start, merged.target_range.end);
                    }
                    
                    merged_results.push(merged);
                    made_merges = true;
                    i += 2; // Skip both matches we just merged
                } else {
                    // Extra logging for our matches of interest
                    if (current.source_range.start == 46104 && current.source_range.end == 46270) ||
                       (next.source_range.start == 46744 && next.source_range.end == 46993) {
                        info!("Failed to merge our matches of interest!");
                    }
                    
                    // Keep current match as is
                    merged_results.push(current.clone());
                    i += 1;
                }
            }
            
            // If we made merges, update the matches list for the next iteration
            if made_merges {
                *matches = merged_results;
                info!("Iteration {}: merged matches, new count: {}", iteration, matches.len());
            }
            
            // Safety check - don't iterate forever
            if iteration >= 10 {
                info!("Reached maximum iterations (10) for adjacent merging");
                break;
            }
        }
        
        info!("Completed adjacent match merging, resulting in {} matches", matches.len());
        
        Ok(())
    }

    // Helper to check if two matches should be merged based on adjacency criteria
    pub(super) fn should_merge_adjacent(&self, match1: &MatchResult, match2: &MatchResult, merge_gap_ratio: f64) -> Result<bool> {
        // First check - source ranges must be adjacent and not overlapping
        if match2.source_range.start <= match1.source_range.end {
            return Ok(false); // Ranges overlap or are in wrong order
        }
        
        // Same check for target ranges
        if match2.target_range.start <= match1.target_range.end {
            return Ok(false); // Ranges overlap or are in wrong order
        }
        
        // Calculate source sizes (in ngrams/positions)
        let source_size1 = match1.source_range.end - match1.source_range.start;
        let source_size2 = match2.source_range.end - match2.source_range.start;
        let combined_source_size = source_size1 + source_size2;
        
        // Calculate target sizes
        let target_size1 = match1.target_range.end - match1.target_range.start;
        let target_size2 = match2.target_range.end - match2.target_range.start;
        let combined_target_size = target_size1 + target_size2;
        
        // Calculate maximum allowed gaps based on combined sizes
        let max_source_gap = (combined_source_size as f64 * merge_gap_ratio).ceil() as usize;
        let max_target_gap = (combined_target_size as f64 * merge_gap_ratio).ceil() as usize;
        
        // Calculate actual gaps
        let source_gap = match2.source_range.start - match1.source_range.end;
        let target_gap = match2.target_range.start - match1.target_range.end;
        
        // Check if gaps are within allowed limits
        let source_within_limit = source_gap <= max_source_gap;
        let target_within_limit = target_gap <= max_target_gap;
        
        if !source_within_limit || !target_within_limit {
            return Ok(false);
        }
        
        // Preview what the merged result would look like
        let combined_source_text = format!("{} {}", match1.source_text, match2.source_text);
        let combined_target_text = format!("{} {}", match1.target_text, match2.target_text);
        
        // Check merged similarity against threshold
        let merged_similarity = self.similarity_calc.compare_texts(&combined_source_text, &combined_target_text)?;
        let meets_merged_threshold = merged_similarity >= self.matcher_config.merged_text_min_confidence;
        
        info!("Adjacent match check: source_gap={}, max_source_gap={}, target_gap={}, max_target_gap={}, similarity={:.3}, meets_threshold={}",
              source_gap, max_source_gap, target_gap, max_target_gap, 
              merged_similarity, meets_merged_threshold);
        
        Ok(source_within_limit && target_within_limit && meets_merged_threshold)
    }

    // Helper to create a new merged match from two adjacent matches
    pub(super) fn create_merged_adjacent(&self, match1: &MatchResult, match2: &MatchResult) -> Result<MatchResult> {
        // Take the widest possible ranges for byte positions
        let source_start = match1.source_range.start;
        let source_end = match2.source_range.end;
        let target_start = match1.target_range.start;
        let target_end = match2.target_range.end;
        
        // Get the gap sizes (for logging/debugging)
        let source_gap = match2.source_range.start - match1.source_range.end;
        let target_gap = match2.target_range.start - match1.target_range.end;
        
        info!("Merging adjacent matches with source gap {} bytes and target gap {} bytes", 
               source_gap, target_gap);
        
        // Get sequence ranges for retrieving gap ngrams
        let source_gap_start_seq = match1.source_seq_range.end;
        let source_gap_end_seq = match2.source_seq_range.start;
        let target_gap_start_seq = match1.target_seq_range.end;
        let target_gap_end_seq = match2.target_seq_range.start;
        
        info!("Source sequence gap: {}..{}, Target sequence gap: {}..{}", 
               source_gap_start_seq, source_gap_end_seq,
               target_gap_start_seq, target_gap_end_seq);
        
        // Retrieve ngrams that fall within the gaps
        let source_gap_ngrams = if source_gap_start_seq < source_gap_end_seq {
            match self.source_storage.get_sequence_range(source_gap_start_seq as u64..source_gap_end_seq as u64) {
                Ok(ngrams) => {
                    info!("Retrieved {} ngrams for source gap", ngrams.len());
                    ngrams
                },
                Err(e) => {
                    warn!("Error retrieving source gap ngrams: {}", e);
                    Vec::new()
                }
            }
        } else {
            info!("No source sequence gap to fill");
            Vec::new()
        };
        
        let target_gap_ngrams = if target_gap_start_seq < target_gap_end_seq {
            match self.target_storage.get_sequence_range(target_gap_start_seq as u64..target_gap_end_seq as u64) {
                Ok(ngrams) => {
                    info!("Retrieved {} ngrams for target gap", ngrams.len());
                    ngrams
                },
                Err(e) => {
                    warn!("Error retrieving target gap ngrams: {}", e);
                    Vec::new()
                }
            }
        } else {
            info!("No target sequence gap to fill");
            Vec::new()
        };
        
        // Reconstruct text from the gap ngrams
        let source_gap_text = if !source_gap_ngrams.is_empty() {
            // Convert to references for the text reconstruction function
            let gap_ngram_refs: Vec<&NGramEntry> = source_gap_ngrams.iter().collect();
            
            // Use the existing optimized text reconstruction function
            let text = self.reconstruct_text_optimized(&gap_ngram_refs);
            
            // Log a sample of the reconstructed text
            if !text.is_empty() {
                let sample = if text.chars().count() > 50 {
                    let valid_index = text
                        .char_indices() // Get (byte_index, char) pairs
                        .nth(50)        // Get the 50th character
                        .map(|(idx, _)| idx) // Extract the byte index
                        .unwrap_or_else(|| text.len()); // Fallback to full length
                    format!("{}...", &text[..valid_index]) // Slice safely
                } else {
                    text.clone()
                };
                
                info!("Source gap text (sample): '{}'", sample);
            }
            
            // Return the reconstructed text with appropriate spacing
            if text.trim().is_empty() {
                " ".to_string()  // Just a space if empty or whitespace
            } else {
                // Add spaces around the text if needed
                if text.starts_with(' ') && text.ends_with(' ') {
                    text
                } else if text.starts_with(' ') {
                    format!("{} ", text)
                } else if text.ends_with(' ') {
                    format!(" {}", text)
                } else {
                    format!(" {} ", text)
                }
            }
        } else {
            // If no ngrams found, just use a space
            info!("No source gap ngrams found, using space");
            " ".to_string()
        };
        
        // Similar processing for target gap text
        let target_gap_text = if !target_gap_ngrams.is_empty() {
            // Convert to references for the text reconstruction function
            let gap_ngram_refs: Vec<&NGramEntry> = target_gap_ngrams.iter().collect();
            
            // Use the existing optimized text reconstruction function
            let text = self.reconstruct_text_optimized(&gap_ngram_refs);
            
            // Log a sample of the reconstructed text
            if !text.is_empty() {
                let sample = if text.chars().count() > 50 {
                    let valid_index = text
                        .char_indices() // Get (byte_index, char) pairs
                        .nth(50)        // Get the 50th character
                        .map(|(idx, _)| idx) // Extract the byte index
                        .unwrap_or_else(|| text.len()); // Fallback to full length
                    format!("{}...", &text[..valid_index]) // Slice safely
                } else {
                    text.clone()
                };

                info!("Target gap text (sample): '{}'", sample);
            }
            
            // Return the reconstructed text with appropriate spacing
            if text.trim().is_empty() {
                " ".to_string()  // Just a space if empty or whitespace
            } else {
                // Add spaces around the text if needed
                if text.starts_with(' ') && text.ends_with(' ') {
                    text
                } else if text.starts_with(' ') {
                    format!("{} ", text)
                } else if text.ends_with(' ') {
                    format!(" {}", text)
                } else {
                    format!(" {} ", text)
                }
            }
        } else {
            // If no ngrams found, just use a space
            info!("No target gap ngrams found, using space");
            " ".to_string()
        };
        
        // Now create the combined texts with the gap text properly inserted
        let combined_source_text = format!("{}{}{}", match1.source_text, source_gap_text, match2.source_text);
        let combined_target_text = format!("{}{}{}", match1.target_text, target_gap_text, match2.target_text);
        
        // Calculate similarity of the combined text
        let similarity = match self.similarity_calc.compare_texts(&combined_source_text, &combined_target_text) {
            Ok(sim) => {
                info!("Calculated similarity for merged text: {:.3}", sim);
                sim
            },
            Err(e) => {
                warn!("Error calculating similarity: {}", e);
                // Fallback to average of original similarities
                (match1.similarity + match2.similarity) / 2.0
            }
        };
        
        // Combine isnad flags - if either was an isnad, the merged result is an isnad
        let is_isnad = match1.is_isnad || match2.is_isnad;
        
        info!("Created merged match with full gap text and similarity {:.3}", similarity);
        
        // Return the new merged match result with complete text
        Ok(MatchResult {
            source_text: combined_source_text,
            target_text: combined_target_text,
            similarity,
            source_range: source_start..source_end,
            target_range: target_start..target_end,
            similarity_metric: match1.similarity_metric,
            is_isnad,
            source_seq_range: match1.source_seq_range.start..match2.source_seq_range.end,
            target_seq_range: match1.target_seq_range.start..match2.target_seq_range.end,
        })
    }
    
    pub(super) fn should_merge_matches(&self, m1: &MatchResult, m2: &MatchResult) -> Result<bool> {
        // Check if source ranges are close enough to merge
        let source_overlap = ranges_overlap(&m1.source_range, &m2.source_range);
        
        // Check if target ranges are close enough to merge
        let target_overlap = ranges_overlap(&m1.target_range, &m2.target_range);
        
        if !source_overlap || !target_overlap {
            return Ok(false);
        }
        
        // Preview what the merged result would look like
        // Use the longer text versions for the test
        let (source_text, target_text) = if m1.source_text.len() > m2.source_text.len() {
            (&m1.source_text, &m1.target_text)
        } else {
            (&m2.source_text, &m2.target_text)
        };
        
        // Calculate similarity for what would be the merged result
        let merged_similarity = self.similarity_calc.compare_texts(source_text, target_text)?;
        
        // Only merge if the merged result would meet the merged text threshold
        let meets_merged_threshold = merged_similarity >= self.matcher_config.merged_text_min_confidence;
        
        trace!("Checking merge compatibility:\n  Source overlap: {}\n  Target overlap: {}\n  Merged similarity: {:.3}\n  Meets threshold: {}", 
            source_overlap, target_overlap, merged_similarity, meets_merged_threshold);
        
        Ok(source_overlap && target_overlap && meets_merged_threshold)
    }
    
    pub(super) fn merge_matches(&self, m1: &MatchResult, m2: &MatchResult) -> Result<MatchResult> {
        // Take the widest possible ranges
        let source_start = m1.source_range.start.min(m2.source_range.start);
        let source_end = m1.source_range.end.max(m2.source_range.end);
        let target_start = m1.target_range.start.min(m2.target_range.start);
        let target_end = m1.target_range.end.max(m2.target_range.end);
    
        // Use the longer text versions
        let (source_text, target_text) = if m1.source_text.len() > m2.source_text.len() {
            (m1.source_text.clone(), m1.target_text.clone())
        } else {
            (m2.source_text.clone(), m2.target_text.clone())
        };
    
        // Calculate similarity for the merged result
        let similarity = self.similarity_calc.compare_texts(&source_text, &target_text)?;
        
        // Combine isnad flags - if either was an isnad, the merged result is an isnad
        let is_isnad = m1.is_isnad || m2.is_isnad;
    
        Ok(MatchResult {
            source_text,
            target_text,
            similarity,
            source_range: source_start..source_end,
            target_range: target_start..target_end,
            similarity_metric: m1.similarity_metric,
            is_isnad,
            source_seq_range: m1.source_seq_range.start..m2.source_seq_range.end,
            target_seq_range: m1.target_seq_range.start..m2.target_seq_range.end,
        })
    }

    pub(super) fn remove_contained_matches(&self, matches: &mut Vec<MatchResult>) {
        if matches.len() <= 1 {
            return; // Nothing to do with 0 or 1 matches
        }
        
        // Step 1: Sort matches by source start position, then by source range size (ascending)
        // This creates an order where potential containers come before the matches they might contain
        matches.sort_by(|a, b| {
            a.source_range.start.cmp(&b.source_range.start)
                .then_with(|| (b.source_range.end - b.source_range.start)
                    .cmp(&(a.source_range.end - a.source_range.start)))
        });
        
        // Step 2: Use a more efficient algorithm to identify contained matches
        let mut to_keep = vec![true; matches.len()];
        let mut keep_count = matches.len();
        
        // Outer loop - for each potential "container" match
        for i in 0..matches.len() {
            // Skip if this match has already been marked for removal
            if !to_keep[i] {
                continue;
            }
            
            let container = &matches[i];
            
            // Inner loop - only check potential "containees" that come after this match
            // This is the key optimization - we only check matches where:
            // 1. They start at or after our container (guaranteed by the sort)
            // 2. They end before or at our container's end
            for j in (i+1)..matches.len() {
                // Skip if already marked for removal
                if !to_keep[j] {
                    continue;
                }
                
                let potential_containee = &matches[j];
                
                // Early exit: If this match starts beyond the container's end, we can
                // stop checking as all subsequent matches will also start beyond it
                if potential_containee.source_range.start >= container.source_range.end {
                    break;
                }
                
                // Check containment: A match is contained if its source and target ranges
                // are both fully within another match's ranges
                if potential_containee.source_range.start >= container.source_range.start && 
                   potential_containee.source_range.end <= container.source_range.end &&
                   potential_containee.target_range.start >= container.target_range.start && 
                   potential_containee.target_range.end <= container.target_range.end {
                    
                    to_keep[j] = false;
                    keep_count -= 1;
                }
            }
        }
        
        // Step 3: Only rebuild the vector if we found matches to remove
        if keep_count < matches.len() {
            let retained = matches.len() - keep_count;
            info!("Removing {} contained matches", retained);
            
            // Create a new vector with only the matches we want to keep
            let filtered = matches
                .iter()
                .zip(to_keep.iter())
                .filter_map(|(m, &keep)| if keep { Some(m.clone()) } else { None })
                .collect::<Vec<MatchResult>>();
            
            // Replace the original vector
            *matches = filtered;
        }
    }

    pub(super) fn reconstruct_text_optimized<'a, T>(&self, items: &[&'a T]) -> String 
    where 
        T: TextContent + 'a 
    {
        if items.is_empty() {
            trace!("reconstruct_text_optimized called with empty items");
            return String::new();
        }

        // Sort items by position
        let mut sorted_items = items.to_vec();
        sorted_items.sort_by_key(|item| item.get_position());
        
        trace!("Reconstructing text from {} sorted items", sorted_items.len());
        trace!("Item positions: {:?}", sorted_items.iter().map(|i| i.get_position()).collect::<Vec<_>>());
        
        match self.ngram_type {
            NGramType::Character => {
                trace!("Using Character ngram reconstruction");
                // Preallocate with estimated capacity
                let mut text = String::with_capacity(items.len() * 10);
                
                // Start with first item's text
                text.push_str(sorted_items[0].get_cleaned_text());
                let mut last_position = sorted_items[0].get_position();
                trace!("Starting with text: '{}', position: {}", text, last_position);
                
                // Process subsequent items
                for (i, item) in sorted_items.iter().enumerate().skip(1) {
                    let current_position = item.get_position();
                    let current_text = item.get_cleaned_text();
                    
                    // Calculate gap
                    let gap = current_position.saturating_sub(last_position);
                    debug!("Item {}: text='{}', position={}, gap={}", i, current_text, current_position, gap);
                    
                    // Only break for sequential mode (not for proximity)
                    if !self.matcher_config.proximity_matching && gap > self.matcher_config.max_gap {
                        debug!("Gap {} exceeds max gap {}, stopping (sequential mode)", gap, self.matcher_config.max_gap);
                        break;
                    }

                    // For sequential positions based on configured stride
                    if gap == self.processor_config.generator.stride {
                        // Get the last 'stride' characters (or fewer if the text is shorter)
                        let chars: Vec<char> = current_text.chars().collect();
                        let start_index = chars.len().saturating_sub(self.processor_config.generator.stride);
                        
                        debug!("Sequential item with stride {}, appending last {} chars", 
                            self.processor_config.generator.stride, chars.len() - start_index);
                        
                        for &c in &chars[start_index..] {
                            text.push(c);
                        }
                    } else if gap > self.processor_config.generator.stride {
                        // For gaps, insert space and full text
                        debug!("Non-sequential item with gap {}, appending full text", gap);
                        text.push(' ');
                        text.push_str(current_text);
                    }

                    last_position = current_position;
                }

                debug!("Final character text (length={}): '{}'", text.len(), text);
                text
            },
            
            NGramType::Word => {
                debug!("Using Word ngram reconstruction");
                
                // Get the first ngram's text
                let first_text = sorted_items[0].get_cleaned_text();
                debug!("Starting with first ngram: '{}'", first_text);
                
                // Extract all words from all ngrams
                let all_words: Vec<Vec<&str>> = sorted_items.iter()
                    .map(|item| item.get_cleaned_text().split_whitespace().collect())
                    .collect();
                    
                trace!("Word breakdown per ngram:");
                for (i, words) in all_words.iter().enumerate() {
                    debug!("  Ngram #{}: {:?}", i, words);
                }
                
                // Reconstruct text maintaining all words
                let mut result = String::new();
                
                // Special handling for n-grams that properly maintains sliding windows
                if sorted_items.len() == 1 {
                    // For single item, just use its text
                    result = first_text.to_string();
                } else {
                    // For n-grams, we need to handle the sliding window pattern
                    // Start with the first ngram
                    result.push_str(first_text);
                    let mut last_pos = sorted_items[0].get_position();
                    
                    for (i, item) in sorted_items.iter().enumerate().skip(1) {
                        let pos = item.get_position();
                        let text = item.get_cleaned_text();
                        let gap = pos.saturating_sub(last_pos);
                        
                        debug!("Processing ngram #{}: '{}', position={}, last_pos={}, gap={}", 
                            i, text, pos, last_pos, gap);
                        
                        // Add words based on gap and patterns
                        if gap == self.processor_config.generator.stride {
                            // Sequential with stride - add the last 'stride' words
                            // since they overlap by (n-stride) words
                            let curr_words: Vec<&str> = text.split_whitespace().collect();
                            
                            // Calculate how many words to add (last 'stride' words)
                            let start_index = curr_words.len().saturating_sub(self.processor_config.generator.stride);
                            
                            debug!("Sequential ngram with stride {} - adding last {} words", 
                                self.processor_config.generator.stride, curr_words.len() - start_index);
                            
                            // Add each of the last words with spaces
                            for word in &curr_words[start_index..] {
                                result.push(' ');
                                result.push_str(word);
                            }
                        } else if gap > self.processor_config.generator.stride {
                            // For any gap, add a space and full text
                            // REMOVED the upper bound check (gap <= self.matcher_config.max_gap)
                            debug!("Non-sequential ngram (gap={}) - adding full text", gap);
                            result.push(' ');
                            result.push_str(text);
                        }
                        // REMOVED the else branch with the break statement
                        
                        last_pos = pos;
                    }
                }
                
                debug!("Final word text (length={}): '{}'", result.len(), result);
                result
            }
        }
    }

    pub(super) fn is_likely_banality(&self, text: &str) -> (bool, f64, usize) {
        
        trace!("BANALITY CHECK: Checking text: '{}'", text);
        
        // Skip empty strings
        if text.is_empty() {
            return (false, 0.0, 0);
        }
        
        // Check if this text contains our traced n-gram
        let has_traced = self.is_traced_ngram(text);
        
        // For automatic detection, we would ideally have n-gram IDs, but here we only have text
        // Since this is a validation step, we should generally continue with phrase-based detection
        // Automatic detection is better used directly with n-gram matches in check_automatic_banality
        if self.banality_detection_mode == "automatic" && has_traced {
            info!("TRACE [banality]: Using phrase-based detection for validation (n-gram IDs not available)");
        }
        
        // Original phrase-based detection logic based on ngram_type
        match self.ngram_type {
            NGramType::Character => {
                // For character ngrams, use direct character comparison
                let total_chars = text.chars().count();
                
                if total_chars == 0 {
                    return (false, 0.0, 0);
                }
                
                // Check for exact phrases first (direct substring matching)
                let mut matched_chars = 0;
                let mut matched_items: Vec<String> = Vec::new();
                
                // Look for banality phrases in the text
                for phrase in &self.banality_phrases {
                    if text.contains(phrase) {
                        matched_chars += phrase.chars().count();
                        
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                    }
                }
                
                // Avoid overcounting
                matched_chars = matched_chars.min(total_chars);
                let density = matched_chars as f64 / total_chars as f64;
                
                // Estimate word count for character-based ngrams by counting spaces + 1
                let estimated_word_count = text.matches(' ').count() + 1;
                
                if density >= self.banality_density_threshold {
                    if has_traced {
                        info!("TRACE [banality]: Traced text detected as banality (character mode) - density: {:.2} ({} of {} chars)",
                            density, matched_chars, total_chars);
                        if !matched_items.is_empty() {
                            info!("TRACE [banality]: Matching items: {:?}", matched_items);
                        }
                    }
                    
                    debug!("Found banality by character density: {:.2} ({} of {} chars)",
                          density, matched_chars, total_chars);
                    return (true, density, estimated_word_count);
                }
                
                if has_traced {
                    info!("TRACE [banality]: Traced text is NOT a banality (character mode) - density: {:.2} (threshold: {:.2})",
                        density, self.banality_density_threshold);
                    if !matched_items.is_empty() {
                        info!("TRACE [banality]: Had some banal elements but below threshold: {:?}", matched_items);
                    }
                }
                
                (false, density, estimated_word_count)
            },
            
            NGramType::Word => {
                // Original word-based implementation
                let words: Vec<&str> = text.split_whitespace().collect();
                let word_count = words.len();
                
                if word_count == 0 {
                    return (false, 0.0, 0);
                }
                
                // Track which words have been covered by phrases
                let mut covered_positions = vec![false; word_count];
                let mut matching_words = 0;
                let mut matched_items: Vec<String> = Vec::new();
                
                // First check for banality phrases
                for phrase in &self.banality_phrases {
                    if text.contains(phrase) {
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                        
                        // Find all occurrences of this phrase in the text
                        let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                        
                        // For each potential starting position in the text
                        for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                            let mut matches = true;
                            
                            // Check if phrase matches at this position
                            for (offset, phrase_word) in phrase_words.iter().enumerate() {
                                if start_pos + offset >= word_count || words[start_pos + offset] != *phrase_word {
                                    matches = false;
                                    break;
                                }
                            }
                            
                            // If phrase matches, mark all its positions as covered
                            if matches {
                                for offset in 0..phrase_words.len() {
                                    if !covered_positions[start_pos + offset] {
                                        covered_positions[start_pos + offset] = true;
                                        matching_words += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Then check for individual words, only counting those not already covered by phrases
                for (pos, word) in words.iter().enumerate() {
                    if !covered_positions[pos] && self.banality_words.contains(&word.to_string()) {
                        matching_words += 1;
                        covered_positions[pos] = true;
                        
                        if has_traced {
                            matched_items.push(format!("word: '{}'", word));
                        }
                    }
                }
                
                let density = matching_words as f64 / word_count as f64;
                
                if density >= self.banality_density_threshold {
                    if has_traced {
                        info!("TRACE [banality]: Traced text detected as banality - density: {:.2} ({} of {} words)",
                            density, matching_words, word_count);
                        info!("TRACE [banality]: Matching items: {:?}", matched_items);
                    }
                    
                    debug!("Found banality by word density: {:.2} ({} of {} words)",
                          density, matching_words, word_count);
                    return (true, density, word_count);
                }
                
                if has_traced {
                    info!("TRACE [banality]: Traced text is NOT a banality - density: {:.2} (threshold: {:.2})",
                        density, self.banality_density_threshold);
                    if !matched_items.is_empty() {
                        info!("TRACE [banality]: Had some banal elements but below threshold: {:?}", matched_items);
                    }
                }
                
                (false, density, word_count)
            }
        }
    }
    
    pub(super) fn is_likely_isnad(&self, text: &str) -> (bool, f64, usize) {
        trace!("ISNAD CHECK: Checking text: '{}'", text);
        
        // Different approach based on ngram type
        match self.ngram_type {
            NGramType::Character => {
                // For character ngrams, use direct character comparison
                let total_chars = text.chars().count();
                
                if total_chars == 0 {
                    return (false, 0.0, 0);
                }
                
                // Check if this text contains our traced n-gram
                let has_traced = self.is_traced_ngram(text);
                
                // Check for exact phrases first (direct substring matching)
                let mut matched_chars = 0;
                let mut matched_items = Vec::new();
                
                // Look for isnad phrases in the text
                for phrase in &self.isnad_phrases {
                    if text.contains(phrase) {
                        matched_chars += phrase.chars().count();
                        
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                    }
                }
                
                // Avoid overcounting
                matched_chars = matched_chars.min(total_chars);
                let density = matched_chars as f64 / total_chars as f64;
                
                // Estimate word count for character-based ngrams by counting spaces + 1
                let estimated_word_count = text.matches(' ').count() + 1;
                
                if density >= self.isnad_density_threshold {
                    if has_traced {
                        info!("TRACE [isnad]: Traced text detected as isnad (character mode) - density: {:.2} ({} of {} chars)",
                            density, matched_chars, total_chars);
                        if !matched_items.is_empty() {
                            info!("TRACE [isnad]: Matching items: {:?}", matched_items);
                        }
                        info!("TRACE [isnad]: Estimated word count: {} (min required: {})", 
                            estimated_word_count, self.isnad_min_length);
                    }
                    
                    debug!("Found isnad by character density: {:.2} ({} of {} chars)",
                          density, matched_chars, total_chars);
                    
                    // Determine if this is a short isnad that should be discarded
                    let is_short = estimated_word_count < self.isnad_min_length;
                    
                    // Return is_short as the first value to indicate if it should be discarded
                    return (is_short, density, estimated_word_count);
                }
                
                if has_traced {
                    info!("TRACE [isnad]: Traced text is NOT an isnad (character mode) - density: {:.2} (threshold: {:.2})",
                        density, self.isnad_density_threshold);
                    if !matched_items.is_empty() {
                        info!("TRACE [isnad]: Had some isnad elements but below threshold: {:?}", matched_items);
                    }
                }
                
                (false, density, estimated_word_count)
            },
            
            NGramType::Word => {
                // Original word-based implementation
                let words: Vec<&str> = text.split_whitespace().collect();
                let word_count = words.len();
                
                if word_count == 0 {
                    return (false, 0.0, 0);
                }
                
                // Check if this text contains our traced n-gram
                let has_traced = self.is_traced_ngram(text);
                
                // Track which words have been covered by phrases
                let mut covered_positions = vec![false; word_count];
                let mut matching_words = 0;
                let mut matched_items = Vec::new();
                
                // First check for isnad phrases
                for phrase in &self.isnad_phrases {
                    if text.contains(phrase) {
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                        
                        // Find all occurrences of this phrase in the text
                        let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                        
                        // For each potential starting position in the text
                        for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                            let mut matches = true;
                            
                            // Check if phrase matches at this position
                            for (offset, phrase_word) in phrase_words.iter().enumerate() {
                                if start_pos + offset >= word_count || words[start_pos + offset] != *phrase_word {
                                    matches = false;
                                    break;
                                }
                            }
                            
                            // If phrase matches, mark all its positions as covered
                            if matches {
                                for offset in 0..phrase_words.len() {
                                    if !covered_positions[start_pos + offset] {
                                        covered_positions[start_pos + offset] = true;
                                        matching_words += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Then check for individual words, only counting those not already covered by phrases
                for (pos, word) in words.iter().enumerate() {
                    if !covered_positions[pos] && self.isnad_words.contains(&word.to_string()) {
                        matching_words += 1;
                        covered_positions[pos] = true;
                        
                        if has_traced {
                            matched_items.push(format!("word: '{}'", word));
                        }
                    }
                }
                
                let density = matching_words as f64 / word_count as f64;
                
                if density >= self.isnad_density_threshold {
                    if has_traced {
                        info!("TRACE [isnad]: Traced text detected as isnad - density: {:.2} ({} of {} words)",
                            density, matching_words, word_count);
                        info!("TRACE [isnad]: Matching items: {:?}", matched_items);
                        info!("TRACE [isnad]: Word count: {} (min required: {})", word_count, self.isnad_min_length);
                    }
                    
                    debug!("Found isnad by word density: {:.2} ({} of {} words)",
                          density, matching_words, word_count);
                    return (true, density, word_count);
                }
                
                if has_traced {
                    info!("TRACE [isnad]: Traced text is NOT an isnad - density: {:.2} (threshold: {:.2})",
                        density, self.isnad_density_threshold);
                    if !matched_items.is_empty() {
                        info!("TRACE [isnad]: Had some isnad elements but below threshold: {:?}", matched_items);
                    }
                }
                
                (false, density, word_count)
            }
        }
    }
    pub fn passes_early_filters(&self, text: &str, _chain_length: usize) -> bool {
        // Check if this contains our traced text
        let has_traced = self.is_traced_ngram(text);
        if has_traced {
            info!("TRACE [early_filters]: Checking traced text");
        }
                
        // Early banality check (if enabled)
        if self.processor_config.early_banality_detection {
            let (is_banal, density) = self.is_early_banal(text);
            if is_banal {
                debug!("Early filter: REJECTED - detected as banal text (density: {:.2})", density);
                if has_traced {
                    info!("TRACE [early_filters]: Traced text REJECTED - detected as banal text (density: {:.2})", density);
                }
                return false;
            }
        }
        
        // Early isnad check (if enabled)
        if self.processor_config.early_isnad_detection {
            let (is_short_isnad, density, word_count) = self.is_early_isnad(text);
            if is_short_isnad {
                debug!("Early filter: REJECTED - detected as short isnad (density: {:.2}, words: {})", 
                    density, word_count);
                if has_traced {
                    info!("TRACE [early_filters]: Traced text REJECTED - detected as short isnad (density: {:.2}, words: {})", 
                        density, word_count);
                }
                return false;
            }
        }
        
        // All early filters passed
        if has_traced {
            info!("TRACE [early_filters]: Traced text PASSED all early filters");
        }
        
        true
    }
    
    
    // Function for final filtering (post-validation)
    pub fn passes_final_filters(&self, match_result: &MatchResult) -> bool {
        // Check if this match contains our traced n-gram
        let has_traced = self.is_traced_ngram(&match_result.source_text);
        if has_traced {
            info!("TRACE [final_filters]: Checking traced match");
        }
        
        // Length filter for both source and target
        match self.ngram_type {
            NGramType::Word => {
                // For word-based ngrams, check word count for both texts
                let source_word_count = match_result.source_text.split_whitespace().count();
                let target_word_count = match_result.target_text.split_whitespace().count();
                let min_length = self.matcher_config.min_word_length;
                
                if source_word_count < min_length || target_word_count < min_length {
                    debug!("Final filter: REJECTED - word count too low (source: {}, target: {}, min: {})", 
                        source_word_count, target_word_count, min_length);
                    return false;
                }
            },
            NGramType::Character => {
                // For character-based ngrams, check character count for both texts
                let source_char_count = match_result.source_text.chars().count();
                let target_char_count = match_result.target_text.chars().count();
                let min_length = self.matcher_config.min_chars_length;
                
                if source_char_count < min_length || target_char_count < min_length {
                    debug!("Final filter: REJECTED - character count too low (source: {}, target: {}, min: {})", 
                        source_char_count, target_char_count, min_length);
                    return false;
                }
            }
        }
        
        // Check for banality in both source and target
        let (source_is_banal, _source_density, _) = self.is_likely_banality(&match_result.source_text);
        let (target_is_banal, _target_density, _) = self.is_likely_banality(&match_result.target_text);

        if source_is_banal || target_is_banal {
            debug!("Final filter: REJECTED - detected as banality");
            if has_traced {
                info!("TRACE [final_filters]: Traced match REJECTED - detected as banality");
            }
            return false;
        }
        
        // Check for isnad in both source and target
        let (source_is_isnad, _source_isnad_density, source_word_count) = self.is_likely_isnad(&match_result.source_text);
        let (target_is_isnad, _target_isnad_density, target_word_count) = self.is_likely_isnad(&match_result.target_text);

        // Reject if either text is a short isnad (below the threshold)
        if (source_is_isnad && source_word_count < self.isnad_min_length) || 
        (target_is_isnad && target_word_count < self.isnad_min_length) {
            debug!("Final filter: REJECTED - detected as short isnad (source: {}/{} words, target: {}/{} words)", 
                source_is_isnad, source_word_count, 
                target_is_isnad, target_word_count);
            if has_traced {
                info!("TRACE [final_filters]: Traced match REJECTED - detected as short isnad");
            }
            return false;
        }
        
        // All final filters passed
        if has_traced {
            info!("TRACE [final_filters]: Traced match PASSED all final filters");
        }
        
        true
    }
}

pub(super) fn ranges_overlap(r1: &Range<usize>, r2: &Range<usize>) -> bool {
    // Allow small gaps between ranges (5 positions)
    const ADJACENT_THRESHOLD: usize = 30;
    
    // Check if ranges overlap or are very close to each other
    r1.start <= r2.end + ADJACENT_THRESHOLD && r2.start <= r1.end + ADJACENT_THRESHOLD
}