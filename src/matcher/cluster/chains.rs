use bit_vec::BitVec;
use ahash::{AHashMap, AHashSet};
use std::time::Instant;
use log::{info, debug, trace, log_enabled, Level};
use crate::error::{Result, Error};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use crate::matcher::types::{NGramMatch, MatchCandidate};
use crate::config::subsystems::generator::NGramType;
use crate::matcher::cluster::parallel_chains::ParallelProcessingStats;
use crate::storage::StorageBackend;
use super::matcher::ClusteringMatcher;

impl<S: StorageBackend> ClusteringMatcher<S> {
    pub(super) fn group_into_chains_optimized(
        &self,
        matches: &[NGramMatch], 
        max_gap: usize,
        traced_phrase: &str
    ) -> Vec<Vec<usize>> {
        if matches.is_empty() {
            return Vec::new();
        }
        
        // Helper function to check for traced ngram - remains the same
        let contains_traced = |text: &str| !traced_phrase.is_empty() && text.contains(traced_phrase);
        
                              
        // Search for traced-ngram
        let traced_matches: Vec<(usize, &NGramMatch)> = if log_enabled!(log::Level::Info) && !self.matcher_config.traced_phrase.is_empty() {
            matches.iter()
                .enumerate()
                .filter(|(_, m)| contains_traced(&m.ngram.text_content.cleaned_text))
                .collect()
        } else {
            Vec::new() // Empty vec if logging disabled or no traced phrase
        };

        if !traced_matches.is_empty() {
            info!("TRACE [chains]: Found {} matches containing traced n-gram before chain building", 
                traced_matches.len());
                                    
            for (i, (match_idx, m)) in traced_matches.iter().enumerate().take(5) {
                info!("TRACE [chains]: Traced match #{}: source={}, target={}, text='{}' (index: {})", 
                    i+1, m.source_position, m.target_position, m.ngram.text_content.cleaned_text, match_idx);
            }
        } else if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(log::Level::Trace) {
            trace!("TRACE [chains]: NO MATCHES containing traced n-gram found before chain building!");
        }
        
        // Create a map of match indices by source position
        let mut source_groups: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for (idx, m) in matches.iter().enumerate() {
            source_groups.entry(m.source_position)
                .or_insert_with(Vec::new)
                .push(idx);
        }
        
        // Get sorted list of source positions
        let mut source_positions: Vec<usize> = source_groups.keys().cloned().collect();
        source_positions.sort();
        
        let mut all_chains = Vec::new();
        let mut used_pairs: AHashSet<(usize, usize)> = AHashSet::new();
        
        // Track chains with our traced n-gram
        let mut traced_chain_count = 0;
        
        // Process each source position in order
        for &start_pos in &source_positions {
            if let Some(start_indices) = source_groups.get(&start_pos) {
                for &start_idx in start_indices {
                    let start_match = &matches[start_idx];
                    
                    // Skip if already used
                    if used_pairs.contains(&(start_match.source_position, start_match.target_position)) {
                        continue;
                    }
                    
                    // Start new chain with this match index
                    let mut chain = vec![start_idx];
                    let mut current_source = start_match.source_position;
                    let mut current_target = start_match.target_position;
                    used_pairs.insert((current_source, current_target));
                    
                    // Check if starting with our traced n-gram
                    let chain_starts_with_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                        contains_traced(&start_match.ngram.text_content.cleaned_text)
                    } else {
                        false
                    };

                    if chain_starts_with_traced {
                        info!("TRACE [chains]: Starting new chain with traced n-gram at source={}, target={}",
                            current_source, current_target);
                    }
                    
                    // Keep extending the chain as long as possible
                    loop {
                        // Collect all valid next positions first
                        let valid_next_sources: Vec<usize> = source_positions.iter()
                            .filter(|&&pos| pos > current_source && 
                                    pos.saturating_sub(current_source) <= max_gap)
                            .cloned()
                            .collect();
                        
                        if valid_next_sources.is_empty() {
                            break; // No more valid positions
                        }
                        
                        // Track best next match
                        let mut best_match_idx: Option<usize> = None;
                        let mut best_source_gap = usize::MAX;
                        
                        // Find the best next match across all valid positions
                        for next_source in valid_next_sources {
                            if let Some(candidate_indices) = source_groups.get(&next_source) {
                                for &candidate_idx in candidate_indices {
                                    let candidate = &matches[candidate_idx];
                                    
                                    // Skip if used or not advancing target position
                                    if used_pairs.contains(&(candidate.source_position, candidate.target_position)) || 
                                       candidate.target_position <= current_target {
                                        continue;
                                    }
                                    
                                    // OPTIMIZATION: Only compare with the last match in the chain
                                    // instead of checking against all previous matches
                                    let source_gap = candidate.source_position - current_source;
                                    let target_gap = candidate.target_position - current_target;
                                    
                                    // Only consider if both gaps are within limit
                                    if source_gap <= max_gap && target_gap <= max_gap {
                                        // This is a valid next match
                                        // Take the one with smallest position gap
                                        if source_gap < best_source_gap {
                                            best_match_idx = Some(candidate_idx);
                                            best_source_gap = source_gap;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // If we found a next match, add it to chain
                        if let Some(next_idx) = best_match_idx {
                            let next_match = &matches[next_idx];
                            
                            // Check if next match contains traced n-gram
                            let next_has_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                                contains_traced(&next_match.ngram.text_content.cleaned_text)
                            } else {
                                false
                            };
                            
                            if next_has_traced && chain_starts_with_traced {
                                info!("TRACE [chains]: Adding traced n-gram to existing traced chain at position {}",
                                     chain.len());
                            } else if next_has_traced {
                                info!("TRACE [chains]: Adding traced n-gram to chain at position {}", chain.len());
                            }
                            
                            chain.push(next_idx);
                            used_pairs.insert((next_match.source_position, next_match.target_position));
                            current_source = next_match.source_position;
                            current_target = next_match.target_position;
                        } else {
                            break; // No valid next match found
                        }
                    }
                    
                    // Check if chain contains our traced n-gram
                    let chain_has_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                        chain.iter().any(|&idx| contains_traced(&matches[idx].ngram.text_content.cleaned_text))
                    } else {
                        false
                    };
                                        
                    // Minimum chain length
                    if chain.len() >= 2 {
                        if chain_has_traced {
                            traced_chain_count += 1;
                            info!("TRACE [chains]: Built chain #{} containing traced n-gram with {} elements: {}..{} → {}..{}", 
                                traced_chain_count,
                                chain.len(),
                                matches[*chain.first().unwrap()].source_position,
                                matches[*chain.last().unwrap()].source_position,
                                matches[*chain.first().unwrap()].target_position,
                                matches[*chain.last().unwrap()].target_position);
                        }
                               
                        all_chains.push(chain);
                    } else if chain_has_traced {
                        // Log if a chain with traced n-gram is being discarded due to length
                        info!("TRACE [chains]: DISCARDED chain containing traced n-gram with only {} elements",
                            chain.len());
                    }
                }
            }
        }
        
        // Summary for traced n-gram chains
        if traced_chain_count > 0 && log_enabled!(Level::Info) {
            info!("TRACE [chains]: Built {} chains containing traced n-gram (out of {} total chains)",
                traced_chain_count, all_chains.len());
        } else if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
            info!("TRACE [chains]: NO CHAINS containing traced n-gram were built!");
        }
        
        all_chains
    }
    
    pub(super) fn find_dense_regions(&self, matches: &[NGramMatch]) -> Result<Vec<MatchCandidate>> {
        info!("\nStarting dense region detection with {} matches", matches.len());
        
        // Log which matching method we're using
        if self.matcher_config.proximity_matching {
            trace!("Using proximity-based matching with distance: {}", self.matcher_config.proximity_distance);
        } else {
            trace!("Using sequential matching with max gap: {}", self.matcher_config.max_gap);
        }
        
        // Early optimization: If matches exceed threshold, process in chunks using parallel implementation
        if self.processor_config.parallel_dense_regions && 
           matches.len() >= self.processor_config.parallel_dense_threshold {
            info!("Large match set detected ({} matches), using parallel implementation", matches.len());
            
            // The parallel implementation needs to know which matching method to use
            // We'll pass 'self' which contains the configuration
            return super::parallel_chains::find_dense_regions_parallel(self, matches);
        }
        
        // For smaller match sets, use the standard implementation
        info!("Using standard processing for {} matches", matches.len());
        self.find_dense_regions_chunk(matches, None)
    }

    pub(super) fn find_dense_regions_chunk(
        &self, 
        matches: &[NGramMatch],
        stats: Option<&Arc<ParallelProcessingStats>>
    ) -> Result<Vec<MatchCandidate>> {
        let start_time = Instant::now();
        
        // Check if matches contain our traced n-gram
        let traced_matches: Vec<(usize, &NGramMatch)> = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
            matches.iter()
                .enumerate()
                .filter(|(_, m)| self.is_traced_ngram(&m.ngram.text_content.cleaned_text))
                .collect()
        } else {
            Vec::new()
        };
                                
        if !traced_matches.is_empty() {
            info!("TRACE [dense_regions]: Processing {} matches containing traced n-gram", traced_matches.len());
        } else if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Trace) {
            trace!("TRACE [dense_regions]: No matches containing traced n-gram found in this chunk");
        }
        
        // Get effective max gap based on ngram type
        let base_max_gap = self.matcher_config.max_gap;
        let effective_max_gap = match self.ngram_type {
            NGramType::Character => base_max_gap,
            NGramType::Word => base_max_gap,
        };
        
        debug!("Using effective max gap: {} for ngram type: {:?}", effective_max_gap, self.ngram_type);
        
        // Choose between sequential and proximity-based methods
        let clusters = if self.matcher_config.proximity_matching {
            // Use proximity-based clustering
            let proximity_distance = self.matcher_config.proximity_distance;
            trace!("Using proximity-based clustering with distance: {}", proximity_distance);
            self.group_into_proximity_clusters(matches, proximity_distance, &self.matcher_config.traced_phrase)
        } else {
            // Use sequential chain building
            trace!("Using sequential chain building with max gap: {}", effective_max_gap);
            self.group_into_chains_optimized(matches, effective_max_gap, &self.matcher_config.traced_phrase)
        };
        
        // Check for clusters with our traced n-gram
        let traced_clusters: Vec<&Vec<usize>> = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
            clusters.iter()
                .filter(|cluster| cluster.iter().any(|&idx| self.is_traced_ngram(&matches[idx].ngram.text_content.cleaned_text)))
                .collect()
        } else {
            Vec::new()
        };
                                
        if !traced_clusters.is_empty() {
            info!("TRACE [dense_regions]: Found {} clusters containing traced n-gram", traced_clusters.len());
            
            // Show more details about the first few traced clusters
            for (i, cluster) in traced_clusters.iter().enumerate().take(3) {
                info!("TRACE [dense_regions]: Traced cluster #{} has {} elements", 
                    i+1, cluster.len());
                    
                // Get positions of first and last elements (may not be sequential for proximity clusters)
                if !cluster.is_empty() {
                    if self.matcher_config.proximity_matching {
                        // Calculate these expensive operations only when info logging is enabled
                        // This block is already inside an if statement that only executes when
                        // traced_clusters is non-empty, which already has proper logging guards
                        let source_min = cluster.iter().map(|&idx| matches[idx].source_position).min().unwrap_or(0);
                        let source_max = cluster.iter().map(|&idx| matches[idx].source_position).max().unwrap_or(0);
                        let target_min = cluster.iter().map(|&idx| matches[idx].target_position).min().unwrap_or(0);
                        let target_max = cluster.iter().map(|&idx| matches[idx].target_position).max().unwrap_or(0);
                        
                        info!("TRACE [dense_regions]: Source positions: {}..{}, Target positions: {}..{}", 
                            source_min, source_max, target_min, target_max);
                    } else {
                        // For sequential chains, first and last element positions
                        let first_idx = cluster[0];
                        let last_idx = cluster[cluster.len() - 1];
                        
                        info!("TRACE [dense_regions]: Traced chain #{} has {} elements: {}..{} → {}..{}", 
                            i+1, cluster.len(), 
                            matches[first_idx].source_position,
                            matches[last_idx].source_position,
                            matches[first_idx].target_position,
                            matches[last_idx].target_position);
                    }
                }
            }
        } else if !self.matcher_config.traced_phrase.is_empty() && !traced_matches.is_empty() && log_enabled!(Level::Trace) {
            trace!("TRACE [dense_regions]: NO CLUSTERS containing traced n-gram were found despite having matching n-grams!");
        }
               
        let mut candidates = Vec::new();
        
        // Track local statistics for sequential mode
        let mut local_clusters_processed = 0;
        let local_clusters_rejected = 0;
        let mut clusters_rejected_by_filter = 0;
        
        // Track traced candidates for logging
        let mut traced_candidates = 0;
        
        // Process clusters into candidates with detailed debugging
        for (i, cluster) in clusters.iter().enumerate() {
            // Update clusters processed counter
            local_clusters_processed += 1;
            if let Some(stats) = stats {
                stats.chains_processed.fetch_add(1, Ordering::Relaxed);
            }
            
            // Check if this cluster contains our traced n-gram
            let cluster_has_traced = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
                cluster.iter()
                    .any(|&idx| self.is_traced_ngram(&matches[idx].ngram.text_content.cleaned_text))
            } else {
                false
            };
            
            if cluster_has_traced {
                info!("TRACE [dense_regions]: Processing cluster #{} containing traced n-gram with {} elements", 
                    i, cluster.len());
                    
                // Print the individual ngrams in the cluster
                info!("TRACE [dense_regions]: Individual ngrams in cluster:");
                for (j, &ngram_idx) in cluster.iter().enumerate() {
                    let ngram_match = &matches[ngram_idx];
                    info!("TRACE [dense_regions]: Ngram #{}: '{}' (source={}, target={})", 
                        j+1, 
                        ngram_match.ngram.text_content.cleaned_text,
                        ngram_match.source_position,
                        ngram_match.target_position);
                }
            }
            
            // Reconstruct text from the cluster indices
            let cluster_refs: Vec<&NGramMatch> = cluster.iter()
                .map(|&idx| &matches[idx])
                .collect();
                
            let reconstructed_text = self.reconstruct_text_optimized(&cluster_refs);
            
            // Print the complete reconstructed text for traced clusters
            if cluster_has_traced {
                info!("TRACE [dense_regions]: FULL cluster text: '{}'", reconstructed_text);
            }
            
            // First check if the cluster is banal, using the appropriate detection method
            let (is_banal, auto_density) = if self.banality_detection_mode == "automatic" && self.common_ngrams.is_some() {
                // For automatic detection, we need to convert indices to NGramMatch objects
                let cluster_matches: Vec<NGramMatch> = cluster.iter()
                    .map(|&idx| matches[idx].clone())
                    .collect();
                
                // Use automatic detection with the n-gram IDs from the cluster
                self.check_automatic_banality(&cluster_matches)
            } else {
                // Use traditional phrase-based detection
                self.is_early_banal(&reconstructed_text)
            };
        
            if is_banal {
                clusters_rejected_by_filter += 1;
                
                // Only log at debug level if enabled
                if log_enabled!(Level::Debug) {
                    debug!("Cluster #{} rejected as banal", i);
                }
                
                // We've already optimized cluster_has_traced to only be true
                // when traced_phrase is non-empty and info logging is enabled
                if cluster_has_traced {
                    if self.banality_detection_mode == "automatic" {
                        info!("TRACE [dense_regions]: Traced cluster #{} REJECTED as banal by automatic detection (density: {:.2}%)", 
                            i, auto_density);
                    } else {
                        info!("TRACE [dense_regions]: Traced cluster #{} REJECTED as banal by phrase detection", i);
                    }
                }
                
                continue;
            }
        
            // Then apply other early filters (like isnad detection)
            if !self.passes_early_filters(&reconstructed_text, cluster.len()) {
                clusters_rejected_by_filter += 1;
                
                // Only log at debug level if enabled
                if log_enabled!(Level::Debug) {
                    debug!("Cluster #{} rejected by other early filters", i);
                }
                
                // We've already optimized cluster_has_traced to only be true
                // when traced_phrase is non-empty and info logging is enabled
                if cluster_has_traced {
                    info!("TRACE [dense_regions]: Traced cluster #{} REJECTED by other early filters", i);
                }
                
                continue;
            }
        
            // Create a candidate using the appropriate method based on matching type
            let candidate_result = if self.matcher_config.proximity_matching {
                self.create_candidate_from_cluster(cluster, matches)
            } else {
                self.create_candidate_optimized(cluster, matches)
            };
            
            match candidate_result {
                Ok(candidate) => {
                    // Only log at debug level if enabled
                    if log_enabled!(Level::Debug) {
                        debug!("Created candidate for cluster #{}: source={}..{}, target={}..{}",
                              i, candidate.source_range.start, candidate.source_range.end,
                              candidate.target_range.start, candidate.target_range.end);
                    }
                    
                    // We've already optimized cluster_has_traced to only be true
                    // when traced_phrase is non-empty and info logging is enabled
                    if cluster_has_traced {
                        info!("TRACE [dense_regions]: Created candidate from traced cluster: source={}..{}, target={}..{}",
                              candidate.source_range.start, candidate.source_range.end,
                              candidate.target_range.start, candidate.target_range.end);
                    }
                    
                    // Check if it's a dense region using the appropriate method
                    let is_dense_result = if self.matcher_config.proximity_matching {
                        self.is_dense_cluster(&candidate)
                    } else {
                        self.is_dense_region_optimized(&candidate)
                    };
                    
                    match is_dense_result {
                        Ok(true) => {
                            debug!("Cluster #{} is dense enough to be a candidate", i);
                            
                            if cluster_has_traced {
                                traced_candidates += 1;
                                info!("TRACE [dense_regions]: Traced cluster #{} PASSED density check", i);
                                
                                // Show text for validation
                                info!("TRACE [dense_regions]: Traced candidate text: '{}'", reconstructed_text);
                            }
                            
                            // Add to candidates
                            candidates.push(candidate);
                        },
                        Ok(false) => {
                            debug!("Cluster #{} is NOT dense enough to be a candidate", i);
                            
                            if cluster_has_traced {
                                info!("TRACE [dense_regions]: Traced cluster #{} FAILED density check", i);
                                
                                // Get additional debug info for failing clusters
                                let source_range = candidate.source_range.end - candidate.source_range.start;
                                let target_range = candidate.target_range.end - candidate.target_range.start;
                                info!("TRACE [dense_regions]: Density details - matches: {}, source range: {}, target range: {}", 
                                    candidate.matches.len(), source_range, target_range);
                            }
                        },
                        Err(e) => {
                            debug!("Error checking density for cluster #{}: {}", i, e);
                            
                            if cluster_has_traced {
                                info!("TRACE [dense_regions]: Error checking density for traced cluster #{}: {}", i, e);
                            }
                        }
                    }
                },
                Err(e) => {
                    debug!("Failed to create candidate for cluster #{}: {}", i, e);
                    
                    if cluster_has_traced {
                        info!("TRACE [dense_regions]: Failed to create candidate for traced cluster #{}: {}", i, e);
                    }
                }
            }
        }
        
        // Log local statistics if not in parallel mode and tracing is enabled
        if stats.is_none() && log_enabled!(Level::Trace) {
            trace!("Sequential processing stats:");
            trace!("  Total clusters processed: {}", local_clusters_processed);
            
            // Calculate percentages only when tracing is actually enabled
            let filter_pct = if local_clusters_processed > 0 {
                (clusters_rejected_by_filter as f64 / local_clusters_processed as f64) * 100.0
            } else { 
                0.0 
            };
            trace!("  Clusters rejected by filters: {} ({:.1}%)", 
                clusters_rejected_by_filter, filter_pct);
            
            let banal_pct = if local_clusters_processed > 0 {
                (local_clusters_rejected as f64 / local_clusters_processed as f64) * 100.0
            } else { 
                0.0 
            };
            trace!("  Clusters rejected as banal/isnad: {} ({:.1}%)", 
                local_clusters_rejected, banal_pct);
                
            trace!("  Final candidates: {}", candidates.len());
        }

        
        if traced_candidates > 0 && log_enabled!(Level::Info) {
            info!("TRACE [dense_regions]: Created {} candidates containing traced n-gram", traced_candidates);
        } else if !traced_clusters.is_empty() && log_enabled!(Level::Info) {
            info!("TRACE [dense_regions]: NO CANDIDATES created from clusters containing traced n-gram!");
        }
        
        // Only log timing info if debug logging is enabled
        if log_enabled!(Level::Debug) {
            debug!("Found {} dense regions in chunk (took {:?})", candidates.len(), start_time.elapsed());
        }
        
        Ok(candidates)
    }

    pub(super) fn create_candidate_optimized(&self, chain: &[usize], matches: &[NGramMatch]) -> Result<MatchCandidate> {
        if chain.is_empty() {
            return Err(Error::text("Cannot create candidate from empty chain"));
        }
        
        // Get first and last matches by index
        let first = &matches[chain[0]];
        let last = &matches[chain[chain.len() - 1]];
        
        // Create source and target ranges directly
        let source_range = first.source_position..last.source_position + 1;
        let target_range = first.target_position..last.target_position + 1;
        
        // Basic validation
        let source_size = source_range.end - source_range.start;
        let target_size = target_range.end - target_range.start;
        
        if source_size == 0 || target_size == 0 {
            return Err(Error::text("Zero-length range detected"));
        }
        
        // Quick range ratio validation
        let range_ratio = source_size as f64 / target_size as f64;
        if range_ratio < 0.32 || range_ratio > 3.0 {
            return Err(Error::text(format!(
                "Range size mismatch: source={}, target={}, ratio={:.2}",
                source_size, target_size, range_ratio
            )));
        }
        
        // Create the list of match objects from indices
        let match_objects: Vec<NGramMatch> = chain.iter()
            .map(|&idx| matches[idx].clone())
            .collect();
        
        Ok(MatchCandidate {
            source_range,
            target_range,
            matches: match_objects
        })
    }

    pub(super) fn is_dense_region_optimized(&self, candidate: &MatchCandidate) -> Result<bool> {
        // Check if candidate contains our traced n-gram
        let has_traced = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
            candidate.matches.iter()
                .any(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text))
        } else {
            false
        };
            
        if has_traced {
            info!("TRACE [density]: Checking density for candidate with traced n-gram");
            info!("TRACE [density]: Candidate has {} matches over source range {}..{} and target range {}..{}", 
                candidate.matches.len(),
                candidate.source_range.start, candidate.source_range.end,
                candidate.target_range.start, candidate.target_range.end);
        }
        
        // Quick reject for very small candidates
        if candidate.matches.len() < 2 {
            if has_traced {
                info!("TRACE [density]: REJECTED - Candidate has too few matches: {} (min: 3)", 
                    candidate.matches.len());
            }
            return Ok(false);
        }
        
        // Calculate range sizes
        let source_range = candidate.source_range.end - candidate.source_range.start;
        let target_range = candidate.target_range.end - candidate.target_range.start;
        
        // For small ranges, use a simpler approach
        if source_range < 100 && target_range < 100 {
            // For small ranges, just use match count / range size
            let density = candidate.matches.len() as f64 / source_range.max(target_range) as f64;
            
            // Adjust threshold based on ngram type
            let effective_threshold = match self.ngram_type {
                NGramType::Character => self.matcher_config.min_density,
                NGramType::Word => (self.matcher_config.min_density * 0.7).max(0.1)
            };
            
            let is_dense = density >= effective_threshold;
            
            if has_traced {
                info!("TRACE [density]: Small range calculation - density: {:.3}, threshold: {:.3}, result: {}", 
                    density, effective_threshold, if is_dense { "PASS" } else { "FAIL" });
            }
            
            return Ok(is_dense);
        }
        
        // For larger ranges, use bit vectors for more accurate density calculation
        let mut source_bits = BitVec::from_elem(source_range, false);
        let mut target_bits = BitVec::from_elem(target_range, false);
        
        // Mark positions covered by matches
        for m in &candidate.matches {
            let src_pos = m.source_position - candidate.source_range.start;
            let tgt_pos = m.target_position - candidate.target_range.start;
            
            if src_pos < source_range {
                source_bits.set(src_pos, true);
            }
            if tgt_pos < target_range {
                target_bits.set(tgt_pos, true);
            }
        }
        
        // Calculate density as the fraction of positions covered
        let source_filled = source_bits.iter().filter(|&b| b).count();
        let target_filled = target_bits.iter().filter(|&b| b).count();
        
        let source_density = source_filled as f64 / source_range as f64;
        let target_density = target_filled as f64 / target_range as f64;
        
        // Use the lower density to be conservative
        let density = source_density.min(target_density);
        
        // Adjust threshold based on ngram type
        let effective_threshold = match self.ngram_type {
            NGramType::Character => self.matcher_config.min_density,
            NGramType::Word => (self.matcher_config.min_density * 0.7).max(0.1)
        };
        
        let is_dense = density >= effective_threshold;
        
        if has_traced {
            info!("TRACE [density]: Large range calculation:");
            info!("TRACE [density]: Source density: {:.3} ({} of {} positions filled)", 
                source_density, source_filled, source_range);
            info!("TRACE [density]: Target density: {:.3} ({} of {} positions filled)", 
                target_density, target_filled, target_range);
            info!("TRACE [density]: Final density: {:.3}, threshold: {:.3}, result: {}", 
                density, effective_threshold, if is_dense { "PASS" } else { "FAIL" });
        }
        
        Ok(is_dense)
    }
    
    pub(super) fn group_into_proximity_clusters(
        &self,
        matches: &[NGramMatch], 
        max_distance: usize,
        traced_phrase: &str
    ) -> Vec<Vec<usize>> {  // Changed return type
        if matches.is_empty() {
            return Vec::new();
        }
        
        // Helper function to check for traced ngram
        let contains_traced = |text: &str| !traced_phrase.is_empty() && text.contains(traced_phrase);
        
        let traced_matches: Vec<(usize, &NGramMatch)> = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
            matches.iter()
                .enumerate()  // Keep track of original indices
                .filter(|(_, m)| contains_traced(&m.ngram.text_content.cleaned_text))
                .collect()
        } else {
            Vec::new()
        };
                                
        if !traced_matches.is_empty() {
            info!("TRACE [proximity]: Found {} matches containing traced n-gram before clustering", 
                traced_matches.len());
                                    
            for (i, (match_idx, m)) in traced_matches.iter().enumerate().take(5) {
                // Use match_idx instead of just idx to fix the unused variable warning
                info!("TRACE [proximity]: Traced match #{}: source={}, target={}, text='{}' (index: {})", 
                    i+1, m.source_position, m.target_position, m.ngram.text_content.cleaned_text, match_idx);
            }
        } else if !traced_phrase.is_empty() && log_enabled!(Level::Trace) {
            trace!("TRACE [proximity]: NO MATCHES containing traced n-gram found before clustering!");
        }
        
        // Create a vector to hold all clusters (of indices now)
        let mut all_clusters = Vec::new();
        
        // Track which matches have been processed by index
        let mut processed = AHashSet::new();
        
        // Track clusters with our traced n-gram
        let mut traced_cluster_count = 0;
        
        // Process each match as a potential start of a new cluster
        for (i, start_match) in matches.iter().enumerate() {
            // Skip if this match has already been assigned to a cluster
            if processed.contains(&i) {
                continue;
            }
            
            // Start a new cluster with this match index
            let mut cluster = vec![i];
            processed.insert(i);
            
            // Check if starting with our traced n-gram
            let cluster_starts_with_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                contains_traced(&start_match.ngram.text_content.cleaned_text)
            } else {
                false
            };
            
            if cluster_starts_with_traced {
                info!("TRACE [proximity]: Starting new cluster with traced n-gram at source={}, target={}",
                    start_match.source_position, start_match.target_position);
            }
            
            // Keep expanding the cluster until no more matches can be added
            let mut added_more = true;
            
            while added_more {
                added_more = false;
                
                // Look for unprocessed matches that are close to any match in our current cluster
                for (j, candidate) in matches.iter().enumerate() {
                    // Skip if already processed
                    if processed.contains(&j) {
                        continue;
                    }
                    
                    // Check distance to every match in the current cluster
                    let mut is_close = false;
                    for &cluster_idx in &cluster {
                        let cluster_match = &matches[cluster_idx];
                        
                        // Calculate distance in both source and target dimensions
                        let source_distance = if cluster_match.source_position > candidate.source_position {
                            cluster_match.source_position - candidate.source_position
                        } else {
                            candidate.source_position - cluster_match.source_position
                        };
                        
                        let target_distance = if cluster_match.target_position > candidate.target_position {
                            cluster_match.target_position - candidate.target_position
                        } else {
                            candidate.target_position - cluster_match.target_position
                        };
                        
                        // If both distances are within max_distance, this match is close enough
                        if source_distance <= max_distance && target_distance <= max_distance {
                            is_close = true;
                            break;  // No need to check against other cluster matches
                        }
                    }
                    
                    // If this candidate is close to any match in the cluster, add it
                    if is_close {
                        // Check if this match contains traced n-gram - but only if needed
                        let has_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                            contains_traced(&candidate.ngram.text_content.cleaned_text)
                        } else {
                            false
                        };
                        
                        if has_traced && cluster_starts_with_traced {
                            info!("TRACE [proximity]: Adding traced n-gram to existing traced cluster at position {}", 
                                cluster.len());
                        } else if has_traced {
                            info!("TRACE [proximity]: Adding traced n-gram to cluster at position {}", 
                                cluster.len());
                        }
                        
                        cluster.push(j);  // Store the index, not the cloned object
                        processed.insert(j);
                        added_more = true;  // We added a match, so we need to do another pass
                    }
                }
            }
            
            let cluster_has_traced = if !traced_phrase.is_empty() && log_enabled!(Level::Info) {
                cluster.iter()
                    .any(|&idx| contains_traced(&matches[idx].ngram.text_content.cleaned_text))
            } else {
                false
            };
                    
            // Only keep clusters with at least 2 matches
            if cluster.len() >= 2 {
                if cluster_has_traced {
                    traced_cluster_count += 1;
                    info!("TRACE [proximity]: Built cluster #{} containing traced n-gram with {} elements", 
                        traced_cluster_count, cluster.len());
                        
                    // Show some specific matches in this cluster
                    for (pos, &match_idx) in cluster.iter().enumerate().take(3) {
                        let m = &matches[match_idx];
                        info!("TRACE [proximity]: Match #{} in traced cluster: source={}, target={}", 
                            pos+1, m.source_position, m.target_position);
                    }
                }
                
                all_clusters.push(cluster);
            } else if cluster_has_traced {
                // Log if a cluster with traced n-gram is being discarded due to length
                info!("TRACE [proximity]: DISCARDED cluster containing traced n-gram with only {} elements",
                    cluster.len());
            }
        }
        
        // Summary for traced n-gram clusters
        if traced_cluster_count > 0 && log_enabled!(Level::Info) {
            info!("TRACE [proximity]: Built {} clusters containing traced n-gram (out of {} total clusters)",
                traced_cluster_count, all_clusters.len());
        } else if !traced_phrase.is_empty() && log_enabled!(Level::Trace) {
            trace!("TRACE [proximity]: NO CLUSTERS containing traced n-gram were built!");
        }
        
        all_clusters
    }

    pub(super) fn create_candidate_from_cluster(&self, cluster: &[usize], matches: &[NGramMatch]) -> Result<MatchCandidate> {
        if cluster.is_empty() {
            return Err(Error::text("Cannot create candidate from empty cluster"));
        }
        
        // For proximity-based clusters, find min/max positions in source and target
        // This handles matches that might be in any order
        let source_min = cluster.iter()
            .map(|&idx| matches[idx].source_position)
            .min()
            .unwrap_or(0);
        
        let source_max = cluster.iter()
            .map(|&idx| matches[idx].source_position)
            .max()
            .unwrap_or(0);
        
        let target_min = cluster.iter()
            .map(|&idx| matches[idx].target_position)
            .min()
            .unwrap_or(0);
        
        let target_max = cluster.iter()
            .map(|&idx| matches[idx].target_position)
            .max()
            .unwrap_or(0);
        
        // Create ranges that encompass the entire cluster
        let source_range = source_min..source_max + 1;
        let target_range = target_min..target_max + 1;
        
        // Basic validation
        let source_size = source_range.end - source_range.start;
        let target_size = target_range.end - target_range.start;
        
        if source_size == 0 || target_size == 0 {
            return Err(Error::text("Zero-length range detected"));
        }
        
        // Quick range ratio validation
        let range_ratio = source_size as f64 / target_size as f64;
        if range_ratio < 0.32 || range_ratio > 3.0 {
            return Err(Error::text(format!(
                "Range size mismatch: source={}, target={}, ratio={:.2}",
                source_size, target_size, range_ratio
            )));
        }
        
        // Check if this candidate contains a traced n-gram
        let has_traced = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
            cluster.iter()
                .any(|&idx| self.is_traced_ngram(&matches[idx].ngram.text_content.cleaned_text))
        } else {
            false
        };
            
        if has_traced {
            info!("TRACE [proximity]: Created candidate from traced cluster: source={}..{}, target={}..{}",
                source_min, source_max, target_min, target_max);
            info!("TRACE [proximity]: Cluster contains {} matches over area size {} x {}",
                cluster.len(), source_size, target_size);
        }
        
        // Create the matches list for the candidate by converting indices to objects
        let match_objects: Vec<NGramMatch> = cluster.iter()
            .map(|&idx| matches[idx].clone())
            .collect();
        
        Ok(MatchCandidate {
            source_range,
            target_range,
            matches: match_objects
        })
    }
    pub(super) fn is_dense_cluster(&self, candidate: &MatchCandidate) -> Result<bool> {
        // Check if candidate contains our traced n-gram
        let has_traced = if !self.matcher_config.traced_phrase.is_empty() && log_enabled!(Level::Info) {
            candidate.matches.iter()
                .any(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text))
        } else {
            false
        };
                        
        if has_traced {
            info!("TRACE [density]: Checking proximity-based density for candidate with traced n-gram");
            info!("TRACE [density]: Candidate has {} matches over source range {}..{} and target range {}..{}", 
                candidate.matches.len(),
                candidate.source_range.start, candidate.source_range.end,
                candidate.target_range.start, candidate.target_range.end);
        }
        
        // Quick reject for very small candidates
        if candidate.matches.len() < 2 {
            if has_traced {
                info!("TRACE [density]: REJECTED - Candidate has too few matches: {} (min: 2)", 
                    candidate.matches.len());
            }
            return Ok(false);
        }
        
        // Calculate range sizes
        let source_range = candidate.source_range.end - candidate.source_range.start;
        let target_range = candidate.target_range.end - candidate.target_range.start;
        
        // For proximity-based matching, we use a different density calculation
        // that accounts for the 2D nature of proximity clusters
        
        // Calculate the "area" of the match region
        let area = source_range * target_range;
        
        // Avoid division by zero
        if area == 0 {
            if has_traced {
                info!("TRACE [density]: REJECTED - Zero area (degenerate cluster)");
            }
            return Ok(false);
        }
        
        // Calculate density as matches per square root of area
        // This gives a better measure for 2D clusters than linear density
        // For a cluster with equal dimensions, this is equivalent to matches / side_length
        let match_count = candidate.matches.len();
        let density = match_count as f64 / (area as f64).sqrt();
        
        // Get density threshold - apply proximity-specific factor from configuration
        let base_threshold = self.matcher_config.min_density;
        let adjusted_threshold = base_threshold * self.matcher_config.proximity_density_factor;
        
        // Adjust threshold based on ngram type
        let effective_threshold = match self.ngram_type {
            NGramType::Character => adjusted_threshold,
            NGramType::Word => (adjusted_threshold * 0.7).max(0.1)  // Word n-grams get a reduced threshold
        };
        
        // Check if dense enough
        let is_dense = density >= effective_threshold;
        
        if has_traced {
            info!("TRACE [density]: Proximity calculation - match count: {}, area: {} ({}×{})", 
                match_count, area, source_range, target_range);
            info!("TRACE [density]: Density: {:.3} matches per √area, threshold: {:.3}, result: {}", 
                density, effective_threshold, if is_dense { "PASS" } else { "FAIL" });
                
            // For failed traces, provide more context
            if !is_dense {
                // Calculate what match count would be needed to pass
                let needed_matches = (effective_threshold * (area as f64).sqrt()).ceil() as usize;
                info!("TRACE [density]: Would need {} matches to pass density check (has {})", 
                    needed_matches, match_count);
            }
        }
        
        Ok(is_dense)
    }
}