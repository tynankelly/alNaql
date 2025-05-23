// Enhanced ParallelMatcher in parallel.rs

use rayon::prelude::*;
use log::{info, trace, warn};
use std::time::Instant;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::collections::{HashMap, HashSet};
use crate::error::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::storage::NGramMetadata;
use crate::types::NGramEntry;
use crate::config::subsystems::matcher::MatcherConfig;
use crate::config::subsystems::processor::ProcessorConfig;
use crate::matcher::types::{NGramMatch, TextSpan};
use crate::matcher::algorithms::{
    SimilarityAlgorithm, ExactMatcher, JaccardMatcher, 
    CosineMatcher, LevenshteinMatcher, VSAMatcher
};
use crate::config::subsystems::matcher::SimilarityMetric;
use crate::matcher::{ProcessingPipeline, StorageAccessStrategy};
use crate::storage::StorageBackend;

// Constants for performance tuning
const MIN_CHUNK_SIZE: usize = 2000;
const MAX_CHUNK_SIZE: usize = 20000;
const SIMILARITY_THRESHOLD: f64 = 0.0005;
const SOURCE_MIN_BATCH_SIZE: usize = 100;
const SOURCE_MAX_BATCH_SIZE: usize = 500;

pub struct ParallelMatcher<'a> {
    matcher_config: MatcherConfig,
    processor_config: ProcessorConfig,
    storage: Option<&'a dyn StorageBackend>,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

impl<'a> ParallelMatcher<'a> {
    pub fn new(
        matcher_config: MatcherConfig, 
        processor_config: ProcessorConfig,
        storage: Option<&'a dyn StorageBackend>
    ) -> Result<Self> {
        // Initialize thread pool if thread count specified
        let thread_pool = if processor_config.parallel_thread_count > 0 {
            Some(rayon::ThreadPoolBuilder::new()
                .num_threads(Self::get_thread_count(&processor_config))
                .stack_size(32 * 1024 * 1024)  // 32MB stack
                .build()?)
        } else {
            None // Use rayon's default thread pool
        };
    
        Ok(Self {
            matcher_config,
            processor_config,
            storage,
            thread_pool: thread_pool.map(Arc::new),
        })
    }

    pub fn find_matches_parallel(
        &self,
        source_ngrams: &[NGramEntry],
        storage: &'a dyn StorageBackend
    ) -> Result<Vec<NGramMatch>> {
        let start_time = Instant::now();
        info!("Starting parallel match finding for {} source ngrams", source_ngrams.len());
        
        // Calculate batch size using the existing function (now enhanced)
        let batch_size = self.calculate_source_batch_size(source_ngrams.len());
        info!("Using source batch size of {} ngrams", batch_size);
        
        // Track progress more efficiently with an atomic counter
        let progress_counter = AtomicUsize::new(0);
        let total_ngrams = source_ngrams.len();
        
        // Divide source ngrams into batches
        let source_len = source_ngrams.len();
        let num_batches = (source_len + batch_size - 1) / batch_size;
        
        // Create a collection for results that avoids mutex contention
        // Instead of using a shared Mutex<Vec>, each thread will return its results
        let results: Vec<Vec<NGramMatch>> = (0..num_batches)
            .into_par_iter()
            .map(|batch_idx| -> Result<Vec<NGramMatch>> {
                // Calculate start and end indices for this batch
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(source_len);
                let batch = &source_ngrams[start_idx..end_idx];
                
                // Process each source ngram in this batch 
                let mut batch_matches = Vec::new();
                for source_ngram in batch {
                    // Check if this is a traced ngram
                    let is_traced = !self.matcher_config.traced_phrase.is_empty() && 
                                    source_ngram.text_content.cleaned_text.contains(&self.matcher_config.traced_phrase);
                    
                    // Try optimized path with metadata filtering
                    if let Ok(sequences) = storage.get_sequences_by_prefix_for_ngram(source_ngram) {
                        if !sequences.is_empty() {
                            if is_traced {
                                trace!("TRACE: Found {} prefix candidates for traced ngram", sequences.len());
                            }
                            
                            // Get metadata for filtering
                            if let Ok(metadata_batch) = storage.get_ngram_metadata_batch(&sequences) {
                                // Apply metadata filtering using our new function
                                let filtered_sequences = self.filter_candidates_by_metadata(
                                    source_ngram,
                                    &metadata_batch,
                                    self.matcher_config.ngram_similarity_metric
                                );
                                
                                if is_traced {
                                    trace!("TRACE: Filtered {} → {} candidates using metadata for traced ngram", 
                                          metadata_batch.len(), filtered_sequences.len());
                                }
                                
                                // Fetch full ngrams only for filtered candidates
                                if !filtered_sequences.is_empty() {
                                    if let Ok(filtered_ngrams) = storage.get_ngrams_batch(&filtered_sequences) {
                                        // Get the appropriate algorithm based on the similarity metric
                                        let algorithm = match self.matcher_config.ngram_similarity_metric {
                                            SimilarityMetric::Exact => Box::new(ExactMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                            SimilarityMetric::Jaccard => Box::new(JaccardMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                            SimilarityMetric::Cosine => Box::new(CosineMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                            SimilarityMetric::Levenshtein => Box::new(LevenshteinMatcher::with_config(
                                                self.matcher_config.algorithm_config.levenshtein_max_distance
                                            )) as Box<dyn SimilarityAlgorithm>,
                                            SimilarityMetric::VSA => Box::new(VSAMatcher::with_config(
                                                self.matcher_config.algorithm_config.vsa_min_term_weight
                                            )) as Box<dyn SimilarityAlgorithm>,
                                        };
                                        
                                        // Compute matches using the appropriate algorithm
                                        let source_matches: Vec<NGramMatch> = filtered_ngrams.iter()
                                            .filter_map(|target| {
                                                match algorithm.compare_ngrams(source_ngram, target) {
                                                    Ok(similarity) => {
                                                        if similarity >= self.matcher_config.ngram_min_confidence {
                                                            if is_traced {
                                                                trace!("TRACE: Found match for traced ngram with similarity {:.3}", 
                                                                      similarity);
                                                            }
                                                            
                                                            Some(NGramMatch {
                                                                source_position: source_ngram.sequence_number as usize,
                                                                target_position: target.sequence_number as usize,
                                                                ngram: source_ngram.clone(),
                                                                similarity,
                                                            })
                                                        } else {
                                                            None
                                                        }
                                                    },
                                                    Err(e) => {
                                                        warn!("Error calculating similarity: {}", e);
                                                        None
                                                    }
                                                }
                                            })
                                            .collect();
                                        
                                        // Add matches to batch results
                                        if !source_matches.is_empty() {
                                            let matches_count = source_matches.len();
                                            batch_matches.extend(source_matches);
                                            
                                            if is_traced {
                                                trace!("TRACE: Added {} matches for traced ngram", matches_count);
                                            }
                                        }
                                        
                                        // Skip the fallback path
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Fallback to existing method if optimized path fails
                    match self.find_matches_fallback(source_ngram, &[]) {
                        Ok(matches) => {
                            if !matches.is_empty() {
                                batch_matches.extend(matches);
                            }
                        },
                        Err(e) => {
                            warn!("Error finding matches for ngram {}: {}", 
                                  source_ngram.sequence_number, e);
                        }
                    }
                }
                
                // Update progress counter atomically and log periodically
                let processed = progress_counter.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                if processed % 1000 == 0 || processed == total_ngrams {
                    info!("Processed {}/{} source ngrams ({:.1}%)", 
                         processed, total_ngrams, (processed as f64 / total_ngrams as f64) * 100.0);
                }
                
                Ok(batch_matches)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Combine all results efficiently - no mutex needed
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
                a.source_position.cmp(&b.source_position)
                    .then(a.target_position.cmp(&b.target_position))
            });
            
            // Remove adjacent duplicates
            all_matches.dedup_by(|a, b| {
                a.source_position == b.source_position && 
                a.target_position == b.target_position
            });
            
            let removed = before_count - all_matches.len();
            if removed > 0 {
                info!("Removed {} duplicate matches", removed);
            }
        }
        
        info!("Parallel source processing complete: found {} matches in {:?}", 
             all_matches.len(), start_time.elapsed());
        
        Ok(all_matches)
    }
    
    // In src/matcher/parallel.rs
    fn calculate_source_batch_size(&self, source_count: usize) -> usize {
        // Get system information
        let cpu_count = num_cpus::get();
        
        // NEW: Check memory pressure
        let memory_pressure = match sys_info::mem_info() {
            Ok(info) => 1.0 - (info.avail as f64 / info.total as f64),
            Err(_) => 0.5, // Assume 50% if unknown
        };
        
        // Base size adjusted for memory
        let base_size = if memory_pressure > 0.9 {
            // Critical memory - super small batches
            150
        } else if memory_pressure > 0.8 {
            // High memory - very small batches
            250
        } else {
            // Normal memory
            350
        };
        
        // Calculate adaptive batch size based on data characteristics and available CPUs
        let cpu_based_size = if source_count < 10_000 {
            // For smaller datasets, use reasonable batch size
            (source_count / (cpu_count * 2).max(1)).min(base_size)
        } else {
            // For larger datasets, use smaller batches
            (source_count / (cpu_count * 4).max(1)).min(base_size)
        };
        
        // Keep within smaller bounds
        cpu_based_size.clamp(SOURCE_MIN_BATCH_SIZE, SOURCE_MAX_BATCH_SIZE) // Much smaller range
    }

    pub fn find_matches_fallback(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        let start_time = Instant::now();
        trace!("Finding matches for source ngram: '{}' (sequence: {})", 
              source.text_content.cleaned_text, source.sequence_number);
    
        // For exact matching, try storage-based lookup first
        if let Some(storage) = self.storage {
            trace!("Looking up prefix matches for ngram: '{}' (source={})", 
                   source.text_content.cleaned_text, 
                   source.sequence_number);
            
            if let Ok(sequences) = storage.get_sequences_by_prefix_for_ngram(source) {
                let sequence_count = sequences.len();
                trace!("Found {} sequences with matching prefix", sequence_count);
                
                if !sequences.is_empty() {
                    trace!("First few sequence numbers: {:?}", 
                           sequences.iter().take(5).collect::<Vec<_>>());
                    
                    // Prepare for matching with filtering
                    let source_text = &source.text_content.cleaned_text;
                    let mut all_matches = Vec::new();
                    
                    // Performance tracking
                    let mut total_candidates = 0;
                    let mut filtered_candidates = 0;
                    let mut final_matches = 0;
                    
                    // Calculate optimal batch size based on sequence count
                    let batch_size = if sequences.len() > 100_000 {
                        5_000  // Smaller batches for huge result sets
                    } else if sequences.len() > 10_000 {
                        10_000 // Medium batches for moderate result sets
                    } else {
                        sequences.len() // Process all at once for small result sets
                    };
                    
                    // Process in chunks to maintain efficiency
                    for chunk in sequences.chunks(batch_size) {
                        total_candidates += chunk.len();
                        trace!("Processing chunk of {} prefix matches", chunk.len());
                        
                        // STEP 1: Get metadata first (this is much faster than getting full entries)
                        if let Ok(metadata_batch) = storage.get_ngram_metadata_batch(chunk) {
                            trace!("Retrieved metadata for {} candidate ngrams", metadata_batch.len());
                            
                            // MODIFIED: Metadata filtering for all similarity metrics
                            let filtered_sequences = self.filter_candidates_by_metadata(
                                source,
                                &metadata_batch,
                                self.matcher_config.ngram_similarity_metric
                            );
    
                            trace!("Metadata-based filtering: {} → {} candidates", 
                                metadata_batch.len(), filtered_sequences.len());
                            
                            filtered_candidates += filtered_sequences.len();
                            trace!("Pre-filtered to {} likely matches using metadata", 
                                  filtered_sequences.len());
                            
                            // STEP 3: Only fetch full ngrams for promising candidates
                            if !filtered_sequences.is_empty() {
                                if let Ok(ngrams) = storage.get_ngrams_batch(&filtered_sequences) {
                                    trace!("Retrieved {} full ngrams for final verification", ngrams.len());
                                    
                                    // Get the appropriate similarity algorithm based on configuration
                                    let algorithm = match self.matcher_config.ngram_similarity_metric {
                                        SimilarityMetric::Exact => Box::new(ExactMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                        SimilarityMetric::Jaccard => Box::new(JaccardMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                        SimilarityMetric::Cosine => Box::new(CosineMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                        SimilarityMetric::Levenshtein => Box::new(LevenshteinMatcher::with_config(
                                            self.matcher_config.algorithm_config.levenshtein_max_distance
                                        )) as Box<dyn SimilarityAlgorithm>,
                                        SimilarityMetric::VSA => Box::new(VSAMatcher::with_config(
                                            self.matcher_config.algorithm_config.vsa_min_term_weight
                                        )) as Box<dyn SimilarityAlgorithm>,
                                    };
    
                                    // Process each target ngram with the appropriate similarity algorithm
                                    let chunk_matches: Vec<NGramMatch> = ngrams.into_iter()
                                        .filter_map(|target| {
                                            // Calculate similarity using the configured algorithm
                                            match algorithm.compare_ngrams(source, &target) {
                                                Ok(similarity) => {
                                                    // Only keep matches that meet the confidence threshold
                                                    if similarity >= self.matcher_config.ngram_min_confidence {
                                                        trace!("Match found: {} → {} with similarity {:.3}", 
                                                            source.text_content.cleaned_text,
                                                            target.text_content.cleaned_text,
                                                            similarity);
                                                        
                                                        Some(NGramMatch {
                                                            source_position: source.sequence_number as usize,
                                                            target_position: target.sequence_number as usize,
                                                            ngram: source.clone(),
                                                            similarity,
                                                        })
                                                    } else {
                                                        // Log trace for diagnostics when a near-match is below threshold
                                                        if similarity > 0.0 {
                                                            trace!("Near match rejected: {} → {} with similarity {:.3} (threshold: {:.3})",
                                                                source.text_content.cleaned_text,
                                                                target.text_content.cleaned_text,
                                                                similarity,
                                                                self.matcher_config.ngram_min_confidence);
                                                        }
                                                        None
                                                    }
                                                },
                                                Err(e) => {
                                                    warn!("Error calculating similarity: {}", e);
                                                    None
                                                }
                                            }
                                        })
                                        .collect();
                                    
                                    final_matches += chunk_matches.len();
                                    if !chunk_matches.is_empty() {
                                        trace!("Found {} matches in this chunk", chunk_matches.len());
                                        all_matches.extend(chunk_matches);
                                    }
                                } else {
                                    trace!("Failed to retrieve full ngrams from storage");
                                }
                            }
                        } else {
                            // Fallback to direct lookup if metadata retrieval fails
                            trace!("Metadata retrieval failed, falling back to direct lookup");
                            if let Ok(ngrams) = storage.get_ngrams_batch(chunk) {
                                trace!("Retrieved {} candidate ngrams for comparison", ngrams.len());
                                
                                let chunk_matches: Vec<NGramMatch> = ngrams.into_iter()
                                    .filter_map(|target| {
                                        if target.text_content.cleaned_text == *source_text {
                                            Some(NGramMatch {
                                                source_position: source.sequence_number as usize,
                                                target_position: target.sequence_number as usize,
                                                ngram: source.clone(),
                                                similarity: 1.0,
                                            })
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                
                                final_matches += chunk_matches.len();
                                if !chunk_matches.is_empty() {
                                    trace!("Found {} matches in this chunk", chunk_matches.len());
                                    all_matches.extend(chunk_matches);
                                }
                            } else {
                                trace!("Failed to retrieve chunk of ngrams from storage");
                            }
                        }
                    }
                    
                    // Log performance metrics
                    if total_candidates > 0 {
                        let filtration_ratio = 100.0 * (1.0 - (filtered_candidates as f64 / total_candidates as f64));
                        let overall_ratio = 100.0 * (1.0 - (final_matches as f64 / total_candidates as f64));
                        
                        trace!("Prefix matching performance metrics:");
                        trace!("  Candidates: {} → {} after filtering → {} matches", 
                             total_candidates, filtered_candidates, final_matches);
                        trace!("  Reduction: {:.1}% at pre-filtering, {:.1}% overall", 
                             filtration_ratio, overall_ratio);
                        trace!("  Processing time: {:?}", start_time.elapsed());
                    }
                    
                    trace!("Parallel matching completed in {:?}, found {} matches from {} prefix candidates", 
                          start_time.elapsed(), all_matches.len(), sequence_count);
                    return Ok(all_matches);
                } else {
                    trace!("No matching prefix sequences found in target storage");
                }
                
                // Return empty results if prefix lookup was attempted but found no matches
                trace!("No matches found in prefix index - skipping full scan");
                return Ok(Vec::new());
            } else {
                trace!("Failed to retrieve sequences by prefix for ngram: '{}'", 
                       source.text_content.cleaned_text);
                
                // If prefix lookup failed due to an error, fallback to full scan
                trace!("Prefix lookup failed - falling back to full scan as last resort");
            }
        } else {
            trace!("No storage available for prefix index, using direct comparison");
        }
    
        // This fallback should rarely be used - only when storage is unavailable or prefix lookup fails
        if !targets.is_empty() {
            warn!("WARNING: Performing full target scan - this should be rare!");
            
            // Split targets into chunks for parallel processing
            let batch_size = self.calculate_optimal_batch_size();
            let chunks: Vec<&[NGramEntry]> = targets.chunks(batch_size).collect();
            trace!("Processing {} targets in {} chunks of size {}", 
                targets.len(), chunks.len(), batch_size);
    
            // Process chunks using configured thread pool
            let results = if let Some(pool) = &self.thread_pool {
                pool.install(|| self.process_chunks_parallel(source, &chunks))
            } else {
                self.process_chunks_parallel(source, &chunks)
            }?;
    
            // Merge and process results
            let matches = self.merge_results(results)?;
            
            trace!("Parallel matching completed in {:?}, found {} matches", 
                start_time.elapsed(), matches.len());
            
            Ok(matches)
        } else {
            // If no targets provided and storage lookup failed, return empty results
            trace!("No targets provided and storage lookup failed/unavailable");
            Ok(Vec::new())
        }
    }
    
    // Gets the thread count from the processor config
    fn get_thread_count(config: &ProcessorConfig) -> usize {
        if config.parallel_thread_count > 0 {
            config.parallel_thread_count
        } else {
            num_cpus::get()
        }
    }

    // Calculates the optimal batch size for target chunks
    fn calculate_optimal_batch_size(&self) -> usize {
        if let Ok(mem_info) = sys_info::mem_info() {
            let available_mem = mem_info.avail as usize;
            // Use about 10% of available memory for batch size
            let safe_batch_size = (available_mem / std::mem::size_of::<NGramEntry>()) / 10;
            safe_batch_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
        } else {
            // Fallback to configured size
            self.processor_config.parallel_batch_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
        }
    }

    // Processes chunks of target ngrams in parallel
    fn process_chunks_parallel(
        &self,
        source: &NGramEntry,
        chunks: &[&[NGramEntry]]
    ) -> Result<Vec<Vec<NGramMatch>>> {
        chunks.par_iter()
            .map(|chunk| self.process_chunk(source, chunk))
            .collect()
    }

    // Processes a single chunk of target ngrams
    fn process_chunk(&self, source: &NGramEntry, chunk: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        match self.matcher_config.ngram_similarity_metric {
            SimilarityMetric::Exact => self.process_chunk_exact(source, chunk),
            _ => self.process_chunk_with_filters(source, chunk)
        }
    }
    pub fn calculate_adaptive_batch_size(&self, source_count: usize) -> usize {
        // Get system information
        let cpu_count = num_cpus::get();
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize,
            Err(_) => 1024 * 1024 * 1024, // Default to 1GB if can't determine
        };
        
        // Estimate NGramEntry size (typically ranges from 100-500 bytes)
        let estimated_entry_size = 300;
        
        // Calculate memory-based batch size
        // Use up to 20% of available memory for batching
        let memory_based_size = (available_memory / 5) / estimated_entry_size;
        
        // Calculate CPU-based batch size
        // For small datasets, use fewer larger batches
        // For large datasets, use more smaller batches for better load balancing
        let cpu_based_size = if source_count < 10_000 {
            source_count / cpu_count.max(1)
        } else {
            source_count / (cpu_count * 2).max(1)
        };
        
        // Choose smaller of the two approaches, but keep within reasonable bounds
        cpu_based_size.min(memory_based_size).clamp(100, 2000)
    }

    pub fn create_optimized_thread_pool(&self) -> Result<rayon::ThreadPool> {
    // Determine thread count based on config or system
    let thread_count = if self.processor_config.parallel_thread_count > 0 {
        self.processor_config.parallel_thread_count
    } else {
        // Use 75% of available cores by default to avoid system saturation
        (num_cpus::get() * 3 / 4).max(1)
    };
    
    // Create thread pool with optimized settings
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        // Larger stack size for complex recursive operations
        .stack_size(32 * 1024 * 1024)  // 32MB stack
        // Thread naming for easier debugging
        .thread_name(|i| format!("matcher-worker-{}", i))
        .build()
        .map_err(|e| crate::error::Error::Storage(format!("Failed to create thread pool: {}", e)))
}

    // Processes a chunk using exact matching
    fn process_chunk_exact(&self, source: &NGramEntry, chunk: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        let mut matches = Vec::new();
        let source_text = &source.text_content.cleaned_text;
        let source_len = source_text.len();

        trace!("Processing chunk with {} targets for exact matching", chunk.len());
        trace!("Source text: '{}' (len: {})", source_text, source_len);
        
        let mut match_count = 0;
        for target in chunk {
            // Simple length check first for efficiency
            if target.text_content.cleaned_text.len() == source_len {
                // Direct text comparison
                if target.text_content.cleaned_text == *source_text {
                    trace!("Exact match found: '{}' (source={}, target={})", 
                        source_text, source.sequence_number, target.sequence_number);
                    match_count += 1;
                    
                    matches.push(NGramMatch {
                        source_position: source.sequence_number as usize,
                        target_position: target.sequence_number as usize,
                        ngram: source.clone(),
                        similarity: 1.0,
                    });
                }
            }
        }
        
        if match_count > 0 {
            trace!("Found {} exact matches in this chunk", match_count);
        } else {
            trace!("No exact matches found in this chunk");
        }

        Ok(matches)
    }

    // Merges results from parallel processing
    fn merge_results(&self, chunk_results: Vec<Vec<NGramMatch>>) -> Result<Vec<NGramMatch>> {
        let mut seen_positions = HashSet::new();
        let mut merged = Vec::new();

        for chunk_matches in chunk_results {
            for match_result in chunk_matches {
                let pos_key = (match_result.source_position, match_result.target_position);
                if !seen_positions.contains(&pos_key) {
                    seen_positions.insert(pos_key);
                    merged.push(match_result);
                }
            }
        }

        // Sort by position and similarity
        merged.sort_by(|a, b| {
            a.source_position.cmp(&b.source_position)
                .then(a.target_position.cmp(&b.target_position))
                .then(b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal))
        });

        Ok(merged)
    }

    // Process a chunk with similarity-based filtering
    fn process_chunk_with_filters(&self, source: &NGramEntry, chunk: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        let mut matches = Vec::new();
        let source_span = TextSpan {
            original_text: source.text_content.original_slice.clone(),
            ngrams: vec![source.clone()],
        };

        for target in chunk {
            // Full filtering pipeline for non-exact matching
            if !self.passes_initial_filters(source, target) {
                continue;
            }

            let quick_similarity = self.calculate_quick_similarity(source, target);
            if quick_similarity >= SIMILARITY_THRESHOLD {
                let target_span = TextSpan {
                    original_text: target.text_content.original_slice.clone(),
                    ngrams: vec![target.clone()],
                };

                let algorithm = self.get_similarity_algorithm()?;
                let similarity = algorithm.compare_spans(&source_span, &target_span)?;

                if similarity >= self.matcher_config.ngram_min_confidence {
                    matches.push(NGramMatch {
                        source_position: source.sequence_number as usize,
                        target_position: target.sequence_number as usize,
                        ngram: source.clone(),
                        similarity,
                    });
                }
            }
        }

        Ok(matches)
    }

    // Helper functions from the existing implementation
    fn passes_initial_filters(&self, source: &NGramEntry, target: &NGramEntry) -> bool {
        // Length ratio check
        let len_ratio = target.text_content.cleaned_text.len() as f64 
            / source.text_content.cleaned_text.len() as f64;
        
        if !(0.7..=1.3).contains(&len_ratio) {
            return false;
        }

        // Quick character set comparison
        let source_chars: HashSet<_> = source.text_content.cleaned_text.chars().collect();
        let target_chars: HashSet<_> = target.text_content.cleaned_text.chars().collect();
        let overlap = source_chars.intersection(&target_chars).count();
        let union = source_chars.union(&target_chars).count();
        
        let char_similarity = overlap as f64 / union as f64;
        char_similarity >= SIMILARITY_THRESHOLD
    }

    fn calculate_quick_similarity(&self, source: &NGramEntry, target: &NGramEntry) -> f64 {
        let source_prefix: String = source.text_content.cleaned_text.chars().take(3).collect();
        let target_prefix: String = target.text_content.cleaned_text.chars().take(3).collect();
        
        if source_prefix == target_prefix {
            1.0
        } else {
            let source_freq = count_char_frequencies(&source.text_content.cleaned_text);
            let target_freq = count_char_frequencies(&target.text_content.cleaned_text);
            calculate_frequency_similarity(&source_freq, &target_freq)
        }
    }

    fn get_similarity_algorithm(&self) -> Result<Box<dyn SimilarityAlgorithm + '_>> {
        crate::matcher::algorithms::SimilarityAlgorithmFactory::create(
            self.matcher_config.ngram_similarity_metric,
            &self.matcher_config
        )
    }
    
    // Add this method to decide between pipeline and batch processing
    pub fn find_matches_with_optimal_strategy(
        &self,
        source_ngrams: &[NGramEntry],
        storage: &'a dyn StorageBackend
    ) -> Result<Vec<NGramMatch>> {
        // Get workload characteristics
        let ngram_count = source_ngrams.len();
        let text_size = source_ngrams.iter()
            .map(|ng| ng.text_content.cleaned_text.len())
            .sum();
            
        // Check if we should use the pipeline
        if self.should_use_pipeline(ngram_count, text_size) {
            info!("Using pipeline processing for {} ngrams", ngram_count);
            self.find_matches_pipeline(source_ngrams, storage)
        } else {
            info!("Using batch parallel processing for {} ngrams", ngram_count);
            self.find_matches_parallel(source_ngrams, storage)
        }
    }
    
    // Method to use pipeline processing
    pub fn find_matches_pipeline(
        &self,
        source_ngrams: &[NGramEntry],
        storage: &'a dyn StorageBackend
    ) -> Result<Vec<NGramMatch>> {
        let start_time = Instant::now();
        info!("Starting pipeline processing for {} source ngrams", source_ngrams.len());
        
        // Use the existing create_optimal_pipeline method instead of duplicating the logic
        let pipeline = self.create_optimal_pipeline(source_ngrams.len());
        
        // Add the storage strategy based on config if needed
        let pipeline = if self.processor_config.storage_access_strategy == "pooled" {
            pipeline.with_storage_strategy(StorageAccessStrategy::Pooled(
                self.processor_config.storage_pool_size
            ))
        } else if self.processor_config.storage_access_strategy == "per_worker" {
            pipeline.with_storage_strategy(StorageAccessStrategy::PerWorker)
        } else {
            pipeline.with_storage_strategy(StorageAccessStrategy::Shared)
        };
        
        // Process using the pipeline
        let result = pipeline.process(source_ngrams, self, storage);
        
        info!("Pipeline processing completed in {:?}", start_time.elapsed());
        result
    }
    
    // Helper to create an optimally configured pipeline
    fn create_optimal_pipeline(&self, source_size: usize) -> ProcessingPipeline {
        // Get system resources
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize * 1024, // Convert KB to bytes
            Err(_) => 1024 * 1024 * 1024, // Default to 1GB if can't determine
        };
        
        // Create and return the pipeline with optimal settings
        ProcessingPipeline::with_optimal_settings(source_size, available_memory)
    }
    
    // Decision logic for whether to use pipeline
    fn should_use_pipeline(&self, ngram_count: usize, text_size: usize) -> bool {
        // Get configuration values (these should be added to your config)
        let auto_select = self.processor_config.auto_select_strategy;
        let ngram_threshold = self.processor_config.pipeline_ngram_threshold;
        let memory_threshold = self.processor_config.pipeline_memory_threshold;
        
        // If auto-selection is disabled, check manual setting
        if !auto_select {
            return self.processor_config.processing_mode == "pipeline";
        }
        
        // Get system memory
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize * 1024,
            Err(_) => 1024 * 1024 * 1024, // Default to 1GB
        };
        
        // Estimate memory requirements
        let estimated_memory = self.estimate_memory_requirements(ngram_count, text_size);
        let memory_ratio = estimated_memory as f64 / available_memory as f64;
        
        // Decision logic
        let use_pipeline = 
            ngram_count > ngram_threshold ||
            memory_ratio > memory_threshold;
            
        // Log the decision factors
        info!("Pipeline decision: {} (ngrams: {}, memory ratio: {:.2})",
            if use_pipeline { "yes" } else { "no" },
            ngram_count,
            memory_ratio
        );
        
        use_pipeline
    }
    
    // Helper to estimate memory requirements
    fn estimate_memory_requirements(&self, ngram_count: usize, text_size: usize) -> usize {
        // Base memory overhead
        let base_memory = 50 * 1024 * 1024; // 50MB base
        
        // Memory per ngram (rough estimate)
        let ngram_memory = ngram_count * 500; // ~500 bytes per ngram
        
        // Text processing overhead
        let text_memory = text_size * 2; // 2x for processing overhead
        
        base_memory + ngram_memory + text_memory
    }
    
    pub fn find_matches_for_chunk(&self, chunk: &[NGramEntry], storage: &'a dyn StorageBackend) -> Result<Vec<NGramMatch>> {
        // Start timing this chunk processing
        let chunk_start = Instant::now();
        trace!("Processing chunk of {} ngrams", chunk.len());
        
        // Pre-allocate with a reasonable capacity - most ngrams won't match
        // so we allocate for about 10% of the chunk size as a starting point
        let mut matches = Vec::with_capacity(chunk.len() / 10);
        
        // Track how many ngrams actually produced matches
        let mut matching_ngrams = 0;
        
        // Process each source ngram in the chunk
        for source_ngram in chunk {
            // Check if this is a traced ngram for debugging
            let is_traced = !self.matcher_config.traced_phrase.is_empty() && 
                            source_ngram.text_content.cleaned_text.contains(&self.matcher_config.traced_phrase);
            
            if is_traced {
                trace!("TRACE: Processing traced ngram: '{}' (sequence: {})", 
                      source_ngram.text_content.cleaned_text, source_ngram.sequence_number);
            }
            
            // Try optimized path using metadata filtering
            if let Ok(sequences) = storage.get_sequences_by_prefix_for_ngram(source_ngram) {
                if !sequences.is_empty() {
                    if is_traced {
                        trace!("TRACE: Found {} prefix matches for traced ngram", sequences.len());
                    }
                    
                    // Get metadata for efficient filtering
                    if let Ok(metadata_batch) = storage.get_ngram_metadata_batch(&sequences) {
                        // Apply metadata filtering using our new function
                        let filtered_sequences = self.filter_candidates_by_metadata(
                            source_ngram,
                            &metadata_batch,
                            self.matcher_config.ngram_similarity_metric
                        );
                        
                        if is_traced {
                            trace!("TRACE: Filtered {} → {} candidates using metadata for traced ngram", 
                                  metadata_batch.len(), filtered_sequences.len());
                        }
                        
                        // Only fetch full ngrams for promising candidates
                        if !filtered_sequences.is_empty() {
                            if let Ok(filtered_ngrams) = storage.get_ngrams_batch(&filtered_sequences) {
                                // Get the appropriate similarity algorithm
                                let algorithm = match self.matcher_config.ngram_similarity_metric {
                                    SimilarityMetric::Exact => Box::new(ExactMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                    SimilarityMetric::Jaccard => Box::new(JaccardMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                    SimilarityMetric::Cosine => Box::new(CosineMatcher::new()) as Box<dyn SimilarityAlgorithm>,
                                    SimilarityMetric::Levenshtein => Box::new(LevenshteinMatcher::with_config(
                                        self.matcher_config.algorithm_config.levenshtein_max_distance
                                    )) as Box<dyn SimilarityAlgorithm>,
                                    SimilarityMetric::VSA => Box::new(VSAMatcher::with_config(
                                        self.matcher_config.algorithm_config.vsa_min_term_weight
                                    )) as Box<dyn SimilarityAlgorithm>,
                                };
                                
                                // Process each target ngram with the appropriate similarity algorithm
                                let ngram_matches: Vec<NGramMatch> = filtered_ngrams.iter()
                                    .filter_map(|target| {
                                        // Calculate similarity using the configured algorithm
                                        match algorithm.compare_ngrams(source_ngram, target) {
                                            Ok(similarity) => {
                                                // Only keep matches that meet the confidence threshold
                                                if similarity >= self.matcher_config.ngram_min_confidence {
                                                    if is_traced {
                                                        trace!("TRACE: Match found for traced ngram: {} → {} with similarity {:.3}", 
                                                              source_ngram.text_content.cleaned_text,
                                                              target.text_content.cleaned_text,
                                                              similarity);
                                                    }
                                                    
                                                    Some(NGramMatch {
                                                        source_position: source_ngram.sequence_number as usize,
                                                        target_position: target.sequence_number as usize,
                                                        ngram: source_ngram.clone(),
                                                        similarity,
                                                    })
                                                } else {
                                                    None
                                                }
                                            },
                                            Err(e) => {
                                                warn!("Error calculating similarity: {}", e);
                                                None
                                            }
                                        }
                                    })
                                    .collect();
                                    
                                // Add matches to results if any found
                                if !ngram_matches.is_empty() {
                                    matching_ngrams += 1;
                                    
                                    // Store the length before moving ngram_matches
                                    let matches_count = ngram_matches.len();
                                    
                                    // Extend the matches vector
                                    matches.extend(ngram_matches);
                                    
                                    if is_traced {
                                        trace!("TRACE: Added {} matches for traced ngram to results", matches_count);
                                    }
                                }
                                
                                // Skip the fallback call
                                continue;
                            }
                        }
                    }
                }
            }
            
            // Fallback to original method if optimized path fails
            match self.find_matches_fallback(source_ngram, &[]) {
                Ok(ngram_matches) => {
                    // Only do work if we actually found matches
                    if !ngram_matches.is_empty() {
                        // Record that this ngram produced matches
                        matching_ngrams += 1;
                        
                        // Store the length before moving
                        let matches_count = ngram_matches.len();
                        
                        // Add all matches from this ngram to our result set
                        matches.extend(ngram_matches);
                        
                        // Detailed logging for significant matches
                        if matches_count > 10 || is_traced {
                            trace!("Found {} matches for ngram '{}' (sequence: {})",
                                  matches_count, 
                                  source_ngram.text_content.cleaned_text,
                                  source_ngram.sequence_number);
                        }
                    } else if is_traced {
                        trace!("TRACE: No matches found for traced ngram using fallback method");
                    }
                },
                Err(e) => {
                    // Log the error but continue processing other ngrams
                    warn!("Error matching ngram {}: {}", source_ngram.sequence_number, e);
                }
            }
        }
        
        // If we found a significant number of matches, log detailed stats
        if matches.len() > 0 {
            let elapsed = chunk_start.elapsed();
            trace!("Chunk processing complete: {} ngrams produced {} matches in {:?}",
                  matching_ngrams, matches.len(), elapsed);
                  
            // Calculate and log match density for diagnostic purposes
            let match_density = matching_ngrams as f64 / chunk.len() as f64;
            trace!("Match density: {:.2}% of ngrams produced matches", match_density * 100.0);
        }
        
        Ok(matches)
    }
    /// Filters ngram candidates based on metadata before fetching full content
    /// Returns a filtered list of sequence numbers that pass the metadata checks
    fn filter_candidates_by_metadata(
        &self,
        source: &NGramEntry, 
        metadata_batch: &[NGramMetadata],
        similarity_metric: SimilarityMetric
    ) -> Vec<u64> {
        if metadata_batch.is_empty() {
            return Vec::new();
        }

        let source_text = &source.text_content.cleaned_text;
        let source_text_len = source_text.len();
        
        // Try to get source metadata from the database first
        let mut source_metadata: Option<NGramMetadata> = None;
        if let Some(storage) = self.storage {
            // Try to get metadata for source ngram
            if let Ok(metadata_vec) = storage.get_ngram_metadata_batch(&[source.sequence_number]) {
                if !metadata_vec.is_empty() {
                    source_metadata = Some(metadata_vec[0].clone());
                }
            }
        }
        
        // Choose filtering thresholds based on similarity metric
        let (length_ratio_min, length_ratio_max) = match similarity_metric {
            SimilarityMetric::Exact => (0.98, 1.02),      // Very strict for exact matching
            SimilarityMetric::Levenshtein => (0.80, 1.20), // More lenient for edit distance
            SimilarityMetric::Jaccard => (0.9, 1.1),    // Moderate for Jaccard
            SimilarityMetric::Cosine => (0.9, 1.1),     // Moderate for Cosine
            SimilarityMetric::VSA => (0.90, 1.10),        // Slightly stricter for VSA
        };
        
        // For exact matching, we can do more filtering with hash and first_chars
        let is_exact_match = similarity_metric == SimilarityMetric::Exact;
        
        // Calculate source values only if we couldn't get metadata
        let (source_hash, source_first_chars) = if source_metadata.is_none() && is_exact_match {
            // Fallback implementation of hash calculation
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            source_text.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Fallback implementation of first_chars extraction
            let mut first_chars = [0u8; 8];
            let bytes = source_text.as_bytes();
            for i in 0..first_chars.len() {
                if i < bytes.len() {
                    first_chars[i] = bytes[i];
                }
            }
            
            (hash, first_chars)
        } else {
            (0, [0u8; 8]) // Dummy values when not using exact matching or metadata available
        };
        
        // Apply the filters
        metadata_batch.iter()
            .filter_map(|meta| {
                // Length-based filter (always applied)
                let length_ratio = if let Some(ref source_meta) = source_metadata {
                    // Use exact length from metadata if available
                    meta.text_length as f64 / source_meta.text_length as f64
                } else {
                    // Fallback to length from text content
                    meta.text_length as f64 / source_text_len as f64
                };
                
                let length_ok = length_ratio >= length_ratio_min && length_ratio <= length_ratio_max;
                
                if !length_ok {
                    if log::log_enabled!(log::Level::Trace) {
                        trace!("Filtered out sequence {} by length ratio {:.2} (outside {}..{})", 
                            meta.sequence_number, length_ratio, length_ratio_min, length_ratio_max);
                    }
                    return None;
                }
                
                // For exact matching, we can do additional checks
                if is_exact_match {
                    if let Some(ref source_meta) = source_metadata {
                        // Use metadata for comparison if available
                        if meta.text_hash != 0 && source_meta.text_hash != 0 && 
                        meta.text_hash != source_meta.text_hash {
                            trace!("Filtered out sequence {} by hash mismatch", 
                                meta.sequence_number);
                            return None;
                        }
                        
                        if meta.first_chars != [0u8; 8] && source_meta.first_chars != [0u8; 8] &&
                        meta.first_chars != source_meta.first_chars {
                            trace!("Filtered out sequence {} by first chars mismatch", 
                                meta.sequence_number);
                            return None;
                        }
                    } else {
                        // Use calculated values as fallback
                        if meta.text_hash != 0 && source_hash != meta.text_hash {
                            trace!("Filtered out sequence {} by calculated hash mismatch", 
                                meta.sequence_number);
                            return None;
                        }
                        
                        if meta.first_chars != [0u8; 8] && meta.first_chars != source_first_chars {
                            trace!("Filtered out sequence {} by calculated first chars mismatch", 
                                meta.sequence_number);
                            return None;
                        }
                    }
                }
                
                // This candidate passed all filters
                Some(meta.sequence_number)
            })
            .collect()
    }
}

// Helper functions for character frequency analysis
fn count_char_frequencies(text: &str) -> HashMap<char, usize> {
    let mut freqs = HashMap::new();
    for c in text.chars() {
        *freqs.entry(c).or_insert(0) += 1;
    }
    freqs
}

fn calculate_frequency_similarity(freq1: &HashMap<char, usize>, freq2: &HashMap<char, usize>) -> f64 {
    let mut overlap = 0;
    let mut total = 0;

    for (c, count1) in freq1 {
        if let Some(count2) = freq2.get(c) {
            overlap += count1.min(count2);
        }
        total += count1;
    }

    for count2 in freq2.values() {
        total += count2;
    }

    if total == 0 {
        0.0
    } else {
        (2 * overlap) as f64 / total as f64
    }
}
