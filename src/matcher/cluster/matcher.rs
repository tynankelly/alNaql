// src/matcher/cluster/matcher.rs
use ahash::{AHashMap, AHashSet};
use std::time::{Instant, Duration};
use std::sync::Mutex;
use std::sync::{Arc, RwLock};
use std::ops::Range;
use std::time::{SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use indicatif::ProgressBar;
use log::{info, debug, warn, trace, error};
use std::path::{Path, PathBuf};
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::io::Write;
use crate::TempFileManager;
use crate::utils::MemoryManager;
use crate::error::Result;
use crate::matcher::parallel::ParallelMatcher;
use crate::error::Error;
use crate::types::NGramEntry;
use crate::storage::StorageBackend;
use crate::storage::lmdb::cache::METADATA_CACHE;
use crate::config::subsystems::matcher::MatcherConfig;
use crate::config::subsystems::generator::NGramType;
use crate::config::subsystems::processor::ProcessorConfig;
use crate::matcher::concurrent::{MemoryMonitor, MemoryPressure};
use crate::matcher::similarity::SimilarityCalculator;
use crate::matcher::{StorageAccessStrategy, SimpleMatchResult, ProcessingPipeline, MatchResult, MatchCandidate};
use crate::matcher::types::NGramMatch;
use crate::matcher::banality::CommonNgrams;

// TextContent trait definition
pub trait TextContent: Clone {
    fn get_cleaned_text(&self) -> &str;
    fn get_position(&self) -> usize;
}

impl TextContent for NGramEntry {
    fn get_cleaned_text(&self) -> &str {
        &self.text_content.cleaned_text
    }

    fn get_position(&self) -> usize {
        self.sequence_number as usize
    }
}

impl TextContent for NGramMatch {
    fn get_cleaned_text(&self) -> &str {
        &self.ngram.text_content.cleaned_text
    }

    fn get_position(&self) -> usize {
        self.source_position
    }
}

pub struct ClusteringMatcher<S: StorageBackend> {
    pub source_storage: S,
    pub target_storage: S,
    pub matcher_config: MatcherConfig,
    pub processor_config: ProcessorConfig, 
    pub similarity_calc: SimilarityCalculator,
    pub ngram_type: NGramType,
    pub isnad_phrases: AHashSet<String>,
    pub isnad_words: AHashSet<String>,
    pub isnad_density_threshold: f64,
    pub isnad_min_length: usize,
    pub banality_phrases: AHashSet<String>,
    pub banality_words: AHashSet<String>,
    pub banality_density_threshold: f64,
    pub banality_detection_mode: String,
    pub common_ngrams: Option<CommonNgrams>,
    pub ngram_cache: Arc<RwLock<AHashMap<(u64, u64, bool), Vec<NGramEntry>>>>,
    pub text_cache: Arc<RwLock<AHashMap<Range<usize>, String>>>,
    pub similarity_cache: Arc<RwLock<AHashMap<(String, String), f64>>>,
}

impl<S: StorageBackend> std::fmt::Debug for ClusteringMatcher<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusteringMatcher")
            .field("matcher_config", &self.matcher_config)
            .field("processor_config", &self.processor_config)
            .field("ngram_type", &self.ngram_type)
            .field("isnad_phrases", &self.isnad_phrases)
            .field("isnad_words", &self.isnad_words)
            .field("isnad_density_threshold", &self.isnad_density_threshold)
            .field("isnad_min_length", &self.isnad_min_length)
            .field("banality_phrases", &self.banality_phrases)
            .field("banality_words", &self.banality_words)
            .field("banality_density_threshold", &self.banality_density_threshold)
            .finish_non_exhaustive()
    }
}

impl<S: StorageBackend + Clone> Clone for ClusteringMatcher<S> {
    fn clone(&self) -> Self {
        Self {
            source_storage: self.source_storage.clone(),
            target_storage: self.target_storage.clone(),
            matcher_config: self.matcher_config.clone(),
            processor_config: self.processor_config.clone(),
            similarity_calc: SimilarityCalculator::new(
                self.matcher_config.clone(), 
                self.ngram_type
            ).expect("Failed to create SimilarityCalculator"),
            ngram_type: self.ngram_type,
            isnad_phrases: self.isnad_phrases.clone(),
            isnad_words: self.isnad_words.clone(),
            isnad_density_threshold: self.isnad_density_threshold,
            isnad_min_length: self.isnad_min_length,
            banality_phrases: self.banality_phrases.clone(),
            banality_words: self.banality_words.clone(),
            banality_density_threshold: self.banality_density_threshold,
            banality_detection_mode: self.banality_detection_mode.clone(),
            common_ngrams: self.common_ngrams.clone(),
            ngram_cache: Arc::clone(&self.ngram_cache),
            text_cache: Arc::clone(&self.text_cache),
            similarity_cache: Arc::clone(&self.similarity_cache),
        }
    }
}

impl<S: StorageBackend> ClusteringMatcher<S> {
    // Inside ClusteringMatcher impl
    pub fn new(
        source_storage: S,
        target_storage: S,
        matcher_config: MatcherConfig,
        processor_config: ProcessorConfig,
        ngram_type: NGramType,
    ) -> Result<Self> {
        if processor_config.banality_detection_mode == "automatic" {
            info!("SYSTEM: Initializing with AUTOMATIC banality detection mode");
        }
        let similarity_calc = SimilarityCalculator::new(matcher_config.clone(), ngram_type)?;
        
        // Load isnad phrases (existing code)
        let isnad_phrases = match Self::load_terms_from_file("filters/isnad_phrases.txt") {
            Ok(phrases) => phrases,
            Err(e) => {
                warn!("Failed to load isnad phrases: {}. Proceeding with empty set.", e);
                AHashSet::new()
            }
        };
        
        // Load isnad words (existing code)
        let isnad_words = match Self::load_terms_from_file("filters/isnad_words.txt") {
            Ok(words) => words,
            Err(e) => {
                warn!("Failed to load isnad words: {}. Proceeding with empty set.", e);
                AHashSet::new()
            }
        };
        
        // Load banality phrases (existing code)
        let banality_phrases = match Self::load_terms_from_file("filters/banalities_phrases.txt") {
            Ok(phrases) => phrases,
            Err(e) => {
                warn!("Failed to load banality phrases: {}. Proceeding with empty set.", e);
                AHashSet::new()
            }
        };
        
        // Load banality words (existing code)
        let banality_words = match Self::load_terms_from_file("filters/banalities_words.txt") {
            Ok(words) => words,
            Err(e) => {
                warn!("Failed to load banality words: {}. Proceeding with empty set.", e);
                AHashSet::new()
            }
        };
        
        // Default configuration values
        let isnad_density_threshold = 0.65;
        let isnad_min_length = 25;
        let banality_density_threshold = 0.67;
        
        // Initialize with the configured banality detection mode
        let banality_detection_mode = processor_config.banality_detection_mode.clone();
        // Initialize caches with reasonable capacities
        let ngram_cache = Arc::new(RwLock::new(AHashMap::with_capacity(1000)));
        let text_cache = Arc::new(RwLock::new(AHashMap::with_capacity(5000)));
        let similarity_cache = Arc::new(RwLock::new(AHashMap::with_capacity(10000)));
        // Create the matcher instance with all fields
        let mut matcher = Self {
            source_storage,
            target_storage,
            matcher_config,
            processor_config,
            similarity_calc,
            ngram_type,
            isnad_phrases,
            isnad_words,
            isnad_density_threshold,
            isnad_min_length,
            banality_phrases,
            banality_words,
            banality_density_threshold,
            banality_detection_mode,
            common_ngrams: None,
            ngram_cache,
            text_cache,
            similarity_cache,
        };
        
        // If using automatic mode, initialize the CommonNgrams structure
        if matcher.banality_detection_mode == "automatic" {
            info!("SYSTEM: Preparing automatic banality detection");
            
            let mut common_ngrams = crate::matcher::banality::CommonNgrams::new(
                matcher.processor_config.banality_auto_proportion,
                matcher.processor_config.banality_auto_threshold
            );
            
            // Analyze the source database to identify common n-grams
            info!("SYSTEM: Analyzing source database for common n-grams...");
            match common_ngrams.analyze_database(&matcher.source_storage) {
                Ok(_) => {
                    info!("SYSTEM: Automatic banality detection ready: {}", common_ngrams.stats());
                    matcher.common_ngrams = Some(common_ngrams);
                },
                Err(e) => {
                    warn!("SYSTEM: Failed to analyze database: {}", e);
                    warn!("SYSTEM: Falling back to phrase-based detection");
                    matcher.banality_detection_mode = "phrase".to_string();
                }
            }
        } else {
            info!("SYSTEM: Using phrase-based banality detection");
        }
        
        Ok(matcher)
    }

    fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            return format!("{}s", total_secs);
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            return format!("{}m {}s", mins, secs);
        } else {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            return format!("{}h {}m", hours, mins);
        }
    }
    
    pub fn find_matches(
        &self,
        source_range: Range<u64>,
        target_range: Range<u64>,
        output_path: Option<&Path>,
        progress_bar: Option<&ProgressBar>,
        early_escape_threshold: f64,  // New parameter, default will be 0.9 (90%) 
    ) -> Result<usize> {
        let start_time = Instant::now();
        info!("Starting match finding process");
        
        // Create a unique temporary directory for this processing run
        let temp_dir = PathBuf::from("temp/matches")
            .join(format!("run_{}", SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()));
        info!("DEBUG: Creating temporary directory at: {:?}", temp_dir);
        
        // Create the TempFileManager
        let temp_manager = match TempFileManager::new(&temp_dir, "matches") {
            Ok(manager) => manager,
            Err(e) => return Err(Error::Io(e)),
        };
    
        // Wrap everything in a closure to ensure cleanup happens regardless of success/failure
        let result = (|| {
            // Initialize a vector to track temporary file paths instead of storing all matches
            let mut temp_file_paths: Vec<PathBuf> = Vec::new();
            
            // Create output file writer if path provided
            let mut writer = if let Some(path) = output_path {
                Some(std::io::BufWriter::new(File::create(path)?))
            } else {
                None
            };
            
            // Initialize discarded content logger if enabled
            if self.processor_config.log_discarded_matches {
                let discarded_path = Path::new(&self.processor_config.discarded_log_path);
                // Create parent directories if they don't exist
                if let Some(parent) = discarded_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                
                if let Err(e) = crate::utils::DISCARDED_LOGGER.init(discarded_path) {
                    warn!("Failed to initialize discarded content logger: {}", e);
                } else {
                    info!("Discarded content will be logged to {:?}", discarded_path);
                }
            }
            
            // Get target ngrams
            info!("Retrieving target ngrams from range {}..{}", 
                 target_range.start, target_range.end);
            let target_ngrams = self.target_storage.get_sequence_range(target_range.clone())?;
            info!("Retrieved {} target ngrams", target_ngrams.len());
            
            // Process source ngrams in chunks
            let chunk_size = 10_000; 
            let total_chunks = ((source_range.end - source_range.start) as usize + chunk_size - 1) / chunk_size;
            
            // Initialize progress bar if provided
            if let Some(pb) = progress_bar {
                pb.set_length(total_chunks as u64);
                pb.set_position(0);
                pb.set_message("Starting chunk processing...");
            }
            
            // Track chunk times for ETA calculation
            let mut chunk_times = Vec::with_capacity(total_chunks.min(10));
            
            // Log initial memory state
            if let Ok(mem_info) = sys_info::mem_info() {
                let used_mem_mb = (mem_info.total - mem_info.avail) / 1024;
                info!("Initial memory usage: {} MB", used_mem_mb);
            }
            
            for chunk_index in 0..total_chunks {
                // Calculate chunk range
                let chunk_start = source_range.start + (chunk_index as u64 * chunk_size as u64);
                let chunk_end = (chunk_start + chunk_size as u64).min(source_range.end);
                
                let chunk_start_time = Instant::now();
                info!("Processing chunk {}/{}: source ngrams {}..{}", 
                     chunk_index + 1, total_chunks, chunk_start, chunk_end);
                
                // Get and process this chunk
                let source_chunk = self.source_storage.get_sequence_range(chunk_start..chunk_end)?;
                
                // Use parallel or sequential matching based on configuration
                let ngram_matches_time = Instant::now();
                let chunk_matches = self.find_ngram_matches(&source_chunk, &target_ngrams)?;
                info!("Found {} initial matches in {:?}", 
                     chunk_matches.len(), ngram_matches_time.elapsed());
                
                // Add early escape check for the first chunk
                if chunk_index == 0 && !source_chunk.is_empty() {
                    // Calculate match rate
                    let unique_matched_source_ngrams: std::collections::HashSet<_> = chunk_matches.iter()
                        .map(|m| m.source_position)
                        .collect();
                    
                    let match_rate = unique_matched_source_ngrams.len() as f64 / source_chunk.len() as f64;
                    
                    info!("First chunk match rate: {:.2}% ({} of {} source ngrams have matches)",
                         match_rate * 100.0, unique_matched_source_ngrams.len(), source_chunk.len());
                    
                    // If match rate exceeds threshold, this is likely a duplicate
                    if match_rate >= early_escape_threshold {
                        info!("EARLY ESCAPE: Match rate {:.2}% exceeds threshold {:.2}%. These databases appear to be duplicates or very similar.",
                             match_rate * 100.0, early_escape_threshold * 100.0);
                        
                        // If a progress bar is provided, update it
                        if let Some(pb) = progress_bar {
                            pb.finish_with_message(format!(
                                "Early termination: {:.2}% match rate indicates duplicate databases",
                                match_rate * 100.0
                            ));
                        }
                        
                        // Create an empty output file to mark that we processed this pair
                        if let Some(path) = output_path {
                            if let Ok(mut file) = File::create(path) {
                                let _ = writeln!(file, "{{\"early_escape\": true, \"match_rate\": {:.4}, \"threshold\": {:.4}}}",
                                               match_rate, early_escape_threshold);
                            }
                        }
                        
                        // Return special value to indicate early escape
                        return Ok(usize::MAX);
                    }
                }
                           
                if !chunk_matches.is_empty() {
                    // Find dense regions - choose between parallel and sequential processing
                    let dense_time = Instant::now();
                    
                    let candidates = if self.processor_config.parallel_dense_regions && 
                                     chunk_matches.len() >= self.processor_config.parallel_dense_threshold {
                        info!("Using parallel dense region finding for {} matches", chunk_matches.len());
                        super::parallel_chains::find_dense_regions_parallel(self, &chunk_matches)?
                    } else {
                        info!("Using sequential dense region finding for {} matches", chunk_matches.len());
                        self.find_dense_regions_chunk(&chunk_matches, None)?
                    };
                    
                    info!("Found {} dense regions in {:?}", 
                         candidates.len(), dense_time.elapsed());
                    
                    if !candidates.is_empty() {
                        // Validate matches WITHOUT merging
                        let validation_time = Instant::now();
                        
                        // Use validate_without_merging instead
                        let chunk_valid_matches = self.validate_without_merging(
                            candidates.clone(), &source_chunk, &target_ngrams)?;
                        
                        // Get the count of valid matches
                        let chunk_matches_count = chunk_valid_matches.len();
                        info!("Validated {} matches in {:?} (not merged yet)", 
                             chunk_matches_count, validation_time.elapsed());
                        
                        // CHANGE: Instead of collecting in all_matches, write to temp file
                        if !chunk_valid_matches.is_empty() {
                            let file_path = temp_manager.write_matches(&chunk_valid_matches, chunk_index)?;
                            info!("Wrote {} matches from chunk {} to temporary file: {:?}",
                                 chunk_matches_count, chunk_index + 1, file_path);
                            temp_file_paths.push(file_path.clone());
                            info!("DEBUG: Added temp file to paths list: {:?}", file_path);
                            info!("DEBUG: Current temp files count: {}", temp_file_paths.len());
                        }
                    }
                }
                
                // Help garbage collection by explicitly dropping large data
                drop(chunk_matches);
                drop(source_chunk);
                           
                // Force memory cleanup every few chunks
                if chunk_index % 3 == 0 {
                    info!("Triggering memory cleanup after chunk {}", chunk_index);
                    // Brief pause to allow memory to be reclaimed
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            
                // After processing the chunk, update the progress bar
                let chunk_duration = chunk_start_time.elapsed();
                info!("Chunk {}/{} completed in {:?}", 
                     chunk_index + 1, total_chunks, chunk_duration);
                
                // Log memory usage after each chunk
                if let Ok(mem_info) = sys_info::mem_info() {
                    let used_mem_mb = (mem_info.total - mem_info.avail) / 1024;
                    info!("Memory usage after chunk {}: {} MB", chunk_index + 1, used_mem_mb);
                }
                     
                // Track chunk times for ETA estimation
                chunk_times.push(chunk_duration);
                
                // Keep only the last 5 chunk times for better accuracy as processing may speed up/slow down
                if chunk_times.len() > 5 {
                    chunk_times.remove(0);
                }
                
                // Update progress bar if provided
                if let Some(pb) = progress_bar {
                    pb.inc(1);
                    
                    // Calculate average chunk time and estimate remaining time
                    if !chunk_times.is_empty() {
                        // Calculate average chunk time
                        let total_millis: u128 = chunk_times.iter()
                            .map(|d| d.as_millis())
                            .sum();
                        let avg_chunk_time = Duration::from_millis((total_millis / chunk_times.len() as u128) as u64);
                        
                        // Calculate remaining time
                        let chunks_remaining = total_chunks - (chunk_index + 1);
                        let estimated_remaining = avg_chunk_time.mul_f32(chunks_remaining as f32);
                        
                        // Calculate overall progress statistics
                        let processed_pct = (chunk_index + 1) as f32 / total_chunks as f32 * 100.0;
                        // Format a detailed progress message
                        pb.set_message(format!(
                            "Chunk {}/{} ({:.1}%) - Found {} temp files so far - Avg: {} per chunk - ETA: {}",
                            chunk_index + 1,
                            total_chunks,
                            processed_pct,
                            temp_file_paths.len(),
                            Self::format_duration(avg_chunk_time),
                            Self::format_duration(estimated_remaining)
                        ));
                    } else {
                        // Simpler message for the first chunk
                        pb.set_message(format!(
                            "Chunk {}/{} completed - Created {} temp files so far",
                            chunk_index + 1,
                            total_chunks,
                            temp_file_paths.len()
                        ));
                    }
                }
            }
            
            // Process the temporary files with streaming merge
            if !temp_file_paths.is_empty() {
                info!("Processing final merging from {} temporary files", temp_file_paths.len());
                info!("DEBUG: About to merge the following temp files:");
                for (i, path) in temp_file_paths.iter().enumerate() {
                    info!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, path.exists());
                }
                let merging_start_time = Instant::now();
                
                // Call our new streaming merge method (we'll implement this next)
                let total_match_count = self.stream_merge_from_temp_files(
                    &temp_file_paths, 
                    &temp_manager, 
                    writer.as_mut()
                )?;
                
                info!("Final merging complete: {} matches after merging (took {:?})", 
                     total_match_count, merging_start_time.elapsed());
                
                     info!("Completed all chunks, found {} total matches in {:?}", 
                     total_match_count, start_time.elapsed());
                
                Ok(total_match_count)
            } else {
                // No matches found in any chunks
                info!("No matches found in any chunks");
                Ok(0)
            }
        })();
        
        // Always clean up temporary files, regardless of processing outcome
        if let Err(e) = temp_manager.cleanup() {
            warn!("Error cleaning up temporary files: {}", e);
        }
        
        // Return the result
        result
    }
    
    // In matcher.rs, modify find_ngram_matches function
    // Corrected find_ngram_matches function
    fn find_ngram_matches(
        &self,
        source_ngrams: &[NGramEntry],
        target_ngrams: &[NGramEntry]
    ) -> Result<Vec<NGramMatch>> {
        // Add tracing for source n-grams
        let traced_sources: Vec<&NGramEntry> = source_ngrams.iter()
            .filter(|ng| self.is_traced_ngram(&ng.text_content.cleaned_text))
            .collect();
            
        if !traced_sources.is_empty() {
            info!("TRACE [parallel]: Found {} source n-grams containing traced text", traced_sources.len());
            for (i, ng) in traced_sources.iter().enumerate() {
                info!("TRACE [parallel]: Source traced n-gram #{}: '{}' (sequence: {})", 
                    i+1, ng.text_content.cleaned_text, ng.sequence_number);
            }
        }
        
        // Add tracing for target n-grams
        let traced_targets: Vec<&NGramEntry> = target_ngrams.iter()
            .filter(|ng| self.is_traced_ngram(&ng.text_content.cleaned_text))
            .collect();
            
        if !traced_targets.is_empty() {
            info!("TRACE [find_ngram_matches]: Found {} target n-grams containing traced text", traced_targets.len());
            for (i, ng) in traced_targets.iter().enumerate() {
                self.log_traced_ngram("find_ngram_matches",
                                &format!("Target traced n-gram #{} (sequence: {})", i+1, ng.sequence_number),
                                &ng.text_content.cleaned_text);
            }
        }

        // Original decision code with return value
        if self.should_use_parallel(target_ngrams) {
            info!("Using parallel processing for ngram matching");
            self.find_ngram_matches_parallel(source_ngrams, target_ngrams)
        } else {
            info!("Using sequential processing for ngram matching");
            self.find_ngram_matches_sequential(source_ngrams, target_ngrams)
        }
    }

    // In matcher.rs, modify find_ngram_matches_sequential
    fn find_ngram_matches_sequential(
        &self,
        source_ngrams: &[NGramEntry],
        target_ngrams: &[NGramEntry]
    ) -> Result<Vec<NGramMatch>> {
        let mut all_matches = Vec::new();
        
        for source_ngram in source_ngrams {
            // Check if this is our traced n-gram
            if self.is_traced_ngram(&source_ngram.text_content.cleaned_text) {
                info!("TRACE [sequential]: Processing traced source n-gram: '{}' (seq: {})", 
                    source_ngram.text_content.cleaned_text, source_ngram.sequence_number);
            }
            
            let matches = self.similarity_calc.find_matching_ngrams(
                source_ngram, target_ngrams, Some(&self.target_storage))?;
            
            // Check if we found matches for our traced n-gram
            if self.is_traced_ngram(&source_ngram.text_content.cleaned_text) {
                info!("TRACE [sequential]: Found {} matches for traced n-gram", matches.len());
                for (i, m) in matches.iter().enumerate().take(5) { // Show first 5 to avoid spam
                    info!("TRACE [sequential]: Match #{} - source pos: {}, target pos: {}, similarity: {:.3}", 
                        i+1, m.source_position, m.target_position, m.similarity);
                }
            }
            
            all_matches.extend(matches);
        }
        
        Ok(all_matches)
    }

    // Parallel processing with pipeline support
    // In matcher.rs, modify find_ngram_matches_parallel
    fn find_ngram_matches_parallel(
        &self,
        source_ngrams: &[NGramEntry],
        _target_ngrams: &[NGramEntry]
    ) -> Result<Vec<NGramMatch>> {
        // Count traced ngrams before processing
        let traced_sources: Vec<&NGramEntry> = source_ngrams.iter()
            .filter(|ng| self.is_traced_ngram(&ng.text_content.cleaned_text))
            .collect();
            
        if !traced_sources.is_empty() {
            info!("TRACE [parallel]: Found {} source n-grams containing traced text", traced_sources.len());
            for (i, ng) in traced_sources.iter().enumerate() {
                info!("TRACE [parallel]: Source traced n-gram #{}: '{}' (sequence: {})", 
                    i+1, ng.text_content.cleaned_text, ng.sequence_number);
            }
        }

        // Create the parallel matcher with target storage
        let parallel_matcher = self.create_parallel_matcher()?;
        
        // Determine whether to use pipeline or standard parallel processing
        let result = if self.should_use_pipeline(source_ngrams) {
            info!("TRACE [parallel]: Using pipeline processing strategy for {} source ngrams", source_ngrams.len());
            
            // Create and configure the pipeline
            let pipeline = self.create_optimized_pipeline(source_ngrams.len());
            
            // Execute using the pipeline
            pipeline.process(source_ngrams, &parallel_matcher, &self.target_storage)
        } else {
            info!("TRACE [parallel]: Using standard parallel processing for {} source ngrams", source_ngrams.len());
            
            // Execute using standard parallel processing
            parallel_matcher.find_matches_parallel(source_ngrams, &self.target_storage)
        };
        
        // Add trace for matches found
        if let Ok(ref matches) = result {
            let traced_matches: Vec<&NGramMatch> = matches.iter()
                .filter(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text))
                .collect();
                
            if !traced_matches.is_empty() {
                info!("TRACE [parallel]: Found {} matches for traced n-gram", traced_matches.len());
                for (i, m) in traced_matches.iter().enumerate().take(5) { // Show first 5 to avoid spam
                    info!("TRACE [parallel]: Match #{} - source pos: {}, target pos: {}, similarity: {:.3}", 
                        i+1, m.source_position, m.target_position, m.similarity);
                }
            } else if !traced_sources.is_empty() {
                trace!("TRACE [parallel]: NO MATCHES found for traced n-gram despite having source candidates");
            }
        }
        
        result
    }
    
    // Create a parallel matcher instance
    fn create_parallel_matcher(&self) -> Result<ParallelMatcher<'_>> {
        ParallelMatcher::new(
            self.matcher_config.clone(),
            self.processor_config.clone(),
            Some(&self.target_storage)
        )
    }
    
    // Determine whether parallel processing should be used
    fn should_use_parallel(&self, target_ngrams: &[NGramEntry]) -> bool {
        if target_ngrams.is_empty() {
            info!("Using prefix-based matching with storage");
            self.processor_config.parallel_ngram_matching
        } else {
            // Original size-based decision for full target array
            let total_size = target_ngrams.len() * std::mem::size_of::<NGramEntry>();
            self.processor_config.parallel_ngram_matching && 
            total_size >= self.processor_config.parallel_min_file_size
        }
    }
    
    // Determine whether to use pipeline or standard parallel
    fn should_use_pipeline(&self, source_ngrams: &[NGramEntry]) -> bool {
        // If pipeline processing is disabled, always return false
        if !self.processor_config.pipeline_processing {
            return false;
        }
        
        // Get system information
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize * 1024, // Convert KB to bytes
            Err(_) => {
                warn!("Cannot determine available memory, disabling pipeline processing");
                return false;
            }
        };
        
        // Calculate workload characteristics
        let ngram_count = source_ngrams.len();
        let total_text_size: usize = source_ngrams.iter()
            .take(100) // Sample to avoid excessive computation
            .map(|ng| ng.text_content.cleaned_text.len())
            .sum::<usize>() * source_ngrams.len() / 100;
        
        // Use the processor config's built-in method for deciding on pipeline usage
        let should_use_pipeline = self.processor_config.should_use_pipeline(
            ngram_count, 
            total_text_size,
            available_memory
        );
        
        // Log the decision factors
        info!("Pipeline decision: {} (ngrams: {}, threshold: {})",
             if should_use_pipeline { "yes" } else { "no" },
             ngram_count,
             self.processor_config.pipeline_ngram_threshold);
        
        should_use_pipeline
    }
    
    // Create an optimally configured pipeline
    fn create_optimized_pipeline(&self, source_size: usize) -> ProcessingPipeline {
        // Get system resources
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize * 1024, // Convert KB to bytes
            Err(_) => 1024 * 1024 * 1024, // Default to 1GB if can't determine
        };
        
        // Calculate optimal settings based on source size and available memory
        let chunk_size = if source_size < 10_000 {
            self.processor_config.min_pipeline_batch_size 
        } else if source_size < 100_000 {
            // For medium datasets, use a value between min and max
            let mid_size = (self.processor_config.min_pipeline_batch_size + 
                             self.processor_config.max_pipeline_batch_size) / 2;
            mid_size
        } else {
            // For large datasets, use max size
            self.processor_config.max_pipeline_batch_size
        };
        
        // Adjust result capacity based on available memory
        let result_capacity = if available_memory > 8 * 1024 * 1024 * 1024 { // Over 8GB
            self.processor_config.pipeline_queue_capacity * 2000
        } else {
            self.processor_config.pipeline_queue_capacity * 1000
        };
        
        // Start with basic pipeline settings
        let mut pipeline = ProcessingPipeline::new(
            chunk_size,
            result_capacity,
            self.processor_config.parallel_thread_count
        );
        
        // Configure storage access strategy
        match self.processor_config.storage_access_strategy.as_str() {
            "pooled" => {
                pipeline = pipeline.with_storage_strategy(
                    StorageAccessStrategy::Pooled(self.processor_config.storage_pool_size)
                );
            },
            "per_worker" => {
                pipeline = pipeline.with_storage_strategy(
                    StorageAccessStrategy::PerWorker
                );
            },
            _ => {
                // Default to shared
                pipeline = pipeline.with_storage_strategy(
                    StorageAccessStrategy::Shared
                );
            }
        }
        
        // Check if memory_checks should be enabled
        let memory_checks_enabled = self.processor_config.force_sequential_memory_threshold > 0.0;
        
        if !memory_checks_enabled {
            pipeline = pipeline.without_memory_checks();
        }
        
        pipeline
    }
    // src/matcher/cluster/matcher.rs

    pub fn find_matches_with_rayon(
        &self,
        source_range: Range<u64>,
        target_range: Range<u64>,
        output_path: Option<&Path>,
        progress_bar: Option<&ProgressBar>,
        early_escape_threshold: f64  // New parameter
    ) -> Result<usize> {
        let start_time = Instant::now();
        info!("Starting Rayon-based match finding process");
        info!("Processing source range: {} to {}", source_range.start, source_range.end);
        info!("Processing target range: {} to {}", target_range.start, target_range.end);
        
        // Create a unique temporary directory for this processing run
        let temp_dir = PathBuf::from("temp/matches")
            .join(format!("run_{}", SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()));
                
        // Add debugging
        info!("DEBUG: Creating temporary directory at: {:?}", temp_dir);
        
        // Create the TempFileManager
        let temp_manager = match TempFileManager::new(&temp_dir, "matches") {
            Ok(manager) => manager,
            Err(e) => return Err(Error::Io(e)),
        };
    
        // Wrap everything in a closure to ensure cleanup happens regardless of success/failure
        let result = (|| {
            // Initialize writer outside the parallel processing
            let mut writer = if let Some(path) = output_path {
                Some(std::io::BufWriter::new(File::create(path)?))
            } else {
                None
            };
            
            // Initialize memory monitor
            let memory_monitor = Arc::new(MemoryMonitor::new(
                self.processor_config.memory_warn_threshold_mb, 
                self.processor_config.memory_critical_threshold_mb,
                100 // Check interval in milliseconds
            ));
            
            // Get target ngrams
            info!("Retrieving target ngrams from range {}..{}", 
                target_range.start, target_range.end);
            let target_ngrams = Arc::new(self.target_storage.get_sequence_range(target_range.clone())?);
            info!("Retrieved {} target ngrams", target_ngrams.len());
            
            // Configure Rayon thread pool if specified in config
            let rayon_pool = if self.processor_config.parallel_thread_count > 0 {
                Some(rayon::ThreadPoolBuilder::new()
                    .num_threads(self.processor_config.parallel_thread_count)
                    .build()?)
            } else {
                None
            };
            
            // Create dynamic chunk sizes based on source range
            let source_size = (source_range.end - source_range.start) as usize;
            let base_chunk_size = self.calculate_optimal_chunk_size(source_size);
            info!("Using base chunk size of {} ngrams", base_chunk_size);
            
            // Create adaptive chunk sizes for work stealing
            // Smaller chunks at the beginning and end, larger in the middle
            let mut chunk_ranges = Vec::new();
            let min_chunk_size = base_chunk_size / 2;  // Changed from divide by 4
            let max_chunk_size = base_chunk_size * 2;  // Keep the same

            let mut current_pos = source_range.start;
            while current_pos < source_range.end {
                // Calculate dynamic chunk size based on position
                let progress = (current_pos - source_range.start) as f64 / 
                            (source_range.end - source_range.start) as f64;
                
                // Use a smoother, less extreme parabolic function
                // This gives more consistent chunk sizes
                let chunk_size_factor = 0.8 + 0.4 * (1.0 - (2.0 * progress - 1.0).powi(2)); // Changed formula
                let chunk_size = (min_chunk_size as f64 + 
                                ((max_chunk_size - min_chunk_size) as f64 * chunk_size_factor)) as u64;
                
                let end_pos = (current_pos + chunk_size).min(source_range.end);
                chunk_ranges.push(current_pos..end_pos);
                current_pos = end_pos;
            }
            
            let total_chunks = chunk_ranges.len();
            info!("Created {} work items with adaptive chunk sizes", total_chunks);
            
            // Use a thread-safe collection to store temporary file paths
            let temp_file_paths = Arc::new(Mutex::new(Vec::new()));
            
            // Track progress with atomic counter
            let processed_counter = Arc::new(AtomicUsize::new(0));
            
            // Initialize progress bar if provided
            if let Some(pb) = progress_bar {
                pb.set_length(total_chunks as u64);
                pb.set_position(0);
                pb.set_message("Starting adaptive chunk processing...");
            }
            
            // Flag for early escape detection
            let early_escape_flag = Arc::new(AtomicUsize::new(0)); // 0=no escape, 1=escape
            let first_chunk_processed = Arc::new(AtomicUsize::new(0)); // Track if first chunk processed
            
            // Processing function that will be run in parallel
            let process_chunk = {
                let matcher = self; // Needed for method access
                let target_ngrams = Arc::clone(&target_ngrams);
                let memory_monitor = Arc::clone(&memory_monitor);
                let temp_file_paths = Arc::clone(&temp_file_paths);
                let processed_counter = Arc::clone(&processed_counter);
                let progress_bar = progress_bar.cloned();
                let temp_manager = Arc::new(temp_manager.clone());
                let early_escape_flag = Arc::clone(&early_escape_flag);
                let first_chunk_processed = Arc::clone(&first_chunk_processed);
                let output_path = output_path.map(|p| p.to_path_buf());
                
                move |chunk_range: Range<u64>, chunk_index: usize| -> Result<()> {
                    // Skip if early escape has been triggered
                    if early_escape_flag.load(Ordering::SeqCst) == 1 {
                        trace!("Skipping chunk {} due to early escape flag", chunk_index);
                        return Ok(());
                    }
    
                    // Check memory pressure occasionally
                    if memory_monitor.should_check() {
                        if let Some(pressure) = memory_monitor.check_memory() {
                            match pressure {
                                MemoryPressure::Warning => {
                                    warn!("Memory pressure detected: {} MB", 
                                        memory_monitor.get_memory_usage_mb());
                                    // Brief pause to allow memory reclamation
                                    std::thread::sleep(Duration::from_millis(50));
                                },
                                MemoryPressure::Critical => {
                                    warn!("Critical memory pressure: {} MB", 
                                        memory_monitor.get_memory_usage_mb());
                                    // Longer pause
                                    std::thread::sleep(Duration::from_millis(200));
                                }
                            }
                        }
                    }
                    
                    let chunk_start_time = Instant::now();
                    let chunk_size = (chunk_range.end - chunk_range.start) as usize;
                    
                    // Process this chunk
                    let source_chunk = matcher.source_storage.get_sequence_range(chunk_range.clone())?;
                    
                    // Find matches
                    let ngram_matches = matcher.find_ngram_matches(&source_chunk, &target_ngrams)?;
                    info!("Chunk {}: Found {} initial ngram matches", chunk_index, ngram_matches.len());
                    
                    // Check if this is the first chunk to be processed
                    let was_first = first_chunk_processed.fetch_add(1, Ordering::SeqCst) == 0;
                    
                    if was_first && !source_chunk.is_empty() {
                        // Calculate match rate for early escape check
                        let unique_matched_source_ngrams: std::collections::HashSet<_> = ngram_matches.iter()
                            .map(|m| m.source_position)
                            .collect();
                        
                        let match_rate = unique_matched_source_ngrams.len() as f64 / source_chunk.len() as f64;
                        
                        info!("First chunk match rate: {:.2}% ({} of {} source ngrams have matches)",
                             match_rate * 100.0, unique_matched_source_ngrams.len(), source_chunk.len());
                        
                        // If match rate exceeds threshold, mark for early escape
                        if match_rate >= early_escape_threshold {
                            info!("EARLY ESCAPE: Match rate {:.2}% exceeds threshold {:.2}%. These databases appear to be duplicates or very similar.",
                                 match_rate * 100.0, early_escape_threshold * 100.0);
                            
                            // Update progress bar if provided
                            if let Some(ref pb) = progress_bar {
                                pb.set_message(format!(
                                    "Early termination: {:.2}% match rate indicates duplicate databases",
                                    match_rate * 100.0
                                ));
                            }
                            
                            // Create an empty output file to mark that we processed this pair
                            if let Some(path) = &output_path {
                                if let Ok(mut file) = File::create(path) {
                                    let _ = writeln!(file, "{{\"early_escape\": true, \"match_rate\": {:.4}, \"threshold\": {:.4}}}",
                                                   match_rate, early_escape_threshold);
                                }
                            }
                            
                            // Set flag to skip remaining chunks
                            early_escape_flag.store(1, Ordering::SeqCst);
                            return Ok(());
                        }
                    }
                    
                    if !ngram_matches.is_empty() {
                        // Find dense regions
                        let dense_candidates = if matcher.processor_config.parallel_dense_regions && 
                                            ngram_matches.len() >= matcher.processor_config.parallel_dense_threshold {
                            matcher.find_dense_regions_parallel(&ngram_matches)?
                        } else {
                            matcher.find_dense_regions_chunk(&ngram_matches, None)?
                        };
                        
                        info!("Chunk {}: Found {} dense regions", chunk_index, dense_candidates.len());
                        
                        if !dense_candidates.is_empty() {
                            // Validate matches
                            let valid_matches = matcher.validate_without_merging(
                                dense_candidates, &source_chunk, &target_ngrams)?;
                            
                            let valid_count = valid_matches.len();
                            info!("Chunk {}: Found {} valid matches", chunk_index, valid_count);
                            
                            // Write to temp file instead of sending to results collector
                            if !valid_matches.is_empty() {
                                let temp_file_path = temp_manager.write_matches(&valid_matches, chunk_index)?;
                                
                                info!("Chunk {}: Wrote {} matches to temporary file {:?}",
                                     chunk_index, valid_count, temp_file_path);
                                
                                // Add debugging
                                info!("DEBUG: Added temp file to paths list: {:?}", temp_file_path);
                                
                                // Save file path to our shared collection
                                if let Ok(mut paths) = temp_file_paths.lock() {
                                    paths.push(temp_file_path);
                                    info!("DEBUG: Current temp files count: {}", paths.len());
                                } else {
                                    warn!("Failed to lock temp_file_paths mutex");
                                }
                            }
                        }
                    }
                    
                    // Update progress
                    let processed = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    if let Some(ref pb) = progress_bar {
                        pb.set_position(processed as u64);
                        
                        // Use chunk_index in your message
                        let message = format!(
                            "Processed chunk #{} ({}/{} total) - {} ngrams in {:?}",
                            chunk_index + 1,
                            processed,
                            total_chunks,
                            chunk_size,
                            chunk_start_time.elapsed()
                        );
                        pb.set_message(message);
                    }
                    
                    Ok(())
                }
            };
            
            // Execute in parallel using Rayon
            if let Some(pool) = rayon_pool {
                // Use custom Rayon pool
                pool.install(|| {
                    chunk_ranges.par_iter().enumerate().for_each(|(i, range)| {
                        if let Err(e) = process_chunk(range.clone(), i) {
                            error!("Error processing chunk {}: {}", i, e);
                        }
                        // Clean up thread-local cache every few chunks
                        if i % 5 == 0 {
                            METADATA_CACHE.trim_thread_local_caches(Duration::from_secs(120));
                        }
                    });
                });
            } else {
                // Use default Rayon thread pool
                chunk_ranges.par_iter().enumerate().for_each(|(i, range)| {
                    if let Err(e) = process_chunk(range.clone(), i) {
                        error!("Error processing chunk {}: {}", i, e);
                    }
                    // Clean up thread-local cache every few chunks
                    if i % 5 == 0 {
                        METADATA_CACHE.trim_thread_local_caches(Duration::from_secs(120));
                    }
                });
            }
            
            // Check if early escape was triggered
            if early_escape_flag.load(Ordering::SeqCst) == 1 {
                info!("Finishing early due to high similarity detected in first chunk");
                return Ok(usize::MAX); // Special value to indicate early escape
            }
            
            // Collect all temporary file paths
            let file_paths = if let Ok(paths) = temp_file_paths.lock() {
                paths.clone()
            } else {
                warn!("Failed to lock temp_file_paths mutex when retrieving paths");
                Vec::new()
            };
            
            info!("All processing threads completed, collected {} temporary files", file_paths.len());
            
            // Debug: verify all temp files
            if !file_paths.is_empty() {
                info!("DEBUG: About to merge the following temp files:");
                for (i, path) in file_paths.iter().enumerate() {
                    info!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, path.exists());
                }
            }
            
            // Process final merging with our streaming approach
            if !file_paths.is_empty() {
                let merging_start_time = Instant::now();
                info!("Processing final merging for {} matches from all chunks", file_paths.len());
                
                // Use our new streaming merge method with the correct writer reference
                let total_match_count = self.stream_merge_from_temp_files(&file_paths, &temp_manager, writer.as_mut())?;
                
                info!("Final merging complete: {} matches after merging (took {:?})", 
                    total_match_count, merging_start_time.elapsed());
                info!("Completed all chunks, found {} total matches in {:?}", 
                    total_match_count, start_time.elapsed());
                
                Ok(total_match_count)
            } else {
                info!("No matches found in any chunks");
                Ok(0)
            }
        })();
        
        // Always clean up temporary files, regardless of processing outcome
        // Comment out for debugging if you want to see the temp files
        if let Err(e) = temp_manager.cleanup() {
            warn!("Error cleaning up temporary files: {}", e);
        }
        
        // Return the result
        result
    }

    fn find_dense_regions_parallel(&self, matches: &[NGramMatch]) -> Result<Vec<MatchCandidate>> {
        // Reuse the existing parallel implementation
        super::parallel_chains::find_dense_regions_parallel(self, matches)
    }

    fn calculate_optimal_chunk_size(&self, source_size: usize) -> u64 {
        // Get system information
        let cpu_count = num_cpus::get();
        
        // Check available memory - CRITICAL ADDITION
        let memory_pressure = match sys_info::mem_info() {
            Ok(info) => 1.0 - (info.avail as f64 / info.total as f64),
            Err(_) => 0.5, // Assume 50% if we can't determine
        };
        
        // Use source_size to adjust the chunk size - BUT MUCH SMALLER VALUES
        let base_size = match source_size {
            s if s < 10_000 => 500,          // Reduced from original 1_000
            s if s < 100_000 => 1_000,       // Reduced from original 5_000
            s if s < 1_000_000 => 2_000,     // Reduced from original 10_000
            _ => 4_000                        // Reduced from original 20_000
        };
        
        // Apply memory pressure adjustment - NEW STEP
        // Scale down chunk size based on memory pressure
        let memory_adjusted = if memory_pressure > 0.9 {
            // Critical memory pressure - tiny chunks
            base_size / 4
        } else if memory_pressure > 0.8 {
            // High memory pressure - small chunks
            base_size / 2
        } else {
            // Normal memory pressure
            base_size
        };
        
        // Adjust for available CPUs
        let cpu_adjusted = if cpu_count <= 4 {
            memory_adjusted
        } else if cpu_count <= 16 {
            // More parallelism with smaller chunks
            (memory_adjusted / 3).max(250)
        } else {
            // Much more parallelism with even smaller chunks
            (memory_adjusted / 4).max(200)
        };
        
        // Keep within much smaller reasonable bounds
        cpu_adjusted.clamp(200, 2_000) as u64  // Drastically reduced from 1_000-25_000
    }

    fn load_terms_from_file(filename: &str) -> Result<AHashSet<String>> {
        let file = match File::open(filename) {
            Ok(file) => file,
            Err(e) => return Err(Error::Io(e))
        };
        
        let reader = BufReader::new(file);
        let mut terms = AHashSet::new();
        
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            
            // Skip empty lines and comments
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                terms.insert(trimmed.to_string());
            }
        }
        
        info!("Loaded {} terms from {}", terms.len(), filename);
        Ok(terms)
    }
    pub(super) fn is_early_isnad(&self, text: &str) -> (bool, f64, usize) {
        // Skip empty strings
        if text.is_empty() {
            return (false, 0.0, 0);
        }
    
        // Check if this contains our traced text
        let has_traced = self.is_traced_ngram(text);
        if has_traced {
            info!("TRACE [isnad]: Checking traced text: '{}'", text);
        }
    
        // Get all words in the text
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        
        if word_count == 0 {
            return (false, 0.0, 0);
        }
    
        // Track which words are covered by phrases or individual matches
        let mut covered_positions = vec![false; word_count];
        let mut matching_words = 0;
        let mut matched_items = Vec::new();
        
        // Check for isnad phrases
        for phrase in &self.isnad_phrases {
            if text.contains(phrase) {
                if has_traced {
                    info!("TRACE [isnad]: Found isnad phrase: '{}'", phrase);
                    matched_items.push(format!("phrase: '{}'", phrase));
                }
                
                // Find where this phrase occurs in the word sequence
                let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                
                // For each potential starting position in the text
                for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                    let mut phrase_matches = true;
                    
                    // Check if phrase matches at this position
                    for (offset, phrase_word) in phrase_words.iter().enumerate() {
                        if start_pos + offset >= word_count || words[start_pos + offset] != *phrase_word {
                            phrase_matches = false;
                            break;
                        }
                    }
                    
                    // If phrase matches, mark all its positions as covered
                    if phrase_matches {
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
        
        // Check for individual isnad words
        for (pos, word) in words.iter().enumerate() {
            if !covered_positions[pos] && self.isnad_words.contains(&word.to_string()) {
                covered_positions[pos] = true;
                matching_words += 1;
                
                if has_traced {
                    matched_items.push(format!("word: '{}'", word));
                }
            }
        }
        
        // Calculate the actual density
        let density = matching_words as f64 / word_count as f64;
        
        if has_traced {
            info!("TRACE [isnad]: Word-level analysis: {}/{} matching words, density: {:.2}", 
                 matching_words, word_count, density);
            if !matched_items.is_empty() {
                info!("TRACE [isnad]: Matching isnad terms: {:?}", matched_items);
            }
        }
        
        // Use threshold for detection
        let is_isnad = density >= self.isnad_density_threshold;
        
        if has_traced {
            if is_isnad {
                info!("TRACE [isnad]: Detected as isnad with density {:.2} (threshold: {:.2})",
                     density, self.isnad_density_threshold);
            } else {
                info!("TRACE [isnad]: NOT an isnad - density {:.2} below threshold {:.2}",
                     density, self.isnad_density_threshold);
            }
        }
        
        // Only discard short isnads (below min_length threshold)
        let should_discard = is_isnad && word_count < self.isnad_min_length;
        
        if is_isnad && has_traced {
            if should_discard {
                info!("TRACE [isnad]: REJECTING as short isnad ({} words < min threshold {})",
                    word_count, self.isnad_min_length);
            } else {
                info!("TRACE [isnad]: KEEPING as longer isnad ({} words >= min threshold {})",
                    word_count, self.isnad_min_length);
            }
        }
        
        (should_discard, density, word_count)
    }
    /// Lightweight check for likely banality that can be used during early stages of matching
    /// Returns (is_banal, density) - simplified version of is_likely_banality
    pub(super) fn is_early_banal(&self, text: &str) -> (bool, f64) {
        // Skip empty strings
        if text.is_empty() {
            return (false, 0.0);
        }
        
        // Check if this contains our traced text
        let has_traced = self.is_traced_ngram(text);
        
        // If we're in automatic mode, skip early banality detection completely
        if self.banality_detection_mode == "automatic" {
            if has_traced {
                info!("TRACE [early_banal]: Skipping early banality check for traced text - using automatic mode");
            }
            return (false, 0.0);
        }
        
        // Split text into words
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        
        if word_count == 0 {
            return (false, 0.0);
        }
        
        // Count matched words and phrases
        let mut matching_words = 0;
        let mut matched_items: Vec<String> = Vec::new();
        
        // Check phrases - just count them but don't try precise positioning
        for phrase in &self.banality_phrases {
            if text.contains(phrase) {
                // Estimate word count in phrase
                let phrase_words = phrase.split_whitespace().count();
                matching_words += phrase_words;
                
                if has_traced {
                    matched_items.push(format!("phrase: '{}'", phrase));
                }
            }
        }
        
        // Check individual words
        for word in &words {
            // Check if this word is in the banality_words
            if self.banality_words.contains(&word.to_string()) {
                matching_words += 1;
                
                if has_traced {
                    matched_items.push(format!("word: '{}'", word));
                }
            }
        }
        
        // Avoid overcounting
        matching_words = matching_words.min(word_count);
        
        let density = matching_words as f64 / word_count as f64;
        
        // Use slightly lower threshold for early detection
        let early_threshold = (self.banality_density_threshold * 0.9).max(0.5);
        
        if has_traced {
            if density >= early_threshold {
                info!("TRACE [early_banal]: Traced text banal with density {:.2} ({}/{} words)",
                     density, matching_words, word_count);
                if !matched_items.is_empty() {
                    info!("TRACE [early_banal]: Banal items: {:?}", matched_items);
                }
            } else {
                info!("TRACE [early_banal]: Traced text NOT banal - density {:.2} (threshold: {:.2})",
                     density, early_threshold);
                if !matched_items.is_empty() {
                    info!("TRACE [early_banal]: Has banal items but below threshold: {:?}", matched_items);
                }
            }
        }
        
        (density >= early_threshold, density)
    }
    pub(super) fn check_automatic_banality(&self, matches: &[NGramMatch]) -> (bool, f64) {
        if matches.is_empty() || self.banality_detection_mode != "automatic" {
            return (false, 0.0);
        }
        
        if let Some(ref common_ngrams) = self.common_ngrams {
            // Extract n-gram IDs directly from the matches
            let ngram_ids: Vec<u64> = matches.iter()
                .map(|m| m.ngram.sequence_number)
                .collect();
            
            // Check if any matches contain our traced n-gram
            let has_traced = matches.iter()
                .any(|m| self.is_traced_ngram(&m.ngram.text_content.cleaned_text));
            
            // Use the common n-grams to check if this is banal
            let (is_banal, density) = common_ngrams.is_likely_banal(&ngram_ids);
            
            if has_traced {
                if is_banal {
                    info!("TRACE [automatic_banality]: Traced matches detected as banal with density {:.2}%",
                         density);
                } else {
                    info!("TRACE [automatic_banality]: Traced matches NOT banal - density {:.2}% (threshold: {:.1}%)",
                         density, common_ngrams.threshold());
                }
            }
            
            return (is_banal, density);
        }
        
        (false, 0.0)
    }

    /// Check if the given text contains our target n-gram
    pub fn is_traced_ngram(&self, text: &str) -> bool {
        // Skip if traced_phrase is empty
        if self.matcher_config.traced_phrase.is_empty() {
            return false;
        }
        
        text.contains(&self.matcher_config.traced_phrase)
    }

    /// Logs information about a traced n-gram
    fn log_traced_ngram(&self, location: &str, text: &str, extra_info: &str) {
        if self.is_traced_ngram(text) {
            info!("TRACE [{}]: Found traced n-gram in text: '{}'", location, text);
            info!("TRACE [{}]: {}", location, extra_info);
        }
    }

    /// Performs streaming merging of matches from temporary files
    ///
    /// This method reads matches from temporary files in batches, performs merging operations,
    /// and writes the results directly to the output file. After processing, it deletes the
    /// temporary directory that housed the files.
    ///
    /// # Arguments
    /// * `temp_file_paths` - Paths to the temporary files containing matches
    /// * `temp_manager` - The temporary file manager
    /// * `writer` - Optional writer to the output file
    ///
    /// # Returns
    /// The total number of merged matches
    fn stream_merge_from_temp_files(
        &self,
        temp_file_paths: &[PathBuf],
        temp_manager: &TempFileManager,
        mut writer: Option<&mut std::io::BufWriter<File>>
    ) -> Result<usize> {
        info!("Starting streaming merge from {} temporary files", temp_file_paths.len());
        info!("DEBUG: Temp directory: {:?}", temp_manager.get_temp_dir());
        info!("DEBUG: Verifying all temp files exist:");
        let mut missing_files = 0;
        for (i, path) in temp_file_paths.iter().enumerate() {
            let exists = path.exists();
            info!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, exists);
            if !exists {
                missing_files += 1;
            }
        }
        info!("DEBUG: Missing files: {}/{}", missing_files, temp_file_paths.len());
        
        // If no temporary files, return early
        if temp_file_paths.is_empty() {
            return Ok(0);
        }
        
        // Memory-adaptive batch processing constants
        // Use smaller values during high memory pressure
        let memory_state = MemoryManager::get_memory_usage_percent().unwrap_or(50.0);
        let is_memory_constrained = memory_state > 70.0;
        
        // Define batch sizes based on memory pressure
        let batch_size = if is_memory_constrained {
            // Smaller batch size under memory pressure
            500
        } else {
            1000
        };
        
        let merge_buffer_size = if is_memory_constrained {
            // Smaller buffer size under memory pressure
            2000
        } else {
            5000
        };
        
        info!("Using memory-adaptive settings: batch_size={}, buffer_size={} (memory usage: {:.1}%)",
             batch_size, merge_buffer_size, memory_state);
        
        // Initialize counters
        let mut total_read: usize = 0;
        let mut total_written: usize = 0;
        let mut batch_count: usize = 0;
        
        // Create a buffer for accumulating matches for merging
        let mut merge_buffer: Vec<MatchResult> = Vec::with_capacity(merge_buffer_size);
        
        // Process each temporary file
        for (file_index, file_path) in temp_file_paths.iter().enumerate() {
            info!("Processing temporary file {}/{}: {:?}", 
                file_index + 1, temp_file_paths.len(), file_path);
            
            // Read matches from the file in smaller chunks to manage memory
            let file = match File::open(file_path) {
                Ok(f) => f,
                Err(e) => {
                    warn!("Error opening temp file {}: {}", file_path.display(), e);
                    continue;
                }
            };
            
            let reader = std::io::BufReader::new(file);
            let mut file_matches_count = 0;
            
            // Process the file line by line to avoid loading entire file
            for line in reader.lines() {
                match line {
                    Ok(line_text) => {
                        match serde_json::from_str::<MatchResult>(&line_text) {
                            Ok(match_result) => {
                                // Add to merge buffer
                                merge_buffer.push(match_result);
                                file_matches_count += 1;
                                
                                // If buffer reaches threshold, process it
                                if merge_buffer.len() >= merge_buffer_size {
                                    // Merge and write current buffer
                                    let merged_count = self.merge_and_write_buffer(&mut merge_buffer, &mut writer)?;
                                    total_written += merged_count;
                                    batch_count += 1;
                                    
                                    // Log progress
                                    debug!("Processed batch {} with {} matches from buffer, wrote {} merged matches", 
                                        batch_count, merge_buffer_size, merged_count);
                                    
                                    // After processing a large batch, force some cleanup
                                    drop(Vec::<u8>::with_capacity(1024 * 1024 * 10)); // 10MB
                                    
                                    // Clear buffer for next batch
                                    merge_buffer.clear();
                                    merge_buffer.shrink_to_fit(); // Release capacity back to system
                                    
                                    // Short pause to allow system to reclaim memory
                                    std::thread::sleep(std::time::Duration::from_millis(10));
                                }
                            },
                            Err(e) => {
                                warn!("Error parsing match from line: {}", e);
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Error reading line from file: {}", e);
                    }
                }
            }
            
            total_read += file_matches_count;
            info!("Read {} matches from file {}", file_matches_count, file_index + 1);
            
            // Check if we need to apply memory backpressure
            if let Some(memory_usage) = MemoryManager::get_memory_usage_percent() {
                if memory_usage > 80.0 {
                    info!("High memory usage ({:.1}%) after processing file {}, pausing for memory reclamation", 
                         memory_usage, file_index + 1);
                    
                    // Force garbage collection
                    drop(Vec::<u8>::with_capacity(1024 * 1024 * 20)); // 20MB
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
        
        // Process any remaining matches in the buffer
        if !merge_buffer.is_empty() {
            debug!("Processing final merge buffer with {} matches", merge_buffer.len());
            let merged_count = self.merge_and_write_buffer(&mut merge_buffer, &mut writer)?;
            total_written += merged_count;
        }
        
        // Ensure writer is flushed
        if let Some(w) = writer.as_mut() {
            if let Err(e) = w.flush() {
                warn!("Error flushing output file: {}", e);
            }
        }
        
        // Get the temp directory for manual cleanup
        let temp_dir = temp_manager.get_temp_dir().to_path_buf();
        
        // First, call the TempFileManager's cleanup to delete the individual files
        if let Err(e) = temp_manager.cleanup() {
            warn!("Error during temp file cleanup: {}", e);
        } else {
            info!("Temporary files successfully cleaned up");
        }
        
        // Now, delete the entire temporary directory
        if temp_dir.exists() {
            info!("Removing temporary directory: {:?}", temp_dir);
            match std::fs::remove_dir_all(&temp_dir) {
                Ok(_) => info!("Successfully removed temporary directory"),
                Err(e) => warn!("Failed to remove temporary directory: {}", e),
            }
        }
        
        info!("Streaming merge complete: read {} matches, wrote {} after merging", 
            total_read, total_written);
        
        Ok(total_written)
    }

/// Helper method to merge a buffer of matches and write the results
///
/// # Arguments
/// * `buffer` - Buffer of matches to merge
/// * `writer` - Optional writer to the output file
///
/// # Returns
/// The number of merged matches written
fn merge_and_write_buffer(
    &self,
    buffer: &mut Vec<MatchResult>,
    writer: &mut Option<&mut std::io::BufWriter<File>>
) -> Result<usize> {
    // Early check for empty buffer
    if buffer.is_empty() {
        return Ok(0);
    }
    
    debug!("Merging buffer with {} matches", buffer.len());
    
    // Perform the merging operations using existing logic
    // Sort matches by source range start for consistent processing
    buffer.sort_by(|a, b| a.source_range.start.cmp(&b.source_range.start));
    
    // Create an R-tree for spatial indexing
    debug!("Building R-tree spatial index for {} matches...", buffer.len());
    let rtree = super::spatial::build_rtree(&buffer);
    
    // Merge overlapping matches using R-tree
    let mut merged_results = Vec::new();
    let mut processed = AHashSet::new();
    
    for (i, match1) in buffer.iter().enumerate() {
        // Skip if already processed
        if processed.contains(&i) {
            continue;
        }
    
        let mut current_match = match1.clone();
        processed.insert(i);
        
        // Find potential overlapping candidates using R-tree
        let candidates = super::spatial::find_overlapping_candidates(&rtree, &current_match);
        
        for &j in &candidates {
            if i == j || processed.contains(&j) {
                continue;
            }
    
            let match2 = &buffer[j];
            
            // Check if these matches should be merged
            if self.should_merge_matches(&current_match, match2)? {
                // Merge the matches
                let merged_match = self.merge_matches(&current_match, match2)?;
                current_match = merged_match;
                processed.insert(j);
            }
        }
    
        merged_results.push(current_match);
    }
    
    debug!("After overlapping merging: {} matches (from {} original)", 
         merged_results.len(), buffer.len());
    
    // Remove contained matches
    self.remove_contained_matches(&mut merged_results);
    debug!("After removing contained matches: {} matches", merged_results.len());
    
    // Merge adjacent, non-overlapping matches
    self.merge_adjacent_matches_with_rtree(&mut merged_results)?;
    debug!("After merging adjacent matches: {} matches", merged_results.len());
    
    // Apply final filters and mark isnads
    let mut filtered_matches = Vec::new();
    
    for match_result in merged_results {
        // Apply all final filters
        if !self.passes_final_filters(&match_result) {
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
    }
    
    // Write the filtered matches to the output file if provided
    if let Some(w) = writer.as_mut() {
        for match_result in &filtered_matches {
            // Convert to SimpleMatchResult for JSON output
            let simple = SimpleMatchResult::from(match_result.clone());
            
            // Write to file
            writeln!(w, "{}", serde_json::to_string(&simple)?)
                .map_err(|e| Error::Io(e))?;
        }
        
        // Flush to ensure data is written
        w.flush().map_err(|e| Error::Io(e))?;
    }
    
    let final_count = filtered_matches.len();
    debug!("Wrote {} matches after filtering", final_count);
    
    Ok(final_count)
}
}
