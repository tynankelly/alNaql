// src/matcher/orchestrator.rs
// Main orchestration module that coordinates the entire matching process
// Replaces the ClusteringMatcher from matcher_legacy.rs

// ============================================================================
// STANDARD LIBRARY IMPORTS
// ============================================================================
use ahash::AHashSet;
use dashmap::DashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

// ============================================================================
// EXTERNAL CRATE IMPORTS
// ============================================================================
use log::{debug, info, warn, error, trace};
use serde_json;

// ============================================================================
// ERROR AND CORE TYPES
// ============================================================================
use crate::error::{Error, Result};

// ============================================================================
// CONFIGURATION IMPORTS
// ============================================================================
use crate::config::{
    AlNaqlConfig,
    NGramMatchingStrategy,
    NGramType,
    MatchingConfig,
};

// ============================================================================
// STORAGE IMPORTS
// ============================================================================
use crate::storage::LMDBStorage;
use crate::storage::cache::METADATA_CACHE;
// ============================================================================
// MATCHER MODULE IMPORTS - New Modular Structure
// ============================================================================
// Core matcher types
use crate::matcher::grid_orchestrator;
use crate::utils::resume_manager::ResumeManager;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
// New column-based types
use crate::matcher::types::{
    SequenceMatch,
    ReconstructedSegment,
    ValidatedSegment,
    ColumnBatch,
    BufferedChunkMessage,
    FinalMatchResult,
};
// N-gram matching strategies
use crate::matcher::ngrams_matcher::{
    ngrams_sequential,
    ngrams_parallel,
    ngrams_parallel_opt,
};

// Clustering and validation
use crate::matcher::clusters::{
    chains,
    text_classification::TextClassifier,
    validation,
};
use crate::matcher::clusters::text_classification;
use crate::matcher::buffered_writer::create_buffered_writer;

// ============================================================================
// UTILITY IMPORTS
// ============================================================================
use crate::utils::tempfile::TempFileManager;
use crate::matcher::orchestrator::text_classification::AutoBanal;
use crate::utils::binary_io::{BinaryWriter, DataType};
use crate::utils::progress_manager::{ProgressContext, ProgressPhase, format_duration};
// ============================================================================
// MODULE STRUCTURE
// ============================================================================

/// Shared state for caching and optimization across all processing threads
/// This struct contains all the expensive-to-create shared data that should
/// be reused across chunks and threads
#[derive(Clone)]
pub struct MatcherSharedState {
    /// Cache for frequently accessed sequences and their column data
    pub sequence_cache: Arc<DashMap<u64, ColumnBatch>>,
    
    /// Cache for text segments
    pub text_cache: Arc<DashMap<(u64, u64), String>>,
    
    /// Cache for computed similarity scores to avoid recalculation
    /// Key: (source_text, target_text) tuple
    pub similarity_cache: Arc<DashMap<(String, String), f64>>,
    
    /// Memory pressure flag for coordinating memory management
    pub memory_pressure_flag: Arc<AtomicBool>,
    
    /// Processing statistics shared across all threads
    pub stats: Arc<ProcessingStats>,
}

/// Statistics tracked across the entire matching process
#[derive(Debug, Default)]
pub struct ProcessingStats {
    /// Total number of n-gram comparisons performed
    pub total_comparisons: AtomicUsize,
    
    /// Number of cache hits
    pub cache_hits: AtomicUsize,
    
    /// Number of cache misses  
    pub cache_misses: AtomicUsize,
    
    /// Number of n-grams filtered before comparison
    pub filtered_early: AtomicUsize,
    
    /// Number of memory cleanup cycles triggered
    pub memory_cleanups: AtomicUsize,
    
    /// Number of matches found
    pub matches_found: AtomicUsize, 
    
    /// Number of candidates validated
    pub candidates_validated: AtomicUsize,
}

/// Main orchestrator that coordinates the entire matching process
pub struct MatcherOrchestrator {
    /// Configuration for matching behavior
    pub config: Arc<AlNaqlConfig>,
    /// Source storage reference for source document n-grams
    pub source_storage: Arc<LMDBStorage>,
    /// Target storage reference for target document n-grams
    pub target_storage: Arc<LMDBStorage>,
    /// Shared state for caching and optimization
    pub shared_state: Arc<MatcherSharedState>,
    /// Text classifier for filtering (banality, isnad detection)
    pub text_classifier: Arc<RwLock<TextClassifier>>,
    /// N-gram type being processed
    pub ngram_type: NGramType,
    /// Similarity calculator for text comparison
    pub similarity_calc: SimilarityCalculator,

}

// ============================================================================
// IMPLEMENTATION - CONSTRUCTORS AND BASIC METHODS
// ============================================================================

impl MatcherSharedState {
    /// Create a new shared state with empty caches
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            sequence_cache: Arc::new(DashMap::with_capacity(cache_capacity / 4)),
            text_cache: Arc::new(DashMap::with_capacity(cache_capacity / 2)),
            similarity_cache: Arc::new(DashMap::with_capacity(cache_capacity)),
            memory_pressure_flag: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(ProcessingStats::default()),
        }
    }
    
    /// Clear all caches to free memory
    pub fn clear_caches(&self) {
        self.sequence_cache.clear();
        self.text_cache.clear();
        self.similarity_cache.clear();
        self.stats.memory_cleanups.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current cache sizes for monitoring
    pub fn get_cache_sizes(&self) -> (usize, usize, usize) {
        let ngram_size = self.sequence_cache.len();
        let text_size = self.text_cache.len();
        let sim_size = self.similarity_cache.len();
        (ngram_size, text_size, sim_size)
    }
}

impl MatcherOrchestrator {
    /// Create a new MatcherOrchestrator
    pub fn new(
        source_storage: Arc<LMDBStorage>,
        target_storage: Arc<LMDBStorage>,
        config: Arc<AlNaqlConfig>,
        ngram_type: NGramType,
    ) -> Result<Self> {
        info!("Creating MatcherOrchestrator with {} configuration",
            match ngram_type {
                NGramType::Word => "word",
                NGramType::Character => "character",
            });
        
        // Log the configuration mode
        if config.matcher.banality_detection_mode == "automatic" {
            debug!("SYSTEM: Initializing with AUTOMATIC banality detection mode");
        } else {
            debug!("SYSTEM: Using phrase-based banality detection");
        }
        
        // Create the similarity calculator with the full config and ngram type
        // The ? operator handles any initialization errors
        let similarity_calc = SimilarityCalculator::new(
            Arc::clone(&config),
        )?;
        
        // Determine cache capacity based on configuration
        let cache_capacity = config.matcher.cache_capacity;
        debug!("Initializing shared state with cache capacity: {}", cache_capacity);
        
        let shared_state = Arc::new(MatcherSharedState::new(cache_capacity));
        
        // Create text classifier with the correct arguments
        // Note: This also returns a Result, hence the ? operator
        let text_classifier = TextClassifier::new(&config, None)?;
        
        Ok(Self {
            config: config,
            source_storage,
            target_storage,
            shared_state,
            text_classifier: Arc::new(RwLock::new(text_classifier)),
            ngram_type,
            similarity_calc,
        })
    }
    
}

// ============================================================================
// MAIN ENTRY POINTS - COPIED FROM matcher_legacy.rs
// ============================================================================

impl MatcherOrchestrator {
    /// Sequential match finding - processes source chunks one at a time
    pub fn find_matches(
        &mut self,
        source_range: Range<u64>,
        target_range: Range<u64>,
        output_path: Option<&Path>,
        early_escape_threshold: f64,
        config: &AlNaqlConfig,
        progress_context: Option<ProgressContext>,
    ) -> Result<usize> {
        let start_time = Instant::now();
        info!("Starting match finding process");
        
        // Create a unique temporary directory for this processing run
        let temp_dir = config.files.temp_dir
            .join("matches")
            .join(format!("run_{}", SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()));
        debug!("DEBUG: Creating temporary directory at: {:?}", temp_dir);
        
        // Create the TempFileManager
        let temp_manager = match crate::utils::tempfile::TempFileManager::new(&temp_dir, "matches") {
            Ok(manager) => manager,
            Err(e) => return Err(Error::Io(e)),
        };

        // Wrap everything in a closure to ensure cleanup happens regardless of success/failure
        let result = (|| {
            // Initialize a vector to track temporary file paths instead of storing all matches
            let mut temp_file_paths: Vec<PathBuf> = Vec::new();
            
            // Create output file writer if path provided
            let mut writer = if let Some(path) = output_path {
                Some(BufWriter::new(File::create(path)?))
            } else {
                None
            };
            
            // Initialize discarded content logger if enabled
            if self.config.debug.log_discarded_matches {
                let discarded_path = Path::new(&self.config.debug.discarded_log_path);
                // Create parent directories if they don't exist
                if let Some(parent) = discarded_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                
                if let Err(e) = crate::utils::DISCARDED_LOGGER.init(discarded_path) {
                    warn!("Failed to initialize discarded content logger: {}", e);
                } else {
                    debug!("Discarded content will be logged to {:?}", discarded_path);
                }
            }

            // Process source ngrams in chunks
            let chunk_size = 10_000; 
            let total_chunks = ((source_range.end - source_range.start) as usize + chunk_size - 1) / chunk_size;
                        
            // Track chunk times for ETA calculation
            let mut chunk_times = Vec::with_capacity(total_chunks.min(10));
            
            // Log initial memory state (simplified - we don't have sys_info)
            debug!("Starting chunk processing with {} total chunks", total_chunks);
            
            for chunk_index in 0..total_chunks {
                // Calculate chunk range (unchanged)
                let chunk_start = source_range.start + (chunk_index as u64 * chunk_size as u64);
                let chunk_end = (chunk_start + chunk_size as u64).min(source_range.end);
                let chunk_start_time = Instant::now();
                
                info!("Processing chunk {}/{}: source sequences {}..{}",  // Note: "sequences" not "ngrams"
                    chunk_index + 1, total_chunks, chunk_start, chunk_end);
                
                // Create sequence list - no storage call needed
                let source_chunk: Vec<u64> = (chunk_start..chunk_end).collect();
                
                // Dispatch matching with sequences instead of NGramEntry objects
                let ngram_matches_time = Instant::now();
                let chunk_matches = self.dispatch_ngram_matching(
                    &source_chunk,
                    target_range.clone(),
                    &self.similarity_calc,
                )?;
            
                info!("Found {} initial matches in {:?} (n-gram matching took {:?})", 
                    chunk_matches.len(), chunk_start_time.elapsed(), ngram_matches_time.elapsed());
                
                // Add early escape check for the first chunk
                if chunk_index == 0 && !source_chunk.is_empty() {
                    // Calculate match rate
                    let unique_matched_source_sequences: AHashSet<_> = chunk_matches.iter()
                        .map(|m| m.source_sequence)
                        .collect();
                    
                    let match_rate = unique_matched_source_sequences.len() as f64 / source_chunk.len() as f64;
                    
                    info!("First chunk match rate: {:.2}% ({} of {} source sequences have matches)",
                        match_rate * 100.0, 
                        unique_matched_source_sequences.len(), 
                        source_chunk.len()
                    );
                    
                    // If match rate exceeds threshold, this is likely a duplicate
                    if match_rate >= early_escape_threshold {
                        info!("EARLY ESCAPE: Match rate {:.2}% exceeds threshold {:.2}%. These databases appear to be duplicates or very similar.",
                            match_rate * 100.0, early_escape_threshold * 100.0);
                                                
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
                                
                // Track chunk timing
                let chunk_duration = chunk_start_time.elapsed();
                chunk_times.push(chunk_duration);
                                
                if !chunk_matches.is_empty() {
                    // Find dense regions
                    let dense_time = Instant::now();
                    let mut candidates = chains::find_dense_regions(
                        &chunk_matches,
                        &self.config,
                        Some(&self.source_storage),
                        None,
                    )?;
                    
                    info!("Found {} dense regions in {:?}", 
                        candidates.len(), dense_time.elapsed());
                    
                    if !candidates.is_empty() {
                        // Validate matches WITHOUT merging
                        let validation_time = Instant::now();
                        
                        // Sort candidates by source position for efficient processing
                        candidates.sort_by_key(|c| c.source_start);

                        // Acquire the read lock before calling validate_without_merging
                        let text_classifier_guard = self.text_classifier.read()
                            .expect("Failed to acquire read lock on text classifier for validation");

                        let validated = validation::validate_without_merging(
                            &candidates,                    // Vec<SequenceCluster>
                            &self.source_storage,          // Source storage reference
                            &self.target_storage,          // Target storage reference  
                            &self.similarity_calc,         // For text similarity checks
                            &self.config,          // Config
                            &*text_classifier_guard,       // Dereference the guard to get &TextClassifier
                            self.ngram_type,
                        )?;
                        
                        // Get the count of valid matches
                        info!("Validated {} matches in {:?} (not merged yet)", 
                            validated.len(), validation_time.elapsed());

                        // Write validated matches to temp file
                        if !validated.is_empty() {
                            let file_path = temp_manager.write_matches(&validated, chunk_index)?;
                            info!("Wrote {} validated matches from chunk {} to temporary file: {:?}",
                                validated.len(), chunk_index + 1, file_path);
                            temp_file_paths.push(file_path.clone());
                            debug!("DEBUG: Added temp file to paths list: {:?}", file_path);
                            debug!("DEBUG: Current temp files count: {}", temp_file_paths.len());
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
                    
                    // Progress update
                    if let Some(ref ctx) = progress_context {
                        ctx.update_progress(
                            (chunk_index + 1) as u64,
                            total_chunks as u64,
                            Some(format!("Chunk {}/{} ({:.1}%) | ETA: {}", 
                                chunk_index + 1, 
                                total_chunks,
                                processed_pct,
                                format_duration(estimated_remaining)
                            ))
                        );
                        
                        // Update match count if we found matches
                        if !temp_file_paths.is_empty() {
                            ctx.update_matches(temp_file_paths.len());
                        }
                    }
                } else {
                    // First chunk - no ETA yet
                    if let Some(ref ctx) = progress_context {
                        ctx.update_progress(
                            1,
                            total_chunks as u64,
                            Some(format!("Chunk 1/{}", total_chunks))
                        );
                    }
                }
            }
            
            // Process the temporary files with streaming merge
            if !temp_file_paths.is_empty() {
                info!("Processing final merging from {} temporary files", temp_file_paths.len());
                debug!("DEBUG: About to merge the following temp files:");
                for (i, path) in temp_file_paths.iter().enumerate() {
                    debug!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, path.exists());
                }
                let merging_start_time = Instant::now();
                
                // Call our new streaming merge method (we'll implement this next)
                let total_match_count = self.stream_merge_temp_files(
                    &temp_file_paths, 
                    &temp_manager, 
                    writer.as_mut(),
                    &config,
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

    pub fn find_matches_parallel(
        &self,
        source_range: Range<u64>,
        target_range: Range<u64>,
        output_path: Option<&Path>,
        early_escape_threshold: f64,
        config: &AlNaqlConfig,
        progress_context: Option<ProgressContext>,
    ) -> Result<usize> {
        let start_time = Instant::now();
        info!("Starting Rayon-based match finding process");

        // Read debug configuration
        let keep_temp_files = config.debug.keep_temp_files.unwrap_or(false);
        let keep_on_error = config.debug.keep_temp_files_on_error.unwrap_or(false);
        
        // Create temp directory path
        let temp_dir = PathBuf::from("temp/matches")
            .join(format!("run_{}", SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()));
        
        // Create TempFileManager with config
        let mut temp_manager = match TempFileManager::new(&temp_dir, "matches") {
            Ok(manager) => manager,
            Err(e) => return Err(Error::Io(e)),
        };
        temp_manager.set_keep_files(keep_temp_files);
    
        // Wrap everything in a closure to ensure cleanup happens regardless of success/failure
        let result = (|| {
            // Initialize writer outside the parallel processing
            let mut writer = if let Some(path) = output_path {
                Some(std::io::BufWriter::new(File::create(path)?))
            } else {
                None
            };
            
            // Check if we should use buffered writer
            let use_buffered_writer = self.config.matcher.use_buffered_parallel;
            
            // Track progress with atomic counter
            let processed_counter = Arc::new(AtomicUsize::new(0));

            // Calculate chunks
            let num_cpus = num_cpus::get();
            let chunk_ranges = create_simple_chunks(
                source_range, 
                num_cpus,
                &self.config.matcher
            );
            let total_chunks = chunk_ranges.len();
            
            // Flag for early escape detection
            let early_escape_flag = Arc::new(AtomicUsize::new(0)); // 0=no escape, 1=escape
            let first_chunk_processed = Arc::new(AtomicUsize::new(0)); // Track if first chunk processed
            
            // This will hold temp file paths from either path
            let final_temp_file_paths: Vec<PathBuf>;
            
            if use_buffered_writer {
                // ================================================================
                // BUFFERED WRITER
                // ================================================================
                info!("Using buffered writer for parallel processing");
    
                // Create buffered writer with channel - now just pass the config directly
                let (sender, writer_handle) = create_buffered_writer(
                    &self.config,  // Pass the whole config
                    temp_manager.clone(),
                );
                
                // Wrap sender in Arc<Mutex> for sharing across threads
                let sender = Arc::new(Mutex::new(sender));
                // Clone for parallel access
                let progress_ctx = progress_context.clone();
                // Processing function for buffered writer path
                let process_chunk = {
                    let matcher = self;
                    let processed_counter = Arc::clone(&processed_counter);
                    let early_escape_flag = Arc::clone(&early_escape_flag);
                    let first_chunk_processed = Arc::clone(&first_chunk_processed);
                    let sender = Arc::clone(&sender);
                    
                    move |chunk_range: Range<u64>, chunk_index: usize| -> Result<()> {
                        // Skip if early escape has been triggered
                        if early_escape_flag.load(Ordering::SeqCst) == 1 {
                            trace!("Chunk {}: Skipping due to early escape flag", chunk_index);
                            return Ok(());
                        }
                                                
                        // Convert range to vec of sequence IDs
                        let source_chunk: Vec<u64> = chunk_range.collect();
                        let chunk_size = source_chunk.len();
                        
                        debug!("Chunk {}: Processing {} source sequences", chunk_index, chunk_size);
                        
                        // Process this chunk using the dispatcher
                        let chunk_matches = matcher.dispatch_ngram_matching(
                            &source_chunk,
                            target_range.clone(),
                            &matcher.similarity_calc,
                        )?;
                        
                        debug!("Chunk {}: Found {} raw matches", chunk_index, chunk_matches.len());
                        
                        // Early escape detection for first chunk
                        if chunk_index == 0 && !chunk_matches.is_empty() {
                            // Calculate match rate for early escape check
                            // We're counting how many unique source sequences found at least one match
                            let unique_matched_source_sequences: std::collections::HashSet<_> = chunk_matches
                                .iter()
                                .map(|m| m.source_sequence)
                                .collect();
                            
                            let match_rate = unique_matched_source_sequences.len() as f64 / chunk_size as f64;
                            
                            info!("First chunk match rate: {:.2}% ({} of {} source sequences have matches)",
                                match_rate * 100.0,
                                unique_matched_source_sequences.len(),
                                chunk_size
                            );
                            
                            // If match rate exceeds threshold, mark for early escape
                            if match_rate >= early_escape_threshold {
                                info!("EARLY ESCAPE: Match rate {:.2}% exceeds threshold {:.2}%. These databases appear to be duplicates or very similar.",
                                    match_rate * 100.0,
                                    early_escape_threshold * 100.0
                                );
                                
                                early_escape_flag.store(1, Ordering::SeqCst);
                                
                                // Still process these matches before escaping
                                let mut candidates = chains::find_dense_regions(&chunk_matches, &self.config, None, None)?;
                                if !candidates.is_empty() {
                                    candidates.sort_by_key(|c| c.source_start);
                                    
                                    let text_classifier_guard = matcher.text_classifier.read()
                                        .expect("Failed to acquire read lock on text classifier");
                                    
                                    let validated = validation::validate_without_merging(
                                        &candidates,
                                        &matcher.source_storage,
                                        &matcher.target_storage,
                                        &matcher.similarity_calc,
                                        &self.config,
                                        &*text_classifier_guard,
                                        self.ngram_type
                                    )?;
                                    
                                    if !validated.is_empty() {
                                        // Send to buffered writer
                                        sender.lock().unwrap().send(BufferedChunkMessage::Matches {
                                            chunk_index,
                                            matches: validated,
                                        }).map_err(|e| Error::async_err(format!(
                                            "Failed to send matches to writer: {}", e
                                        )))?;
                                    }
                                }
                                
                                first_chunk_processed.store(1, Ordering::SeqCst);
                                return Ok(());
                            }
                        }
                        
                        // Process matches through clustering and validation
                        if !chunk_matches.is_empty() {
                            let mut candidates = chains::find_dense_regions(&chunk_matches, &self.config, None, None)?;
                            
                            info!("Chunk {}: Found {} dense region candidates", chunk_index, candidates.len());
                            
                            if !candidates.is_empty() {
                                candidates.sort_by_key(|c| c.source_start);
                                
                                let text_classifier_guard = matcher.text_classifier.read()
                                    .expect("Failed to acquire read lock on text classifier");
                                
                                let validated = validation::validate_without_merging(
                                    &candidates,
                                    &matcher.source_storage,
                                    &matcher.target_storage,
                                    &matcher.similarity_calc,
                                    &self.config,
                                    &*text_classifier_guard,
                                    self.ngram_type
                                )?;
                                
                                let valid_count = validated.len();
                                info!("Chunk {}: Validated {} matches", chunk_index, valid_count);
                                
                                // Send to buffered writer instead of writing directly
                                if !validated.is_empty() {
                                    sender.lock().unwrap().send(BufferedChunkMessage::Matches {
                                        chunk_index,
                                        matches: validated,
                                    }).map_err(|e| Error::async_err(format!(
                                        "Failed to send matches to writer: {}", e
                                    )))?;
                                    
                                    debug!("Chunk {}: Sent {} matches to buffered writer", 
                                        chunk_index, valid_count);
                                }
                            }
                        }
                        
                        // Update progress
                        let processed = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;

                        // Update progress context
                        if let Some(ref ctx) = progress_ctx {
                            ctx.update_progress(
                                processed as u64,
                                total_chunks as u64,
                                Some(format!("Completed {}/{} chunks", processed, total_chunks))
                            );
                        }
                        
                        Ok(())
                    }
                };
                
                // Process ALL chunks in parallel
                info!("Processing {} chunks in parallel with buffered writer", chunk_ranges.len());
                
                chunk_ranges
                    .par_iter()
                    .enumerate()
                    .for_each(|(chunk_idx, range)| {
                        if early_escape_flag.load(Ordering::SeqCst) == 1 {
                            return;
                        }
                        
                        let chunk_start = Instant::now();
                        
                        match process_chunk(range.clone(), chunk_idx) {
                            Ok(()) => {
                                debug!("Chunk {} completed in {:?}", 
                                    chunk_idx + 1, chunk_start.elapsed());
                            }
                            Err(e) => {
                                error!("Error processing chunk {}: {}", chunk_idx, e);
                            }
                        }
                        
                        // Clean up caches periodically
                        if chunk_idx % 5 == 0 {
                            METADATA_CACHE.trim_thread_local_caches(Duration::from_secs(120));
                        }
                    });
                
                info!("All {} chunks processed, shutting down writer", chunk_ranges.len());
                
                // Send explicit shutdown signal to writer
                sender.lock().unwrap().send(BufferedChunkMessage::Shutdown)
                    .map_err(|e| Error::async_err(format!("Failed to send shutdown signal: {}", e)))?;

                // Now we can drop the sender Arc
                drop(sender);
                
                // Wait for writer thread to complete
                match writer_handle.join() {
                    Ok(Ok(writer_result)) => {
                        info!("Writer thread completed successfully: {} matches written to {} files",
                            writer_result.total_matches_written,
                            writer_result.temp_file_paths.len()
                        );
                        final_temp_file_paths = writer_result.temp_file_paths;
                    }
                    Ok(Err(e)) => {
                        return Err(Error::async_err(format!("Writer thread error: {}", e)));
                    }
                    Err(_) => {
                        return Err(Error::async_err("Writer thread panicked".to_string()));
                    }
                }
                
            } else {
                // ================================================================
                // DIRECT FILE WRITING
                // ================================================================
                info!("Using direct file writing for parallel processing");
                
                // Use a thread-safe collection to store temporary file paths
                let temp_file_paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
                // Clone for parallel access
                let progress_ctx = progress_context.clone();
                // Processing function for direct writing path
                let process_chunk = {
                    let matcher = self;
                    let temp_file_paths = Arc::clone(&temp_file_paths);
                    let processed_counter = Arc::clone(&processed_counter);
                    let temp_manager = Arc::new(temp_manager.clone());
                    let early_escape_flag = Arc::clone(&early_escape_flag);
                    let first_chunk_processed = Arc::clone(&first_chunk_processed);
                    
                    move |chunk_range: Range<u64>, chunk_index: usize| -> Result<()> {
                        // Skip if early escape has been triggered
                        if early_escape_flag.load(Ordering::SeqCst) == 1 {
                            trace!("Chunk {}: Skipping due to early escape flag", chunk_index);
                            return Ok(());
                        }
                                                
                        // Convert range to vec of sequence IDs
                        let source_chunk: Vec<u64> = chunk_range.collect();
                        let chunk_size = source_chunk.len();
                        
                        debug!("Chunk {}: Processing {} source sequences", chunk_index, chunk_size);
                        
                        // Process this chunk using the dispatcher
                        let chunk_matches = matcher.dispatch_ngram_matching(
                            &source_chunk,
                            target_range.clone(),
                            &matcher.similarity_calc,
                        )?;
                        
                        debug!("Chunk {}: Found {} raw matches", chunk_index, chunk_matches.len());
                        
                        // Early escape detection for first chunk
                        if chunk_index == 0 && !chunk_matches.is_empty() {
                            // Calculate match rate for early escape check
                            // We're counting how many unique source sequences found at least one match
                            let unique_matched_source_sequences: std::collections::HashSet<_> = chunk_matches
                                .iter()
                                .map(|m| m.source_sequence)
                                .collect();
                            
                            let match_rate = unique_matched_source_sequences.len() as f64 / chunk_size as f64;
                            
                            info!("First chunk match rate: {:.2}% ({} of {} source sequences have matches)",
                                match_rate * 100.0,
                                unique_matched_source_sequences.len(),
                                chunk_size
                            );
                            
                            // If match rate exceeds threshold, mark for early escape
                            if match_rate >= early_escape_threshold {
                                info!("EARLY ESCAPE: Match rate {:.2}% exceeds threshold {:.2}%. These databases appear to be duplicates or very similar.",
                                    match_rate * 100.0,
                                    early_escape_threshold * 100.0
                                );
                                
                                early_escape_flag.store(1, Ordering::SeqCst);
                                first_chunk_processed.store(1, Ordering::SeqCst);
                                return Ok(());
                            }
                        }
                        
                        // Process matches through clustering and validation
                        if !chunk_matches.is_empty() {
                            let mut candidates = chains::find_dense_regions(&chunk_matches, &self.config, None, None)?;
                            
                            info!("Chunk {}: Found {} dense region candidates", chunk_index, candidates.len());
                            
                            if !candidates.is_empty() {
                                candidates.sort_by_key(|c| c.source_start);
                                
                                let text_classifier_guard = matcher.text_classifier.read()
                                    .expect("Failed to acquire read lock on text classifier");
                                
                                let validated = validation::validate_without_merging(
                                    &candidates,
                                    &matcher.source_storage,
                                    &matcher.target_storage,
                                    &matcher.similarity_calc,
                                    &self.config,
                                    &*text_classifier_guard,
                                    self.ngram_type
                                )?;
                                
                                let valid_count = validated.len();
                                info!("Chunk {}: Validated {} matches", chunk_index, valid_count);
                                
                                // Direct file writing (old path)
                                if !validated.is_empty() {
                                    let temp_file_path = temp_manager.write_matches(&validated, chunk_index)?;
                                    
                                    debug!("Chunk {}: Wrote {} matches to temporary file: {:?}", 
                                        chunk_index, valid_count, temp_file_path);
                                    
                                    match temp_file_paths.lock() {
                                        Ok(mut paths) => {
                                            paths.push(temp_file_path);
                                            debug!("Chunk {}: Recorded temp file path, total files now: {}", 
                                                chunk_index, paths.len());
                                        },
                                        Err(e) => {
                                            return Err(Error::async_err(format!(
                                                "Could not record temp file path for chunk {}: {}", 
                                                chunk_index, e
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Update progress
                        let processed = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;

                        // Update progress context  
                        if let Some(ref ctx) = progress_ctx {
                            ctx.update_progress(
                                processed as u64,
                                total_chunks as u64,
                                Some(format!("Completed {}/{} chunks", processed, total_chunks))
                            );
                        }
                                        
                        Ok(())
                    }
                };
                
                // Process ALL chunks in parallel
                info!("Processing {} chunks in parallel with direct writing", chunk_ranges.len());
                
                chunk_ranges
                    .par_iter()
                    .enumerate()
                    .for_each(|(chunk_idx, range)| {
                        if early_escape_flag.load(Ordering::SeqCst) == 1 {
                            return;
                        }
                        
                        let chunk_start = Instant::now();
                        
                        match process_chunk(range.clone(), chunk_idx) {
                            Ok(()) => {
                                debug!("Chunk {} completed in {:?}", 
                                    chunk_idx + 1, chunk_start.elapsed());
                            }
                            Err(e) => {
                                error!("Error processing chunk {}: {}", chunk_idx, e);
                            }
                        }
                        
                        // Clean up caches periodically
                        if chunk_idx % 5 == 0 {
                            METADATA_CACHE.trim_thread_local_caches(Duration::from_secs(120));
                        }
                    });
                
                info!("All {} chunks processed", chunk_ranges.len());
                
                // Collect temp file paths from mutex
                final_temp_file_paths = if let Ok(paths) = temp_file_paths.lock() {
                    paths.clone()
                } else {
                    warn!("Failed to lock temp_file_paths mutex when retrieving paths");
                    Vec::new()
                };
            }
            
            // ================================================================
            // COMMON PATH: Process results from either approach
            // ================================================================
            
            // Check if early escape was triggered
            if early_escape_flag.load(Ordering::SeqCst) == 1 {
                info!("Finishing early due to high similarity detected in first chunk");
                return Ok(usize::MAX); // Special value to indicate early escape
            }
            
            debug!("All processing threads completed, collected {} temporary files", 
                final_temp_file_paths.len());
            
            // Debug: verify all temp files
            if !final_temp_file_paths.is_empty() {
                debug!("DEBUG: About to merge the following temp files:");
                for (i, path) in final_temp_file_paths.iter().enumerate() {
                    debug!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, path.exists());
                }
            }
            
            // Process final merging with our streaming approach
            if !final_temp_file_paths.is_empty() {
                let merging_start_time = Instant::now();
                info!("Processing final merging for {} matches from all chunks", 
                    final_temp_file_paths.len());
                
                // Use our streaming merge method with the correct writer reference
                let total_match_count = self.stream_merge_temp_files(
                    &final_temp_file_paths, 
                    &temp_manager, 
                    writer.as_mut(), 
                    &config
                )?;
                
                info!("Final merging complete: {} matches after merging (took {:?})", 
                    total_match_count, merging_start_time.elapsed());
                info!("Completed all chunks, found {} total matches in {:?}", 
                    total_match_count, start_time.elapsed());

                // Finish progress bar
                Ok(total_match_count)
            } else {
                info!("No matches found in any chunks");
                Ok(0)
            }
        })();
        
        // Handle cleanup based on result and config
        match result {
            Ok(count) => {
                // Success - cleanup unless keep_temp_files is true
                if !keep_temp_files {
                    if let Err(e) = temp_manager.cleanup() {
                        warn!("Error cleaning up temporary files: {}", e);
                    }
                    // Also remove the directory
                    if temp_dir.exists() {
                        if let Err(e) = fs::remove_dir_all(&temp_dir) {
                            warn!("Failed to remove temporary directory: {}", e);
                        }
                    }
                } else {
                    info!("✓ Temporary files kept for debugging at: {:?}", temp_dir);
                    info!("  To clean up manually, run: rm -rf {:?}", temp_dir);
                }
                Ok(count)
            }
            Err(e) => {
                // Error - check if we should keep files
                if keep_on_error || keep_temp_files {
                    info!("⚠ Error occurred, keeping temporary files for debugging");
                    info!("  Location: {:?}", temp_dir);
                    info!("  Error: {}", e);
                    info!("  To clean up manually, run: rm -rf {:?}", temp_dir);
                } else {
                    // Clean up even on error
                    if let Err(cleanup_err) = temp_manager.cleanup() {
                        warn!("Error cleaning up after failure: {}", cleanup_err);
                    }
                    if temp_dir.exists() {
                        let _ = fs::remove_dir_all(&temp_dir);
                    }
                }
                Err(e)
            }
        }
    }


    /// Grid-based streaming match finding - processes entire dataset with memory-aware streaming
    /// This is optimized for very large datasets that may exceed available memory
    /// 
    /// # Parameters
    /// * `source_range` - Range of source ngram IDs to process
    /// * `target_range` - Range of target ngram IDs to process  
    /// * `output_path` - Optional path to write final results
    /// * `resume_enabled` - Whether to attempt resuming from previous interrupted run
    pub fn find_matches_grid_streaming(
        &self,
        source_range: Range<u64>,
        target_range: Range<u64>,
        output_path: Option<&Path>,
        resume_enabled: bool,
        source_db_path: &Path,
        target_db_path: &Path,
        progress_context: Option<ProgressContext>,
    ) -> Result<usize> {
        let start_time = Instant::now();
        info!("Starting grid-based streaming match finding process");
        info!("Dataset size: {} source × {} target ngrams", 
            source_range.end - source_range.start,
            target_range.end - target_range.start);
        
        // Determine the run directory based on source and target databases
        // Note: You'll need to pass the actual DB paths - this might need adjustment
        // based on how you access the database paths from the orchestrator
        let run_dir = ResumeManager::get_run_directory(source_db_path, target_db_path);
        
        // Check if debug mode wants to keep temp files
        let keep_temp_files = self.config.debug.keep_temp_files.unwrap_or(false);
        
        // ================================================================
        // Resume Path - if enabled and resumable state exists
        // ================================================================
        if resume_enabled {
            match ResumeManager::find_resumable_run(source_db_path, target_db_path)? {
                Some(resume_state) => {
                    info!("Resumable run found!");
                    info!("{}", resume_state.summary());
                    
                    // Attempt to resume from the last completed stage
                    match grid_orchestrator::execute_from_stage(
                        self,
                        resume_state.resume_from_stage,
                        resume_state.input_file,
                        source_range,
                        target_range,
                        output_path,
                        run_dir.clone(),
                        Some(resume_state.metadata),
                        progress_context.as_ref(),
                    ) {
                        Ok(count) => {
                            info!("Successfully resumed and completed: {} matches", count);
                            
                            // Clean up if not in debug mode
                            if !keep_temp_files {
                                info!("Cleaning up temporary files...");
                                if let Err(e) = ResumeManager::cleanup_run(source_db_path, target_db_path) {
                                    warn!("Failed to clean up run directory: {}", e);
                                }
                            } else {
                                info!("Keeping temporary files for debugging at {:?}", run_dir);
                            }
                            
                            return Ok(count);
                        }
                        Err(e) => {
                            // Resume failed - return error to let job tracker mark as failed
                            error!("Failed to resume grid streaming: {}", e);
                            return Err(e);
                        }
                    }
                }
                None => {
                    info!("No resumable run found, starting fresh");
                }
            }
        }
        
        // ================================================================
        // Fresh Run Path - either resume disabled or no resumable state
        // ================================================================
        
        // Initialize new run with metadata
        let run_dir = ResumeManager::initialize_run(source_db_path, target_db_path)?;
        info!("Initialized new run at {:?}", run_dir);
        
        // Execute the pipeline stages in sequence
        let result = (|| {
            // Phase 1: Setup infrastructure
            let (temp_manager, memory_manager, total_comparisons) = 
                grid_orchestrator::setup_infrastructure(
                    &source_range,
                    &target_range,
                    run_dir.clone(),
                )?;
            
            // Phase 2: Extract source sequences

            // Enter grid processing phase
            if let Some(ref ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::GridProcessing);
            }

            let source_sequences = grid_orchestrator::extract_source_sequences(
                &source_range
            )?;
            
            // Phase 3: Configure memory scaling
            let streaming_config = grid_orchestrator::configure_memory_scaling(
                self,
                &memory_manager,
                &source_sequences,
                &target_range,
            )?;
            
            // Phase 4/5: Grid matching
            let grid_result = grid_orchestrator::execute_grid_matching(
                self,
                &source_sequences,
                target_range.clone(),
                &streaming_config,
                &memory_manager,
                &temp_manager,
                progress_context.as_ref(),
            )?;
            let matches_written = grid_result.count();
            // Update metadata after grid matching
            ResumeManager::update_stage_completion(
                &run_dir,
                crate::utils::resume_manager::PipelineStage::GridMatching,
                "grid_matches.tmp",
                matches_written,
            )?;
            
            // Phase 5.5: Sorting and deduplication

            // Transition to sorting phase
            if let Some(ref ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Sorting);
            }

            let sorted_result = grid_orchestrator::execute_sorting(
                grid_result,  // Pass the PipelineData directly
                &memory_manager,
                &temp_manager,
            )?;
            
            // Update metadata after sorting
            let sorted_count = sorted_result.count();
            ResumeManager::update_stage_completion(
                &run_dir,
                crate::utils::resume_manager::PipelineStage::Sorting,
                "grid_matches_sorted.tmp",
                sorted_count,
            )?;
            
            // Phase 6: Clustering

            // Transition to clustering phase
            if let Some(ref ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Clustering);
            }

            let clustered_result = grid_orchestrator::execute_clustering(
                self,
                sorted_result,  // Pass the PipelineData directly from sorting
                &streaming_config,
                &memory_manager,
                &temp_manager,
                progress_context.as_ref(),
            )?;

            // Extract count for metadata
            let clusters_written = clustered_result.count();

            // Update metadata after clustering
            ResumeManager::update_stage_completion(
                &run_dir,
                crate::utils::resume_manager::PipelineStage::Clustering,
                "clusters.tmp",
                clusters_written,
            )?;
            
            // Phase 7: Validation
            // Transition to validation phase
            if let Some(ref ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Validation);
            }

            let validated_result = grid_orchestrator::execute_validation(
                self,
                clustered_result,  // Pass the PipelineData from clustering
                &streaming_config,
                &memory_manager,  // Add this parameter - execute_validation needs it
                &temp_manager,
                progress_context.as_ref(),
            )?;

            // Extract count for metadata
            let validated_written = validated_result.count();

            // Update metadata after validation
            ResumeManager::update_stage_completion(
                &run_dir,
                crate::utils::resume_manager::PipelineStage::Validation,
                "validated.tmp",
                validated_written,
            )?;
            
            // Phase 8: Final merge and output

            // Transition to merge phase
            if let Some(ref ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::FinalMerge);
            }

            let final_count = grid_orchestrator::execute_final_merge(
                self,
                validated_result,  // Pass the PipelineData from validation
                output_path,
                &streaming_config,
                &temp_manager,
                start_time,
                total_comparisons,
                progress_context.as_ref(),
            )?;

            // Update metadata after final merge
            ResumeManager::update_stage_completion(
                &run_dir,
                crate::utils::resume_manager::PipelineStage::FinalMerge,
                "final",
                final_count,
            )?;
            // Clean up temporary files based on config
            if !keep_temp_files {
                grid_orchestrator::cleanup_temp_files(temp_manager, &run_dir, false)?;
            } else {
                info!("Keeping temporary files for debugging at {:?}", run_dir);
            }
            
            Ok(final_count)
        })();
        
        // Mark as complete
        if let Some(ref ctx) = progress_context {
            ctx.enter_phase(ProgressPhase::Complete);
        }
        // Return the result
        result
    }
    

    /// Dispatch to the appropriate n-gram matching implementation based on configuration
    fn dispatch_ngram_matching(
        &self,
        source_chunk: &[u64],  // Just sequence IDs
        target_range: Range<u64>,
        similarity_calc: &SimilarityCalculator,
    ) -> Result<Vec<SequenceMatch>> {
        let strategy = self.config.matcher.ngram_matching_strategy.clone();
            
        // Track the operation in shared stats
        let start_time = Instant::now();
        
        let matches = match strategy {
            NGramMatchingStrategy::Sequential => {
                debug!("Using sequential n-gram matching for {} source ngrams against target range {}..{}",
                    source_chunk.len(), target_range.start, target_range.end);
                
                // Call the sequential implementation
                ngrams_sequential::process(
                    &self.config,
                    source_chunk,
                    &self.source_storage,
                    &self.target_storage,
                    similarity_calc,
                )?
            },
            
            NGramMatchingStrategy::Parallel => {
                debug!("Using parallel n-gram matching for {} source ngrams against target range {}..{}",
                    source_chunk.len(), target_range.start, target_range.end);
                
                // Call the parallel implementation with the range
                ngrams_parallel::process_parallel(
                    &self.config,
                    source_chunk,
                    &self.source_storage,
                    &self.target_storage,
                    similarity_calc,
                    
                )?
            },
            
            NGramMatchingStrategy::ParallelOpt => {
                debug!("Using optimized parallel n-gram matching for {} source ngrams against target range {}..{}",
                    source_chunk.len(), target_range.start, target_range.end);
                
                // Call the optimized parallel implementation with the range
                ngrams_parallel_opt::process_parallel_opt(
                    &self.config,
                    source_chunk,
                    &self.source_storage,
                    &self.target_storage,
                    similarity_calc,
                )?
            },

        };
        
        // Update statistics
        let elapsed = start_time.elapsed();
        let target_count = (target_range.end - target_range.start) as usize;
        let comparisons = source_chunk.len() * target_count;
        self.shared_state.stats.total_comparisons.fetch_add(comparisons, Ordering::Relaxed);
        self.shared_state.stats.matches_found.fetch_add(matches.len(), Ordering::Relaxed);
        
        debug!("N-gram matching completed in {:?}: {} matches from {} potential comparisons",
            elapsed, matches.len(), comparisons);
        
        // Only perform traced phrase checking if debug logging is enabled
        if log::log_enabled!(log::Level::Debug) && !self.config.debug.traced_phrase.is_empty() {
            // Fetch and check only when we're actually going to log
            let source_sequences: Vec<u64> = matches.iter()
                .map(|m| m.source_sequence)
                .take(100)  // Maybe limit to first 100 for performance
                .collect();
            
            let source_texts = self.source_storage.get_text_bytes_batch(&source_sequences)?;
            
            let traced_count = source_texts.values()
                .filter_map(|bytes| String::from_utf8(bytes.clone()).ok())
                .filter(|text| text.contains(&self.config.debug.traced_phrase))
                .count();
            
            if traced_count > 0 {
                // Extrapolate if we only checked a sample
                let estimated_total = if matches.len() > 100 {
                    (traced_count * matches.len()) / source_sequences.len()
                } else {
                    traced_count
                };
                
                debug!("TRACE: Found {} matches containing traced phrase '{}' (estimated from sample)",
                    estimated_total, self.config.debug.traced_phrase);
            }
        }
        
        Ok(matches)
    }
    
    // ========================================================================
    // MERGING METHODS - Copied and adapted from matcher_legacy.rs
    // ========================================================================
    
    /// Stream merge temp files into final output
    /// Reads matches from temporary files in batches, performs merging operations,
    /// and writes the results directly to the output file
    fn stream_merge_temp_files(
        &self,
        temp_file_paths: &[PathBuf],
        temp_manager: &TempFileManager,
        mut writer: Option<&mut BufWriter<File>>,
        config: &AlNaqlConfig,
    ) -> Result<usize> {
        info!("Starting streaming merge from {} temporary files", temp_file_paths.len());
        
        // Debug: verify temp files exist
        let temp_dir = temp_manager.get_temp_dir();
        debug!("DEBUG: Temp directory: {:?}", temp_dir);
        debug!("DEBUG: Verifying all temp files exist:");
        let mut missing_files = 0;
        for (i, path) in temp_file_paths.iter().enumerate() {
            let exists = path.exists();
            debug!("DEBUG:   File {}: {:?} (exists: {})", i+1, path, exists);
            if !exists {
                missing_files += 1;
            }
        }
        if missing_files > 0 {
            warn!("DEBUG: Missing files: {}/{}", missing_files, temp_file_paths.len());
        }
        
        // If no temporary files, return early
        if temp_file_paths.is_empty() {
            return Ok(0);
        }
        
        // Memory-adaptive batch processing
        let memory_usage = self.get_memory_usage_percent();
        let is_memory_constrained = memory_usage > 70.0;
        
        // Define batch sizes based on memory pressure
        let batch_size = if is_memory_constrained {
            500  // Smaller batch size under memory pressure
        } else {
            1000
        };
        
        let merge_buffer_size = if is_memory_constrained {
            2000  // Smaller buffer size under memory pressure
        } else {
            5000
        };
        
        info!("Using memory-adaptive settings: batch_size={}, buffer_size={} (memory usage: {:.1}%)",
            batch_size, merge_buffer_size, memory_usage);
        
        // Initialize counters
        let mut total_read: usize = 0;
        let mut total_written: usize = 0;
        let mut batch_count: usize = 0;
        
        // Create a buffer for accumulating matches for merging
        let mut merge_buffer: Vec<ReconstructedSegment> = Vec::with_capacity(merge_buffer_size);
        
        // Process each temporary file using BINARY STREAMING
        for (file_index, file_path) in temp_file_paths.iter().enumerate() {
            debug!("Processing temporary binary file {}/{}: {:?}", 
                file_index + 1, temp_file_paths.len(), file_path);
            
            // Stream matches from the binary file
            let stream_iter = match temp_manager.stream_matches(file_path) {
                Ok(iter) => iter,
                Err(e) => {
                    warn!("Error opening binary temp file {}: {}", file_path.display(), e);
                    continue;
                }
            };
            
            let mut file_matches_count = 0;
            
            // Process matches one by one from the binary stream
            for match_result in stream_iter {
                match match_result {
                    Ok(segment) => {
                        // Add to merge buffer
                        merge_buffer.push(segment);
                        file_matches_count += 1;
                        
                        // If buffer reaches threshold, process it
                        if merge_buffer.len() >= merge_buffer_size {
                            // Merge and write current buffer
                            let merged_count = self.merge_and_write_buffer(&mut merge_buffer, &mut writer, &config)?;
                            total_written += merged_count;
                            batch_count += 1;
                            
                            debug!("Processed batch {} with {} matches, wrote {} merged matches", 
                                batch_count, merge_buffer_size, merged_count);
                            
                            // Clear buffer for next batch
                            merge_buffer.clear();
                            merge_buffer.shrink_to_fit(); // Release capacity
                            
                            // Brief pause to allow memory reclamation
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                    Err(e) => {
                        warn!("Error reading match from binary stream: {}", e);
                    }
                }
            }
            
            total_read += file_matches_count;
            debug!("Read {} matches from binary file {}", file_matches_count, file_index + 1);
            
            // Check memory pressure after each file
            let current_memory = self.get_memory_usage_percent();
            if current_memory > 80.0 {
                info!("High memory usage ({:.1}%) after processing file {}, pausing for memory reclamation", 
                    current_memory, file_index + 1);
                
                // Force memory cleanup
                self.perform_memory_cleanup();
                thread::sleep(Duration::from_millis(200));
            }
        }
        
        // Process any remaining matches in the buffer
        if !merge_buffer.is_empty() {
            debug!("Processing final merge buffer with {} matches", merge_buffer.len());
            let merged_count = self.merge_and_write_buffer(&mut merge_buffer, &mut writer, &config)?;
            total_written += merged_count;
        }
        
        // Ensure writer is flushed
        if let Some(w) = writer.as_mut() {
            if let Err(e) = w.flush() {
                warn!("Error flushing output file: {}", e);
            }
        }
        
        // Clean up temp directory
        let temp_dir = temp_manager.get_temp_dir();
        
        // First, call TempFileManager's cleanup
        if let Err(e) = temp_manager.cleanup() {
            warn!("Error during temp file cleanup: {}", e);
        } else {
            debug!("Temporary files successfully cleaned up");
        }
        
        // Now delete the entire temporary directory
        if temp_dir.exists() {
            debug!("Removing temporary directory: {:?}", temp_dir);
            match fs::remove_dir_all(&temp_dir) {
                Ok(_) => info!("Successfully removed temporary directory"),
                Err(e) => warn!("Failed to remove temporary directory: {}", e),
            }
        }
        
        info!("Streaming merge complete: read {} matches, wrote {} after merging", 
            total_read, total_written);
        
        Ok(total_written)
    }
    
    /// Helper method to merge a buffer of matches and write results
    /// This delegates most of the actual merging logic to the validation module
    fn merge_and_write_buffer(
        &self,
        buffer: &mut Vec<ReconstructedSegment>,
        writer: &mut Option<&mut BufWriter<File>>,
        config: &AlNaqlConfig,
    ) -> Result<usize> {
        // Early check for empty buffer
        if buffer.is_empty() {
            return Ok(0);
        }
        
        debug!("Merging buffer with {} matches", buffer.len());
        
        // Sort matches by source range start for consistent processing
        buffer.sort_by(|a, b| a.source_start_seq.cmp(&b.source_start_seq));
        

        // ============================================================================
        // VALIDATION PIPELINE - Transform raw matches into high-quality results
        // ============================================================================
        // This pipeline progressively refines matches through four stages:
        // 1. Merge overlapping matches (handles transitive overlaps)
        // 2. Merge adjacent matches (with quality checks)
        // 3. Remove contained matches (eliminate redundancy)
        // 4. Apply final quality filters (similarity, length, banality)

        // Stage 1: Merge Overlapping Matches
        // This creates a new vector with overlapping matches combined
        let mut validated_matches = validation::merge_overlapping_matches(
            buffer,
            &self.source_storage,
            &self.target_storage,
            &self.config,
        );

        debug!(
            "After merging overlaps: {} matches (reduced from {})", 
            validated_matches.len(),
            buffer.len()
        );

        // Stage 2: Merge Adjacent Matches
        // This modifies the vector in place, potentially fetching gap text from storage
        // Acquire the read lock for the merge operation
        let text_classifier_guard = self.text_classifier.read()
            .expect("Failed to acquire read lock on text classifier for merging");

        if let Err(e) = validation::merge_nearby_matches(
            &mut validated_matches,
            &self.config,          // MatcherConfig with adjacent_merge_ratio
            &self.source_storage,          // For fetching gap text from source
            &self.target_storage,          // For fetching gap text from target
            &self.similarity_calc,         // To recalculate similarities after merging
            &*text_classifier_guard,       // Dereference the guard to get &TextClassifier
        ) {
            // Handle error...
            warn!("Failed to merge adjacent matches: {}. Continuing with {} unmerged matches.",
                e, validated_matches.len());
        }
        debug!(
            "After merging adjacent: {} matches", 
            validated_matches.len()
        );

        // Stage 3: Remove Contained Matches
        // This modifies in place, removing matches fully contained within others
        // No return value - the vector is modified directly
        validation::remove_contained_matches(&mut validated_matches);
        debug!(
            "After removing contained: {} matches", 
            validated_matches.len()
        );

        // Stage 4: Apply Final Quality Filters
        // This is different - it's a predicate function that works on individual matches
        // We use retain() to keep only matches that pass all quality checks
        // Right before final filtering, initialize AutoBanal if needed

        if self.config.matcher.banality_detection_mode == "automatic" {
            // Check if AutoBanal is already initialized using a read lock
            let needs_initialization = {
                let classifier = self.text_classifier.read()
                    .expect("Failed to acquire read lock to check AutoBanal status");
                !classifier.has_auto_banal()  // Now we can call the method on the actual TextClassifier
            }; // Read lock is released here
            
            if needs_initialization {
                info!("Initializing automatic banality detection for final filtering...");
                
                let proportion = self.config.matcher.banality_auto_proportion;
                let threshold = self.config.matcher.banality_auto_threshold;
                
                // Do the expensive analysis before taking the write lock
                let mut auto_banal = AutoBanal::new(proportion, threshold);
                auto_banal.analyze_databases(&self.source_storage, &self.target_storage, &config.matcher)?;
                
                info!("Banality analysis complete: {} common text patterns identified", 
                    auto_banal.common_text_count());
                
                // Now acquire a write lock to modify the classifier
                {
                    let mut classifier = self.text_classifier.write()
                        .expect("Failed to acquire write lock to set AutoBanal");
                    classifier.set_auto_banal(Some(auto_banal));
                } // Write lock is released here
            }
        }
        let initial_count = validated_matches.len();
        let traced_phrase = if self.config.debug.traced_phrase.is_empty() {
            None
        } else {
            Some(self.config.debug.traced_phrase.as_str())
        };

        let final_validated_matches: Vec<ValidatedSegment> = validated_matches
            .into_iter()
            .filter_map(|segment| {
                validation::passes_final_filters(
                    segment,
                    &self.similarity_calc,
                    &self.config,
                    &*text_classifier_guard,
                    self.ngram_type,
                    traced_phrase,
                    &self.source_storage,
                    &self.target_storage,
                )
            })
            .collect();
        debug!("Final filtering: {} matches passed (from {} input)", 
            final_validated_matches.len(), initial_count);

        // Write the matches to output file if provided
        if let Some(w) = writer.as_mut() {
            // Now use final_validated_matches, not validated_matches
            let final_results = validation::convert_to_final_results(
                &final_validated_matches,  // <- Use the ValidatedSegment vector
                &self.source_storage,
                &self.target_storage,
            )?;
            
            // Write each result as a JSON line
            for result in &final_results {
                writeln!(w, "{}", serde_json::to_string(&result)?)
                    .map_err(|e| Error::Io(e))?;
            }
            
            // Ensure all data is written to disk immediately
            w.flush().map_err(|e| Error::Io(e))?;
            
            debug!("Wrote {} matches to output file", final_results.len());
        }

        // Return the count of matches that passed all filters
        Ok(final_validated_matches.len())
    }
    
    /// Get current memory usage percentage
    fn get_memory_usage_percent(&self) -> f64 {
        // Default to 50% if we can't determine
        50.0
    }
    
}

// ============================================================================
// HELPER METHODS FOR MEMORY MANAGEMENT
// ============================================================================

impl MatcherOrchestrator {
    
    /// Perform memory cleanup when pressure is detected
    fn perform_memory_cleanup(&self) {
        info!("Performing memory cleanup due to pressure");
        self.shared_state.clear_caches();
        
        // Brief pause to allow OS to reclaim memory
        thread::sleep(Duration::from_millis(100));
    }
    
    pub fn write_final_output(
        &self,
        results: &[FinalMatchResult],
        base_output_path: &Path,
        config: &AlNaqlConfig,
    ) -> Result<Vec<PathBuf>> {
        let mut output_paths = Vec::new();
        
        // Generate output paths based on config
        let paths = config.get_output_paths(base_output_path);
        
        for path in paths {
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            
            match ext {
                "bin" | "lz4" => {
                    // Write binary output
                    info!("Writing {} matches to binary file: {:?}", results.len(), path);
                    
                    let file = File::create(&path)?;
                    let compress = ext == "lz4" || config.compress_final_output();
                    let mut writer = BinaryWriter::new(file, compress, DataType::FinalMatchResult)?;
                    
                    for result in results {
                        writer.write_item(result)?;
                    }
                    writer.flush()?;
                    
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        info!("Binary file size: {:.2} MB", metadata.len() as f64 / (1024.0 * 1024.0));
                    }
                },
                
                "jsonl" | "json" => {
                    // Write JSON output
                    info!("Writing {} matches to JSON file: {:?}", results.len(), path);
                    
                    let file = File::create(&path)?;
                    let mut writer = BufWriter::with_capacity(256 * 1024, file);
                    
                    for result in results {
                        writeln!(writer, "{}", serde_json::to_string(result)?)?;
                    }
                    writer.flush()?;
                    
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        info!("JSON file size: {:.2} MB", metadata.len() as f64 / (1024.0 * 1024.0));
                    }
                },
                
                _ => {
                    warn!("Unknown output format for file: {:?}", path);
                    continue;
                }
            }
            
            output_paths.push(path);
        }
        
        // Log comparison if both formats were written
        if output_paths.len() == 2 {
            if let (Ok(bin_meta), Ok(json_meta)) = (
                std::fs::metadata(&output_paths[0]),
                std::fs::metadata(&output_paths[1])
            ) {
                let bin_size = bin_meta.len() as f64 / (1024.0 * 1024.0);
                let json_size = json_meta.len() as f64 / (1024.0 * 1024.0);
                let reduction = ((json_size - bin_size) / json_size) * 100.0;
                
                info!("Output size comparison: Binary {:.2} MB vs JSON {:.2} MB ({:.1}% reduction)",
                    bin_size, json_size, reduction);
            }
        }
        
        info!("Successfully wrote {} matches to {} output file(s)", 
            results.len(), output_paths.len());
        
        Ok(output_paths)
    }
}

/// Create fixed-size chunks for parallel processing
/// This replaces both calculate_optimal_chunk_size and create_adaptive_chunk_ranges
fn create_simple_chunks(
    source_range: Range<u64>, 
    num_cpus: usize,
    config: &MatchingConfig
) -> Vec<Range<u64>> {
    let total_size = (source_range.end - source_range.start) as usize;
    
    // Calculate chunk size: aim for 2 chunks per CPU core
    // with reasonable bounds to prevent over-fragmentation
    // Use configured values instead of hard-coded ones
    let chunk_size = (total_size / (num_cpus * config.source_chunks_per_cpu))
        .max(config.min_source_chunk)
        .min(config.max_source_chunk);
    
    // Create evenly-sized chunks
    let mut chunk_ranges = Vec::new();
    let mut current_pos = source_range.start;
    
    while current_pos < source_range.end {
        let end_pos = (current_pos + chunk_size as u64).min(source_range.end);
        chunk_ranges.push(current_pos..end_pos);
        current_pos = end_pos;
    }
    
    // Log what we created - much simpler than before
    info!("Created {} chunks of ~{} ngrams each (total: {} ngrams, {} CPUs available)", 
        chunk_ranges.len(), 
        chunk_size,
        total_size,
        num_cpus
    );
    
    chunk_ranges
}