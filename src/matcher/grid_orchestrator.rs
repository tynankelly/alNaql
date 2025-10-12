// matcher/grid_orchestrator.rs
//! Grid-based streaming pipeline stages
//! 
//! This module contains the individual stages of the grid streaming pipeline,
//! extracted from the main orchestrator for better modularity and resume support.
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::ops::Range;
use std::time::Instant;
use log::{info, debug, warn};

use crate::error::{Result, Error};
use crate::config::AlNaqlConfig;
use crate::matcher::MatcherOrchestrator;
use crate::matcher::clusters;
use crate::matcher::clusters::streaming_final_merge;
use crate::matcher::clusters::validation;

use crate::matcher::types::{PipelineData, ReconstructedSegment, SequenceCluster, SequenceMatch, ValidatedSegment};
use crate::matcher::ngrams_matcher::ngrams_grid;
use crate::utils::resource_coordinator::memory_manager::MemoryManager;
use crate::utils::MemoryState;
use crate::utils::progress_manager::{ProgressContext, ProgressPhase};
use crate::utils::tempfile::TempFileManager;
use crate::utils::binary_io::{BinaryWriter, BinaryReader, DataType};

// ============================================================================
// INFRASTRUCTURE SETUP
// ============================================================================

/// Phase 1: Setup infrastructure for grid streaming
pub fn setup_infrastructure(
    source_range: &Range<u64>,
    target_range: &Range<u64>,
    run_dir: PathBuf,  // Pass in the run directory (either new or resumed)
) -> Result<(TempFileManager, MemoryManager, u64)> {
    info!("Setting up grid streaming infrastructure");
    
    // Create temp file manager for this run
    let temp_manager = TempFileManager::new(&run_dir, "grid")
        .map_err(|e| Error::Io(e))?;
    
    // Initialize memory manager
    let memory_manager = MemoryManager::new();
    memory_manager.log_memory_status();
   let initial_state = memory_manager.monitor_memory_thresholds();
   info!("Memory Manager initialized with state: {}", initial_state);
    
    // Calculate total comparisons for progress and statistics
    let total_comparisons = (source_range.end - source_range.start) * 
                           (target_range.end - target_range.start);
        
    info!("Infrastructure setup complete");
    Ok((temp_manager, memory_manager, total_comparisons))
}

// ============================================================================
// EXTRACT SOURCE SEQUENCES
// ============================================================================

/// Phase 2: Extract source sequence IDs for grid processing
pub fn extract_source_sequences(
    source_range: &Range<u64>,
) -> Result<Vec<u64>> {
    info!("Extracting source sequence IDs for range {:?}", source_range);
    
    let start_time = Instant::now();
    let mut source_sequences = Vec::new();
    
    for seq_id in source_range.clone() {
        source_sequences.push(seq_id);
    }
    
    info!("Extracted {} source sequences in {:?}", 
        source_sequences.len(), start_time.elapsed());
    
    Ok(source_sequences)
}

// ============================================================================
// CONFIGURE MEMORY SCALING
// ============================================================================

/// Phase 3: Configure memory-aware processing parameters
pub fn configure_memory_scaling(
    orchestrator: &MatcherOrchestrator,
    memory_manager: &MemoryManager,
    source_sequences: &[u64],
    target_range: &Range<u64>,
) -> Result<Arc<AlNaqlConfig>> { 
    info!("Configuring memory-aware processing parameters");
    
    // Just clone config without modifications (maintain original logic)
    let streaming_config = orchestrator.config.clone();
    
    // Calculate dataset sizes for logging
    let source_count = source_sequences.len();
    let target_count = (target_range.end - target_range.start) as usize;
    let total_comparisons = source_count * target_count;
    
    // Log memory status
    let memory_state = memory_manager.monitor_memory_thresholds();
    info!("Memory status before processing: {}", memory_state);
    
    info!("Configuration complete for {} comparisons", total_comparisons);

    
    Ok(streaming_config)
}

// ============================================================================
// GRID MATCHING
// ============================================================================

/// Phase 4/5: Execute grid matching and write results
pub fn execute_grid_matching(
    orchestrator: &MatcherOrchestrator,
    source_sequences: &[u64],
    target_range: Range<u64>,
    streaming_config: &AlNaqlConfig,
    memory_manager: &MemoryManager,
    temp_manager: &TempFileManager,
    progress_context: Option<&ProgressContext>,
) -> Result<PipelineData<SequenceMatch>> {  // CHANGE: Return type
    info!("Starting grid processing phase");
    
    // Prepare the path but DON'T create the file yet
    let grid_matches_path = temp_manager.get_temp_dir().join("grid_matches.tmp");
    
    info!("Starting grid-based streaming processing...");
    
    // Call the streaming grid processor with the PATH, not a writer
    let result = ngrams_grid::process_grid_streaming(
        streaming_config,
        source_sequences,
        &orchestrator.source_storage,
        &orchestrator.target_storage,
        &orchestrator.similarity_calc,
        memory_manager,
        grid_matches_path.clone(),
        target_range.clone(),
        progress_context,
    )?;
    
    // Handle the result based on whether it's in memory or on disk
    match result {
        ngrams_grid::GridProcessingResult::OnDisk { path, count } => {
            info!("Grid processing complete: {} matches written to disk", count);
            Ok(PipelineData::OnDisk { path, count })  // CHANGE: Return PipelineData
        }
        ngrams_grid::GridProcessingResult::InMemory(matches) => {
            let count = matches.len();
            info!("Grid processing complete: {} matches kept in memory", count);
            
            // CHANGE: Don't write to disk anymore - just return the in-memory data
            Ok(PipelineData::InMemory(matches))
        }
    }
}

// ============================================================================
// SORTING AND DEDUPLICATION
// ============================================================================

// In src/matcher/grid_orchestrator.rs

/// Phase 5.5: Sort and deduplicate grid matches
pub fn execute_sorting(
    input: PipelineData<SequenceMatch>,
    memory_manager: &MemoryManager,
    temp_manager: &TempFileManager,
) -> Result<PipelineData<SequenceMatch>> {
    let input_count = input.count();
    info!("Sorting and deduplicating {} grid matches...", input_count);
    
    match input {
        PipelineData::InMemory(mut matches) => {
            // Check if we can sort in memory
            let memory_state = memory_manager.monitor_memory_thresholds();
            
            if matches!(memory_state, MemoryState:: AboveNormal | MemoryState::Normal | MemoryState::BelowNormal) {
                // Sort in memory using parallel sort
                info!("Sorting {} matches in memory", matches.len());
                let sort_start = Instant::now();
                
                radix_sort_matches(&mut matches);
                
                // Deduplicate
                let before_dedup = matches.len();
                matches.dedup_by(|a, b| {
                    a.source_sequence == b.source_sequence && 
                    a.target_sequence == b.target_sequence
                });
                
                let removed = before_dedup - matches.len();
                info!("In-memory sort complete in {:?}: {} unique matches (removed {} duplicates, {:.1}% reduction)",
                    sort_start.elapsed(), matches.len(), removed,
                    (removed as f64 / before_dedup as f64) * 100.0);
                
                // Check memory again after sorting
                let post_sort_state = memory_manager.monitor_memory_thresholds();
                if matches!(post_sort_state, MemoryState::High) {
                    // Memory pressure developed during sorting - spill to disk
                    info!("Memory pressure after sorting, spilling to disk");
                    
                    let sorted_path = temp_manager.get_temp_dir().join("grid_matches_sorted.tmp");
                    let file = File::create(&sorted_path)?;
                    let mut writer = BinaryWriter::new(file, true, DataType::SequenceMatch)?;
                    
                    for match_item in &matches {
                        writer.write_item(match_item)?;
                    }
                    writer.flush()?;
                    
                    Ok(PipelineData::OnDisk { 
                        path: sorted_path, 
                        count: matches.len() 
                    })
                } else {
                    // Memory still OK - keep in memory
                    Ok(PipelineData::InMemory(matches))
                }
            } else {
                // Memory pressure before sorting - spill first, then use external sort
                info!("Memory pressure detected, spilling to disk for external sort");
                
                // Write unsorted matches to temp file
                let temp_path = temp_manager.get_temp_dir().join("grid_matches_unsorted.tmp");
                let file = File::create(&temp_path)?;
                let mut writer = BinaryWriter::new(file, true, DataType::SequenceMatch)?;
                
                for match_item in matches {
                    writer.write_item(&match_item)?;
                }
                writer.flush()?;
                
                // Now use external sort
                let sorted_path = temp_manager.get_temp_dir().join("grid_matches_sorted.tmp");
                let unique_count = sort_and_dedup_matches_file(
                    &temp_path,
                    &sorted_path,
                    memory_manager
                )?;
                
                // Clean up unsorted temp file
                let _ = std::fs::remove_file(temp_path);
                
                Ok(PipelineData::OnDisk { 
                    path: sorted_path, 
                    count: unique_count 
                })
            }
        }
        
        PipelineData::OnDisk { path, count } => {
            // Already on disk - use existing external sort
            info!("Sorting {} matches from disk using external sort", count);
            
            let sorted_path = temp_manager.get_temp_dir().join("grid_matches_sorted.tmp");
            let unique_count = sort_and_dedup_matches_file(
                &path,
                &sorted_path,
                memory_manager
            )?;
            
            let removed = count.saturating_sub(unique_count);
            info!("External sort complete: {} unique matches (removed {} duplicates, {:.1}% reduction)",
                unique_count, removed, (removed as f64 / count as f64) * 100.0);
            
            Ok(PipelineData::OnDisk { 
                path: sorted_path, 
                count: unique_count 
            })
        }
    }
}

// ============================================================================
// CLUSTERING
// ============================================================================

pub fn execute_clustering(
    orchestrator: &MatcherOrchestrator,
    input: PipelineData<SequenceMatch>,
    streaming_config: &AlNaqlConfig,
    memory_manager: &MemoryManager,
    temp_manager: &TempFileManager,
    progress_context: Option<&ProgressContext>,
) -> Result<PipelineData<SequenceCluster>> {
    info!("Starting clustering phase...");
    
    let clusters_path = temp_manager.get_temp_dir().join("clusters.tmp");
    let input_count = input.count();
    
    // Calculate expected chunks (from existing code)
    let chunk_size = streaming_config.execution.grid_cluster_chunks;
    let expected_chunks = ((input_count / chunk_size) + if input_count % chunk_size > 0 { 1 } else { 0 }) * 12 / 10;
    info!("Expecting approximately {} chunks for streaming", expected_chunks);
    
    match input {
        PipelineData::InMemory(matches) => {
            // For now, write to disk and use existing clustering
            // (Later you could add in-memory clustering if desired)
            info!("Writing {} matches to disk for clustering", matches.len());
            
            let matches_path = temp_manager.get_temp_dir().join("matches_for_clustering.tmp");
            let file = File::create(&matches_path)?;
            let mut writer = BinaryWriter::new(file, true, DataType::SequenceMatch)?;
            for m in matches {
                writer.write_item(&m)?;
            }
            writer.flush()?;
            
            // Use existing disk-based clustering
            let cluster_count = clusters::streaming::stream_clustering(
                &matches_path,
                &clusters_path,
                streaming_config,
                &orchestrator.source_storage,
                memory_manager,
                progress_context,
                expected_chunks
            )?;
            
            // Clean up temp file
            let _ = std::fs::remove_file(matches_path);
            
            Ok(PipelineData::OnDisk { path: clusters_path, count: cluster_count })
        }
        PipelineData::OnDisk { path, count } => {
            // Already on disk - use existing streaming clustering
            info!("Clustering {} matches from disk", count);
            
            let cluster_count = clusters::streaming::stream_clustering(
                &path,
                &clusters_path,
                streaming_config,
                &orchestrator.source_storage,
                memory_manager,
                progress_context,
                expected_chunks
            )?;
            
            Ok(PipelineData::OnDisk { path: clusters_path, count: cluster_count })
        }
    }
}
// ============================================================================
// VALIDATION
// ============================================================================

pub fn execute_validation(
    orchestrator: &MatcherOrchestrator,
    input: PipelineData<SequenceCluster>,
    streaming_config: &AlNaqlConfig,
    memory_manager: &MemoryManager,  // ADD THIS PARAMETER
    temp_manager: &TempFileManager,
    progress_context: Option<&ProgressContext>,
) -> Result<PipelineData<ReconstructedSegment>> {  // CHANGE TO ReconstructedSegment
    info!("Starting validation phase...");
    
    let validated_path = temp_manager.get_temp_dir().join("validated.tmp");
    
    // Get text classifier
    let text_classifier_guard = orchestrator.text_classifier.read()
        .expect("Failed to acquire read lock on text classifier");
    
    match input {
        PipelineData::InMemory(clusters) => {
            // Check if we can validate in memory
            let memory_state = memory_manager.monitor_memory_thresholds();
            
            if matches!(memory_state, MemoryState::Normal | MemoryState::BelowNormal) {
                // Validate in memory using existing function
                info!("Validating {} clusters in memory", clusters.len());
                
                let validated = validation::validate_without_merging(
                    &clusters,
                    &orchestrator.source_storage,
                    &orchestrator.target_storage,
                    &orchestrator.similarity_calc,
                    streaming_config,
                    &*text_classifier_guard,
                    orchestrator.ngram_type,
                )?;
                
                drop(text_classifier_guard);
                
                let validated_count = validated.len();
                info!("In-memory validation complete: {} validated segments", validated_count);
                
                // Check memory after validation (text segments can be large)
                let post_validation_state = memory_manager.monitor_memory_thresholds();
                if matches!(post_validation_state, MemoryState::High) {
                    // Memory pressure after validation - spill to disk
                    info!("Memory pressure after validation, spilling to disk");
                    
                    let file = File::create(&validated_path)?;
                    // Use ReconstructedSegment DataType
                    let mut writer = BinaryWriter::new(file, true, DataType::ReconstructedSegment)?;
                    
                    for segment in &validated {
                        writer.write_item(segment)?;
                    }
                    writer.flush()?;
                    
                    Ok(PipelineData::OnDisk {
                        path: validated_path,
                        count: validated_count,
                    })
                } else {
                    // Keep in memory
                    Ok(PipelineData::InMemory(validated))
                }
            } else {
                // Memory pressure - write clusters to disk first, then use streaming
                info!("Memory pressure detected, using disk-based validation");
                
                let clusters_temp = temp_manager.get_temp_dir().join("clusters_for_validation.tmp");
                let file = File::create(&clusters_temp)?;
                let mut writer = BinaryWriter::new(file, true, DataType::SequenceCluster)?;
                
                for cluster in clusters {
                    writer.write_item(&cluster)?;
                }
                writer.flush()?;
                
                // Use streaming validation
                let validated_count = validation::stream_validation(
                    &clusters_temp,
                    &validated_path,
                    streaming_config,
                    &orchestrator.source_storage,
                    &orchestrator.target_storage,
                    &orchestrator.similarity_calc,
                    &*text_classifier_guard,
                    progress_context
                )?;
                
                drop(text_classifier_guard);
                
                // Clean up temp file
                let _ = std::fs::remove_file(clusters_temp);
                
                info!("Disk-based validation complete: {} validated segments", validated_count);
                
                Ok(PipelineData::OnDisk {
                    path: validated_path,
                    count: validated_count,
                })
            }
        }
        
        PipelineData::OnDisk { path, count } => {
            // Already on disk - use streaming validation
            info!("Validating {} clusters from disk", count);
            
            let validated_count = validation::stream_validation(
                &path,
                &validated_path,
                streaming_config,
                &orchestrator.source_storage,
                &orchestrator.target_storage,
                &orchestrator.similarity_calc,
                &*text_classifier_guard,
                progress_context
            )?;
            
            drop(text_classifier_guard);
            
            info!("Disk-based validation complete: {} validated segments", validated_count);
            
            Ok(PipelineData::OnDisk {
                path: validated_path,
                count: validated_count,
            })
        }
    }
}

// ============================================================================
// FINAL MERGE AND OUTPUT
// ============================================================================

/// Phase 8: Final merging - decides whether to use streaming or in-memory
/// based on dataset size

pub fn execute_final_merge(
    orchestrator: &MatcherOrchestrator,
    input: PipelineData<ReconstructedSegment>,
    output_path: Option<&Path>,
    streaming_config: &AlNaqlConfig,
    temp_manager: &TempFileManager,
    start_time: Instant,
    total_comparisons: u64,
    progress_context: Option<&ProgressContext>,
) -> Result<usize> {
    
    match input {
        PipelineData::InMemory(segments) => {
            // Data is already in memory - use in-memory merge directly
            let segment_count = segments.len();
            info!("Final merge: {} segments already in memory, using in-memory merge", segment_count);
            
            // Write to temp file for the in-memory merge function (it expects a file path)
            let temp_path = temp_manager.get_temp_dir().join("segments_for_merge.tmp");
            let file = File::create(&temp_path)?;
            let mut writer = BinaryWriter::new(file, true, DataType::ReconstructedSegment)?;
            
            for segment in segments {
                writer.write_item(&segment)?;
            }
            writer.flush()?;
            
            // Use the in-memory merge function
            let result = execute_final_merge_in_memory(
                orchestrator,
                &temp_path,
                output_path,
                streaming_config,
                temp_manager,
                start_time,
                total_comparisons,
            )?;
            
            // Clean up temp file
            let _ = std::fs::remove_file(temp_path);
            
            Ok(result)
        }
        
        PipelineData::OnDisk { path, count: _ } => {
            // Data is on disk - use existing logic to decide streaming vs in-memory
            
            // Check file size to decide whether to stream
            let metadata = std::fs::metadata(&path)?;
            let file_size_mb = metadata.len() / (1024 * 1024);
            
            // Estimate match count (rough estimate: ~100 bytes per match in compressed binary format)
            let estimated_matches = metadata.len() / 80;
            
            // Reduce threshold in parallel mode
            let effective_threshold = if std::env::var("ALNAQL_PARALLEL_MODE").is_ok() {
                10_000  // Much smaller in parallel mode
            } else {
                50_000  // Use configured threshold
            };
            
            info!("Final merge input: {} MB, ~{} matches (threshold: {})",
                file_size_mb, estimated_matches, effective_threshold);
            
            if estimated_matches > effective_threshold as u64 {
                // Use streaming for large datasets
                info!("Dataset exceeds threshold ({} > {}), using streaming merge",
                    estimated_matches, effective_threshold);
                
                streaming_final_merge::execute_streaming(
                    orchestrator,
                    &path,
                    output_path,
                    streaming_config,
                    temp_manager,
                    start_time,
                    progress_context,
                )
            } else {
                // Use original in-memory approach for smaller datasets
                info!("Dataset within threshold ({} <= {}), using in-memory merge",
                    estimated_matches, effective_threshold);
                
                execute_final_merge_in_memory(
                    orchestrator,
                    &path,
                    output_path,
                    streaming_config,
                    temp_manager,
                    start_time,
                    total_comparisons,
                )
            }
        }
    }
}
/// Phase 8: Final merging and output to user-specified path
pub fn execute_final_merge_in_memory(
    orchestrator: &MatcherOrchestrator,
    input_path: &Path,
    output_path: Option<&Path>,
    streaming_config: &AlNaqlConfig,
    temp_manager: &TempFileManager,
    start_time: Instant,
    total_comparisons: u64,
) -> Result<usize> {
    info!("Starting final merging phase...");
    
    // Read all validated matches from the binary temp file
    let final_matches = temp_manager.read_matches(input_path)?;
    
    info!("Loaded {} validated matches for merging", final_matches.len());
    
    // Apply the 4-stage validation pipeline (EXACTLY as in original)
    
    // Stage 1: merged_matches is created and populated
    let mut merged_matches = validation::merge_overlapping_matches(
        &mut final_matches.clone(),
        &orchestrator.source_storage,
        &orchestrator.target_storage,
        streaming_config,
    );
    info!("Stage 1: {} matches after merging overlaps", merged_matches.len());

    // Get text classifier for stages 2 and 4
    let text_classifier_guard = orchestrator.text_classifier.read()
        .expect("Failed to acquire read lock on text classifier");

    // Stage 2: merge_nearby_matches modifies merged_matches IN-PLACE
    validation::merge_nearby_matches(
        &mut merged_matches,  // Pass mutable reference
        streaming_config,
        &orchestrator.source_storage,
        &orchestrator.target_storage,
        &orchestrator.similarity_calc,
        &*text_classifier_guard,
    )?;
    // merged_matches has been modified but the variable still exists!
    info!("Stage 2: {} matches after merging adjacents", merged_matches.len());

    // Stage 3: remove_contained_matches ALSO modifies in-place
    validation::remove_contained_matches(&mut merged_matches);
    info!("Stage 3: {} matches after removing contained", merged_matches.len());

    // Stage 4: Apply final filters directly to the ReconstructedSegments
    let filtered_segments: Vec<ValidatedSegment> = merged_matches
        .into_iter()
        .filter_map(|segment| {
            validation::passes_final_filters(
                segment,
                &orchestrator.similarity_calc,
                streaming_config,
                &*text_classifier_guard,
                streaming_config.generator.ngram_type,
                None,
                &orchestrator.source_storage,
                &orchestrator.target_storage,
            )
        })
        .collect();

    info!("Stage 4: {} matches after final filters", filtered_segments.len());

    // Convert to final results
    let final_results = validation::convert_to_final_results(
        &filtered_segments,
        &orchestrator.source_storage,
        &orchestrator.target_storage,
    )?;

    drop(text_classifier_guard);
    
    // Write final output using orchestrator's write_final_output method
    // (This respects the configuration's output format)
    let final_count = if let Some(path) = output_path {
        orchestrator.write_final_output(&final_results, path, streaming_config)?;
        final_results.len()
    } else {
        info!("No output path specified, {} matches ready", final_results.len());
        final_results.len()
    };
    
    // Report statistics (matching original implementation)
    let elapsed = start_time.elapsed();
    info!("=========================================");
    info!("Grid Streaming Processing Complete!");
    info!("=========================================");
    info!("Total time: {:?}", elapsed);
    info!("Total comparisons: {}", total_comparisons);
    info!("Throughput: {:.2} comparisons/second", 
        total_comparisons as f64 / elapsed.as_secs_f64());
    info!("Final matches: {}", final_count);
    info!("=========================================");
    
    Ok(final_count)
}

// ============================================================================
// CLEANUP
// ============================================================================

/// Clean up temporary files and directories
pub fn cleanup_temp_files(
    temp_manager: TempFileManager,
    run_dir: &Path,
    keep_for_debugging: bool,
) -> Result<()> {
    if keep_for_debugging {
        info!("Keeping temporary files for debugging at {:?}", run_dir);
        return Ok(());
    }
    
    // First, call TempFileManager's cleanup
    if let Err(e) = temp_manager.cleanup() {
        warn!("Error during temp file cleanup: {}", e);
    } else {
        debug!("Temporary files successfully cleaned up");
    }
    
    // Now delete the entire temporary directory
    if run_dir.exists() {
        debug!("Removing temporary directory: {:?}", run_dir);
        match fs::remove_dir_all(run_dir) {
            Ok(_) => info!("Successfully removed temporary directory"),
            Err(e) => warn!("Failed to remove temporary directory: {}", e),
        }
    }
    
    Ok(())
}

// ============================================================================
// RESUME SUPPORT FUNCTIONS
// ============================================================================

/// Execute grid streaming from a specific stage (for resume)
pub fn execute_from_stage(
    orchestrator: &MatcherOrchestrator,
    start_stage: crate::utils::resume_manager::PipelineStage,
    input_file: Option<PathBuf>,
    source_range: Range<u64>,
    target_range: Range<u64>,
    output_path: Option<&Path>,
    run_dir: PathBuf,
    existing_metadata: Option<crate::utils::resume_manager::RunMetadata>,
    progress_context: Option<&ProgressContext>,
) -> Result<usize> {
    use crate::utils::resume_manager::{PipelineStage, ResumeManager};
    
    info!("Executing grid streaming from stage: {:?}", start_stage);
    
    // Update progress display if we're resuming
    if let Some(ctx) = progress_context {
        if start_stage != PipelineStage::GridMatching {
            let phase = match start_stage {
                PipelineStage::Sorting => ProgressPhase::Sorting,
                PipelineStage::Clustering => ProgressPhase::Clustering,
                PipelineStage::Validation => ProgressPhase::Validation,
                PipelineStage::FinalMerge => ProgressPhase::FinalMerge,
                _ => ProgressPhase::GridProcessing,
            };
            ctx.enter_phase(phase);
        }
    }
    
    // Track overall start time for statistics
    let overall_start_time = Instant::now();
    
    // Setup infrastructure
    let (temp_manager, memory_manager, total_comparisons) = setup_infrastructure(
        &source_range,
        &target_range,
        run_dir.clone(),
    )?;
    
    let streaming_config = orchestrator.config.clone();
    
    // Execute pipeline from the appropriate stage
    match start_stage {
        PipelineStage::GridMatching => {
            // Starting fresh - execute all stages
            let source_sequences = extract_source_sequences(&source_range)?;
            let streaming_config = configure_memory_scaling(
                orchestrator, &memory_manager, &source_sequences, &target_range
            )?;
            
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::GridProcessing);
            }
            
            // Grid matching
            let grid_result = execute_grid_matching(
                orchestrator, &source_sequences, target_range, 
                &streaming_config, &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::GridMatching, "grid_matches.tmp", grid_result.count()
            )?;
            
            // Sorting
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Sorting);
            }
            let sorted_result = execute_sorting(grid_result, &memory_manager, &temp_manager)?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Sorting, "grid_matches_sorted.tmp", sorted_result.count()
            )?;
            
            // Clustering
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Clustering);
            }
            let clustered_result = execute_clustering(
                orchestrator, sorted_result, &streaming_config, 
                &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Clustering, "clusters.tmp", clustered_result.count()
            )?;
            
            // Validation
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Validation);
            }
            let validated_result = execute_validation(
                orchestrator, clustered_result, &streaming_config,
                &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Validation, "validated.tmp", validated_result.count()
            )?;
            
            // Final merge
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::FinalMerge);
            }
            let final_count = execute_final_merge(
                orchestrator, validated_result, output_path, 
                &streaming_config, &temp_manager, overall_start_time, total_comparisons, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::FinalMerge, "final", final_count
            )?;
            
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(final_count);
        },
        
        PipelineStage::Sorting => {
            // Resume from sorting
            let input_path = input_file.expect("Input file required for sorting stage");
            let matches_count = existing_metadata
                .and_then(|m| m.stage_counts.get("grid_matching").copied())
                .unwrap_or(0);
            
            // Create PipelineData from checkpoint
            let sorted_result = execute_sorting(
                PipelineData::OnDisk { path: input_path, count: matches_count },
                &memory_manager,
                &temp_manager
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Sorting, "grid_matches_sorted.tmp", sorted_result.count()
            )?;
            
            // Continue with clustering
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Clustering);
            }
            let clustered_result = execute_clustering(
                orchestrator, sorted_result, &streaming_config,
                &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Clustering, "clusters.tmp", clustered_result.count()
            )?;
            
            // Continue with validation
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Validation);
            }
            let validated_result = execute_validation(
                orchestrator, clustered_result, &streaming_config,
                &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Validation, "validated.tmp", validated_result.count()
            )?;
            
            // Final merge
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::FinalMerge);
            }
            let final_count = execute_final_merge(
                orchestrator, validated_result, output_path,
                &streaming_config, &temp_manager, overall_start_time, total_comparisons, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::FinalMerge, "final", final_count
            )?;
            
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(final_count);
        },
        
        PipelineStage::Clustering => {
            // Resume from clustering
            let input_path = input_file.expect("Input file required for clustering stage");
            let unique_matches = existing_metadata
                .and_then(|m| m.stage_counts.get("sorting").copied())
                .unwrap_or(0);
            
            // Create PipelineData from checkpoint
            let clustered_result = execute_clustering(
                orchestrator,
                PipelineData::OnDisk { path: input_path, count: unique_matches },
                &streaming_config,
                &memory_manager,
                &temp_manager,
                progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Clustering, "clusters.tmp", clustered_result.count()
            )?;
            
            // Continue with validation
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::Validation);
            }
            let validated_result = execute_validation(
                orchestrator, clustered_result, &streaming_config,
                &memory_manager, &temp_manager, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Validation, "validated.tmp", validated_result.count()
            )?;
            
            // Final merge
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::FinalMerge);
            }
            let final_count = execute_final_merge(
                orchestrator, validated_result, output_path,
                &streaming_config, &temp_manager, overall_start_time, total_comparisons, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::FinalMerge, "final", final_count
            )?;
            
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(final_count);
        },
        
        PipelineStage::Validation => {
            // Resume from validation
            let input_path = input_file.expect("Input file required for validation stage");
            let cluster_count = existing_metadata
                .and_then(|m| m.stage_counts.get("clustering").copied())
                .unwrap_or(0);
            
            // Create PipelineData from checkpoint
            let validated_result = execute_validation(
                orchestrator,
                PipelineData::OnDisk { path: input_path, count: cluster_count },
                &streaming_config,
                &memory_manager,
                &temp_manager,
                progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::Validation, "validated.tmp", validated_result.count()
            )?;
            
            // Final merge
            if let Some(ctx) = progress_context {
                ctx.enter_phase(ProgressPhase::FinalMerge);
            }
            let final_count = execute_final_merge(
                orchestrator, validated_result, output_path,
                &streaming_config, &temp_manager, overall_start_time, total_comparisons, progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::FinalMerge, "final", final_count
            )?;
            
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(final_count);
        },
        
        PipelineStage::FinalMerge => {
            // Resume from final merge
            let input_path = input_file.expect("Input file required for final merge stage");
            let validated_count = existing_metadata
                .and_then(|m| m.stage_counts.get("validation").copied())
                .unwrap_or(0);
            
            // Create PipelineData from checkpoint and execute final merge
            let final_count = execute_final_merge(
                orchestrator,
                PipelineData::OnDisk { path: input_path, count: validated_count },
                output_path,
                &streaming_config,
                &temp_manager,
                overall_start_time,
                total_comparisons,
                progress_context
            )?;
            ResumeManager::update_stage_completion(
                &run_dir, PipelineStage::FinalMerge, "final", final_count
            )?;
            
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(final_count);
        },
        
        PipelineStage::Complete => {
            info!("Pipeline already complete, nothing to do");
            cleanup_temp_files(temp_manager, &run_dir, false)?;
            return Ok(0);
        },
    }
}

/// Sort a file of JSON matches by source_sequence then target_sequence with deduplication
fn sort_and_dedup_matches_file(
    input_path: &Path,
    output_path: &Path,
    memory_manager: &MemoryManager,
) -> Result<usize> {
    
    let start_time = Instant::now();
    
    // Determine chunk size based on available memory
    let memory_state = memory_manager.monitor_memory_thresholds();
    let chunk_size = match memory_state {
        MemoryState::Low => 20_000_000,
        MemoryState::BelowNormal => 10_000_000,
        MemoryState::Normal => 5_000_000,
        _ => 2_000_000,
    };
    
    info!("Sorting: using chunk size of {} matches (memory state: {})", 
        chunk_size, memory_state);
    
    // Phase 1: Split into sorted chunks
    let input_file = File::open(input_path)?;
    let mut reader = BinaryReader::new(input_file)?;

    let mut chunk_files = Vec::new();
    let mut current_chunk = Vec::with_capacity(chunk_size);
    let mut total_matches: usize = 0;

    while let Some(match_item) = reader.read_item::<SequenceMatch>()? {
        current_chunk.push(match_item);
        total_matches += 1;
        
        if current_chunk.len() >= chunk_size {
            let chunk_path = write_sorted_chunk(&current_chunk, chunk_files.len())?;
            chunk_files.push(chunk_path);
            current_chunk.clear();
            
            info!("Sorting: created chunk {} with {} matches ({} total matches read)", 
                chunk_files.len(), chunk_size, total_matches);
        }
    }
    
    if !current_chunk.is_empty() {
        let chunk_path = write_sorted_chunk(&current_chunk, chunk_files.len())?;
        chunk_files.push(chunk_path);
        info!("Sorting: created final chunk {} with {} matches", 
            chunk_files.len(), current_chunk.len());
    }
    
    info!("Sorting: created {} sorted chunks from {} total matches, starting merge with deduplication...", 
        chunk_files.len(), total_matches);
    
    // Phase 2: K-way merge with deduplication
    let unique_matches = merge_sorted_chunks_with_dedup(chunk_files, output_path)?;
    
    let duplicates_removed = total_matches.saturating_sub(unique_matches);
    let dedup_percent = if total_matches > 0 {
        (duplicates_removed as f64 / total_matches as f64) * 100.0
    } else {
        0.0
    };
    
    info!("Sorting complete: {} unique matches (removed {} duplicates, {:.1}% reduction) in {:?}",
        unique_matches, duplicates_removed, dedup_percent, start_time.elapsed());
    
    Ok(unique_matches)
}

fn write_sorted_chunk(
    matches: &[SequenceMatch],
    chunk_id: usize,
) -> Result<PathBuf> {
    use std::env;
    
    // Sort the chunk using radix sort
    let mut sorted = matches.to_vec();
    radix_sort_matches(&mut sorted);
    
    // Create temp directory if needed
    let temp_dir = env::temp_dir().join("alnaql_sort");
    std::fs::create_dir_all(&temp_dir)?;
    
    let chunk_path = temp_dir.join(format!("chunk_{}.bin", chunk_id));
    let file = File::create(&chunk_path)?;
    
    let mut writer = BinaryWriter::new(file, true, DataType::SequenceMatch)?;
    
    for match_item in &sorted {
        writer.write_item(&match_item)?;
    }
    
    writer.flush()?;
    
    debug!("Wrote sorted chunk {} with {} matches", chunk_id, sorted.len());
    Ok(chunk_path)
}

fn merge_sorted_chunks_with_dedup(
    chunk_files: Vec<PathBuf>,
    output_path: &Path,
) -> Result<usize> {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    
    #[derive(Clone)]
    struct HeapEntry {
        source_seq: u64,
        target_seq: u64,
        reader_idx: usize,
        match_data: SequenceMatch,
    }
    
    impl Eq for HeapEntry {}
    
    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.source_seq == other.source_seq && self.target_seq == other.target_seq
        }
    }
    
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            (self.source_seq, self.target_seq).cmp(&(other.source_seq, other.target_seq))
        }
    }
    
    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    
    info!("Starting merge of {} sorted chunk files with inline deduplication", chunk_files.len());
    
    // Open all chunk files
    let mut readers = Vec::new();
    for path in &chunk_files {
        let file = File::open(path)?;
        let reader = BinaryReader::new(file)?;
        
        if reader.data_type() != DataType::SequenceMatch {
            return Err(Error::storage(
                format!("Expected SequenceMatch data in chunk file, found {:?}", reader.data_type())
            ));
        }
        
        readers.push(reader);
    }
    
    debug!("Successfully opened {} chunk readers", readers.len());
    
    // Create output writer
    let output_file = File::create(output_path)?;
    let mut writer = BinaryWriter::new(
        output_file, 
        true,
        DataType::SequenceMatch
    )?;
    
    // Initialize heap with first element from each reader
    let mut heap = BinaryHeap::new();
    
    for (reader_idx, reader) in readers.iter_mut().enumerate() {
        if let Some(match_item) = reader.read_item::<SequenceMatch>()? {
            heap.push(Reverse(HeapEntry {
                source_seq: match_item.source_sequence,
                target_seq: match_item.target_sequence,
                reader_idx,
                match_data: match_item,
            }));
        }
    }
    
    // Track last written match for inline deduplication
    let mut last_written: Option<(u64, u64)> = None;
    let mut unique_count = 0;
    let mut duplicate_count = 0;
    
    // K-way merge with inline deduplication
    while let Some(Reverse(entry)) = heap.pop() {
        // Check if this is a duplicate of the last written match
        let is_duplicate = if let Some((last_src, last_tgt)) = last_written {
            entry.source_seq == last_src && entry.target_seq == last_tgt
        } else {
            false
        };
        
        if !is_duplicate {
            // Write this match
            writer.write_item(&entry.match_data)?;
            last_written = Some((entry.source_seq, entry.target_seq));
            unique_count += 1;
            
            // Periodic flush
            if unique_count % 100_000 == 0 {
                writer.flush()?;
                debug!("Merged {} unique matches so far (skipped {} duplicates)", 
                       unique_count, duplicate_count);
            }
        } else {
            duplicate_count += 1;
        }
        
        // Read next item from the same reader
        if let Some(next_match) = readers[entry.reader_idx].read_item::<SequenceMatch>()? {
            heap.push(Reverse(HeapEntry {
                source_seq: next_match.source_sequence,
                target_seq: next_match.target_sequence,
                reader_idx: entry.reader_idx,
                match_data: next_match,
            }));
        }
    }
    
    writer.flush()?;
    
    info!("Merge complete: {} unique matches written ({} duplicates skipped)", 
          unique_count, duplicate_count);
    
    // Clean up chunk files
    for path in chunk_files {
        let _ = std::fs::remove_file(path);
    }
    
    Ok(unique_count)
}
/// Radix sort for SequenceMatch
/// Radix sort for SequenceMatch
#[inline]
fn radix_sort_matches(matches: &mut Vec<SequenceMatch>) {
    if matches.len() < 1000 {
        matches.sort_unstable_by_key(|m| (m.source_sequence, m.target_sequence));
        return;
    }
    
    // Create sortable tuples: (key, original_match)
    let mut tuples: Vec<(u128, SequenceMatch)> = matches
        .iter()
        .map(|m| {
            let key = ((m.source_sequence as u128) << 64) | (m.target_sequence as u128);
            (key, m.clone())
        })
        .collect();
    
    // Sort by the u128 key
    radsort::sort_by_key(&mut tuples, |t| t.0);
    
    // Extract sorted matches
    *matches = tuples.into_iter().map(|(_, m)| m).collect();
}