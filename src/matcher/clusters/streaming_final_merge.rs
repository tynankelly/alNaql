// src/matcher/streaming_final_merge.rs
//! Streaming implementation of final merge for large datasets
//! 
//! This module handles the final merge stage when the validated matches
//! are too large to fit in memory. It processes the data in chunks and
//! applies the 4-stage validation pipeline incrementally.

use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Write, BufReader, BufWriter, BufRead};
use std::cmp::min;
use std::time::Instant;
use log::{info, debug, warn};

use crate::error::{Result, Error};
use crate::config::AlNaqlConfig;
use crate::matcher::MatcherOrchestrator;
use crate::matcher::clusters::validation;
use crate::matcher::types::{ReconstructedSegment, ValidatedSegment, FinalMatchResult};
use crate::utils::binary_io::{BinaryReader, DataType};
use crate::utils::progress_manager::ProgressContext;
use crate::utils::tempfile::TempFileManager;

// ============================================================================
// CHUNK READER
// ============================================================================

/// Stream chunks of ReconstructedSegments from a binary file
struct ChunkStreamReader {
    reader: BinaryReader<File>,
    target_chunk_size: usize,
    overlap_buffer: Vec<ReconstructedSegment>,
    finished: bool,
}

impl ChunkStreamReader {
    /// Create a new chunk stream reader
    fn new(input_path: &Path, chunk_size: usize) -> Result<Self> {
        let file = File::open(input_path)?;
        let reader = BinaryReader::new(file)?;
        
        // Verify we're reading the right data type
        if reader.data_type() != DataType::ReconstructedSegment {
            return Err(Error::storage(
                format!("Expected ReconstructedSegment data in {}, found {:?}", 
                    input_path.display(), reader.data_type())
            ));
        }
        
        Ok(ChunkStreamReader {
            reader,
            target_chunk_size: chunk_size,
            overlap_buffer: Vec::new(),
            finished: false,
        })
    }
    
    /// Read the next chunk of matches
    fn read_next_chunk(&mut self, config: &AlNaqlConfig) -> Result<Option<Vec<ReconstructedSegment>>> {
        if self.finished {
            return Ok(None);
        }
        
        let mut chunk = Vec::new();
        
        // Add any overlap from the previous chunk
        if !self.overlap_buffer.is_empty() {
            debug!("Adding {} matches from overlap buffer", self.overlap_buffer.len());
            chunk.extend(self.overlap_buffer.drain(..));
        }
        
        // Read new matches up to target_chunk_size + window
        let window_size = ((self.target_chunk_size as f64) * 
                          config.execution.boundary_window_percent / 100.0) as usize;
        let target_read = self.target_chunk_size + window_size;
        
        while chunk.len() < target_read {
            match self.reader.read_item::<ReconstructedSegment>()? {
                Some(segment) => chunk.push(segment),
                None => {
                    self.finished = true;
                    break;
                }
            }
        }
        
        if chunk.is_empty() {
            return Ok(None);
        }
        
        // If this is the last chunk, return everything
        if self.finished {
            info!("Final chunk: {} matches (EOF reached)", chunk.len());
            return Ok(Some(chunk));
        }
        
        // Find the boundary for this chunk
        let boundary = find_chunk_boundary(&chunk, self.target_chunk_size, config);
        
        // Split the chunk at the boundary
        let to_process = chunk.drain(..boundary).collect::<Vec<_>>();
        self.overlap_buffer = chunk;
        
        info!("Chunk ready: {} matches to process, {} held for next chunk", 
            to_process.len(), self.overlap_buffer.len());
        
        if self.overlap_buffer.len() > self.target_chunk_size / 2 {
            warn!("Large overlap buffer ({} matches) - may indicate insufficient gaps", 
                self.overlap_buffer.len());
        }
        
        Ok(Some(to_process))
    }
}

// ============================================================================
// BOUNDARY DETECTION
// ============================================================================

/// Find a safe boundary to split a chunk of matches
fn find_chunk_boundary(
    matches: &[ReconstructedSegment],
    target_chunk_size: usize,
    config: &AlNaqlConfig,
) -> usize {
    if matches.len() <= target_chunk_size {
        return matches.len();
    }
    
    let safety_multiplier = config.execution.boundary_safety_multiplier;
    let safe_gap_threshold = config.matcher.adjacent_merge_ratio * safety_multiplier;
    let window_percent = config.execution.boundary_window_percent;
    
    let window_size = ((target_chunk_size as f64) * window_percent / 100.0) as usize;
    let search_start = target_chunk_size.saturating_sub(window_size);
    let search_end = min(target_chunk_size + window_size, matches.len());
    
    debug!("Searching for boundary between indices {} and {}", search_start, search_end);
    
    let mut best_boundary = search_end;
    let mut best_gap = 0u64;
    
    for i in search_start..search_end.saturating_sub(1) {
        let current = &matches[i];
        let next = &matches[i + 1];
        
        let source_gap = next.source_start_seq.saturating_sub(current.source_end_seq);
        let target_gap = next.target_start_seq.saturating_sub(current.target_end_seq);
        let min_gap = source_gap.min(target_gap);
        
        let combined_size = (current.source_end_seq - current.source_start_seq) +
                          (next.source_end_seq - next.source_start_seq);
        let required_gap = (combined_size as f64 * safe_gap_threshold) as u64;
        
        if min_gap >= required_gap {
            debug!("Found safe boundary at index {} with gap {}", i + 1, min_gap);
            return i + 1;
        }
        
        if min_gap > best_gap {
            best_gap = min_gap;
            best_boundary = i + 1;
        }
    }
    
    info!("No safe gap found, using largest gap {} at index {}", best_gap, best_boundary);
    best_boundary
}

// ============================================================================
// PIPELINE PROCESSING
// ============================================================================

/// Apply the 4-stage merge pipeline to a chunk of matches
fn apply_merge_pipeline_to_chunk(
    mut chunk: Vec<ReconstructedSegment>,
    orchestrator: &MatcherOrchestrator,
    config: &AlNaqlConfig,
) -> Result<Vec<FinalMatchResult>> {
    
    info!("Applying merge pipeline to chunk of {} matches", chunk.len());
    
    // Stage 1: Merge overlapping matches
    let mut merged_matches = validation::merge_overlapping_matches(
        &mut chunk,
        &orchestrator.source_storage,
        &orchestrator.target_storage,
        config,
    );
    
    debug!("Stage 1: {} matches after merging overlaps", merged_matches.len());
    
    // Stage 2: Merge nearby matches
    {
        let text_classifier_guard = orchestrator.text_classifier.read()
            .expect("Failed to acquire read lock on text classifier");
        
        validation::merge_nearby_matches(
            &mut merged_matches,
            config,
            &orchestrator.source_storage,
            &orchestrator.target_storage,
            &orchestrator.similarity_calc,
            &*text_classifier_guard,
        )?;
    }
    
    debug!("Stage 2: {} matches after merging adjacent", merged_matches.len());
    
    // Stage 3: Remove contained matches
    validation::remove_contained_matches(&mut merged_matches);
    
    debug!("Stage 3: {} matches after removing contained", merged_matches.len());
    
    // Stage 4: Apply final filters
    let filtered_segments: Vec<ValidatedSegment> = {
        let text_classifier_guard = orchestrator.text_classifier.read()
            .expect("Failed to acquire read lock on text classifier");
        
        merged_matches
            .into_iter()
            .filter_map(|segment| {
                validation::passes_final_filters(
                    segment,
                    &orchestrator.similarity_calc,
                    config,
                    &*text_classifier_guard,
                    config.generator.ngram_type,
                    None,
                    &orchestrator.source_storage,
                    &orchestrator.target_storage,
                )
            })
            .collect()
    };
    
    debug!("Stage 4: {} matches after final filters", filtered_segments.len());
    
    // Convert to final results
    let final_results = validation::convert_to_final_results(
        &filtered_segments,
        &orchestrator.source_storage,
        &orchestrator.target_storage,
    )?;
    
    info!("Pipeline complete: {} final matches from {} input", 
        final_results.len(), chunk.len());
    
    Ok(final_results)
}

// ============================================================================
// OUTPUT COMBINATION
// ============================================================================

/// Combine binary chunk files
fn combine_binary_chunks(
    chunk_files: &[PathBuf],
    output_path: &Path,
) -> Result<()> {
    use crate::utils::binary_io::{BinaryReader, BinaryWriter, DataType};
    
    let out_file = File::create(output_path)?;
    let mut writer = BinaryWriter::new(out_file, true, DataType::FinalMatchResult)?;
    
    for chunk_file in chunk_files {
        let file = File::open(chunk_file)?;
        let mut reader = BinaryReader::new(file)?;
        
        while let Some(item) = reader.read_item::<FinalMatchResult>()? {
            writer.write_item(&item)?;
        }
    }
    
    writer.flush()?;
    Ok(())
}

/// Combine JSON chunk files
fn combine_json_chunks(
    chunk_files: &[PathBuf],
    output_path: &Path,
) -> Result<()> {
    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::new(out_file);
    
    for chunk_file in chunk_files {
        let file = File::open(chunk_file)?;
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            writeln!(writer, "{}", line?)?;
        }
    }
    
    writer.flush()?;
    Ok(())
}
// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

pub fn execute_streaming(
    orchestrator: &MatcherOrchestrator,
    input_path: &Path,
    output_path: Option<&Path>,
    config: &AlNaqlConfig,
    temp_manager: &TempFileManager,
    start_time: Instant,
    _progress_context: Option<&ProgressContext>,  // Prefix with _ - not using progress bars here
) -> Result<usize> {
    info!("Starting STREAMING final merge for large dataset");
    
    let mut chunk_reader = ChunkStreamReader::new(input_path, config.execution.final_merge_chunk_size)?;
    let base_output = output_path.unwrap_or(&Path::new("output/matches/final_output"));
    let temp_dir = temp_manager.get_temp_dir();
    
    let mut binary_chunks = Vec::new();
    let mut json_chunks = Vec::new();
    let mut total_written = 0;
    let mut chunk_index = 0;
    
    // Process chunks
    while let Some(chunk) = chunk_reader.read_next_chunk(config)? {
        chunk_index += 1;
        
        let processed = apply_merge_pipeline_to_chunk(chunk, orchestrator, config)?;
        
        if !processed.is_empty() {
            let chunk_base = temp_dir.join(format!("chunk_{:04}", chunk_index));
            let created_paths = orchestrator.write_final_output(&processed, &chunk_base, config)?;
            
            for path in created_paths {
                if path.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == "lz4" || e == "bin")
                    .unwrap_or(false) {
                    binary_chunks.push(path);
                } else {
                    json_chunks.push(path);
                }
            }
            
            total_written += processed.len();
        }
        
        // Log progress for each chunk
        info!("Final merge: processed chunk {}, {} matches written so far", 
            chunk_index, total_written);
    }
    
    info!("Processed {} chunks total, combining into final output...", chunk_index);
    
    // Combine binary chunks if any
    if !binary_chunks.is_empty() {
        let binary_output = base_output.with_extension("bin.lz4");
        info!("Combining {} binary chunks into {:?}", binary_chunks.len(), binary_output);
        combine_binary_chunks(&binary_chunks, &binary_output)?;
    }
    
    // Combine JSON chunks if any
    if !json_chunks.is_empty() {
        let json_output = base_output.with_extension("jsonl");
        info!("Combining {} JSON chunks into {:?}", json_chunks.len(), json_output);
        combine_json_chunks(&json_chunks, &json_output)?;
    }
    
    // Clean up temporary chunk files
    for chunk_file in binary_chunks.iter().chain(json_chunks.iter()) {
        if let Err(e) = std::fs::remove_file(chunk_file) {
            debug!("Failed to remove temp chunk file: {}", e);
        }
    }
    
    let elapsed = start_time.elapsed();
    info!("Streaming merge complete: {} total chunks, {} matches written in {:?}", 
        chunk_index, total_written, elapsed);
    
    Ok(total_written)
}