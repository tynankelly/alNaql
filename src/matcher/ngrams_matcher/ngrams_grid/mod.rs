// src/matcher/ngrams_matcher/ngrams_grid/mod.rs
// Public interface for grid-based n-gram matching

mod dimensions;
mod cell;

// Imports for main functions
use std::sync::Arc;
use std::time::{Instant, Duration};
use rayon::prelude::*;
use log::{info, debug, error};
use std::fs::File;
use std::path::PathBuf;
use crate::utils::binary_io::{BinaryWriter, DataType};
use std::sync::mpsc;
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{Result, Error};
use crate::matcher::types::SequenceMatch;
use crate::config::AlNaqlConfig;
use crate::matcher::similarity_algorithms::similarity::SimilarityCalculator;
use crate::storage::LMDBStorage;
use crate::utils::progress_manager::ProgressContext;
use crate::utils::resource_coordinator::{MemoryManager, MemoryState};
use self::dimensions::calculate_grid_dimensions;
use self::cell::{GridCell, process_cell};

/// Result of grid processing - either in memory or on disk
pub enum GridProcessingResult {
    InMemory(Vec<SequenceMatch>),
    OnDisk { path: PathBuf, count: usize },
}

impl GridProcessingResult {
    pub fn count(&self) -> usize {
        match self {
            GridProcessingResult::InMemory(matches) => matches.len(),
            GridProcessingResult::OnDisk { count, .. } => *count,
        }
    }
}

pub fn process_grid_streaming(
    config: &AlNaqlConfig,
    source_sequences: &[u64],
    source_storage: &LMDBStorage,
    target_storage: &LMDBStorage,
    similarity_calc: &SimilarityCalculator,
    memory_manager: &MemoryManager,
    output_path: PathBuf,
    target_range: std::ops::Range<u64>,
    progress_context: Option<&ProgressContext>,
) -> Result<GridProcessingResult> {
    let start_time = Instant::now();
    
    // Get dataset sizes
    let source_count = source_sequences.len();
    let target_count = (target_range.end - target_range.start) as usize;
    
    info!("Starting streaming grid processing: {} source × {} target ngrams", 
        source_count, target_count);
    
    // Calculate grid dimensions
    let (source_partitions, target_partitions) = calculate_grid_dimensions(
        source_count,
        target_count,
        config.matcher.comparisons_per_cell,
    );
    
    // Create grid cells
    let grid_cells = create_grid_cells(
        source_count,
        target_count,
        source_partitions,
        target_partitions,
    );
    
    let total_cells = grid_cells.len();
    info!("Created {} grid cells ({} × {}), processing ALL in parallel", 
        total_cells, source_partitions, target_partitions);
        
    // Create channel for streaming results
    let (sender, receiver) = mpsc::sync_channel::<Vec<SequenceMatch>>(100);
    
    // Counter for progress tracking
    let matches_found = Arc::new(AtomicUsize::new(0));
    let matches_counter = matches_found.clone();
    let cells_completed = Arc::new(AtomicUsize::new(0));
    let cells_counter = cells_completed.clone();
    
    // Create memory_manager for the writer thread
    let writer_memory_manager = MemoryManager::new();
    
    // Spawn ADAPTIVE writer thread
    let writer_handle = thread::spawn(move || -> Result<GridProcessingResult> {
        // Start optimistically - try to keep everything in memory
        let mut in_memory_matches: Vec<SequenceMatch> = Vec::new();
        let mut writer: Option<BinaryWriter<File>> = None;
        let mut total_written = 0usize;
        let mut last_memory_check = Instant::now();
        
        // Memory check parameters
        const CHECK_INTERVAL_MATCHES: usize = 100_000;  // Check every 100k matches
        const CHECK_INTERVAL_SECS: u64 = 30;  // Or every 30 seconds
        
        loop {
            match receiver.recv() {
                Ok(cell_matches) => {
                    // Only check memory if we haven't already spilled
                    if writer.is_none() && 
                       (in_memory_matches.len() >= CHECK_INTERVAL_MATCHES || 
                        last_memory_check.elapsed() > Duration::from_secs(CHECK_INTERVAL_SECS)) {
                        
                        let memory_state = writer_memory_manager.monitor_memory_thresholds();
                        
                        // Check if we need to start spilling
                        if matches!(memory_state, MemoryState::High) {
                            info!("Memory pressure detected ({:?}), spilling matches to disk", memory_state);
                            info!("Had {} matches in memory before spilling", in_memory_matches.len());
                            
                            // Create the writer NOW
                            let grid_file = File::create(&output_path)?;
                            writer = Some(BinaryWriter::new(grid_file, true, DataType::SequenceMatch)?);
                            
                            // Dump accumulated matches to disk
                            if let Some(ref mut w) = writer {
                                for m in in_memory_matches.drain(..) {
                                    w.write_item(&m)?;
                                    total_written += 1;
                                }
                                w.flush()?; // Flush after dumping memory buffer
                            }
                        }
                        
                        last_memory_check = Instant::now();
                    }
                    
                    // Add new matches
                    if let Some(ref mut w) = writer {
                        // We're in disk mode - write directly
                        for m in cell_matches {
                            w.write_item(&m)?;
                            total_written += 1;
                        }
                        
                        // Periodic flush when writing to disk
                        if total_written % 10_000 == 0 {
                            w.flush()?;
                        }
                    } else {
                        // Still accumulating in memory
                        in_memory_matches.extend(cell_matches);
                    }
                }
                Err(_) => {
                    // Channel closed, finish up
                    break;
                }
            }
        }
        
        // Final processing - return appropriate result
        if let Some(mut w) = writer {
            // We spilled to disk - make sure everything is written
            for m in in_memory_matches {
                w.write_item(&m)?;
                total_written += 1;
            }
            w.flush()?;
            
            info!("Grid processing complete: {} matches written to disk", total_written);
            Ok(GridProcessingResult::OnDisk { 
                path: output_path, 
                count: total_written 
            })
        } else {
            // Everything stayed in memory!
            let match_count = in_memory_matches.len();
            info!("Grid processing complete: {} matches kept in memory", match_count);
            Ok(GridProcessingResult::InMemory(in_memory_matches))
        }
    });
    
    // [REST OF THE FUNCTION REMAINS THE SAME - parallel processing of cells]
    // Process ALL cells in parallel
    let grouped_cells = group_cells_by_target(grid_cells);
    let num_groups = grouped_cells.len();
    info!("Processing {} column groups for cache efficiency", num_groups);

    // Process column groups sequentially, cells within each group in parallel
    let process_result = grouped_cells
        .into_iter()
        .enumerate()
        .try_for_each(|(col_idx, column_cells)| -> Result<()> {
            let cells_in_column = column_cells.len();
            debug!("Processing column group {}/{} with {} cells", 
                   col_idx + 1, num_groups, cells_in_column);
            
            // Process all cells in this column group in parallel
            column_cells
                .into_par_iter()
                .enumerate()
                .try_for_each(|(within_col_idx, cell)| -> Result<()> {
                    // Calculate actual cell index for memory checks
                    let cell_idx = col_idx * cells_in_column + within_col_idx;
                    
                    // Check memory periodically during processing
                    if cell_idx % 10 == 0 {
                        let state = memory_manager.monitor_memory_thresholds();
                        if matches!(state, MemoryState::High) {
                            // Only fail on Critical, let High be handled by writer thread
                            return Err(Error::Storage(
                                "Critical memory state reached during grid processing".to_string()
                            ));
                        }
                    }
                    
                    // Process the cell
                    let cell_matches = process_cell(
                        &cell,
                        source_sequences,
                        source_storage,
                        target_storage,
                        config,
                        similarity_calc,
                    )?;
                    
                    // Track progress
                    if !cell_matches.is_empty() {
                        let match_count = cell_matches.len();
                        matches_counter.fetch_add(match_count, Ordering::Relaxed);
                        
                        // Send results to writer thread
                        sender.send(cell_matches)
                            .map_err(|e| Error::Storage(
                                format!("Failed to send matches to writer thread: {}", e)
                            ))?;
                    }
                    
                    let completed = cells_counter.fetch_add(1, Ordering::Relaxed) + 1;

                    // Update progress bar
                    if let Some(ctx) = progress_context {
                        let current_matches = matches_counter.load(Ordering::Relaxed);
                        ctx.update_grid_progress(completed, total_cells, current_matches);
                    }

                    if completed % 10 == 0 || completed == total_cells {
                        debug!("Progress: {}/{} cells completed", completed, total_cells);
                    }
                    
                    Ok(())
                })
        });
    
    // Handle any errors from processing
    if let Err(e) = process_result {
        error!("Error during grid processing: {}", e);
        return Err(e);
    }
    
    // Close channel (drops sender)
    drop(sender);
    
    // Wait for writer thread to finish and get result
    let result = writer_handle.join()
        .map_err(|_| Error::Storage("Writer thread panicked".to_string()))??;
    
    let elapsed = start_time.elapsed();
    let total_matches = result.count();
    info!("Grid processing finished in {:?} with {} total matches", elapsed, total_matches);
    
    Ok(result)
}


/// Create grid cells based on partitioning
fn create_grid_cells(
    source_count: usize,
    target_count: usize,
    source_partitions: usize,
    target_partitions: usize,
) -> Vec<GridCell> {
    let mut cells = Vec::new();
    
    let source_chunk_size = (source_count + source_partitions - 1) / source_partitions;
    let target_chunk_size = (target_count + target_partitions - 1) / target_partitions;
    
    for row in 0..source_partitions {
        let source_start = row * source_chunk_size;
        
        // Skip this entire row if it starts beyond the source data
        if source_start >= source_count {
            continue;
        }
        
        let source_end = ((row + 1) * source_chunk_size).min(source_count);
        
        for col in 0..target_partitions {
            let target_start = col * target_chunk_size;
            
            // Skip this cell if it starts beyond the target data
            if target_start >= target_count {
                continue;
            }
            
            let target_end = ((col + 1) * target_chunk_size).min(target_count);
            
            cells.push(GridCell {
                id: (row, col),
                source_range: source_start..source_end,
                target_range: (target_start as u64)..(target_end as u64),
            });
        }
    }
    
    cells
}

/// Group cells by target partition for cache efficiency
fn group_cells_by_target(cells: Vec<GridCell>) -> Vec<Vec<GridCell>> {
    let mut groups: Vec<Vec<GridCell>> = Vec::new();
    
    // Find max column index
    let max_col = cells.iter().map(|c| c.id.1).max().unwrap_or(0);
    
    // Group cells by column (target partition)
    for col in 0..=max_col {
        let mut group = Vec::new();
        for cell in &cells {
            if cell.id.1 == col {
                group.push(cell.clone());
            }
        }
        if !group.is_empty() {
            groups.push(group);
        }
    }
    
    groups
}