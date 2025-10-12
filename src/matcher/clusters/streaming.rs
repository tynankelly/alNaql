// src/matcher/clusters/streaming.rs

use std::fs::File;
use std::path::Path;
use log::{info, debug, trace};
use rayon::prelude::*;
use crossbeam_channel::bounded;
use std::thread;
use std::time::Duration;

use crate::error::Result;
use crate::storage::LMDBStorage;
use crate::config::AlNaqlConfig;
use crate::utils::resource_coordinator::MemoryManager;
use crate::utils::progress_manager::ProgressContext;
use crate::matcher::types::{SequenceMatch, SequenceCluster};
use crate::utils::binary_io::{BinaryReader, BinaryWriter, DataType};

use super::chains::find_dense_regions;

/// Stream clustering with binary I/O and parallel processing
pub fn stream_clustering(
    input_path: &Path,
    output_path: &Path,
    config: &AlNaqlConfig,
    source_storage: &LMDBStorage,
    memory_manager: &MemoryManager,
    progress_context: Option<&ProgressContext>,
    expected_chunks: usize,
) -> Result<usize> {
    info!("Starting bounded parallel clustering with binary I/O");
    
    // Use chunk size from config
    let chunk_size = config.execution.grid_cluster_chunks;
    
    // Get parallelism from config or default to CPU cores
    let parallel_chunks = config.execution.parallel_clustering_chunks
        .unwrap_or_else(|| rayon::current_num_threads());
    
    info!("Using chunk size of {} from config, processing {} chunks in parallel", 
        chunk_size, parallel_chunks);
    
    // Create bounded channel (2x for buffering)
    let (chunk_sender, chunk_receiver) = bounded(parallel_chunks * 2);
    
    // Clone what we need for the reader thread
    let reader_max_gap = config.matcher.max_gap;
    let input_path = input_path.to_owned();
    
    // Spawn reader thread with binary reading
    let reader_handle = thread::spawn(move || {
        read_chunks_streaming_binary(
            &input_path,
            chunk_size,
            reader_max_gap,
            chunk_sender
        )
    });
    
    // Create binary writer for output
    let output_file = File::create(output_path)?;
    let mut writer = BinaryWriter::new(
        output_file,
        true, // Always compress clustering output
        DataType::SequenceCluster
    )?;
    
    let mut total_clusters_written = 0;
    let mut chunks_processed = 0;
    
    loop {
        // Collect up to parallel_chunks chunks
        let mut batch: Vec<(usize, Vec<SequenceMatch>)> = Vec::new();
        
        for _ in 0..parallel_chunks {
            match chunk_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk_data) => {
                    batch.push((chunks_processed + batch.len(), chunk_data));
                }
                Err(_) => break, // Timeout or channel closed
            }
        }
        
        if batch.is_empty() {
            // Check if reader is done
            if reader_handle.is_finished() {
                break;
            }
            continue; // Wait for more chunks
        }
        
        let batch_size = batch.len();
        
        // Check memory state before processing batch
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == crate::utils::resource_coordinator::MemoryState::High {
            info!("High memory detected, processing batch sequentially");
            // Process sequentially to reduce memory pressure
            for (_chunk_idx, chunk) in batch {
                let clusters = find_dense_regions(&chunk, config, Some(source_storage), None)?;
                
                // Write clusters using binary format
                for cluster in clusters {
                    writer.write_item(&cluster)?;
                    total_clusters_written += 1;
                }
            }
        } else {
            // Process batch in parallel
            debug!("Processing batch of {} chunks in parallel", batch_size);
            
            let batch_clusters: Vec<Vec<SequenceCluster>> = batch
                .par_iter()
                .map(|(_, chunk)| {
                    find_dense_regions(chunk, config, Some(source_storage), None)
                        .unwrap_or_else(|e| {
                            debug!("Error processing chunk: {}", e);
                            Vec::new()
                        })
                })
                .collect();
            
            // Write all clusters from batch using binary format
            for chunk_clusters in batch_clusters {
                for cluster in chunk_clusters {
                    writer.write_item(&cluster)?;
                    total_clusters_written += 1;
                }
            }
        }
        
        chunks_processed += batch_size;
        
        // Periodic flush
        if chunks_processed % 10 == 0 {
            writer.flush()?;
        }
        
        // Update progress
        if let Some(ctx) = progress_context {
            // Don't show more than 100% even if we process more chunks than expected
            ctx.update_progress(
                chunks_processed as u64,
                expected_chunks as u64,  // ← Use estimate instead of 0
                Some(format!("Processed {} chunks | {} clusters found", 
                    chunks_processed, total_clusters_written))
            );
        }

    }
    
    // Wait for reader thread to complete and get both counts
    let (total_read, chunks_sent) = reader_handle.join()
        .map_err(|_| crate::error::Error::Storage("Reader thread panicked".to_string()))??;
    
    // Final flush
    writer.flush()?;
    
    // Log output file size
    if let Ok(metadata) = std::fs::metadata(output_path) {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        info!("Clustering output: {} candidates in {:.2} MB (compressed binary)",
            total_clusters_written, size_mb);
    }
    
    info!("Clustering complete: {} matches read, {} candidates written in {} chunks (reader sent {} chunks)", 
        total_read, total_clusters_written, chunks_processed, chunks_sent);
    
    Ok(total_clusters_written)
}

/// Reader thread function that produces chunks with binary input
fn read_chunks_streaming_binary(
    input_path: &Path,
    chunk_size: usize,
    max_gap: usize,
    sender: crossbeam_channel::Sender<Vec<SequenceMatch>>,
) -> Result<(usize, usize)> {
    // Open file and create binary reader
    let file = File::open(input_path)?;
    let mut reader = BinaryReader::new(file)?;
    
    // Verify we're reading the right type
    if reader.data_type() != DataType::SequenceMatch {
        return Err(crate::error::Error::storage(
            format!("Expected SequenceMatch data, found {:?}", reader.data_type())
        ));
    }
    
    let mut total_read = 0;
    let mut current_chunk: Vec<SequenceMatch> = Vec::with_capacity(chunk_size);
    let mut overlap_buffer: Vec<SequenceMatch> = Vec::new();
    let mut chunks_sent = 0;
    
    // Read matches from binary format
    while let Some(sequence_match) = reader.read_item::<SequenceMatch>()? {
        current_chunk.push(sequence_match);
        total_read += 1;
        
        // Process chunk when it reaches the target size
        if current_chunk.len() >= chunk_size {
            // Add overlap buffer from previous chunk to the beginning
            if !overlap_buffer.is_empty() {
                debug!("Adding {} matches from overlap buffer", overlap_buffer.len());
                let mut combined_chunk = overlap_buffer.clone();
                combined_chunk.append(&mut current_chunk);
                current_chunk = combined_chunk;
                overlap_buffer.clear();
            }
            
            // Find natural boundary for clustering
            let boundary_index = find_cluster_boundary(&current_chunk, max_gap);
            
            trace!("Reader chunk {}: {} matches (boundary at {})", 
                chunks_sent + 1, current_chunk.len(), boundary_index);
            
            // Split the chunk and create owned copies
            let process_chunk: Vec<SequenceMatch> = current_chunk[..boundary_index].to_vec();
            overlap_buffer = current_chunk[boundary_index..].to_vec();
            
            // Clear and shrink current_chunk
            current_chunk.clear();
            current_chunk.shrink_to_fit();
            
            // Send chunk - this will block if channel is full (backpressure!)
            sender.send(process_chunk)
                .map_err(|e| crate::error::Error::Storage(
                    format!("Failed to send chunk: {}", e)
                ))?;
            
            chunks_sent += 1;
        }
    }
    
    // Process remaining matches
    if !overlap_buffer.is_empty() {
        current_chunk = overlap_buffer;
    }
    
    if !current_chunk.is_empty() {
        info!("Reader sending final chunk: {} matches", current_chunk.len());
        sender.send(current_chunk)
            .map_err(|e| crate::error::Error::Storage(
                format!("Failed to send final chunk: {}", e)
            ))?;
        chunks_sent += 1;
    }
    
    info!("Reader complete: read {} matches, sent {} chunks", total_read, chunks_sent);
    Ok((total_read, chunks_sent))
}
/// Find a natural boundary in a chunk of SequenceMatch for clustering
/// This looks for gaps in sequence numbers to identify natural break points
fn find_cluster_boundary(
    matches: &[SequenceMatch],
    max_gap: usize,
) -> usize {
    // [KEEP YOUR EXISTING IMPLEMENTATION - NO CHANGES]
    // For small chunks, process everything - overhead isn't worth it
    if matches.len() <= 100 {
        return matches.len();
    }
    
    // Start searching from 80% through the chunk
    // This ensures we process most data while looking for boundaries
    let start_search = (matches.len() * 8) / 10;
    
    // Extract positions from the search region and sort them
    // We sort because matches might not be perfectly ordered in the chunk
    let mut positions: Vec<u64> = matches[start_search..]
        .iter()
        .map(|m| m.source_sequence)
        .collect();
    
    positions.sort_unstable();
    
    // Look for gaps larger than max_gap in the sorted positions
    for i in 1..positions.len() {
        // Calculate gap between consecutive sorted positions
        let gap = positions[i].saturating_sub(positions[i-1]) as usize;
        
        if gap > max_gap {
            // Found a gap! Now find the exact match at this boundary
            let boundary_position = positions[i];
            
            // Search through original matches to find where this boundary occurs
            for (idx, sequence_match) in matches.iter().enumerate().skip(start_search) {
                if sequence_match.source_sequence >= boundary_position {
                    debug!("Found natural boundary at position {} (gap: {})",
                        boundary_position, gap);
                    return idx;
                }
            }
        }
    }
    
    // No natural boundary found - keep last 100 matches as overlap buffer
    // This ensures we don't lose potential clusters at chunk boundaries
    matches.len().saturating_sub(100)
}