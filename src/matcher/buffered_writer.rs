// src/matcher/buffered_writer.rs
// Memory-buffered writer for parallel match processing with real memory monitoring

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use log::{info, debug, warn};
use sys_info;

use crate::error::{Result, Error};
use crate::matcher::types::ReconstructedSegment;
use crate::utils::tempfile::TempFileManager;
use crate::utils::resource_coordinator::{MemoryManager, MemoryState};
use crate::config::AlNaqlConfig;

// Import our types from matcher/types.rs
use super::types::{
    BufferedChunkMessage,
    WriterState,
    WriterResult,
};

// ============================================================================
// BUFFERED WRITER
// ============================================================================

/// Manages buffered writing of matches to temporary files
pub struct BufferedWriter {
    memory_threshold: usize,
    match_threshold: usize,
    time_threshold: u64,
    temp_manager: TempFileManager,
    receiver: mpsc::Receiver<BufferedChunkMessage>,
    memory_manager: MemoryManager,
}

impl BufferedWriter {
    /// Create a new BufferedWriter from config
    pub fn new(
        config: &AlNaqlConfig,
        temp_manager: TempFileManager,
        receiver: mpsc::Receiver<BufferedChunkMessage>,
    ) -> Self {
        debug!("Initializing BufferedWriter with thresholds: memory={} MB, matches={}, time={}s",
            config.matcher.parallel_buffer_memory_limit / (1024 * 1024),
            config.matcher.parallel_buffer_match_limit,
            config.matcher.parallel_buffer_time_limit
        );
        
        Self {
            memory_threshold: config.matcher.parallel_buffer_memory_limit,
            match_threshold: config.matcher.parallel_buffer_match_limit,
            time_threshold: config.matcher.parallel_buffer_time_limit,
            temp_manager,
            receiver,
            memory_manager: MemoryManager::new(),
        }
    }
    
    /// Main writer loop - receives matches and writes to temp files
    pub fn run(self) -> Result<WriterResult> {
        info!("Starting buffered writer thread");
        let start_time = Instant::now();
        
        // Initialize state
        let mut state = WriterState::new();
        
        // Track some basic stats
        let mut total_messages = 0;
        let mut flush_count = 0;
        
        // Get baseline memory usage before we start buffering
        let mut baseline_memory = self.get_current_memory_usage_mb();
        info!("Baseline memory usage: {} MB", baseline_memory);
        
        // Log initial memory state
        self.memory_manager.log_memory_status();
        
        // Main receive loop
        loop {
            // Check if we should flush based on time
            if self.should_flush_by_time(&state) {
                debug!("Time threshold reached, flushing buffer");
                self.flush_buffer(&mut state, &mut flush_count)?;
                // Update baseline after flush
                baseline_memory = self.get_current_memory_usage_mb();
            }
            
            // Try to receive with a timeout so we can check time periodically
            match self.receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(message) => {
                    total_messages += 1;
                    
                    match message {
                        BufferedChunkMessage::Matches { chunk_index, matches } => {
                            let match_count = matches.len();
                            debug!("Received {} matches from chunk {}", match_count, chunk_index);
                            
                            // Add to buffer
                            state.buffer.extend(matches);
                            
                            // Check ACTUAL memory usage in MB
                            let current_memory = self.get_current_memory_usage_mb();
                            let memory_increase_mb = current_memory.saturating_sub(baseline_memory);
                            
                            debug!("Buffer now has {} matches, memory increased by {} MB from baseline",
                                state.buffer.len(),
                                memory_increase_mb
                            );
                            
                            // Convert MB to bytes for comparison with threshold
                            let memory_increase_bytes = memory_increase_mb * 1024 * 1024;
                            
                            // Check if we should flush based on actual memory
                            if self.should_flush(&state, memory_increase_bytes) {
                                let reason = if memory_increase_bytes >= self.memory_threshold {
                                    "memory threshold"
                                } else {
                                    "match count threshold"
                                };
                                
                                info!("Flushing {} matches due to {} ({} MB increase)",
                                    state.buffer.len(),
                                    reason,
                                    memory_increase_mb
                                );
                                
                                self.flush_buffer(&mut state, &mut flush_count)?;
                                
                                // Update baseline after flush
                                baseline_memory = self.get_current_memory_usage_mb();
                                debug!("New baseline memory after flush: {} MB", baseline_memory);
                            }
                            
                            // Also check memory pressure
                            let memory_state = self.memory_manager.monitor_memory_thresholds();
                            if memory_state == MemoryState::High {
                                warn!("High memory pressure detected, forcing flush");
                                self.flush_buffer(&mut state, &mut flush_count)?;
                                baseline_memory = self.get_current_memory_usage_mb();
                            }
                        },
                        
                        BufferedChunkMessage::Shutdown => {
                            info!("Received shutdown signal");
                            break;
                        }
                    }
                },
                
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Timeout is expected, just check time threshold and continue
                    continue;
                },
                
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    warn!("Channel disconnected, shutting down writer");
                    break;
                }
            }
        }
        
        // Final flush if buffer has content
        if !state.buffer.is_empty() {
            info!("Final flush of {} remaining matches", state.buffer.len());
            self.flush_buffer(&mut state, &mut flush_count)?;
        }
        
        let elapsed = start_time.elapsed();
        info!("Buffered writer complete: {} total matches written to {} files in {:?}",
            state.total_written,
            state.temp_file_paths.len(),
            elapsed
        );
        
        info!("Writer statistics: {} messages processed, {} flushes performed",
            total_messages, flush_count
        );
        
        // Log final memory state
        self.memory_manager.log_memory_status();
        
        Ok(WriterResult {
            temp_file_paths: state.temp_file_paths,
            total_matches_written: state.total_written,
        })
    }
    
    /// Get current process memory usage in MB
    fn get_current_memory_usage_mb(&self) -> usize {
        if let Ok(mem_info) = sys_info::mem_info() {
            let used_mem_mb = ((mem_info.total - mem_info.avail) / 1024) as usize;
            used_mem_mb
        } else {
            debug!("Could not get memory info, using 0");
            0
        }
    }
    
    /// Check if we should flush based on memory or match count
    fn should_flush(&self, state: &WriterState, memory_increase_bytes: usize) -> bool {
        // Check memory threshold
        if memory_increase_bytes >= self.memory_threshold {
            debug!("Memory threshold reached: {} >= {}",
                memory_increase_bytes, self.memory_threshold
            );
            return true;
        }
        
        // Check match count threshold
        if state.buffer.len() >= self.match_threshold {
            debug!("Match count threshold reached: {} >= {}",
                state.buffer.len(), self.match_threshold
            );
            return true;
        }
        
        false
    }
    
    /// Check if we should flush based on time
    fn should_flush_by_time(&self, state: &WriterState) -> bool {
        // Only flush by time if buffer has content
        if state.buffer.is_empty() {
            return false;
        }
        
        let elapsed = state.last_flush.elapsed().as_secs();
        elapsed >= self.time_threshold
    }
    
    /// Flush buffer to a temporary file
    fn flush_buffer(&self, state: &mut WriterState, flush_count: &mut usize) -> Result<()> {
    if state.buffer.is_empty() {
        debug!("Buffer is empty, skipping flush");
        return Ok(());
    }
    
    let flush_start = Instant::now();
    let match_count = state.buffer.len();
    
    // Calculate approximate memory usage before flush
    let approx_memory_mb = (state.buffer.len() * std::mem::size_of::<ReconstructedSegment>()) / (1024 * 1024);
    
    debug!("Starting flush {} of {} matches (~{} MB uncompressed)", 
        *flush_count + 1, match_count, approx_memory_mb);
    
    // Use temp_manager to write matches - NOW USES BINARY WITH COMPRESSION
    let temp_file_path = self.temp_manager
        .write_matches(&state.buffer, *flush_count)
        .map_err(|e| Error::Storage(format!("Failed to write temp file: {}", e)))?;
    
    // Log the file size to show compression benefit
    if let Ok(metadata) = std::fs::metadata(&temp_file_path) {
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        info!("Wrote {} matches to binary file {:?} ({:.2} MB compressed, ~{} MB uncompressed - {:.1}% reduction)",
            match_count, 
            temp_file_path, 
            file_size_mb,
            approx_memory_mb,
            ((approx_memory_mb as f64 - file_size_mb) / approx_memory_mb as f64) * 100.0
        );
    } else {
        debug!("Wrote {} matches to binary file {:?}", match_count, temp_file_path);
    }
    
    // Update state
    state.temp_file_paths.push(temp_file_path);
    state.total_written += match_count;
    state.buffer.clear();
    state.buffer.shrink_to_fit(); // Release memory back to OS
    state.last_flush = Instant::now();
    *flush_count += 1;
    
    let flush_duration = flush_start.elapsed();
    info!("Flush {} complete: {} matches written to compressed binary in {:?}",
        flush_count, match_count, flush_duration
    );
    
    // Force memory reclamation
    drop(Vec::<u8>::with_capacity(0)); // Hint to OS to reclaim
    
    Ok(())
}
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a buffered writer with its channel and thread handle
/// This is a convenience function for easy integration
pub fn create_buffered_writer(
    config: &AlNaqlConfig,
    temp_manager: TempFileManager,
) -> (mpsc::Sender<BufferedChunkMessage>, thread::JoinHandle<Result<WriterResult>>) {
    // Create channel
    let (sender, receiver) = mpsc::channel::<BufferedChunkMessage>();
    
    // Clone config for the thread
    let config_clone = config.clone();
    
    // Spawn writer thread
    let handle = thread::spawn(move || {
        let writer = BufferedWriter::new(&config_clone, temp_manager, receiver);
        writer.run()
    });
    
    (sender, handle)
}

/// Create a bounded channel variant for back-pressure
/// This prevents unlimited buffering in the channel itself
pub fn create_buffered_writer_bounded(
    config: &AlNaqlConfig,
    temp_manager: TempFileManager,
    channel_size: usize,
) -> (mpsc::SyncSender<BufferedChunkMessage>, thread::JoinHandle<Result<WriterResult>>) {
    // Create bounded channel for back-pressure
    let (sender, receiver) = mpsc::sync_channel::<BufferedChunkMessage>(channel_size);
    
    // Clone config for the thread
    let config_clone = config.clone();
    
    // Spawn writer thread with sync_channel receiver
    let handle = thread::spawn(move || {
        let writer = BufferedWriter::new(&config_clone, temp_manager, receiver);
        writer.run()
    });
    
    (sender, handle)
}
