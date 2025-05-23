// memory_manager.rs - Enhanced with better use of all fields

use std::time::{Duration, Instant};
use parking_lot::Mutex;
use log::{info, debug, warn};
use sys_info;

// Constants for memory management
const HIGH_MEMORY_THRESHOLD: f64 = 80.0;    // Percentage of memory usage that triggers aggressive cleanup
const NORMAL_MEMORY_THRESHOLD: f64 = 70.0;  // Percentage that triggers standard cleanup
const LOW_MEMORY_THRESHOLD: f64 = 30.0;     // Percentage below which we can increase batch sizes
const MIN_CLEANUP_INTERVAL: Duration = Duration::from_secs(5); // Minimum time between forced cleanups
const MEMORY_DEGRADATION_THRESHOLD: f64 = 0.30; // 30% degradation from initial memory triggers warning

/// Handles memory monitoring and adaptive processing parameters
pub struct MemoryManager {
    last_cleanup: Mutex<Instant>,
    initial_memory: u64,
    min_batch_size: usize,
    max_batch_size: usize,
    current_batch_size: Mutex<usize>,
    last_memory_check: Mutex<Instant>,
    normal_operating_range: (f64, f64), // (low_bound, high_bound) of normal operating range
}

impl MemoryManager {
    /// Create a new MemoryManager with initial parameters
    pub fn new(min_batch_size: usize, max_batch_size: usize) -> Self {
        let initial_memory = Self::get_available_memory().unwrap_or(0);
        let initial_batch_size = (min_batch_size + max_batch_size) / 2;
        
        // Define normal operating range as a buffer around NORMAL_MEMORY_THRESHOLD
        let normal_operating_range = (NORMAL_MEMORY_THRESHOLD - 5.0, NORMAL_MEMORY_THRESHOLD + 5.0);
        
        info!("Initializing MemoryManager with initial available memory: {} bytes", initial_memory);
        info!("Normal operating memory range: {:.1}% - {:.1}%", 
              normal_operating_range.0, normal_operating_range.1);
        
        Self {
            last_cleanup: Mutex::new(Instant::now()),
            initial_memory,
            min_batch_size,
            max_batch_size,
            current_batch_size: Mutex::new(initial_batch_size),
            last_memory_check: Mutex::new(Instant::now()),
            normal_operating_range,
        }
    }
    
    /// Get current available system memory
    pub fn get_available_memory() -> Option<u64> {
        sys_info::mem_info().ok().map(|info| info.avail)
    }
    
    /// Get total system memory
    pub fn get_total_memory() -> Option<u64> {
        sys_info::mem_info().ok().map(|info| info.total)
    }
    
    /// Calculate memory usage percentage
    pub fn get_memory_usage_percent() -> Option<f64> {
        if let Ok(mem_info) = sys_info::mem_info() {
            let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
            Some(usage_percent)
        } else {
            None
        }
    }
    
    /// Check if memory has degraded significantly from initial state
    pub fn check_memory_degradation(&self) -> Option<f64> {
        if self.initial_memory == 0 {
            return None;
        }
        
        if let Some(current_memory) = Self::get_available_memory() {
            // Only check every 30 seconds to avoid excessive logging
            let mut last_check = self.last_memory_check.lock();
            if last_check.elapsed() < Duration::from_secs(30) {
                return None;
            }
            *last_check = Instant::now();
            
            // Calculate degradation percentage
            let initial_gb = self.initial_memory as f64 / (1024.0 * 1024.0 * 1024.0);
            let current_gb = current_memory as f64 / (1024.0 * 1024.0 * 1024.0);
            let degradation = 1.0 - (current_memory as f64 / self.initial_memory as f64);
            
            if degradation > MEMORY_DEGRADATION_THRESHOLD {
                warn!("Memory has degraded by {:.1}% from initial state ({:.2} GB → {:.2} GB)", 
                     degradation * 100.0, initial_gb, current_gb);
                return Some(degradation);
            }
        }
        
        None
    }
    
    /// Check if memory usage exceeds any threshold and take appropriate action
    pub fn monitor_memory_thresholds(&self) -> MemoryState {
        if let Some(usage_percent) = Self::get_memory_usage_percent() {
            if usage_percent > HIGH_MEMORY_THRESHOLD {
                debug!("HIGH memory usage detected: {:.1}%", usage_percent);
                return MemoryState::High;
            } else if usage_percent > self.normal_operating_range.1 {
                debug!("ABOVE NORMAL memory usage: {:.1}%", usage_percent);
                return MemoryState::AboveNormal;
            } else if usage_percent < LOW_MEMORY_THRESHOLD {
                debug!("LOW memory usage detected: {:.1}%", usage_percent);
                return MemoryState::Low;
            } else if usage_percent < self.normal_operating_range.0 {
                debug!("BELOW NORMAL memory usage: {:.1}%", usage_percent);
                return MemoryState::BelowNormal;
            } else {
                // Within normal operating range
                return MemoryState::Normal;
            }
        }
        
        MemoryState::Unknown
    }
    
    /// Check if memory usage exceeds the high threshold
    pub fn should_force_cleanup(&self) -> bool {
        // Only force cleanup if enough time has passed since the last cleanup
        let mut last_cleanup = self.last_cleanup.lock();
        if last_cleanup.elapsed() < MIN_CLEANUP_INTERVAL {
            return false;
        }
        
        if let Some(usage_percent) = Self::get_memory_usage_percent() {
            if usage_percent > HIGH_MEMORY_THRESHOLD {
                *last_cleanup = Instant::now();
                debug!("High memory usage detected: {:.1}%, forcing cleanup", usage_percent);
                return true;
            } else if usage_percent > self.normal_operating_range.1 {
                // Check if memory has degraded over time
                if let Some(degradation) = self.check_memory_degradation() {
                    if degradation > MEMORY_DEGRADATION_THRESHOLD {
                        *last_cleanup = Instant::now();
                        debug!("Memory degradation detected ({:.1}%), forcing cleanup", degradation * 100.0);
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Perform memory cleanup actions
    pub async fn force_cleanup(&self) -> MemoryState {
        debug!("Performing forced memory cleanup");
        
        // First check current state to decide cleanup intensity
        let current_state = self.monitor_memory_thresholds();
        
        // Force drop of a large vector to encourage garbage collection
        match current_state {
            MemoryState::High => {
                // Aggressive cleanup for high memory state
                debug!("Performing aggressive cleanup for HIGH memory state");
                drop(Vec::<u8>::with_capacity(1024 * 1024 * 50)); // 50MB
                tokio::time::sleep(Duration::from_millis(200)).await;
            },
            MemoryState::AboveNormal => {
                // Medium cleanup for above normal state
                debug!("Performing medium cleanup for ABOVE NORMAL memory state");
                drop(Vec::<u8>::with_capacity(1024 * 1024 * 20)); // 20MB
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
            _ => {
                // Light cleanup for other states
                debug!("Performing light cleanup");
                drop(Vec::<u8>::with_capacity(1024 * 1024 * 10)); // 10MB
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
        
        // Update last cleanup time
        *self.last_cleanup.lock() = Instant::now();
        
        // Return the new state after cleanup
        self.monitor_memory_thresholds()
    }
    
    /// Get the current recommended batch size based on memory conditions
    pub fn get_batch_size(&self) -> usize {
        *self.current_batch_size.lock()
    }
    
    /// Adjust batch size based on current memory conditions
    pub fn adjust_batch_size(&self) -> bool {
        let memory_state = self.monitor_memory_thresholds();
        let mut batch_size = self.current_batch_size.lock();
        let old_batch_size = *batch_size;
        
        match memory_state {
            MemoryState::High => {
                // Drastically reduce batch size when memory usage is high
                *batch_size = (*batch_size / 4).max(self.min_batch_size);
                debug!("HIGH memory usage, drastically reduced batch size: {} → {}", 
                      old_batch_size, *batch_size);
            },
            MemoryState::AboveNormal => {
                // Moderately reduce batch size 
                *batch_size = (*batch_size / 2).max(self.min_batch_size);
                debug!("ABOVE NORMAL memory usage, reduced batch size: {} → {}", 
                      old_batch_size, *batch_size);
            },
            MemoryState::Low => {
                // Significantly increase batch size when memory usage is low
                *batch_size = (*batch_size * 2).min(self.max_batch_size);
                debug!("LOW memory usage, increased batch size: {} → {}", 
                      old_batch_size, *batch_size);
            },
            MemoryState::BelowNormal => {
                // Moderately increase batch size
                *batch_size = (*batch_size * 3 / 2).min(self.max_batch_size);
                debug!("BELOW NORMAL memory usage, moderately increased batch size: {} → {}", 
                      old_batch_size, *batch_size);
            },
            MemoryState::Normal => {
                // Keep batch size stable in normal range, but check for memory degradation
                if let Some(degradation) = self.check_memory_degradation() {
                    if degradation > MEMORY_DEGRADATION_THRESHOLD {
                        // If memory has degraded over time, slightly reduce batch size
                        *batch_size = (*batch_size * 4 / 5).max(self.min_batch_size);
                        debug!("NORMAL usage but detected memory degradation, slightly reducing batch size: {} → {}", 
                              old_batch_size, *batch_size);
                    }
                }
            },
            MemoryState::Unknown => {
                // Cannot determine memory state, keep current batch size
                debug!("Unable to determine memory state, keeping current batch size");
            }
        }
        
        // Return whether batch size was changed
        old_batch_size != *batch_size
    }
    
    /// Calculate the optimal chunk size for memory-mapped file handling
    /// based on available memory and initial memory
    pub fn calculate_optimal_chunk_size(&self) -> usize {
        if let Some(avail_mem) = Self::get_available_memory() {
            // Compare with initial memory to adjust behavior
            let memory_ratio = if self.initial_memory > 0 {
                avail_mem as f64 / self.initial_memory as f64
            } else {
                1.0
            };
            
            // Adjust percentage based on available vs initial memory
            let percent = if memory_ratio > 0.8 {
                // If memory is still close to initial state, use more
                0.05 // 5% of available memory
            } else if memory_ratio > 0.5 {
                // If moderate degradation, use less
                0.03 // 3% of available memory
            } else {
                // If significant degradation, be very conservative
                0.01 // 1% of available memory
            };
            
            // Use a percentage of available memory for a chunk
            let chunk_size = (avail_mem as f64 * percent) as usize;
            
            // Ensure the chunk size is reasonable
            let min_size = 1 * 1024 * 1024; // 1MB minimum
            let max_size = 8 * 1024 * 1024; // 8MB maximum
            
            debug!("Calculated chunk size: {}MB (memory ratio: {:.2}, using {:.1}% of available memory)",
                  chunk_size / (1024 * 1024), memory_ratio, percent * 100.0);
                  
            chunk_size.clamp(min_size, max_size)
        } else {
            // Default to 4MB if memory info is unavailable
            debug!("Unable to get memory info, using default 4MB chunk size");
            4 * 1024 * 1024
        }
    }
    
    /// Calculate optimal memory settings for the FastStringBuilder
    pub fn calculate_builder_memory_limit(&self) -> usize {
        if let Some(avail_mem) = Self::get_available_memory() {
            // Compare with initial memory
            let memory_ratio = if self.initial_memory > 0 {
                avail_mem as f64 / self.initial_memory as f64
            } else {
                1.0
            };
            
            // Adjust based on available vs initial memory
            let percent = if memory_ratio > 0.8 {
                0.3 // 30% if memory is plentiful
            } else if memory_ratio > 0.5 {
                0.2 // 20% if moderate memory available
            } else {
                0.1 // 10% if memory is constrained
            };
            
            // Calculate memory limit in MB
            ((avail_mem as f64 * percent) / (1024.0 * 1024.0)) as usize
        } else {
            // Default to 64MB if memory info is unavailable
            64
        }
    }
    
    /// Check if system is in a memory-constrained state
    pub fn is_memory_constrained(&self) -> bool {
        if let Some(usage_percent) = Self::get_memory_usage_percent() {
            usage_percent > NORMAL_MEMORY_THRESHOLD
        } else {
            false // Assume not constrained if we can't get info
        }
    }
    
    /// Log current memory status
    pub fn log_memory_status(&self) {
        if let Ok(mem_info) = sys_info::mem_info() {
            let total_mb = mem_info.total / 1024;
            let avail_mb = mem_info.avail / 1024;
            let used_mb = (mem_info.total - mem_info.avail) / 1024;
            let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
            
            // Compare with initial state
            let initial_mb = self.initial_memory / 1024;
            let initial_percent = 
                if self.initial_memory > 0 {
                    (avail_mb as f64 / initial_mb as f64) * 100.0
                } else { 
                    0.0 
                };
            
            info!("Memory status: {:.1}% used ({} MB used / {} MB total, {} MB available)", 
                 usage_percent, used_mb, total_mb, avail_mb);
                 
            if self.initial_memory > 0 {
                info!("Memory compared to initial state: {:.1}% remaining ({} MB initial → {} MB current)",
                     initial_percent, initial_mb, avail_mb);
            }
            
            // Report current state
            info!("Memory state: {}", self.monitor_memory_thresholds());
        } else {
            warn!("Unable to get memory information");
        }
    }
}

// Add an enum to represent memory states for clearer state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryState {
    Low,           // Below LOW_MEMORY_THRESHOLD
    BelowNormal,   // Between LOW_MEMORY_THRESHOLD and lower bound of normal range
    Normal,        // Within normal operating range
    AboveNormal,   // Between upper bound of normal range and HIGH_MEMORY_THRESHOLD
    High,          // Above HIGH_MEMORY_THRESHOLD
    Unknown,       // Cannot determine state
}

impl std::fmt::Display for MemoryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryState::Low => write!(f, "LOW"),
            MemoryState::BelowNormal => write!(f, "BELOW NORMAL"),
            MemoryState::Normal => write!(f, "NORMAL"),
            MemoryState::AboveNormal => write!(f, "ABOVE NORMAL"),
            MemoryState::High => write!(f, "HIGH"),
            MemoryState::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// Implementation of Drop to ensure cleanup on destruction
impl Drop for MemoryManager {
    fn drop(&mut self) {
        debug!("MemoryManager being dropped, forcing final cleanup");
        drop(Vec::<u8>::with_capacity(1024 * 1024 * 10)); // 10MB
    }
}