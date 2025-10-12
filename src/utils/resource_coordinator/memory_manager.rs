// src/utils/resource_coordinator/memory_manager.rs

use std::sync::Mutex;
use std::time::{Duration, Instant};
use sys_info;
use log::{info, warn, debug};
use serde::{Serialize, Deserialize};

// Import shared state
use super::SHARED_STATE;
use std::sync::atomic::Ordering;


/// Memory thresholds for different states
const HIGH_MEMORY_THRESHOLD: f64 = 85.0;        // Above this is HIGH state
const LOW_MEMORY_THRESHOLD: f64 = 40.0;         // Below this is LOW state
const MEMORY_CHECK_INTERVAL: Duration = Duration::from_secs(60);  // Don't check too frequently
const MEMORY_DEGRADATION_THRESHOLD: f64 = 0.5;  // 50% degradation from initial

/// MemoryManager is responsible for monitoring system memory usage
/// and reporting state via shared atomics

pub struct MemoryManager {
    last_cleanup: Mutex<Instant>,
    initial_memory: u64,
    last_memory_check: Mutex<Instant>,
    normal_operating_range: (f64, f64),
}

impl MemoryManager {
    /// Create a new MemoryManager instance
    pub fn new() -> Self {
        let initial_memory = Self::get_available_memory().unwrap_or(0);
        
        // Set normal operating range based on system memory
        let normal_operating_range = if let Some(total_mem) = Self::get_total_memory() {
            let total_gb = total_mem as f64 / (1024.0 * 1024.0 * 1024.0);
            if total_gb > 64.0 {
                // High memory system: 50-75% usage is normal
                (50.0, 75.0)
            } else if total_gb > 32.0 {
                // Medium memory system: 55-80% usage is normal
                (55.0, 80.0)
            } else {
                // Low memory system: 60-85% usage is normal
                (60.0, 85.0)
            }
        } else {
            // Default range
            (55.0, 80.0)
        };
        
        info!("Initializing MemoryManager with initial available memory: {} MB", 
              initial_memory / (1024 * 1024));
        info!("Normal operating memory range: {:.1}% - {:.1}%", 
              normal_operating_range.0, normal_operating_range.1);
        
        Self {
            last_cleanup: Mutex::new(Instant::now()),
            initial_memory,
            last_memory_check: Mutex::new(Instant::now()),
            normal_operating_range,
        }
    }
    
    /// Get current available system memory in bytes
    pub fn get_available_memory() -> Option<u64> {
        sys_info::mem_info().ok().map(|info| info.avail * 1024)
    }
    
    /// Get total system memory in bytes
    pub fn get_total_memory() -> Option<u64> {
        sys_info::mem_info().ok().map(|info| info.total * 1024)
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
            let mut last_check = self.last_memory_check.lock().unwrap();
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
    
    /// Monitor memory and update shared state atomics
    pub fn monitor_memory_thresholds(&self) -> MemoryState {
        let memory_state = if let Some(usage_percent) = Self::get_memory_usage_percent() {
            if usage_percent > HIGH_MEMORY_THRESHOLD {
                debug!("HIGH memory usage detected: {:.1}%", usage_percent);
                MemoryState::High
            } else if usage_percent > self.normal_operating_range.1 {
                debug!("ABOVE NORMAL memory usage: {:.1}%", usage_percent);
                MemoryState::AboveNormal
            } else if usage_percent >= self.normal_operating_range.0 && usage_percent <= self.normal_operating_range.1 {
                debug!("Normal memory usage: {:.1}%", usage_percent);
                MemoryState::Normal
            } else if usage_percent < LOW_MEMORY_THRESHOLD {
                debug!("LOW memory usage: {:.1}%", usage_percent);
                MemoryState::Low
            } else {
                debug!("BELOW NORMAL memory usage: {:.1}%", usage_percent);
                MemoryState::BelowNormal
            }
        } else {
            warn!("Unable to determine memory usage");
            MemoryState::Unknown
        };
        
        // Update shared state with memory pressure (0-100 scale)
        let pressure = match memory_state {
            MemoryState::Low => 0,
            MemoryState::BelowNormal => 25,
            MemoryState::Normal => 50,
            MemoryState::AboveNormal => 75,
            MemoryState::High => 100,
            MemoryState::Unknown => 50, // Default to normal
        };
        SHARED_STATE.memory_pressure.store(pressure, Ordering::Relaxed);
        
        // Update backpressure flag based on high memory
        SHARED_STATE.backpressure.store(
            matches!(memory_state, MemoryState::High),
            Ordering::Relaxed
        );
        
        memory_state
    }
    
    /// Check memory status and log if needed (monitoring only)
    pub fn check_and_log_if_needed(&self) -> Option<MemoryState> {
        let mut last_cleanup = self.last_cleanup.lock().unwrap();
        
        if last_cleanup.elapsed() < MEMORY_CHECK_INTERVAL {
            return None;
        }
        
        *last_cleanup = Instant::now();
        
        let memory_state = self.monitor_memory_thresholds();
        
        // Check for memory degradation
        if let Some(degradation) = self.check_memory_degradation() {
            warn!("Significant memory degradation detected: {:.1}%", degradation * 100.0);
        }
        
        // Log if memory pressure is high
        if matches!(memory_state, MemoryState::High | MemoryState::AboveNormal) {
            self.log_memory_status();
        }
        
        Some(memory_state)
    }
    
    /// Log current memory status
    pub fn log_memory_status(&self) {
        if let Ok(mem_info) = sys_info::mem_info() {
            let total_mb = mem_info.total / 1024;
            let avail_mb = mem_info.avail / 1024;
            let used_mb = (mem_info.total - mem_info.avail) / 1024;
            let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
            
            info!("Memory Status: {:.1}% used ({} MB / {} MB), {} MB available", 
                 usage_percent, used_mb, total_mb, avail_mb);
            
            // Log comparison to initial memory if significant time has passed
            if self.initial_memory > 0 {
                let initial_mb = self.initial_memory / (1024 * 1024);
                let current_avail_mb = (mem_info.avail * 1024) / (1024 * 1024);
                let percent_of_initial = (current_avail_mb as f64 / initial_mb as f64) * 100.0;
                debug!("Memory compared to initial: {:.1}% remaining ({} MB → {} MB)",
                     percent_of_initial, initial_mb, current_avail_mb);
            }
            
            // Report current state
            info!("Memory state: {}", self.monitor_memory_thresholds());
        } else {
            warn!("Unable to get memory information");
        }
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
        
        // Update last cleanup time with proper error handling
        if let Ok(mut last_cleanup) = self.last_cleanup.lock() {
            *last_cleanup = Instant::now();
        }
        
        // Return the new state after cleanup
        self.monitor_memory_thresholds()
    }
    /// Get current memory pressure as a percentage (0-100)
    pub fn get_memory_pressure(&self) -> usize {
        use crate::utils::resource_coordinator::SHARED_STATE;
        
        SHARED_STATE.memory_pressure.load(Ordering::Relaxed) as usize
    }
        
    /// Force a memory cleanup/collection
    pub fn force_memory_cleanup(&self) -> MemoryState {
        // Use the correct method name for METADATA_CACHE
        use crate::storage::cache::METADATA_CACHE;
        
        // Clear all caches using the correct method
        METADATA_CACHE.clear_all_caches();
        
        // Get current state after cleanup
        self.monitor_memory_thresholds()
    }
}

/// Memory states for clearer state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

impl Drop for MemoryManager {
    fn drop(&mut self) {
        debug!("MemoryManager being dropped");
    }
}
