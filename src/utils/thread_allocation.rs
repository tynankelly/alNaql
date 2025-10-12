// This module provides smart thread allocation for comparison jobs
// The defaults are tuned based on performance testing of n-gram matching workloads

use sys_info;
use log::{info, debug, warn};
use crate::config::execution::ExecutionConfig;

/// System state information needed for thread allocation decisions
#[derive(Debug, Clone)]
pub struct SystemState {
    pub cpu_cores: usize,
    pub available_memory_mb: usize,
    pub memory_usage_percent: f64,
    pub total_memory_mb: usize,
}

impl SystemState {
    /// Gather current system state
    pub fn assess() -> Result<Self, String> {
        let cpu_cores = num_cpus::get();
        
        let mem_info = sys_info::mem_info()
            .map_err(|e| format!("Failed to get memory info: {}", e))?;
        
        let total_memory_mb = (mem_info.total / 1024) as usize;
        let available_memory_mb = (mem_info.avail / 1024) as usize;
        let memory_usage_percent = ((mem_info.total - mem_info.avail) as f64 
                                   / mem_info.total as f64) * 100.0;
        
        Ok(SystemState {
            cpu_cores,
            available_memory_mb,
            memory_usage_percent,
            total_memory_mb,
        })
    }
    
    /// Check if system is under memory pressure
    pub fn has_memory_pressure(&self) -> bool {
        self.memory_usage_percent > 85.0 || self.available_memory_mb < 2048
    }
}

/// Resource allocation for a comparison job
/// Returned by ComparisonCoordinator to specify allocated resources
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub memory_mb: usize,
    pub thread_budget: usize,
    pub optimal_threads: usize,
    pub min_threads: usize,
    pub max_threads: usize,
}

impl ResourceAllocation {
    /// Create a resource allocation from execution config for sequential mode
    pub fn sequential(execution_config: &ExecutionConfig, total_memory_mb: usize, total_threads: usize) -> Self {
        ResourceAllocation {
            memory_mb: total_memory_mb,
            thread_budget: execution_config.optimal_threads_per_comparison.min(total_threads),
            optimal_threads: execution_config.optimal_threads_per_comparison,
            min_threads: execution_config.min_threads_per_comparison,
            max_threads: execution_config.max_threads_per_comparison.min(total_threads),
        }
    }
    
    /// Create a resource allocation for parallel mode
    pub fn parallel(
        execution_config: &ExecutionConfig, 
        memory_per_job: usize,
        threads_per_job: usize
    ) -> Self {
        ResourceAllocation {
            memory_mb: memory_per_job,
            thread_budget: threads_per_job,
            optimal_threads: execution_config.optimal_threads_per_comparison,
            min_threads: execution_config.min_threads_per_comparison,
            max_threads: execution_config.max_threads_per_comparison,
        }
    }
}

/// Thread allocation configuration
/// These values are tuned based on performance testing
#[derive(Debug, Clone)]
pub struct ThreadAllocationConfig {
    /// Optimal thread count for normal operations (your tuned value)
    pub optimal_threads: usize,
    
    /// Minimum threads to maintain parallelism
    pub min_threads: usize,
    
    /// Maximum threads (even on large machines)
    pub max_threads: usize,
    
    /// Memory pressure threshold (percent)
    pub memory_pressure_threshold: f64,
    
    /// Reduction factor when under memory pressure
    pub memory_pressure_reduction: f64,
}

impl ThreadAllocationConfig {
    /// Create thread allocation config from execution config
    /// This is the primary constructor for normal operation
    pub fn from_config(execution_config: &ExecutionConfig) -> Self {
        ThreadAllocationConfig {
            optimal_threads: execution_config.optimal_threads_per_comparison,
            min_threads: execution_config.min_threads_per_comparison,
            max_threads: execution_config.max_threads_per_comparison,
            memory_pressure_threshold: execution_config.memory_pressure_threshold,
            memory_pressure_reduction: execution_config.memory_pressure_reduction_factor,
        }
    }
    
    /// Create thread allocation config with a specific thread budget
    /// Used when running parallel comparisons where resources are divided
    pub fn from_config_with_budget(execution_config: &ExecutionConfig, thread_budget: usize) -> Self {
        ThreadAllocationConfig {
            // Use budget as optimal, but still respect configured min/max
            optimal_threads: thread_budget
                .max(execution_config.min_threads_per_comparison)
                .min(execution_config.max_threads_per_comparison),
            min_threads: execution_config.min_threads_per_comparison,
            max_threads: execution_config.max_threads_per_comparison
                .min(thread_budget + 2), // Allow slight flex above budget if configured max is higher
            memory_pressure_threshold: execution_config.memory_pressure_threshold,
            memory_pressure_reduction: execution_config.memory_pressure_reduction_factor,
        }
    }
    /// Get a string describing the source of this configuration
    pub fn source_description(&self) -> String {
        // Check if values match the deprecated defaults
        if self.optimal_threads == 10 && 
           self.min_threads == 2 && 
           self.max_threads == 12 &&
           self.memory_pressure_threshold == 85.0 &&
           self.memory_pressure_reduction == 0.5 {
            "default (hardcoded)".to_string()
        } else {
            "config file".to_string()
        }
    }
}

/// Calculate optimal thread count for a comparison job
pub fn calculate_thread_count(
    system_state: &SystemState,
    config: &ThreadAllocationConfig,
) -> usize {
    // Start with your tuned optimal value
    let mut threads = config.optimal_threads;
    
    // Log what we're working with - ENHANCED LOGGING
    debug!("Thread calculation starting with:");
    debug!("  Optimal threads: {} (from config)", config.optimal_threads);
    debug!("  Min threads: {}", config.min_threads);
    debug!("  Max threads: {}", config.max_threads);
    debug!("  System cores: {}", system_state.cpu_cores);
    debug!("  Available memory: {} MB", system_state.available_memory_mb);
    debug!("  Memory usage: {:.1}%", system_state.memory_usage_percent);
    
    // Check for extreme memory pressure
    if system_state.has_memory_pressure() {
        warn!("Memory pressure detected: {}% used, only {}MB available",
              system_state.memory_usage_percent,
              system_state.available_memory_mb);
        
        let reduced = (threads as f64 * config.memory_pressure_reduction) as usize;
        info!("Reduced threads from {} to {} due to memory pressure (factor: {})", 
              threads, reduced, config.memory_pressure_reduction);
        threads = reduced;
    }
    
    // Clamp to configured bounds
    let before_clamp = threads;
    threads = threads.clamp(config.min_threads, config.max_threads);
    if threads != before_clamp {
        debug!("Clamped threads from {} to {} (bounds: {}-{})",
              before_clamp, threads, config.min_threads, config.max_threads);
    }
    
    // Final sanity check: never exceed actual CPU cores
    if threads > system_state.cpu_cores {
        debug!("Capping threads from {} to {} (CPU core limit)", 
              threads, system_state.cpu_cores);
        threads = system_state.cpu_cores;
    }
    
    info!("Thread allocation decision: {} threads (optimal: {}, cores: {})",
          threads, config.optimal_threads, system_state.cpu_cores);
    
    threads
}

/// Calculate thread count using resource allocation
/// This is the new preferred method for resource-aware thread calculation
pub fn calculate_thread_count_with_allocation(
    system_state: &SystemState,
    allocation: &ResourceAllocation,
    execution_config: &ExecutionConfig,
) -> usize {
    // Create config from allocation
    let thread_config = ThreadAllocationConfig::from_config_with_budget(
        execution_config,
        allocation.thread_budget
    );
    
    // Use existing calculation logic
    calculate_thread_count(system_state, &thread_config)
}

/// Get thread count with overrides, using new allocation-based approach
pub fn get_thread_count_with_allocation(
    system_state: &SystemState,
    allocation: &ResourceAllocation,
    execution_config: &ExecutionConfig,
) -> usize {
    // Check for environment variable override (useful for testing)
    if let Ok(override_threads) = std::env::var("ALNAQL_THREADS") {
        if let Ok(threads) = override_threads.parse::<usize>() {
            info!("Using thread count override from ALNAQL_THREADS: {}", threads);
            info!("  (Overriding allocated budget of {} threads)", allocation.thread_budget);
            return threads.clamp(1, system_state.cpu_cores);
        }
    }
    
    // Normal calculation with allocation
    calculate_thread_count_with_allocation(system_state, allocation, execution_config)
}