// config/execution.rs - Execution strategy and parallelization configuration

use serde::{Serialize, Deserialize};
use log::warn;
use crate::error::{Error, Result};
use crate::config::loader::Configurable;

/// Configuration for execution strategy and parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    // ========== Basic Processing Mode ==========
    /// Overall processing mode ("sequential" or "parallel")
    pub processing_mode: String,

    pub max_concurrent_files: usize,
        
    // ========== Storage Access ==========
    /// Strategy for accessing storage ("shared", "exclusive", "pooled")
    pub storage_access_strategy: String,
    
    // ========== Dense Region Processing ==========
    /// Enable parallel processing for dense region finding
    pub parallel_dense_regions: bool,
    
    /// Minimum number of matches to trigger parallel dense region processing
    pub parallel_dense_threshold: usize,
    
    /// Minimum chunk size for dense region processing
    pub min_dense_chunk_size: usize,
    
    /// Maximum chunk size for dense region processing
    pub max_dense_chunk_size: usize,

    /// Number of cluster chunks to process in parallel
    pub parallel_clustering_chunks: Option<usize>,

    /// Size of chunks during grid streaming process of clusters
    pub grid_cluster_chunks: usize,
    
    //===========Validation Parallel Processing===
    /// Batch size for parallel validation processing
    /// Higher = more memory, better throughput
    pub validation_batch_size: usize,

    // ========== Streaming Final Merge ==========
    /// Size of chunks for streaming final merge when dataset is too large
    pub final_merge_chunk_size: usize,
    
    /// Window size for finding chunk boundaries (as percentage of chunk size)
    /// E.g., 1.0 = 1% window on each side of target boundary
    pub boundary_window_percent: f64,
    
    /// Safety multiplier for merge gap ratio when finding boundaries
    /// E.g., 3.0 = require 3x the merge_gap_ratio to consider it safe
    pub boundary_safety_multiplier: f64,

    // ========== Parallel Comparison Processing ==========
    /// Maximum number of comparisons to run in parallel (1 = sequential)
    pub max_parallel_comparisons: usize,

    /// Minimum memory per comparison (MB) when running in parallel
    pub min_memory_per_comparison: usize,

    /// Minimum threads per comparison when running in parallel  
    pub min_threads_per_comparison: usize,

    // Number of optimal threads per comparison between two texts
    pub optimal_threads_per_comparison: usize,

    // Maxmimum number of threads per comparison
    pub max_threads_per_comparison: usize,

    // Memory threshold that triggers reduction in number of threads
    pub memory_pressure_threshold: f64,

    // Factor by which number of threads is reduced under memory pressure
    pub memory_pressure_reduction_factor: f64,

    /// Enable database handle pooling for parallel comparisons
    pub enable_db_pooling: bool,

    /// Strategy for parallel execution ("unrestricted", "smart", "pooled")
    pub parallel_strategy: String,

    
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            // Basic mode
            processing_mode: "sequential".to_string(),
            max_concurrent_files: 4,            
            
            // Storage access
            storage_access_strategy: "shared".to_string(),
            
            // Dense region processing
            parallel_dense_regions: true,
            parallel_dense_threshold: 10_000,
            min_dense_chunk_size: 5_000,
            max_dense_chunk_size: 50_000,
            parallel_clustering_chunks: Some(0),
            grid_cluster_chunks: 100_000,

            // Validation parallel processing
            validation_batch_size: 5_000,

            // Streaming final merge defaults
            final_merge_chunk_size: 1_000_000,
            boundary_window_percent: 1.0,
            boundary_safety_multiplier: 3.0,

            // Parallel comparison defaults
            max_parallel_comparisons: 1,  // Sequential by default
            min_memory_per_comparison: 4096,  // 4GB minimum
            min_threads_per_comparison: 8,
            optimal_threads_per_comparison: 10,
            max_threads_per_comparison: 12,
            memory_pressure_threshold: 85.0,
            memory_pressure_reduction_factor: 0.5, 
            enable_db_pooling: false,
            parallel_strategy: "smart".to_string(),
        }
    }
}

impl Configurable for ExecutionConfig {
    fn section_name() -> &'static str {
        "execution"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            // Ngram generation processing mode
            "processing_mode" => {
                let mode = value.trim_matches('"').to_lowercase();
                match mode.as_str() {
                    "sequential" | "parallel" => {
                        self.processing_mode = mode;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid processing_mode (must be 'sequential' or 'parallel'): {}", value)
                    )),
                }
            },
            "max_concurrent_files" => {
                let num = crate::config::loader::parse::usize(value)?;
                if num == 0 {
                    return Err(Error::Config(format!("Invalid max_concurrent_files (must be > 0): {}", value)));
                }
                self.max_concurrent_files = num;
                Ok(true)
            },
            
            // Storage access
            "storage_access_strategy" => {
                let strategy = value.trim_matches('"').to_lowercase();
                match strategy.as_str() {
                    "shared" | "exclusive" | "pooled" => {
                        self.storage_access_strategy = strategy;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid storage_access_strategy (must be 'shared', 'exclusive', or 'pooled'): {}", value)
                    )),
                }
            },
            
            // Dense region processing
            "parallel_dense_regions" => {
                self.parallel_dense_regions = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "parallel_dense_threshold" => {
                let threshold = crate::config::loader::parse::usize(value)?;
                if threshold == 0 {
                    return Err(Error::Config(
                        format!("Invalid parallel_dense_threshold (must be > 0): {}", value)
                    ));
                }
                self.parallel_dense_threshold = threshold;
                Ok(true)
            },
            "min_dense_chunk_size" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        format!("Invalid min_dense_chunk_size (must be > 0): {}", value)
                    ));
                }
                self.min_dense_chunk_size = size;
                Ok(true)
            },
            "max_dense_chunk_size" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        format!("Invalid max_dense_chunk_size (must be > 0): {}", value)
                    ));
                }
                self.max_dense_chunk_size = size;
                Ok(true)
            },

            "parallel_clustering_chunks" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        format!("Invalid parallel_clustering_chunks (must be > 0): {}", value)
                    ));
                }
                self.parallel_clustering_chunks = Some(size);
                Ok(true)
            },

            "grid_cluster_chunks" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        format!("Invalid grid_cluster_chunks (must be > 0): {}", value)
                    ));
                }
                self.grid_cluster_chunks = size;
                Ok(true)
            },
            // Validataion parallel processing
            "validation_batch_size" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        "validation_batch_size must be greater than 0".to_string()
                    ));
                }
                self.validation_batch_size = size;
                Ok(true)
            },
            // Streaming final merge settings
            "final_merge_chunk_size" => {
                let size = crate::config::loader::parse::usize(value)?;
                if size == 0 {
                    return Err(Error::Config(
                        format!("Invalid final_merge_chunk_size (must be > 0): {}", value)
                    ));
                }
                self.final_merge_chunk_size = size;
                Ok(true)
            },
            "boundary_window_percent" => {
                let percent = crate::config::loader::parse::f64(value)?;
                if percent <= 0.0 || percent > 10.0 {
                    return Err(Error::Config(
                        format!("Invalid boundary_window_percent (must be 0.0 < x <= 10.0): {}", value)
                    ));
                }
                self.boundary_window_percent = percent;
                Ok(true)
            },
            "boundary_safety_multiplier" => {
                let multiplier = crate::config::loader::parse::f64(value)?;
                if multiplier <= 0.0 || multiplier > 10.0 {
                    return Err(Error::Config(
                        format!("Invalid boundary_safety_multiplier (must be 0.0 < x <= 10.0): {}", value)
                    ));
                }
                self.boundary_safety_multiplier = multiplier;
                Ok(true)
            },
            // Parallel comparison settings
            "max_parallel_comparisons" => {
                // Check if it's "auto"
                let trimmed = value.trim_matches('"').to_lowercase();
                if trimmed == "auto" {
                    self.max_parallel_comparisons = 0;  // 0 means auto-calculate
                } else {
                    let num = crate::config::loader::parse::usize(value)?;
                    // 0 is now reserved for "auto", so actual values must be > 0
                    if num == 0 {
                        return Err(Error::Config(
                            format!("Invalid max_parallel_comparisons (must be > 0 or 'auto'): {}", value)
                        ));
                    }
                    self.max_parallel_comparisons = num;
                }
                Ok(true)
            },
            "min_memory_per_comparison" => {
                let mb = crate::config::loader::parse::usize(value)?;
                if mb < 1024 {  // Minimum 1GB
                    return Err(Error::Config(
                        format!("Invalid min_memory_per_comparison (must be >= 1024 MB): {}", value)
                    ));
                }
                self.min_memory_per_comparison = mb;
                Ok(true)
            },
            "min_threads_per_comparison" => {
                let threads = crate::config::loader::parse::usize(value)?;
                if threads == 0 {
                    return Err(Error::Config(
                        format!("Invalid min_threads_per_comparison (must be > 0): {}", value)
                    ));
                }
                self.min_threads_per_comparison = threads;
                Ok(true)
            },
            "optimal_threads_per_comparison" => {
                let threads = crate::config::loader::parse::usize(value)?;
                if threads == 0 {
                    return Err(Error::Config(
                        format!("Invalid optimal_threads_per_comparison (must be > 0): {}", value)
                    ));
                }
                self.optimal_threads_per_comparison = threads;
                Ok(true)
            },
            "max_threads_per_comparison" => {
                let threads = crate::config::loader::parse::usize(value)?;
                if threads == 0 {
                    return Err(Error::Config(
                        format!("Invalid max_threads_per_comparison (must be > 0): {}", value)
                    ));
                }
                self.max_threads_per_comparison = threads;
                Ok(true)
            },
            "memory_pressure_threshold" => {
                let threshold = crate::config::loader::parse::f64(value)?;
                if threshold <= 50.0 || threshold >= 95.0 {
                    return Err(Error::Config(
                        format!("Invalid memory_pressure_threshold (must be 50.0 < x < 95.0): {}", value)
                    ));
                }
                self.memory_pressure_threshold = threshold;
                Ok(true)
            },
            "memory_pressure_reduction_factor" => {
                let factor = crate::config::loader::parse::f64(value)?;
                if factor <= 0.1 || factor >= 0.9 {
                    return Err(Error::Config(
                        format!("Invalid memory_pressure_reduction_factor (must be 0.1 < x < 0.9): {}", value)
                    ));
                }
                self.memory_pressure_reduction_factor = factor;
                Ok(true)
            },
            "enable_db_pooling" => {
                self.enable_db_pooling = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "parallel_strategy" => {
                let strategy = value.trim_matches('"').to_lowercase();
                match strategy.as_str() {
                    "unrestricted" | "smart" | "pooled" => {
                        self.parallel_strategy = strategy;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid parallel_strategy (must be 'unrestricted', 'smart', or 'pooled'): {}", value)
                    )),
                }
            },
            
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Validate processing mode
        match self.processing_mode.as_str() {
            "sequential" | "parallel" => {},
            _ => return Err(Error::Config(
                format!("Invalid processing_mode: {}", self.processing_mode)
            )),
        }
                    
        // Validate storage access strategy
        match self.storage_access_strategy.as_str() {
            "shared" | "exclusive" | "pooled" => {},
            _ => return Err(Error::Config(
                format!("Invalid storage_access_strategy: {}", self.storage_access_strategy)
            )),
        }
                
        // Validate dense region settings
        if self.parallel_dense_threshold == 0 {
            return Err(Error::Config(
                "parallel_dense_threshold must be greater than 0".to_string()
            ));
        }
        
        if self.min_dense_chunk_size == 0 {
            return Err(Error::Config(
                "min_dense_chunk_size must be greater than 0".to_string()
            ));
        }
        
        if self.max_dense_chunk_size == 0 {
            return Err(Error::Config(
                "max_dense_chunk_size must be greater than 0".to_string()
            ));
        }
        
        if self.min_dense_chunk_size > self.max_dense_chunk_size {
            return Err(Error::Config(
                format!("min_dense_chunk_size ({}) cannot be larger than max_dense_chunk_size ({})",
                    self.min_dense_chunk_size, self.max_dense_chunk_size)
            ));
        }
        // Validate batch size
        if self.validation_batch_size < 100 || self.validation_batch_size > 100_000 {
            return Err(Error::Config(
                format!("validation_batch_size must be between 100 and 100,000, got {}", 
                    self.validation_batch_size)
            ));
        }
        // Validate streaming final merge settings
        if self.final_merge_chunk_size == 0 {
            return Err(Error::Config(
                "final_merge_chunk_size must be greater than 0".to_string()
            ));
        }
        
        if self.boundary_window_percent <= 0.0 || self.boundary_window_percent > 10.0 {
            return Err(Error::Config(
                format!("boundary_window_percent must be between 0.0 and 10.0, got {}", 
                    self.boundary_window_percent)
            ));
        }
        
        if self.boundary_safety_multiplier <= 0.0 || self.boundary_safety_multiplier > 10.0 {
            return Err(Error::Config(
                format!("boundary_safety_multiplier must be between 0.0 and 10.0, got {}", 
                    self.boundary_safety_multiplier)
            ));
        }

        // Validate parallel comparison settings
        if self.min_memory_per_comparison < 1024 {
            return Err(Error::Config(
                format!("min_memory_per_comparison must be at least 1024 MB (1GB), got {}", 
                    self.min_memory_per_comparison)
            ));
        }

        if self.min_threads_per_comparison == 0 {
            return Err(Error::Config(
                "min_threads_per_comparison must be greater than 0".to_string()
            ));
        }

        // Validate thread allocation hierarchy
        if self.min_threads_per_comparison > self.optimal_threads_per_comparison {
            return Err(Error::Config(
                format!("min_threads_per_comparison ({}) must be <= optimal_threads_per_comparison ({})",
                    self.min_threads_per_comparison, self.optimal_threads_per_comparison)
            ));
        }

        if self.optimal_threads_per_comparison > self.max_threads_per_comparison {
            return Err(Error::Config(
                format!("optimal_threads_per_comparison ({}) must be <= max_threads_per_comparison ({})",
                    self.optimal_threads_per_comparison, self.max_threads_per_comparison)
            ));
        }

        // Validate memory pressure settings
        if self.memory_pressure_threshold <= 50.0 || self.memory_pressure_threshold >= 95.0 {
            return Err(Error::Config(
                format!("memory_pressure_threshold must be between 50.0 and 95.0, got {}",
                    self.memory_pressure_threshold)
            ));
        }

        if self.memory_pressure_reduction_factor <= 0.1 || self.memory_pressure_reduction_factor >= 0.9 {
            return Err(Error::Config(
                format!("memory_pressure_reduction_factor must be between 0.1 and 0.9, got {}",
                    self.memory_pressure_reduction_factor)
            ));
        }

        // Validate parallel strategy
        match self.parallel_strategy.as_str() {
            "unrestricted" | "smart" | "pooled" => {},
            _ => return Err(Error::Config(
                format!("Invalid parallel_strategy: {}", self.parallel_strategy)
            )),
        }
        Ok(())
    }
}

impl ExecutionConfig {
    /// Check if parallel processing is enabled
    pub fn is_parallel(&self) -> bool {
        self.processing_mode == "parallel"
    }
    
    /// Check if sequential processing is enabled
    pub fn is_sequential(&self) -> bool {
        self.processing_mode == "sequential"
    }
        
    /// Check if match count qualifies for parallel dense region processing
    pub fn should_use_parallel_dense(&self, match_count: usize) -> bool {
        self.parallel_dense_regions && match_count >= self.parallel_dense_threshold
    }
    
     /// Check if parallel comparison processing is enabled
    pub fn is_parallel_comparisons(&self) -> bool {
        self.max_parallel_comparisons > 1
    }
    
    pub fn get_resources_per_comparison(&self, total_memory_mb: usize, total_threads: usize) -> (usize, usize) {
        if self.max_parallel_comparisons <= 1 {
            // Sequential mode - use optimal threads (capped by system)
            let threads = self.optimal_threads_per_comparison.min(total_threads);
            (total_memory_mb, threads)
        } else {
            // Parallel mode - divide resources
            let memory_per = (total_memory_mb / self.max_parallel_comparisons)
                .max(self.min_memory_per_comparison);
            
            // Start with optimal, but adjust if total threads are limited
            let divided_threads = total_threads / self.max_parallel_comparisons;
            let threads_per = if divided_threads >= self.optimal_threads_per_comparison {
                // We have enough threads to give each job optimal count
                self.optimal_threads_per_comparison
            } else {
                // Limited threads, divide them but respect minimum
                divided_threads.max(self.min_threads_per_comparison)
            }.min(self.max_threads_per_comparison); // Never exceed max
            
            (memory_per, threads_per)
        }
    }

    /// Check if database pooling is recommended
    pub fn should_enable_pooling(&self) -> bool {
        // Recommend pooling when running multiple comparisons
        self.max_parallel_comparisons > 1 && 
        (self.parallel_strategy == "pooled" || self.enable_db_pooling)
    }
    
    /// Validate pooling configuration
    pub fn validate_pooling(&self) -> Result<()> {
        if self.parallel_strategy == "pooled" && !self.enable_db_pooling {
            warn!("Parallel strategy is 'pooled' but enable_db_pooling is false");
        }
        
        if self.enable_db_pooling && self.max_parallel_comparisons == 1 {
            warn!("Database pooling enabled but max_parallel_comparisons is 1 (no benefit)");
        }
        
        Ok(())
    }
}