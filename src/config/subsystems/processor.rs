// src/config/subsystems/processor.rs

use serde::{Serialize, Deserialize};
use log::{info, warn};
use crate::error::{Error, Result};
use crate::config::FromIni;
use super::{
    generator::GeneratorConfig,
    text_processing::TextProcessingConfig,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    // Basic processing settings
    pub batch_size: usize,
    pub processing_mode: String,
    pub max_concurrent_files: usize,
    pub force_sequential_memory_threshold: f64,
    pub target_memory_percent: f64,

    // Subsystem configurations
    pub text_processing: TextProcessingConfig,
    pub generator: GeneratorConfig,

    // Matching settings
    pub similarity_threshold: f64,
    pub max_gap: usize,

    // New parallel processing settings
    pub parallel_ngram_matching: bool,
    pub parallel_batch_size: usize,
    pub parallel_min_file_size: usize,
    pub parallel_thread_count: usize,

    // Pipeline settings
    pub pipeline_processing: bool,
    pub auto_select_strategy: bool,
    pub pipeline_ngram_threshold: usize,
    pub pipeline_memory_threshold: f64,
    pub min_pipeline_batch_size: usize,
    pub max_pipeline_batch_size: usize,
    pub pipeline_queue_capacity: usize,
    pub storage_access_strategy: String,
    pub storage_pool_size: usize,
    // Enable/disable parallel dense region finding
    pub parallel_dense_regions: bool,

    // Minimum number of matches required to use parallel processing
    pub parallel_dense_threshold: usize,

    // Minimum size of each chunk for dense region finding (to avoid overhead of tiny chunks)
    pub min_dense_chunk_size: usize,

    // Maximum size of each chunk for dense region finding (for memory efficiency)
    pub max_dense_chunk_size: usize,
    pub early_isnad_detection: bool,
    pub early_banality_detection: bool,

    // Whether to log discarded matches for review
    pub log_discarded_matches: bool,

    // Path for logging discarded matches
    pub discarded_log_path: String,
    // Banality detection configuration
    pub banality_detection_mode: String,  // "phrase" or "automatic"
    pub banality_auto_proportion: f64,    // Percentage of most common n-grams to consider
    pub banality_auto_threshold: f64,     // Threshold percentage for detection
    pub common_ngrams_file: String,       // Path to common n-grams file

    pub memory_warn_threshold_mb: u64,
    pub memory_critical_threshold_mb: u64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            // Existing defaults
            batch_size: 1000,
            processing_mode: "sequential".to_string(),
            max_concurrent_files: 4,
            force_sequential_memory_threshold: 80.0,
            target_memory_percent: 70.0,
            text_processing: TextProcessingConfig::default(),
            generator: GeneratorConfig::default(),
            similarity_threshold: 0.85,
            max_gap: 10,
            parallel_ngram_matching: false,
            parallel_batch_size: 5000,
            parallel_min_file_size: 1024 * 1024,
            parallel_thread_count: 0,
            pipeline_processing: true,
            auto_select_strategy: true,
            pipeline_ngram_threshold: 50000,
            pipeline_memory_threshold: 0.7,
            min_pipeline_batch_size: 500,
            max_pipeline_batch_size: 2000,
            pipeline_queue_capacity: 16,
            storage_access_strategy: "shared".to_string(),
            storage_pool_size: 4,
            // Default to enabled
            parallel_dense_regions: true,

            // Default threshold of 10,000 matches
            parallel_dense_threshold: 10_000,

            // Default min/max chunk sizes
            min_dense_chunk_size: 5_000,
            max_dense_chunk_size: 50_000,
            early_isnad_detection: true,
            early_banality_detection: true,
            log_discarded_matches: false,
            discarded_log_path: "logs/discarded_matches.log".to_string(),
            banality_detection_mode: "phrase".to_string(),
            banality_auto_proportion: 5.0,
            banality_auto_threshold: 70.0,
            common_ngrams_file: "filters/common_ngrams.txt".to_string(),
            memory_warn_threshold_mb: 1024 * 8, // 8GB warning level
            memory_critical_threshold_mb: 1024 * 12, // 12GB critical level
        }
    }
}

impl FromIni for ProcessorConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        // First try subsystem configs
        match section_name {
            "text_processing" => return self.text_processing.from_ini_section(section_name, key, value),
            "generator" => return self.generator.from_ini_section(section_name, key, value),
            _ => {}
        }

        // Then handle processor-specific settings
        if section_name != "processor" {
            return None;
        }

        match key {
            "batch_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.batch_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid batch_size (must be > 0): {}", value)
                    ))),
                }
            },
            "processing_mode" => {
                let mode = value.trim_matches('"').to_lowercase();
                match mode.as_str() {
                    "sequential" | "parallel" => {
                        self.processing_mode = mode;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid processing_mode (must be 'sequential' or 'parallel'): {}", value)
                    ))),
                }
            },
            "max_concurrent_files" => {
                match value.parse() {
                    Ok(num) if num > 0 => {
                        self.max_concurrent_files = num;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid max_concurrent_files (must be > 0): {}", value)
                    ))),
                }
            },
            "force_sequential_memory_threshold" => {
                match value.parse::<f64>() {
                    Ok(threshold) if (0.0..=100.0).contains(&threshold) => {
                        self.force_sequential_memory_threshold = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid force_sequential_memory_threshold (must be between 0 and 100): {}", value)
                    ))),
                }
            },
            "target_memory_percent" => {
                match value.parse::<f64>() {
                    Ok(percent) if (0.0..=100.0).contains(&percent) => {
                        self.target_memory_percent = percent;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid target_memory_percent (must be between 0 and 100): {}", value)
                    ))),
                }
            },
            "similarity_threshold" => {
                match value.parse::<f64>() {
                    Ok(threshold) if (0.0..=1.0).contains(&threshold) => {
                        self.similarity_threshold = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid similarity_threshold (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            "max_gap" => {
                match value.parse() {
                    Ok(gap) if gap > 0 => {
                        self.max_gap = gap;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid max_gap (must be > 0): {}", value)
                    ))),
                }
            },
            "parallel_ngram_matching" => {
                match value.parse() {
                    Ok(enabled) => {
                        self.parallel_ngram_matching = enabled;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid parallel_ngram_matching value (must be true/false): {}", value)
                    ))),
                }
            },
            "parallel_batch_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.parallel_batch_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid parallel_batch_size (must be > 0): {}", value)
                    ))),
                }
            },
            "parallel_min_file_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.parallel_min_file_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid parallel_min_file_size (must be > 0): {}", value)
                    ))),
                }
            },
            "log_discarded_matches" => {
                self.log_discarded_matches = value.parse::<bool>().unwrap_or(false);
                Some(Ok(()))
            },
            "discarded_log_path" => {
                self.discarded_log_path = value.to_string();
                Some(Ok(()))
            },
            "parallel_thread_count" => {
                match value.parse::<i64>() {
                    Ok(count) if count >= 0 => {
                        self.parallel_thread_count = count as usize;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid parallel_thread_count (must be >= 0): {}", value)
                    ))),
                }
            },
            "pipeline_processing" => {
                match value.parse() {
                    Ok(enabled) => {
                        self.pipeline_processing = enabled;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid pipeline_processing value (must be true/false): {}", value)
                    ))),
                }
            },
            "auto_select_strategy" => {
                match value.parse::<bool>() {
                    Ok(enabled) => {
                        self.auto_select_strategy = enabled;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid auto_select_strategy value (must be true/false): {}", value)
                    ))),
                }
            },
            "pipeline_ngram_threshold" => {
                match value.parse::<usize>() {
                    Ok(threshold) if threshold > 0 => {
                        self.pipeline_ngram_threshold = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid pipeline_ngram_threshold (must be > 0): {}", value)
                    ))),
                }
            },
            "pipeline_memory_threshold" => {
                match value.parse::<f64>() {
                    Ok(threshold) if (0.0..=1.0).contains(&threshold) => {
                        self.pipeline_memory_threshold = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid pipeline_memory_threshold (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            "min_pipeline_batch_size" => {
                match value.parse::<usize>() {
                    Ok(size) if size > 0 => {
                        self.min_pipeline_batch_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid min_pipeline_batch_size (must be > 0): {}", value)
                    ))),
                }
            },
            "max_pipeline_batch_size" => {
                match value.parse::<usize>() {
                    Ok(size) if size > 0 => {
                        self.max_pipeline_batch_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid max_pipeline_batch_size (must be > 0): {}", value)
                    ))),
                }
            },
            "pipeline_queue_capacity" => {
                match value.parse::<usize>() {
                    Ok(capacity) if capacity > 0 => {
                        self.pipeline_queue_capacity = capacity;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid pipeline_queue_capacity (must be > 0): {}", value)
                    ))),
                }
            },
            "storage_access_strategy" => {
                let strategy = value.trim_matches('"').to_lowercase();
                match strategy.as_str() {
                    "shared" | "pooled" | "per_worker" => {
                        self.storage_access_strategy = strategy;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid storage_access_strategy (must be 'shared', 'pooled', or 'per_worker'): {}", value)
                    ))),
                }
            },
            "storage_pool_size" => {
                match value.parse::<usize>() {
                    Ok(size) if size > 0 => {
                        self.storage_pool_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid storage_pool_size (must be > 0): {}", value)
                    ))),
                }
            },
            "early_isnad_detection" => {
                self.early_isnad_detection = value.parse::<bool>().unwrap_or(false);
                Some(Ok(()))
            },
            "banality_detection_mode" => {
                let mode = value.trim_matches('"').to_lowercase();
                match mode.as_str() {
                    "phrase" | "automatic" => {
                        self.banality_detection_mode = mode;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid banality_detection_mode (must be 'phrase' or 'automatic'): {}", value)
                    ))),
                }
            },
            "banality_auto_proportion" => {
                match value.parse::<f64>() {
                    Ok(proportion) if (0.0..=100.0).contains(&proportion) => {
                        self.banality_auto_proportion = proportion;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid banality_auto_proportion (must be between 0 and 100): {}", value)
                    ))),
                }
            },
            "banality_auto_threshold" => {
                match value.parse::<f64>() {
                    Ok(threshold) if (0.0..=100.0).contains(&threshold) => {
                        self.banality_auto_threshold = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid banality_auto_threshold (must be between 0 and 100): {}", value)
                    ))),
                }
            },
            _ => None,
        }
        
    }
}


impl ProcessorConfig {
    pub fn validate(&self) -> Result<()> {
        // Validate subsystems
        self.text_processing.validate()?;
        self.generator.validate()?;

        // Validate processor settings
        if self.batch_size == 0 {
            return Err(Error::Config(
                "batch_size must be greater than 0".to_string()
            ));
        }

        if !["sequential", "parallel"].contains(&self.processing_mode.as_str()) {
            return Err(Error::Config(
                format!("Invalid processing_mode: {}", self.processing_mode)
            ));
        }

        if self.max_concurrent_files == 0 {
            return Err(Error::Config(
                "max_concurrent_files must be greater than 0".to_string()
            ));
        }

        if !(0.0..=100.0).contains(&self.force_sequential_memory_threshold) {
            return Err(Error::Config(
                "force_sequential_memory_threshold must be between 0 and 100".to_string()
            ));
        }

        if !(0.0..=100.0).contains(&self.target_memory_percent) {
            return Err(Error::Config(
                "target_memory_percent must be between 0 and 100".to_string()
            ));
        }

        if self.target_memory_percent >= self.force_sequential_memory_threshold {
            return Err(Error::Config(
                "target_memory_percent must be less than force_sequential_memory_threshold".to_string()
            ));
        }

        if self.parallel_ngram_matching {
            if self.parallel_batch_size == 0 {
                return Err(Error::Config(
                    "parallel_batch_size must be greater than 0".to_string()
                ));
            }

            if self.parallel_min_file_size == 0 {
                return Err(Error::Config(
                    "parallel_min_file_size must be greater than 0".to_string()
                ));
            }

            // Thread count of 0 is valid (means use num_cpus)
            if self.parallel_thread_count > 0 {
                // If specific thread count given, ensure it's reasonable
                let num_cpus = num_cpus::get();
                if self.parallel_thread_count > num_cpus * 2 {
                    return Err(Error::Config(format!(
                        "parallel_thread_count {} is too high (system has {} CPUs)",
                        self.parallel_thread_count, num_cpus
                    )));
                }
            }
        }
        if self.auto_select_strategy || self.pipeline_processing {
            // Validate pipeline thresholds
            if self.pipeline_ngram_threshold == 0 {
                return Err(Error::Config(
                    "pipeline_ngram_threshold must be greater than 0".to_string()
                ));
            }
            
            if !(0.0..=1.0).contains(&self.pipeline_memory_threshold) {
                return Err(Error::Config(
                    "pipeline_memory_threshold must be between 0 and 1".to_string()
                ));
            }
            
            // Validate batch sizes
            if self.min_pipeline_batch_size == 0 {
                return Err(Error::Config(
                    "min_pipeline_batch_size must be greater than 0".to_string()
                ));
            }
            
            if self.max_pipeline_batch_size == 0 {
                return Err(Error::Config(
                    "max_pipeline_batch_size must be greater than 0".to_string()
                ));
            }
            
            if self.min_pipeline_batch_size > self.max_pipeline_batch_size {
                return Err(Error::Config(
                    "min_pipeline_batch_size must be less than or equal to max_pipeline_batch_size".to_string()
                ));
            }
            
            // Validate queue capacity
            if self.pipeline_queue_capacity == 0 {
                return Err(Error::Config(
                    "pipeline_queue_capacity must be greater than 0".to_string()
                ));
            }
            
            // Validate storage access strategy
            if !["shared", "pooled", "per_worker"].contains(&self.storage_access_strategy.as_str()) {
                return Err(Error::Config(
                    format!("Invalid storage_access_strategy: {}", self.storage_access_strategy)
                ));
            }
            
            // If using pooled strategy, validate pool size
            if self.storage_access_strategy == "pooled" && self.storage_pool_size == 0 {
                return Err(Error::Config(
                    "storage_pool_size must be greater than 0 when using 'pooled' access strategy".to_string()
                ));
            }
            // Validate banality detection settings
            if !["phrase", "automatic"].contains(&self.banality_detection_mode.as_str()) {
                return Err(Error::Config(
                    format!("Invalid banality_detection_mode: {}", self.banality_detection_mode)
                ));
            }

            if !(0.0..=100.0).contains(&self.banality_auto_proportion) {
                return Err(Error::Config(
                    "banality_auto_proportion must be between 0 and 100".to_string()
                ));
            }

            if !(0.0..=100.0).contains(&self.banality_auto_threshold) {
                return Err(Error::Config(
                    "banality_auto_threshold must be between 0 and 100".to_string()
                ));
            }

            // Optional: Additional validation for relationship between threshold and proportion
            if self.banality_auto_threshold < self.banality_auto_proportion {
                warn!("banality_auto_threshold ({}) is lower than banality_auto_proportion ({}), which might lead to unexpected results",
                    self.banality_auto_threshold, self.banality_auto_proportion);
            }
        }

        Ok(())
    }

    // Helper method to determine if parallel processing should be used
    pub fn should_use_parallel(&self, file_size: usize) -> bool {
        self.parallel_ngram_matching && file_size >= self.parallel_min_file_size
    }

    // Helper method to determine if pipeline processing should be used
    pub fn should_use_pipeline(&self, ngram_count: usize, text_size: usize, available_memory: usize) -> bool {
        // If auto-selection is disabled, check manual setting
        if !self.auto_select_strategy {
            return self.processing_mode == "pipeline";
        }
        
        // Estimate memory requirements (simplified example)
        let estimated_memory = (ngram_count * 500) + (text_size * 2) + (50 * 1024 * 1024);
        let memory_ratio = estimated_memory as f64 / available_memory as f64;
        
        // Decision logic
        ngram_count > self.pipeline_ngram_threshold || 
        memory_ratio > self.pipeline_memory_threshold
    }

    // Helper to get effective thread count
    pub fn get_thread_count(&self) -> usize {
        if self.parallel_thread_count > 0 {
            self.parallel_thread_count
        } else {
            num_cpus::get()
        }
    }
    pub fn adjust_for_system(&mut self) -> Result<()> {
        // Get system info
        if let Ok(mem_info) = sys_info::mem_info() {
            let total_mb = mem_info.total / 1024;
            let available_mb = mem_info.avail / 1024;
            let num_cpus = num_cpus::get();
            
            // Log system resource information
            info!("System resources detected: {} MB total memory, {} MB available, {} CPUs",
                total_mb, available_mb, num_cpus);
            
            // Scale batch size based on available memory
            self.batch_size = if total_mb > 16384 {  // More than 16GB
                100_000
            } else if total_mb > 8192 {  // More than 8GB
                50_000
            } else if total_mb > 4096 {  // More than 4GB
                25_000
            } else {
                10_000
            };

            // Optimize parallel processing
            if self.processing_mode == "parallel" {
                // Use 75% of available cores
                self.max_concurrent_files = std::cmp::max(
                    1,
                    (num_cpus as f32 * 0.75) as usize
                );
            }

            // Adjust memory thresholds
            self.force_sequential_memory_threshold = 90.0;  // More aggressive
            self.target_memory_percent = 80.0;  // Allow using more memory
            
            // Adjust parallel batch size based on available memory
            self.parallel_batch_size = if total_mb > 16384 {
                20000
            } else if total_mb > 8192 {
                10000
            } else {
                5000
            };
            
            // ==== Pipeline Configuration Adjustments ====
            
            // Adjust pipeline batch sizes based on available memory
            if total_mb > 16384 {  // More than 16GB
                self.min_pipeline_batch_size = 1000;
                self.max_pipeline_batch_size = 4000;
            } else if total_mb > 8192 {  // More than 8GB
                self.min_pipeline_batch_size = 750;
                self.max_pipeline_batch_size = 3000;
            } else if total_mb > 4096 {  // More than 4GB
                self.min_pipeline_batch_size = 500;
                self.max_pipeline_batch_size = 2000;
            } else {  // Less than 4GB
                self.min_pipeline_batch_size = 250;
                self.max_pipeline_batch_size = 1000;
            }
            
            // Adjust queue capacity based on CPU count and memory
            // Higher memory machines can handle more items in the queue
            let memory_factor = if total_mb > 8192 { 3 } else { 2 };
            self.pipeline_queue_capacity = std::cmp::max(4, num_cpus * memory_factor);
            
            // Adjust ngram threshold based on memory
            // This determines when to use pipeline mode vs batch processing
            self.pipeline_ngram_threshold = if total_mb > 16384 {
                150000  // Very high memory systems
            } else if total_mb > 8192 {
                100000  // High memory systems
            } else if total_mb > 4096 {
                50000   // Medium memory systems
            } else {
                25000   // Low memory systems
            };
            
            // Adjust memory threshold ratio based on available memory
            // Systems with more free memory can have a higher threshold
            // before switching to pipeline mode
            let memory_usage_ratio = 1.0 - (available_mb as f64 / total_mb as f64);
            if memory_usage_ratio > 0.7 {
                // System is already using a lot of memory, be more cautious
                self.pipeline_memory_threshold = 0.5;  // Switch to pipeline earlier
            } else if memory_usage_ratio > 0.5 {
                self.pipeline_memory_threshold = 0.6;
            } else {
                // Plenty of memory available
                self.pipeline_memory_threshold = 0.7;
            }
            
            // Determine optimal storage access strategy based on system
            // For higher memory systems, pooled access can be more efficient
            if total_mb > 16384 && num_cpus > 8 {
                self.storage_access_strategy = "pooled".to_string();
                self.storage_pool_size = num_cpus / 2;  // Use half of available CPUs
            } else {
                // For lower-spec systems, shared access is more memory-efficient
                self.storage_access_strategy = "shared".to_string();
            }
            
            self.pipeline_processing = total_mb > 4096; // Enable pipeline on systems with more than 4GB
        
            info!("System-adjusted configuration: batch_size={}, parallel_batch_size={}, pipeline_threshold={} ngrams, pipeline_enabled={}",
                self.batch_size, self.parallel_batch_size, self.pipeline_ngram_threshold, self.pipeline_processing);
        } else {
            warn!("Could not detect system information, using default settings");
        }

        Ok(())
    }
}