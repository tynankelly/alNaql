// src/config/memory.rs

use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs;
use crate::error::{Error, Result};

// Optimized constants for better performance
const OPTIMAL_CHUNK_SIZE: usize = 4 * 1024 * 1024;  // 4MB chunks for optimal file processing
const MIN_WRITE_BUFFER: usize = 64;                 // 64MB minimum write buffer
const MIN_BLOCK_CACHE: usize = 256;                 // 256MB minimum block cache
const DEFAULT_BATCH_SIZE: usize = 10000;            // Base batch size for calculations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    // General memory limits
    pub max_memory_usage_percent: f64,
    pub min_free_memory_percent: f64,
    
    // Temporary file settings
    pub temp_file_threshold_mb: usize,
    pub temp_buffer_size_mb: usize,
    
    // Batch processing settings
    pub small_batch_size: usize,
    pub large_batch_size: usize,
    pub batch_size_threshold_mb: usize,
    
    // Cleanup settings
    pub cleanup_interval_ngrams: usize,
    pub force_cleanup_memory_percent: f64,
    
    // Checkpoint settings
    pub checkpoint_interval_secs: u64,
    
    // Cache settings
    pub rocksdb_block_cache_mb: usize,
    pub rocksdb_write_buffer_mb: usize,
    pub rocksdb_max_write_buffer_number: i32,
    pub rocksdb_target_file_size_mb: usize,
    pub rocksdb_max_background_jobs: i32,
    pub rocksdb_compression_level: i32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_usage_percent: 75.0,
            min_free_memory_percent: 15.0,
            temp_file_threshold_mb: 256,
            temp_buffer_size_mb: (OPTIMAL_CHUNK_SIZE / (1024 * 1024)).max(4),  // Based on chunk size
            small_batch_size: DEFAULT_BATCH_SIZE / 2,     // 5000
            large_batch_size: DEFAULT_BATCH_SIZE * 2,     // 20000
            batch_size_threshold_mb: (OPTIMAL_CHUNK_SIZE / (1024 * 1024)) * 64, // 256MB
            cleanup_interval_ngrams: DEFAULT_BATCH_SIZE * 100,  // 1M
            force_cleanup_memory_percent: 90.0,
            checkpoint_interval_secs: 600,
            rocksdb_block_cache_mb: MIN_BLOCK_CACHE * 4,  // 1024MB
            rocksdb_write_buffer_mb: MIN_WRITE_BUFFER * 4,  // 256MB
            rocksdb_max_write_buffer_number: 6,
            rocksdb_target_file_size_mb: 256,
            rocksdb_max_background_jobs: 4,
            rocksdb_compression_level: 3,
        }
    }
}

impl MemoryConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("Failed to read memory config: {}", e)))?;
            
        serde_json::from_str(&content)
            .map_err(|e| Error::Config(format!("Failed to parse memory config: {}", e)))
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize memory config: {}", e)))?;
            
        fs::write(path, content)
            .map_err(|e| Error::Config(format!("Failed to write memory config: {}", e)))
    }

    pub fn calculate_batch_size(&self, file_size: u64) -> usize {
        if file_size > (self.batch_size_threshold_mb * 1024 * 1024) as u64 {
            // Calculate optimal batch size based on chunk size
            let num_chunks = (file_size as usize + OPTIMAL_CHUNK_SIZE - 1) / OPTIMAL_CHUNK_SIZE;
            (self.small_batch_size.max(OPTIMAL_CHUNK_SIZE / 1024))
                .min(file_size as usize / num_chunks)
        } else {
            self.large_batch_size
        }
    }

    pub fn should_use_temp_file(&self, size: usize) -> bool {
        size > self.temp_file_threshold_mb * 1024 * 1024
    }

    pub fn get_buffer_size(&self) -> usize {
        self.temp_buffer_size_mb * 1024 * 1024
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_memory_usage_percent > 95.0 || self.max_memory_usage_percent < 10.0 {
            return Err(Error::Config(
                "max_memory_usage_percent must be between 10 and 95".to_string()
            ));
        }

        if self.min_free_memory_percent > 50.0 || self.min_free_memory_percent < 5.0 {
            return Err(Error::Config(
                "min_free_memory_percent must be between 5 and 50".to_string()
            ));
        }

        if self.force_cleanup_memory_percent <= self.max_memory_usage_percent {
            return Err(Error::Config(
                "force_cleanup_memory_percent must be greater than max_memory_usage_percent".to_string()
            ));
        }

        if self.rocksdb_block_cache_mb < MIN_BLOCK_CACHE {
            return Err(Error::Config(
                format!("rocksdb_block_cache_mb must be at least {}", MIN_BLOCK_CACHE)
            ));
        }

        if self.rocksdb_compression_level < 0 || self.rocksdb_compression_level > 9 {
            return Err(Error::Config(
                "rocksdb_compression_level must be between 0 and 9".to_string()
            ));
        }

        Ok(())
    }

    pub fn get_rocksdb_options(&self) -> Vec<(&str, String)> {
        vec![
            ("max_background_jobs", self.rocksdb_max_background_jobs.to_string()),
            ("write_buffer_size", (self.rocksdb_write_buffer_mb * 1024 * 1024).to_string()),
            ("max_write_buffer_number", self.rocksdb_max_write_buffer_number.to_string()),
            ("target_file_size_base", (self.rocksdb_target_file_size_mb * 1024 * 1024).to_string()),
            ("compression_level", self.rocksdb_compression_level.to_string()),
        ]
    }

    pub fn adjust_for_system_memory(&mut self) -> Result<()> {
        if let Ok(mem_info) = sys_info::mem_info() {
            let total_mb = mem_info.total / 1024;
            let num_cpus = num_cpus::get();
            
            // More aggressive memory utilization based on system RAM
            if total_mb >= 32768 {  // 32GB+
                self.rocksdb_block_cache_mb = MIN_BLOCK_CACHE * 16;    // 4GB
                self.rocksdb_write_buffer_mb = MIN_WRITE_BUFFER * 8;   // 512MB
                self.large_batch_size = DEFAULT_BATCH_SIZE * 5;        // 50000
                self.max_memory_usage_percent = 80.0;
                self.temp_buffer_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 4;  // 16MB
                self.rocksdb_max_write_buffer_number = 8;
            } else if total_mb >= 16384 {  // 16GB+
                self.rocksdb_block_cache_mb = MIN_BLOCK_CACHE * 8;     // 2GB
                self.rocksdb_write_buffer_mb = MIN_WRITE_BUFFER * 4;   // 256MB
                self.large_batch_size = DEFAULT_BATCH_SIZE * 2.5 as usize;  // 25000
                self.max_memory_usage_percent = 75.0;
                self.temp_buffer_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 2;  // 8MB
                self.rocksdb_max_write_buffer_number = 6;
            } else if total_mb >= 8192 {   // 8GB+
                self.rocksdb_block_cache_mb = MIN_BLOCK_CACHE * 4;     // 1GB
                self.rocksdb_write_buffer_mb = MIN_WRITE_BUFFER * 2;   // 128MB
                self.large_batch_size = DEFAULT_BATCH_SIZE * 1.5 as usize;  // 15000
                self.max_memory_usage_percent = 70.0;
                self.temp_buffer_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024);  // 4MB
                self.rocksdb_max_write_buffer_number = 4;
            } else {  // Less than 8GB
                self.rocksdb_block_cache_mb = MIN_BLOCK_CACHE;         // 256MB
                self.rocksdb_write_buffer_mb = MIN_WRITE_BUFFER;       // 64MB
                self.large_batch_size = DEFAULT_BATCH_SIZE;            // 10000
                self.max_memory_usage_percent = 65.0;
                self.temp_buffer_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) / 2;  // 2MB
                self.rocksdb_max_write_buffer_number = 3;
            }
    
            // Optimize batch sizes
            self.small_batch_size = (self.large_batch_size / 3).max(DEFAULT_BATCH_SIZE / 2);
            
            // Adjust cleanup thresholds
            self.force_cleanup_memory_percent = (self.max_memory_usage_percent + 15.0).min(95.0);
            
            // Scale temp file threshold more aggressively
            self.temp_file_threshold_mb = ((total_mb / 10) as usize)
                .max(OPTIMAL_CHUNK_SIZE / (1024 * 1024))
                .min(2048);
            
            // Optimize for number of CPU cores
            self.rocksdb_max_background_jobs = ((num_cpus as f32 * 0.75) as i32).max(2);
            
            // Adjust compression based on CPU cores
            if num_cpus >= 16 {
                self.rocksdb_compression_level = 4;
                self.rocksdb_target_file_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 128;  // 512MB
            } else if num_cpus >= 8 {
                self.rocksdb_compression_level = 3;
                self.rocksdb_target_file_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 64;   // 256MB
            } else if num_cpus >= 4 {
                self.rocksdb_compression_level = 2;
                self.rocksdb_target_file_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 32;   // 128MB
            } else {
                self.rocksdb_compression_level = 1;
                self.rocksdb_target_file_size_mb = OPTIMAL_CHUNK_SIZE / (1024 * 1024) * 16;   // 64MB
            }

            // Adjust checkpoint interval based on memory
            self.checkpoint_interval_secs = if total_mb >= 16384 { 900 } else { 600 };
            
            // Adjust cleanup interval based on memory
            self.cleanup_interval_ngrams = if total_mb >= 16384 {
                DEFAULT_BATCH_SIZE * 200  // 2M
            } else {
                DEFAULT_BATCH_SIZE * 100  // 1M
            };
    
            // Validate final settings
            self.validate()?;
        }
    
        Ok(())
    }

    pub fn to_ini(&self) -> String {
        format!(
            "[memory]\n\
            max_memory_usage_percent = {}\n\
            min_free_memory_percent = {}\n\
            temp_file_threshold_mb = {}\n\
            temp_buffer_size_mb = {}\n\
            small_batch_size = {}\n\
            large_batch_size = {}\n\
            batch_size_threshold_mb = {}\n\
            cleanup_interval_ngrams = {}\n\
            force_cleanup_memory_percent = {}\n\
            checkpoint_interval_secs = {}\n\
            \n\
            [rocksdb]\n\
            block_cache_mb = {}\n\
            write_buffer_mb = {}\n\
            max_write_buffer_number = {}\n\
            target_file_size_mb = {}\n\
            max_background_jobs = {}\n\
            compression_level = {}\n",
            self.max_memory_usage_percent,
            self.min_free_memory_percent,
            self.temp_file_threshold_mb,
            self.temp_buffer_size_mb,
            self.small_batch_size,
            self.large_batch_size,
            self.batch_size_threshold_mb,
            self.cleanup_interval_ngrams,
            self.force_cleanup_memory_percent,
            self.checkpoint_interval_secs,
            self.rocksdb_block_cache_mb,
            self.rocksdb_write_buffer_mb,
            self.rocksdb_max_write_buffer_number,
            self.rocksdb_target_file_size_mb,
            self.rocksdb_max_background_jobs,
            self.rocksdb_compression_level
        )
    }

    pub fn from_ini(content: &str) -> Result<Self> {
        let mut config = Self::default();
        let mut current_section = "";

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                current_section = line.trim_matches(|c| c == '[' || c == ']');
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                match (current_section, key) {
                    ("memory", "max_memory_usage_percent") => {
                        config.max_memory_usage_percent = value.parse().map_err(|_| {
                            Error::Config("Invalid max_memory_usage_percent".to_string())
                        })?;
                    },
                    ("memory", "min_free_memory_percent") => {
                        config.min_free_memory_percent = value.parse().map_err(|_| {
                            Error::Config("Invalid min_free_memory_percent".to_string())
                        })?;
                    },
                    ("memory", "temp_file_threshold_mb") => {
                        config.temp_file_threshold_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid temp_file_threshold_mb".to_string())
                        })?;
                    },
                    ("memory", "temp_buffer_size_mb") => {
                        config.temp_buffer_size_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid temp_buffer_size_mb".to_string())
                        })?;
                    },
                    ("memory", "small_batch_size") => {
                        config.small_batch_size = value.parse().map_err(|_| {
                            Error::Config("Invalid small_batch_size".to_string())
                        })?;
                    },
                    ("memory", "large_batch_size") => {
                        config.large_batch_size = value.parse().map_err(|_| {
                            Error::Config("Invalid large_batch_size".to_string())
                        })?;
                    },
                    ("memory", "batch_size_threshold_mb") => {
                        config.batch_size_threshold_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid batch_size_threshold_mb".to_string())
                        })?;
                    },
                    ("memory", "cleanup_interval_ngrams") => {
                        config.cleanup_interval_ngrams = value.parse().map_err(|_| {
                            Error::Config("Invalid cleanup_interval_ngrams".to_string())
                        })?;
                    },
                    ("memory", "force_cleanup_memory_percent") => {
                        config.force_cleanup_memory_percent = value.parse().map_err(|_| {
                            Error::Config("Invalid force_cleanup_memory_percent".to_string())
                        })?;
                    },
                    ("memory", "checkpoint_interval_secs") => {
                        config.checkpoint_interval_secs = value.parse().map_err(|_| {
                            Error::Config("Invalid checkpoint_interval_secs".to_string())
                        })?;
                    },
                    ("rocksdb", "block_cache_mb") => {
                        config.rocksdb_block_cache_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid block_cache_mb".to_string())
                        })?;
                    },
                    ("rocksdb", "write_buffer_mb") => {
                        config.rocksdb_write_buffer_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid write_buffer_mb".to_string())
                        })?;
                    },
                    ("rocksdb", "max_write_buffer_number") => {
                        config.rocksdb_max_write_buffer_number = value.parse().map_err(|_| {
                            Error::Config("Invalid max_write_buffer_number".to_string())
                        })?;
                    },
                    ("rocksdb", "target_file_size_mb") => {
                        config.rocksdb_target_file_size_mb = value.parse().map_err(|_| {
                            Error::Config("Invalid target_file_size_mb".to_string())
                        })?;
                    },
                    ("rocksdb", "max_background_jobs") => {
                        config.rocksdb_max_background_jobs = value.parse().map_err(|_| {
                            Error::Config("Invalid max_background_jobs".to_string())
                        })?;
                    },
                    ("rocksdb", "compression_level") => {
                        config.rocksdb_compression_level = value.parse().map_err(|_| {
                            Error::Config("Invalid compression_level".to_string())
                        })?;
                    },
                    _ => {}
                }
            }
        }

        config.validate()?;
        Ok(config)
    }
}