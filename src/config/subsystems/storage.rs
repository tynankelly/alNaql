// src/config/subsystems/storage.rs

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::{Error, Result};
use crate::config::FromIni;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    // Base path
    pub db_path: PathBuf,
    
    // I/O settings
    pub use_direct_io: bool,
    pub use_fsync: bool,

    // Cache settings
    pub cache_size_mb: usize,

    // General processing settings
    pub batch_size: usize,
    pub parallel_threads: usize,
    pub is_bulk_loading: bool,
    
    // Memory mapping - though LMDB always uses memory mapping,
    // keeping these for API compatibility
    pub allow_mmap_reads: bool,
    pub allow_mmap_writes: bool,
    
    // LMDB-specific settings
    pub lmdb_max_readers: Option<u32>,     // Maximum number of reader slots
    pub lmdb_max_dbs: Option<u32>,         // Maximum number of named databases
    pub lmdb_map_size_mb: Option<usize>,   // Memory map size in megabytes
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            // Base path
            db_path: PathBuf::from("db/ngrams"),
            
            // I/O settings
            use_direct_io: false,
            use_fsync: false,

            // Cache settings
            cache_size_mb: 4096,

            // General processing settings
            batch_size: 1000,
            parallel_threads: 4,
            is_bulk_loading: false,
            
            // Memory mapping settings (LMDB always uses memory mapping)
            allow_mmap_reads: true,
            allow_mmap_writes: true,
            
            // LMDB-specific settings
            lmdb_max_readers: Some(126),      // Default max readers
            lmdb_max_dbs: Some(100),          // Default max databases
            lmdb_map_size_mb: Some(10240),    // Default 10GB map size
        }
    }
}

impl FromIni for StorageConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        if section_name != "storage" {
            return None;
        }

        match key {
            // Base path
            "db_path" | "rocksdb_path" => {  // Support both old and new names
                self.db_path = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },

            // I/O settings
            "use_direct_io" => {
                match value.parse() {
                    Ok(flag) => {
                        self.use_direct_io = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid use_direct_io value (must be true/false): {}", value)
                    ))),
                }
            },
            "use_fsync" => {
                match value.parse() {
                    Ok(flag) => {
                        self.use_fsync = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid use_fsync value (must be true/false): {}", value)
                    ))),
                }
            },

            // Cache settings
            "cache_size_mb" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.cache_size_mb = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid cache_size_mb (must be > 0): {}", value)
                    ))),
                }
            },

            // General processing settings
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
            "parallel_threads" => {
                match value.parse() {
                    Ok(num) if num > 0 => {
                        self.parallel_threads = num;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid parallel_threads (must be > 0): {}", value)
                    ))),
                }
            },
            "is_bulk_loading" => {
                match value.parse() {
                    Ok(flag) => {
                        self.is_bulk_loading = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid is_bulk_loading value (must be true/false): {}", value)
                    ))),
                }
            },

            // Memory mapping settings
            "allow_mmap_reads" => {
                match value.parse() {
                    Ok(flag) => {
                        self.allow_mmap_reads = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid allow_mmap_reads value (must be true/false): {}", value)
                    ))),
                }
            },
            "allow_mmap_writes" => {
                match value.parse() {
                    Ok(flag) => {
                        self.allow_mmap_writes = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid allow_mmap_writes value (must be true/false): {}", value)
                    ))),
                }
            },

            // LMDB-specific settings
            "lmdb_max_readers" => {
                match value.parse() {
                    Ok(readers) if readers > 0 => {
                        self.lmdb_max_readers = Some(readers);
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid lmdb_max_readers (must be > 0): {}", value)
                    ))),
                }
            },
            "lmdb_max_dbs" => {
                match value.parse() {
                    Ok(dbs) if dbs > 0 => {
                        self.lmdb_max_dbs = Some(dbs);
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid lmdb_max_dbs (must be > 0): {}", value)
                    ))),
                }
            },
            "lmdb_map_size_mb" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.lmdb_map_size_mb = Some(size);
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid lmdb_map_size_mb (must be > 0): {}", value)
                    ))),
                }
            },

            // Ignore all other RocksDB-specific settings
            "compression" | "compression_level" |
            "cache_strict_capacity" | "row_cache_size_mb" |
            "write_buffer_size" | "max_write_buffer_number" | "min_write_buffer_number" |
            "max_background_jobs" | "max_background_compactions" | "max_background_flushes" |
            "max_subcompactions" | "target_file_size_mb" | "max_open_files" | 
            "keep_log_file_num" | "arena_block_size" => {
                // Simply ignore these RocksDB-specific settings
                Some(Ok(()))
            },

            // Unknown setting
            _ => None,
        }
    }
}

impl StorageConfig {
    pub fn validate(&self) -> Result<()> {
        // Validate cache settings
        if self.cache_size_mb == 0 {
            return Err(Error::Config(
                "cache_size_mb must be greater than 0".to_string()
            ));
        }

        // Additional LMDB-specific validations
        if let Some(map_size) = self.lmdb_map_size_mb {
            if map_size < 10 {
                return Err(Error::Config(
                    "lmdb_map_size_mb should be at least 10MB".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Adjust settings based on available system resources
    pub fn adjust_for_system(&mut self) -> Result<()> {
        // Get system memory info
        if let Ok(mem_info) = sys_info::mem_info() {
            let total_mb = mem_info.total / 1024;
            let available_mb = mem_info.avail / 1024;
            
            // Adjust cache size based on available memory
            self.cache_size_mb = (available_mb as f64 * 0.3) as usize;  // Use 30% of available memory
            
            // Adjust LMDB map size based on available memory
            // Use 60% of total memory as a default if not specified
            if self.lmdb_map_size_mb.is_none() {
                self.lmdb_map_size_mb = Some((total_mb as f64 * 0.6) as usize);
            }
        }

        // Adjust thread count based on CPU cores
        let num_cpus = num_cpus::get();
        self.parallel_threads = std::cmp::max(2, num_cpus / 2);  // Use half of available cores

        Ok(())
    }

    /// Get a human-readable description of the configuration
    pub fn describe(&self) -> String {
        format!(
            "Storage Configuration:\n\
             - Database Path: {:?}\n\
             - Cache Size: {} MB\n\
             - Batch Size: {}\n\
             - Parallel Threads: {}\n\
             - Memory Mapping: reads={}, writes={}\n\
             - Direct I/O: {}\n\
             - Sync Mode: {}\n\
             - Bulk Loading Mode: {}\n\
             - LMDB Map Size: {} MB\n\
             - LMDB Max Readers: {}\n\
             - LMDB Max Databases: {}",
            self.db_path,
            self.cache_size_mb,
            self.batch_size,
            self.parallel_threads,
            self.allow_mmap_reads,
            self.allow_mmap_writes,
            self.use_direct_io,
            if self.use_fsync { "sync (fsync)" } else { "async" },
            self.is_bulk_loading,
            self.lmdb_map_size_mb.unwrap_or(10240),
            self.lmdb_max_readers.unwrap_or(126),
            self.lmdb_max_dbs.unwrap_or(100)
        )
    }

    /// Get optimal batch size for the current configuration
    pub fn get_optimal_batch_size(&self) -> usize {
        // For LMDB, batch size is primarily about transaction size management
        // We use the configured batch size directly 
        self.batch_size
    }
}