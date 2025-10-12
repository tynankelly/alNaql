// config/storage.rs - Storage backend (LMDB) configuration

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::{Error, Result};
use crate::config::loader::Configurable;


/// Configuration for the LMDB storage backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to the LMDB database
    pub db_path: PathBuf,
    
    /// Whether to force sync on every transaction
    pub use_fsync: bool,
    
    /// Whether we're in bulk loading mode (optimizes for initial load)
    pub is_bulk_loading: bool,
    
    /// Whether to allow memory-mapped reads
    pub allow_mmap_reads: bool,
    
    /// Whether to allow memory-mapped writes
    pub allow_mmap_writes: bool,
    
    /// Maximum number of LMDB reader slots
    pub lmdb_max_readers: Option<u32>,
    
    /// Maximum number of named LMDB databases
    pub lmdb_max_dbs: Option<u32>,

    // ===== COMPACTION CONFIGURATION =====
    
    /// Whether to enable automatic database compaction after generation
    pub enable_post_generation_compaction: bool,
    
    /// Buffer percentage to add to actual data size (e.g., 20.0 for 20% buffer)
    pub compaction_buffer_percentage: f64,
    
    /// Number of records to copy in each batch during compaction
    pub compaction_batch_size: usize,
    
    /// Minimum size for compacted database in MB (prevents too-small databases)
    pub compaction_min_size_mb: u64,
    
    /// Whether to keep the original database after compaction (for debugging)
    pub keep_original_on_compaction: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("ngrams"),
            use_fsync: false,
            is_bulk_loading: false,
            allow_mmap_reads: true,
            allow_mmap_writes: true,
            lmdb_max_readers: Some(126),
            lmdb_max_dbs: Some(100),
            // Compaction defaults
            enable_post_generation_compaction: true,  // Enabled by default to save space
            compaction_buffer_percentage: 20.0,       // 20% buffer is a good balance
            compaction_batch_size: 10000,            // 10k records per batch
            compaction_min_size_mb: 100,             // 100MB minimum
            keep_original_on_compaction: false, 
        }
    }
}

impl Configurable for StorageConfig {
    fn section_name() -> &'static str {
        "storage"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            // Handle both old and new names for db_path
            "db_path" => {
                self.db_path = PathBuf::from(value.trim_matches('"'));
                Ok(true)
            },
            "use_fsync" => {
                self.use_fsync = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "is_bulk_loading" => {
                self.is_bulk_loading = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "allow_mmap_reads" => {
                self.allow_mmap_reads = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "allow_mmap_writes" => {
                self.allow_mmap_writes = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "lmdb_max_readers" => {
                self.lmdb_max_readers = crate::config::loader::parse::optional_u32(value)?;
                Ok(true)
            },
            "lmdb_max_dbs" => {
                self.lmdb_max_dbs = crate::config::loader::parse::optional_u32(value)?;
                Ok(true)
            },
            // Compaction configuration
            "enable_post_generation_compaction" => {
                self.enable_post_generation_compaction = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "compaction_buffer_percentage" => {
                let percentage = crate::config::loader::parse::f64(value)?;
                if percentage < 0.0 || percentage > 100.0 {
                    return Err(Error::Config(format!(
                        "compaction_buffer_percentage must be between 0 and 100, got {}", 
                        percentage
                    )));
                }
                self.compaction_buffer_percentage = percentage;
                Ok(true)
            },
            "compaction_batch_size" => {
                let batch_size = crate::config::loader::parse::usize(value)?;
                if batch_size == 0 {
                    return Err(Error::Config(
                        "compaction_batch_size must be greater than 0".to_string()
                    ));
                }
                self.compaction_batch_size = batch_size;
                Ok(true)
            },
            "compaction_min_size_mb" => {
                let min_size = crate::config::loader::parse::u64(value)?;
                if min_size < 10 {
                    return Err(Error::Config(
                        "compaction_min_size_mb must be at least 10 MB".to_string()
                    ));
                }
                self.compaction_min_size_mb = min_size;
                Ok(true)
            },
            "keep_original_on_compaction" => {
                self.keep_original_on_compaction = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
                
        // Validate LMDB settings if provided
        if let Some(max_readers) = self.lmdb_max_readers {
            if max_readers == 0 {
                return Err(Error::Config(
                    "lmdb_max_readers must be greater than 0".to_string()
                ));
            }
        }
        
        if let Some(max_dbs) = self.lmdb_max_dbs {
            if max_dbs == 0 {
                return Err(Error::Config(
                    "lmdb_max_dbs must be greater than 0".to_string()
                ));
            }
        }
        
        // Create db directory if it doesn't exist
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::Config(
                format!("Failed to create database directory: {}", e)
            ))?;
        }

        // Validate compaction settings
        if self.enable_post_generation_compaction {
            // Ensure buffer percentage is reasonable
            if self.compaction_buffer_percentage < 5.0 {
                return Err(Error::Config(
                    "compaction_buffer_percentage below 5% is not recommended".to_string()
                ));
            }
            
            // Ensure batch size is reasonable
            if self.compaction_batch_size > 1_000_000 {
                return Err(Error::Config(
                    "compaction_batch_size above 1 million may use excessive memory".to_string()
                ));
            }
        }
        
        Ok(())
    }

    
}

impl StorageConfig {
    
    /// Check if we're in bulk loading mode
    pub fn is_bulk_mode(&self) -> bool {
        self.is_bulk_loading
    }
    
    /// Get LMDB-specific environment options
    pub fn get_lmdb_options(&self) -> Vec<(&str, String)> {
        let mut options = Vec::new();
        
        if let Some(max_readers) = self.lmdb_max_readers {
            options.push(("max_readers", max_readers.to_string()));
        }
        
        if let Some(max_dbs) = self.lmdb_max_dbs {
            options.push(("max_dbs", max_dbs.to_string()));
        }
        
        if !self.use_fsync {
            options.push(("no_sync", "true".to_string()));
        }
        
        options
    }
    
    /// Get the database path
    pub fn database_path(&self) -> &PathBuf {
        &self.db_path
    }

    /// Calculate the optimal compacted database size based on actual data size
    pub fn calculate_compacted_size(&self, actual_data_size: u64) -> usize {
        let buffer_multiplier = 1.0 + (self.compaction_buffer_percentage / 100.0);
        let with_buffer = (actual_data_size as f64 * buffer_multiplier) as u64;
        
        // Ensure minimum size
        let min_size_bytes = self.compaction_min_size_mb * 1024 * 1024;
        let final_size = with_buffer.max(min_size_bytes);
        
        // Ensure page alignment (4KB boundary)
        ((final_size + 4095) / 4096 * 4096) as usize
    }
    
    /// Check if compaction would save significant space
    pub fn is_compaction_worthwhile(&self, current_size: u64, optimal_size: u64) -> bool {
        // Only compact if we can save at least 20% or 100MB
        let space_saved = current_size.saturating_sub(optimal_size);
        let percent_saved = (space_saved as f64 / current_size as f64) * 100.0;
        
        space_saved > 100 * 1024 * 1024 || percent_saved > 20.0
    }
}
