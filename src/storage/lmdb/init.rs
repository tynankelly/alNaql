// storage/lmdb/init.rs

use std::fs;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use std::num::NonZeroUsize;
use log::{info, debug, error};
use lmdb_rkv::{Environment, Database, DatabaseFlags, Transaction, Error as LmdbError};
use lru::LruCache;
use parking_lot::RwLock;
use sys_info;

use crate::error::{Error, Result};
use crate::config::subsystems::storage::StorageConfig;
use crate::storage::lmdb::config::calculate_map_size_from_file_size;
use crate::storage::metrics::StorageMetrics;

use super::{LMDBStorage, DB_CACHE};
use super::config::{
    create_env_options,
    CF_MAIN, 
    CF_METADATA,
    CF_NGRAM_METADATA,
    calculate_batch_size
};

// Prefix DB definitions - we'll need to create the same prefix databases that were used in RocksDB
const PREFIX_PARTITIONS: [(&str, &[char]); 30] = [
    // Space prefix
    ("prefix_space", &[' ']),
    
    // Arabic letter partitions with their corresponding characters
    ("prefix_alef", &['ا', 'أ', 'إ', 'آ']),
    ("prefix_ba", &['ب']),
    ("prefix_ta", &['ت']),
    ("prefix_tha", &['ث']),
    ("prefix_jim", &['ج']),
    ("prefix_ha", &['ح']),
    ("prefix_kha", &['خ']),
    ("prefix_dal", &['د']),
    ("prefix_thal", &['ذ']),
    ("prefix_ra", &['ر']),
    ("prefix_zain", &['ز']),
    ("prefix_sin", &['س']),
    ("prefix_shin", &['ش']),
    ("prefix_sad", &['ص']),
    ("prefix_dad", &['ض']),
    ("prefix_ta_marbuta", &['ط']),
    ("prefix_dha", &['ظ']),
    ("prefix_ain", &['ع']),
    ("prefix_ghain", &['غ']),
    ("prefix_fa", &['ف']),
    ("prefix_qaf", &['ق']),
    ("prefix_kaf", &['ك']),
    ("prefix_lam", &['ل']),
    ("prefix_mim", &['م']),
    ("prefix_nun", &['ن']),
    ("prefix_ha_final", &['ه', 'ة']),
    ("prefix_waw", &['و', 'ؤ']),
    ("prefix_ya", &['ي', 'ى', 'ئ']),
    ("prefix_hamza", &['ء']),
];

impl LMDBStorage {
    /// Creates a new LMDBStorage instance with the given path and configuration
    pub fn new<P: AsRef<Path>>(path: P, config: StorageConfig, file_size: Option<u64>) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        
        // Make sure the directory exists
        if !path_buf.exists() {
            fs::create_dir_all(&path_buf)
            .map_err(|e| Error::storage(format!("Failed to create database directory: {}", e)))?;
        }
        
        // Get environment flags
        let (env_flags, max_readers, max_dbs, mut map_size) = create_env_options(&config);
        
        // Override map size based on file size if available
        if let Some(file_size) = file_size {
            let calculated_map_size = calculate_map_size_from_file_size(file_size);
            
            // Only use calculated size if it's larger than the configured size
            if calculated_map_size > map_size {
                info!("Overriding configured map size ({} MB) with calculated size ({} MB) based on file size",
                    map_size / (1024 * 1024),
                    calculated_map_size / (1024 * 1024));
                map_size = calculated_map_size;
            }
        }
        
        // Log environment setup
        info!(
            "Creating LMDB environment at {:?} with map_size={} MB, max_readers={}, max_dbs={}",
            path_buf,
            map_size / (1024 * 1024),
            max_readers,
            max_dbs
        );
        
        // Wrap environment creation in a proper error handler
        let env = Self::create_environment(&path_buf, env_flags, max_readers, max_dbs, map_size)?;
        
        // Create database handles for main, metadata, and ngram_metadata
        let main_db = Self::create_database(&env, CF_MAIN, DatabaseFlags::empty())?;
        let metadata_db = Self::create_database(&env, CF_METADATA, DatabaseFlags::empty())?;
        let ngram_metadata_db = Self::create_database(&env, CF_NGRAM_METADATA, DatabaseFlags::empty())?;
        
        // Create prefix databases with DUP_SORT flag for multiple values per key
        let mut prefix_dbs = HashMap::new();
        for &(db_name, _chars) in &PREFIX_PARTITIONS {
            let db = Self::create_database(&env, db_name, DatabaseFlags::empty())?;
            prefix_dbs.insert(db_name.to_string(), db);
        }
        
        // Calculate initial batch size based on available memory
        let mem_info = sys_info::mem_info()
            .map_err(|e| Error::Config(format!("Failed to get system memory info: {}", e)))?;
        
        let available_memory = (mem_info.total as f64 * 0.7) as usize * 1024; // 70% of total RAM in bytes
        let initial_batch_size = calculate_batch_size(available_memory);

        // Calculate memory-based cache size for prefix cache
        let total_mem_mb = mem_info.total / 1024;
        
        // Allocate a significant portion of memory to prefix cache (10-15%)
        let prefix_cache_size = {
            // Base calculation: 15% of total memory for prefix cache
            let calculated_size = (total_mem_mb as f64 * 0.15) as usize;
            
            // Minimum 512MB, maximum 8GB, but never more than 25% of total memory
            let max_allowed = (total_mem_mb as f64 * 0.25) as usize;
            
            // Apply constraints
            calculated_size.clamp(512, max_allowed.min(8 * 1024))
        };
        
        // Convert from MB to entry count (rough estimate based on average entry size)
        let estimated_entries = (prefix_cache_size * 1024 * 1024) / 100;
        
        // Create cache with calculated size
        let prefix_cache = Arc::new(RwLock::new(
            LruCache::new(NonZeroUsize::new(estimated_entries).unwrap())
        ));
        
        // Calculate metadata cache size (using 10% of total memory)
        let metadata_cache_size = {
            let calculated_size = (total_mem_mb as f64 * 0.10) as usize;
            calculated_size.clamp(100, 2048)
        };

        // Convert MB to entry count (each entry ~40 bytes)
        let metadata_entries = (metadata_cache_size * 1024 * 1024) / 40;

        // Create the metadata cache
        let metadata_cache = Arc::new(RwLock::new(
            LruCache::new(NonZeroUsize::new(metadata_entries).unwrap())
        ));
    
        // Create and initialize the storage instance
        let mut storage = Self {
            env: Arc::new(env),
            main_db,
            metadata_db,
            ngram_metadata_db,
            prefix_dbs,
            metrics: Arc::new(StorageMetrics::default()),
            is_bulk_loading: false,
            current_memory_usage: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            adaptive_batch_size: initial_batch_size,
            prefix_cache,
            metadata_cache,
            db_path: path_buf.clone(),
            map_size,
        };

        // Ensure all databases exist and are accessible
        storage.ensure_databases_exist()?;  

        // Add to global DB cache
        DB_CACHE.insert(path_buf.clone(), Arc::new(parking_lot::Mutex::new(storage.clone())));
    
        info!("Initialized LMDB storage at {:?} with batch size: {}", path_buf, initial_batch_size);
        
        Ok(storage)
    }

    /// Helper method to create an LMDB environment with error handling
    fn create_environment(
        path: &Path,
        flags: lmdb_rkv::EnvironmentFlags,
        max_readers: u32,
        max_dbs: u32,
        map_size: usize
    ) -> Result<Environment> {
        debug!("Creating LMDB environment at {:?}", path);
        
        // Create environment builder with configurated settings
        match Environment::new()
            .set_flags(flags)
            .set_max_readers(max_readers)
            .set_max_dbs(max_dbs)
            .set_map_size(map_size)
            .open(path) {
            Ok(env) => {
                debug!("Successfully created LMDB environment");
                Ok(env)
            },
            Err(e) => {
                // Special handling for common LMDB errors
                match e {
                    // Keep MapResized if it exists in your library
                    e if e.to_string().contains("map resize") || e.to_string().contains("map full") => {
                        // Try again with double the map size
                        error!("Map size too small, attempting with larger size: {} MB", map_size*2/(1024*1024));
                        Self::create_environment(path, flags, max_readers, max_dbs, map_size * 2)
                    },
                    // Handle map full errors
                    e if e.to_string().contains("map full") => {
                        error!("LMDB map full. Please increase map size in configuration.");
                        Err(Error::storage(format!("LMDB map full: {}", e)))
                    },
                    // Handle busy environment errors
                    e if e.to_string().contains("busy") || e.to_string().contains("in use") => {
                        error!("LMDB environment is busy. Another process may be using it.");
                        Err(Error::storage(format!("LMDB environment busy: {}", e)))
                    },
                    // Handle version incompatibility
                    e if e.to_string().contains("version") || e.to_string().contains("compat") => {
                        error!("LMDB version mismatch. Database was created with an incompatible version.");
                        Err(Error::storage(format!("LMDB version mismatch: {}", e)))
                    },
                    // Default case for all other errors
                    _ => {
                        error!("Failed to open LMDB environment: {}", e);
                        Err(Error::storage(format!("Failed to open LMDB environment: {}", e)))
                    }
                }
            }
        }
    }

    /// Helper method to create a database handle with error handling
    fn create_database(env: &Environment, name: &str, flags: DatabaseFlags) -> Result<Database> {
        debug!("Creating database: {}", name);
        
        match env.create_db(Some(name), flags) {
            Ok(db) => {
                debug!("Successfully created database: {}", name);
                Ok(db)
            },
            Err(e) => {
                match e {
                    LmdbError::NotFound => {
                        // Database doesn't exist yet, retry with creation flags
                        debug!("Database {} doesn't exist yet, creating", name);
                        match env.create_db(Some(name), flags) {
                            Ok(db) => Ok(db),
                            Err(e) => {
                                error!("Failed to create database {}: {}", name, e);
                                Err(Error::Database(format!("Failed to create database {}: {}", name, e)))
                            }
                        }
                    },
                    LmdbError::DbsFull => {
                        error!("Maximum number of databases reached. Increase max_dbs in configuration.");
                        Err(Error::Database(format!("LMDB max databases reached: {}", e)))
                    },
                    _ => {
                        error!("Failed to open database {}: {}", name, e);
                        Err(Error::Database(format!("Failed to open database {}: {}", name, e)))
                    }
                }
            }
        }
    }

    /// Ensures the database has all needed tables/databases
    pub(crate) fn ensure_databases_exist(&mut self) -> Result<()> {
        // Check if all needed databases exist
        // For LMDB, databases are created when opened, so this is mostly a validation step
        
        // Check main databases
        for name in &[CF_MAIN, CF_METADATA, CF_NGRAM_METADATA] {
            // Try a simple operation to see if database exists and is accessible
            let txn = self.env.begin_ro_txn()
                .map_err(|e| Error::Database(format!("Failed to start transaction: {}", e)))?;
            
            let _db = unsafe { txn.open_db(Some(name)) }
                .map_err(|e| {
                    if let LmdbError::NotFound = e {
                        // Database doesn't exist, we'll create it
                        debug!("Database {} needs to be created", name);
                        Error::storage(format!("Database {} not found", name))
                    } else {
                        Error::Database(format!("Error opening database {}: {}", name, e))
                    }
                })?;
            
            // Database exists and is accessible
            debug!("Database {} exists and is accessible", name);
            
            // End transaction
            txn.abort();
        }
        
        // Check prefix databases
        for (prefix_name, _) in &PREFIX_PARTITIONS {
            let txn = self.env.begin_ro_txn()
                .map_err(|e| Error::Database(format!("Failed to start transaction: {}", e)))?;
            
            let _db = unsafe { txn.open_db(Some(prefix_name)) }
                .map_err(|e| {
                    if let LmdbError::NotFound = e {
                        // Database doesn't exist, we'll need to create it
                        debug!("Prefix database {} needs to be created", prefix_name);
                        Error::storage(format!("Database {} not found", prefix_name))
                    } else {
                        Error::Database(format!("Error opening database {}: {}", prefix_name, e))
                    }
                })?;
            
            // End transaction
            txn.abort();
        }
        
        Ok(())
    }

    /// Verify the LMDB environment is usable
    pub fn verify_environment(&self) -> Result<()> {
        debug!("Verifying LMDB environment");
        
        // Start a read transaction
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start transaction: {}", e)))?;
        
        // Get environment stats (this works based on your code)
        match (*self.env).stat() {
            Ok(stat) => {
                debug!("Environment stats: entries={}, branch_pages={}, leaf_pages={}, overflow_pages={}",
                      stat.entries(), stat.branch_pages(), stat.leaf_pages(), stat.overflow_pages());
            },
            Err(e) => {
                error!("Failed to get environment stats: {}", e);
                return Err(Error::Database(format!("Failed to get environment stats: {}", e)));
            }
        }
        
        // End transaction
        txn.abort();
        
        debug!("LMDB environment verification completed successfully");
        Ok(())
    }

    /// Check if the LMDB environment needs to be resized
    pub fn check_map_size(&self) -> Result<bool> {
        // We know (*self.env).stat() works - this is confirmed
        let env_stat = (*self.env).stat()
            .map_err(|e| Error::Database(format!("Failed to get environment stats: {}", e)))?;
        
        // Get pages used
        let pages_used = env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages();
        
        // Since we can't directly query the map size through stat or info methods,
        // we need to use the value that was configured when creating the environment
        
        // Start a read transaction (we know this works)
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start transaction: {}", e)))?;
        
        // End transaction, since we're not using it for anything
        txn.abort();
        
        // Use the map_size from when we created the environment
        // This assumes you're storing it or have a way to get it
        // For example, add a field to LMDBStorage to store map_size when created
        let map_size = self.map_size; // You would need to add this field
        
        // If you don't have a field like this, you need to either:
        // 1. Add one, or
        // 2. Use a conservative estimate
        
        // Calculate percentage
        let total_pages = map_size / 4096; // 4KB page size
        let used_percent = (pages_used as f64 * 100.0) / total_pages as f64;
        
        debug!("LMDB map usage: {:.1}% ({} of {} pages)",
            used_percent, pages_used, total_pages);
        
        // Return true if we're over 80% full
        Ok(used_percent > 80.0)
    }

    /// Resize the LMDB environment to a new map size
    #[allow(dead_code)]  // This might be used later
    pub fn resize_environment(&mut self) -> Result<()> {
        // LMDB doesn't support resizing an open environment
        // We need to close and reopen it with a new size
        
        error!("LMDB environment resize requested, but this requires reopening the database");
        error!("Current implementation does not support dynamic resizing");
        error!("Please increase the map_size in configuration and restart the application");
        
        Err(Error::Database("LMDB environment resize not supported at runtime".to_string()))
    }
}