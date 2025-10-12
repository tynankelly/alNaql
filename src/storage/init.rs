// storage/lmdb/init.rs
// Database initialization with atomic creation and comprehensive error handling

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Duration;
use log::{info, debug, trace, error, warn};
use lmdb_rkv::{Environment, EnvironmentFlags, Database, DatabaseFlags, Transaction, RwTransaction, Error as LmdbError};
use lru::LruCache;
use parking_lot::RwLock;
use sys_info;
use libc; // For error code constants (EAGAIN, ENOSPC, EIO, etc.)

use crate::error::{Error, Result};
use crate::config::storage::StorageConfig;
use crate::utils::resource_coordinator::environment_manager::{
    EnvironmentHandle, EnvironmentConfig, close_environment, get_or_create_environment
};
use crate::storage::metrics::StorageMetrics;
use super::LMDBStorage;
use super::config::{
    create_env_options,
    CF_MAIN, 
    CF_METADATA,
    CF_NGRAM_METADATA,
    calculate_batch_size,
    calculate_map_size_from_file_size
};

/// Struct to hold all database handles for atomic creation
struct DatabaseHandles {
    sequences_db: Database,
    text_bytes_db: Database,
    char_frequencies_db: Database,
    position_db: Database,
    metadata_db: Database,
    ngram_metadata_db: Database,
    prefix_index_db: Database,
    prefix_bytes_db: Database,
    percentile_db: Database,
}

impl LMDBStorage {
    /// Creates new database for writing with expansion (generate_ngrams.rs)
    pub fn create_for_writing<P: AsRef<Path>>(
        path: P,
        config: StorageConfig,
        source_file_size: u64,
        prefix_index_depth: usize,
    ) -> Result<LMDBStorage> {
        let path_buf = path.as_ref().to_path_buf();
        
        info!("Creating LMDB storage for writing at {:?} from {} MB source file", 
            path_buf, source_file_size / (1024 * 1024));
        
        // Step 1: Remove existing database if present
        if path_buf.exists() {
            info!("Removing existing database at {:?}", path_buf);
            fs::remove_dir_all(&path_buf)
                .map_err(|e| Error::storage(format!("Failed to remove existing database: {}", e)))?;
        }
        
        // Step 2: Create fresh directory
        fs::create_dir_all(&path_buf)
            .map_err(|e| Error::storage(format!("Failed to create database directory: {}", e)))?;
        
        // Step 3: Calculate map size with expansion for source file
        let map_size = calculate_map_size_from_file_size(source_file_size);
        info!("Calculated map size: {} MB for {} MB source file ({}x expansion)", 
            map_size / (1024 * 1024), 
            source_file_size / (1024 * 1024),
            (map_size as u64) / source_file_size.max(1));
        
        // Step 4: Calculate initial batch size
        let mem_info = sys_info::mem_info()
            .map_err(|e| Error::storage(format!("Failed to get memory info: {}", e)))?;
        
        let base_batch_size = calculate_batch_size(mem_info.total as usize * 1024);
        
        // Adjust batch size based on source file size
        let initial_batch_size = if source_file_size < 10 * 1024 * 1024 { // < 10MB
            let adjusted = base_batch_size * 2;
            debug!("Small source file, increasing batch size to {}", adjusted);
            adjusted
        } else if source_file_size > 100 * 1024 * 1024 { // > 100MB
            let adjusted = base_batch_size / 2;
            debug!("Large source file, reducing batch size to {}", adjusted);
            adjusted
        } else {
            debug!("Medium source file, using base batch size {}", base_batch_size);
            base_batch_size
        };
        
        // Step 5: Get environment options (no longer includes map_size)
        let (env_flags, max_readers, max_dbs) = create_env_options(&config);
        
        // Step 6: Build environment config with our calculated map_size
        let env_config = EnvironmentConfig {
            flags: env_flags | EnvironmentFlags::NO_SYNC | EnvironmentFlags::WRITE_MAP, // Write optimizations
            max_readers,
            max_dbs,
            map_size,  // Our calculated size, not from config!
        };
        
        info!("Creating LMDB environment with map_size={} MB, max_readers={}, max_dbs={}",
            map_size / (1024 * 1024), max_readers, max_dbs);
        
        // Step 7: Create or get environment (may force close cached one)
        close_environment(&path_buf).ok(); // Force close any cached environment
        
        let env_handle = get_or_create_environment(&path_buf, env_config)
            .map_err(|e| {
                let error_str = e.to_string();
                if error_str.contains("ENOMEM") {
                    Error::Database(format!(
                        "Insufficient memory to map {} MB. Consider reducing source file size.",
                        map_size / (1024 * 1024)
                    ))
                } else if error_str.contains("ENOSPC") {
                    Error::Database(format!(
                        "Insufficient disk space for {} MB database.",
                        map_size / (1024 * 1024)
                    ))
                } else {
                    Error::Database(format!("Failed to create environment: {}", e))
                }
            })?;
        
        // Step 8: Clean up stale readers
        if let Ok(stale_count) = env_handle.cleanup_stale_readers() {
            if stale_count > 0 {
                info!("Cleaned up {} stale readers", stale_count);
            }
        }
        
        // Step 9: Create all databases atomically
        let env_arc = env_handle.environment();
        let handles = Self::create_all_databases(env_arc)?;
        
        // Step 10: Build storage directly with our calculated values
        Self::build_storage_directly(
            path_buf,
            config,
            env_handle,
            handles,
            map_size,           // Pass our calculated size
            initial_batch_size, // Pass our calculated batch size
            prefix_index_depth,
        )
    }
    
    /// Helper: Directly build storage without re-reading map_size from environment
    fn build_storage_directly(
        path: PathBuf,
        config: StorageConfig,
        env_handle: EnvironmentHandle,
        handles: DatabaseHandles,
        map_size: usize,
        initial_batch_size: usize,
        prefix_index_depth: usize,
    ) -> Result<LMDBStorage> {
        trace!("Building storage instance with provided map_size: {} MB", 
            map_size / (1024 * 1024));
        
        // Create caches
        let cache_size = NonZeroUsize::new(10000)
            .ok_or_else(|| Error::Config("Invalid cache size".to_string()))?;
        
        let prefix_cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        let metadata_cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        // Create metrics
        let metrics = Arc::new(StorageMetrics::new());
        
        // Build storage struct directly with provided values
        let storage = LMDBStorage {
            env_handle,
            sequences_db: handles.sequences_db,
            text_bytes_db: handles.text_bytes_db,
            char_frequencies_db: handles.char_frequencies_db,
            position_db: handles.position_db,
            metadata_db: handles.metadata_db,
            ngram_metadata_db: handles.ngram_metadata_db,
            prefix_index_db: handles.prefix_index_db,
            prefix_bytes_db: handles.prefix_bytes_db,
            percentile_db: handles.percentile_db,
            metrics,
            is_bulk_loading: Arc::new(AtomicBool::new(config.is_bulk_loading)),
            current_memory_usage: Arc::new(AtomicUsize::new(0)),
            adaptive_batch_size: Arc::new(AtomicUsize::new(initial_batch_size)),
            prefix_cache,
            metadata_cache,
            db_path: path.clone(),
            map_size,
            prefix_index_depth,
        };
        
        info!("Created LMDB storage at {:?} with batch_size={}, map_size={} MB", 
            path, initial_batch_size, map_size / (1024 * 1024));
        
        Ok(storage)
    }

    pub fn open_for_reading<P: AsRef<Path>>(
    path: P,
    config: StorageConfig,
) -> Result<LMDBStorage> {
    let path_buf = path.as_ref().to_path_buf();
    let data_mdb_path = path_buf.join("data.mdb");
    
    info!("Opening LMDB storage for reading at {:?}", path_buf);
    
    // Step 1: Verify database exists
    if !data_mdb_path.exists() {
        return Err(Error::Database(format!(
            "Database file not found: {:?}. Cannot open for reading.",
            data_mdb_path
        )));
    }
    
    // Step 1.5: Try to read metadata from metadata.db in parent directory
    // BUT ONLY if we're not already opening a metadata.db (to prevent infinite recursion)
    let mut prefix_index_depth = 2; // Default fallback
    
    let is_metadata_db = path_buf.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name == "metadata.db")
        .unwrap_or(false);
    
    if !is_metadata_db {
        let parent_dir = path_buf.parent().unwrap_or(&path_buf);
        let metadata_db_path = parent_dir.join("metadata.db");
        
        if metadata_db_path.exists() {
            info!("Found metadata.db at {:?}, attempting to read prefix_index_depth", metadata_db_path);
            
            // Try to open the metadata database temporarily
            // This will now safely skip the metadata lookup since it IS a metadata.db
            match LMDBStorage::open_for_reading(&metadata_db_path, config.clone()) {
                Ok(mut metadata_storage) => {
                    // Try to read the metadata summary
                    match metadata_storage.get_metadata_summary() {
                        Ok(summary) => {
                            prefix_index_depth = summary.prefix_index_depth;
                            info!("Successfully read prefix_index_depth={} from metadata.db", prefix_index_depth);
                        },
                        Err(e) => {
                            warn!("Failed to read metadata summary from metadata.db: {}", e);
                        }
                    }
                    // Close the metadata database
                    let _ = metadata_storage.close();
                },
                Err(e) => {
                    warn!("Failed to open metadata.db: {}", e);
                }
            }
        } else {
            info!("No metadata.db found in parent directory, using default prefix_index_depth={}", prefix_index_depth);
        }
    } else {
        debug!("Opening metadata.db itself, skipping metadata lookup to prevent recursion");
    }
    
    // Step 2: Get exact file size - no expansion!
    let file_metadata = std::fs::metadata(&data_mdb_path)
        .map_err(|e| Error::storage(format!("Failed to read database file metadata: {}", e)))?;
    
    let map_size = file_metadata.len() as usize;
    info!("Opening existing database with size: {} MB", map_size / (1024 * 1024));
    
    // Step 3: Calculate batch size for read operations
    let mem_info = sys_info::mem_info()
        .map_err(|e| Error::storage(format!("Failed to get memory info: {}", e)))?;
    
    let initial_batch_size = calculate_batch_size(mem_info.total as usize * 1024);
    debug!("Using batch size {} for read operations", initial_batch_size);
    
    // Step 4: Get environment options (no longer includes map_size)
    let (env_flags, max_readers, max_dbs) = create_env_options(&config);
    
    // Step 5: Build environment config with exact map_size
    let env_config = EnvironmentConfig {
        flags: env_flags | EnvironmentFlags::NO_SYNC, // Read optimizations
        max_readers,
        max_dbs,
        map_size,  // Exact size from file, no expansion!
    };
    
    info!("Opening LMDB environment with map_size={} MB (exact), max_readers={}, max_dbs={}",
        map_size / (1024 * 1024), max_readers, max_dbs);
    
    // Step 6: Force close any cached environment with wrong size
    close_environment(&path_buf).ok();
    
    // Step 7: Get or create environment with correct size
    let env_handle = get_or_create_environment(&path_buf, env_config)
        .map_err(|e| Error::Database(format!("Failed to open environment for reading: {}", e)))?;
    
    // Step 8: Clean up stale readers if any
    if let Ok(stale_count) = env_handle.cleanup_stale_readers() {
        if stale_count > 0 {
            info!("Cleaned up {} stale readers", stale_count);
        }
    }
    
    // Step 9: Open existing databases (NOT create)
    let env_arc = env_handle.environment();
    let handles = Self::open_existing_databases(env_arc)?;
    
    // Step 10: Build storage with exact values and the prefix_index_depth we found
    Self::build_storage_directly(
        path_buf,
        config,
        env_handle,
        handles,
        map_size,           // Exact size from file
        initial_batch_size, // For read batching
        prefix_index_depth, // Pass the depth we found from metadata.db
    )
}
    
    /// Helper: Opens existing databases without creating them
    fn open_existing_databases(env: &Arc<Environment>) -> Result<DatabaseHandles> {
        trace!("Opening existing databases for reading");
        
        // Use write transaction to get persistent handles
        let txn = env.begin_rw_txn()
            .map_err(|e| Error::Database(format!("Failed to begin transaction for opening databases: {}", e)))?;
        
        // Use create_db which opens existing databases (doesn't actually create if they exist)
            let sequences_db = unsafe {
            txn.create_db(Some(CF_MAIN), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open main database: {}", e)))?
        };

        let text_bytes_db = unsafe {
            txn.create_db(Some("text_bytes"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open text_bytes database: {}", e)))?
        };
                
        let char_frequencies_db = unsafe {
            txn.create_db(Some("char_frequencies"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open char_frequencies database: {}", e)))?
        };
        
        let position_db = unsafe {
            txn.create_db(Some("position"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open position database: {}", e)))?
        };
        
        let metadata_db = unsafe {
            txn.create_db(Some(CF_METADATA), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open metadata database: {}", e)))?
        };
        
        let ngram_metadata_db = unsafe {
            txn.create_db(Some(CF_NGRAM_METADATA), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open ngram_metadata database: {}", e)))?
        };
        
        // Open prefix system databases
        let prefix_index_db = unsafe {
            txn.create_db(Some("prefix_index"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open prefix_index database: {}", e)))?
        };

        let prefix_bytes_db = unsafe {
            txn.create_db(Some("prefix_bytes"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open prefix_bytes database: {}", e)))?
        };

        let percentile_db = unsafe {
            txn.create_db(Some("percentiles"), DatabaseFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to open percentiles database: {}", e)))?
        };
        
        // Commit to make handles persistent
        txn.commit()
            .map_err(|e| Error::Database(format!("Failed to commit database handles: {}", e)))?;
        
        info!("Successfully committed creation of 10 databases (8 column + 2 prefix)");
        
        Ok(DatabaseHandles {
            sequences_db,
            text_bytes_db,
            char_frequencies_db,
            position_db,
            metadata_db,
            ngram_metadata_db,
            prefix_index_db,
            prefix_bytes_db,
            percentile_db,
        })
    }

    fn create_all_databases(env: &Arc<Environment>) -> Result<DatabaseHandles> {
        info!("Atomically creating all LMDB databases");
        
        // Retry logic for concurrent initialization
        const MAX_ATTEMPTS: u32 = 3;
        let mut attempts = 0;
        
        loop {
            attempts += 1;
            
            // Try to begin write transaction
            match env.begin_rw_txn() {
                Ok(txn) => {
                    // Successfully got write lock, proceed with creation
                    // Pass ownership of transaction to helper method
                    return Self::create_databases_with_txn(txn);
                },
                Err(e) => {
                    // Check if it's a concurrent access issue
                    if let LmdbError::Other(code) = &e {
                        if *code == libc::EAGAIN && attempts < MAX_ATTEMPTS {
                            warn!("Database initialization in progress by another process, \
                                   attempt {}/{}", attempts, MAX_ATTEMPTS);
                            std::thread::sleep(Duration::from_millis(100 * attempts as u64));
                            continue;
                        }
                    }
                    
                    // Not a retry-able error or max attempts reached
                    return Err(Error::Database(format!(
                        "Failed to begin database creation transaction after {} attempts: {}", 
                        attempts, e
                    )));
                }
            }
        }
    }
    
    /// Helper method to create databases within a transaction
    /// Separated for cleaner error handling and retry logic
    /// Takes ownership of transaction since commit consumes it
    fn create_databases_with_txn(mut txn: RwTransaction) -> Result<DatabaseHandles> {
        // Create main databases using the helper that accepts a transaction
        debug!("Creating main databases");
        let sequences_db = Self::create_database(&mut txn, CF_MAIN, DatabaseFlags::empty())?;
        let text_bytes_db = Self::create_database(&mut txn, "text_bytes", DatabaseFlags::empty())?;
        let char_frequencies_db = Self::create_database(&mut txn, "char_frequencies", DatabaseFlags::empty())?;
        let position_db = Self::create_database(&mut txn, "position", DatabaseFlags::empty())?;
        
        let metadata_db = Self::create_database(&mut txn, CF_METADATA, DatabaseFlags::empty())?;
        let ngram_metadata_db = Self::create_database(&mut txn, CF_NGRAM_METADATA, DatabaseFlags::empty())?;
        
        // Create prefix system databases
        trace!("Creating prefix index and prefix bytes databases");
        let prefix_index_db = Self::create_database(&mut txn, "prefix_index", DatabaseFlags::empty())?;
        let prefix_bytes_db = Self::create_database(&mut txn, "prefix_bytes", DatabaseFlags::empty())?;

        // Create common ngram percentage database
        let percentile_db = Self::create_database(&mut txn, "percentiles", DatabaseFlags::empty())?;        
        
        // Commit with enhanced error handling
        match txn.commit() {
            Ok(()) => {
                info!("Successfully committed creation of 9 databases (7 column + 2 prefix)");
            },
            Err(e) => {
                error!("Failed to commit database creation: {}", e);
                
                // Provide specific guidance based on error type
                let error_msg = if let LmdbError::Other(code) = &e {
                    match *code {
                        libc::ENOSPC => {
                            "No disk space available to commit database creation. \
                             Free up disk space and retry."
                        },
                        libc::EIO => {
                            "I/O error during database commit. Check disk health \
                             and filesystem integrity."
                        },
                        libc::ENOMEM => {
                            "Out of memory during commit. The system may be under \
                             severe memory pressure."
                        },
                        _ => "Failed to commit database creation transaction"
                    }
                } else {
                    "Failed to commit database creation transaction"
                };
                
                return Err(Error::Database(format!("{}: {}", error_msg, e)));
            }
        }
        
        Ok(DatabaseHandles {
            sequences_db,
            text_bytes_db,
            char_frequencies_db,
            position_db,
            metadata_db,
            ngram_metadata_db,
            prefix_index_db,
            prefix_bytes_db,
            percentile_db,
        })
    }
    
   
    
    /// ✅ STEP 2.1: Updated helper method to accept transaction parameter
    /// This allows atomic creation of multiple databases in a single transaction
    fn create_database(txn: &mut RwTransaction, name: &str, flags: DatabaseFlags) -> Result<Database> {
        trace!("Creating database: {}", name);
        
        // ✅ Use the passed-in transaction instead of creating a new one
        let database = unsafe {
            match txn.create_db(Some(name), flags) {
                Ok(db) => {
                    trace!("Successfully created database: {}", name);
                    db
                },
                Err(LmdbError::NotFound) => {
                    // Database doesn't exist yet, try again
                    debug!("Database {} doesn't exist yet, creating with CREATE flag", name);
                    txn.create_db(Some(name), flags)
                        .map_err(|e| Error::Database(format!("Failed to create database {}: {}", name, e)))?
                },
                Err(LmdbError::DbsFull) => {
                    error!("Maximum number of databases reached. Increase max_dbs in configuration.");
                    return Err(Error::Database(format!("LMDB max databases reached for {}", name)));
                },
                Err(e) => {
                    error!("Failed to create database {}: {}", name, e);
                    return Err(Error::Database(format!("Failed to create database {}: {}", name, e)));
                }
            }
        };
        
        // ✅ NO COMMIT HERE - the caller will commit when all databases are created
        Ok(database)
    }

    /// Verify the LMDB environment is usable with enhanced diagnostics
    pub fn verify_environment(&self) -> Result<()> {
        debug!("Verifying LMDB environment at {:?}", self.db_path);
        
        // Get environment through the handle
        let env = self.env_handle.environment();
        
        // Check we can read environment info
        let info = env.info()
            .map_err(|e| {
                error!("Failed to get environment info: {}", e);
                Error::Database(format!(
                    "Failed to get environment info. The database may be corrupted \
                     or inaccessible: {}", e
                ))
            })?;
        
        // Check we can get stats
        let stat = env.stat()
            .map_err(|e| {
                error!("Failed to get environment statistics: {}", e);
                Error::Database(format!(
                    "Failed to get environment statistics. The database may be corrupted: {}", e
                ))
            })?;
        
        debug!("Environment info: map_size={} MB, max_readers={}, num_readers={}", 
               info.map_size() / (1024 * 1024), info.max_readers(), info.num_readers());
        
        debug!("Environment stat: page_size={}, depth={}, entries={}", 
               stat.page_size(), stat.depth(), stat.entries());
        
        // Warn if reader usage is high
        let reader_usage = (info.num_readers() as f64 / info.max_readers() as f64) * 100.0;
        if reader_usage > 80.0 {
            warn!("High reader usage: {:.1}% ({}/{}). Consider increasing max_readers.",
                  reader_usage, info.num_readers(), info.max_readers());
        }
        
        // Verify we can read from the main database
        let txn = env.begin_ro_txn()
            .map_err(|e| {
                error!("Failed to create verification transaction: {}", e);
                Error::Database(format!(
                    "Failed to create read transaction for verification. \
                     The environment may be locked or corrupted: {}", e
                ))
            })?;
        
        // Just try to get stats on the main database
        let db_stat = txn.stat(self.sequences_db)
            .map_err(|e| {
                error!("Failed to read main database stats: {}", e);
                Error::Database(format!(
                    "Failed to read main database statistics. \
                     The database may be corrupted and need reinitialization: {}", e
                ))
            })?;
        
        debug!("Main database stats: {} entries, {} pages", 
               db_stat.entries(), db_stat.leaf_pages() + db_stat.branch_pages());
        
        // Drop transaction explicitly for clarity
        drop(txn);
        
        debug!("Environment verification successful");
        Ok(())
    }

    /// Creates a new database with exact size for compaction
    /// This bypasses the automatic size calculation and uses the provided exact size
    pub fn create_for_compaction<P: AsRef<Path>>(
        path: P,
        config: StorageConfig,
        exact_map_size: usize,
        prefix_index_depth: usize,
    ) -> Result<LMDBStorage> {
        let path_buf = path.as_ref().to_path_buf();
        
        info!("Creating LMDB storage for compaction at {:?} with exact size {} MB", 
            path_buf, exact_map_size / (1024 * 1024));
        
        // Step 1: Remove existing database if present
        if path_buf.exists() {
            info!("Removing existing database at {:?}", path_buf);
            fs::remove_dir_all(&path_buf)
                .map_err(|e| Error::storage(format!("Failed to remove existing database: {}", e)))?;
        }
        
        // Step 2: Create fresh directory
        fs::create_dir_all(&path_buf)
            .map_err(|e| Error::storage(format!("Failed to create database directory: {}", e)))?;
        
        // Step 3: Use the exact map size provided (no calculation/expansion)
        let map_size = exact_map_size;
        info!("Using exact map size: {} MB", map_size / (1024 * 1024));
        
        // Step 4: Calculate batch size for write operations
        let mem_info = sys_info::mem_info()
            .map_err(|e| Error::storage(format!("Failed to get memory info: {}", e)))?;
        
        let initial_batch_size = calculate_batch_size(mem_info.total as usize * 1024);
        
        // Step 5: Get environment options
        let (env_flags, max_readers, max_dbs) = create_env_options(&config);
        
        // Step 6: Build environment config with exact map_size
        let env_config = EnvironmentConfig {
            flags: env_flags | EnvironmentFlags::NO_SYNC | EnvironmentFlags::WRITE_MAP, // Write optimizations
            max_readers,
            max_dbs,
            map_size,  // Use exact size, no expansion
        };
        
        info!("Creating compaction LMDB environment with map_size={} MB, max_readers={}, max_dbs={}",
            map_size / (1024 * 1024), max_readers, max_dbs);
        
        // Step 7: Create environment
        close_environment(&path_buf).ok(); // Force close any cached environment
        
        let env_handle = get_or_create_environment(&path_buf, env_config)
            .map_err(|e| {
                let error_str = e.to_string();
                if error_str.contains("ENOMEM") {
                    Error::Database(format!(
                        "Insufficient memory to map {} MB for compaction.",
                        map_size / (1024 * 1024)
                    ))
                } else if error_str.contains("ENOSPC") {
                    Error::Database(format!(
                        "Insufficient disk space for {} MB compaction database.",
                        map_size / (1024 * 1024)
                    ))
                } else {
                    Error::Database(format!("Failed to create compaction environment: {}", e))
                }
            })?;
        
        info!("Successfully created compaction LMDB environment");
        
        // Step 8: Create all databases atomically (same as create_for_writing)
        let env_arc = env_handle.environment();
        let handles = Self::create_all_databases(env_arc)?;
        
        // Step 9: Build storage with exact values
        let mut storage = Self::build_storage_directly(
            path_buf.clone(),
            config.clone(),
            env_handle,
            handles,
            map_size,
            initial_batch_size,
            prefix_index_depth,
        )?;
        
        // Step 10: Store initial metadata about this being a compaction database
        // Get metadata_db handle before the closure to avoid borrow issues
        let metadata_db = storage.metadata_db;
        storage.with_write_scope(|mut scope| {
            // Store database creation metadata
            let metadata = vec![
                ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
                ("created_for".to_string(), "compaction".to_string()),
                ("exact_map_size".to_string(), map_size.to_string()),
                ("prefix_index_depth".to_string(), prefix_index_depth.to_string()),
                ("original_path".to_string(), path_buf.to_string_lossy().to_string()),
            ];
            
            for (key, value) in metadata {
                scope.txn()
                    .put(
                        metadata_db,
                        &key.as_bytes(),
                        &value.as_bytes(),
                        lmdb_rkv::WriteFlags::empty(),
                    )
                    .map_err(|e| Error::Database(format!("Failed to write compaction metadata: {}", e)))?;
            }
            
            scope.commit()
        })?;
        
        info!("Successfully initialized compaction LMDB storage at {:?}", path_buf);
        
        Ok(storage)
    }

}