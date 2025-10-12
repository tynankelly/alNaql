// storage/mod.rs - Unified LMDB storage implementation (duplicates removed)

//==============================================================================
// MODULE DECLARATIONS
//==============================================================================
pub mod config;
pub mod init;
pub mod batch;
pub mod prefix;
pub mod pool;
pub mod query;
pub mod maintenance;
pub mod metadata;
pub mod cache;
pub mod scopes;
pub mod metrics;

pub use self::maintenance::{DatabaseSizeStats, PageStats};
pub use pool::{DatabasePool, PoolStats, PooledDatabase, create_pool_from_config};

//==============================================================================
// IMPORTS
//==============================================================================

// Standard imports
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use log::{info, warn, debug};
use lru::LruCache;
use parking_lot::RwLock;
use lmdb_rkv::{Transaction, Database, Cursor};
use memmap2::Mmap;

// Crate imports
use crate::error::{Error, Result};
use crate::config::storage::StorageConfig;
use crate::types::NGramMetadata;

// Resource coordinator imports
use crate::utils::resource_coordinator::environment_manager::{
    EnvironmentHandle,
    close_environment,
    check_and_perform_maintenance,
};

// Local submodule imports
use self::metrics::StorageMetrics;
use self::scopes::{ReadScope, WriteScope};

//==============================================================================
// TYPE DEFINITIONS
//==============================================================================

pub type SequenceID = u64;


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileMetadata {
    pub file_path: String,
    pub file_size: u64,
    pub total_ngrams: u64,
    pub processed_at: std::time::SystemTime,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetadataSummary {
    pub total_files: u64,
    pub total_ngrams: u64,
    pub last_updated: std::time::SystemTime,
    pub prefix_index_depth: usize,
}

impl Default for MetadataSummary {
    fn default() -> Self {
        Self {
            total_files: 0,
            total_ngrams: 0,
            last_updated: std::time::SystemTime::now(),
            prefix_index_depth: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_ngrams: u64,
    pub total_bytes: u64,
    pub unique_sequences: u64,
    pub unique_cleaned_texts: u64,
    pub compression_ratio: f64,
}

//==============================================================================
// LMDBSTORAGE STRUCT
//==============================================================================

pub struct LMDBStorage {
    pub(crate) env_handle: EnvironmentHandle,
    pub(crate) sequences_db: Database,
    pub(crate) metadata_db: Database,
    pub(crate) ngram_metadata_db: Database,
    pub(crate) prefix_index_db: Database,
    pub(crate) prefix_bytes_db: Database,
    pub(crate) prefix_index_depth: usize,
    pub(crate) metrics: Arc<StorageMetrics>,
    pub(crate) is_bulk_loading: Arc<AtomicBool>,
    pub(crate) current_memory_usage: Arc<AtomicUsize>,
    pub(crate) adaptive_batch_size: Arc<AtomicUsize>,
    pub(crate) prefix_cache: Arc<RwLock<LruCache<Vec<u8>, Vec<u64>>>>,
    pub(crate) metadata_cache: Arc<RwLock<LruCache<u64, NGramMetadata>>>,
    pub(crate) db_path: PathBuf,
    pub(crate) map_size: usize,
    pub(crate) text_bytes_db: Database,
    pub(crate) char_frequencies_db: Database,
    pub(crate) position_db: Database,
    pub(crate) percentile_db: Database,
}

impl std::fmt::Debug for LMDBStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LMDBStorage")
            .field("db_path", &self.db_path.display())
            .field("is_bulk_loading", &self.is_bulk_loading.load(Ordering::Relaxed))
            .field("adaptive_batch_size", &self.adaptive_batch_size.load(Ordering::Relaxed))
            .field("current_memory_usage", &self.current_memory_usage.load(Ordering::Relaxed))
            .field("metadata_cache_size", &self.metadata_cache.read().len())
            .field("has_prefix_index", &true)
            .field("has_prefix_column", &true)
            .finish()
    }
}

// Explicit Send + Sync markers
unsafe impl Send for LMDBStorage {}
unsafe impl Sync for LMDBStorage {}

//==============================================================================
// SCOPE METHODS - TRANSACTION MANAGEMENT (UNIQUE TO MOD.RS)
//==============================================================================

impl LMDBStorage {
    /// Execute operation with read-only transaction scope
    pub fn with_read_scope<F, R>(&self, f: F) -> Result<R>
    where 
        F: FnOnce(ReadScope<'_>) -> Result<R>
    {
        // First attempt
        match self.env_handle.environment().begin_ro_txn() {
            Ok(txn) => {
                let scope = ReadScope::new(txn, self);
                f(scope)
            },
            Err(e) => {
                let error_str = e.to_string();
                
                // Check if it's a readers full error
                if error_str.contains("MDB_READERS_FULL") || 
                   error_str.contains("readers full") ||
                   error_str.contains("No more readers") {
                    // Try to clean up stale readers
                    warn!("Reader slots full, attempting cleanup");
                    
                    // Cleanup the stale readers
                    match self.env_handle.cleanup_stale_readers() {
                        Ok(cleaned) => {
                            info!("Cleaned {} stale readers, retrying transaction", cleaned);
                        },
                        Err(e) => {
                            warn!("Failed to clean stale readers: {}", e);
                            // Continue anyway - maybe the cleanup partially worked
                        }
                    }
                    
                    // Retry once after cleanup attempt
                    let txn = self.env_handle.environment().begin_ro_txn()
                        .map_err(|e| Error::Database(format!(
                            "No reader slots available even after cleanup: {}", e
                        )))?;
                    
                    let scope = ReadScope::new(txn, self);
                    f(scope)
                } else {
                    // For other errors, just propagate them
                    Err(Error::Database(format!("Failed to begin read transaction: {}", e)))
                }
            }
        }
    }
    
    /// Execute operation with read-write transaction scope
    pub fn with_write_scope<F, R>(&mut self, f: F) -> Result<R>
    where 
        F: FnOnce(WriteScope<'_>) -> Result<R>
    {
        match self.env_handle.environment().begin_rw_txn() {
            Ok(txn) => {
                let scope = WriteScope::new(txn, self);
                f(scope)
            },
            Err(e) => {
                let error_str = e.to_string();
                
                // Write transactions can also fail due to reader slots!
                if error_str.contains("MDB_READERS_FULL") || 
                   error_str.contains("readers full") ||
                   error_str.contains("No more readers") {
                    warn!("Reader slots full on write transaction, attempting cleanup");
                    
                    match self.env_handle.cleanup_stale_readers() {
                        Ok(cleaned) => {
                            info!("Cleaned {} stale readers, retrying write transaction", cleaned);
                        },
                        Err(e) => {
                            warn!("Failed to clean stale readers: {}", e);
                        }
                    }
                    
                    // Retry once
                    let txn = self.env_handle.environment().begin_rw_txn()
                        .map_err(|e| Error::Database(format!(
                            "No reader slots available for write even after cleanup: {}", e
                        )))?;
                    
                    let scope = WriteScope::new(txn, self);
                    f(scope)
                } else {
                    Err(Error::Database(format!("Failed to begin write transaction: {}", e)))
                }
            }
        }
    }
}

//==============================================================================
// COORDINATION HELPER METHODS (UNIQUE TO MOD.RS)
//==============================================================================

impl LMDBStorage {

    /// Check if bulk loading is enabled
    pub fn is_bulk_loading(&self) -> bool {
        self.is_bulk_loading.load(Ordering::Relaxed)
    }
    
    /// Set bulk loading mode
    pub fn set_bulk_loading(&self, enabled: bool) {
        self.is_bulk_loading.store(enabled, Ordering::Relaxed);
        debug!("Bulk loading mode set to: {}", enabled);
    }
    
    /// Get current memory usage
    pub fn get_current_memory_usage(&self) -> usize {
        self.current_memory_usage.load(Ordering::Relaxed)
    }
    
    /// Get adaptive batch size
    pub fn get_adaptive_batch_size(&self) -> usize {
        self.adaptive_batch_size.load(Ordering::Relaxed)
    }
    
    /// Update adaptive batch size
    pub fn update_adaptive_batch_size(&self, new_size: usize) {
        self.adaptive_batch_size.store(new_size, Ordering::Relaxed);
        debug!("Updated adaptive batch size to: {}", new_size);
    }
    
    /// Update memory usage tracking
    pub fn update_memory_usage(&self, bytes_delta: isize) {
        if bytes_delta > 0 {
            self.current_memory_usage.fetch_add(bytes_delta as usize, Ordering::Relaxed);
        } else {
            self.current_memory_usage.fetch_sub((-bytes_delta) as usize, Ordering::Relaxed);
        }
    }
    
    /// Reset memory usage counter
    pub fn reset_memory_usage(&self) {
        self.current_memory_usage.store(0, Ordering::Relaxed);
    }

    /// Store percentile data for ngrams
    pub fn store_percentiles(&mut self, percentiles: ahash::AHashMap<u64, u8>) -> Result<()> {
        self.with_write_scope(|mut scope| {  // Add mut here
            let percentile_db = scope.storage().percentile_db;
            
            for (hash, percentile) in percentiles {
                let key = hash.to_be_bytes();
                let value = [percentile];  // Single byte value
                
                scope.txn().put(
                    percentile_db,
                    &key,
                    &value,
                    lmdb_rkv::WriteFlags::empty()
                ).map_err(|e| Error::Database(format!("Failed to store percentile: {}", e)))?;
            }
            
            scope.commit()
        })
    }

/// Get percentile for a single ngram hash
pub fn get_percentile(&self, hash: u64) -> Result<Option<u8>> {
    use lmdb_rkv::Transaction;  // Import the trait
    
    self.with_read_scope(|scope| {
        let percentile_db = scope.storage().percentile_db;
        let key = hash.to_be_bytes();
        
        match scope.txn().get(percentile_db, &key) {
            Ok(value) => Ok(Some(value[0])),
            Err(lmdb_rkv::Error::NotFound) => Ok(None),
            Err(e) => Err(Error::Database(format!("Failed to get percentile: {}", e)))
        }
    })
}

/// Batch get percentiles for multiple hashes
pub fn get_percentiles_batch(&self, hashes: &[u64]) -> Result<ahash::AHashMap<u64, u8>> {
    use lmdb_rkv::Transaction;  // Import the trait
    
    self.with_read_scope(|scope| {
        let percentile_db = scope.storage().percentile_db;
        let mut results = ahash::AHashMap::new();
        
        for &hash in hashes {
            let key = hash.to_be_bytes();
            if let Ok(value) = scope.txn().get(percentile_db, &key) {
                results.insert(hash, value[0]);
            }
        }
        
        Ok(results)
    })
}
}

//==============================================================================
// CACHE ACCESS HELPERS (UNIQUE TO MOD.RS)
//==============================================================================

impl LMDBStorage {
    /// Get prefix cache statistics
    pub fn get_prefix_cache_stats(&self) -> (usize, usize) {
        let cache = self.prefix_cache.read();
        (cache.len(), cache.cap().get())
    }
    
    /// Get metadata cache statistics
    pub fn get_metadata_cache_stats(&self) -> (usize, usize) {
        let cache = self.metadata_cache.read();
        (cache.len(), cache.cap().get())
    }
    
    /// Clear prefix cache
    pub fn clear_prefix_cache(&self) {
        let cleared_count = self.prefix_cache.read().len();
        self.prefix_cache.write().clear();
        debug!("Cleared {} entries from prefix cache", cleared_count);
    }
    
    /// Clear metadata cache
    pub fn clear_metadata_cache(&self) {
        let cleared_count = self.metadata_cache.read().len();
        self.metadata_cache.write().clear();
        debug!("Cleared {} entries from metadata cache", cleared_count);
    }
    
    /// Clear all caches
    pub fn clear_all_caches(&self) {
        self.clear_prefix_cache();
        self.clear_metadata_cache();
    }
}

/// Handle for memory-mapped database access
pub struct MmapHandle {
    pub mmap: Mmap,
    pub len: usize,
    pub entry_size: usize,
}

impl MmapHandle {
    pub fn get_entry(&self, index: usize) -> Option<&[u8]> {
        let offset = index * self.entry_size;
        if offset + self.entry_size <= self.len {
            Some(&self.mmap[offset..offset + self.entry_size])
        } else {
            None
        }
    }
}

//==============================================================================
// STORAGE STATE INFO (UNIQUE TO MOD.RS)
//==============================================================================

#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub db_path: PathBuf,
    pub is_bulk_loading: bool,
    pub current_memory_usage: usize,
    pub adaptive_batch_size: usize,
    pub has_prefix_index: bool,
    pub has_prefix_column: bool,
    pub prefix_cache_stats: (usize, usize),
    pub metadata_cache_stats: (usize, usize),
}

impl LMDBStorage {
    /// Get comprehensive storage information
    pub fn get_storage_info(&self) -> StorageInfo {
        StorageInfo {
            db_path: self.db_path.clone(),
            is_bulk_loading: self.is_bulk_loading(),
            current_memory_usage: self.get_current_memory_usage(),
            adaptive_batch_size: self.get_adaptive_batch_size(),
            has_prefix_index: true,
            has_prefix_column: true,
            prefix_cache_stats: self.get_prefix_cache_stats(),
            metadata_cache_stats: self.get_metadata_cache_stats(),
        }
    }
}

//==============================================================================
// UNIQUE PUBLIC METHODS (NOT FOUND IN SUBMODULES)
//==============================================================================

impl LMDBStorage {
    /// Perform database compaction (LMDB doesn't support this - unique to mod.rs)
    pub fn compact(&mut self) -> Result<()> {
        info!("LMDB compaction not supported - performing sync instead");
        self.env_handle.environment().sync(true)
            .map_err(|e| Error::Database(format!("Failed to sync environment: {}", e)))
    }

    /// Close the storage backend (unique implementation in mod.rs)
    pub fn close(&mut self) -> Result<()> {
        info!("Closing LMDB storage at {:?}", self.db_path);
        
        // Clear all caches
        {
            let prefix_cache_size = self.prefix_cache.read().len();
            self.prefix_cache.write().clear();
            
            let metadata_cache_size = self.metadata_cache.read().len();
            self.metadata_cache.write().clear();
            
            debug!("Cleared {} prefix entries and {} metadata entries from caches", 
                prefix_cache_size, metadata_cache_size);
        }
        
        // Reset atomic counters
        self.reset_memory_usage();
        self.set_bulk_loading(false);
        
        // Final sync to disk
        self.env_handle.environment().sync(true)
            .map_err(|e| Error::Database(format!("Failed to sync during close: {}", e)))?;
        
        info!("Successfully closed LMDB storage");
        Ok(())
    }

    /// Copy all data from this storage to another storage instance
    /// Used for database compaction after generation
    pub fn copy_all_data_to(&self, target: &mut LMDBStorage) -> Result<()> {
        use std::time::Instant;
        let start_time = Instant::now();
        
        info!("Starting database copy for compaction...");
        
        // Track total records copied
        let mut total_records = 0u64;
        let batch_size = 10000; // Configurable batch size
        
        // Copy each database in order of importance
        // Metadata first, then core data, then indexes
        
        // 1. Copy metadata_db
        info!("Copying metadata database...");
        let copied = self.copy_database_contents(
            self.metadata_db,
            target.metadata_db,
            target,
            batch_size,
            "metadata"
        )?;
        total_records += copied;
        
        // 2. Copy ngram_metadata_db
        info!("Copying ngram metadata database...");
        let copied = self.copy_database_contents(
            self.ngram_metadata_db,
            target.ngram_metadata_db,
            target,
            batch_size,
            "ngram_metadata"
        )?;
        total_records += copied;
        
        // 3. Copy sequences_db (registry of all sequences)
        info!("Copying sequences database...");
        let copied = self.copy_database_contents(
            self.sequences_db,
            target.sequences_db,
            target,
            batch_size,
            "sequences"
        )?;
        total_records += copied;
        
        // 4. Copy text_bytes_db
        info!("Copying text bytes database...");
        let copied = self.copy_database_contents(
            self.text_bytes_db,
            target.text_bytes_db,
            target,
            batch_size,
            "text_bytes"
        )?;
        total_records += copied;
                
        // 5. Copy char_frequencies_db
        info!("Copying character frequencies database...");
        let copied = self.copy_database_contents(
            self.char_frequencies_db,
            target.char_frequencies_db,
            target,
            batch_size,
            "char_frequencies"
        )?;
        total_records += copied;
        
        // 6. Copy position_db
        info!("Copying position database...");
        let copied = self.copy_database_contents(
            self.position_db,
            target.position_db,
            target,
            batch_size,
            "position"
        )?;
        total_records += copied;
        
        // 7. Copy prefix_index_db
        info!("Copying prefix index database...");
        let copied = self.copy_database_contents(
            self.prefix_index_db,
            target.prefix_index_db,
            target,
            batch_size,
            "prefix_index"
        )?;
        total_records += copied;
        
        // 8. Copy prefix_bytes_db
        info!("Copying prefix bytes database...");
        let copied = self.copy_database_contents(
            self.prefix_bytes_db,
            target.prefix_bytes_db,
            target,
            batch_size,
            "prefix_bytes"
        )?;
        total_records += copied;
        
        // 9. Copy percentile_db
        info!("Copying percentile database...");
        let copied = self.copy_database_contents(
            self.percentile_db,
            target.percentile_db,
            target,
            batch_size,
            "percentile"
        )?;
        total_records += copied;
        
        let elapsed = start_time.elapsed();
        info!("Database copy completed in {:?}", elapsed);
        info!("Total records copied: {}", total_records);
        
        // Ensure all data is synced to disk in the target
        target.env_handle.environment().sync(true)
            .map_err(|e| Error::Database(format!("Failed to sync target database: {}", e)))?;
        
        Ok(())
    }
    
    /// Helper method to copy contents of a single database
    fn copy_database_contents(
        &self,
        source_db: Database,
        target_db: Database,
        target_storage: &mut LMDBStorage,
        batch_size: usize,
        db_name: &str,
    ) -> Result<u64> {
        let mut records_copied = 0u64;
        let mut batch = Vec::with_capacity(batch_size);
        
        // Read from source
        self.with_read_scope(|scope| {
            let mut cursor = scope.txn()
                .open_ro_cursor(source_db)
                .map_err(|e| Error::Database(format!("Failed to open cursor on {} database: {}", db_name, e)))?;
            
            // Iterate through all records
            for result in cursor.iter() {
                let (key, value) = result
                    .map_err(|e| Error::Database(format!("Failed to read from {} database: {}", db_name, e)))?;
                
                // Add to batch
                batch.push((key.to_vec(), value.to_vec()));
                
                // Write batch when it reaches the size limit
                if batch.len() >= batch_size {
                    let batch_to_write = std::mem::replace(&mut batch, Vec::with_capacity(batch_size));
                    records_copied += self.write_batch_to_target(
                        target_storage,
                        target_db,
                        batch_to_write,
                        db_name
                    )?;
                    
                    if records_copied % 100_000 == 0 {
                        debug!("Copied {} records from {} database...", records_copied, db_name);
                    }
                }
            }
            
            // Write remaining batch
            if !batch.is_empty() {
                records_copied += self.write_batch_to_target(
                    target_storage,
                    target_db,
                    batch,
                    db_name
                )?;
            }
            
            Ok(records_copied)
        })
    }
    
    /// Helper to write a batch of records to the target database
    fn write_batch_to_target(
        &self,
        target_storage: &mut LMDBStorage,
        target_db: Database,
        batch: Vec<(Vec<u8>, Vec<u8>)>,
        db_name: &str,
    ) -> Result<u64> {
        let batch_len = batch.len() as u64;
        
        target_storage.with_write_scope(|mut scope| {
            for (key, value) in batch {
                scope.txn()
                    .put(target_db, &key, &value, lmdb_rkv::WriteFlags::empty())
                    .map_err(|e| Error::Database(format!(
                        "Failed to write to {} database: {}", db_name, e
                    )))?;
            }
            scope.commit()?;
            Ok(())
        })?;
        
        Ok(batch_len)
    }
}

//==============================================================================
// FACTORY FUNCTIONS
//==============================================================================

/// Create a new storage instance
/// Opens existing database for reading (used by find_matches)
pub fn open_storage<P: AsRef<Path>>(
    path: P,
    config: StorageConfig,
) -> Result<LMDBStorage> {
    LMDBStorage::open_for_reading(path, config)
}

/// Creates new database with expansion (used by generate_ngrams)
pub fn create_storage_for_generation<P: AsRef<Path>>(
    path: P,
    config: StorageConfig,
    source_file_size: u64,
    prefix_index_depth: usize,  // ADD THIS PARAMETER
) -> Result<LMDBStorage> {
    LMDBStorage::create_for_writing(path, config, source_file_size, prefix_index_depth)
}

/// Creates a small database without expansion (for metadata storage)
pub fn create_metadata_storage<P: AsRef<Path>>(
    path: P,
    config: StorageConfig,
) -> Result<LMDBStorage> {
    // Use a reasonable fixed size for metadata databases - they're always small
    let metadata_db_size = 3 * 1024 * 1024; // 10MB should be plenty for metadata
    
    // Metadata databases don't use prefix indexing, just pass a default
    let prefix_index_depth = 2;  // Default, won't actually be used
    
    LMDBStorage::create_for_compaction(path, config, metadata_db_size, prefix_index_depth)
}

/// Creates a new database with exact size for compaction
/// This bypasses the file-size based calculation and uses the exact size provided
pub fn create_storage_for_compaction<P: AsRef<Path>>(
    path: P,
    config: StorageConfig,
    exact_map_size: usize,
    prefix_index_depth: usize,
) -> Result<LMDBStorage> {
    LMDBStorage::create_for_compaction(path, config, exact_map_size, prefix_index_depth)
}

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/// Verify if a storage backend is properly closed
pub fn verify_storage_closed<P: AsRef<Path>>(path: P) -> bool {
    match close_environment(path.as_ref()) {
        Ok(was_managed) => !was_managed,
        Err(_) => true,
    }
}

/// Check if a storage backend is closed
pub fn is_storage_closed<P: AsRef<Path>>(path: P) -> bool {
    verify_storage_closed(path)
}

/// Perform maintenance on storage systems
pub fn perform_storage_maintenance() -> Result<()> {
    check_and_perform_maintenance()?;
    Ok(())
}

//==============================================================================
// COMPATIBILITY METHODS
//==============================================================================

impl LMDBStorage {
    /// Compatibility shim for remove_from_cache calls
    pub fn remove_from_cache<P: AsRef<Path>>(path: P) -> Result<()> {
        log::debug!("remove_from_cache called for {:?} - delegating to environment manager", path.as_ref());
        close_environment(path.as_ref())?;
        Ok(())
    }
    
    /// Compatibility shim for verify_closed calls
    pub fn verify_closed<P: AsRef<Path>>(path: P) -> bool {
        verify_storage_closed(path)
    }
}
