// storage/lmdb/mod.rs

// Declare submodules
pub mod config;
pub mod init;
pub mod batch;
pub mod prefix;
pub mod query;
pub mod maintenance;
pub mod metadata;
pub mod cache;

// Standard imports
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use dashmap::DashMap;
use std::collections::HashMap;
use log::{info, debug, warn};
use lazy_static::lazy_static;
use lru::LruCache;
use parking_lot::RwLock;
use std::ops::Range;
use lmdb_rkv::{Environment, Database};

// Import from crate
use crate::error::{Error, Result};
use crate::types::NGramEntry;
use crate::storage::metrics::StorageMetrics;
use crate::storage::NGramMetadata;
use crate::config::subsystems::storage::StorageConfig;
use super::{
    StorageBackend, StorageStats,
};

// Define DB_CACHE - this is used across the app
lazy_static! {
    pub static ref DB_CACHE: DashMap<PathBuf, Arc<parking_lot::Mutex<LMDBStorage>>> = DashMap::new();
}

// Core LMDBStorage struct definition
#[derive(Clone)]
pub struct LMDBStorage {
    pub(crate) env: Arc<Environment>,
    pub(crate) main_db: Database,
    pub(crate) metadata_db: Database,
    pub(crate) ngram_metadata_db: Database,
    pub(crate) prefix_dbs: HashMap<String, Database>,
    pub(crate) metrics: Arc<StorageMetrics>,
    pub(crate) is_bulk_loading: bool,
    pub(crate) current_memory_usage: Arc<AtomicUsize>,
    pub(crate) adaptive_batch_size: usize,
    pub(crate) prefix_cache: Arc<RwLock<LruCache<Vec<u8>, Vec<u64>>>>,
    pub(crate) metadata_cache: Arc<RwLock<LruCache<u64, NGramMetadata>>>,
    pub(crate) db_path: PathBuf,
    pub(crate) map_size: usize,
}

// Debug impl needs to be manual due to complex struct
impl std::fmt::Debug for LMDBStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LMDBStorage")
            .field("db_path", &self.db_path.display())
            .field("is_bulk_loading", &self.is_bulk_loading)
            .field("adaptive_batch_size", &self.adaptive_batch_size)
            .field("current_memory_usage", &self.current_memory_usage.load(Ordering::Relaxed))
            .field("metadata_cache_size", &self.metadata_cache.read().len())
            .finish()
    }
}

impl LMDBStorage {
    // Core utility methods used across multiple modules
    /// Create a new storage backend instance with the given path and configuration
    pub fn create_storage<P: AsRef<Path>>(path: P, config: StorageConfig, file_size: Option<u64>) -> Result<LMDBStorage> {
        LMDBStorage::new(path, config, file_size)
    }

    /// Create a storage instance or reuse an existing one from the cache
    pub fn get_or_create_storage<P: AsRef<Path>>(path: P, config: StorageConfig, file_size: Option<u64>) -> Result<LMDBStorage> {
        let path_buf = path.as_ref().to_path_buf();
        
        // Try to get from cache first
        if let Some(cached) = DB_CACHE.get(&path_buf) {
            log::debug!("Reusing cached database connection for {:?}", path_buf);
            let storage = cached.lock().clone();
            return Ok(storage);
        }
        
        // Not in cache, create new
        log::debug!("Creating new database connection for {:?}", path_buf);
        Self::create_storage(path, config, file_size)
    }
    
    /// Serialize an NGramEntry to bytes
    pub(crate) fn serialize_entry(entry: &NGramEntry) -> Result<Vec<u8>> {
        bincode::serialize(entry)
            .map_err(|e| Error::Serialization(format!("Bincode serialization failed: {}", e)))
    }

    /// Deserialize bytes to an NGramEntry
    pub(crate) fn deserialize_entry(bytes: &[u8]) -> Result<NGramEntry> {
        bincode::deserialize(bytes)
            .map_err(|e| Error::Serialization(format!("Bincode deserialization failed: {}", e)))
    }

    /// Create a database key from a sequence number
    pub(crate) fn create_ngram_key(sequence: u64) -> Vec<u8> {
        sequence.to_be_bytes().to_vec()
    }
    /// Explicitly close the database connection and release all resources
    pub fn close(&mut self) -> Result<()> {
        // Get the path before dropping the environment
        let path = self.db_path.clone();
        
        // Log what we're doing
        info!("Explicitly closing LMDB database at {:?}", path);
        
        // First flush any pending writes - LMDB doesn't need this, but keep the API consistent
        self.cleanup()?;
        
        // Clear the prefix cache to release memory
        self.prefix_cache.write().clear();
        
        // Clear the existing metadata cache
        self.metadata_cache.write().clear();
        
        // Clear the thread-local metadata cache (NEW ADDITION)
        cache::METADATA_CACHE.clear_thread_local();
        
        // Remove this instance from the DB_CACHE
        Self::remove_from_cache(&path);
        
        // Force memory cleanup
        self.force_memory_cleanup()?;
        
        Ok(())
    }
    /// Remove a database from the global cache by path
    pub fn remove_from_cache(path: &Path) {
        if DB_CACHE.contains_key(path) {
            // Remove from the cache
            if let Some((_, instance)) = DB_CACHE.remove(path) {
                // Try to get exclusive access to the instance for proper cleanup
                match Arc::try_unwrap(instance) {
                    Ok(mutex) => {
                        // Try to lock the mutex
                        match mutex.try_lock() {
                            Some(mut guard) => {
                                // Force release of Arc references in the instance
                                if let Err(e) = guard.force_memory_cleanup() {
                                    warn!("Error during forced cleanup: {}", e);
                                }
                                
                                // Drop the guard explicitly to ensure proper release
                                drop(guard);
                            },
                            None => {
                                warn!("Failed to lock mutex for DB cleanup");
                            }
                        }
                    },
                    Err(_) => {
                        warn!("Failed to get exclusive ownership of DB");
                    }
                }
                
                info!("Removed database from cache: {:?}", path);
            }
        } else {
            debug!("Database not found in cache: {:?}", path);
        }
    }

    /// Verify if a database is properly closed and resources released
    pub fn verify_closed(path: &Path) -> bool {
        // Check if it's still in the cache
        let in_cache = DB_CACHE.contains_key(path);
        
        // LMDB doesn't maintain lock files the same way as RocksDB, but
        // we can check for our presence in the cache
        debug!("Database {:?} status: in_cache={}", path, in_cache);
        
        !in_cache
    }

    /// Force memory cleanup
    pub fn force_memory_cleanup(&mut self) -> Result<()> {
        self.current_memory_usage.store(0, Ordering::Relaxed);
        Ok(())
    }

}

// StorageBackend trait implementation
impl StorageBackend for LMDBStorage {
    fn store_ngram(&mut self, entry: &NGramEntry) -> Result<()> {
        // This will be implemented in batch.rs
        // We're calling directly into the module implementation
        self.store_ngram(entry)
    }

    fn store_ngrams_batch(&mut self, entries: &[NGramEntry]) -> Result<()> {
        // This will be implemented in batch.rs
        self.store_ngrams_batch(entries)
    }

    fn get_sequence_range(&self, range: Range<u64>) -> Result<Vec<NGramEntry>> {
        // This will be implemented in query.rs
        self.get_sequence_range(range)
    }

    fn get_stats(&self) -> Result<StorageStats> {
        // Call the implementation in maintenance.rs with the new name
        self.calculate_storage_stats()
    }

    fn cleanup(&mut self) -> Result<()> {
        // This will be implemented in maintenance.rs
        self.cleanup()
    }

    fn compact(&mut self) -> Result<()> {
        // This is a no-op for LMDB since it doesn't need manual compaction like RocksDB
        Ok(())
    }

    fn find_ngrams_by_prefix(&self, prefix: &[u8]) -> Result<Vec<NGramEntry>> {
        // This will be implemented in query.rs
        self.find_ngrams_by_prefix(prefix)
    }

    fn get_sequences_by_prefix(&self, prefix: &[u8]) -> Result<Vec<u64>> {
        // This will be implemented in prefix.rs
        self.get_sequences_by_prefix(prefix)
    }

    fn get_sequences_by_prefix_for_ngram(&self, ngram: &NGramEntry) -> Result<Vec<u64>> {
        // This will be implemented in prefix.rs
        self.get_sequences_by_prefix_for_ngram(ngram)
    }

    fn get_ngrams_batch(&self, sequences: &[u64]) -> Result<Vec<NGramEntry>> {
        // This will be implemented in batch.rs
        self.get_ngrams_batch(sequences)
    }

    fn calculate_text_hash(&self, text: &str) -> u64 {
        // This will be implemented in metadata.rs
        self.calculate_text_hash(text)
    }

    fn get_ngram_metadata_batch(&self, sequences: &[u64]) -> Result<Vec<NGramMetadata>> {
        // This will be implemented in metadata.rs
        self.get_ngram_metadata_batch(sequences)
    }

    fn close(&mut self) -> Result<()> {
        // Call the implementation method
        self.close()
    }
}