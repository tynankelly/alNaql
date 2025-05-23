// storage/mod.rs

pub mod metrics;
pub mod lmdb;

use std::ops::Range;
use std::path::Path;

use crate::error::Result;
use crate::types::NGramEntry;
use crate::config::subsystems::storage::StorageConfig;

// Shared types and definitions for both storage backends
pub type SequenceID = u64;

/// NGram metadata structure for efficient lookups and filtering
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NGramMetadata {
    pub sequence_number: u64,
    pub text_length: usize,
    pub text_hash: u64,
    pub first_chars: [u8; 8],
}

/// Metadata about processed files
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileMetadata {
    pub file_path: String,
    pub file_size: u64,
    pub total_ngrams: u64,
    pub processed_at: std::time::SystemTime,
}

/// Summary of database metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetadataSummary {
    pub total_files: u64,
    pub total_ngrams: u64,
    pub last_updated: std::time::SystemTime,
}

impl Default for MetadataSummary {
    fn default() -> Self {
        Self {
            total_files: 0,
            total_ngrams: 0,
            last_updated: std::time::SystemTime::now(),
        }
    }
}

/// Statistics about the storage backend
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_ngrams: u64,
    pub total_bytes: u64,
    pub unique_sequences: u64,
    pub unique_cleaned_texts: u64,
    pub compression_ratio: f64,
}

/// The StorageBackend trait defines the interface that all storage implementations must provide
pub trait StorageBackend: Send + Sync {
    /// Store a single NGram entry
    fn store_ngram(&mut self, entry: &NGramEntry) -> Result<()>;
    
    /// Store a batch of NGram entries
    fn store_ngrams_batch(&mut self, entries: &[NGramEntry]) -> Result<()>;
    
    /// Retrieve all NGram entries within a sequence range
    fn get_sequence_range(&self, range: Range<u64>) -> Result<Vec<NGramEntry>>;
    
    /// Get storage statistics
    fn get_stats(&self) -> Result<StorageStats>;
    
    /// Perform cleanup operations (like flushing)
    fn cleanup(&mut self) -> Result<()>;
    
    /// Perform database compaction
    fn compact(&mut self) -> Result<()>;
    
    /// Find NGram entries with a specific text prefix
    fn find_ngrams_by_prefix(&self, prefix: &[u8]) -> Result<Vec<NGramEntry>>;
    
    /// Find sequence numbers for NGrams with a given prefix
    fn get_sequences_by_prefix(&self, prefix: &[u8]) -> Result<Vec<u64>>;
    
    /// Find sequence numbers for NGrams with the same prefix as the given NGram
    fn get_sequences_by_prefix_for_ngram(&self, ngram: &NGramEntry) -> Result<Vec<u64>>;
    
    /// Batch retrieve NGram entries by sequence number
    fn get_ngrams_batch(&self, sequences: &[u64]) -> Result<Vec<NGramEntry>>;
    
    /// Calculate a hash for the given text
    fn calculate_text_hash(&self, text: &str) -> u64;
    
    /// Retrieve metadata for a batch of NGram sequences
    fn get_ngram_metadata_batch(&self, sequences: &[u64]) -> Result<Vec<NGramMetadata>>;
    
    /// Explicitly close the database
    fn close(&mut self) -> Result<()>;
}

/// Create a new storage backend instance with the given path and configuration
pub fn create_storage<P: AsRef<Path>>(path: P, config: StorageConfig, file_size: Option<u64>) -> Result<lmdb::LMDBStorage> {
    lmdb::LMDBStorage::new(path, config, file_size)
}

/// Create a storage instance or reuse an existing one from the cache
pub fn get_or_create_storage<P: AsRef<Path>>(path: P, config: StorageConfig, file_size: Option<u64>) -> Result<lmdb::LMDBStorage> {
    let path_buf = path.as_ref().to_path_buf();
    
    // Try to get from cache first
    if let Some(cached) = lmdb::DB_CACHE.get(&path_buf) {
        log::debug!("Reusing cached database connection for {:?}", path_buf);
        let storage = cached.lock().clone();
        return Ok(storage);
    }
    
    // Not in cache, create new
    log::debug!("Creating new database connection for {:?}", path_buf);
    lmdb::LMDBStorage::new(path, config, file_size)
}

/// Verify if a storage backend is properly closed
pub fn verify_storage_closed<P: AsRef<Path>>(path: P) -> bool {
    lmdb::LMDBStorage::verify_closed(path.as_ref())
}

/// Check for any database locks in a directory
pub fn check_locks_in_directory<P: AsRef<Path>>(dir_path: P) -> Vec<String> {
    let path = dir_path.as_ref();
    
    if !path.exists() || !path.is_dir() {
        return vec![format!("Directory not found or not a directory: {:?}", path)];
    }
    
    let mut results = Vec::new();
    
    // LMDB databases have a lock file named data.mdb-lock
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                let lock_path = entry_path.join("data.mdb-lock");
                if lock_path.exists() {
                    // LMDB locks are managed by the OS
                    results.push(format!("Found LMDB lock at {:?}", lock_path));
                }
            }
        }
    }
    
    results
}

/// Clear all database connections from cache
pub fn clear_storage_cache() {
    lmdb::DB_CACHE.clear();
}