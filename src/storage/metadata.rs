// storage/lmdb/metadata.rs
// Metadata operations - All functions converted to scoped transactions

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, Instant, Duration};
use log::{info, trace, warn, debug, error};
use lmdb_rkv::{Transaction, WriteFlags, Cursor};

use crate::error::{Error, Result};
use crate::storage::{FileMetadata, MetadataSummary};
use crate::storage::cache::METADATA_CACHE;
use crate::types::NGramMetadata;

use super::LMDBStorage;
// REMOVED unused imports: CF_METADATA and CF_NGRAM_METADATA

// Constants for fixed batch sizes
const METADATA_STORE_BATCH_SIZE: usize = 100;      // For store_metadata batching
const METADATA_REBUILD_BATCH_SIZE: usize = 5000;   // For rebuild operations
const METADATA_PRELOAD_BATCH_SIZE: usize = 1000;   // For preload operations
const MAX_RETRY_ATTEMPTS: usize = 3;               // Standard retry limit

impl LMDBStorage {
    /// Calculate a hash for a given text
    pub fn calculate_text_hash(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Extract the first N characters of text for metadata storage
    /// Returns UTF-8 bytes of exactly prefix_depth characters (or fewer if text is shorter)
    pub fn extract_first_chars(&self, text: &str, prefix_depth: usize) -> Vec<u8> {
        // Take exactly prefix_depth characters, respecting UTF-8 boundaries
        let prefix_str: String = text
            .chars()
            .take(prefix_depth)
            .collect();
        
        // Convert to bytes for storage - these will always be valid UTF-8
        prefix_str.into_bytes()
    }
    
    /// Store file metadata
    pub fn store_metadata(&mut self, metadata: &[FileMetadata], prefix_index_depth: usize) -> Result<()> {
        if metadata.is_empty() {
            debug!("No metadata to store, returning early");
            return Ok(());
        }
        
        let start_time = Instant::now();
        trace!("Storing metadata for {} files", metadata.len());
        
        // Create summary OUTSIDE scope (avoids borrow issues)
        let summary = MetadataSummary {
            total_files: metadata.len() as u64,
            total_ngrams: metadata.iter().map(|m| m.total_ngrams).sum(),
            last_updated: SystemTime::now(),
            prefix_index_depth,
        };
        
        // Serialize summary OUTSIDE scope
        let serialized_summary = bincode::serialize(&summary)
            .map_err(|e| Error::Serialization(format!("Failed to serialize metadata summary: {}", e)))?;
        
        // Process in fixed-size batches
        for (batch_idx, metadata_chunk) in metadata.chunks(METADATA_STORE_BATCH_SIZE).enumerate() {
            trace!("Processing metadata batch {}", batch_idx);
            
            // Pre-serialize all file metadata OUTSIDE scope to avoid borrow issues
            let serialized_file_metadata: Vec<(Vec<u8>, Vec<u8>)> = metadata_chunk
                .iter()
                .filter_map(|file_meta| {
                    let key = file_meta.file_path.as_bytes().to_vec();
                    match bincode::serialize(file_meta) {
                        Ok(value) => Some((key, value)),
                        Err(e) => {
                            warn!("Failed to serialize metadata for {}: {}", file_meta.file_path, e);
                            None
                        }
                    }
                })
                .collect();
            
            if serialized_file_metadata.is_empty() {
                warn!("No valid metadata to store in batch {}", batch_idx);
                continue;
            }
            
            // Retry logic wrapping the entire scope
            let mut retry_count = 0;
            loop {
                retry_count += 1;
                
                let operation_result = self.with_write_scope(|mut scope| {
                    // Get database handle first (immutable borrow)
                    let metadata_db = scope.storage().metadata_db;
                    
                    // Store the summary (only in first batch)
                    if batch_idx == 0 {
                        scope.txn().put(
                            metadata_db, 
                            b"summary", 
                            &serialized_summary, 
                            WriteFlags::empty()
                        ).map_err(|e| Error::Database(format!("Failed to store summary: {}", e)))?;
                    }
                    
                    // Store individual file metadata
                    for (key, value) in &serialized_file_metadata {
                        scope.txn().put(metadata_db, key, value, WriteFlags::empty())
                            .map_err(|e| Error::Database(format!("Failed to store file metadata: {}", e)))?;
                    }
                    
                    scope.commit()
                });
                
                match operation_result {
                    Ok(_) => {
                        trace!("Successfully committed metadata batch {}", batch_idx);
                        break;
                    },
                    Err(e) if retry_count < MAX_RETRY_ATTEMPTS && should_retry_lmdb_error(&e.to_string()) => {
                        debug!("Metadata storage failed (attempt {}), retrying: {}", retry_count, e);
                        std::thread::sleep(Duration::from_millis(10 * retry_count as u64));
                        continue;
                    },
                    Err(e) => return Err(e),
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        info!("Stored metadata for {} files in {:?}", metadata.len(), elapsed);
        self.metrics.increment_writes();
        
        Ok(())
    }
    
    /// Get the stored metadata summary
    pub fn get_metadata_summary(&self) -> Result<MetadataSummary> {
        self.with_read_scope(|scope| {
            // Get database handle first
            let metadata_db = scope.storage().metadata_db;
            
            // Try to get the summary
            match scope.txn().get(metadata_db, b"summary") {
                Ok(summary_bytes) => {
                    match bincode::deserialize::<MetadataSummary>(summary_bytes) {
                        Ok(summary) => {
                            debug!("Retrieved metadata summary: {} files, {} total ngrams", 
                                  summary.total_files, summary.total_ngrams);
                            Ok(summary)
                        },
                        Err(e) => {
                            warn!("Failed to deserialize metadata summary: {}", e);
                            Err(Error::Serialization(format!(
                                "Failed to deserialize database stats: {}", e
                            )))
                        }
                    }
                },
                Err(lmdb_rkv::Error::NotFound) => {
                    debug!("No metadata summary found, returning default");
                    Ok(MetadataSummary::default())
                },
                Err(e) => {
                    warn!("Failed to retrieve metadata summary: {}", e);
                    Err(Error::Database(format!(
                        "Failed to get database stats: {}", e
                    )))
                }
            }
        })
    }

    /// Rebuild metadata for all NGram entries in the database
    /// FIXED: Now correctly takes &mut self since it performs write operations
    pub fn rebuild_metadata(&mut self, start_sequence: u64, end_sequence: u64) -> Result<()> {
        let start_time = Instant::now();
        trace!("Rebuilding metadata for sequences {}..{}", start_sequence, end_sequence);
        
        if start_sequence >= end_sequence {
            warn!("Invalid sequence range: {} >= {}", start_sequence, end_sequence);
            return Ok(());
        }
        
        let total_sequences = end_sequence - start_sequence;
        let mut total_processed = 0u64;
        
        // Process in fixed-size batches - no more adaptive sizing
        // TODO: Integrate with ResourceCoordinator for dynamic batch sizing
        for batch_start in (start_sequence..end_sequence).step_by(METADATA_REBUILD_BATCH_SIZE) {
            let batch_end = (batch_start + METADATA_REBUILD_BATCH_SIZE as u64).min(end_sequence);
            
            // Progress reporting
            if total_processed > 0 && total_processed % 50000 == 0 {
                let progress = (total_processed as f64 / total_sequences as f64) * 100.0;
                trace!("Metadata rebuild progress: {:.1}% ({}/{})", 
                    progress, total_processed, total_sequences);
            }
            
            // Process this batch using the helper function
            let batch_count = self.rebuild_metadata_batch(batch_start, batch_end)?;
            total_processed += batch_count;
        }
        
        let elapsed = start_time.elapsed();
        info!("Completed metadata rebuild of {} entries in {:?} ({:.1} entries/sec)", 
            total_processed, elapsed, total_processed as f64 / elapsed.as_secs_f64());
        
        Ok(())
    }

    /// Helper function to rebuild metadata for a batch of sequences
    fn rebuild_metadata_batch(&mut self, batch_start: u64, batch_end: u64) -> Result<u64> {
        let mut entries_to_process = Vec::new();
        
        // First, read all entries in this batch using a read scope
        self.with_read_scope(|scope| {
            let main_db = scope.storage().sequences_db;
            
            for sequence in batch_start..batch_end {
                let key = sequence.to_be_bytes();
                
                match scope.txn().get(main_db, &key) {
                    Ok(bytes) => {
                        match bincode::deserialize::<crate::types::NGramEntry>(bytes) {
                            Ok(entry) => {
                                entries_to_process.push(entry);
                            }
                            Err(e) => {
                                trace!("Skipping sequence {}: deserialization error: {}", sequence, e);
                            }
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        // This is normal - not all sequences exist
                        trace!("Sequence {} not found", sequence);
                    },
                    Err(e) => {
                        debug!("Error reading sequence {}: {}", sequence, e);
                    }
                }
            }
            Ok(())
        })?;
        
        let entries_count = entries_to_process.len() as u64;
        
        if entries_to_process.is_empty() {
            trace!("No entries found in batch {}-{}", batch_start, batch_end);
            return Ok(0);
        }
        
        // Pre-calculate all metadata OUTSIDE the write scope
        let metadata_entries: Vec<(u64, NGramMetadata)> = entries_to_process
            .iter()
            .map(|entry| {
                let metadata = NGramMetadata {
                    sequence_number: entry.sequence_number,
                    text_length: entry.text_content.cleaned_text.len() as u64,
                    text_hash: self.calculate_text_hash(&entry.text_content.cleaned_text),
                };
                (entry.sequence_number, metadata)
            })
            .collect();
        
        // Now write all metadata in a single write scope
        self.with_write_scope(|mut scope| {
            let metadata_db = scope.storage().ngram_metadata_db;
            
            for (sequence, metadata) in &metadata_entries {
                let key = sequence.to_be_bytes();
                let serialized = bincode::serialize(metadata)
                    .map_err(|e| Error::Serialization(format!("Failed to serialize metadata: {}", e)))?;
                
                scope.txn().put(metadata_db, &key, &serialized, WriteFlags::empty())
                    .map_err(|e| Error::Database(format!("Failed to store metadata for {}: {}", sequence, e)))?;
            }
            
            scope.commit()
        })?;
        
        trace!("Processed batch {}-{}: {} entries", batch_start, batch_end-1, entries_count);
        Ok(entries_count)
    }

    /// Preload metadata into cache for faster access
    pub fn preload_metadata(&self, start_sequence: u64, count: usize) -> Result<usize> {
        trace!("Preloading {} metadata entries starting from sequence {}", count, start_sequence);
        
        if count == 0 {
            return Ok(0);
        }
        
        let start_time = Instant::now();
        let mut total_loaded = 0;
        let end_sequence = start_sequence + count as u64;
        
        // Process in fixed-size batches
        // TODO: Integrate with ResourceCoordinator for dynamic batch sizing
        for batch_start in (start_sequence..end_sequence).step_by(METADATA_PRELOAD_BATCH_SIZE) {
            let batch_end = (batch_start + METADATA_PRELOAD_BATCH_SIZE as u64).min(end_sequence);
            
            // Read and cache metadata for this batch
            let batch_loaded = self.with_read_scope(|scope| {
                let metadata_db = scope.storage().ngram_metadata_db;
                let mut cache = scope.storage().metadata_cache.write();
                let mut loaded_count = 0;
                
                for sequence in batch_start..batch_end {
                    // Skip if already cached
                    if cache.contains(&sequence) {
                        continue;
                    }
                    
                    let key = sequence.to_be_bytes();
                    
                    match scope.txn().get(metadata_db, &key) {
                        Ok(bytes) => {
                            if let Ok(metadata) = bincode::deserialize::<NGramMetadata>(bytes) {
                                cache.put(sequence, metadata.clone());
                                METADATA_CACHE.insert(sequence, metadata);
                                loaded_count += 1;
                            }
                        },
                        Err(lmdb_rkv::Error::NotFound) => continue,
                        Err(e) => {
                            trace!("Error preloading metadata for sequence {}: {}", sequence, e);
                        }
                    }
                }
                
                Ok(loaded_count)
            })?;
            
            total_loaded += batch_loaded;
        }
        
        let elapsed = start_time.elapsed();
        let rate = if elapsed.as_secs() > 0 {
            total_loaded as f64 / elapsed.as_secs_f64()
        } else {
            total_loaded as f64
        };
        
        // Report cache statistics
        let (current_size, capacity) = {
            let cache = self.metadata_cache.read();
            (cache.len(), cache.cap().get())
        };
        let usage_percent = (current_size as f64 / capacity as f64) * 100.0;
        
        info!("Preloaded {} metadata entries in {:?} ({:.1} entries/sec, {:.1}% cache usage)",
            total_loaded, elapsed, rate, usage_percent);
        
        Ok(total_loaded)
    }

    /// Retrieve metadata for a batch of NGram sequences
    pub fn get_ngram_metadata_batch(&self, sequences: &[u64]) -> Result<Vec<NGramMetadata>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        // ADD THIS DEBUG
        if sequences.contains(&1) {
            debug!("Attempting to get metadata for sequence 1 from database: {:?}", self.db_path);
            
            // Check what's actually in the metadata database
            self.with_read_scope(|scope| {
                let metadata_db = scope.storage().ngram_metadata_db;
                let cursor_result = scope.txn().open_ro_cursor(metadata_db);
                
                match cursor_result {
                    Ok(mut cursor) => {
                        let mut count = 0;
                        let mut first_sequences = Vec::new();
                        
                        for entry in cursor.iter() {
                            match entry {
                                Ok((key, _)) => {
                                    if key.len() == 8 {
                                        let seq = u64::from_be_bytes(key.try_into().unwrap());
                                        if count < 5 {
                                            first_sequences.push(seq);
                                        }
                                        count += 1;
                                    }
                                },
                                Err(e) => {
                                    warn!("Error iterating cursor: {}", e);
                                }
                            }
                        }
                        
                        error!("Database {:?} contains {} metadata entries", self.db_path, count);
                        error!("First sequences in database: {:?}", first_sequences);
                    },
                    Err(e) => {
                        error!("Failed to open cursor: {}", e);
                    }
                }
                Ok(())
            })?;
        }
        
        trace!("Retrieving metadata for {} sequences", sequences.len());
        let mut metadata_vec = Vec::with_capacity(sequences.len());
        let mut cache_hits = 0;
        let mut cache_misses = Vec::new();
        
        // First, check instance cache
        {
            let cache = self.metadata_cache.read();
            for &sequence in sequences {
                if let Some(metadata) = cache.peek(&sequence) {
                    metadata_vec.push(metadata.clone());
                    cache_hits += 1;
                    
                    // Update thread-local cache
                    METADATA_CACHE.insert(sequence, metadata.clone());
                } else {
                    cache_misses.push(sequence);
                }
            }
        }
        
        // For cache misses, query the database
        if !cache_misses.is_empty() {
            // Process in reasonable chunks
            const CHUNK_SIZE: usize = 1000;
            
            for chunk in cache_misses.chunks(CHUNK_SIZE) {
                self.with_read_scope(|scope| {
                    // Get database handle first
                    let metadata_db = scope.storage().ngram_metadata_db;
                    
                    // Create a temporary vector for this chunk's results
                    let mut chunk_results = Vec::new();
                    
                    for &sequence in chunk {
                        let key = sequence.to_be_bytes();
                        
                        match scope.txn().get(metadata_db, &key) {
                            Ok(bytes) => {
                                match bincode::deserialize::<NGramMetadata>(bytes) {
                                    Ok(metadata) => {
                                        chunk_results.push((sequence, metadata));
                                    }
                                    Err(e) => {
                                        error!("Failed to deserialize metadata for {}: {}", sequence, e);
                                    }
                                }
                            },
                            Err(lmdb_rkv::Error::NotFound) => {
                                error!("Metadata not found for sequence {}", sequence);
                            },
                            Err(e) => {
                                error!("Error reading metadata for sequence {}: {}", sequence, e);
                            }
                        }
                    }
                    
                    // Update caches AFTER reading all data (to avoid multiple scope borrows)
                    let mut cache = scope.storage().metadata_cache.write();
                    for (sequence, metadata) in chunk_results {
                        metadata_vec.push(metadata.clone());
                        cache.put(sequence, metadata.clone());
                        METADATA_CACHE.insert(sequence, metadata);
                    }
                    
                    Ok(())
                })?;
            }
        }
        
        trace!("Retrieved {} metadata entries ({} cache hits, {} database queries)", 
            metadata_vec.len(), cache_hits, cache_misses.len());
        
        Ok(metadata_vec)
    }
}

/// Helper function to determine if an LMDB error should be retried
fn should_retry_lmdb_error(error_str: &str) -> bool {
    error_str.contains("Resource temporarily unavailable") ||
    error_str.contains("MDB_READERS_FULL") ||
    error_str.contains("Database lock") ||
    error_str.contains("I/O error")
}