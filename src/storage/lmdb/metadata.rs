// storage/lmdb/metadata.rs

use log::{info, trace, warn, debug};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, Instant, Duration};
use lmdb_rkv::{Transaction, WriteFlags};

use crate::error::{Error, Result};
use crate::storage::{FileMetadata, MetadataSummary, NGramMetadata};
use crate::storage::lmdb::cache::METADATA_CACHE;


use super::LMDBStorage;
use super::config::{CF_METADATA, CF_NGRAM_METADATA};

impl LMDBStorage {
    /// Calculate a hash for a given text
    pub fn calculate_text_hash(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Extract the first few characters of a text for quick comparison
    pub fn extract_first_chars(&self, text: &str) -> [u8; 8] {
        let mut result = [0; 8];
        let bytes = text.as_bytes();
        
        for i in 0..result.len() {
            if i < bytes.len() {
                result[i] = bytes[i];
            }
        }
        
        result
    }
    
    /// Store file metadata
    pub fn store_metadata(&mut self, metadata: &[FileMetadata]) -> Result<()> {
        debug!("Storing metadata for {} files", metadata.len());
        
        // Create a summary of the metadata
        let summary = MetadataSummary {
            total_files: metadata.len() as u64,
            total_ngrams: metadata.iter().map(|m| m.total_ngrams).sum(),
            last_updated: SystemTime::now(),
        };
        
        // Serialize the summary
        let serialized = bincode::serialize(&summary)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        
        // Begin a write transaction
        let mut txn = self.env.begin_rw_txn()
            .map_err(|e| Error::Database(format!("Failed to start write transaction: {}", e)))?;
        
        // Store the summary in the metadata database - using the database directly
        // We're using the constant here to make it clear this is metadata storage
        debug!("Storing metadata summary in {} database", CF_METADATA);
        txn.put(self.metadata_db, b"summary", &serialized, WriteFlags::empty())
            .map_err(|e| Error::Database(format!("Failed to store metadata summary: {}", e)))?;
        
        // For each file metadata, we could also store individual entries
        for (i, file_md) in metadata.iter().enumerate() {
            let key = format!("file:{}", i);
            let serialized = bincode::serialize(file_md)
                .map_err(|e| Error::Serialization(e.to_string()))?;
                
            txn.put(self.metadata_db, &key, &serialized, WriteFlags::empty())
                .map_err(|e| Error::Database(format!("Failed to store file metadata: {}", e)))?;
        }
        
        // Commit the transaction
        txn.commit()
            .map_err(|e| Error::Database(format!("Failed to commit metadata transaction: {}", e)))?;
        
        info!("Stored metadata summary with {} files and {} total ngrams", 
              metadata.len(), summary.total_ngrams);
       
        Ok(())
    }

    /// Preload metadata for a range of sequences to improve subsequent access performance
    pub fn preload_metadata_range(&self, range: std::ops::Range<u64>) -> Result<usize> {
        if range.start >= range.end {
            return Ok(0);
        }
        
        info!("Preloading metadata for range {}..{} ({} sequences)", 
            range.start, range.end, range.end - range.start);
        
        let start_time = Instant::now();
        
        // Process in batches to avoid excessive memory usage
        let batch_size = 10_000;
        let mut total_loaded = 0;
        let mut last_log_time = Instant::now();
        
        // Begin a single read transaction for the entire operation
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        // Process the range in batches
        for batch_start in (range.start..range.end).step_by(batch_size) {
            let batch_end = (batch_start + batch_size as u64).min(range.end);
            let batch_range = batch_start..batch_end;
            
            let batch_keys: Vec<u64> = batch_range.collect();
            
            // Get write access to the existing cache once per batch
            let mut local_cache = self.metadata_cache.write();
            
            for &sequence in &batch_keys {
                // Skip if already in cache (both caches)
                if local_cache.contains(&sequence) || METADATA_CACHE.get(sequence).is_some() {
                    continue;
                }
                
                // Create the key for this sequence
                let key = Self::create_ngram_key(sequence);
                
                // Try to fetch the metadata
                match txn.get(self.ngram_metadata_db, &key) {
                    Ok(bytes) => {
                        match bincode::deserialize::<NGramMetadata>(bytes) {
                            Ok(metadata) => {
                                // Cache in both places
                                local_cache.put(sequence, metadata.clone());
                                METADATA_CACHE.insert(sequence, metadata);
                                
                                total_loaded += 1;
                            },
                            Err(e) => {
                                // Log but continue with other entries
                                trace!("Skipping sequence {}: deserialization error: {}", sequence, e);
                            }
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        // This is normal - not all sequences exist
                        trace!("Sequence {} not found, skipping", sequence);
                    },
                    Err(e) => {
                        // Log but continue with other entries
                        trace!("Error retrieving sequence {}: {}", sequence, e);
                    }
                }
            }
            
            // Periodic progress updates
            if last_log_time.elapsed() > Duration::from_secs(5) {
                let progress_pct = (batch_end - range.start) as f64 / (range.end - range.start) as f64 * 100.0;
                info!("Preloading progress: {:.1}% complete ({} of {} sequences loaded)",
                    progress_pct, total_loaded, range.end - range.start);
                last_log_time = Instant::now();
            }
        }
        
        // End transaction
        txn.abort();
        
        let elapsed = start_time.elapsed();
        let rate = if elapsed.as_secs() > 0 {
            total_loaded as f64 / elapsed.as_secs_f64()
        } else {
            total_loaded as f64
        };
        
        info!("Preloaded {} metadata entries in {:?} ({:.1} entries/sec)",
            total_loaded, elapsed, rate);
        
        Ok(total_loaded)
    }
    /// Batch retrieval of ngram metadata with caching and optimized processing
    pub fn get_ngram_metadata_batch(&self, sequences: &[u64]) -> Result<Vec<NGramMetadata>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        
        // Pre-allocate with exact capacity
        let mut metadata_vec = Vec::with_capacity(sequences.len());
        let mut missing_sequences = Vec::new();
        
        // ENHANCEMENT: First check the fast thread-local cache for each sequence
        // This has no lock contention and is extremely fast
        for &seq in sequences {
            if let Some(metadata) = METADATA_CACHE.get(seq) {
                metadata_vec.push((*metadata).clone());
            } else {
                missing_sequences.push(seq);
            }
        }
        
        // If everything was in thread-local cache, return early
        if missing_sequences.is_empty() {
            trace!("All {} metadata entries found in thread-local cache", sequences.len());
            return Ok(metadata_vec);
        }
        
        // For remaining sequences, check the existing cache
        let mut still_missing = Vec::new();
        {
            let mut cache = self.metadata_cache.write();
            for &seq in &missing_sequences {
                if let Some(metadata) = cache.get(&seq) {
                    metadata_vec.push(metadata.clone());
                    
                    // ENHANCEMENT: Populate thread-local cache from existing cache
                    METADATA_CACHE.insert(seq, metadata.clone());
                } else {
                    still_missing.push(seq);
                }
            }
        }
        
        // If everything was found between the two caches, return early
        if still_missing.is_empty() {
            trace!("All {} metadata entries found in combined caches", sequences.len());
            return Ok(metadata_vec);
        }
        
        // Record DB lookups for statistics
        METADATA_CACHE.record_db_batch_lookup(still_missing.len());
        
        trace!("Fetching {} metadata entries from {} database", 
               still_missing.len(), CF_NGRAM_METADATA);
        
        // Start a read transaction for database access
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        // Use smaller batch size for metadata lookups
        let adaptive_batch_size = match still_missing.len() {
            n if n < 100 => n,
            n if n < 500 => 100,
            _ => 500
        };
        
        // Process in chunks for memory efficiency
        for chunk in still_missing.chunks(adaptive_batch_size) {
            trace!("Processing metadata batch of size {}", chunk.len());
            
            let batch_start = Instant::now();
            let mut batch_hits = 0;
            
            // Process each sequence in the batch
            let mut cache = self.metadata_cache.write();
            
            for &seq in chunk {
                let key = Self::create_ngram_key(seq);
                
                match txn.get(self.ngram_metadata_db, &key) {
                    Ok(bytes) => {
                        match bincode::deserialize::<NGramMetadata>(bytes) {
                            Ok(metadata) => {
                                // Add to results
                                metadata_vec.push(metadata.clone());
                                
                                // Update both caches
                                cache.put(seq, metadata.clone());
                                METADATA_CACHE.insert(seq, metadata);
                                
                                batch_hits += 1;
                            },
                            Err(e) => {
                                warn!("Failed to deserialize metadata for sequence {}: {}", seq, e);
                            }
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        trace!("Metadata for sequence {} not found", seq);
                    },
                    Err(e) => {
                        warn!("Error fetching metadata for sequence {}: {}", seq, e);
                    }
                }
            }
            
            trace!("Batch processed in {:?}: {}/{} hits", 
                  batch_start.elapsed(), batch_hits, chunk.len());
        }
        
        // End transaction
        txn.abort();
        
        trace!("Retrieved {} metadata entries (out of {} requested)", 
             metadata_vec.len(), sequences.len());
        
        Ok(metadata_vec)
    }
    
    /// Rebuild metadata for all entries in the database
    pub fn rebuild_metadata(&self) -> Result<()> {
        info!("Starting metadata rebuild for {} database...", CF_NGRAM_METADATA);
        let start_time = Instant::now();
        
        // Use a conservative maximum sequence number
        const MAX_SAFE_SEQUENCE: u64 = 10_000_000; 
        
        // Process in batches for efficiency
        let batch_size = 10000;
        let mut total_processed = 0;
        let mut last_found_seq = 0;
        
        // Process sequences in batches with adaptive scanning
        for batch_start in (1..=MAX_SAFE_SEQUENCE).step_by(batch_size as usize) {
            let batch_end = (batch_start + batch_size).min(MAX_SAFE_SEQUENCE + 1);
            
            // Start a read transaction
            let ro_txn = self.env.begin_ro_txn()
                .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
            
            // Collect entries to process
            let mut batch_entries = Vec::new();
            let mut found_any = false;
            
            for seq in batch_start..batch_end {
                let key = Self::create_ngram_key(seq);
                
                match ro_txn.get(self.main_db, &key) {
                    Ok(value) => {
                        found_any = true;
                        last_found_seq = seq;
                        
                        if let Ok(entry) = Self::deserialize_entry(value) {
                            batch_entries.push((key, entry));
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        // Entry not found, skip
                    },
                    Err(e) => {
                        ro_txn.abort();
                        return Err(Error::Database(format!("Error reading entry: {}", e)));
                    }
                }
            }
            
            // End read transaction
            ro_txn.abort();
            
            // Process collected entries
            if !batch_entries.is_empty() {
                // Store the batch length before consuming the vector
                let batch_len = batch_entries.len();
                
                // Start a write transaction
                let mut rw_txn = self.env.begin_rw_txn()
                    .map_err(|e| Error::Database(format!("Failed to start write transaction: {}", e)))?;
                
                for (key, entry) in batch_entries {
                    // Create metadata
                    let metadata = NGramMetadata {
                        sequence_number: entry.sequence_number,
                        text_length: entry.text_content.cleaned_text.len(),
                        text_hash: self.calculate_text_hash(&entry.text_content.cleaned_text),
                        first_chars: self.extract_first_chars(&entry.text_content.cleaned_text),
                    };
                    
                    // Serialize metadata
                    let serialized = bincode::serialize(&metadata)
                        .map_err(|e| Error::Serialization(e.to_string()))?;
                    
                    // Store in metadata database
                    if let Err(e) = rw_txn.put(self.ngram_metadata_db, &key, &serialized, WriteFlags::empty()) {
                        warn!("Failed to store metadata for key {:?}: {}", key, e);
                    }
                }
                
                // Commit transaction
                rw_txn.commit()
                    .map_err(|e| Error::Database(format!("Failed to commit transaction: {}", e)))?;
                
                total_processed += batch_len;
                info!("Processed batch {}-{}: {} entries ({} total)", 
                    batch_start, batch_end-1, batch_len, total_processed);
            }
            
            // Break early if we've processed a significant range with no entries
            if !found_any && batch_start > last_found_seq + batch_size * 10 {
                info!("No entries found in range {}..{}, stopping scan", 
                    batch_start, batch_end);
                break;
            }
        }
        
        let elapsed = start_time.elapsed();
        info!("Completed metadata rebuild of {} entries in {:?} ({:.1} entries/sec)", 
            total_processed, elapsed, total_processed as f64 / elapsed.as_secs_f64());
        
        Ok(())
    }
    
    /// Get the database statistics
    pub fn get_database_stats(&self) -> Result<MetadataSummary> {
        debug!("Retrieving database statistics from {} database", CF_METADATA);
        
        // Begin a read transaction
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        // Try to get the summary
        let result = match txn.get(self.metadata_db, b"summary") {
            Ok(summary_bytes) => {
                let summary: MetadataSummary = bincode::deserialize(summary_bytes)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
                Ok(summary)
            },
            Err(lmdb_rkv::Error::NotFound) => {
                debug!("No metadata summary found, returning default");
                // Return default summary if none exists
                Ok(MetadataSummary::default())
            },
            Err(e) => {
                Err(Error::Database(format!("Failed to get database stats: {}", e)))
            }
        };
        
        // End transaction
        txn.abort();
        
        result
    }
}