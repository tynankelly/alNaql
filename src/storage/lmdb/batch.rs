// storage/lmdb/batch.rs

use log::{info, debug, trace, warn};
use lmdb_rkv::{WriteFlags, Transaction, Error as LmdbError};
use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicUsize;

use crate::error::{Error, Result};
use crate::types::NGramEntry;
use crate::storage::NGramMetadata;

use super::LMDBStorage;

// Constants for transaction retry logic
const MAX_RETRY_ATTEMPTS: usize = 5;
const BASE_RETRY_DELAY_MS: u64 = 10;

impl LMDBStorage {
    /// Stores a single NGram entry
    pub fn store_ngram(&mut self, entry: &NGramEntry) -> Result<()> {
        // Simply delegate to the batch function for consistency
        self.store_ngrams_batch_internal(&[entry.clone()])?;
        self.metrics.increment_writes();
        Ok(())
    }

    /// Stores a batch of NGram entries with verification
    pub(crate) fn store_ngrams_batch_internal(&mut self, chunk: &[NGramEntry]) -> Result<()> {
        debug!("Storing batch of {} ngrams", chunk.len());
        
        // Begin a write transaction with retry logic
        let mut attempt = 0;
        let start_time = Instant::now();
        
        while attempt < MAX_RETRY_ATTEMPTS {
            attempt += 1;
            
            // Begin transaction
            let mut txn = match self.env.begin_rw_txn() {
                Ok(txn) => txn,
                Err(e) => {
                    // Check if we should retry based on error type
                    if should_retry_transaction_error(&e) && attempt < MAX_RETRY_ATTEMPTS {
                        debug!("Transaction start failed (attempt {}), retrying: {}", attempt, e);
                        apply_retry_backoff(attempt);
                        continue;
                    }
                    return Err(Error::Database(format!("Failed to start write transaction: {}", e)));
                }
            };
            
            // Perform operations, tracking any errors
            let mut operation_error = None;
            
            for entry in chunk {
                let key = Self::create_ngram_key(entry.sequence_number);
                let serialized = match Self::serialize_entry(entry) {
                    Ok(data) => data,
                    Err(e) => {
                        operation_error = Some(e);
                        break;
                    }
                };
                
                // Add to main database
                if let Err(e) = txn.put(self.main_db, &key, &serialized, WriteFlags::empty()) {
                    operation_error = Some(Error::Database(format!("Failed to store ngram: {}", e)));
                    break;
                }
                
                // Add metadata
                let metadata = NGramMetadata {
                    sequence_number: entry.sequence_number,
                    text_length: entry.text_content.cleaned_text.len(),
                    text_hash: self.calculate_text_hash(&entry.text_content.cleaned_text),
                    first_chars: self.extract_first_chars(&entry.text_content.cleaned_text),
                };
                
                match bincode::serialize(&metadata) {
                    Ok(serialized_meta) => {
                        if let Err(e) = txn.put(self.ngram_metadata_db, &key, &serialized_meta, WriteFlags::empty()) {
                            operation_error = Some(Error::Database(format!("Failed to store metadata: {}", e)));
                            break;
                        }
                    },
                    Err(e) => {
                        operation_error = Some(Error::Serialization(e.to_string()));
                        break;
                    }
                }
            }
            
            // If we had an error during operations, abort transaction and return the error
            if let Some(e) = operation_error {
                txn.abort();
                return Err(e);
            }
            
            // Commit transaction
            match txn.commit() {
                Ok(_) => {
                    // Log extra info if we retried
                    if attempt > 1 {
                        debug!("Transaction succeeded after {} attempts in {:?}", 
                               attempt, start_time.elapsed());
                    }
                    
                    // Update metrics
                    self.metrics.increment_batch_writes(chunk.len());
                    
                    // Verify a few entries if in debug mode
                    if cfg!(debug_assertions) && !chunk.is_empty() {
                        self.verify_written_entries(&chunk[..5.min(chunk.len())])?;
                    }
                    
                    return Ok(());
                },
                Err(e) => {
                    // Check if we should retry based on error type
                    if should_retry_transaction_error(&e) && attempt < MAX_RETRY_ATTEMPTS {
                        debug!("Transaction commit failed (attempt {}), retrying: {}", attempt, e);
                        apply_retry_backoff(attempt);
                        continue;
                    }
                    return Err(Error::Database(format!("Failed to commit transaction: {}", e)));
                }
            }
        }
        
        Err(Error::Database(format!("Failed to complete transaction after {} attempts", MAX_RETRY_ATTEMPTS)))
    }
    
    /// Verify that entries were correctly written
    fn verify_written_entries(&self, entries: &[NGramEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        
        debug!("Verifying {} written entries", entries.len());
        
        // Begin a read transaction
        let txn = match self.env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => return Err(Error::Database(format!("Failed to start read transaction for verification: {}", e))),
        };
        
        for entry in entries {
            let key = Self::create_ngram_key(entry.sequence_number);
            
            match txn.get(self.main_db, &key) {
                Ok(value) => {
                    match Self::deserialize_entry(value) {
                        Ok(stored_entry) => {
                            // Verify text content matches
                            if stored_entry.text_content.cleaned_text != entry.text_content.cleaned_text {
                                txn.abort();
                                return Err(Error::Storage(format!(
                                    "Content mismatch for key {}: expected '{}' but got '{}'",
                                    entry.sequence_number,
                                    entry.text_content.cleaned_text,
                                    stored_entry.text_content.cleaned_text
                                )));
                            }
                        },
                        Err(e) => {
                            warn!("Failed to deserialize entry for key {}: {}", entry.sequence_number, e);
                            txn.abort();
                            return Err(Error::Storage(format!(
                                "Verification failed for key {}: {}",
                                entry.sequence_number, e
                            )));
                        }
                    }
                },
                Err(e) => {
                    warn!("Failed to verify key {}: {}", entry.sequence_number, e);
                    txn.abort();
                    return Err(Error::Storage(format!(
                        "Key {} not found during verification: {}",
                        entry.sequence_number, e
                    )));
                }
            }
        }
        
        // End transaction
        txn.abort();
        
        debug!("Verification completed successfully for {} entries", entries.len());
        Ok(())
    }

    /// Store multiple NGram entries in batches
    pub fn store_ngrams_batch(&mut self, entries: &[NGramEntry]) -> Result<()> {
        // Determine optimal batch size
        let batch_size = if self.is_bulk_loading {
            // Larger batches during bulk loading
            5000
        } else {
            // Smaller batches for normal operation
            1000
        };
        
        debug!("Processing {} entries in batches of {}", entries.len(), batch_size);
        
        // Set up sync tracking
        const SYNC_FREQUENCY: usize = 10; // Sync every 10 batches
        let mut batches_since_sync = 0;
        
        // Set up stale reader check tracking
        const READER_CHECK_FREQUENCY: usize = 50; // Check for stale readers less frequently
        static READER_CHECK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        
        let start_time = Instant::now();
        
        // Process in chunks
        for (i, chunk) in entries.chunks(batch_size).enumerate() {
            trace!("Storing batch {}: {} entries", i + 1, chunk.len());
            self.store_ngrams_batch_internal(chunk)?;
            
            // Periodic disk sync
            batches_since_sync += 1;
            if batches_since_sync >= SYNC_FREQUENCY {
                // Perform a lightweight sync periodically
                match self.env.sync(false) { // false = less strict sync for intermediate points
                    Ok(_) => {
                        trace!("Performed periodic sync after {} batches", SYNC_FREQUENCY);
                    },
                    Err(e) => {
                        warn!("Periodic database sync failed: {}", e);
                        // Continue despite error - don't return early
                    }
                }
                batches_since_sync = 0;
            }
            
            // Periodic stale reader check (with much lower frequency than sync)
            let check_count = READER_CHECK_COUNTER.fetch_add(1, Ordering::Relaxed);
            if check_count % READER_CHECK_FREQUENCY == 0 {
                match self.check_stale_readers() {
                    Ok(cleaned_count) => {
                        if cleaned_count > 0 {
                            info!("Cleaned up {} stale reader entries during batch operation", cleaned_count);
                        }
                    },
                    Err(e) => {
                        // Just log and continue, don't fail the operation
                        warn!("Stale reader check failed during batch operation: {}", e);
                    }
                }
            }
        }
        
        // Always do a full sync at the end if we processed any entries
        if !entries.is_empty() {
            match self.env.sync(true) { // true = force sync to ensure durability
                Ok(_) => {
                    debug!("Final sync completed after storing {} entries", entries.len());
                },
                Err(e) => {
                    warn!("Final database sync failed: {}", e);
                    // Continue despite error - the data is still in the database
                }
            }
            
            // Also do a final stale reader check to clean up
            match self.check_stale_readers() {
                Ok(cleaned_count) => {
                    if cleaned_count > 0 {
                        debug!("Cleaned up {} stale reader entries after batch completion", cleaned_count);
                    }
                },
                Err(e) => {
                    warn!("Final stale reader check failed: {}", e);
                }
            }
        }
        
        let duration = start_time.elapsed();
        
        // Log performance metrics
        if entries.len() > 1000 {
            let entries_per_second = entries.len() as f64 / duration.as_secs_f64();
            info!("Completed storing {} entries in {} batches ({:.1} entries/sec, took {:?})", 
                entries.len(), 
                (entries.len() + batch_size - 1) / batch_size,
                entries_per_second,
                duration);
        } else {
            debug!("Completed storing {} entries in {} batches (took {:?})", 
                entries.len(), 
                (entries.len() + batch_size - 1) / batch_size,
                duration);
        }
        
        Ok(())
    }

    /// Retrieve multiple NGram entries by their sequence numbers with optimized batching
    pub fn get_ngrams_batch(&self, sequences: &[u64]) -> Result<Vec<NGramEntry>> {
        // Quick return for empty input
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        
        // Adaptive batch size for transaction management
        let adaptive_batch_size = match sequences.len() {
            n if n < 100 => n,
            n if n < 500 => 100,
            _ => 500
        };
        
        // Pre-allocate result vector with exact capacity needed
        let mut ngrams = Vec::with_capacity(sequences.len());
        
        // Process in batches to avoid large transactions
        for (i, chunk) in sequences.chunks(adaptive_batch_size).enumerate() {
            trace!("Processing batch {}: {} sequences", i + 1, chunk.len());
            
            // Begin a read transaction
            let txn = match self.env.begin_ro_txn() {
                Ok(txn) => txn,
                Err(e) => return Err(Error::Database(format!("Failed to start read transaction: {}", e))),
            };
            
            // Retrieve each ngram in the chunk
            for &seq in chunk {
                let key = Self::create_ngram_key(seq);
                
                match txn.get(self.main_db, &key) {
                    Ok(value) => {
                        if let Ok(entry) = Self::deserialize_entry(value) {
                            ngrams.push(entry);
                        }
                    },
                    Err(e) => {
                        // Only log at trace level since missing entries aren't necessarily errors
                        trace!("NGram with sequence {} not found: {}", seq, e);
                    }
                }
            }
            
            // End transaction
            txn.abort();
        }
        
        debug!("Retrieved {} NGrams from {} requested sequences", 
              ngrams.len(), sequences.len());
        Ok(ngrams)
    }
    
}

/// Helper function to determine if a transaction error is retryable
fn should_retry_transaction_error(error: &LmdbError) -> bool {
    match error {
        LmdbError::MapResized => true,   // Map size changed, can retry
        LmdbError::MapFull => false,     // Map is full, no point in retrying
        e if e.to_string().contains("busy") => true, // Fallback if no Busy variant
        e if e.to_string().contains("read-only") => false, // Fallback
        _ => false,                      // Don't retry for other errors
    }
}

/// Helper function to apply exponential backoff for retries
fn apply_retry_backoff(attempt: usize) {
    // Calculate delay with exponential backoff and some jitter
    let base_delay = BASE_RETRY_DELAY_MS;
    let exp_factor = 2u64.pow(attempt as u32 - 1);
    let jitter = fastrand::u64(0..10); // 0-10ms of jitter
    let delay_ms = base_delay * exp_factor + jitter;
    
    debug!("Transaction retry {} - sleeping for {}ms", attempt, delay_ms);
    std::thread::sleep(Duration::from_millis(delay_ms));
}