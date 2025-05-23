// storage/lmdb/maintenance.rs

use std::path::{Path, PathBuf};
use std::time::{Instant, Duration};
use std::sync::atomic::Ordering;
use log::{debug, info, trace, warn, error};
use lmdb_rkv::{Transaction, Environment, Cursor};

use crate::error::{Error, Result};
use crate::storage::StorageStats;

use super::LMDBStorage;
use super::prefix::PREFIX_PARTITIONS;

impl LMDBStorage {
    /// Verify the current state of the database
    pub fn verify_database_state(&self) -> Result<()> {
        debug!("Verifying LMDB database state");
        
        // Start a read transaction
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        // Count entries in main database as a basic verification
        let mut count = 0;
        {
            // Create cursor in a new scope so it gets dropped before we abort the transaction
            let mut cursor = txn.open_ro_cursor(self.main_db)
                .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
            
            // Use iterator to count entries
            for result in cursor.iter() {
                // We don't need the key-value contents, just count successful entries
                if result.is_ok() {
                    count += 1;
                    
                    // For very large databases, limit the counting for verification
                    if count > 1_000_000 {
                        debug!("Limiting database verification count to 1 million entries");
                        break;
                    }
                } else if let Err(e) = result {
                    // Optionally log iterator errors
                    debug!("Error during cursor iteration: {}", e);
                }
            }
            
            // The cursor will be dropped when this scope ends
        }
        
        debug!("Database contains at least {} entries", count);
        
        // Get environment statistics (if available)
        if let Ok(env_stat) = (*self.env).stat() {
            debug!("LMDB environment stat: page_size={}, entries={}", 
                env_stat.page_size(), env_stat.entries());
        } else {
            debug!("Environment statistics not available");
        }
        
        // Now it's safe to abort the transaction because the cursor is dropped
        txn.abort();
        
        debug!("Database verification completed successfully");
        Ok(())
    }
    
    /// Display a comprehensive overview of the database state
    /// Display a comprehensive overview of the database state
    pub fn display_complete_system_state(&self, db_paths: &[PathBuf]) -> Result<()> {
        trace!("\n=== Complete System State Overview ===");
        
        for db_path in db_paths {
            trace!("\nExamining database: {:?}", db_path);
            
            // Try to open the environment directly
            let env_result = Environment::new().open(db_path);
            
            if let Ok(env) = env_result {
                // Get basic environment information
                if let Ok(env_stat) = env.stat() {
                    trace!("Environment statistics:");
                    trace!("  Page size: {} bytes", env_stat.page_size());
                    trace!("  Total entries: {}", env_stat.entries());
                }
                
                // If this is our current database path, use our in-memory handles
                if db_path == &self.db_path {
                    trace!("\n=== Database Contents ===");
                    
                    // Check main database
                    let txn = self.env.begin_ro_txn()
                        .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
                    
                    // Sample some NGram entries
                    trace!("\n=== NGram Entries Sample ===");

                    // Declare count outside the cursor scope so it's available later
                    let mut count = 0;
                    let mut sample_count = 0;

                    {
                        // Use a scope to ensure cursor is dropped before transaction ends
                        let mut cursor = txn.open_ro_cursor(self.main_db)
                            .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
                        
                        // FIXED: Handle Result from cursor iterator
                        for result in cursor.iter() {
                            if let Ok((key, value)) = result {
                                count += 1;
                                
                                // Sample first 10 entries
                                if sample_count < 10 {
                                    if key.len() == 8 && !value.is_empty() {
                                        if let Ok(ngram) = Self::deserialize_entry(value) {
                                            trace!("Ngram {}:", ngram.sequence_number);
                                            trace!("  Text: '{}'", ngram.text_content.cleaned_text);
                                            trace!("  Start Byte: {}", ngram.text_position.start_byte);
                                            trace!("  End Byte: {}", ngram.text_position.end_byte);
                                            trace!(""); // Empty line for readability
                                            sample_count += 1;
                                        }
                                    }
                                }
                                
                                // For very large databases, limit counting
                                if count > 1_000_000 {
                                    trace!("Limiting count to 1 million entries");
                                    break;
                                }
                            }
                        }
                    } // Cursor is dropped here

                    // Now count is still accessible here
                    info!("Total number of ngrams in main DB: {}", count);
                    // Display prefix mappings with sequence counts
                    info!("\n=== Prefix Database Summary ===");
                    let mut total_prefix_sequences = 0;
                    
                    for partition in PREFIX_PARTITIONS.iter() {
                        let db_name = &partition.db_name;
                        
                        if let Some(&db) = self.prefix_dbs.get(db_name) {
                            let mut partition_sequences = 0;
                            let mut prefix_count = 0;
                            
                            // Create a new scope for the cursor
                            {
                                let mut cursor = match txn.open_ro_cursor(db) {
                                    Ok(cursor) => cursor,
                                    Err(e) => {
                                        warn!("Failed to open cursor for {}: {}", db_name, e);
                                        continue;
                                    }
                                };
                                
                                // Process each key and its values
                                let mut current_key: Option<&[u8]> = None;
                                
                                // FIXED: Handle Result from cursor iterator
                                for result in cursor.iter() {
                                    if let Ok((key, _)) = result {
                                        // Check if this is a new key
                                        if current_key.map_or(true, |k| k != key) {
                                            prefix_count += 1;
                                            current_key = Some(key);
                                        }
                                        
                                        // Count each value
                                        partition_sequences += 1;
                                    }
                                }
                            } // Cursor is dropped here
                            
                            total_prefix_sequences += partition_sequences;
                            
                            // Always show the summary for each partition
                            trace!("Database '{}': {} prefixes, {} total sequences", 
                                db_name, 
                                prefix_count,
                                partition_sequences);
                            
                            // First character(s) for this partition
                            let chars_str = partition.characters.iter()
                                .map(|c| format!("'{}'", c))
                                .collect::<Vec<_>>()
                                .join(", ");
                            
                            trace!("  Characters: {}", chars_str);
                        }
                    }
                    
                    info!("Total sequences across all prefix databases: {}", total_prefix_sequences);
                    
                    // End transaction
                    txn.abort();
                }
            } else {
                warn!("Could not open database at {:?}: {:?}", db_path, env_result.err());
            }
        }
        
        trace!("\n=== Complete System State Overview Complete ===");
        Ok(())
    }
    
    /// Calculate storage statistics for the database
    pub fn calculate_storage_stats(&self) -> Result<StorageStats> {
        info!("Gathering LMDB storage statistics");
        let start_time = Instant::now();
        
        // Start a read transaction
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        let mut total_ngrams = 0;
        let mut total_bytes = 0;
        let mut unique_texts = std::collections::HashSet::new();
        
        // Process database entries in a separate scope to ensure cursor is dropped
        {
            // Create cursor for main database
            let mut cursor = txn.open_ro_cursor(self.main_db)
                .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
            
            // Use iterator to process entries
            for item in cursor.iter() {
                // FIXED: Properly handle the Result from the cursor iterator
                if let Ok((key, value)) = item {
                    total_bytes += key.len() + value.len();
                    
                    if let Ok(entry) = Self::deserialize_entry(value) {
                        total_ngrams += 1;
                        unique_texts.insert(entry.text_content.cleaned_text);
                    }
                    
                    if total_ngrams % 10000 == 0 {
                        trace!("Processed {} entries", total_ngrams);
                    }
                }
            }
        } // Cursor is dropped here
        
        // Get database statistics if available
        let db_stat = match (*self.env).stat() {
            Ok(stat) => stat,
            Err(e) => {
                warn!("Could not get environment statistics: {}", e);
                // Skip stats - we'll handle this case below
                txn.abort(); // Explicitly abort transaction before returning
                return Ok(StorageStats {
                    total_ngrams: total_ngrams as u64,
                    total_bytes: total_bytes as u64,
                    unique_sequences: total_ngrams as u64,
                    unique_cleaned_texts: unique_texts.len() as u64,
                    compression_ratio: 1.0, // Default when stats not available
                });
            }
        };
        
        // End transaction now that we're done with the cursor
        txn.abort();
        
        let elapsed = start_time.elapsed();
        info!("Statistics gathered in {:?}: {} ngrams, {} unique texts, {} total bytes",
            elapsed, total_ngrams, unique_texts.len(), total_bytes);
        
        // Calculate compression ratio (LMDB doesn't compress, but we estimate based on page usage)
        let compression_ratio = if db_stat.branch_pages() + db_stat.leaf_pages() > 0 {
            // Estimate ratio by comparing data size to page usage
            let page_size = db_stat.page_size();
            let pages_used = db_stat.branch_pages() + db_stat.leaf_pages() + db_stat.overflow_pages();
            let storage_used = pages_used as f64 * page_size as f64;
            
            if storage_used > 0.0 && total_bytes > 0 {
                storage_used / total_bytes as f64
            } else {
                1.0 // Default if no data
            }
        } else {
            1.0 // Default if no page info
        };
        
        Ok(StorageStats {
            total_ngrams: total_ngrams as u64,
            total_bytes: total_bytes as u64,
            unique_sequences: total_ngrams as u64,
            unique_cleaned_texts: unique_texts.len() as u64,
            compression_ratio,
        })
    }

    /// Prepare database for bulk loading
    pub fn prepare_bulk_load(&mut self) -> Result<()> {
        info!("Preparing LMDB database for bulk loading");
        
        // Verify empty state
        self.verify_database_state()?;
        
        // LMDB doesn't need special preparation for bulk loading
        // But we can set a flag to optimize our write patterns
        self.is_bulk_loading = true;
        
        // Get basic database information - dereference Arc<Environment>
        if let Ok(env_stat) = (*self.env).stat() {
            info!("Current LMDB environment: page_size={}, entries={}", 
                  env_stat.page_size(), env_stat.entries());
        }
        
        info!("Bulk loading activated");
        
        Ok(())
    }

    /// Finish bulk loading and prepare database for normal operation
    pub fn finish_bulk_load(&mut self) -> Result<()> {
        if self.is_bulk_loading {
            info!("Finishing LMDB bulk loading");
            
            // Check data before finishing
            self.verify_database_state()?;
            
            // Force sync to disk
            self.env.sync(true)
                .map_err(|e| Error::Database(format!("Failed to sync environment after bulk load: {}", e)))?;
            
            self.is_bulk_loading = false;
            
            // Clean up caches
            self.prefix_cache.write().clear();
            self.metadata_cache.write().clear();
            
            info!("Bulk loading completed, database ready for normal operation");
        }
        
        Ok(())
    }

    /// Implement cleanup (commit pending transactions, sync to disk, and clean up resources)
    pub fn cleanup(&mut self) -> Result<()> {
        trace!("Running cleanup on LMDB database");
        
        // Step 1: Sync database to disk to ensure durability
        let sync_start = Instant::now();
        match self.env.sync(true) {
            Ok(_) => {
                debug!("Successfully synchronized LMDB environment to disk in {:?}", sync_start.elapsed());
            },
            Err(e) => {
                warn!("Failed to sync LMDB environment to disk: {}", e);
                // Continue with cleanup despite sync error - other cleanup operations may still be important
            }
        }
        
        // Step 2: Check for and clean up stale readers (if supported)
        if let Ok(cleaned_count) = self.check_stale_readers() {
            if cleaned_count > 0 {
                info!("Cleaned up {} stale reader entries during cleanup", cleaned_count);
            }
        }
        
        // Step 3: Clear caches if they're getting too large
        let _cache_cleanup_start = Instant::now();
        {
            let prefix_cache = self.prefix_cache.read();
            let prefix_cache_size = prefix_cache.len();
            let prefix_cache_capacity = prefix_cache.cap().get();
            
            if prefix_cache_size > prefix_cache_capacity / 2 {
                drop(prefix_cache); // Release the read lock before acquiring write lock
                debug!("Clearing prefix cache during cleanup ({} of {} entries)", 
                    prefix_cache_size, prefix_cache_capacity);
                self.prefix_cache.write().clear();
            }
        }
        
        {
            let metadata_cache = self.metadata_cache.read();
            let metadata_cache_size = metadata_cache.len();
            let metadata_cache_capacity = metadata_cache.cap().get();
            
            if metadata_cache_size > metadata_cache_capacity / 2 {
                drop(metadata_cache); // Release the read lock before acquiring write lock
                debug!("Clearing metadata cache during cleanup ({} of {} entries)", 
                    metadata_cache_size, metadata_cache_capacity);
                self.metadata_cache.write().clear();
            }
        }
        
        // Step 4: Reset memory usage tracking
        self.current_memory_usage.store(0, Ordering::Relaxed);
        
        // Step 5: Log database statistics if in debug mode
        if log::log_enabled!(log::Level::Debug) {
            if let Ok(env_stat) = self.env.stat() {
                debug!("LMDB environment status after cleanup:");
                debug!("  Page size: {} bytes", env_stat.page_size());
                debug!("  Branch pages: {}", env_stat.branch_pages());
                debug!("  Leaf pages: {}", env_stat.leaf_pages());
                debug!("  Overflow pages: {}", env_stat.overflow_pages());
                debug!("  Entries: {}", env_stat.entries());
                
                // Calculate usage percentage based on map size
                let total_pages = self.map_size / env_stat.page_size() as usize;
                let pages_used = env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages();
                let usage_percent = (pages_used as f64 * 100.0) / total_pages as f64;
                
                if usage_percent > 80.0 {
                    warn!("LMDB map usage is high: {:.1}%", usage_percent);
                } else {
                    debug!("  Map usage: {:.1}%", usage_percent);
                }
            }
        }
        
        let total_duration = sync_start.elapsed();
        if total_duration > Duration::from_millis(100) {
            // Only log timing details if cleanup took a significant amount of time
            debug!("LMDB cleanup completed in {:?}", total_duration);
        } else {
            trace!("LMDB cleanup completed in {:?}", total_duration);
        }
        
        Ok(())
    }

    /// Check for database locks in a directory
    pub fn check_locks_in_directory(dir_path: &Path) -> Vec<String> {
        let mut results = Vec::new();
        
        if !dir_path.exists() || !dir_path.is_dir() {
            return vec![format!("Directory not found or not a directory: {:?}", dir_path)];
        }
        
        // LMDB uses a single lockfile per environment
        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Check for lock file
                    let lock_path = path.join("lock.mdb");
                    if lock_path.exists() {
                        results.push(format!("Found LMDB lock at {:?}", lock_path));
                        
                        // Check if we can open the environment
                        match Environment::new().open(&path) {
                            Ok(_) => {
                                results.push(format!("  Environment at {:?} can be opened (not locked)", path));
                            },
                            Err(e) => {
                                results.push(format!("  Environment at {:?} cannot be opened: {}", path, e));
                            }
                        }
                    }
                }
            }
        }
        
        results
    }

    /// Check for stale readers
    pub fn check_stale_readers(&self) -> Result<usize> {
        // LMDB's Rust binding doesn't directly expose a reader_check method
        // Implement a fallback approach
        debug!("Reader check functionality not available in this LMDB binding");
        
        // Return 0 to indicate no readers were cleaned up
        Ok(0)
    }

    /// Get map usage percentage - useful for monitoring LMDB space
    pub fn get_map_usage_percentage(&self) -> Result<f64> {
        // Get environment statistics - dereference Arc<Environment>
        let env_stat = (*self.env).stat()
            .map_err(|e| Error::Database(format!("Failed to get environment statistics: {}", e)))?;
        
        // Calculate page usage
        let page_size = env_stat.page_size() as usize;
        let pages_used = env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages();
        let total_pages = self.map_size / page_size;
        
        // Calculate percentage
        let percentage = if total_pages > 0 {
            (pages_used as f64 * 100.0) / total_pages as f64
        } else {
            0.0
        };
        
        Ok(percentage)
    }

    /// Check map size usage and log warnings if approaching capacity
    pub fn check_map_size_usage(&self) -> Result<()> {
        let percentage = self.get_map_usage_percentage()?;
        
        if percentage > 70.0 && percentage <= 80.0 {
            info!("LMDB map usage at {:.1}% - moderate usage", percentage);
        } else if percentage > 80.0 && percentage <= 90.0 {
            warn!("LMDB map usage at {:.1}% - approaching capacity", percentage);
        } else if percentage > 90.0 {
            error!("LMDB map usage at {:.1}% - database nearly full! Data insertion may fail soon.", 
                percentage);
            error!("Consider increasing map size in configuration or reducing data volume.");
        }
        
        Ok(())
    }

}