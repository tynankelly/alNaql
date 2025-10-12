// storage/lmdb/maintenance.rs

use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;
use log::{info, debug, warn, trace, error};
use lmdb_rkv::{Transaction, Cursor};

use crate::error::{Error, Result};
use super::LMDBStorage;

/// Detailed statistics for database size calculation
#[derive(Debug, Clone)]
pub struct DatabaseSizeStats {
    /// Total bytes used across all databases (actual data)
    pub total_bytes_used: u64,
    /// Total number of entries across all databases
    pub total_entries: u64,
    /// Number of entries per database
    pub entries_per_db: Vec<(String, u64)>,
    /// Actual data size per database (sum of key + value sizes)
    pub data_size_per_db: Vec<(String, u64)>,
    /// Page-level statistics from LMDB
    pub page_stats: PageStats,
    /// Estimated optimal size for compaction (with buffer)
    pub optimal_compact_size: usize,
}

/// Page-level statistics from LMDB environment
#[derive(Debug, Clone)]
pub struct PageStats {
    /// Size of each page in bytes
    pub page_size: u64,
    /// Total number of pages in use
    pub pages_used: u64,
    /// Branch pages (internal B-tree nodes)
    pub branch_pages: u64,
    /// Leaf pages (contain actual data)
    pub leaf_pages: u64,
    /// Overflow pages (for large values)
    pub overflow_pages: u64,
    /// Total map size
    pub map_size: u64,
    /// Percentage of map in use
    pub usage_percentage: f64,
}

impl LMDBStorage {
    
    /// Prepare database for bulk loading operations
    /// Uses scoped transactions for verification only
    pub fn prepare_for_bulk_loading(&mut self) -> Result<()> {
        info!("Preparing LMDB database for bulk loading at {:?}", self.db_path);
        
        // Step 1: Verify environment state before starting bulk operations
        let verification_start = Instant::now();
        
        // Use read scope for verification - following established pattern
        let entry_count = self.with_read_scope(|scope| {
            debug!("Verifying database state before bulk loading");
            
            // Get database handle first (borrow checker pattern)
            let main_db = scope.storage().sequences_db;
            
            // Create cursor for verification
            let mut cursor = scope.txn().open_ro_cursor(main_db)
                .map_err(|e| Error::storage(format!(
                    "Failed to open cursor for verification: {}", e
                )))?;
            
            // Check first few entries to see if database is populated
            let mut count = 0;
            for result in cursor.iter_start() {
                if result.is_ok() {
                    count += 1;
                }
                
                // Only check a few entries to avoid lengthy verification
                if count >= 10 {
                    break;
                }
            }
            
            Ok(count)
        })?;
        
        debug!("Database verification completed in {:?}, found {} entries", 
            verification_start.elapsed(), entry_count);
        
        // Step 2: Set bulk loading flag (atomic operation, no transaction needed)
        self.is_bulk_loading.store(true, Ordering::Release);
        
        // Step 3: Clear caches to start with maximum available memory
        {
            let mut prefix_cache = self.prefix_cache.write();
            let prefix_size = prefix_cache.len();
            prefix_cache.clear();
            debug!("Cleared {} entries from prefix cache", prefix_size);
            
            let mut metadata_cache = self.metadata_cache.write();
            let metadata_size = metadata_cache.len();
            metadata_cache.clear();
            debug!("Cleared {} entries from metadata cache", metadata_size);
        }
        
        // Step 4: Log initial state for debugging (no transaction needed)
        let env = self.env_handle.environment();
        if let Ok(env_stat) = env.stat() {
            debug!("Database prepared for bulk loading:");
            debug!("  Page size: {} bytes", env_stat.page_size());
            debug!("  Current entries: {}", env_stat.entries());
            debug!("  Current pages used: {}", 
                env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages());
        }
        
        info!("Database ready for bulk loading operations");
        Ok(())
    }

    /// Finalize bulk loading and return to normal operation mode
    /// No transactions needed - only state management and sync
    pub fn finalize_bulk_loading(&mut self) -> Result<()> {
        let operation_start = Instant::now();
        info!("Finalizing bulk loading for database at {:?}", self.db_path);
        
        // Get environment through the handle
        let env = self.env_handle.environment();
        
        // Step 1: Force sync to disk for durability
        let sync_start = Instant::now();
        match env.sync(true) {
            Ok(_) => {
                info!("Successfully synchronized database to disk after bulk loading in {:?}", 
                    sync_start.elapsed());
            },
            Err(e) => {
                warn!("Failed to sync database after bulk loading: {}", e);
                // Continue despite sync error - we can still transition state
            }
        }
        
        // Step 2: Transition out of bulk loading mode (atomic operation)
        self.is_bulk_loading.store(false, Ordering::Release);
        
        // Step 3: Clear all caches to start fresh in normal mode
        // This is for CORRECTNESS, not memory management - we want clean state
        {
            debug!("Clearing caches after bulk loading for clean state");
            
            let mut prefix_cache = self.prefix_cache.write();
            let prefix_cache_size = prefix_cache.len();
            prefix_cache.clear();
            
            let mut metadata_cache = self.metadata_cache.write();
            let metadata_cache_size = metadata_cache.len();
            metadata_cache.clear();
            
            debug!("Cleared {} prefix entries and {} metadata entries from caches", 
                prefix_cache_size, metadata_cache_size);
        }
        
        // Step 4: Reset memory usage tracking (kept for API compatibility)
        self.current_memory_usage.store(0, Ordering::Relaxed);
        
        // Step 5: Log final database statistics
        if let Ok(env_stat) = env.stat() {
            info!("Bulk loading completed:");
            info!("  Final entries: {}", env_stat.entries());
            info!("  Final pages used: {}", 
                env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages());
            
            // Calculate final map usage
            let total_pages = self.map_size / env_stat.page_size() as usize;
            let pages_used = env_stat.branch_pages() + env_stat.leaf_pages() + env_stat.overflow_pages();
            let usage_percent = (pages_used as f64 * 100.0) / total_pages as f64;
            info!("  Map usage: {:.1}%", usage_percent);
        }
        
        // Log completion with performance metrics
        let total_duration = operation_start.elapsed();
        info!("Bulk loading finalization completed in {:?}", total_duration);
        info!("Database ready for normal operation");
        
        Ok(())
    }

    /// Finish bulk loading and prepare database for normal operation
    /// This method transitions the database from high-speed bulk loading mode
    /// back to normal operation mode with full safety guarantees
    pub fn finish_bulk_loading(&mut self) -> Result<()> {
        // Check if we're actually in bulk loading mode
        // The atomic load ensures we get the current state safely
        if !self.is_bulk_loading.load(Ordering::Acquire) {
            debug!("finish_bulk_loading called but not in bulk loading mode");
            return Ok(());
        }

        info!("Finishing LMDB bulk loading for database at {:?}", self.db_path);
        let operation_start = Instant::now();

        // Step 1: Force sync to disk for durability
        // CRITICAL FIX: Use env.sync(true) directly instead of transaction commit
        // With NO_SYNC flag, transaction commits don't sync to disk
        let env = self.env_handle.environment();
        let sync_start = Instant::now();
        
        match env.sync(true) {
            Ok(_) => {
                info!("Successfully synchronized database to disk in {:?}", 
                    sync_start.elapsed());
            },
            Err(e) => {
                // Sync failure is critical - data may not be persistent
                error!("Failed to sync database after bulk loading: {}", e);
                return Err(Error::Database(format!(
                    "Failed to sync database after bulk loading: {}", e
                )));
            }
        }
        
        // Step 2: Gather statistics about what we loaded
        // This helps verify the bulk load was successful
        let stats_result = self.with_read_scope(|scope| {
            // Get the environment through the scope for statistics
            let env = scope.storage().env_handle.environment();
            
            if let Ok(env_stat) = env.stat() {
                debug!("LMDB environment status after bulk loading:");
                debug!("  Page size: {} bytes", env_stat.page_size());
                debug!("  Entries: {}", env_stat.entries());
                
                // Now get database-specific statistics for the main sequences database
                let db_stat = scope.txn().stat(self.sequences_db)
                    .map_err(|e| Error::Database(format!("Failed to get database stats: {}", e)))?;
                
                let branch_pages = db_stat.branch_pages() as u64;
                let leaf_pages = db_stat.leaf_pages() as u64;
                let overflow_pages = db_stat.overflow_pages() as u64;
                let page_size = env_stat.page_size() as u64;
                
                // Calculate how much space we're using
                let pages_used = branch_pages + leaf_pages + overflow_pages;
                let bytes_used = pages_used * page_size;
                
                info!("Database statistics after bulk loading:");
                info!("  Total entries: {}", env_stat.entries());
                info!("  Database size: {} MB", bytes_used / (1024 * 1024));
                info!("  Pages used: {} (branch: {}, leaf: {}, overflow: {})", 
                    pages_used, branch_pages, leaf_pages, overflow_pages);
                
                // Calculate and warn about map usage
                // This is important because LMDB has a fixed map size
                if self.map_size > 0 {
                    let usage_percent = (bytes_used as f64 * 100.0) / self.map_size as f64;
                    if usage_percent > 80.0 {
                        warn!("LMDB map usage is high after bulk loading: {:.1}%", usage_percent);
                        warn!("Consider increasing map_size in configuration if you plan to add more data");
                    } else {
                        info!("  Map usage: {:.1}% of {} MB", 
                            usage_percent, self.map_size / (1024 * 1024));
                    }
                }
            }
            
            Ok(())
        });
        
        // Log any issues with statistics gathering, but don't fail the operation
        if let Err(e) = stats_result {
            warn!("Failed to gather statistics after bulk loading: {}", e);
        }
        
        // Step 3: Clean up stale readers
        // During heavy bulk loading, reader slots can sometimes get stuck
        let reader_cleanup_start = Instant::now();
        match self.env_handle.cleanup_stale_readers() {
            Ok(cleaned_count) => {
                if cleaned_count > 0 {
                    info!("Cleaned up {} stale readers after bulk loading", cleaned_count);
                }
                debug!("Reader cleanup completed in {:?}", reader_cleanup_start.elapsed());
            },
            Err(e) => {
                // Reader cleanup failure is concerning but not fatal
                // The system can continue, though performance might degrade
                warn!("Failed to clean up stale readers after bulk loading: {}", e);
                warn!("This may impact performance - consider manual cleanup");
            }
        }
        
        // Step 4: Transition out of bulk loading mode
        // This atomic store ensures all threads see the change immediately
        self.is_bulk_loading.store(false, Ordering::Release);
        info!("Bulk loading mode disabled - write operations will now use normal safety flags");
        
        // Step 5: Clear all caches to start fresh
        // Bulk loading may have bypassed normal cache updates, so we clear everything
        // to ensure caches are rebuilt correctly during normal operation
        {
            let prefix_cache_size = self.prefix_cache.read().len();
            let metadata_cache_size = self.metadata_cache.read().len();
            
            if prefix_cache_size > 0 || metadata_cache_size > 0 {
                debug!("Clearing caches after bulk loading:");
                debug!("  Prefix cache entries to clear: {}", prefix_cache_size);
                debug!("  Metadata cache entries to clear: {}", metadata_cache_size);
                
                self.prefix_cache.write().clear();
                self.metadata_cache.write().clear();
                
                info!("Cleared {} cache entries total", prefix_cache_size + metadata_cache_size);
            }
        }
        
        // Step 6: Reset memory usage tracking
        // During bulk loading, memory tracking might not be accurate
        // Reset it so normal operations start with correct accounting
        let previous_usage = self.current_memory_usage.swap(0, Ordering::Relaxed);
        if previous_usage > 0 {
            debug!("Reset memory usage counter (was {} bytes)", previous_usage);
        }
        
        // Step 7: Quick verification that prefix storage is intact
        // This is a sanity check to ensure our indexing worked
        debug!("Running quick verification of prefix storage integrity");
        match self.verify_prefix_storage() {
            Ok(_) => debug!("Prefix storage verification passed"),
            Err(e) => warn!("Prefix storage verification failed: {}. Search functionality may be impaired.", e),
        }
        
        // Log completion with timing information
        let total_duration = operation_start.elapsed();
        info!("Bulk loading finalization completed in {:?}", total_duration);
        
        // Provide guidance if the operation took a long time
        if total_duration > Duration::from_secs(10) {
            info!("Note: Bulk loading finalization took over 10 seconds.");
            info!("This is normal for large datasets, but consider:");
            info!("  - Running this during low-usage periods");
            info!("  - Monitoring system resources during bulk operations");
        }
        
        info!("Database at {:?} ready for normal operation", self.db_path);
        Ok(())
    }

    /// Perform basic cleanup - sync to disk only
    /// Simplified version without memory management
    pub fn cleanup(&mut self) -> Result<()> {
        trace!("Running cleanup on LMDB database at {:?}", self.db_path);
        
        // Get the environment through the handle
        let env = self.env_handle.environment();
        
        // Sync database to disk to ensure durability
        let sync_start = Instant::now();
        match env.sync(true) {
            Ok(_) => {
                debug!("Successfully synchronized LMDB environment to disk in {:?}", 
                    sync_start.elapsed());
            },
            Err(e) => {
                warn!("Failed to sync LMDB environment to disk: {}", e);
                // Return error as sync failure in cleanup is more serious
                return Err(Error::Database(format!("Cleanup sync failed: {}", e)));
            }
        }
        
        trace!("LMDB cleanup completed");
        Ok(())
    }

    /// Get comprehensive size statistics for database compaction
    pub fn get_size_statistics(&self) -> Result<DatabaseSizeStats> {
        info!("Gathering comprehensive database size statistics");
        
        // Get environment statistics
        let env = self.env_handle.environment();
        let env_stat = env.stat()
            .map_err(|e| Error::Database(format!("Failed to get environment stats: {}", e)))?;
        let env_info = env.info()
            .map_err(|e| Error::Database(format!("Failed to get environment info: {}", e)))?;
        
        // Calculate page-level stats
        let page_size = env_stat.page_size() as u64;
        let total_entries = env_stat.entries() as u64;
        
        // Initialize tracking variables
        let mut entries_per_db = Vec::new();
        let mut data_size_per_db = Vec::new();
        let mut total_data_bytes = 0u64;
        let mut total_branch_pages = 0u64;
        let mut total_leaf_pages = 0u64;
        let mut total_overflow_pages = 0u64;
        
        // Gather stats for each database
        self.with_read_scope(|scope| {
            // List of all databases with their names
            let databases = vec![
                ("sequences", self.sequences_db),
                ("text_bytes", self.text_bytes_db),
                ("char_frequencies", self.char_frequencies_db),
                ("position", self.position_db),
                ("metadata", self.metadata_db),
                ("ngram_metadata", self.ngram_metadata_db),
                ("prefix_index", self.prefix_index_db),
                ("prefix_bytes", self.prefix_bytes_db),
                ("percentile", self.percentile_db),
            ];
            
            for (name, db) in databases {
                // Get database-specific statistics
                let db_stat = scope.txn().stat(db)
                    .map_err(|e| Error::Database(format!("Failed to get stats for {}: {}", name, e)))?;
                
                let db_entries = db_stat.entries() as u64;
                entries_per_db.push((name.to_string(), db_entries));
                
                // Accumulate page statistics
                total_branch_pages += db_stat.branch_pages() as u64;
                total_leaf_pages += db_stat.leaf_pages() as u64;
                total_overflow_pages += db_stat.overflow_pages() as u64;
                
                // Calculate actual data size
                let data_size = if db_entries == 0 {
                    0
                } else if db_entries > 100_000 {
                    // For large databases, sample to estimate size
                    self.estimate_database_size(scope.txn(), db, name)?
                } else {
                    // For smaller databases, do actual calculation
                    self.calculate_database_size(scope.txn(), db, name)?
                };
                
                data_size_per_db.push((name.to_string(), data_size));
                total_data_bytes += data_size;
                
                debug!("Database '{}': {} entries, {} bytes data, {} branch pages, {} leaf pages", 
                    name, db_entries, data_size, 
                    db_stat.branch_pages(), db_stat.leaf_pages());
            }
            
            Ok(())
        })?;
        
        // Calculate page usage
        let pages_used = total_branch_pages + total_leaf_pages + total_overflow_pages;
        let bytes_used = pages_used * page_size;
        let map_size = env_info.map_size() as u64;
        let usage_percentage = if map_size > 0 {
            (bytes_used as f64 / map_size as f64) * 100.0
        } else {
            0.0
        };
        
        // Calculate optimal compact size
        // Use actual data size + 20% buffer, but ensure minimum size
        let optimal_compact_size = ((total_data_bytes as f64 * 1.2) as usize)
            .max(100 * 1024 * 1024); // Minimum 100MB
        
        // Ensure page alignment (4KB boundary)
        let optimal_compact_size = ((optimal_compact_size + 4095) / 4096) * 4096;
        
        info!("Statistics complete:");
        info!("  Total data size: {} MB", total_data_bytes / (1024 * 1024));
        info!("  Total entries: {}", total_entries);
        info!("  Pages used: {} ({} MB)", pages_used, bytes_used / (1024 * 1024));
        info!("  Current map size: {} MB", map_size / (1024 * 1024));
        info!("  Map usage: {:.1}%", usage_percentage);
        info!("  Optimal compact size: {} MB", optimal_compact_size / (1024 * 1024));
        info!("  Space savings potential: {} MB", 
            (map_size as isize - optimal_compact_size as isize) / (1024 * 1024));
        
        Ok(DatabaseSizeStats {
            total_bytes_used: total_data_bytes,
            total_entries,
            entries_per_db,
            data_size_per_db,
            page_stats: PageStats {
                page_size,
                pages_used,
                branch_pages: total_branch_pages,
                leaf_pages: total_leaf_pages,
                overflow_pages: total_overflow_pages,
                map_size,
                usage_percentage,
            },
            optimal_compact_size,
        })
    }
    
    /// Calculate exact size of a database by iterating all entries
    fn calculate_database_size(
        &self,
        txn: &lmdb_rkv::RoTransaction,
        db: lmdb_rkv::Database,
        db_name: &str,
    ) -> Result<u64> {
        trace!("Calculating exact size for database '{}'", db_name);
        let mut total_size = 0u64;
        let mut entry_count = 0u64;
        
        let mut cursor = txn.open_ro_cursor(db)
            .map_err(|e| Error::Database(format!("Failed to open cursor for {}: {}", db_name, e)))?;
        
        for result in cursor.iter() {
            let (key, value) = result
                .map_err(|e| Error::Database(format!("Failed to read from {}: {}", db_name, e)))?;
            total_size += key.len() as u64 + value.len() as u64;
            entry_count += 1;
            
            // Log progress for large databases
            if entry_count % 10000 == 0 {
                trace!("  Processed {} entries from '{}'...", entry_count, db_name);
            }
        }
        
        trace!("Database '{}': {} entries, {} bytes total", db_name, entry_count, total_size);
        Ok(total_size)
    }
    
    /// Estimate database size by sampling entries
    fn estimate_database_size(
        &self,
        txn: &lmdb_rkv::RoTransaction,
        db: lmdb_rkv::Database,
        db_name: &str,
    ) -> Result<u64> {
        trace!("Estimating size for large database '{}'", db_name);
        
        let db_stat = txn.stat(db)
            .map_err(|e| Error::Database(format!("Failed to get stats for {}: {}", db_name, e)))?;
        
        let total_entries = db_stat.entries() as u64;
        if total_entries == 0 {
            return Ok(0);
        }
        
        // Sample up to 1000 entries to estimate average size
        const SAMPLE_SIZE: u64 = 1000;
        let mut sample_total = 0u64;
        let mut sample_count = 0u64;
        
        let mut cursor = txn.open_ro_cursor(db)
            .map_err(|e| Error::Database(format!("Failed to open cursor for {}: {}", db_name, e)))?;
        
        // Calculate skip interval for even sampling
        let skip_interval = (total_entries / SAMPLE_SIZE).max(1);
        let mut current_pos = 0u64;
        
        for result in cursor.iter() {
            let (key, value) = result
                .map_err(|e| Error::Database(format!("Failed to read from {}: {}", db_name, e)))?;
            
            // Sample at intervals
            if current_pos % skip_interval == 0 && sample_count < SAMPLE_SIZE {
                sample_total += key.len() as u64 + value.len() as u64;
                sample_count += 1;
            }
            
            current_pos += 1;
            
            // Stop if we've collected enough samples
            if sample_count >= SAMPLE_SIZE {
                break;
            }
        }
        
        // Extrapolate to full database
        if sample_count > 0 {
            let avg_entry_size = sample_total / sample_count;
            let estimated_total = avg_entry_size * total_entries;
            
            debug!("Database '{}': Sampled {} entries, avg size {} bytes, estimated total {} MB", 
                db_name, sample_count, avg_entry_size, estimated_total / (1024 * 1024));
            
            Ok(estimated_total)
        } else {
            warn!("Failed to sample any entries from '{}'", db_name);
            Ok(0)
        }
    }
    
    /// Get a quick summary of database sizes (simpler version for quick checks)
    pub fn get_storage_summary(&self) -> Result<(u64, f64)> {
        let env = self.env_handle.environment();
        let env_stat = env.stat()
            .map_err(|e| Error::Database(format!("Failed to get environment stats: {}", e)))?;
        let env_info = env.info()
            .map_err(|e| Error::Database(format!("Failed to get environment info: {}", e)))?;
        
        let page_size = env_stat.page_size() as u64;
        let map_size = env_info.map_size() as u64;
        
        // Get total pages by summing all databases
        let mut total_pages = 0u64;
        
        self.with_read_scope(|scope| {
            let databases = vec![
                self.char_frequencies_db, self.position_db, self.metadata_db,
                self.ngram_metadata_db, self.prefix_index_db, self.prefix_bytes_db,
                self.percentile_db,
            ];
            
            for db in databases {
                if let Ok(db_stat) = scope.txn().stat(db) {
                    total_pages += db_stat.branch_pages() as u64;
                    total_pages += db_stat.leaf_pages() as u64;
                    total_pages += db_stat.overflow_pages() as u64;
                }
            }
            Ok(())
        })?;
        
        let bytes_used = total_pages * page_size;
        let usage_percentage = if map_size > 0 {
            (bytes_used as f64 / map_size as f64) * 100.0
        } else {
            0.0
        };
        
        debug!("Quick summary: {} MB used of {} MB ({:.1}%)", 
            bytes_used / (1024 * 1024),
            map_size / (1024 * 1024),
            usage_percentage);
        
        Ok((bytes_used, usage_percentage))
    }
}
