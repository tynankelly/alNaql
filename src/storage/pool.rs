// src/storage/pool.rs

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use lru::LruCache;
use log::{info, debug, warn};

use crate::error::{Result, Error};
use crate::storage::{LMDBStorage, open_storage, StorageConfig};

/// Pool of reusable database handles
pub struct DatabasePool {
    /// LRU cache of open databases
    cache: Arc<Mutex<LruCache<PathBuf, Arc<LMDBStorage>>>>,
    
    /// Reference counting for each database
    references: Arc<Mutex<HashMap<PathBuf, usize>>>,
    
    /// Track last access time for cleanup
    last_access: Arc<Mutex<HashMap<PathBuf, Instant>>>,
    
    /// Configuration
    max_cached: usize,
    max_readers_per_db: usize,
    
    /// Storage config for opening new databases
    storage_config: StorageConfig,
}

impl DatabasePool {
    /// Create a new database pool
    pub fn new(max_cached: usize, storage_config: StorageConfig) -> Self {
        info!("Creating DatabasePool with max_cached={}", max_cached);
        
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(max_cached).unwrap()
            ))),
            references: Arc::new(Mutex::new(HashMap::new())),
            last_access: Arc::new(Mutex::new(HashMap::new())),
            max_cached,
            max_readers_per_db: 4,  // Allow up to 4 concurrent readers per database
            storage_config,
        }
    }
    
    /// Get or open a database handle
    pub fn get_or_open(&self, path: &Path) -> Result<Arc<LMDBStorage>> {
        let path_buf = path.to_path_buf();
        
        // Check if we've hit the reader limit for this database
        {
            let refs = self.references.lock().unwrap();
            if let Some(&count) = refs.get(&path_buf) {
                if count >= self.max_readers_per_db {
                    warn!("DatabasePool: Reader limit ({}) reached for {:?}", 
                        self.max_readers_per_db, path);
                    return Err(Error::Storage(format!(
                        "Maximum concurrent readers ({}) reached for database {:?}", 
                        self.max_readers_per_db, path
                    )));
                }
            }
        }
        
        // First, try to get from cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(storage) = cache.get(&path_buf) {
                // Found in cache, increment reference count
                let mut refs = self.references.lock().unwrap();
                let ref_count = refs.entry(path_buf.clone()).or_insert(0);
                *ref_count += 1;
                
                // Update last access time
                let mut last_access = self.last_access.lock().unwrap();
                last_access.insert(path_buf.clone(), Instant::now());
                
                debug!("DatabasePool: Returning cached handle for {:?} (refs: {})", 
                    path, ref_count);
                
                return Ok(storage.clone());
            }
        }
        
        // Not in cache, need to open new database
        info!("DatabasePool: Opening new database at {:?}", path);
        
        // Open the database using your existing function
        let storage = Arc::new(open_storage(path, self.storage_config.clone())?);
        
        // Add to cache
        {
            let mut cache = self.cache.lock().unwrap();
            
            // Check if we're at capacity and need to evict
            if cache.len() >= self.max_cached {
                // LRU will automatically evict the least recently used
                debug!("DatabasePool: Cache at capacity, LRU eviction will occur");
            }
            
            cache.put(path_buf.clone(), storage.clone());
        }
        
        // Initialize reference count
        {
            let mut refs = self.references.lock().unwrap();
            refs.insert(path_buf.clone(), 1);
        }
        
        // Set initial access time
        {
            let mut last_access = self.last_access.lock().unwrap();
            last_access.insert(path_buf.clone(), Instant::now());
        }
        
        Ok(storage)
    }
    
    /// Try to get a database handle with a timeout
    pub fn try_get_or_open(&self, path: &Path, timeout: Duration) -> Result<Arc<LMDBStorage>> {
        let start = Instant::now();
        let path_buf = path.to_path_buf();
        
        loop {
            // Check current reference count
            let current_refs = {
                let refs = self.references.lock().unwrap();
                refs.get(&path_buf).copied().unwrap_or(0)
            };
            
            // If under limit, try to get the handle
            if current_refs < self.max_readers_per_db {
                match self.get_or_open(path) {
                    Ok(handle) => return Ok(handle),
                    Err(e) => {
                        // If it's not a reader limit error, return it
                        if !e.to_string().contains("Maximum concurrent readers") {
                            return Err(e);
                        }
                        // Otherwise, continue waiting
                    }
                }
            }
            
            // Check timeout
            if start.elapsed() >= timeout {
                return Err(Error::Storage(format!(
                    "Timeout waiting for database handle for {:?} ({}ms)", 
                    path, timeout.as_millis()
                )));
            }
            
            // Brief sleep before retry
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    
    /// Release a database handle (decrement reference count)
    pub fn release(&self, path: &Path) {
        let path_buf = path.to_path_buf();
        
        let mut refs = self.references.lock().unwrap();
        
        if let Some(count) = refs.get_mut(&path_buf) {
            *count = count.saturating_sub(1);
            
            debug!("DatabasePool: Released handle for {:?} (refs: {})", path, *count);
            
            // Don't remove from cache even if refs reach 0
            // Let LRU handle eviction based on usage patterns
            if *count == 0 {
                refs.remove(&path_buf);
                debug!("DatabasePool: No active references for {:?}, keeping in cache for reuse", path);
            }
        }
    }
    
    /// Check if a database is currently in use
    pub fn is_in_use(&self, path: &Path) -> bool {
        let refs = self.references.lock().unwrap();
        refs.get(&path.to_path_buf()).copied().unwrap_or(0) > 0
    }
    
    /// Check if database has reached reader limit
    pub fn is_at_reader_limit(&self, path: &Path) -> bool {
        let refs = self.references.lock().unwrap();
        refs.get(&path.to_path_buf()).copied().unwrap_or(0) >= self.max_readers_per_db
    }
    
    /// Get current reference count for a database
    pub fn get_reference_count(&self, path: &Path) -> usize {
        let refs = self.references.lock().unwrap();
        refs.get(&path.to_path_buf()).copied().unwrap_or(0)
    }
    
    /// Set maximum readers per database
    pub fn set_max_readers(&mut self, max_readers: usize) {
        self.max_readers_per_db = max_readers;
        info!("DatabasePool: Set max_readers_per_db to {}", max_readers);
    }
    
    /// Get maximum readers per database
    pub fn get_max_readers(&self) -> usize {
        self.max_readers_per_db
    }
    
    /// Clear all cached databases (for cleanup)
    pub fn clear(&self) -> Result<()> {
        info!("DatabasePool: Clearing all cached databases");
        
        // Check if any databases are still in use
        {
            let refs = self.references.lock().unwrap();
            if !refs.is_empty() {
                let in_use: Vec<_> = refs.iter()
                    .filter(|(_, count)| **count > 0)
                    .map(|(path, count)| format!("{:?}: {}", path, count))
                    .collect();
                
                if !in_use.is_empty() {
                    warn!("DatabasePool: Clearing while databases still in use: {:?}", in_use);
                }
            }
        }
        
        // Clear the cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.clear();
        }
        
        // Clear reference counts
        {
            let mut refs = self.references.lock().unwrap();
            refs.clear();
        }
        
        // Clear access times
        {
            let mut last_access = self.last_access.lock().unwrap();
            last_access.clear();
        }
        
        info!("DatabasePool: Cleared successfully");
        Ok(())
    }
    
    /// Get pool statistics for monitoring
    pub fn get_stats(&self) -> PoolStats {
        let cache = self.cache.lock().unwrap();
        let refs = self.references.lock().unwrap();
        
        let total_cached = cache.len();
        let total_in_use = refs.values().sum::<usize>();
        let databases_with_refs = refs.len();
        
        PoolStats {
            total_cached,
            total_in_use,
            databases_with_refs,
            max_capacity: self.max_cached,
            max_readers_per_db: self.max_readers_per_db,
        }
    }
    
    /// Clean up databases that haven't been accessed recently
    pub fn cleanup_stale(&self, max_age: Duration) -> usize {
        let mut cleaned = 0;
        let now = Instant::now();
        
        // Find stale entries
        let stale_entries: Vec<PathBuf> = {
            let refs = self.references.lock().unwrap();
            let last_access = self.last_access.lock().unwrap();
            
            last_access.iter()
                .filter(|(path, last)| {
                    // Only clean if not in use and older than max_age
                    refs.get(*path).copied().unwrap_or(0) == 0 
                        && now.duration_since(**last) > max_age
                })
                .map(|(path, _)| path.clone())
                .collect()
        };
        
        // Remove stale entries from cache
        if !stale_entries.is_empty() {
            let mut cache = self.cache.lock().unwrap();
            for path in &stale_entries {
                cache.pop(path);
                cleaned += 1;
                debug!("DatabasePool: Removed stale entry {:?}", path);
            }
            
            // Also remove from last_access
            let mut last_access = self.last_access.lock().unwrap();
            for path in &stale_entries {
                last_access.remove(path);
            }
        }
        
        if cleaned > 0 {
            info!("DatabasePool: Cleaned {} stale entries", cleaned);
        }
        
        cleaned
    }
}

/// Statistics about the database pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_cached: usize,
    pub total_in_use: usize,
    pub databases_with_refs: usize,
    pub max_capacity: usize,
    pub max_readers_per_db: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, 
            "DatabasePool: {}/{} cached, {} in use, {} with refs, max {} readers/db",
            self.total_cached,
            self.max_capacity,
            self.total_in_use,
            self.databases_with_refs,
            self.max_readers_per_db
        )
    }
}

/// RAII guard for database handle from pool
pub struct PooledDatabase {
    storage: Arc<LMDBStorage>,
    pool: Arc<DatabasePool>,
    path: PathBuf,
}

impl PooledDatabase {
    pub fn new(storage: Arc<LMDBStorage>, pool: Arc<DatabasePool>, path: PathBuf) -> Self {
        Self { storage, pool, path }
    }
    
    /// Get the underlying storage
    pub fn storage(&self) -> &Arc<LMDBStorage> {
        &self.storage
    }
}

impl Drop for PooledDatabase {
    fn drop(&mut self) {
        // Automatically release when dropped
        self.pool.release(&self.path);
    }
}

impl std::ops::Deref for PooledDatabase {
    type Target = LMDBStorage;
    
    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

/// Create a database pool from configuration
pub fn create_pool_from_config(config: &crate::AlNaqlConfig) -> Arc<DatabasePool> {
    let max_cached = config.execution.max_parallel_comparisons * 2; // Cache 2x parallel comparisons
    let pool = DatabasePool::new(max_cached, config.storage.clone());
    Arc::new(pool)
}