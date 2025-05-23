use std::cell::RefCell;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use ahash::AHashMap;
use lazy_static::lazy_static;
use log::{debug, info, trace};

use crate::storage::NGramMetadata;

// Thread-local cache with size limit and timestamp tracking
thread_local! {
    static LOCAL_METADATA_CACHE: RefCell<ThreadLocalCache> = 
        RefCell::new(ThreadLocalCache::new(2000)); // More conservative size limit
}

// Structure to manage thread-local cache with size limits
struct ThreadLocalCache {
    data: AHashMap<u64, (Arc<NGramMetadata>, Instant)>, // Added timestamp
    max_size: usize,
}

impl ThreadLocalCache {
    fn new(max_size: usize) -> Self {
        Self {
            data: AHashMap::with_capacity(max_size / 2), // Start with half capacity
            max_size,
        }
    }

    fn get(&self, key: &u64) -> Option<&Arc<NGramMetadata>> {
        self.data.get(key).map(|(metadata, _)| metadata)
    }

    fn insert(&mut self, key: u64, value: Arc<NGramMetadata>) {
        // Check if we need to evict
        if self.data.len() >= self.max_size {
            self.evict();
        }
        
        // Insert with current timestamp
        self.data.insert(key, (value, Instant::now()));
    }

    fn evict(&mut self) {
        // Simple strategy: evict 25% of oldest entries
        let to_evict = self.max_size / 4;
        if to_evict == 0 {
            return;
        }
    
        // Collect entries sorted by age
        let mut entries: Vec<_> = self.data.iter().collect();
        entries.sort_by_key(|(_, (_, timestamp))| *timestamp);
    
        // First collect the keys to remove
        let keys_to_remove: Vec<u64> = entries.iter()
            .take(to_evict)
            .map(|(key, _)| **key)
            .collect();
    
        // Then remove them in a separate loop
        for key in keys_to_remove {
            self.data.remove(&key);
        }
    
        trace!("Thread-local cache: evicted {} oldest entries", to_evict);
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn clear(&mut self) {
        self.data.clear();
    }
}

// Statistics tracking for monitoring cache performance
#[derive(Default)]
struct CacheStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
    evictions: AtomicUsize,
    thread_local_evictions: AtomicUsize,
    db_lookups: AtomicUsize,
}

// Rest of the MetadataCache implementation
pub struct MetadataCache {
    // Main cache with read-write lock
    shared_cache: RwLock<AHashMap<u64, (Arc<NGramMetadata>, Instant)>>,
    // Configuration
    max_size: usize,
    ttl: Duration,
    stats: CacheStats,
    // New: thread-local cache size
    thread_local_max_size: usize,
}

impl MetadataCache {
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        // Use 2000 as default thread-local cache size
        Self::with_thread_local_size(max_size, ttl_seconds, 2000)
    }
    
    pub fn with_thread_local_size(max_size: usize, ttl_seconds: u64, thread_local_size: usize) -> Self {
        debug!("Initializing metadata cache with max_size={}, ttl={}s, thread_local_size={}", 
               max_size, ttl_seconds, thread_local_size);
        Self {
            shared_cache: RwLock::new(AHashMap::with_capacity(max_size)),
            max_size,
            ttl: Duration::from_secs(ttl_seconds),
            stats: CacheStats::default(),
            thread_local_max_size: thread_local_size,
        }
    }

    pub fn get(&self, key: u64) -> Option<Arc<NGramMetadata>> {
        // First check thread-local cache (no locking required)
        let mut found = None;
        
        // Initialize thread-local cache with the configured size if needed
        LOCAL_METADATA_CACHE.with(|cache| {
            // We can safely initialize this every time because it's only created once per thread
            let cache_ref = &mut *cache.borrow_mut();
            
            // Ensure the thread-local cache has the right size configuration
            if cache_ref.max_size != self.thread_local_max_size {
                debug!("Updating thread-local cache size: {} -> {}", 
                      cache_ref.max_size, self.thread_local_max_size);
                cache_ref.max_size = self.thread_local_max_size;
            }
            
            // Rest of the method stays the same...
            if let Some(metadata) = cache_ref.get(&key) {
                found = Some(metadata.clone());
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!("Thread-local cache hit for sequence {}", key);
            }
        });
        
        if found.is_some() {
            return found;
        }
        
        // If not in thread-local cache, try shared cache
        let shared_read = self.shared_cache.read().unwrap();
        if let Some((metadata, timestamp)) = shared_read.get(&key) {
            // Check if entry is still valid
            if timestamp.elapsed() < self.ttl {
                // Update thread-local cache
                let metadata_clone = metadata.clone();
                LOCAL_METADATA_CACHE.with(|cache| {
                    cache.borrow_mut().insert(key, metadata_clone.clone());
                });
                
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!("Shared cache hit for sequence {}", key);
                return Some(metadata.clone());
            }
            // If entry expired, let it fall through to a miss
            trace!("Shared cache expired entry for sequence {}", key);
        }
        
        // Cache miss
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        trace!("Cache miss for sequence {}", key);
        None
    }

    pub fn insert(&self, key: u64, metadata: NGramMetadata) {
        let metadata_arc = Arc::new(metadata);
        
        // Update thread-local cache
        LOCAL_METADATA_CACHE.with(|cache| {
            cache.borrow_mut().insert(key, metadata_arc.clone());
        });
        
        // Update shared cache with eviction policy
        let mut shared_write = self.shared_cache.write().unwrap();
        
        // Evict if necessary
        if shared_write.len() >= self.max_size {
            self.evict_entries(&mut shared_write);
        }
        
        // Insert new entry
        shared_write.insert(key, (metadata_arc, Instant::now()));
        trace!("Inserted metadata for sequence {} into cache", key);
    }
    
    fn evict_entries(&self, cache: &mut AHashMap<u64, (Arc<NGramMetadata>, Instant)>) {
        // Evict expired entries and oldest entries if necessary
        let mut to_remove = Vec::new();
        
        // First pass: mark expired entries
        for (key, (_, timestamp)) in cache.iter() {
            if timestamp.elapsed() > self.ttl {
                to_remove.push(*key);
            }
        }
        
        // If not enough expired entries, evict oldest
        if to_remove.len() < self.max_size / 5 {
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, (_, timestamp))| *timestamp);
            
            // Take the oldest 20% entries
            let additional_evictions = (self.max_size / 5).saturating_sub(to_remove.len());
            to_remove.extend(entries.iter().take(additional_evictions).map(|(k, _)| **k));
        }
        
        // Perform evictions
        let evicted_count = to_remove.len();
        for key in to_remove {
            cache.remove(&key);
        }
        
        if evicted_count > 0 {
            self.stats.evictions.fetch_add(evicted_count, Ordering::Relaxed);
            debug!("Evicted {} entries from metadata cache", evicted_count);
        }
    }
    
    pub fn clear_thread_local(&self) {
        LOCAL_METADATA_CACHE.with(|cache| {
            let len = cache.borrow().len();
            cache.borrow_mut().clear();
            debug!("Cleared thread-local metadata cache ({} entries)", len);
        });
    }
    
    pub fn record_db_lookup(&self) {
        self.stats.db_lookups.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_db_batch_lookup(&self, count: usize) {
        self.stats.db_lookups.fetch_add(count, Ordering::Relaxed);
    }
    
    pub fn print_stats(&self) {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
        let db_lookups = self.stats.db_lookups.load(Ordering::Relaxed);
        let db_avoided = if total > 0 { hits as f64 / total as f64 * db_lookups as f64 } else { 0.0 };
        
        info!("Metadata cache stats: {} hits, {} misses ({:.1}% hit rate), {} evictions, {} DB lookups ({:.0} avoided)",
            hits, misses, hit_rate, 
            self.stats.evictions.load(Ordering::Relaxed),
            db_lookups, db_avoided);
        
        // Get info about the shared cache size
        let shared_size = self.shared_cache.read().unwrap().len();
        info!("Shared metadata cache: {}/{} entries ({:.1}% full)", 
            shared_size, self.max_size, (shared_size as f64 / self.max_size as f64) * 100.0);
        
        // Thread-local cache stats - best effort reporting
        let mut thread_cache_sizes = Vec::new();
        LOCAL_METADATA_CACHE.with(|cache| {
            thread_cache_sizes.push(cache.borrow().len());
        });
        
        let avg_thread_size = if !thread_cache_sizes.is_empty() {
            thread_cache_sizes.iter().sum::<usize>() as f64 / thread_cache_sizes.len() as f64
        } else {
            0.0
        };
        
        let thread_evictions = self.stats.thread_local_evictions.load(Ordering::Relaxed);
        
        info!("Thread-local metadata cache: {:.1} average entries per thread, {} total evictions",
            avg_thread_size, thread_evictions);
    }
    
    // New method to periodically clean up thread local caches
    pub fn trim_thread_local_caches(&self, age_threshold: Duration) {
        LOCAL_METADATA_CACHE.with(|cache| {
            let mut cache_mut = cache.borrow_mut();
            let before_size = cache_mut.len();
            
            // Remove entries older than the threshold
            let mut to_remove = Vec::new();
            for (key, (_, timestamp)) in &cache_mut.data {
                if timestamp.elapsed() > age_threshold {
                    to_remove.push(*key);
                }
            }
            
            for key in to_remove {
                cache_mut.data.remove(&key);
            }
            
            let removed = before_size - cache_mut.len();
            if removed > 0 {
                self.stats.thread_local_evictions.fetch_add(removed, Ordering::Relaxed);
                debug!("Trimmed thread-local cache: removed {} old entries", removed);
            }
        });
    }
}

// Create a global instance with more conservative settings
// 500k entries with 5-minute TTL and 2000 thread-local entries
lazy_static! {
    pub static ref METADATA_CACHE: MetadataCache = MetadataCache::with_thread_local_size(500_000, 300, 2000);
}