// pipeline.rs - Optimized implementation
use std::time::{Instant, Duration};
use std::thread;
use log::{info, debug, warn, trace};
use rayon::prelude::*;
use std::sync::Mutex;
use ahash::AHashMap;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use crate::error::Result;
use crate::types::NGramEntry;
use crate::matcher::types::NGramMatch;
use crate::matcher::parallel::ParallelMatcher;
use crate::storage::StorageBackend;

/// Strategies for accessing storage from pipeline workers
pub enum StorageAccessStrategy {
    /// Each worker uses the same storage instance
    Shared,
    
    /// Each worker gets a connection pool entry
    Pooled(usize), // pool size
    
    /// Each worker creates its own storage connection
    PerWorker,
}

/// Processing pipeline for handling very large text matching workloads efficiently
pub struct ProcessingPipeline {
    /// Number of ngrams to process in each chunk
    chunk_size: usize,
    
    /// Capacity of the internal result collection
    result_capacity: usize,
    
    /// Number of worker threads to spawn
    num_workers: usize,
    
    /// Memory limit per worker (0 for no limit)
    memory_limit_per_worker: usize,
    
    /// Strategy for how workers access storage
    storage_access_strategy: StorageAccessStrategy,
    
    /// Enable memory pressure monitoring
    enable_memory_checks: bool,
    
    /// Connection pool for the Pooled strategy (initialized lazily)
    connection_pool: Option<Mutex<AHashMap<usize, usize>>>,
    
    /// Worker connections map for the PerWorker strategy (not fully implemented)
    worker_connections: Mutex<AHashMap<usize, usize>>,
}

#[allow(dead_code)]
struct StorageConnectionPool<'a> {
    connections: Vec<&'a dyn StorageBackend>,
    next_conn_idx: Mutex<usize>,
}

impl<'a> StorageConnectionPool<'a> {
    #[allow(dead_code)]
    fn new(size: usize, storage: &'a dyn StorageBackend) -> Self {
        // For now, we're using the same storage reference for all connections
        // In a real implementation, we would create multiple independent connections
        let connections = vec![storage; size];
        
        Self {
            connections,
            next_conn_idx: Mutex::new(0),
        }
    }
    
    #[allow(dead_code)]
    fn get_connection(&self) -> Result<&'a dyn StorageBackend> {
        let mut idx = self.next_conn_idx.lock().unwrap();
        
        // Round-robin allocation of connections
        let conn = self.connections[*idx];
        *idx = (*idx + 1) % self.connections.len();
        
        Ok(conn)
    }
}

impl ProcessingPipeline {
    pub fn new(chunk_size: usize, result_capacity: usize, workers: usize) -> Self {
        Self {
            chunk_size,
            result_capacity,
            num_workers: workers,
            memory_limit_per_worker: 0, // No limit by default
            storage_access_strategy: StorageAccessStrategy::Shared,
            enable_memory_checks: true,
            connection_pool: None,
            worker_connections: Mutex::new(AHashMap::new()),
        }
    }
    
    /// Sets a memory limit per worker
    pub fn with_memory_limit(mut self, limit_per_worker: usize) -> Self {
        self.memory_limit_per_worker = limit_per_worker;
        self
    }
    
    /// Sets the storage access strategy
    pub fn with_storage_strategy(mut self, strategy: StorageAccessStrategy) -> Self {
        self.storage_access_strategy = strategy;
        self
    }
    
    /// Disables memory checks for maximum performance
    pub fn without_memory_checks(mut self) -> Self {
        self.enable_memory_checks = false;
        self
    }
    
    /// Creates a pipeline with settings optimized for the given workload and system
    pub fn with_optimal_settings(source_size: usize, system_memory: usize) -> Self {
        // Calculate optimal settings based on workload and system resources
        let cpu_count = num_cpus::get();
        
        // Adjust chunk size based on data size
        let chunk_size = if source_size < 10_000 {
            // For smaller datasets, use relatively larger chunks
            1000
        } else if source_size < 100_000 {
            // For medium datasets, use moderate chunks
            750
        } else {
            // For large datasets, use smaller chunks
            500
        };
        
        // Determine worker count based on CPU count and memory constraints
        let memory_per_cpu = system_memory / cpu_count;
        
        // Use more workers when memory allows
        let worker_count = if memory_per_cpu > 1024 * 1024 * 1024 {
            // More than 1GB per CPU, can use more workers
            cpu_count.min(16)
        } else {
            // Limited memory, use fewer workers
            cpu_count.min(8)
        };
        
        // Estimate result capacity
        let result_capacity = source_size / 5; // Estimate ~20% match rate
        
        // Create the pipeline with these settings
        Self {
            chunk_size,
            result_capacity,
            num_workers: worker_count,
            memory_limit_per_worker: 0,
            storage_access_strategy: StorageAccessStrategy::Shared,
            enable_memory_checks: true,
            connection_pool: None,
            worker_connections: Mutex::new(AHashMap::new()),
        }
    }

    /// Process ngrams using the pipeline strategy
    pub fn process<'a>(
        &self,
        source_ngrams: &[NGramEntry],
        matcher: &'a ParallelMatcher<'a>,
        storage: &'a dyn StorageBackend
    ) -> Result<Vec<NGramMatch>> {
        let start_time = Instant::now();
        info!("Starting pipeline processing for {} source ngrams", source_ngrams.len());
        
        // Create a counter to track progress
        let processed_counter = Arc::new(AtomicUsize::new(0));
        let total_ngrams = source_ngrams.len();
        
        // Calculate an adaptive chunk size based on workload and system resources
        let adaptive_chunk_size = self.calculate_adaptive_chunk_size(source_ngrams);
        
        // Create chunks using the adaptive size
        let chunks: Vec<&[NGramEntry]> = source_ngrams.chunks(adaptive_chunk_size).collect();
        let chunks_total = chunks.len();
        
        info!("Processing {} ngrams in {} chunks of approximately {} ngrams each", 
             total_ngrams, chunks_total, adaptive_chunk_size);
        
        // Configure rayon thread pool if needed
        let pool = if self.num_workers > 0 && self.num_workers != rayon::current_num_threads() {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_workers)
                .stack_size(48 * 1024 * 1024)  // Increased stack size for complex operations
                .build() {
                Ok(pool) => Some(pool),
                Err(e) => {
                    warn!("Failed to create custom thread pool: {}. Using default pool.", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Use arc to share the reference to self across threads
        let self_arc = Arc::new(self);
        
        // Function to process a chunk and update progress
        let process_chunk = |chunk_idx: usize, chunk: &[NGramEntry]| -> Result<Vec<NGramMatch>> {
            let chunk_start = Instant::now();
            trace!("Starting processing of chunk {}/{} with {} ngrams", 
                 chunk_idx + 1, chunks_total, chunk.len());
            
            // Check memory pressure if enabled
            if self_arc.enable_memory_checks && self_arc.check_memory_pressure() {
                debug!("Memory pressure detected before processing chunk {}, applying backpressure", chunk_idx + 1);
                thread::sleep(Duration::from_millis(150)); // Slightly longer pause
            }
            
            // Get the appropriate storage for this worker/chunk
            let worker_id = rayon::current_thread_index().unwrap_or(chunk_idx);
            let worker_storage = match self_arc.get_worker_storage(worker_id, storage) {
                Ok(s) => s,
                Err(e) => {
                    warn!("Failed to get worker storage: {}, falling back to shared storage", e);
                    storage
                }
            };
            
            // Process the chunk with appropriate storage
            let result = matcher.find_matches_for_chunk(chunk, worker_storage);
            
            // Update progress counter
            let processed = processed_counter.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
            let progress_pct = (processed as f64 / total_ngrams as f64) * 100.0;
            
            // Log progress periodically
            if chunk_idx % 5 == 0 || chunk_idx == chunks_total - 1 {
                info!("Pipeline: Processed {}/{} ngrams ({:.1}%) - Chunk {} completed in {:?}",
                     processed, total_ngrams, progress_pct, chunk_idx + 1, chunk_start.elapsed());
            }
            
            // Check memory pressure again after processing
            if self_arc.enable_memory_checks && self_arc.check_memory_pressure() {
                debug!("Memory pressure detected after processing chunk {}, applying backpressure", chunk_idx + 1);
                thread::sleep(Duration::from_millis(150));
            }
            
            result
        };
        
        // Process chunks in parallel using the configured thread pool
        let results = if let Some(ref pool) = pool {
            // Use custom thread pool
            pool.install(|| {
                chunks.par_iter()
                    .enumerate()
                    .map(|(idx, chunk)| process_chunk(idx, chunk))
                    .collect::<Result<Vec<_>>>()
            })
        } else {
            // Use default rayon thread pool
            chunks.par_iter()
                .enumerate()
                .map(|(idx, chunk)| process_chunk(idx, chunk))
                .collect::<Result<Vec<_>>>()
        }?;
        
        // Merge results efficiently - precompute total capacity
        let total_matches: usize = results.iter().map(|r| r.len()).sum();
        let mut all_matches = Vec::with_capacity(total_matches.max(self.result_capacity));
        
        // Extend with all results
        for chunk_matches in results {
            all_matches.extend(chunk_matches);
        }
        
        // Deduplicate if needed
        if all_matches.len() > 1 {
            let before_count = all_matches.len();
            
            // Sort for deduplication
            all_matches.sort_by(|a, b| {
                a.source_position.cmp(&b.source_position)
                    .then(a.target_position.cmp(&b.target_position))
            });
            
            // Remove adjacent duplicates
            all_matches.dedup_by(|a, b| {
                a.source_position == b.source_position && 
                a.target_position == b.target_position
            });
            
            let after_count = all_matches.len();
            if after_count < before_count {
                info!("Pipeline: Removed {} duplicate matches", before_count - after_count);
            }
        }
        
        let elapsed = start_time.elapsed();
        info!("Pipeline processing complete: found {} matches in {:?}",
            all_matches.len(), elapsed);
        
        Ok(all_matches)
    }
    
    // Helper method to calculate an adaptive chunk size based on workload and system resources
    fn calculate_adaptive_chunk_size(&self, source_ngrams: &[NGramEntry]) -> usize {
        if source_ngrams.is_empty() {
            return self.chunk_size; // Use the default size
        }
        
        // Get system information
        let available_memory = match sys_info::mem_info() {
            Ok(info) => info.avail as usize * 1024, // Convert KB to bytes
            Err(_) => 1024 * 1024 * 1024, // Default to 1GB
        };
        
        let cpu_count = num_cpus::get();
        
        // Sample ngrams to get average size
        let sample_size = source_ngrams.len().min(100);
        let avg_ngram_size = if sample_size > 0 {
            source_ngrams.iter()
                .take(sample_size)
                .map(|ng| ng.text_content.cleaned_text.len())
                .sum::<usize>() / sample_size
        } else {
            50 // Default size if empty
        };
        
        // Calculate memory-aware chunk size
        let memory_based_size = if avg_ngram_size > 0 {
            // Use about 5% of available memory per chunk
            ((available_memory / 20) / avg_ngram_size).min(15_000) // Increased from 10_000
        } else {
            self.chunk_size
        };
        
        // Calculate CPU-based chunk size
        let cpu_based_size = if source_ngrams.len() < 10_000 {
            // For smaller datasets, use larger chunks
            (source_ngrams.len() / cpu_count.max(1)).max(500) // Increased from 100
        } else {
            // For larger datasets, use smaller chunks for better load balancing
            (source_ngrams.len() / (cpu_count * 3).max(1)).max(750) // Increased from 500, reduced divisor
        };
        
        // Use the smaller of the two calculated sizes, but keep within reasonable bounds
        let adaptive_size = memory_based_size.min(cpu_based_size).max(500); // Increased from 100
        
        // Keep within reasonable bounds relative to the configured size
        let min_size = (self.chunk_size / 3).max(500); // Changed from divide by 4, increased minimum
        let max_size = self.chunk_size * 4;
        
        adaptive_size.clamp(min_size, max_size)
    }
    
    // Check system memory usage
    fn check_memory_pressure(&self) -> bool {
        if let Ok(mem_info) = sys_info::mem_info() {
            let available_percent = (mem_info.avail as f64 / mem_info.total as f64) * 100.0;
            let memory_pressure = 100.0 - available_percent;
            
            // If memory pressure is high (less than 15% free), return true
            if memory_pressure > 85.0 {
                warn!("High memory pressure detected: {:.1}% memory used", memory_pressure);
                return true;
            }
        }
        false
    }
    
    // Get the appropriate storage for a worker thread
    fn get_worker_storage<'a>(
        &self,
        worker_id: usize,
        storage: &'a dyn StorageBackend
    ) -> Result<&'a dyn StorageBackend> {
        match &self.storage_access_strategy {
            StorageAccessStrategy::Shared => {
                // Simplest strategy: all workers use the same storage reference
                trace!("Worker {} using shared storage connection", worker_id);
                Ok(storage)
            },
            
            StorageAccessStrategy::Pooled(pool_size) => {
                let pool_size = *pool_size;
                
                // Simplified implementation: just calculate connection index
                // without using the mutex for now
                let conn_idx = worker_id % pool_size;
                
                debug!("Worker {} assigned to connection pool slot {}/{}", 
                      worker_id, conn_idx, pool_size);
                
                // Update the connection map if we want to track it
                if let Some(pool) = &self.connection_pool {
                    let mut pool_map = pool.lock().unwrap();
                    pool_map.insert(worker_id, conn_idx);
                }
                
                // In reality, we would use the index to get a specific connection
                // For now, just return the shared storage
                Ok(storage)
            },
            
            StorageAccessStrategy::PerWorker => {
                // For the PerWorker strategy, we would create and store connections
                // in the worker_connections map
                debug!("Worker {} would use dedicated connection (not implemented)", worker_id);
                
                // Mark this worker as having requested a connection
                let mut connections = self.worker_connections.lock().unwrap();
                connections.entry(worker_id).or_insert(1);
                
                Ok(storage)
            }
        }
    }
}