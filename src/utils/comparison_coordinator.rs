// src/utils/comparison_coordinator.rs

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::thread;

use log::{info, debug, warn, error};

use crate::error::{Result, Error};
use crate::config::AlNaqlConfig;
use crate::config::storage::StorageConfig;
use crate::storage::{LMDBStorage, DatabasePool, open_storage};
use crate::utils::job_tracker::ComparisonJob;
use crate::utils::job_scheduler::{JobScheduler, ExecutionBatch};
use crate::utils::resource_coordinator::{SHARED_STATE};
use crate::utils::thread_allocation::{SystemState, ResourceAllocation};

/// Manages resource allocation for parallel comparison execution
pub struct ComparisonCoordinator {
    // Configuration
    max_parallel: usize,
    strategy: String,
    
    // Resource limits per comparison
    memory_per_job: usize,
    threads_per_job: usize,
    
    // Active job tracking
    active_jobs: Arc<Mutex<HashSet<usize>>>,  // job comparison_number
    
    // Database reference counting for smart scheduling
    database_refs: Arc<Mutex<HashMap<PathBuf, usize>>>,
    
    // Metrics (using atomics like your MemoryManager)
    pub completed_jobs: Arc<AtomicUsize>,
    pub failed_jobs: Arc<AtomicUsize>,
    pub total_jobs: Arc<AtomicUsize>,
    
    // Resource limits
    total_memory_mb: usize,
    total_threads: usize,
}

impl ComparisonCoordinator {
    /// Create a new coordinator from config
    pub fn new(config: &AlNaqlConfig) -> Result<Self> {
        let system_state = SystemState::assess()
            .map_err(|e| Error::Config(format!("Failed to assess system state: {}", e)))?;
        
        // Calculate available resources (reserve 20% for system)
        let available_memory = (system_state.total_memory_mb as f64 * 0.8) as usize;
        let available_threads = (system_state.cpu_cores as f64 * 0.8) as usize;
        
        // Determine max_parallel - either use configured value or auto-calculate
        let max_parallel = if config.execution.max_parallel_comparisons == 0 {
            // Auto-calculate based on resources
            let memory_based_limit = available_memory / config.execution.min_memory_per_comparison;
            let thread_based_limit = available_threads / config.execution.min_threads_per_comparison;
            
            // Take the minimum of the two limits
            let calculated = memory_based_limit.min(thread_based_limit);
            
            // Ensure at least 1, cap at reasonable maximum (e.g., 32)
            let final_value = calculated.max(1).min(32);
            
            info!("Auto-calculated max_parallel_comparisons: {}", final_value);
            info!("  Memory limit: {} MB available / {} MB per job = max {} jobs", 
                available_memory, config.execution.min_memory_per_comparison, memory_based_limit);
            info!("  Thread limit: {} threads available / {} threads per job = max {} jobs",
                available_threads, config.execution.min_threads_per_comparison, thread_based_limit);
            
            final_value
        } else {
            config.execution.max_parallel_comparisons
        };
        
        // Use config's calculation method with new optimal thread values
        let (memory_per_job, threads_per_job) = 
            config.execution.get_resources_per_comparison(available_memory, available_threads);

        // Log the thread allocation strategy being used
        info!("Thread allocation strategy:");
        info!("  Min threads per job: {}", config.execution.min_threads_per_comparison);
        info!("  Optimal threads per job: {}", config.execution.optimal_threads_per_comparison);
        info!("  Max threads per job: {}", config.execution.max_threads_per_comparison);
        info!("  Calculated threads per job: {}", threads_per_job);
        
        info!("ComparisonCoordinator initialized:");
        info!("  Max parallel comparisons: {}", max_parallel);
        info!("  Memory per comparison: {} MB", memory_per_job);
        info!("  Threads per comparison: {}", threads_per_job);
        info!("  Strategy: {}", config.execution.parallel_strategy);
        
        Ok(Self {
            max_parallel,
            strategy: config.execution.parallel_strategy.clone(),
            memory_per_job,
            threads_per_job,
            active_jobs: Arc::new(Mutex::new(HashSet::new())),
            database_refs: Arc::new(Mutex::new(HashMap::new())),
            completed_jobs: Arc::new(AtomicUsize::new(0)),
            failed_jobs: Arc::new(AtomicUsize::new(0)),
            total_jobs: Arc::new(AtomicUsize::new(0)),
            total_memory_mb: available_memory,
            total_threads: available_threads,
        })
    }
    
    /// Check if a new job can start based on resources and strategy
    pub fn can_start_job(&self, job: &ComparisonJob) -> bool {
        // Check active job count
        let active = self.active_jobs.lock().unwrap();
        if active.len() >= self.max_parallel {
            debug!("Cannot start job {}: max parallel limit reached ({}/{})", 
                job.comparison_number, active.len(), self.max_parallel);
            return false;
        }
        
        // Check memory pressure using SHARED_STATE (following your pattern)
        let memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
        if memory_pressure > 75 {
            debug!("Cannot start job {}: high memory pressure ({})", 
                job.comparison_number, memory_pressure);
            return false;
        }
        
        // Check strategy-specific constraints
        match self.strategy.as_str() {
            "smart" => {
                // Check for database conflicts
                if self.has_database_conflict(job) {
                    debug!("Cannot start job {}: database conflict", job.comparison_number);
                    return false;
                }
            },
            "pooled" => {
                // Similar to smart, but we'll allow more concurrent access with pooling
                if self.has_heavy_database_conflict(job) {
                    debug!("Cannot start job {}: heavy database usage", job.comparison_number);
                    return false;
                }
            },
            "unrestricted" => {
                // No additional checks
            },
            _ => {}
        }
        
        true
    }
    
    /// Check if databases are already in use (for smart scheduling)
    fn has_database_conflict(&self, job: &ComparisonJob) -> bool {
        let db_refs = self.database_refs.lock().unwrap();
        
        // Don't allow any concurrent access to same database
        let source_refs = db_refs.get(&job.source_db_path).copied().unwrap_or(0);
        let target_refs = db_refs.get(&job.target_db_path).copied().unwrap_or(0);
        
        source_refs > 0 || target_refs > 0
    }
    
    /// Check for heavy database usage (for pooled strategy)
    fn has_heavy_database_conflict(&self, job: &ComparisonJob) -> bool {
        let db_refs = self.database_refs.lock().unwrap();
        
        // Allow up to 2 concurrent readers per database with pooling
        let source_refs = db_refs.get(&job.source_db_path).copied().unwrap_or(0);
        let target_refs = db_refs.get(&job.target_db_path).copied().unwrap_or(0);
        
        source_refs >= 2 || target_refs >= 2
    }
    
    /// Register a job as active and update resource tracking
    pub fn register_job(&self, job: &ComparisonJob) -> JobHandle {
        debug!("Registering job {} with coordinator", job.comparison_number);
        
        // Add to active set
        {
            let mut active = self.active_jobs.lock().unwrap();
            active.insert(job.comparison_number);
        }
        
        // Update database reference counts
        {
            let mut db_refs = self.database_refs.lock().unwrap();
            *db_refs.entry(job.source_db_path.clone()).or_insert(0) += 1;
            *db_refs.entry(job.target_db_path.clone()).or_insert(0) += 1;
        }
        
        // Update SHARED_STATE atomics (following your pattern)
        SHARED_STATE.active_comparisons.fetch_add(1, Ordering::Relaxed);
        SHARED_STATE.total_threads_allocated.fetch_add(self.threads_per_job as u32, Ordering::Relaxed);
        
        info!("Job {} registered. Active comparisons: {}", 
            job.comparison_number,
            SHARED_STATE.active_comparisons.load(Ordering::Relaxed));
        
        // Return RAII handle for automatic cleanup
        JobHandle {
            job_id: job.comparison_number,
            coordinator: self.clone(),
            source_path: job.source_db_path.clone(),
            target_path: job.target_db_path.clone(),
        }
    }
    /// Try to acquire a job slot with comprehensive conflict and resource checking
    /// This is a final verification step before a worker commits to processing a job
    /// 
    /// Returns true if the job can be started immediately, false otherwise
    /// This is a read-only check that doesn't modify any state
    pub fn try_acquire_job_slot(
        &self, 
        job: &ComparisonJob, 
        active_jobs: &HashSet<usize>,
        scheduler: &JobScheduler
    ) -> bool {
        // ========== STEP 1: Basic Resource Checks ==========
        // These are fast checks that don't require locks
        
        // Check memory pressure first (cheapest check)
        let memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
        if memory_pressure > 80 {  // Slightly higher threshold for final check
            debug!(
                "try_acquire_job_slot: Job {} rejected - critical memory pressure ({}%)", 
                job.comparison_number, memory_pressure
            );
            return false;
        }
        
        // Check if we're at max parallel limit
        // Note: We check against the active_jobs parameter, not our internal state
        // This ensures we're using the most recent information
        if active_jobs.len() >= self.max_parallel {
            debug!(
                "try_acquire_job_slot: Job {} rejected - at parallel limit ({}/{})",
                job.comparison_number, active_jobs.len(), self.max_parallel
            );
            return false;
        }
        
        // ========== STEP 2: Conflict Verification ==========
        // Use the scheduler to check for conflicts with the current active set
        if scheduler.conflicts_with_any(job.comparison_number, active_jobs) {
            debug!(
                "try_acquire_job_slot: Job {} rejected - conflicts with active jobs {:?}",
                job.comparison_number, active_jobs
            );
            return false;
        }
        
        // ========== STEP 3: Strategy-Specific Database Checks ==========
        // These checks ensure database access patterns align with our strategy
        match self.strategy.as_str() {
            "smart" => {
                // Smart strategy: No concurrent access to same database
                let db_refs = self.database_refs.lock().unwrap();
                
                let source_refs = db_refs.get(&job.source_db_path).copied().unwrap_or(0);
                let target_refs = db_refs.get(&job.target_db_path).copied().unwrap_or(0);
                
                if source_refs > 0 {
                    debug!(
                        "try_acquire_job_slot: Job {} rejected - source db {} in use ({} refs)",
                        job.comparison_number, 
                        job.source_db_path.display(),
                        source_refs
                    );
                    return false;
                }
                
                if target_refs > 0 {
                    debug!(
                        "try_acquire_job_slot: Job {} rejected - target db {} in use ({} refs)",
                        job.comparison_number,
                        job.target_db_path.display(), 
                        target_refs
                    );
                    return false;
                }
            },
            
            "pooled" => {
                // Pooled strategy: Allow limited concurrent access (max 2 per database)
                let db_refs = self.database_refs.lock().unwrap();
                
                let source_refs = db_refs.get(&job.source_db_path).copied().unwrap_or(0);
                let target_refs = db_refs.get(&job.target_db_path).copied().unwrap_or(0);
                
                // Check if either database is oversubscribed
                const MAX_POOLED_REFS: usize = 2;
                
                if source_refs >= MAX_POOLED_REFS {
                    debug!(
                        "try_acquire_job_slot: Job {} rejected - source db {} oversubscribed ({}/{} refs)",
                        job.comparison_number,
                        job.source_db_path.display(),
                        source_refs,
                        MAX_POOLED_REFS
                    );
                    return false;
                }
                
                if target_refs >= MAX_POOLED_REFS {
                    debug!(
                        "try_acquire_job_slot: Job {} rejected - target db {} oversubscribed ({}/{} refs)",
                        job.comparison_number,
                        job.target_db_path.display(),
                        target_refs,
                        MAX_POOLED_REFS
                    );
                    return false;
                }
            },
            
            "unrestricted" => {
                // Unrestricted mode: No database-specific checks
                // Conflicts were already checked above, that's sufficient
            },
            
            _ => {
                // Unknown strategy - default to safe behavior (smart mode)
                warn!("try_acquire_job_slot: Unknown strategy '{}', using smart mode checks", self.strategy);
                
                let db_refs = self.database_refs.lock().unwrap();
                if db_refs.contains_key(&job.source_db_path) || db_refs.contains_key(&job.target_db_path) {
                    return false;
                }
            }
        }
        
        // ========== STEP 4: Final System State Check ==========
        // One last check for any rapid state changes
        
        // Re-verify our internal active job tracking matches expectations
        {
            let internal_active = self.active_jobs.lock().unwrap();
            
            // If internal tracking significantly differs from parameter, something is wrong
            let diff = (internal_active.len() as i32 - active_jobs.len() as i32).abs();
            if diff > 1 {  // Allow for 1 job difference due to race conditions
                warn!(
                    "try_acquire_job_slot: Job {} - active job mismatch (internal: {}, param: {})",
                    job.comparison_number,
                    internal_active.len(),
                    active_jobs.len()
                );
                // Don't reject, but log this concerning state
            }
            
            // Ensure this specific job isn't somehow already active
            if internal_active.contains(&job.comparison_number) {
                error!(
                    "try_acquire_job_slot: Job {} is already active! This shouldn't happen",
                    job.comparison_number
                );
                return false;
            }
        }
        
        // ========== SUCCESS: All checks passed ==========
        debug!(
            "try_acquire_job_slot: Job {} approved for execution (strategy: {}, active: {})",
            job.comparison_number,
            self.strategy,
            active_jobs.len()
        );
        
        true
    }

    /// Register a job for continuous execution mode
    /// This is a thin wrapper around register_job that adds worker tracking
    pub fn register_job_continuous(
        &self, 
        job: &ComparisonJob, 
        worker_id: usize
    ) -> JobHandle {
        debug!(
            "Registering job {} for continuous execution (worker {})",
            job.comparison_number,
            worker_id
        );
        
        // Could add worker-specific tracking here if needed in the future
        // For now, just delegate to the standard register_job
        self.register_job(job)
    }
    /// Release a job (called by JobHandle drop)
    fn release_job(&self, job_id: usize, source_path: &PathBuf, target_path: &PathBuf) {
        debug!("Releasing job {} from coordinator", job_id);
        
        // Remove from active set
        {
            let mut active = self.active_jobs.lock().unwrap();
            active.remove(&job_id);
        }
        
        // Decrement database reference counts
        {
            let mut db_refs = self.database_refs.lock().unwrap();
            
            if let Some(count) = db_refs.get_mut(source_path) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    db_refs.remove(source_path);
                }
            }
            
            if let Some(count) = db_refs.get_mut(target_path) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    db_refs.remove(target_path);
                }
            }
        }
        
        // Update SHARED_STATE
        SHARED_STATE.active_comparisons.fetch_sub(1, Ordering::Relaxed);
        SHARED_STATE.total_threads_allocated.fetch_sub(self.threads_per_job as u32, Ordering::Relaxed);
        
        info!("Job {} released. Active comparisons: {}", 
            job_id,
            SHARED_STATE.active_comparisons.load(Ordering::Relaxed));
    }
    
    /// Wait for a slot to become available
    pub fn wait_for_slot(&self, job: &ComparisonJob, timeout: Duration) -> bool {
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            if self.can_start_job(job) {
                return true;
            }
            
            // Brief sleep to avoid busy-waiting
            thread::sleep(Duration::from_millis(100));
        }
        
        false
    }
    
    /// Get current resource allocation for logging
    pub fn get_resource_status(&self) -> String {
        let active = self.active_jobs.lock().unwrap();
        let db_refs = self.database_refs.lock().unwrap();
        
        format!(
            "Active: {}/{}, Completed: {}, Failed: {}, DBs in use: {}",
            active.len(),
            self.max_parallel,
            self.completed_jobs.load(Ordering::Relaxed),
            self.failed_jobs.load(Ordering::Relaxed),
            db_refs.len()
        )
    }
    
    /// Get per-job resource allocation
    pub fn get_threads_per_job(&self) -> usize {
        self.threads_per_job
    }
    
    /// Get per-job memory allocation
    pub fn get_memory_per_job(&self) -> usize {
        self.memory_per_job
    }

    /// Check if we should use database pooling based on strategy
    pub fn should_use_pool(&self) -> bool {
        self.strategy == "pooled" || 
        (self.strategy == "smart" && self.max_parallel > 2)
    }
    
    /// Get a database handle (pooled or direct)
    pub fn get_database(
        &self,
        path: &Path,
        pool: Option<&Arc<DatabasePool>>,
        config: &StorageConfig,
    ) -> Result<Arc<LMDBStorage>> {
        if let Some(pool) = pool {
            // Use pool if available
            pool.get_or_open(path)
        } else {
            // Open directly
            Ok(Arc::new(open_storage(path, config.clone())?))
        }
    }

    /// Create execution batches from jobs
    pub fn create_execution_batches(
        &self,
        jobs: Vec<ComparisonJob>,
        config: &AlNaqlConfig,
    ) -> Vec<ExecutionBatch> {
        let scheduler = JobScheduler::new(jobs, config);
        scheduler.get_execution_order(self.max_parallel)
    }
    
    /// Check if job can run given current state
    pub fn can_run_with_current(&self, job: &ComparisonJob, _running_jobs: &HashSet<usize>) -> bool {
        // First check basic resources
        if !self.can_start_job(job) {
            return false;
        }
        
        // For smart scheduling, check for conflicts with running jobs
        if self.strategy == "smart" {
            let db_refs = self.database_refs.lock().unwrap();
            
            // Check if databases are in use by other jobs
            if db_refs.get(&job.source_db_path).copied().unwrap_or(0) > 0 ||
               db_refs.get(&job.target_db_path).copied().unwrap_or(0) > 0 {
                return false;
            }
        }
        
        true
    }
    /// Get resource allocation for a job
    pub fn get_resource_allocation(&self, config: &AlNaqlConfig) -> ResourceAllocation {
        ResourceAllocation::parallel(
            &config.execution,
            self.memory_per_job,
            self.threads_per_job
        )
    }
}

// Implement Clone to allow sharing between threads
impl Clone for ComparisonCoordinator {
    fn clone(&self) -> Self {
        Self {
            max_parallel: self.max_parallel,
            strategy: self.strategy.clone(),
            memory_per_job: self.memory_per_job,
            threads_per_job: self.threads_per_job,
            active_jobs: self.active_jobs.clone(),
            database_refs: self.database_refs.clone(),
            completed_jobs: self.completed_jobs.clone(),
            failed_jobs: self.failed_jobs.clone(),
            total_jobs: self.total_jobs.clone(),
            total_memory_mb: self.total_memory_mb,
            total_threads: self.total_threads,
        }
    }
}

/// RAII handle for automatic job cleanup
pub struct JobHandle {
    job_id: usize,
    coordinator: ComparisonCoordinator,
    source_path: PathBuf,
    target_path: PathBuf,
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        self.coordinator.release_job(self.job_id, &self.source_path, &self.target_path);
    }
}
