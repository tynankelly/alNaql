// src/utils/job_queue.rs
// Dynamic job queue manager for continuous parallel execution

use std::collections::{VecDeque, HashSet, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Instant, Duration};

use crossbeam_channel::{Sender, Receiver, bounded};
use log::{info, debug, warn, error, trace};

use crate::error::Result;
use crate::config::AlNaqlConfig;
use crate::utils::job_tracker::{ComparisonJob, JobSizeCategory};
use crate::utils::job_scheduler::JobScheduler;

/// Message types for worker communication
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    JobCompleted {
        job_id: usize,
        worker_id: usize,
        result: std::result::Result<usize, String>,  // Use std::result::Result directly
        duration: Duration,
    },
    JobStarted {
        job_id: usize,
        worker_id: usize,
    },
    WorkerIdle {
        worker_id: usize,
    },
    WorkerError {
        worker_id: usize,
        error: String,
    },
    Shutdown,
}

/// Status of a job in the queue system
#[derive(Debug, Clone, PartialEq)]
pub enum QueuedJobStatus {
    Pending,
    Active,
    Completed,
    Failed,
}
/// Resource utilization state for adaptive job selection
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResourceState {
    Low,      // <40% utilization
    Good,     // 40-70%
    High,     // 70-85%
    Critical, // >85%
}

impl ResourceState {
    /// Determine current state from memory pressure
    fn from_memory_pressure(pressure: u32) -> Self {
        match pressure {
            0..=40 => ResourceState::Low,
            41..=70 => ResourceState::Good,
            71..=85 => ResourceState::High,
            _ => ResourceState::Critical,
        }
    }
}

/// Tracks job state and metrics
#[derive(Debug, Clone)]
pub struct JobMetrics {
    pub status: QueuedJobStatus,
    pub enqueue_time: Instant,
    pub start_time: Option<Instant>,
    pub complete_time: Option<Instant>,
    pub worker_id: Option<usize>,
    pub attempts: usize,
}

/// Priority calculation result for job selection
#[derive(Debug, Clone)]
struct JobPriority {
    job: ComparisonJob,
    score: i32,
    conflict_count: usize,
    is_independent: bool,
}

/// Main job queue manager for continuous execution
pub struct JobQueue {
    // Core queue data
    pending: Arc<Mutex<VecDeque<ComparisonJob>>>,
    active: Arc<RwLock<HashSet<usize>>>,
    completed: Arc<RwLock<HashSet<usize>>>,
    failed: Arc<RwLock<HashSet<usize>>>,
    
    // Job tracking
    all_jobs: Arc<Vec<ComparisonJob>>,
    job_metrics: Arc<Mutex<HashMap<usize, JobMetrics>>>,
    
    // Scheduling
    scheduler: Arc<JobScheduler>,
    config: Arc<AlNaqlConfig>,
    
    // Worker communication
    worker_sender: Sender<WorkerMessage>,
    worker_receiver: Receiver<WorkerMessage>,
    
    // Statistics
    total_jobs: usize,
    jobs_completed: Arc<AtomicUsize>,
    jobs_failed: Arc<AtomicUsize>,
    
    // Control
    shutdown: Arc<AtomicBool>,
}

impl JobQueue {
    /// Create a new job queue with all jobs to be processed
    pub fn new(jobs: Vec<ComparisonJob>, config: &AlNaqlConfig) -> Result<Self> {
        info!("Initializing JobQueue with {} jobs", jobs.len());
        
        let total_jobs = jobs.len();
        
        // Create scheduler for conflict detection
        let scheduler = Arc::new(JobScheduler::new(jobs.clone(), config));
        
        // Initialize metrics for all jobs
        let mut job_metrics = HashMap::new();
        let now = Instant::now();
        for job in &jobs {
            job_metrics.insert(
                job.comparison_number,
                JobMetrics {
                    status: QueuedJobStatus::Pending,
                    enqueue_time: now,
                    start_time: None,
                    complete_time: None,
                    worker_id: None,
                    attempts: 0,
                }
            );
        }
        
        // Create communication channels for worker coordination
        let (worker_sender, worker_receiver) = bounded(config.execution.max_parallel_comparisons * 2);
        
        // Initialize pending queue with all jobs
        let pending = VecDeque::from(jobs.clone());
        
        Ok(JobQueue {
            pending: Arc::new(Mutex::new(pending)),
            active: Arc::new(RwLock::new(HashSet::new())),
            completed: Arc::new(RwLock::new(HashSet::new())),
            failed: Arc::new(RwLock::new(HashSet::new())),
            
            all_jobs: Arc::new(jobs),
            job_metrics: Arc::new(Mutex::new(job_metrics)),
            
            scheduler,
            config: Arc::new(config.clone()),
            
            worker_sender,
            worker_receiver,
            
            total_jobs,
            jobs_completed: Arc::new(AtomicUsize::new(0)),
            jobs_failed: Arc::new(AtomicUsize::new(0)),
            
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Get the next runnable job that doesn't conflict with active jobs
    pub fn get_next_runnable(&self) -> Option<ComparisonJob> {
        if self.shutdown.load(Ordering::Relaxed) {
            return None;
        }
        
        let active = self.active.read().unwrap();
        let mut pending = self.pending.lock().unwrap();
        
        if pending.is_empty() {
            return None;
        }
        
        // Find all runnable jobs with their priorities
        let mut candidates = Vec::new();
        
        for job in pending.iter() {
            if !self.conflicts_with_active(job, &active) {
                let priority = self.calculate_priority(job, &active);
                candidates.push(priority);
            }
        }
        
        if candidates.is_empty() {
            trace!("No runnable jobs found (all conflict with active jobs)");
            return None;
        }
        
        // Sort by priority score (higher is better)
        candidates.sort_by(|a, b| b.score.cmp(&a.score));
        
        // Get the highest priority job
        let best = candidates.into_iter().next()?;
        
        debug!(
            "Selected job {} with priority score {} (conflicts: {}, independent: {})",
            best.job.comparison_number,
            best.score,
            best.conflict_count,
            best.is_independent
        );
        
        // Remove from pending queue
        pending.retain(|j| j.comparison_number != best.job.comparison_number);
        
        Some(best.job)
    }
    
    /// Try to get a runnable job with timeout
    pub fn try_get_runnable(&self, timeout: Duration) -> Option<ComparisonJob> {
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            if let Some(job) = self.get_next_runnable() {
                return Some(job);
            }
            
            // Brief sleep to avoid busy waiting
            std::thread::sleep(Duration::from_millis(50));
            
            // Check if we should shutdown
            if self.shutdown.load(Ordering::Relaxed) {
                return None;
            }
        }
        
        None
    }
    
    /// Mark a job as active (being processed)
    pub fn mark_active(&self, job_id: usize, worker_id: usize) -> Result<()> {
        let mut active = self.active.write().unwrap();
        active.insert(job_id);
        
        // Update metrics
        let mut metrics = self.job_metrics.lock().unwrap();
        if let Some(metric) = metrics.get_mut(&job_id) {
            metric.status = QueuedJobStatus::Active;
            metric.start_time = Some(Instant::now());
            metric.worker_id = Some(worker_id);
            metric.attempts += 1;
        }
        
        debug!(
            "Job {} marked active (worker {}). Active jobs: {:?}",
            job_id, worker_id,
            active.iter().cloned().collect::<Vec<_>>()
        );
        
        Ok(())
    }
    
    /// Mark a job as complete
    pub fn mark_complete(&self, job_id: usize) -> Result<()> {
        // Remove from active
        let mut active = self.active.write().unwrap();
        active.remove(&job_id);
        
        // Add to completed
        let mut completed = self.completed.write().unwrap();
        completed.insert(job_id);
        
        // Update metrics
        let mut metrics = self.job_metrics.lock().unwrap();
        if let Some(metric) = metrics.get_mut(&job_id) {
            metric.status = QueuedJobStatus::Completed;
            metric.complete_time = Some(Instant::now());
        }
        
        // Update counter
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        
        let completed_count = self.jobs_completed.load(Ordering::Relaxed);
        let failed_count = self.jobs_failed.load(Ordering::Relaxed);
        let remaining = self.total_jobs - completed_count - failed_count;
        
        info!(
            "Job {} completed. Progress: {}/{} complete, {} failed, {} remaining, {} active",
            job_id, completed_count, self.total_jobs, failed_count, remaining,
            active.len()
        );
        
        Ok(())
    }
    
    /// Mark a job as failed
    pub fn mark_failed(&self, job_id: usize, can_retry: bool) -> Result<()> {
        // Remove from active
        let mut active = self.active.write().unwrap();
        active.remove(&job_id);
        
        if can_retry {
            // Get the job and add it back to pending queue
            if let Some(job) = self.get_job_by_id(job_id) {
                let mut pending = self.pending.lock().unwrap();
                pending.push_back(job);
                
                let mut metrics = self.job_metrics.lock().unwrap();
                if let Some(metric) = metrics.get_mut(&job_id) {
                    metric.status = QueuedJobStatus::Pending;
                    warn!("Job {} failed (attempt {}), requeueing for retry", 
                          job_id, metric.attempts);
                }
            }
        } else {
            // Add to failed set
            let mut failed = self.failed.write().unwrap();
            failed.insert(job_id);
            
            // Update metrics
            let mut metrics = self.job_metrics.lock().unwrap();
            if let Some(metric) = metrics.get_mut(&job_id) {
                metric.status = QueuedJobStatus::Failed;
                metric.complete_time = Some(Instant::now());
            }
            
            self.jobs_failed.fetch_add(1, Ordering::Relaxed);
            error!("Job {} failed permanently", job_id);
        }
        
        Ok(())
    }
    
    /// Check if all jobs are done (complete or failed)
    pub fn is_empty(&self) -> bool {
        let pending = self.pending.lock().unwrap();
        let active = self.active.read().unwrap();
        
        pending.is_empty() && active.is_empty()
    }
    
    /// Get current queue status
    pub fn get_status(&self) -> QueueStatus {
        let pending = self.pending.lock().unwrap();
        let active = self.active.read().unwrap();
        let completed = self.completed.read().unwrap();
        let failed = self.failed.read().unwrap();
        
        QueueStatus {
            pending: pending.len(),
            active: active.len(),
            completed: completed.len(),
            failed: failed.len(),
            total: self.total_jobs,
        }
    }
    
    /// Signal shutdown to all workers
    pub fn shutdown(&self) {
        info!("JobQueue shutting down");
        self.shutdown.store(true, Ordering::Relaxed);
    }
    
    /// Check if shutdown was requested
    pub fn should_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
    
    // ========== Private Helper Methods ==========
    
    /// Check if a job conflicts with any active jobs
    fn conflicts_with_active(&self, job: &ComparisonJob, active_jobs: &HashSet<usize>) -> bool {
        // Use the scheduler's public method to check conflicts
        self.scheduler.conflicts_with_any(job.comparison_number, active_jobs)
    }
    
    /// Get a job by its ID
    fn get_job_by_id(&self, job_id: usize) -> Option<ComparisonJob> {
        self.all_jobs
            .iter()
            .find(|j| j.comparison_number == job_id)
            .cloned()
    }
    
    /// Get metrics for a specific job
    pub fn get_job_metrics(&self, job_id: usize) -> Option<JobMetrics> {
        let metrics = self.job_metrics.lock().unwrap();
        metrics.get(&job_id).cloned()
    }
    
    /// Get list of active job IDs
    pub fn get_active_jobs(&self) -> Vec<usize> {
        let active = self.active.read().unwrap();
        active.iter().cloned().collect()
    }
    
    /// Get performance statistics
    pub fn get_statistics(&self) -> QueueStatistics {
        let metrics = self.job_metrics.lock().unwrap();
        
        let mut total_wait_time = Duration::ZERO;
        let mut total_run_time = Duration::ZERO;
        let mut completed_count = 0;
        
        for metric in metrics.values() {
            if metric.status == QueuedJobStatus::Completed {
                completed_count += 1;
                
                if let Some(start) = metric.start_time {
                    total_wait_time += start.duration_since(metric.enqueue_time);
                    
                    if let Some(complete) = metric.complete_time {
                        total_run_time += complete.duration_since(start);
                    }
                }
            }
        }
        
        let avg_wait_time = if completed_count > 0 {
            total_wait_time / completed_count as u32
        } else {
            Duration::ZERO
        };
        
        let avg_run_time = if completed_count > 0 {
            total_run_time / completed_count as u32
        } else {
            Duration::ZERO
        };
        
        QueueStatistics {
            total_jobs: self.total_jobs,
            completed: self.jobs_completed.load(Ordering::Relaxed),
            failed: self.jobs_failed.load(Ordering::Relaxed),
            average_wait_time: avg_wait_time,
            average_run_time: avg_run_time,
            queue_efficiency: self.calculate_efficiency(),
        }
    }
    
    /// Get a clone of the worker message sender for a worker thread
    pub fn get_worker_sender(&self) -> Sender<WorkerMessage> {
        self.worker_sender.clone()
    }
    
    /// Get the worker message receiver (should only be called once by the coordinator)
    pub fn get_worker_receiver(&self) -> &Receiver<WorkerMessage> {
        &self.worker_receiver
    }
    
    /// Send a worker message
    pub fn send_worker_message(&self, msg: WorkerMessage) -> Result<()> {
        self.worker_sender.send(msg)
            .map_err(|e| crate::error::Error::Io(
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            ))
    }
    
    /// Get a reference to the scheduler (for conflict checking)
    pub fn get_scheduler(&self) -> &JobScheduler {
        &self.scheduler
    }
    
    /// Calculate queue efficiency (0.0 - 1.0)
    fn calculate_efficiency(&self) -> f64 {
        let status = self.get_status();
        
        // Efficiency based on keeping workers busy
        let max_active = self.config.execution.max_parallel_comparisons;
        let current_active = status.active;
        
        if status.pending == 0 && current_active < max_active {
            // Queue is starved, can't keep all workers busy
            current_active as f64 / max_active as f64
        } else {
            // Queue has work, efficiency is 1.0 if all slots used
            (current_active as f64 / max_active as f64).min(1.0)
        }
    }

    /// Calculate priority score for a job with enhanced factors
    fn calculate_priority(&self, job: &ComparisonJob, active_jobs: &HashSet<usize>) -> JobPriority {
        let mut score = 0i32;
        
        // Check total conflicts (not just with active) using public method
        let conflict_count = self.scheduler.get_conflict_count(job.comparison_number);
        
        // ========== PRIORITY FACTORS ==========
        
        // 1. HIGHEST PRIORITY: Independent jobs (no conflicts at all)
        let is_independent = conflict_count == 0;
        if is_independent {
            score += 10000;  // Increased from 1000 for stronger preference
        }
        
        // 2. Fewer total conflicts is better
        // Scale the penalty based on conflict severity
        if conflict_count > 0 {
            score -= (conflict_count as i32) * 100;  // Increased penalty from 10
        }
        
        // 3. Jobs accessing different databases than active jobs
        // This rewards jobs that won't compete for the same resources
        let database_diversity_score = self.calculate_database_diversity(job, active_jobs);
        score += database_diversity_score;
        
        // 4. Job size/complexity estimation
        // Smaller jobs get higher priority to maximize throughput
        let size_score = self.calculate_size_priority(job);
        score += size_score;
        
        // 5. Jobs that have been waiting longer (prevent starvation)
        if let Ok(metrics) = self.job_metrics.lock() {
            if let Some(metric) = metrics.get(&job.comparison_number) {
                let wait_time = metric.enqueue_time.elapsed().as_secs();
                // Exponential growth to prevent starvation
                let wait_bonus = (wait_time as f64).sqrt() as i32 * 10;
                score += wait_bonus.min(500); // Cap at 500 to avoid dominance
            }
        }
        
        // 6. Failed attempts get lower priority (but not too low to prevent starvation)
        if let Ok(metrics) = self.job_metrics.lock() {
            if let Some(metric) = metrics.get(&job.comparison_number) {
                // Reduce penalty for retries to ensure they eventually run
                score -= (metric.attempts as i32 - 1) * 20; // Only penalize retries, not first attempt
            }
        }
        
        // 7. Historical performance bonus
        // Jobs from database pairs that historically complete quickly
        if let Some(historical_score) = self.get_historical_performance_score(job) {
            score += historical_score;
        }

        // 8. Resource-aware size diversity boost
        // Adapt job selection based on current memory pressure
        let resource_state = self.get_resource_state();
        let size_category = job.get_size_category();

        let resource_boost = match (resource_state, size_category) {
            // Low utilization: strongly prefer large jobs to use resources
            (ResourceState::Low, JobSizeCategory::Large) => 1500,
            (ResourceState::Low, JobSizeCategory::Medium) => 500,
            (ResourceState::Low, JobSizeCategory::Small) => -200,
            
            // Good utilization: slight preference for maintaining mix
            (ResourceState::Good, JobSizeCategory::Medium) => 200,
            (ResourceState::Good, _) => 0,
            
            // High utilization: prefer smaller jobs to avoid pressure
            (ResourceState::High, JobSizeCategory::Small) => 1000,
            (ResourceState::High, JobSizeCategory::Medium) => -200,
            (ResourceState::High, JobSizeCategory::Large) => -1000,
            
            // Critical: strongly avoid large jobs
            (ResourceState::Critical, JobSizeCategory::Small) => 2000,
            (ResourceState::Critical, JobSizeCategory::Medium) => -1000,
            (ResourceState::Critical, JobSizeCategory::Large) => -5000,
        };

        score += resource_boost;

        // Log significant resource-based adjustments
        if resource_boost.abs() >= 1000 {
            trace!(
                "Job {} (size: {:?}) got resource boost {} due to {:?} state",
                job.comparison_number, size_category, resource_boost, resource_state
            );
        }

        JobPriority {
            job: job.clone(),
            score,
            conflict_count,
            is_independent,
        }
    }
    
    /// Calculate database diversity score
    /// Rewards jobs that use different databases than currently active jobs
    fn calculate_database_diversity(&self, job: &ComparisonJob, active_jobs: &HashSet<usize>) -> i32 {
        if active_jobs.is_empty() {
            return 0;
        }
        
        let mut diversity_score = 0i32;
        let mut shares_source = false;
        let mut shares_target = false;
        
        // Check how many active jobs share databases with this job
        for &active_id in active_jobs {
            if let Some(active_job) = self.get_job_by_id(active_id) {
                if active_job.source_db_path == job.source_db_path ||
                   active_job.target_db_path == job.source_db_path {
                    shares_source = true;
                }
                if active_job.source_db_path == job.target_db_path ||
                   active_job.target_db_path == job.target_db_path {
                    shares_target = true;
                }
            }
        }
        
        // Reward jobs that don't share databases
        if !shares_source {
            diversity_score += 500;  // Bonus for unique source
        }
        if !shares_target {
            diversity_score += 500;  // Bonus for unique target
        }
        
        // Extra bonus if completely independent
        if !shares_source && !shares_target {
            diversity_score += 200;  // Additional bonus for total independence
        }
        
        diversity_score
    }
    
    /// Calculate size-based priority
    /// Smaller jobs get higher priority to maximize throughput
    fn calculate_size_priority(&self, job: &ComparisonJob) -> i32 {
        // Use the estimated complexity from ComparisonJob
        let estimated_size = job.get_estimated_complexity();
        
        // Convert to priority score (inverse relationship)
        match estimated_size {
            size if size < 1_000_000 => 400,           // Very small: high priority
            size if size < 10_000_000 => 200,          // Small: good priority  
            size if size < 100_000_000 => 0,           // Medium: neutral
            size if size < 1_000_000_000 => -200,      // Large: lower priority
            _ => -400,                                  // Very large: lowest priority
        }
    }
    
    /// Get historical performance score for similar jobs
    fn get_historical_performance_score(&self, job: &ComparisonJob) -> Option<i32> {
        // Look for completed jobs with the same database pair
        let completed = self.completed.read().ok()?;
        let metrics = self.job_metrics.lock().ok()?;
        
        let mut similar_durations = Vec::new();
        
        for &completed_id in completed.iter() {
            if let Some(completed_job) = self.get_job_by_id(completed_id) {
                // Check if it's the same database pair (in either order)
                let same_pair = (completed_job.source_db_path == job.source_db_path &&
                                completed_job.target_db_path == job.target_db_path) ||
                               (completed_job.source_db_path == job.target_db_path &&
                                completed_job.target_db_path == job.source_db_path);
                
                if same_pair {
                    if let Some(metric) = metrics.get(&completed_id) {
                        if let (Some(start), Some(end)) = (metric.start_time, metric.complete_time) {
                            let duration = end.duration_since(start).as_secs();
                            similar_durations.push(duration);
                        }
                    }
                }
            }
        }
        
        if similar_durations.is_empty() {
            return None;
        }
        
        // Calculate average duration
        let avg_duration = similar_durations.iter().sum::<u64>() / similar_durations.len() as u64;
        
        // Convert to score (faster historical times = higher priority)
        Some(match avg_duration {
            d if d < 60 => 300,        // Very fast: < 1 minute
            d if d < 300 => 150,       // Fast: < 5 minutes  
            d if d < 900 => 0,         // Medium: < 15 minutes
            d if d < 1800 => -150,     // Slow: < 30 minutes
            _ => -300,                 // Very slow: > 30 minutes
        })
    }
    /// Return a job to the pending queue (used when coordinator check fails)
    pub fn return_job(&self, job: ComparisonJob) -> Result<()> {
        let mut pending = self.pending.lock().unwrap();
        
        // Add to front of queue so it gets picked up quickly
        pending.push_front(job.clone());
        
        debug!(
            "Job {} returned to pending queue (now {} pending)",
            job.comparison_number,
            pending.len()
        );
        
        Ok(())
    }
    
    /// Get current resource state based on system memory pressure
    fn get_resource_state(&self) -> ResourceState {
        use crate::utils::resource_coordinator::SHARED_STATE;
        let pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
        ResourceState::from_memory_pressure(pressure.into())
    }
}

/// Current status of the queue
#[derive(Debug, Clone)]
pub struct QueueStatus {
    pub pending: usize,
    pub active: usize,
    pub completed: usize,
    pub failed: usize,
    pub total: usize,
}

impl QueueStatus {
    pub fn is_finished(&self) -> bool {
        self.pending == 0 && self.active == 0
    }
    
    pub fn progress_percentage(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            ((self.completed + self.failed) as f64 / self.total as f64) * 100.0
        }
    }
}

impl std::fmt::Display for QueueStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Queue Status: {} pending, {} active, {} completed, {} failed ({}% done)",
            self.pending, self.active, self.completed, self.failed,
            self.progress_percentage() as i32
        )
    }
}

/// Performance statistics for the queue
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    pub total_jobs: usize,
    pub completed: usize,
    pub failed: usize,
    pub average_wait_time: Duration,
    pub average_run_time: Duration,
    pub queue_efficiency: f64,
}

impl std::fmt::Display for QueueStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Queue Statistics:\n\
             - Total jobs: {}\n\
             - Completed: {}\n\
             - Failed: {}\n\
             - Avg wait time: {:.2}s\n\
             - Avg run time: {:.2}s\n\
             - Queue efficiency: {:.1}%",
            self.total_jobs,
            self.completed,
            self.failed,
            self.average_wait_time.as_secs_f64(),
            self.average_run_time.as_secs_f64(),
            self.queue_efficiency * 100.0
        )
    }
}
