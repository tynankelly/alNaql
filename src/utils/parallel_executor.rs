// src/utils/parallel_executor.rs

use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use std::time::Duration;

use log::{info, debug, warn, error};
use rayon::prelude::*;

use crate::error::Result;
use crate::config::{AlNaqlConfig, JobConfig};
use crate::utils::{
    job_tracker::{JobTracker, ComparisonJob, JobStatus},
    comparison_coordinator::ComparisonCoordinator,
    resource_coordinator::memory_manager::MemoryManager,
    progress_manager::ProgressManager,
};
use crate::storage::pool::create_pool_from_config;


/// Simple parallel execution without batching (for unrestricted mode)
pub fn execute_parallel_unrestricted(
    jobs: Vec<ComparisonJob>,
    config: &Arc<AlNaqlConfig>,
    tracker: &mut JobTracker,
    memory_manager: &Arc<MemoryManager>,
    job_config: &JobConfig,
    progress_manager: Arc<ProgressManager>,
    process_comparison_fn: impl Fn(ComparisonJob, &Arc<AlNaqlConfig>, &mut JobTracker, usize, &Arc<MemoryManager>, &JobConfig, Arc<ProgressManager>) -> std::result::Result<usize, String> + Send + Sync,) -> Result<()> {
    
    info!("Starting UNRESTRICTED parallel execution - all {} jobs at once", jobs.len());
    
    // Create pool for unrestricted mode too if enabled
    let db_pool = if config.execution.should_enable_pooling() {
        info!("Initializing database pool for unrestricted mode");
        Some(create_pool_from_config(config))
    } else {
        None
    };
    
    // Just run all jobs in parallel without scheduling
    let tracker = Arc::new(Mutex::new(tracker));
    let process_fn = Arc::new(process_comparison_fn);
    
    jobs.par_iter().enumerate().for_each(|(idx, job)| {
        info!("Starting job {} in unrestricted mode", job.comparison_number);

        // Start progress tracking
        let job_name = format!("{} vs {}",
            job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
            job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
        );
        progress_manager.start_job(job.comparison_number, job_name)
            .unwrap_or_else(|e| warn!("Failed to start progress tracking: {}", e));
        
        // Update status
        {
            let mut t = tracker.lock().unwrap();
            if let Some(tracked_job) = t.get_job_mut(idx) {
                tracked_job.status = JobStatus::InProgress;
            }
            let _ = t.save();
        }
        
        // Set pool flag if using pool
        if db_pool.is_some() {
            std::env::set_var("ALNAQL_USE_DB_POOL", "true");
        } else {
            std::env::remove_var("ALNAQL_USE_DB_POOL");
        }
        
        // Run the job
        let result = {
            let mut t = tracker.lock().unwrap();
            process_fn(
                job.clone(),
                config,
                &mut **t,
                idx,
                memory_manager,
                job_config,
                progress_manager.clone(),
            )
        };
        
        // Update result
        {
            let mut t = tracker.lock().unwrap();
            if let Some(tracked_job) = t.get_job_mut(idx) {
                match result {
                    Ok(count) => {
                        tracked_job.status = JobStatus::Completed;
                        tracked_job.match_count = Some(count);
                        info!("Job {} completed: {} matches", job.comparison_number, count);
                        
                        // Update progress manager
                        if count == usize::MAX {
                            progress_manager.early_escape_job(job.comparison_number, "High similarity detected".to_string())
                                .unwrap_or_else(|e| warn!("Failed to update progress: {}", e));
                        } else {
                            progress_manager.complete_job(job.comparison_number, count)
                                .unwrap_or_else(|e| warn!("Failed to complete job progress: {}", e));
                        }
                    },
                    Err(e) => {
                        tracked_job.status = JobStatus::Failed;
                        tracked_job.error_message = Some(e.clone());
                        error!("Job {} failed: {}", job.comparison_number, 
                            tracked_job.error_message.as_ref().unwrap());
                        
                        // Update progress manager
                        progress_manager.fail_job(job.comparison_number, e)
                            .unwrap_or_else(|e| warn!("Failed to mark job failed in progress: {}", e));
                    }
                }
            }
            let _ = t.save();
        }
    });
    
    // Clean up pool if used
    if let Some(pool) = db_pool {
        info!("Final pool stats (unrestricted): {}", pool.get_stats());
        pool.clear()?;
    }
    
    Ok(())
}

// Add this to src/utils/parallel_executor.rs

use crate::utils::job_queue::{JobQueue, WorkerMessage};
use std::thread;
use std::collections::HashSet;
use crossbeam_channel::Sender;
use log::trace;
/// Execute jobs continuously with a fixed number of workers
/// This maintains exactly N parallel jobs at all times instead of processing in batches
pub fn execute_parallel_continuous(
    jobs: Vec<ComparisonJob>,
    config: &Arc<AlNaqlConfig>,
    tracker_arg: &mut JobTracker,  // Renamed to avoid shadowing
    memory_manager: &Arc<MemoryManager>,
    job_config: &JobConfig,
    progress_manager: Arc<ProgressManager>,
    process_comparison_fn: impl Fn(ComparisonJob, &Arc<AlNaqlConfig>, &mut JobTracker, usize, &Arc<MemoryManager>, &JobConfig, Arc<ProgressManager>) -> std::result::Result<usize, String> + Send + Sync + 'static,
) -> Result<()> {
    
    info!("========================================");
    info!("CONTINUOUS PARALLEL EXECUTION MODE");
    info!("========================================");
    info!("Configuration:");
    info!("  Total jobs: {}", jobs.len());
    info!("  Max parallel: {}", config.execution.max_parallel_comparisons);
    info!("  Strategy: {}", config.execution.parallel_strategy);
    info!("  Database pooling: {}", config.execution.enable_db_pooling);
    
    if jobs.is_empty() {
        info!("No jobs to process");
        return Ok(());
    }
    
    let start_time = std::time::Instant::now();
    
    // Initialize components
    let coordinator = Arc::new(ComparisonCoordinator::new(config)?);
    let job_queue = Arc::new(JobQueue::new(jobs.clone(), config)?);
    
    // Initialize database pool if enabled
    let db_pool = if config.execution.should_enable_pooling() {
        info!("Initializing database pool for continuous execution");
        Some(create_pool_from_config(config))
    } else {
        None
    };
    
    // Log resource allocation details
    info!("Resource allocation per job:");
    info!("  Memory: {} MB", coordinator.get_memory_per_job());
    info!("  Threads: {}", coordinator.get_threads_per_job());
    
    // Clone the tracker so we can move it into threads
    // We'll merge changes back at the end
    let tracker_clone = tracker_arg.clone();
    let tracker = Arc::new(Mutex::new(tracker_clone));
    
    // Wrap the process function for sharing
    let process_fn = Arc::new(process_comparison_fn);
    
    // Create worker threads
    let num_workers = config.execution.max_parallel_comparisons;
    let mut worker_handles = Vec::new();
    
    info!("Spawning {} worker threads", num_workers);
    
    for worker_id in 0..num_workers {
        let job_queue = Arc::clone(&job_queue);
        let coordinator = Arc::clone(&coordinator);
        let config = config.clone();
        let tracker = Arc::clone(&tracker);
        let memory_manager = Arc::clone(&memory_manager);
        let job_config = job_config.clone();
        let progress_manager = Arc::clone(&progress_manager);
        let process_fn = Arc::clone(&process_fn);
        let db_pool = db_pool.clone();
        let worker_sender = job_queue.get_worker_sender();
        
        let handle = thread::spawn(move || {
            worker_thread(
                worker_id,
                job_queue,
                coordinator,
                config.clone(),
                tracker,
                memory_manager,
                job_config,
                progress_manager,
                process_fn,
                db_pool,
                worker_sender,
            )
        });
        
        worker_handles.push(handle);
    }
    
    // Monitor thread - handles worker messages and status updates
    let monitor_handle = {
        let job_queue = Arc::clone(&job_queue);
        let coordinator = Arc::clone(&coordinator);
        
        thread::spawn(move || {
            monitor_thread(job_queue, coordinator)
        })
    };
    
    // Wait for all workers to complete
    for (id, handle) in worker_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => debug!("Worker {} terminated successfully", id),
            Ok(Err(e)) => error!("Worker {} failed: {}", id, e),
            Err(_) => error!("Worker {} panicked", id),
        }
    }
    
    // Signal shutdown to monitor
    job_queue.shutdown();
    job_queue.send_worker_message(WorkerMessage::Shutdown)
        .unwrap_or_else(|e| warn!("Failed to send shutdown to monitor: {}", e));
    
    // Wait for monitor to finish
    if let Err(_) = monitor_handle.join() {
        warn!("Monitor thread panicked");
    }
    
    // Final statistics
    let elapsed = start_time.elapsed();
    let stats = job_queue.get_statistics();
    let status = job_queue.get_status();
    
    info!("========================================");
    info!("CONTINUOUS EXECUTION COMPLETE");
    info!("========================================");
    info!("Results:");
    info!("  Total time: {:.2}s", elapsed.as_secs_f64());
    info!("  Completed: {}/{}", status.completed, status.total);
    info!("  Failed: {}", status.failed);
    info!("  Avg wait time: {:.2}s", stats.average_wait_time.as_secs_f64());
    info!("  Avg run time: {:.2}s", stats.average_run_time.as_secs_f64());
    info!("  Queue efficiency: {:.1}%", stats.queue_efficiency * 100.0);
    info!("  Throughput: {:.2} jobs/sec", 
        status.completed as f64 / elapsed.as_secs_f64());
    
    // Log coordinator final state
    info!("========================================");
    info!("CONTINUOUS EXECUTION COMPLETE");
    info!("========================================");
    info!("Results:");
    info!("  Total time: {:.2}s", elapsed.as_secs_f64());
    info!("  Completed: {}/{}", status.completed, status.total);
    info!("  Failed: {}", status.failed);
    info!("  Avg wait time: {:.2}s", stats.average_wait_time.as_secs_f64());
    info!("  Avg run time: {:.2}s", stats.average_run_time.as_secs_f64());
    info!("  Queue efficiency: {:.1}%", stats.queue_efficiency * 100.0);
    info!("  Throughput: {:.2} jobs/sec", 
        status.completed as f64 / elapsed.as_secs_f64());
    
    // Log coordinator final state
    info!("{}", coordinator.get_resource_status());
    
    // Clean up database pool
    if let Some(pool) = db_pool {
        info!("Cleaning up database pool");
        pool.clear()?;
    }
    
    // Copy tracker changes back to the original
    // This is important! We cloned the tracker at the start, now we need to copy changes back
    {
        let final_tracker = tracker.lock().unwrap();
        *tracker_arg = (*final_tracker).clone();
    }
    
    Ok(())
}

/// Worker thread that continuously processes jobs
fn worker_thread(
    worker_id: usize,
    job_queue: Arc<JobQueue>,
    coordinator: Arc<ComparisonCoordinator>,
    config: Arc<AlNaqlConfig>,
    tracker: Arc<Mutex<JobTracker>>,
    memory_manager: Arc<MemoryManager>,
    job_config: JobConfig,
    progress_manager: Arc<ProgressManager>,
    process_fn: Arc<impl Fn(ComparisonJob, &Arc<AlNaqlConfig>, &mut JobTracker, usize, &Arc<MemoryManager>, &JobConfig, Arc<ProgressManager>) -> std::result::Result<usize, String> + Send + Sync>,
    db_pool: Option<Arc<crate::storage::pool::DatabasePool>>,
    worker_sender: Sender<WorkerMessage>,
) -> Result<()> {
    
    info!("Worker {} started", worker_id);
    let mut jobs_processed = 0;
    
    // Get the thread allocation for this worker
    // This respects your sophisticated thread allocation system
    let thread_count = coordinator.get_threads_per_job();
    debug!("Worker {} will use {} threads per job", worker_id, thread_count);
    
    loop {
        // Check for shutdown
        if job_queue.should_shutdown() {
            debug!("Worker {} shutting down (shutdown signal)", worker_id);
            break;
        }
        
        // Try to get a job
        match job_queue.try_get_runnable(Duration::from_secs(1)) {
            Some(job) => {
                let job_id = job.comparison_number;
                let job_index = job_id - 1;
                let job_start = std::time::Instant::now();
                
                debug!("Worker {} acquired job {}", worker_id, job_id);
                
                // Mark job as started to record start time
                {
                    let mut t = tracker.lock().unwrap();
                    if let Some(tracked_job) = t.get_job_mut(job_index) {
                        tracked_job.mark_started();
                    }
                }
                
                // Send start message
                let _ = worker_sender.send(WorkerMessage::JobStarted {
                    job_id,
                    worker_id,
                });
                
                // Verify with coordinator (belt and suspenders)
                let active_jobs: HashSet<usize> = job_queue.get_active_jobs()
                    .into_iter().collect();
                
                if !coordinator.try_acquire_job_slot(&job, &active_jobs, job_queue.get_scheduler()) {
                    warn!("Worker {} job {} failed coordinator check, returning to queue", 
                        worker_id, job_id);
                    
                    // Return the job to the queue for another worker to pick up
                    job_queue.return_job(job)
                        .unwrap_or_else(|e| error!("Failed to return job {} to queue: {}", job_id, e));
                    
                    continue;
                }
                
                // Register with coordinator and queue
                let _coordinator_handle = coordinator.register_job_continuous(&job, worker_id);
                job_queue.mark_active(job_id, worker_id)?;
                
                // Update progress manager
                let job_name = format!("{} vs {}",
                    job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                    job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
                );
                progress_manager.start_job(job_id, job_name)
                    .unwrap_or_else(|e| warn!("Failed to start progress tracking: {}", e));
                
                // Pre-load databases if using pool
                if let Some(ref pool) = db_pool {
                    let _ = pool.try_get_or_open(&job.source_db_path, Duration::from_secs(2));
                    let _ = pool.try_get_or_open(&job.target_db_path, Duration::from_secs(2));
                    debug!("Worker {} pre-loaded databases for job {}", worker_id, job_id);
                }
                
                // Set thread environment variable for this job
                // This integrates with your existing thread allocation system
                let original_threads = std::env::var("ALNAQL_THREADS").ok();
                std::env::set_var("ALNAQL_THREADS", thread_count.to_string());
                debug!("Worker {} set ALNAQL_THREADS={} for job {}", worker_id, thread_count, job_id);
                
                // Set pool flag if using pool
                if db_pool.is_some() {
                    std::env::set_var("ALNAQL_USE_DB_POOL", "true");
                } else {
                    std::env::remove_var("ALNAQL_USE_DB_POOL");
                }
                
                // Update tracker status
                {
                    let mut t = tracker.lock().unwrap();
                    if let Some(tracked_job) = t.get_job_mut(job_index) {
                        tracked_job.status = JobStatus::InProgress;
                        tracked_job.start_time = Some(std::time::SystemTime::now());
                    }
                    let _ = t.save();
                }
                
                // PROCESS THE JOB
                // Create a temporary tracker for this job (following the batch mode pattern)
                let mut temp_tracker = {
                    let t = tracker.lock().unwrap();
                    (*t).clone()
                };
                
                // Call the actual processing function
                let result = process_fn(
                    job.clone(),
                    &config,
                    &mut temp_tracker,
                    job_index,
                    &memory_manager,
                    &job_config,
                    progress_manager.clone(),
                );
                
                // Merge changes back to main tracker
                {
                    let mut t = tracker.lock().unwrap();
                    if let Some(updated_job) = temp_tracker.get_job(job_index) {
                        if let Some(main_job) = t.get_job_mut(job_index) {
                            *main_job = updated_job.clone();
                        }
                    }
                    let _ = t.save();
                }
                
                // Clean up databases from pool
                if let Some(ref pool) = db_pool {
                    pool.release(&job.source_db_path);
                    pool.release(&job.target_db_path);
                    debug!("Worker {} released database handles for job {}", worker_id, job_id);
                }
                
                // Restore original thread environment
                if let Some(threads) = original_threads {
                    std::env::set_var("ALNAQL_THREADS", threads);
                } else {
                    std::env::remove_var("ALNAQL_THREADS");
                }
                
                // Process result and update queue
                let duration = job_start.elapsed();
                match result {
                    Ok(match_count) => {
                        job_queue.mark_complete(job_id)?;
                        coordinator.completed_jobs.fetch_add(1, Ordering::Relaxed);
                        
                        // Update tracker
                        {
                            let mut t = tracker.lock().unwrap();
                            if let Some(tracked_job) = t.get_job_mut(job_index) {
                                tracked_job.mark_completed(match_count, match_count == usize::MAX);
                            }
                            let _ = t.save();
                        }
                        
                        // Update progress manager
                        if match_count == usize::MAX {
                            progress_manager.early_escape_job(job_id, "High similarity detected".to_string())
                                .unwrap_or_else(|e| warn!("Failed to update progress: {}", e));
                        } else {
                            progress_manager.complete_job(job_id, match_count)
                                .unwrap_or_else(|e| warn!("Failed to complete job progress: {}", e));
                        }
                        
                        // Send completion message
                        let _ = worker_sender.send(WorkerMessage::JobCompleted {
                            job_id,
                            worker_id,
                            result: Ok(match_count),
                            duration,
                        });
                        
                        info!("Worker {} completed job {} ({} matches) in {:.2}s",
                            worker_id, job_id, match_count, duration.as_secs_f64());
                    }
                    Err(e) => {
                        // Determine if we should retry
                        let can_retry = !e.contains("fatal") && !e.contains("corrupt");
                        job_queue.mark_failed(job_id, can_retry)?;
                        coordinator.failed_jobs.fetch_add(1, Ordering::Relaxed);
                        
                        // Update tracker
                        {
                            let mut t = tracker.lock().unwrap();
                            if let Some(tracked_job) = t.get_job_mut(job_index) {
                                tracked_job.status = JobStatus::Failed;
                                tracked_job.error_message = Some(e.clone());
                                tracked_job.end_time = Some(std::time::SystemTime::now());
                            }
                            let _ = t.save();
                        }
                        
                        // Update progress manager
                        progress_manager.fail_job(job_id, e.clone())
                            .unwrap_or_else(|err| warn!("Failed to mark job failed in progress: {}", err));
                        
                        // Send failure message
                        let _ = worker_sender.send(WorkerMessage::JobCompleted {
                            job_id,
                            worker_id,
                            result: Err(e.clone()),
                            duration,
                        });
                        
                        error!("Worker {} job {} failed: {}", worker_id, job_id, e);
                    }
                }
                
                jobs_processed += 1;
                
                // _coordinator_handle drops here, automatically cleaning up resources
            }
            None => {
                // No runnable job available
                if job_queue.is_empty() {
                    debug!("Worker {} exiting (queue empty)", worker_id);
                    break;
                }
                
                // Send idle message periodically
                let _ = worker_sender.send(WorkerMessage::WorkerIdle { worker_id });
                
                // Brief sleep before checking again
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    
    info!("Worker {} terminated after processing {} jobs", worker_id, jobs_processed);
    Ok(())
}

/// Monitor thread that handles worker messages and provides status updates
fn monitor_thread(
    job_queue: Arc<JobQueue>,
    coordinator: Arc<ComparisonCoordinator>,
) -> Result<()> {
    info!("Monitor thread started");
    
    let receiver = job_queue.get_worker_receiver();
    let mut last_status = std::time::Instant::now();
    let mut last_detailed = std::time::Instant::now();
    let status_interval = Duration::from_secs(10);
    let detailed_interval = Duration::from_secs(30);
    
    loop {
        // Check for messages with timeout
        match receiver.recv_timeout(Duration::from_millis(500)) {
            Ok(WorkerMessage::JobCompleted { job_id, worker_id, result, duration }) => {
                debug!("Monitor: Job {} completed by worker {} in {:.2}s",
                    job_id, worker_id, duration.as_secs_f64());
                
                match result {
                    Ok(matches) => {
                        trace!("Monitor: Job {} had {} matches", job_id, matches);
                    }
                    Err(ref e) => {
                        trace!("Monitor: Job {} failed: {}", job_id, e);
                    }
                }
            }
            Ok(WorkerMessage::JobStarted { job_id, worker_id }) => {
                debug!("Monitor: Job {} started by worker {}", job_id, worker_id);
            }
            Ok(WorkerMessage::WorkerIdle { worker_id }) => {
                trace!("Monitor: Worker {} is idle", worker_id);
            }
            Ok(WorkerMessage::WorkerError { worker_id, error }) => {
                error!("Monitor: Worker {} error: {}", worker_id, error);
            }
            Ok(WorkerMessage::Shutdown) => {
                info!("Monitor: Received shutdown signal");
                break;
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Normal timeout, check status
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                debug!("Monitor: Channel disconnected");
                break;
            }
        }
        
        // Periodically log brief status
        if last_status.elapsed() >= status_interval {
            let status = job_queue.get_status();
            info!("Progress: {} active, {} complete, {} failed, {} pending ({}% done)",
                status.active, status.completed, status.failed, status.pending,
                status.progress_percentage() as i32);
            
            last_status = std::time::Instant::now();
        }
        
        // Periodically log detailed status
        if last_detailed.elapsed() >= detailed_interval {
            debug!("{}", coordinator.get_resource_status());
            debug!("{}", job_queue.get_statistics());
            last_detailed = std::time::Instant::now();
        }
        
        // Check if we're done
        if job_queue.is_empty() && job_queue.get_status().active == 0 {
            info!("Monitor: All jobs complete");
            break;
        }
        
        // Check for shutdown
        if job_queue.should_shutdown() {
            info!("Monitor: Shutting down");
            break;
        }
    }
    
    info!("Monitor thread terminated");
    Ok(())
}