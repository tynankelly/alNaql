// Standard library imports
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

// External crates
use env_logger::Builder;
use log::{debug, error, info, warn, LevelFilter};
use rayon;

// Internal crate imports (alnaql)
use alnaql::{
    AlNaqlConfig,
    config::{parse_job_config, ProcessStrategy, JobConfig},
    matcher::MatcherOrchestrator,
    storage::{
        cache::METADATA_CACHE,
        open_storage,
        LMDBStorage,
    },
    utils::{
        cloud_signal::SpotTerminationHandler,
        job_tracker,
        job_tracker::{JobTracker, ComparisonJob},
        ProgressPhase,
        ProgressManager,
        parallel_executor::execute_parallel_continuous,
        resource_coordinator::{
            environment_manager,
            initialize_resource_management,
            memory_manager::{MemoryManager, MemoryState},
            SHARED_STATE,
        },
        thread_allocation::{
        ResourceAllocation,
        SystemState, 
        ThreadAllocationConfig, 
        calculate_thread_count,
    },
    },
};


fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Process a single comparison - backward compatibility wrapper
/// 
/// This wrapper maintains compatibility with parallel_executor which expects
/// a specific function signature. It automatically determines resource allocation
/// based on execution context:
/// - In parallel mode: Uses thread budget from ALNAQL_THREADS env var
/// - In sequential mode: Uses full system resources
///
/// For new code, prefer using process_comparison_with_resources directly.
/// 
/// TODO: Remove this wrapper once parallel_executor is updated to pass
/// resources directly (planned for v2.0)
fn process_comparison(
    job: ComparisonJob,
    config: &Arc<AlNaqlConfig>, 
    tracker: &mut JobTracker,
    job_index: usize,
    memory_manager: &Arc<MemoryManager>,
    job_config: &JobConfig,
    progress_manager: Arc<ProgressManager>,
) -> Result<usize, String> {
    // Determine resource allocation based on execution context
    
    // First, assess system state
    let system_state = SystemState::assess()
        .map_err(|e| format!("Failed to assess system state: {}", e))?;
    
    // Check if we're in parallel mode (environment variable set by coordinator)
    let resources = if let Ok(thread_budget_str) = std::env::var("ALNAQL_THREADS") {
        // Parallel mode - coordinator has set a thread budget
        if let Ok(thread_budget) = thread_budget_str.parse::<usize>() {
            debug!("Wrapper: Using thread budget from parallel coordinator: {}", thread_budget);
            
            // Create ResourceAllocation for parallel mode
            // We don't know the exact memory allocation here, so estimate
            let available_memory = (system_state.total_memory_mb as f64 * 0.8) as usize;
            let memory_per_job = available_memory / config.execution.max_parallel_comparisons;
            
            ResourceAllocation::parallel(
                &config.execution,
                memory_per_job,
                thread_budget
            )
        } else {
            // Fallback if parsing fails
            warn!("Wrapper: Failed to parse ALNAQL_THREADS: {}", thread_budget_str);
            ResourceAllocation::sequential(
                &config.execution,
                system_state.total_memory_mb,
                system_state.cpu_cores
            )
        }
    } else {
        // Sequential mode - no env var set, use full system resources
        debug!("Wrapper: Using sequential resource allocation");
        ResourceAllocation::sequential(
            &config.execution,
            system_state.total_memory_mb,
            system_state.cpu_cores
        )
    };
    
    // Call the new function with resources
    process_comparison_with_resources(
        job,
        config,
        tracker,
        job_index,
        memory_manager,
        job_config,
        progress_manager,
        resources
    )
}

/// Process a single comparison with explicit resource allocation
/// This is the new primary function that accepts resources directly
fn process_comparison_with_resources(
    job: ComparisonJob,
    config: &Arc<AlNaqlConfig>,
    tracker: &mut JobTracker,
    job_index: usize,
    memory_manager: &Arc<MemoryManager>,
    job_config: &JobConfig,
    progress_manager: Arc<ProgressManager>,
    resources: ResourceAllocation,
) -> Result<usize, String> {
    let start_time = std::time::Instant::now();
    info!("========================================");
    info!("Processing comparison {}: {} vs {}", 
        job.comparison_number,
        job.source_db_path.display(),
        job.target_db_path.display(),
    );
    info!("========================================");
    info!("Resource allocation:");
    info!("  Memory: {} MB", resources.memory_mb);
    info!("  Thread budget: {}", resources.thread_budget);
    info!("  Optimal threads: {}", resources.optimal_threads);
    info!("  Min/Max threads: {}/{}", resources.min_threads, resources.max_threads);
    
    // Update job status
    tracker.update_job_status(job_index, job_tracker::JobStatus::InProgress)
        .map_err(|e| e.to_string())?;

    // Create progress context for this job
    let progress_context = alnaql::utils::progress_manager::ProgressContext::new(
        progress_manager.clone(),
        job.comparison_number
    );
    
    // Report memory state at start
    info!("Memory state at start of job {}", job.comparison_number);
    memory_manager.log_memory_status();
        
    // Track open database connections so we can explicitly close them
    let mut source_storage: Option<Arc<LMDBStorage>> = None;
    let mut target_storage: Option<Arc<LMDBStorage>> = None;
    
    // Track peak memory usage
    let initial_memory = job_tracker::ComparisonJob::get_current_memory_usage().unwrap_or(0);
    let mut peak_memory = initial_memory;
    
    // Set early escape threshold as a hardcoded value
    let early_escape_threshold = 0.98;
    info!("Using early escape threshold: {:.2} ({}%)", 
         early_escape_threshold, early_escape_threshold * 100.0);
    
    // STEP 1: Assess system state for this comparison
    let system_state = SystemState::assess()
        .map_err(|e| format!("Failed to assess system state: {}", e))?;
    
    // STEP 2: Create thread allocation config from ResourceAllocation
    // NEW: Use the passed-in resources instead of env var or default
    let thread_config = ThreadAllocationConfig::from_config_with_budget(
        &config.execution,
        resources.thread_budget
    );
    
    info!("Thread configuration source: ResourceAllocation (explicit)");
    info!("  Config values: min={}, optimal={}, max={}", 
        thread_config.min_threads, 
        thread_config.optimal_threads, 
        thread_config.max_threads);
    
    // STEP 3: Calculate optimal threads for THIS comparison
    let thread_count = calculate_thread_count(&system_state, &thread_config);
    
    // STEP 4: Create thread pool for this comparison
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .thread_name(|i| format!("comparison-{}", i))
        .build()
        .map_err(|e| format!("Failed to create thread pool: {}", e))?;
    
    info!("Starting comparison with {} threads", thread_count);
    
    // STEP 5: Run entire comparison within this pool
    let result = pool.install(|| {
        // [REST OF THE FUNCTION BODY REMAINS EXACTLY THE SAME]
        // Step 1: Open database connections for this job only
        debug!("Opening source database: {:?}", job.source_db_path);
                
        // Pass file size parameter instead of None
        let source_db = match open_storage(&job.source_db_path, config.storage.clone()) {
            Ok(storage) => Arc::new(storage),  // Wrap here
            Err(e) => return Err(format!("Failed to open source database: {}", e)),
        };
        
        // Store for later cleanup
        source_storage = Some(source_db.clone());
        // ADD DEBUG CODE HERE - use source_db, not source_storage
        info!("Debug: Checking source ngram_metadata database contents");
        match source_db.get_ngram_metadata_batch(&[1]) {
            Ok(metadata_vec) => {
                info!("Successfully retrieved {} metadata entries for sequence 1", metadata_vec.len());
                if metadata_vec.is_empty() {
                    error!("Metadata batch returned empty for sequence 1!");
                } else {
                    info!("Sequence 1 metadata exists: {:?}", metadata_vec[0].sequence_number);
                }
            },
            Err(e) => error!("Failed to get metadata for sequence 1: {}", e),
        }

        debug!("Opening target database: {:?}", job.target_db_path);

        debug!("Opening target database: {:?}", job.target_db_path);
                
        // Pass file size parameter instead of None
        let target_db = match open_storage(&job.target_db_path, config.storage.clone()) {
            Ok(storage) => Arc::new(storage),  // Wrap here
            Err(e) => return Err(format!("Failed to open target database: {}", e)),
        };
        
        // Store for later cleanup
        target_storage = Some(target_db.clone());
        
        // Check memory after opening target database
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High {
            info!("High memory pressure after opening target database: {}", memory_state);
        };

        // Step 2: Get database statistics
        let source_ngrams = match source_db.get_db_info() {
            Ok(stats) => stats.total_ngrams,
            Err(e) => return Err(format!("Failed to get source stats: {}", e)),
        };
            
        let target_ngrams = match target_db.get_db_info() {
            Ok(stats) => stats.total_ngrams,
            Err(e) => return Err(format!("Failed to get target stats: {}", e)),
        };
        
        // Log original database sizes
        info!("Original comparison setup - Source: {} ngrams, Target: {} ngrams", 
             source_ngrams, target_ngrams);
        
        // Determine if we need to swap roles for better performance
        let (final_source_db, final_target_db, final_source_ngrams, final_target_ngrams) = 
            if source_ngrams <= target_ngrams {
                // Swap roles if source is smaller than target
                info!("Swapping source and target for better performance (using larger database as source)");
                
                // Update the output file name to reflect the swap
                let source_name = job.target_db_path.file_stem().unwrap().to_string_lossy();
                let target_name = job.source_db_path.file_stem().unwrap().to_string_lossy();
                let parent = job.output_path.parent().unwrap_or(Path::new(""));
                let new_output_path = parent.join(format!("matches_{}_{}.jsonl", source_name, target_name));
                
                info!("Updated output path after swap: {:?}", new_output_path);
                
                // Update the job's output path in the tracker
                if let Some(tracked_job) = tracker.get_job_mut(job_index) {
                    tracked_job.output_path = new_output_path.clone();
                    // Save the updated path
                    if let Err(e) = tracker.save() {
                        warn!("Failed to save updated output path: {}", e);
                    }
                }
                
                (target_db, source_db, target_ngrams, source_ngrams)
            } else {
                // Keep original configuration
                (source_db, target_db, source_ngrams, target_ngrams)
            };
        
        // Log final configuration after potential swap
        info!("Processing comparison - Source: {} ngrams, Target: {} ngrams", 
             final_source_ngrams, final_target_ngrams);
        progress_context.update_progress(25, 100, Some("Databases opened".to_string()));
        
        let mut matcher = match MatcherOrchestrator::new(
            final_source_db,
            final_target_db,
            Arc::clone(&config),
            config.generator.ngram_type,
        ) {
            Ok(matcher) => matcher,
            Err(e) => return Err(format!("Failed to create matcher: {}", e)),
        };
        
        progress_context.update_progress(50, 100, Some("Matcher initialized".to_string()));

        // Check memory usage after matcher creation
        if let Some(current_memory) = job_tracker::ComparisonJob::get_current_memory_usage() {
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        
        // Check memory pressure after creating matcher
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High {
            info!("High memory pressure after creating matcher: {}", memory_state);
            // Force memory reclamation
            drop(Vec::<u8>::with_capacity(1024 * 1024 * 30)); // 30MB
            thread::sleep(Duration::from_millis(200));
        }

        progress_context.update_progress(100, 100, Some("Starting comparison".to_string()));
        // Step 4: Use the appropriate find_matches method based on process_strategy
        let match_count = match config.matcher.process_strategy {
            ProcessStrategy::Sequential => {
                info!("Using SEQUENTIAL processing strategy");
                matcher.find_matches(
                    1..(final_source_ngrams + 1),
                    1..(final_target_ngrams + 1),
                    Some(&job.output_path),
                    early_escape_threshold,
                    config,
                    Some(progress_context.clone())
                ).map_err(|e| format!("Failed to find matches: {}", e))?  // Convert Error to String
            },
            ProcessStrategy::Parallel => {
                info!("Using PARALLEL processing strategy");
                matcher.find_matches_parallel(
                    1..(final_source_ngrams + 1),
                    1..(final_target_ngrams + 1),
                    Some(&job.output_path),
                    early_escape_threshold,
                    config,
                    Some(progress_context.clone())
                ).map_err(|e| format!("Failed to find matches with parallel: {}", e))?  // Convert Error to String
            },
            ProcessStrategy::Grid => {
                info!("Using GRID STREAMING processing strategy");
                matcher.find_matches_grid_streaming(
                    1..(final_source_ngrams + 1),
                    1..(final_target_ngrams + 1),
                    Some(&job.output_path),
                    job_config.resume,
                    &job.source_db_path,
                    &job.target_db_path,
                    Some(progress_context.clone()) 
                ).map_err(|e| format!("Failed to find matches with grid: {}", e))?
            }
        };

        // Check for early escape without progress bar
        if match_count == usize::MAX {
            info!("Early termination: High similarity detected (>{:.1}%), likely duplicate databases", 
                early_escape_threshold * 100.0);
        }

        // Check memory after processing
        if let Some(current_memory) = job_tracker::ComparisonJob::get_current_memory_usage() {
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        memory_manager.log_memory_status();

        Ok(match_count)
    });
    
    // Apply memory management before cleanup
    let memory_state = memory_manager.monitor_memory_thresholds();
    if memory_state == MemoryState::High || memory_state == MemoryState::AboveNormal {
        info!("Elevated memory pressure before database cleanup: {}", memory_state);
        info!("Forcing memory reclamation before database cleanup");
        drop(Vec::<u8>::with_capacity(1024 * 1024 * 50)); // 50MB
        thread::sleep(Duration::from_millis(300));
    }
    
    // Step 5: Ensure database connections are properly closed, regardless of success or failure
    info!("Closing database connections for job {}", job.comparison_number);

    // Drop the source database Arc and close environment
    if let Some(db_arc) = source_storage.take() {
        info!("Dropping source database reference: {:?}", job.source_db_path);
        
        // Drop the Arc reference - storage will auto-cleanup when last Arc is dropped
        drop(db_arc);
        
        // Use environment manager to ensure the environment is closed
        match environment_manager::close_environment(&job.source_db_path) {
            Ok(true) => info!("Environment manager successfully closed source database environment"),
            Ok(false) => warn!("Environment manager marked source database for deferred closure"),
            Err(e) => warn!("Environment manager failed to close source database: {}", e),
        }
    }

    // Drop the target database Arc and close environment
    if let Some(db_arc) = target_storage.take() {
        info!("Dropping target database reference: {:?}", job.target_db_path);
        
        // Drop the Arc reference - storage will auto-cleanup when last Arc is dropped
        drop(db_arc);
        
        // Use environment manager to ensure the environment is closed
        match environment_manager::close_environment(&job.target_db_path) {
            Ok(true) => info!("Environment manager successfully closed target database environment"),
            Ok(false) => warn!("Environment manager marked target database for deferred closure"),
            Err(e) => warn!("Environment manager failed to close target database: {}", e),
        }
    }
        
    // Step 6: Verify resources are released
    info!("Verifying database resources for job {}", job.comparison_number);
    // Clear thread-local caches to free memory
    info!("Clearing thread-local metadata caches for job {}", job.comparison_number);
    METADATA_CACHE.clear_thread_local();
    
    // Check memory after database closure
    info!("Memory status after database closure:");
    memory_manager.log_memory_status();
    
    // Verify source database is closed
    let source_closed = match environment_manager::deep_verify_environment(&job.source_db_path) {
        Ok(status) => {
            match status {
                environment_manager::EnvironmentStatus::NotFound |
                environment_manager::EnvironmentStatus::DirectoryExists => {
                    // These statuses indicate the environment is not actively managed
                    info!("Source database properly released: {:?}", job.source_db_path);
                    true
                },
                environment_manager::EnvironmentStatus::ClosurePending => {
                    // Marked for closure with no active transactions
                    info!("Source database pending closure: {:?}", job.source_db_path);
                    true
                },
                status => {
                    // Other statuses indicate it's not fully closed
                    warn!("Source database not fully released: {:?} (status: {:?})", job.source_db_path, status);
                    
                    // Request closure through environment manager
                    if let Err(e) = environment_manager::close_environment(&job.source_db_path) {
                        warn!("Error requesting closure of source database: {}", e);
                    }
                    false
                }
            }
        },
        Err(e) => {
            warn!("Failed to verify environment status: {}", e);
            false
        }
    };
    
    // Verify target database is closed
    let target_closed = LMDBStorage::verify_closed(&job.target_db_path);
    if !target_closed {
        warn!("Target database may not be fully closed: {:?}", job.target_db_path);
        // Try again to remove from cache
        if let Err(e) = LMDBStorage::remove_from_cache(&job.target_db_path) {
            warn!("Error forcing closure of target database: {}", e);
        }
    }
    
    // Apply additional cleanup after verification
    if !source_closed || !target_closed {
        info!("Applying additional memory management for resources that failed to close");
        // Force additional GC
        drop(Vec::<u8>::with_capacity(1024 * 1024 * 30));
        thread::sleep(Duration::from_millis(500));
        
        // Try verification again
        LMDBStorage::verify_closed(&job.source_db_path);
        LMDBStorage::verify_closed(&job.target_db_path);
    }
    
    // Calculate duration
    let duration = start_time.elapsed();
    
    // Update job status based on result
    if let Some(tracked_job) = tracker.get_job_mut(job_index) {
        match &result {
            Ok(count) => {
                if *count == usize::MAX {
                    // Early escape detected
                    tracked_job.mark_completed(0, true);
                    info!("Job {} completed early in {:?} (detected as duplicate)", 
                        job.comparison_number, duration);
                } else {
                    // Normal completion
                    tracked_job.mark_completed(*count, false);
                    info!("Job {} completed in {:?}, found {} matches", 
                        job.comparison_number, duration, count);
                }
                // Set peak memory usage
                tracked_job.peak_memory_mb = Some(peak_memory);
            },
            Err(e) => {
                // Record failure
                tracked_job.mark_failed(e.clone());
                warn!("Job {} failed after {:?}: {}", 
                    job.comparison_number, duration, e);
            }
        }
        
        // Make sure to save the updated job status
        if let Err(e) = tracker.save() {
            warn!("Failed to save final job status: {}", e);
        }

    }
    
    // Final memory management before returning
    info!("Final memory status for job {}:", job.comparison_number);
    memory_manager.log_memory_status();

    // Verify database cleanup
    if !verify_database_cleanup(&job.source_db_path) {
        warn!("Source database may not be fully cleaned up: {}", job.source_db_path.display());
    }
    if !verify_database_cleanup(&job.target_db_path) {
        warn!("Target database may not be fully cleaned up: {}", job.target_db_path.display());
    }
    
    // Ensure thread-local caches are cleared between jobs
    METADATA_CACHE.clear_all_caches();
    // Add explicit pause to allow system to fully release resources
    thread::sleep(Duration::from_millis(200));

    progress_context.enter_phase(ProgressPhase::Complete);
    // Return the result
    result
}

fn check_system_resources() -> String {
    let mut status = Vec::new();
    
    // Check memory usage
    if let Ok(mem_info) = sys_info::mem_info() {
        let used_mem_mb = (mem_info.total - mem_info.avail) / 1024;
        let total_mem_mb = mem_info.total / 1024;
        let used_mem_pct = (used_mem_mb as f64 / total_mem_mb as f64) * 100.0;
        status.push(format!("Memory: {}/{} MB ({:.1}%)", used_mem_mb, total_mem_mb, used_mem_pct));
    } else {
        status.push("Memory: unknown".to_string());
    }
    
    // Count open file handles (Linux specific)
    #[cfg(target_os = "linux")]
    {
        let fd_path = std::path::Path::new("/proc/self/fd");
        if fd_path.exists() {
            match std::fs::read_dir(fd_path) {
                Ok(entries) => {
                    let count = entries.count();
                    status.push(format!("Open file handles: {}", count));
                },
                Err(_) => {
                    status.push("Open file handles: unavailable".to_string());
                }
            }
        }
    }
    
    // Check CPU usage
    if let Ok(cpu_info) = sys_info::loadavg() {
        status.push(format!("CPU load: {:.2}, {:.2}, {:.2}", cpu_info.one, cpu_info.five, cpu_info.fifteen));
    } else {
        status.push("CPU load: unknown".to_string());
    }
    
    // Count active LMDB environments
    let env_stats = match alnaql::utils::resource_coordinator::environment_manager::ENVIRONMENT_MANAGER.read() {
        Ok(manager) => manager.get_stats(),
        Err(_) => alnaql::utils::resource_coordinator::environment_manager::EnvironmentStats {
            total_environments: 0,
            total_transactions: 0,
        }
    };
    status.push(format!("Active DB environments: {}", env_stats.total_environments));
    
    // Return the joined status string
    status.join(" | ")
}

fn verify_database_cleanup(db_path: &Path) -> bool {
    info!("Verifying cleanup for database: {:?}", db_path);
    
    // Check if environment is still tracked by the environment manager
    let env_status = match alnaql::utils::resource_coordinator::environment_manager::deep_verify_environment(db_path) {
        Ok(status) => {
            match status {
                alnaql::utils::resource_coordinator::environment_manager::EnvironmentStatus::NotFound |
                alnaql::utils::resource_coordinator::environment_manager::EnvironmentStatus::DirectoryExists => {
                    // These statuses indicate the environment is not actively managed
                    info!("Environment not tracked: {:?}", db_path);
                    true
                },
                alnaql::utils::resource_coordinator::environment_manager::EnvironmentStatus::ClosurePending => {
                    // Marked for closure with no active transactions
                    info!("Environment pending closure: {:?}", db_path);
                    true  // This is also considered "clean"
                },
                status => {
                    // Other statuses indicate it's not fully closed
                    warn!("Environment still active: {:?} (status: {:?})", db_path, status);
                    false
                }
            }
        },
        Err(e) => {
            warn!("Failed to verify environment status: {}", e);
            false  // Assume not closed if verification fails
        }
    };
    
    // Check lock file status - LMDB uses lock.mdb instead of LOCK
    let lock_path = db_path.join("lock.mdb");
    let lock_exists = lock_path.exists();
    
    // If the lock file exists but the environment is not tracked,
    // it could be a stale lock or owned by another process
    if lock_exists && env_status {
        info!("Lock file exists but environment is not tracked: {:?}", db_path);
    }
    
    // Check if we can open the lock file (will succeed if not locked)
    let can_open_lock = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock_path) 
    {
        Ok(file) => {
            // Close the file handle immediately
            drop(file);
            true
        },
        Err(e) => {
            warn!("Cannot open lock file: {}", e);
            false
        }
    };
        
    // For LMDB, we consider it clean if the environment is not tracked
    // and either the lock file doesn't exist or we can open it
    let is_clean = env_status && (can_open_lock || !lock_exists);
    
    if !is_clean {
        warn!("Database {:?} may not be fully cleaned up!", db_path);
        // Rest of warning code...
    } else {
        info!("Database {:?} is properly cleaned up", db_path);
    }
    
    is_clean
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // Parse command line arguments (includes validation and help handling)
    let job_config = parse_job_config();

    // Initialize the memory manager with reasonable batch sizes
    let memory_manager = Arc::new(MemoryManager::new());
    info!("Memory manager initialized - checking initial memory status");
    memory_manager.log_memory_status();
    
    // Load configuration using found path or default
    let config = if let Some(config_path) = &job_config.config_file {
        info!("Loading configuration from: {}", config_path);
        Arc::new(AlNaqlConfig::from_ini(config_path)?)  // ✅ Wrap in Arc
    } else {
        info!("Loading configuration from default.ini");
        Arc::new(AlNaqlConfig::from_ini("default.ini")?)  // ✅ Wrap in Arc
    };

    // Handle reset without exiting
    if job_config.reset {
        let _ = std::fs::remove_dir_all(&config.files.output_dir);
        println!("Reset complete: state and output directories removed");
        // DON'T return - continue with execution
    }

    // 🚀 Initialize coordinated resource management system
    initialize_resource_management()?;
    
    // Set up logging with minimal configuration
    let timestamp = chrono::Local::now().format("%m_%d_%H_%M");
    fs::create_dir_all("logs")?;
    let log_file = File::create(format!("logs/matcher_{}.log", timestamp))?;

    // Convert config string to LevelFilter
    let log_level = match config.debug.ngram_log.to_lowercase().as_str() {
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        "none" => LevelFilter::Off,
        _ => {
            println!("Invalid log level '{}', defaulting to Info", config.debug.ngram_log);
            LevelFilter::Info
        }
    };

    // Use a simpler configuration to avoid formatting issues
    Builder::new()
        .filter(None, log_level)
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    // Log the startup message if logging is enabled
    if log_level != LevelFilter::Off {
        info!("Starting match finding process with log level: {:?}", log_level);
    }

    // Create and set up termination handler (logger created only if needed)
    let spot_handler = SpotTerminationHandler::new();
    
    // Set up directories
    let source_dir = Path::new(&config.files.ngrams_source_dir);
    let target_dir = Path::new(&config.files.ngrams_target_dir);
    let output_dir = Path::new(&config.files.output_dir);
    info!("Using output directory from config: {:?}", output_dir);

    // Make sure the directory exists
    fs::create_dir_all(output_dir)?;
    
    // Initialize or load job tracker
    let mut tracker = if let Some(path) = &job_config.job_file {
        // Use specified job file
        info!("Using specified job file: {:?}", path);
        let mut tracker = job_tracker::JobTracker::new(Some(path))?;
        tracker.load()?;
        tracker
    } else {
        // Look for existing job file in state directory
        let output_dir = Path::new(&config.files.output_dir);
        match job_tracker::JobTracker::load_existing(output_dir)? {
            Some(tracker) => {
                info!("Found existing job tracker with {} jobs", tracker.jobs.len());
                tracker
            },
            None => {
                // Create new tracker
                let mut tracker = job_tracker::JobTracker::new(None)?;
                
                // Discover jobs
                let jobs = job_tracker::discover_jobs(
                    source_dir,
                    target_dir,
                    output_dir,
                    job_config.self_comparison
                )?;
                
                tracker.jobs = jobs;
                tracker.save()?;
                info!("Created new job tracker with {} jobs", tracker.jobs.len());
                tracker
            }
        }
    };

    // Handle display-only modes
    if job_config.list_jobs {
        println!("\nJob List (Total: {}):", tracker.jobs.len());
        println!("{:<5} {:<15} {:<15} {:<15} {:<10} {:<10}", 
                "ID", "Source", "Target", "Status", "Matches", "Duration");
        println!("{}", "-".repeat(75));
        
        for job in &tracker.jobs {
            let source_name = job.source_db_path.file_stem().unwrap_or_default().to_string_lossy();
            let target_name = job.target_db_path.file_stem().unwrap_or_default().to_string_lossy();
            let status = job.status.as_str();
            let matches = job.match_count.map_or("--".to_string(), |c| c.to_string());
            let duration = job.duration_seconds.map_or("--".to_string(), 
                                                    |d| format!("{:.1}s", d));
            
            println!("{:<5} {:<15} {:<15} {:<15} {:<10} {:<10}", 
                    job.comparison_number, 
                    source_name.to_string(), 
                    target_name.to_string(), 
                    status, 
                    matches, 
                    duration);
        }
    }

    // Check for interrupted jobs
    let interrupted_job_indices: Vec<usize> = tracker.check_for_interrupted_jobs()
        .into_iter()
        .map(|(idx, _)| idx)
        .collect();
    if !interrupted_job_indices.is_empty() && job_config.resume {
        info!("Found {} interrupted jobs that can be resumed", interrupted_job_indices.len());
        for idx in &interrupted_job_indices {
            info!("  Job {}: {:?} -> {:?}", 
                tracker.jobs[*idx].comparison_number,
                tracker.jobs[*idx].source_db_path.file_name().unwrap_or_default(),
                tracker.jobs[*idx].target_db_path.file_name().unwrap_or_default()
            );
        }
        
        // Ask user if they want to resume
        println!("\nFound {} interrupted jobs. Resume them? (y/n)", interrupted_job_indices.len());
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() == "y" {
            info!("User chose to resume interrupted jobs");
        } else {
            info!("User chose not to resume interrupted jobs");
            // Mark them as failed
            tracker.mark_interrupted_jobs_failed()?;
        }
    }

    // Determine which jobs to process
    let statuses_to_process = job_config.get_statuses_to_process();
    info!("Will process jobs with statuses: {:?}", statuses_to_process);
    
    // Get jobs to process with resume handling
    let pending_jobs: Vec<job_tracker::ComparisonJob> = {
        if job_config.resume && !interrupted_job_indices.is_empty() {
            println!("\nFound {} interrupted jobs. Resume them? (y/n)", interrupted_job_indices.len());
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            
            if input.trim().to_lowercase() == "y" {
                info!("Resuming {} interrupted jobs", interrupted_job_indices.len());
                
                // Prepare jobs for resumption using indices
                let mut jobs_to_resume = Vec::new();
                for job_index in &interrupted_job_indices {
                    match tracker.prepare_job_for_resume(*job_index) {
                        Ok(_) => {
                            info!("Prepared job {} for resumption", 
                                tracker.jobs[*job_index].comparison_number);
                            jobs_to_resume.push(tracker.jobs[*job_index].clone());
                        },
                        Err(e) => {
                            warn!("Failed to prepare job {} for resumption: {}", 
                                tracker.jobs[*job_index].comparison_number, e);
                        }
                    }
                }
                
                // Get regular pending jobs
                let mut regular_jobs: Vec<job_tracker::ComparisonJob> = tracker.jobs.iter()
                    .filter(|job| statuses_to_process.contains(&job.status) && !job.is_interrupted())
                    .cloned()
                    .collect();
                
                // Add resumed jobs to the front
                if !jobs_to_resume.is_empty() {
                    info!("Adding {} resumed jobs to the front of the processing queue", jobs_to_resume.len());
                    jobs_to_resume.append(&mut regular_jobs);
                    jobs_to_resume
                } else {
                    regular_jobs
                }
            } else {
                info!("User chose not to resume interrupted jobs");
                if let Err(e) = tracker.mark_interrupted_jobs_failed() {
                    warn!("Error marking interrupted jobs as failed: {}", e);
                }
                
                // Get jobs to process
                tracker.jobs.iter()
                    .filter(|job| statuses_to_process.contains(&job.status) && !job.is_interrupted())
                    .cloned()
                    .collect()
            }
        } else {
            // Get jobs to process
            tracker.jobs.iter()
                .filter(|job| statuses_to_process.contains(&job.status))
                .cloned()
                .collect()
        }
    };

    // Check if we're skipping jobs with existing output files
    let mut jobs_to_process: Vec<job_tracker::ComparisonJob> = Vec::new();
    let mut skipped_jobs = 0;

    for job in pending_jobs {
        let skip_job = job_config.skip_existing && 
                      job.output_path.exists() && 
                      fs::metadata(&job.output_path)
                        .map(|m| m.len() > 0)
                        .unwrap_or(false);
                          
        if skip_job {
            skipped_jobs += 1;
            info!("Skipping job {} because output file already exists: {:?}",
                 job.comparison_number, job.output_path);
            
            // Update the job status in the tracker if it's not already marked
            if let Some(tracked_job) = tracker.jobs.iter_mut()
                .find(|j| j.comparison_number == job.comparison_number) {
                if tracked_job.status != job_tracker::JobStatus::Skipped {
                    tracked_job.mark_skipped(Some("Output file already exists".to_string()));
                    // Save the updated status
                    tracker.save()?;
                }
            }
        } else {
            jobs_to_process.push(job);
        }
    }

    if skipped_jobs > 0 {
        info!("Skipped {} jobs with existing output files", skipped_jobs);
    }

    info!("Processing {} jobs", jobs_to_process.len());
    
    if jobs_to_process.is_empty() {
        println!("No jobs to process.");
        return Ok(());
    }

    // Start processing
    let program_start = Instant::now();
    
    // Determine execution mode and create progress manager
    let execution_mode = if config.execution.max_parallel_comparisons > 1 {
        alnaql::utils::progress_manager::ExecutionMode::Parallel { 
            max_concurrent: config.execution.max_parallel_comparisons 
        }
    } else {
        alnaql::utils::progress_manager::ExecutionMode::Sequential
    };

    // Calculate how many were already completed or skipped
    let previously_completed = tracker.jobs.iter()
        .filter(|job| job.status == job_tracker::JobStatus::Completed || 
                    job.status == job_tracker::JobStatus::Skipped)
        .count();

    // Create progress manager with resume info
    let progress_manager = alnaql::utils::progress_manager::ProgressManager::new(
        tracker.jobs.len(),        // Total jobs from tracker
        previously_completed,      // Pass the completed + skipped count
        execution_mode
    );
    // ========================================================================
    // PARALLEL VS SEQUENTIAL EXECUTION
    // ========================================================================
    
    // Check if parallel mode is enabled
    let use_parallel = config.execution.max_parallel_comparisons > 1;
    
    if use_parallel {
        // ====================================================================
        // PARALLEL EXECUTION - CONTINUOUS MODE
        // ====================================================================
        info!("========================================");
        info!("PARALLEL COMPARISON MODE");
        info!("========================================");
        info!("Configuration:");
        info!("  Max parallel comparisons: {}", config.execution.max_parallel_comparisons);
        info!("  Strategy: {}", config.execution.parallel_strategy);
        info!("  Database pooling: {}", config.execution.enable_db_pooling);
        info!("  Min memory per comparison: {} MB", config.execution.min_memory_per_comparison);
        info!("  Min threads per comparison: {}", config.execution.min_threads_per_comparison);
        
        // Validate we have enough resources
        let system_state = SystemState::assess()
            .map_err(|e| format!("Failed to assess system state: {}", e))?;
        
        let required_memory = config.execution.min_memory_per_comparison * 
                            config.execution.max_parallel_comparisons;
        
        if required_memory > system_state.total_memory_mb {
            warn!("WARNING: Parallel configuration requires {} MB but only {} MB available",
                required_memory, system_state.total_memory_mb);
            warn!("Consider reducing max_parallel_comparisons in config");
        }
        
        info!("Processing {} jobs in continuous parallel mode", jobs_to_process.len());
        info!("Using continuous execution with {} workers", config.execution.max_parallel_comparisons);
        
        // Execute using continuous parallel processing
        // This maintains exactly N parallel jobs at all times
        execute_parallel_continuous(
            jobs_to_process,
            &config,
            &mut tracker,
            &memory_manager,
            &job_config,
            progress_manager.clone(),
            process_comparison,
        )?;
        
        // Log resource utilization
        info!("Resource utilization summary:");
        info!("  Peak memory pressure: {}", SHARED_STATE.memory_pressure.load(Ordering::Relaxed));
        info!("  Max threads allocated: {}", SHARED_STATE.total_threads_allocated.load(Ordering::Relaxed));
        info!("  Final comparison status: {}", SHARED_STATE.get_comparison_status());
        
    } else {
        // SEQUENTIAL EXECUTION MODE
        info!("========================================");
        info!("SEQUENTIAL COMPARISON MODE");
        info!("========================================");
        info!("Processing {} jobs sequentially", jobs_to_process.len());

        // Process jobs sequentially
        for (job_index, job) in jobs_to_process.clone().into_iter().enumerate() {
            let job_number = job.comparison_number;
            
            // Skip if already completed
            if job.status == job_tracker::JobStatus::Completed {
                info!("Job {} already completed, skipping", job_number);
                progress_manager.update_overall_message(format!("Job {} already completed", job_number));
                continue;
            }
            
            // Check for termination
            if spot_handler.is_terminated() {
                warn!("Spot instance termination signal received!");
                tracker.save()?;
                break;
            }
            
            // Clean up resources from previous job (KEEP ALL OF THIS)
            if job_index > 0 {
                info!("Attempting cleanup of previous job resources...");
                
                // Try to clean up previous job's databases
                let prev_job = &jobs_to_process[job_index - 1];
                
                // Try to clean up source database
                if let Err(e) = LMDBStorage::remove_from_cache(&prev_job.source_db_path) {
                    debug!("Could not remove source from cache: {}", e);
                }
                
                // Try to clean up target database  
                if let Err(e) = LMDBStorage::remove_from_cache(&prev_job.target_db_path) {
                    debug!("Could not remove target from cache: {}", e);
                }
                
                // Try environment manager cleanup
                if let Err(e) = environment_manager::close_environment(&prev_job.source_db_path) {
                    debug!("Could not close source environment: {}", e);
                }
                if let Err(e) = environment_manager::close_environment(&prev_job.target_db_path) {
                    debug!("Could not close target environment: {}", e);
                }
                
                // Add a pause to allow resources to be properly released
                thread::sleep(Duration::from_secs(1));
            }
            
            // Log detailed resource state before job
            info!("System resources before job {}:", job_number);
            let before_status = check_system_resources();
            info!("{}", before_status);
            
            // Start tracking this job with progress manager
            let job_name = format!("{} vs {}",
                job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
            );
            
            progress_manager.start_job(job_number, job_name.clone())
                .unwrap_or_else(|e| warn!("Failed to start progress tracking: {}", e));
            
            // Mark job as started to record start time
            if let Some(tracked_job) = tracker.get_job_mut(job_index) {
                tracked_job.mark_started();
            }
            // Process the job
            let job_start = Instant::now();

            match process_comparison(
                job.clone(), 
                &config, 
                &mut tracker, 
                job_index, 
                &memory_manager, 
                &job_config,
                progress_manager.clone()  // PASS THE PROGRESS MANAGER
            ) {
                Ok(match_count) => {
                    // Check for early escape
                    if match_count == usize::MAX {
                        // This was an early escape - mark accordingly
                        info!("Job {} terminated early due to high similarity detection", job_number);
                        
                        progress_manager.early_escape_job(
                            job_number, 
                            format!("{} vs {} - Likely duplicate databases",
                                job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                                job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
                            )
                        ).unwrap_or_else(|e| warn!("Failed to update progress: {}", e));
                        
                    } else {
                        // Normal completion                       
                        // Include duration in the completion
                        let duration = job_start.elapsed();
                        progress_manager.set_message(
                            job_number, 
                            format!("✔ Complete: {} matches found ({})", 
                                match_count, 
                                format_duration(duration))
                        ).ok();
                        
                        progress_manager.complete_job(job_number, match_count)
                            .unwrap_or_else(|e| warn!("Failed to complete job progress: {}", e));
                    }
                },
                Err(e) => {
                    error!("Job {} failed: {}", job_number, e);
                    
                    progress_manager.fail_job(job_number, e.to_string())
                        .unwrap_or_else(|e| warn!("Failed to mark job as failed: {}", e));
                }
            }
            
            // Periodic resource check
            if job_index > 0 && job_index % 10 == 0 {
                info!("Periodic resource check after {} jobs:", job_index);
                check_system_resources();
            }
        }
    }
    
    // ========================================================================
    // POST-PROCESSING (same for both modes)
    // ========================================================================
        
    // Generate final report
    info!("========================================");
    info!("All comparisons complete!");
    info!("========================================");
    
    let final_report = tracker.generate_report();
    println!("\n{}", final_report);
    
    // Save final state
    tracker.save()?;
    info!("Job tracker saved successfully");
    
    // Log execution time
    let total_duration = program_start.elapsed();
    info!("Total execution time: {}", format_duration(total_duration));
    
    // Final resource summary
    if use_parallel {
        info!("========================================");
        info!("PARALLEL EXECUTION SUMMARY");
        info!("========================================");
        info!("Execution mode: {} strategy", config.execution.parallel_strategy);
        info!("Max parallel comparisons: {}", config.execution.max_parallel_comparisons);
        info!("Database pooling: {}", if config.execution.enable_db_pooling { "enabled" } else { "disabled" });
        info!("Final resource state:");
        info!("  {}", SHARED_STATE.get_comparison_status());
        memory_manager.log_memory_status();
    } else {
        info!("========================================");
        info!("SEQUENTIAL EXECUTION SUMMARY");
        info!("========================================");
        memory_manager.log_memory_status();
    }
    
    // Check for termination one last time
    if spot_handler.is_terminated() {
        warn!("Program terminated due to spot instance termination signal");
        info!("Final state saved to job tracker");
    }
    
    Ok(())
}