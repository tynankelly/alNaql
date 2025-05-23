use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::time::{Instant, Duration};
use chrono::Local;
use std::io::Write;
use std::thread;
use std::sync::{Arc, Mutex};
use kuttab::{
    KuttabConfig,
    storage::{
    lmdb::LMDBStorage,
    StorageBackend},
    matcher::ClusteringMatcher,
};
use kuttab::ngram::types::NGramEntry;
use kuttab::utils::job_tracker::JobStatus;
use kuttab::utils::job_tracker;
use kuttab::utils::Logger;
use kuttab::storage::lmdb::cache::METADATA_CACHE;
use kuttab::utils::memory_manager::{MemoryManager, MemoryState};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use log::{info, debug, warn, error, LevelFilter};
use env_logger::Builder;
use kuttab::utils::cloud_signal::SpotTerminationHandler;
/// Configuration for job processing
struct JobConfig {
    /// Path to specific job file (if provided)
    job_file: Option<PathBuf>,
    /// Whether to reprocess completed jobs
    reprocess_completed: bool,
    /// Whether to only retry failed jobs
    retry_failed: bool,
    /// Whether to skip jobs with existing output files
    skip_existing: bool,
    /// Whether to generate a detailed report
    generate_report: bool,
    /// Whether to list all jobs and their status
    list_jobs: bool,
    /// Use Rayon for parallel processing
    use_rayon: bool,
    /// Run in self-comparison mode
    self_comparison: bool,
    /// Path to configuration file
    config_file: Option<String>,
}

impl JobConfig {
    /// Parse command line arguments into configuration
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        
        // Initialize with defaults
        let mut config = JobConfig {
            job_file: None,
            reprocess_completed: false,
            retry_failed: false,
            skip_existing: true, // Default behavior
            generate_report: false,
            list_jobs: false,
            use_rayon: false,
            self_comparison: false,
            config_file: None,
        };
        
        // Parse arguments
        let mut i = 1; // Skip program name
        while i < args.len() {
            match args[i].as_str() {
                "--job-file" => {
                    if i + 1 < args.len() {
                        config.job_file = Some(PathBuf::from(&args[i + 1]));
                        i += 1;
                    }
                },
                "--reprocess-completed" => {
                    config.reprocess_completed = true;
                },
                "--retry-failed" => {
                    config.retry_failed = true;
                },
                "--skip-existing" => {
                    config.skip_existing = true;
                },
                "--no-skip-existing" => {
                    config.skip_existing = false;
                },
                "--generate-report" => {
                    config.generate_report = true;
                },
                "--list-jobs" => {
                    config.list_jobs = true;
                },
                "--rayon" => {
                    config.use_rayon = true;
                },
                "--self-comparison" => {
                    config.self_comparison = true;
                },
                arg if arg.ends_with(".ini") => {
                    config.config_file = Some(arg.to_string());
                },
                _ => {
                    // Unrecognized argument, just ignore
                }
            }
            i += 1;
        }
        
        config
    }
    
    /// Print help information about command line options
    fn print_help() {
        println!("Kuttab Match Finder - Command Line Options:");
        println!("  --job-file <path>        Use specific job tracking file");
        println!("  --reprocess-completed    Reprocess jobs that were already completed");
        println!("  --retry-failed           Only process jobs that previously failed");
        println!("  --skip-existing          Skip jobs with existing output files (default)");
        println!("  --no-skip-existing       Process all jobs even if output exists");
        println!("  --generate-report        Generate a detailed CSV report of all jobs");
        println!("  --list-jobs              Display all jobs and their status");
        println!("  --rayon                  Use Rayon for parallel processing");
        println!("  --self-comparison        Compare source databases against each other");
        println!("  <file.ini>               Use specified INI file for configuration");
        println!();
    }
    /// Validate the configuration for consistency
    /// Returns an error if the configuration is invalid
    fn validate(&self) -> Result<(), String> {
        // Check for incompatible options
        if self.reprocess_completed && self.retry_failed {
            return Err("Cannot use both --reprocess-completed and --retry-failed at the same time".to_string());
        }
        
        // Check that job file exists if specified
        if let Some(path) = &self.job_file {
            if !path.exists() {
                return Err(format!("Specified job file does not exist: {:?}", path));
            }
        }
        
        // Check that config file exists if specified
        if let Some(config_path) = &self.config_file {
            let path = Path::new(config_path);
            if !path.exists() {
                return Err(format!("Specified config file does not exist: {}", config_path));
            }
        }
        
        // Check for listing-only modes for correct display
        if self.list_jobs || self.generate_report {
            // These are fine by themselves, but let's warn about combinations that don't make sense
            if self.reprocess_completed || self.retry_failed {
                println!("Warning: The --reprocess-completed and --retry-failed flags have no effect when only listing jobs");
            }
        }
        
        Ok(())
    }
    
    /// Get list of statuses to process based on configuration
    pub fn get_statuses_to_process(&self) -> Vec<JobStatus> {
        let mut statuses = vec![JobStatus::Pending];
        
        if self.retry_failed {
            statuses.push(JobStatus::Failed);
        }
        
        if self.reprocess_completed {
            statuses.push(JobStatus::Completed);
        }
        
        // Always include in-progress jobs that might have been interrupted
        statuses.push(JobStatus::InProgress);
        
        statuses
    }
}


fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

fn process_comparison(
    job: job_tracker::ComparisonJob, 
    config: &KuttabConfig, 
    m: &MultiProgress,
    use_rayon: bool,  
    tracker: &mut job_tracker::JobTracker,
    job_index: usize,
    memory_manager: &Arc<MemoryManager> // New parameter
) -> Result<usize, String> {
    let start_time = Instant::now();
    
    // NEW: Log initial memory state
    info!("Memory state at start of job {}", job.comparison_number);
    memory_manager.log_memory_status();
    
    // Create a chunk progress bar
    let chunk_pb = m.add(ProgressBar::new(0));
    chunk_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} Chunks: [{wide_bar:.cyan/blue}] {pos}/{len} chunks ({eta})\n{msg}")
        .unwrap());
    
    // Update job status to in_progress and record start time
    if let Some(tracked_job) = tracker.get_job_mut(job_index) {
        tracked_job.mark_started();
        // Save the updated status
        if let Err(e) = tracker.save() {
            warn!("Failed to save job status: {}", e);
        }
    }
    
    // Track open database connections so we can explicitly close them
    let mut source_storage = None;
    let mut target_storage = None;
    
    // Track peak memory usage
    let initial_memory = job_tracker::ComparisonJob::get_current_memory_usage().unwrap_or(0);
    let mut peak_memory = initial_memory;
    
    // Set early escape threshold as a hardcoded value
    let early_escape_threshold = 0.9; // Hardcoded to 90%
    info!("Using early escape threshold: {:.2} ({}%)", 
         early_escape_threshold, early_escape_threshold * 100.0);
    
    // Use a result variable to track the outcome
    let result: Result<usize, String> = (|| {
        // Step 1: Open database connections for this job only
        debug!("Opening source database: {:?}", job.source_db_path);
        let source_db = match LMDBStorage::new(&job.source_db_path, config.storage.clone(), None) {
            Ok(storage) => storage,
            Err(e) => return Err(format!("Failed to open source database: {}", e)),
        };
        
        // Store for later cleanup
        source_storage = Some(source_db.clone());
        
        // NEW: Check memory after opening source database
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High || memory_state == MemoryState::AboveNormal {
            info!("Elevated memory pressure after opening source database: {}", memory_state);
            // Apply backpressure
            thread::sleep(Duration::from_millis(100));
        }
        
        debug!("Opening target database: {:?}", job.target_db_path);
        let target_db = match LMDBStorage::new(&job.target_db_path, config.storage.clone(), None) {
            Ok(storage) => storage,
            Err(e) => return Err(format!("Failed to open target database: {}", e)),
        };
        
        // Store for later cleanup
        target_storage = Some(target_db.clone());
        
        // NEW: Check memory after opening target database
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High {
            info!("High memory pressure after opening target database: {}", memory_state);
            info!("Applying backpressure to allow memory reclamation");
            
            // Force memory reclamation
            drop(Vec::<u8>::with_capacity(1024 * 1024 * 20)); // 20MB
            thread::sleep(Duration::from_millis(200));
        }
        
        // Step 2: Get database statistics
        let source_ngrams = match source_db.get_stats() {
            Ok(stats) => stats.total_ngrams,
            Err(e) => return Err(format!("Failed to get source stats: {}", e)),
        };
            
        let target_ngrams = match target_db.get_stats() {
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
        
        // NEW: Determine chunk size based on memory conditions
        let optimal_chunk_size = memory_manager.calculate_optimal_chunk_size();
        let chunk_size = optimal_chunk_size / std::mem::size_of::<NGramEntry>();
        info!("Using memory-optimized chunk size: {} ngrams", chunk_size);
        
        // Step 3: Create the matcher with potentially swapped databases
        let matcher = match ClusteringMatcher::new(
            final_source_db,
            final_target_db,
            config.matcher.clone(),
            config.processor.clone(),
            config.generator.ngram_type
        ) {
            Ok(matcher) => matcher,
            Err(e) => return Err(format!("Failed to create matcher: {}", e)),
        };
        
        // Check memory usage after matcher creation
        if let Some(current_memory) = job_tracker::ComparisonJob::get_current_memory_usage() {
            if current_memory > peak_memory {
                peak_memory = current_memory;
            }
        }
        
        // NEW: Check memory pressure after creating matcher
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High {
            info!("High memory pressure after creating matcher: {}", memory_state);
            // Force memory reclamation
            drop(Vec::<u8>::with_capacity(1024 * 1024 * 30)); // 30MB
            thread::sleep(Duration::from_millis(200));
        }
        
        // Step 4: Use the appropriate find_matches method based on use_rayon flag
        let match_count = if use_rayon {
            info!("Using Rayon-based parallel processing with work stealing");
            match matcher.find_matches_with_rayon(
                0..(final_source_ngrams + 1),
                0..(final_target_ngrams + 1),
                Some(&job.output_path),
                Some(&chunk_pb),
                early_escape_threshold
            ) {
                Ok(count) => {
                    // Check memory usage after processing
                    if let Some(current_memory) = job_tracker::ComparisonJob::get_current_memory_usage() {
                        if current_memory > peak_memory {
                            peak_memory = current_memory;
                        }
                    }
                    
                    // NEW: Check memory after main processing
                    memory_manager.log_memory_status();
                    
                    // Check for early escape special value
                    if count == usize::MAX {
                        // Early escape detected
                        chunk_pb.finish_with_message(
                            format!("Early termination: High similarity detected (>{:.1}%), likely duplicate databases", 
                                  early_escape_threshold * 100.0)
                        );
                        m.remove(&chunk_pb); // Add this line
                        // Return special value to indicate early escape
                        usize::MAX
                    } else {
                        // Normal completion
                        chunk_pb.finish_with_message(format!("Completed: found {} matches (using Rayon)", count));
                        m.remove(&chunk_pb); // Add this line
                        count
                    }
                },
                Err(e) => {
                    // Show error in progress bar
                    chunk_pb.finish_with_message(format!("Error with Rayon: {}", e));
                    m.remove(&chunk_pb); // Add this line
                    return Err(format!("Failed to find matches with Rayon: {}", e));
                },
            }
        } else {
            // Original processing method
            match matcher.find_matches(
                0..(final_source_ngrams + 1),
                0..(final_target_ngrams + 1),
                Some(&job.output_path),
                Some(&chunk_pb),
                early_escape_threshold
            ) {
                Ok(count) => {
                    // Check memory usage after processing
                    if let Some(current_memory) = job_tracker::ComparisonJob::get_current_memory_usage() {
                        if current_memory > peak_memory {
                            peak_memory = current_memory;
                        }
                    }
                    
                    // NEW: Check memory after main processing
                    memory_manager.log_memory_status();
                    
                    // Check for early escape special value
                    if count == usize::MAX {
                        // Early escape detected
                        chunk_pb.finish_with_message(
                            format!("Early termination: High similarity detected (>{:.1}%), likely duplicate databases", 
                                early_escape_threshold * 100.0)
                        );
                        m.remove(&chunk_pb); // Add this line
                        // Return special value to indicate early escape
                        usize::MAX
                    } else {
                        // Normal completion
                        chunk_pb.finish_with_message(format!("Completed: found {} matches", count));
                        m.remove(&chunk_pb); // Add this line
                        count
                    }
                },
                Err(e) => {
                    // Show error in progress bar
                    chunk_pb.finish_with_message(format!("Error: {}", e));
                    m.remove(&chunk_pb); // Add this line
                    return Err(format!("Failed to find matches: {}", e));
                },
            }
        };
        
        Ok(match_count)
    })();
    
    // NEW: Apply memory management before cleanup
    let memory_state = memory_manager.monitor_memory_thresholds();
    if memory_state == MemoryState::High || memory_state == MemoryState::AboveNormal {
        info!("Elevated memory pressure before database cleanup: {}", memory_state);
        info!("Forcing memory reclamation before database cleanup");
        drop(Vec::<u8>::with_capacity(1024 * 1024 * 50)); // 50MB
        thread::sleep(Duration::from_millis(300));
    }
    
    // Step 5: Ensure database connections are properly closed, regardless of success or failure
    info!("Closing database connections for job {}", job.comparison_number);
    
    // Close source database with enhanced logging
    if let Some(ref mut db) = source_storage {
        info!("Explicitly closing source database: {:?}", job.source_db_path);
        match db.close() {
            Ok(_) => info!("Source database successfully closed"),
            Err(e) => warn!("Error closing source database: {}", e),
        }
    }
    
    // Close target database with enhanced logging
    if let Some(ref mut db) = target_storage {
        info!("Explicitly closing target database: {:?}", job.target_db_path);
        match db.close() {
            Ok(_) => info!("Target database successfully closed"),
            Err(e) => warn!("Error closing target database: {}", e),
        }
    }
    
    // Step 6: Verify resources are released
    info!("Verifying database resources for job {}", job.comparison_number);
    // Clear thread-local caches to free memory
    info!("Clearing thread-local metadata caches for job {}", job.comparison_number);
    METADATA_CACHE.clear_thread_local();
    
    // NEW: Check memory after database closure
    info!("Memory status after database closure:");
    memory_manager.log_memory_status();
    
    // Verify source database is closed
    let source_closed = LMDBStorage::verify_closed(&job.source_db_path);
    if !source_closed {
        warn!("Source database may not be fully closed: {:?}", job.source_db_path);
        // Try again to remove from cache
        LMDBStorage::remove_from_cache(&job.source_db_path);
    }
    
    // Verify target database is closed
    let target_closed = LMDBStorage::verify_closed(&job.target_db_path);
    if !target_closed {
        warn!("Target database may not be fully closed: {:?}", job.target_db_path);
        // Try again to remove from cache
        LMDBStorage::remove_from_cache(&job.target_db_path);
    }
    
    // NEW: Apply additional cleanup after verification
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
    
    // NEW: Final memory management before returning
    info!("Final memory status for job {}:", job.comparison_number);
    memory_manager.log_memory_status();
    
    // Ensure thread-local caches are cleared between jobs
    METADATA_CACHE.clear_thread_local();
    // Add explicit pause to allow system to fully release resources
    thread::sleep(Duration::from_millis(200));
    m.remove(&chunk_pb);
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
    
    // Count active RocksDB instances in cache
    let db_cache_size = kuttab::storage::lmdb::DB_CACHE.len();
    status.push(format!("Active DB cache entries: {}", db_cache_size));
    
    status.join(" | ")
}

fn verify_database_cleanup(db_path: &Path) -> bool {
    info!("Verifying cleanup for database: {:?}", db_path);
    
    // Check if it's still in the global cache
    let in_cache = kuttab::storage::lmdb::DB_CACHE.contains_key(db_path);
    
    // Check lock file status
    let lock_path = db_path.join("LOCK");
    let lock_exists = lock_path.exists();
    
    // Try to open the lock file (will succeed if not locked)
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
    
    // Count .sst files to see if they're accessible
    let mut sst_count = 0;
    if let Ok(entries) = std::fs::read_dir(db_path) {
        for entry in entries.flatten() {
            if let Some(extension) = entry.path().extension() {
                if extension == "sst" {
                    sst_count += 1;
                }
            }
        }
    }
    
    // Log detailed status
    info!("Database {:?} cleanup status:", db_path);
    info!("  In global cache: {}", in_cache);
    info!("  Lock file exists: {}", lock_exists);
    info!("  Can open lock file: {}", can_open_lock);
    info!("  SST files count: {}", sst_count);
    
    let is_clean = !in_cache && (can_open_lock || !lock_exists);
    
    if !is_clean {
        warn!("Database {:?} may not be fully cleaned up!", db_path);
        
        // Attempt emergency cleanup
        if in_cache {
            warn!("Forcing removal from global cache");
            LMDBStorage::remove_from_cache(db_path);
        }
        
        // Try to check if any processes have the lock file open
        #[cfg(target_os = "linux")]
        {
            // Try using lsof to see what's holding the lock
            let lock_path_str = lock_path.to_string_lossy();
            let output = std::process::Command::new("lsof")
                .arg(lock_path_str.as_ref())
                .output();
                
            if let Ok(output) = output {
                if !output.stdout.is_empty() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    warn!("Processes holding lock file open:\n{}", stdout);
                } else {
                    info!("No processes found holding the lock file open");
                }
            }
        }
    } else {
        info!("Database {:?} is properly cleaned up", db_path);
    }
    
    is_clean
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments into config
    let job_config = JobConfig::from_args();

    // Print help if --help was specified
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        JobConfig::print_help();
        return Ok(());
    }
    
    // Validate the configuration
    if let Err(error) = job_config.validate() {
        eprintln!("Configuration error: {}", error);
        eprintln!("Use --help to see available options.");
        return Err(error.into());
    }

    // Initialize the memory manager with reasonable batch sizes
    let memory_manager = Arc::new(MemoryManager::new(500, 10000));
    info!("Memory manager initialized - checking initial memory status");
    memory_manager.log_memory_status();
    
    // Load configuration using found path or default
    let config = if let Some(config_path) = &job_config.config_file {
        info!("Loading configuration from: {}", config_path);
        KuttabConfig::from_ini(config_path)?
    } else {
        info!("Loading configuration from default.ini");
        KuttabConfig::from_ini("default.ini")?
    };
    
    // Set up logging with minimal configuration
    let timestamp = chrono::Local::now().format("%m_%d_%H_%M");
    fs::create_dir_all("logs")?;
    let log_file = File::create(format!("logs/matcher_{}.log", timestamp))?;

    // Convert config string to LevelFilter
    let log_level = match config.generator.ngram_log.to_lowercase().as_str() {
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        "none" => LevelFilter::Off,
        _ => {
            println!("Invalid log level '{}', defaulting to Info", config.generator.ngram_log);
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

    // Create the logger for spot termination handling
    let spot_logger = Arc::new(Mutex::new(Logger::with_file(
        Path::new("logs").join(format!("spot_handler_{}.log", 
            Local::now().format("%Y%m%d_%H%M%S"))))?
    ));

    // Create and set up termination handler
    let mut spot_handler = SpotTerminationHandler::new(spot_logger);
    
    // Set up directories
    let source_dir = Path::new(&config.files.ngrams_source_dir);
    let target_dir = Path::new(&config.files.ngrams_target_dir);
    let output_dir = Path::new(&config.files.output_dir);
    info!("Using output directory from config: {:?}", output_dir);

    // Make sure the directory exists
    fs::create_dir_all(output_dir)?;
    
    // Initialize or load job tracker
    let tracker = if let Some(path) = &job_config.job_file {
        // Use specified job file
        info!("Using specified job file: {:?}", path);
        let mut tracker = job_tracker::JobTracker::new(Some(path))?;
        tracker.load()?;
        tracker
    } else {
        // Look for existing job file in state directory
        let state_dir = Path::new(&config.files.state_dir);
        match job_tracker::JobTracker::load_existing(state_dir)? {
            Some(tracker) => {
                info!("Found existing job tracker with {} jobs", tracker.jobs.len());
                tracker
            },
            None => {
                // Create new tracker - using ngram directories correctly
                let mut tracker = job_tracker::JobTracker::new(None)?;
                
                // Now using the correct ngram directories
                let jobs = job_tracker::discover_jobs(
                    source_dir,  // Ngram source directory
                    target_dir,  // Ngram target directory
                    &Path::new(&config.files.output_dir),
                    job_config.self_comparison
                )?;
                
                // Add jobs to tracker
                for job in jobs {
                    tracker.add_job(job);
                }
                
                // Save the tracker
                tracker.save()?;
                info!("Created new job tracker with {} jobs", tracker.jobs.len());
                tracker
            }
        }
    };
    
    // Wrap tracker in Arc<Mutex> for sharing with spot handler
    let tracker_arc = Arc::new(Mutex::new(tracker));
    if let Err(e) = spot_handler.register_job_tracker(tracker_arc.clone()) {
        // Log the error
        error!("Failed to register job tracker with spot termination handler: {}", e);
        
        // Print to stderr for visibility
        eprintln!("ERROR: Could not register job tracker: {}", e);
        
        // Exit the program with error code since this is a critical initialization step
        // If you prefer not to exit, you could just log the error and continue with reduced functionality
        std::process::exit(1);
    }

    // Get the paths of all databases for shutdown handling
    let mut all_db_paths = Vec::new();
    for job in &tracker_arc.lock().unwrap().jobs {
        all_db_paths.push(job.source_db_path.clone());
        all_db_paths.push(job.target_db_path.clone());
    }
    // Make sure paths are unique
    all_db_paths.sort();
    all_db_paths.dedup();

    // Register database paths
    spot_handler.register_databases(all_db_paths);

    // Start the termination handler
    spot_handler.start()?;

    // Get tracker back for normal use
    let mut tracker = tracker_arc.lock().unwrap();

    // Generate and print report if requested
    if job_config.generate_report {
        let report = tracker.generate_report();
        println!("\n{}", report);
        
        // Save the report to a file
        let report_path = output_dir.join(format!("report_{}.txt", 
                                                 Local::now().format("%Y%m%d_%H%M%S")));
        let mut file = File::create(&report_path)?;
        file.write_all(report.as_bytes())?;
        println!("Report saved to {:?}", report_path);
        
        // Exit if we only want to generate a report
        if !job_config.list_jobs {
            return Ok(());
        }
    }
    
    // List jobs if requested
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
        println!("\nUse other command options to process jobs. Use --help for details.");
        
        // Exit if we only want to list jobs
        return Ok(());
    }
    
    // Get the list of jobs to process based on command line options
    let statuses_to_process = job_config.get_statuses_to_process();
    info!("Processing jobs with statuses: {:?}", statuses_to_process);

    // Check for interrupted jobs and determine what jobs to process
    let pending_jobs = {
        // Check for interrupted jobs
        let interrupted_jobs = tracker.check_for_interrupted_jobs();
        
        if interrupted_jobs.is_empty() {
            // No interrupted jobs, just get jobs to process
            tracker.jobs.iter()
                .filter(|job| statuses_to_process.contains(&job.status))
                .cloned()
                .collect()
        } else {
            // Collect information from interrupted jobs before mutating tracker
            let job_info: Vec<(usize, String, String, String)> = interrupted_jobs.iter()
                .map(|(idx, job)| (
                    *idx,
                    job.get_source_display(),
                    job.get_target_display(),
                    job.get_checkpoint_info()
                ))
                .collect();
            
            println!("\nFound {} interrupted jobs that can be resumed:", job_info.len());
            
            // Display information about each interrupted job
            for (i, (_, source, target, checkpoint_info)) in job_info.iter().enumerate() {
                println!("  {}. {} vs {} - {}", 
                        i + 1, 
                        source, 
                        target,
                        checkpoint_info);
            }
            
            // Prompt user
            println!("\nWould you like to resume these jobs? (y/n)");
            let mut input = String::new();
            let should_resume = if std::io::stdin().read_line(&mut input).is_ok() {
                let response = input.trim().to_lowercase();
                response == "y" || response == "yes"
            } else {
                false
            };
            
            if should_resume {
                info!("User chose to resume interrupted jobs");
                let mut jobs_to_resume = Vec::new();
                
                // Prepare each job for resumption using the collected indices
                for (job_index, _, _, _) in &job_info {
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

    let total_comparisons = jobs_to_process.len();
    info!("Found {} jobs to process", total_comparisons);
    
    // Set up progress tracking with enhanced styling
    let m = MultiProgress::new();

    // Set up overall progress tracking with more detailed template
    let overall_pb = m.add(ProgressBar::new(total_comparisons as u64));
    overall_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} Overall: [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | ETA: {eta_precise} | Elapsed: {elapsed_precise}\n{msg}")
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "));

    // Add a status line progress bar for current job details
    let status_pb = m.add(ProgressBar::new(100));
    status_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.red} Current: {prefix} | {wide_msg}")
        .unwrap());
    status_pb.set_prefix("Waiting to start...");
    status_pb.set_message("Preparing job queue");

    // Add a statistics progress bar to show ongoing stats
    let stats_pb = m.add(ProgressBar::new(100));
    stats_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.blue} Stats: {msg}")
        .unwrap());
    stats_pb.set_message(format!("Completed: 0 | Failed: 0 | Skipped: {} | Matches: 0", skipped_jobs));

    // Log initial system state
    info!("Initial system resources before processing:");
    let initial_status = check_system_resources();
    info!("{}", initial_status);
    
    // Track overall statistics - FIX: Make all counters the same type (usize)
    let start_time = Instant::now();
    let mut completed_jobs: usize = 0;
    let mut total_matches: usize = 0;
    let mut errors: Vec<(usize, String)> = Vec::new();
    let mut early_escape_count: usize = 0;
    
    // Process jobs sequentially
    info!("Processing {} comparisons sequentially", jobs_to_process.len());

    for job in jobs_to_process {
        let job_number = job.comparison_number;
        
        // Check for termination before starting each job
        if spot_handler.is_terminated() {
            info!("⚠️ AWS Spot Termination notice received! Saving state and exiting...");
            tracker.save()?;
            break;
        }
        
        // Find the index of this job in the tracker
        let job_index = match tracker.jobs.iter().position(|j| j.comparison_number == job_number) {
            Some(idx) => idx,
            None => {
                warn!("Job {} not found in tracker, skipping", job_number);
                continue;
            }
        };
        
        info!("Starting job {}/{}: source={:?}, target={:?}", 
              job_number, total_comparisons, 
              job.source_db_path.file_stem().unwrap_or_default(),
              job.target_db_path.file_stem().unwrap_or_default());
        
        // Check memory state before starting a new job
        info!("Checking memory state before job {}", job_number);
        memory_manager.log_memory_status();
        
        // Apply memory-aware processing
        let memory_state = memory_manager.monitor_memory_thresholds();
        if memory_state == MemoryState::High {
            info!("High memory pressure detected, forcing cleanup before job {}", job_number);
            // Force system to reclaim memory
            drop(Vec::<u8>::with_capacity(1024 * 1024 * 50)); // 50MB
            thread::sleep(Duration::from_millis(500));
        }
        
        // Update status progress bar at the start of each job
        status_pb.set_prefix(format!("Job {}/{}", job_number, total_comparisons));
        status_pb.set_message(format!(
            "⟳ PROCESSING: {} vs {}", 
            job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
            job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
        ));
        
        // Check for leftover database resources before starting
        let source_clean_before = LMDBStorage::verify_closed(&job.source_db_path);
        let target_clean_before = LMDBStorage::verify_closed(&job.target_db_path);
        
        if !source_clean_before || !target_clean_before {
            warn!("Database resources from previous jobs still present! Attempting cleanup...");
            LMDBStorage::remove_from_cache(&job.source_db_path);
            LMDBStorage::remove_from_cache(&job.target_db_path);
            thread::sleep(Duration::from_secs(1));
        }
        
        // Log detailed resource state before job
        info!("System resources before job {}:", job_number);
        let before_status = check_system_resources();
        info!("{}", before_status);
        
        // Process the job
        let job_start = Instant::now();
        
        match process_comparison(job.clone(), &config, &m, job_config.use_rayon, &mut tracker, job_index, &memory_manager) {
            Ok(match_count) => {
                completed_jobs += 1;
                
                // Check for early escape
                if match_count == usize::MAX {
                    // This was an early escape - mark accordingly
                    early_escape_count += 1;
                    info!("Job {} terminated early due to high similarity detection", job_number);
                    
                    // Update status progress bar
                    status_pb.set_prefix(format!("Job {}/{}", job_number, total_comparisons));
                    status_pb.set_message(format!(
                        "✓ EARLY ESCAPE: {} vs {} - Likely duplicate databases", 
                        job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                        job.target_db_path.file_stem().unwrap_or_default().to_string_lossy()
                    ));
                    
                    // Update overall progress bar
                    overall_pb.set_position(completed_jobs as u64);
                    overall_pb.set_message(format!(
                        "Job {}/{}: Early escape detected ({} total)", 
                        job_number, 
                        total_comparisons,
                        early_escape_count
                    ));
                } else {
                    // Normal completion
                    total_matches += match_count;
                    
                    // Update status progress bar
                    status_pb.set_prefix(format!("Job {}/{}", job_number, total_comparisons));
                    status_pb.set_message(format!(
                        "✓ COMPLETED: {} vs {} - Found {} matches in {}", 
                        job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                        job.target_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                        match_count,
                        format_duration(job_start.elapsed())
                    ));
                    
                    // Update overall progress bar
                    overall_pb.set_position(completed_jobs as u64);
                    overall_pb.set_message(format!(
                        "Progress: {}/{} jobs - {} matches so far - Running for {}", 
                        completed_jobs, 
                        total_comparisons,
                        total_matches,
                        format_duration(start_time.elapsed())
                    ));
                    
                    info!("Completed job {}: Found {} matches in {}", 
                         job_number, match_count, format_duration(job_start.elapsed()));
                }
                
                // Update stats progress bar
                stats_pb.set_message(format!(
                    "Completed: {} | Failed: {} | Skipped: {} | Early Escape: {} | Total Matches: {}", 
                    (completed_jobs - errors.len()), // Fixed: All usize now
                    errors.len(),
                    skipped_jobs,
                    early_escape_count,
                    total_matches
                ));
            },
            Err(error) => {
                completed_jobs += 1;
                errors.push((job_number, error.clone()));
                
                // Update status progress bar
                status_pb.set_prefix(format!("Job {}/{}", job_number, total_comparisons));
                status_pb.set_message(format!(
                    "⨯ FAILED: {} vs {} - Error: {}", 
                    job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                    job.target_db_path.file_stem().unwrap_or_default().to_string_lossy(),
                    error
                ));
                
                // Update overall progress bar
                overall_pb.set_position(completed_jobs as u64);
                overall_pb.set_message(format!(
                    "Job {}/{}: Error encountered ({} total errors)", 
                    job_number, 
                    total_comparisons,
                    errors.len()
                ));
                
                // Update stats progress bar
                stats_pb.set_message(format!(
                    "Completed: {} | Failed: {} | Skipped: {} | Early Escape: {} | Total Matches: {}", 
                    (completed_jobs - errors.len()), // Fixed: All usize now
                    errors.len(),
                    skipped_jobs,
                    early_escape_count,
                    total_matches
                ));
                
                warn!("Error in job {}: {}", job_number, error);
            }
        }
        
        // Thorough verification and emergency cleanup after job
        info!("Running thorough resource verification after job {}", job_number);
        let source_clean = verify_database_cleanup(&job.source_db_path);
        let target_clean = verify_database_cleanup(&job.target_db_path);
        
        // Handle cleanup issues between jobs
        if !source_clean || !target_clean {
            warn!("Database cleanup issues detected after job {}. Applying extra cleanup measures.", job_number);
            
            // Force cache clearing
            info!("Forcing removal from global cache");
            LMDBStorage::remove_from_cache(&job.source_db_path);
            LMDBStorage::remove_from_cache(&job.target_db_path);
            
            // Add a longer pause for system to recover
            info!("Pausing for extended cleanup...");
            thread::sleep(Duration::from_secs(2));
            
            // Check again after extended cleanup
            let source_clean_retry = LMDBStorage::verify_closed(&job.source_db_path);
            let target_clean_retry = LMDBStorage::verify_closed(&job.target_db_path);
            
            if !source_clean_retry || !target_clean_retry {
                warn!("Persistent database issues detected even after extended cleanup!");
            } else {
                info!("Extended cleanup successful");
            }
        }
        
        // Log detailed resource state after job
        info!("System resources after job {}:", job_number);
        let after_status = check_system_resources();
        info!("{}", after_status);

        // Check memory status after job completion
        info!("Memory status after job {} completion:", job_number);
        memory_manager.log_memory_status();

        // Add memory-aware delay between jobs
        let memory_state = memory_manager.monitor_memory_thresholds();
        let pause_duration = match memory_state {
            MemoryState::High => {
                info!("HIGH memory pressure after job. Applying extended cleanup pause.");
                // Force additional cleanup
                drop(Vec::<u8>::with_capacity(1024 * 1024 * 100)); // 100MB
                Duration::from_secs(2)
            },
            MemoryState::AboveNormal => {
                info!("ABOVE NORMAL memory pressure after job. Applying standard cleanup pause.");
                Duration::from_secs(1)
            },
            _ => {
                info!("Normal memory state after job. Applying regular pause.");
                Duration::from_millis(500)
            }
        };

        info!("Pausing for {:?} between jobs to allow memory reclamation", pause_duration);
        thread::sleep(pause_duration);
    }
    
    // Final resource check
    info!("Final system resources:");
    let final_status = check_system_resources();
    info!("{}", final_status);
    
    // Display final statistics
    let total_duration = start_time.elapsed();
    info!("\nProcessing complete!");
    info!("Total time: {}", format_duration(total_duration));
    info!("Total matches found: {}", total_matches);
    info!("Successful jobs: {}/{}", (completed_jobs - errors.len() - early_escape_count), total_comparisons);  
    info!("Skipped jobs (already processed): {}/{}", skipped_jobs, total_comparisons);
    info!("Early escapes (likely duplicates): {}/{}", early_escape_count, total_comparisons);
    
    if !errors.is_empty() {
        info!("Errors encountered: {}", errors.len());
        for (job_num, error) in &errors {
            warn!("Job {}: {}", job_num, error);
        }
    }
    
    // Generate and save final report to text file
    let final_report = tracker.generate_report();
    let report_path = output_dir.join(format!("final_report_{}.txt", 
                                            Local::now().format("%Y%m%d_%H%M%S")));
    let mut file = File::create(&report_path)?;
    file.write_all(final_report.as_bytes())?;
    info!("Final report saved to {:?}", report_path);

    // Export detailed statistics to CSV
    let csv_path = output_dir.join(format!("detailed_stats_{}.csv", 
                                        Local::now().format("%Y%m%d_%H%M%S")));
    if let Err(e) = tracker.export_detailed_csv(&csv_path) {
        warn!("Failed to export detailed CSV report: {}", e);
    } else {
        info!("Detailed statistics exported to {:?}", csv_path);
    }
    // Final metadata cache cleanup
    METADATA_CACHE.clear_thread_local();
    METADATA_CACHE.print_stats();

    // Complete the progress bar message
    overall_pb.finish_with_message(format!(
        "Complete! Found {} matches in {} jobs ({} successful, {} failed, {} skipped, {} duplicates)", 
        total_matches,
        total_comparisons,
        (completed_jobs - errors.len() - early_escape_count),
        errors.len(),
        skipped_jobs,
        early_escape_count
    ));

    // Also complete the other progress bars
    status_pb.finish_with_message("All jobs completed");
    stats_pb.finish_with_message(format!(
        "Final Results: {} matches found - {} jobs processed",
        total_matches,
        completed_jobs
    ));

    spot_handler.stop()?;

    Ok(())
}