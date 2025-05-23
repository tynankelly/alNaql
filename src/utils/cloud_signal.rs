// utils/cloud_signal.rs
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use log::{info, warn, error};

use crate::error::{Error, Result};
use crate::utils::job_tracker::{JobStatus, JobTracker};
use crate::utils::logger::Logger;
use crate::storage::lmdb::LMDBStorage;

/// Polling interval for spot termination checks
const SPOT_CHECK_INTERVAL: Duration = Duration::from_secs(2);
/// EC2 metadata server address
const EC2_METADATA_HOST: &str = "169.254.169.254";
/// Spot termination path
const SPOT_TERMINATION_PATH: &str = "/latest/meta-data/spot/termination-time";

/// Manages spot instance termination detection and graceful shutdown
pub struct SpotTerminationHandler {
    /// Whether termination has been detected
    termination_detected: Arc<AtomicBool>,
    /// Whether the handler is running
    running: Arc<AtomicBool>,
    /// The termination detector thread
    detector_thread: Option<JoinHandle<()>>,
    /// Database paths to close on termination
    db_paths: Vec<PathBuf>,
    /// Job tracker for saving state
    job_tracker: Option<Arc<Mutex<JobTracker>>>,
    /// Logger for events
    logger: Arc<Mutex<Logger>>,
    /// When termination was detected
    termination_time: Arc<Mutex<Option<Instant>>>,
    /// Count of jobs interrupted by termination
    interrupted_job_count: Arc<AtomicUsize>,
}

impl SpotTerminationHandler {
    /// Create a new spot termination handler
    pub fn new(logger: Arc<Mutex<Logger>>) -> Self {
        Self {
            termination_detected: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            detector_thread: None,
            db_paths: Vec::new(),
            job_tracker: None,
            logger,
            termination_time: Arc::new(Mutex::new(None)),
            interrupted_job_count: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Register a job tracker to save on termination
/// Returns a Result to indicate success or failure
pub fn register_job_tracker(&mut self, tracker: Arc<Mutex<JobTracker>>) -> Result<()> {
    // Validate the job tracker by attempting to lock it
    let lock_result = match tracker.try_lock() {
        Ok(guard) => {
            // Get basic information about the job tracker
            let job_count = guard.jobs.len();
            let pending_jobs = guard.jobs.iter().filter(|j| j.status == JobStatus::Pending).count();
            
            // Log successful validation
            info!("Successfully validated job tracker with {} total jobs ({} pending)",
                job_count, pending_jobs);
            
            // Try a test save to ensure write access works
            match guard.save() {
                Ok(_) => {
                    info!("Successfully verified job tracker save functionality");
                    true
                },
                Err(e) => {
                    error!("Job tracker save validation failed: {}", e);
                    // We'll still register it, but with a warning
                    warn!("Registering job tracker despite save validation failure");
                    false
                }
            }
        },
        Err(e) => {
            error!("Failed to validate job tracker - could not acquire lock: {:?}", e);
            return Err(Error::async_err(format!("Job tracker validation failed: {:?}", e)));
        }
    };
    
    // If validation passed, register the tracker
    self.job_tracker = Some(tracker);
    
    // Log registration with information about validation
    if lock_result {
        info!("Job tracker successfully registered with termination handler");
    } else {
        warn!("Job tracker registered with warnings - spot termination handling may be degraded");
    }
    
    // Add association note to log
    if let Ok(mut logger_guard) = self.logger.lock() {
        if lock_result {
            let _ = logger_guard.log("Spot termination handler registered job tracker: fully validated");
        } else {
            let _ = logger_guard.log("Spot termination handler registered job tracker: with warnings");
        }
    }
    
    Ok(())
}
    
    /// Register database paths to clean up on termination
    pub fn register_databases(&mut self, paths: Vec<PathBuf>) {
        self.db_paths.extend(paths);
    }
    
    /// Start the termination detector
    pub fn start(&mut self) -> Result<()> {
        // Don't start if already running
        if self.detector_thread.is_some() {
            return Ok(());
        }
        
        // Set up shared state
        let termination_detected = self.termination_detected.clone();
        let running = self.running.clone();
        let db_paths = self.db_paths.clone();
        let job_tracker = self.job_tracker.clone();
        let logger = self.logger.clone();
        let termination_time = self.termination_time.clone();
        let interrupted_job_count = self.interrupted_job_count.clone(); // Clone this too
        
        // Mark as running
        self.running.store(true, Ordering::SeqCst);
        
        // Start detector thread
        self.detector_thread = Some(thread::spawn(move || {
            info!("Starting spot termination detector thread");
            
            // Log the startup
            if let Ok(mut logger_guard) = logger.lock() {
                let _ = logger_guard.log("Spot termination detector started");
            }
            
            // Check if we're running on EC2
            let is_ec2 = match check_if_ec2() {
                true => {
                    info!("Running on EC2, spot termination detection active");
                    true
                },
                false => {
                    info!("Not running on EC2, spot termination detection passive");
                    false
                }
            };
            
            // Main detector loop
            while running.load(Ordering::SeqCst) {
                // Only check for termination if we're on EC2
                if is_ec2 && !termination_detected.load(Ordering::SeqCst) {
                    if check_spot_termination() {
                        // Termination detected!
                        info!("⚠️ SPOT TERMINATION NOTICE DETECTED ⚠️");
                        
                        // Update shared state
                        termination_detected.store(true, Ordering::SeqCst);
                        
                        // Record termination time
                        if let Ok(mut guard) = termination_time.lock() {
                            *guard = Some(Instant::now());
                        }
                        
                        // Log the termination notice
                        if let Ok(mut logger_guard) = logger.lock() {
                            let _ = logger_guard.log("⚠️ SPOT TERMINATION NOTICE DETECTED ⚠️");
                            let _ = logger_guard.log("Starting graceful shutdown procedures");
                        }
                        
                        // Execute shutdown sequence
                        info!("Starting graceful shutdown sequence");
                        
                        // First, save job tracker state and mark in-progress jobs
                        if let Some(tracker) = &job_tracker {
                            info!("Saving job tracker state and marking in-progress jobs");
                            
                            // Use our helper function to acquire the lock with timeout
                            match try_lock_with_timeout(tracker, Duration::from_secs(5)) {
                                Some(mut guard) => {
                                    // Find in-progress jobs
                                    let mut interrupted_count = 0;
                                    for job in &mut guard.jobs {
                                        if job.status == JobStatus::InProgress {
                                            // Mark with termination message
                                            let now = SystemTime::now();
                                            let time_str = chrono::DateTime::<chrono::Local>::from(now)
                                                .format("%Y-%m-%d %H:%M:%S").to_string();
                                            
                                            job.error_message = Some(format!(
                                                "Job interrupted by spot instance termination at {}", time_str
                                            ));
                                            
                                            // Keep InProgress status for proper resumption detection
                                            job.last_checkpoint = Some(now);
                                            interrupted_count += 1;
                                        }
                                    }
                                    
                                    if interrupted_count > 0 {
                                        info!("Marked {} jobs as interrupted", interrupted_count);
                                        interrupted_job_count.fetch_add(interrupted_count, Ordering::SeqCst);
                                    }
                                    
                                    // Save the tracker
                                    if let Err(e) = guard.save() {
                                        error!("Failed to save job tracker during termination: {}", e);
                                    } else {
                                        info!("Job tracker saved successfully with interruption markers");
                                    }
                                },
                                None => {
                                    error!("Failed to acquire job tracker lock within timeout during termination");
                                }
                            }
                        } else {
                            warn!("No job tracker registered, skipping job state handling");
                        }
                        
                        // Close databases with retries and verification
                        info!("Initiating database shutdown sequence");
                        let db_count = db_paths.len();
                        info!("Attempting to close {} databases", db_count);

                        // Track closure results
                        let mut successful_closures = 0;
                        let mut failed_closures = 0;

                        // Set retry parameters
                        let max_retries = 3;
                        let retry_delay = Duration::from_millis(200);

                        for path in &db_paths {
                            info!("Closing database at {:?}", path);
                            let mut success = false;
                            
                            // Attempt closure with retries
                            for attempt in 1..=max_retries {
                                // Remove from cache
                                LMDBStorage::remove_from_cache(path);
                                
                                // Check if successful
                                if LMDBStorage::verify_closed(path) {
                                    info!("Successfully closed database: {:?} (attempt {})", path, attempt);
                                    success = true;
                                    break;
                                }
                                
                                // If not successful and more retries remain
                                if attempt < max_retries {
                                    warn!("Database closure attempt {} failed for {:?}, retrying after delay...", 
                                        attempt, path);
                                        
                                    // Increase delay for each retry attempt
                                    thread::sleep(retry_delay.mul_f32(attempt as f32));
                                    
                                    // Force memory cleanup between attempts
                                    drop(Vec::<u8>::with_capacity(1024 * 1024 * (5 * attempt))); // 5MB, 10MB, 15MB...
                                    
                                    // Try more aggressively on subsequent attempts
                                    info!("Using aggressive cleanup on retry {}", attempt + 1);
                                    LMDBStorage::remove_from_cache(path);
                                    
                                    // Additional sleep to give OS time to release resources
                                    thread::sleep(Duration::from_millis(100));
                                }
                            }
                            
                            // Final verification and accounting
                            if success {
                                successful_closures += 1;
                            } else {
                                failed_closures += 1;
                                error!("Failed to close database {:?} after {} attempts", path, max_retries);
                                
                                // Log emergency error
                                if let Ok(mut logger_guard) = logger.lock() {
                                    let message = format!(
                                        "EMERGENCY: Failed to properly close database at {:?} - possible corruption risk", 
                                        path
                                    );
                                    let _ = logger_guard.log(&message);
                                }
                                
                                // Last resort: try to force cleanup using OS-specific methods
                                #[cfg(target_os = "linux")]
                                {
                                    warn!("Attempting OS-level cleanup for database at {:?}", path);
                                    let lock_path = path.join("LOCK");
                                    if lock_path.exists() {
                                        // Try to invoke sync to flush filesystem buffers
                                        if let Err(e) = std::process::Command::new("sync").status() {
                                            error!("Failed to run 'sync' command: {}", e);
                                        }
                                        
                                        warn!("Database at {:?} might require manual recovery on instance restart", path);
                                    }
                                }
                            }
                        }

                        // Report overall database closure results
                        info!("Database closure complete: {} succeeded, {} failed", 
                            successful_closures, failed_closures);

                        if failed_closures > 0 {
                            warn!("Some databases failed to close properly. Manual verification recommended on restart.");
                            
                            if let Ok(mut logger_guard) = logger.lock() {
                                let message = format!(
                                    "WARNING: {} of {} databases did not close cleanly during termination",
                                    failed_closures, db_count
                                );
                                let _ = logger_guard.log(&message);
                            }
                        } else {
                            info!("All databases closed successfully");
                        }
                        
                        // Log completion
                        info!("Graceful shutdown sequence completed");
                        if let Ok(mut logger_guard) = logger.lock() {
                            let _ = logger_guard.log("Graceful shutdown sequence completed");
                        }
                    }
                }
                
                // Sleep before next check
                thread::sleep(SPOT_CHECK_INTERVAL);
            }
            
            info!("Termination detector thread stopping");
        }));
        
        info!("Spot termination handler started");
        Ok(())
    }
    
    /// Check if a termination has been detected
    pub fn is_terminated(&self) -> bool {
        self.termination_detected.load(Ordering::SeqCst)
    }
    
    /// Get the time since termination was detected
    pub fn time_since_termination(&self) -> Option<Duration> {
        if let Ok(guard) = self.termination_time.lock() {
            guard.map(|time| time.elapsed())
        } else {
            None
        }
    }
    /// Get detailed information about the termination event
    pub fn get_termination_details(&self) -> String {
        if !self.termination_detected.load(Ordering::SeqCst) {
            return "No termination detected".to_string();
        }
        
        // Get time of termination
        let time_str = if let Ok(guard) = self.termination_time.lock() {
            if let Some(time) = *guard {
                let elapsed = time.elapsed();
                let datetime = chrono::DateTime::<chrono::Local>::from(
                    std::time::SystemTime::now() - elapsed
                );
                format!("{} ({} ago)", 
                    datetime.format("%Y-%m-%d %H:%M:%S"), 
                    self.format_duration(elapsed))
            } else {
                "Unknown time".to_string()
            }
        } else {
            "Cannot access termination time".to_string()
        };
        
        // Get count of interrupted jobs
        let job_count = self.interrupted_job_count.load(Ordering::SeqCst);
        
        format!(
            "Termination detected at: {}\n\
            Jobs interrupted: {}\n\
            Current status: {}", 
            time_str,
            job_count,
            if self.running.load(Ordering::SeqCst) { "Handler still running" } else { "Handler stopped" }
        )
    }

    // Helper function to format duration in a human-readable way
    fn format_duration(&self, duration: Duration) -> String {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            format!("{}s", total_secs)
        } else if total_secs < 3600 {
            format!("{}m {}s", total_secs / 60, total_secs % 60)
        } else {
            format!("{}h {}m {}s", 
                total_secs / 3600, 
                (total_secs % 3600) / 60, 
                total_secs % 60)
        }
    }
    
    /// Stop the termination detector
    pub fn stop(&mut self) -> Result<()> {
        // Signal the thread to stop
        self.running.store(false, Ordering::SeqCst);
        
        // Wait for the thread to finish
        if let Some(thread) = self.detector_thread.take() {
            if let Err(e) = thread.join() {
                error!("Failed to join termination detector thread: {:?}", e);
                // Using the correct error type from your error system
                return Err(Error::async_err(format!("Failed to join termination detector thread: {:?}", e)));
            }
        }
        
        info!("Spot termination handler stopped");
        Ok(())
    }
    
}

impl Drop for SpotTerminationHandler {
    fn drop(&mut self) {
        // Stop the thread when dropped
        let _ = self.stop();
    }
}

/// Check if we're running on an EC2 instance
fn check_if_ec2() -> bool {
    // Simple check - just try connecting to the metadata service
    match TcpStream::connect((EC2_METADATA_HOST, 80)) {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Check if a spot termination notice has been issued
fn check_spot_termination() -> bool {
    // Try to connect to the EC2 metadata service with a short timeout
    match TcpStream::connect((EC2_METADATA_HOST, 80)) {
        Ok(mut stream) => {
            // Set a timeout
            let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            
            // Send HTTP request
            let request = format!(
                "GET {} HTTP/1.0\r\nHost: {}\r\n\r\n",
                SPOT_TERMINATION_PATH, EC2_METADATA_HOST
            );
            
            if let Err(_) = stream.write_all(request.as_bytes()) {
                return false;
            }
            
            // Read response
            let mut buffer = [0; 1024];
            match stream.read(&mut buffer) {
                Ok(size) if size > 0 => {
                    let response = String::from_utf8_lossy(&buffer[..size]);
                    
                    // If we get a 200 response, termination is imminent
                    response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200")
                },
                _ => false,
            }
        },
        Err(_) => false,
    }
    
}
/// Attempt to acquire a mutex lock with timeout to avoid deadlocks
/// Returns Some(guard) if successful, None if timeout occurs
fn try_lock_with_timeout<T>(mutex: &Mutex<T>, timeout: Duration) -> Option<std::sync::MutexGuard<T>> {
    let start_time = Instant::now();
    
    while start_time.elapsed() < timeout {
        match mutex.try_lock() {
            Ok(guard) => return Some(guard),
            Err(_) => {
                // Short sleep before retry to reduce CPU usage
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
    
    // If we get here, the timeout occurred
    None
}