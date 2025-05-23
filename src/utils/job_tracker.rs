// src/utils/job_tracker.rs
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use serde::{Serialize, Deserialize};
use log::{info, warn};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write, ErrorKind, Error as IoError};
use std::error::Error as StdError;
use std::thread;
use std::fmt;
use chrono::Local;
use crate::KuttabConfig;

/// Custom error type for file operations
#[derive(Debug)]
pub enum FileError {
    IoError(IoError),
    LockError(String),
    CsvError(csv::Error),
    ParseError(String),
    TimeoutError,
    Other(String),
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::IoError(e) => write!(f, "I/O error: {}", e),
            FileError::LockError(msg) => write!(f, "File lock error: {}", msg),
            FileError::CsvError(e) => write!(f, "CSV error: {}", e),
            FileError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            FileError::TimeoutError => write!(f, "Operation timed out"),
            FileError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl StdError for FileError {}

impl From<IoError> for FileError {
    fn from(error: IoError) -> Self {
        FileError::IoError(error)
    }
}

impl From<csv::Error> for FileError {
    fn from(error: csv::Error) -> Self {
        FileError::CsvError(error)
    }
}

/// Result type for file operations
pub type FileResult<T> = std::result::Result<T, FileError>;

/// A simple file locking mechanism for safe concurrent access
pub struct FileLock {
    lock_path: PathBuf,
    lock_file: Option<File>,
    acquired: bool,
}

impl FileLock {
    /// Create a new file lock for the given path
    pub fn new(path: &Path) -> Self {
        let lock_path = path.with_extension("lock");
        FileLock {
            lock_path,
            lock_file: None,
            acquired: false,
        }
    }
    
    /// Try to acquire the lock with timeout
    pub fn acquire(&mut self, timeout_ms: u64) -> FileResult<()> {
        if self.acquired {
            return Ok(());
        }
        
        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.lock_path)
            {
                Ok(file) => {
                    self.lock_file = Some(file);
                    self.acquired = true;
                    info!("Acquired lock: {:?}", self.lock_path);
                    return Ok(());
                },
                Err(ref e) if e.kind() == ErrorKind::AlreadyExists => {
                    // Lock exists, check if it's stale (older than 5 minutes)
                    if let Ok(metadata) = fs::metadata(&self.lock_path) {
                        if let Ok(modified) = metadata.modified() {
                            if let Ok(duration) = SystemTime::now().duration_since(modified) {
                                if duration > Duration::from_secs(300) { // 5 minutes
                                    // Stale lock, try to remove it
                                    info!("Removing stale lock: {:?}", self.lock_path);
                                    if let Err(e) = fs::remove_file(&self.lock_path) {
                                        return Err(FileError::LockError(format!("Failed to remove stale lock: {}", e)));
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                    
                    // Check timeout
                    if start.elapsed() > timeout {
                        return Err(FileError::TimeoutError);
                    }
                    
                    // Wait before retrying
                    thread::sleep(Duration::from_millis(100));
                },
                Err(e) => {
                    return Err(FileError::LockError(format!("Failed to create lock file: {}", e)));
                }
            }
        }
    }
    
    /// Release the lock
    pub fn release(&mut self) -> FileResult<()> {
        if self.acquired {
            // Close file handle
            self.lock_file = None;
            
            // Remove lock file
            if let Err(e) = fs::remove_file(&self.lock_path) {
                if e.kind() != ErrorKind::NotFound {
                    return Err(FileError::LockError(format!("Failed to remove lock file: {}", e)));
                }
            }
            
            self.acquired = false;
            info!("Released lock: {:?}", self.lock_path);
        }
        
        Ok(())
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Ensure lock is released when object is dropped
        if self.acquired {
            if let Err(e) = self.release() {
                warn!("Failed to release lock on drop: {}", e);
            }
        }
    }
}
// Add this implementation to the FileError enum in job_tracker.rs
impl FileError {
    /// Add context to an error
    pub fn with_context(self, context: String) -> Self {
        match self {
            FileError::IoError(e) => 
                FileError::IoError(IoError::new(e.kind(), format!("{}: {}", context, e))),
            FileError::CsvError(e) => 
                FileError::CsvError(csv::Error::from(IoError::new(ErrorKind::Other, 
                                   format!("{}: {}", context, e)))),
            FileError::LockError(msg) => 
                FileError::LockError(format!("{}: {}", context, msg)),
            FileError::ParseError(msg) => 
                FileError::ParseError(format!("{}: {}", context, msg)),
            FileError::TimeoutError => 
                FileError::Other(format!("{}: Operation timed out", context)),
            FileError::Other(msg) => 
                FileError::Other(format!("{}: {}", context, msg)),
        }
    }
}

/// Load jobs from a CSV file with file locking protection
pub fn load_jobs_from_csv(path: &Path) -> FileResult<Vec<ComparisonJob>> {
    // Check if file exists
    if !path.exists() {
        // Check if backup exists
        let backup_path = path.with_extension("bak");
        if backup_path.exists() {
            info!("Main CSV not found, attempting to recover from backup: {}", 
                backup_path.display());
            return load_jobs_from_csv(&backup_path);
        }
        return Ok(Vec::new());
    }
    
    // Try to acquire lock
    let mut lock = FileLock::new(path);
    match lock.acquire(5000) { // 5 second timeout
        Ok(_) => {}
        Err(FileError::TimeoutError) => {
            warn!("Lock timeout. File may be in use. Attempting read-only access...");
            // Continue without lock for read-only operation
        }
        Err(e) => return Err(e),
    }
    
    // Open the file with retry logic
    let file = match retry_operation(|| File::open(path), 3, 100) {
        Ok(f) => f,
        Err(e) => {
            let _ = lock.release();
            
            // Check for backup if main file is corrupted
            let backup_path = path.with_extension("bak");
            if backup_path.exists() {
                warn!("Failed to open main CSV: {}. Attempting recovery from backup", e);
                return load_jobs_from_csv(&backup_path);
            }
            
            return Err(FileError::IoError(e));
        }
    };
    
    let reader = BufReader::new(file);
    
    // Create CSV reader
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true) // Allow missing fields
        .trim(csv::Trim::All)
        .from_reader(reader);
    
    let mut jobs = Vec::new();
    let mut errors = Vec::new();
    
    // Read all records with error recovery
    for (i, result) in csv_reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                // Log error but continue parsing other records
                warn!("Error parsing record #{}: {}. Skipping.", i+1, e);
                errors.push(format!("Row {}: {}", i+1, e));
                continue;
            }
        };
        
        // Parse job fields (with robust error handling)
        if record.len() >= 7 {
            let source_path = PathBuf::from(record.get(0).unwrap_or_default());
            let target_path = PathBuf::from(record.get(1).unwrap_or_default());
            let output_path = PathBuf::from(record.get(2).unwrap_or_default());
            
            let comparison_number = match record.get(3).unwrap_or_default().parse::<usize>() {
                Ok(num) => num,
                Err(_) => {
                    warn!("Invalid comparison number in row {}, defaulting to 0", i+1);
                    0
                }
            };
            
            let status_str = record.get(4).unwrap_or_default();
            let status = match JobStatus::from_str(status_str) {
                Ok(s) => s,
                Err(_) => {
                    warn!("Invalid job status: {} in row {}, defaulting to Pending", status_str, i+1);
                    JobStatus::Pending
                }
            };
            
            let match_count = record.get(5)
                .unwrap_or_default()
                .parse::<usize>()
                .ok();
                
            let error_message = if !record.get(6).unwrap_or_default().is_empty() {
                Some(record.get(6).unwrap_or_default().to_string())
            } else {
                None
            };
            
            // Create job with basic fields
            let mut job = ComparisonJob {
                source_db_path: source_path,
                target_db_path: target_path,
                output_path,
                comparison_number,
                status,
                start_time: None,
                end_time: None,
                duration_seconds: None,
                peak_memory_mb: None,
                match_count,
                early_escape_detected: None,
                error_message,
                last_checkpoint: None,
                processed_chunks: None,
                total_chunks: None,
                last_processed_position: None,
            };
            
            // Parse optional fields if present
            if record.len() > 7 {
                job.duration_seconds = record.get(7)
                    .unwrap_or_default()
                    .parse::<f64>()
                    .ok();
            }
            
            if record.len() > 8 {
                job.peak_memory_mb = record.get(8)
                    .unwrap_or_default()
                    .parse::<u64>()
                    .ok();
            }
            
            if record.len() > 9 {
                job.early_escape_detected = match record.get(9).unwrap_or_default() {
                    "true" | "1" => Some(true),
                    "false" | "0" => Some(false),
                    _ => None,
                };
            }
            
            jobs.push(job);
        } else {
            warn!("Row {} has insufficient columns ({}), skipping", i+1, record.len());
            errors.push(format!("Row {}: insufficient columns", i+1));
        }
    }
    
    // Release lock
    lock.release()?;
    
    // Report any errors encountered
    if !errors.is_empty() {
        warn!("Encountered {} errors while loading CSV, but recovered {} jobs", 
              errors.len(), jobs.len());
    }
    
    info!("Loaded {} jobs from {}", jobs.len(), path.display());
    Ok(jobs)
}

/// Save jobs to a CSV file with atomic write and file locking
pub fn save_jobs_to_csv(jobs: &[ComparisonJob], path: &Path) -> FileResult<()> {
    // Create temporary file path
    let temp_path = path.with_extension("tmp");
    
    // Ensure directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    
    // Try to acquire lock
    let mut lock = FileLock::new(path);
    lock.acquire(5000)?; // 5 second timeout
    
    // Create backup if file exists
    if path.exists() {
        let backup_path = path.with_extension("bak");
        if let Err(e) = fs::copy(path, &backup_path) {
            warn!("Failed to create backup: {}", e);
        } else {
            info!("Created backup at {}", backup_path.display());
        }
    }
    
    // Create temporary file
    let file = match File::create(&temp_path) {
        Ok(f) => f,
        Err(e) => {
            // Release lock before returning error
            let _ = lock.release();
            return Err(FileError::IoError(e));
        }
    };
    let writer = BufWriter::new(file);
    
    // Create CSV writer - this doesn't return a Result, it returns the writer directly
    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_writer(writer);
    
    // Write headers - with error handling
    if let Err(e) = csv_writer.write_record(&[
        "source_db_path",
        "target_db_path",
        "output_path",
        "comparison_number",
        "status",
        "match_count",
        "error_message",
        "duration_seconds",
        "peak_memory_mb",
        "early_escape_detected"
    ]) {
        let _ = lock.release();
        return Err(FileError::CsvError(e));
    }
    
    // Write all job records with enhanced error handling
    for (i, job) in jobs.iter().enumerate() {
        if let Err(e) = csv_writer.write_record(&[
            job.source_db_path.to_string_lossy().to_string(),
            job.target_db_path.to_string_lossy().to_string(),
            job.output_path.to_string_lossy().to_string(),
            job.comparison_number.to_string(),
            job.status.as_str().to_string(),
            job.match_count.map_or("".to_string(), |c| c.to_string()),
            job.error_message.clone().unwrap_or_default(),
            job.duration_seconds.map_or("".to_string(), |d| d.to_string()),
            job.peak_memory_mb.map_or("".to_string(), |m| m.to_string()),
            job.early_escape_detected.map_or("".to_string(), |e| e.to_string()),
        ]) {
            let _ = lock.release();
            return Err(FileError::CsvError(e)
                .with_context(format!("Failed writing job {} of {}", i+1, jobs.len())));
        }
    }
    
    // Flush and close the writer with error handling
    if let Err(e) = csv_writer.flush() {
        let _ = lock.release();
        return Err(FileError::IoError(e));
    }
    drop(csv_writer);
    
    // Rename temp file to final destination (atomic operation)
    if let Err(e) = fs::rename(&temp_path, path) {
        let _ = lock.release();
        // Try to recover by copying instead of renaming
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            return Err(FileError::IoError(
                IoError::new(ErrorKind::Other, 
                    format!("Failed to rename AND copy temp file: rename error: {}, copy error: {}", e, copy_err))
            ));
        }
        // Succeeded with copy, try to clean up temp file
        let _ = fs::remove_file(&temp_path);
    }
    
    // Release lock
    lock.release()?;
    
    info!("Saved {} jobs to {}", jobs.len(), path.display());
    Ok(())
}

/// Find the most recent CSV file in a directory
pub fn find_latest_csv(dir: &Path) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    
    let mut latest_path = None;
    let mut latest_time = SystemTime::UNIX_EPOCH;
    
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            // Check if it's a CSV file
            if path.extension().map_or(false, |ext| ext == "csv") {
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > latest_time {
                            latest_time = modified;
                            latest_path = Some(path);
                        }
                    }
                }
            }
        }
    }
    
    latest_path
}
/// Helper function to retry operations with exponential backoff
fn retry_operation<T, F>(operation: F, max_attempts: u32, initial_delay_ms: u64) -> std::io::Result<T>
where
    F: Fn() -> std::io::Result<T>,
{
    let mut attempts = 0;
    let mut delay = initial_delay_ms;
    
    loop {
        attempts += 1;
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempts >= max_attempts {
                    return Err(e);
                }
                
                // Only retry for certain error types
                match e.kind() {
                    ErrorKind::PermissionDenied |
                    ErrorKind::ConnectionRefused |
                    ErrorKind::ConnectionReset |
                    ErrorKind::ConnectionAborted |
                    ErrorKind::Interrupted |
                    ErrorKind::WouldBlock => {
                        warn!("Retry attempt {}/{}: {} (waiting {}ms)", 
                              attempts, max_attempts, e, delay);
                        thread::sleep(Duration::from_millis(delay));
                        delay *= 2; // Exponential backoff
                    },
                    _ => return Err(e), // Don't retry other error types
                }
            }
        }
    }
}
/// Status of a comparison job
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl JobStatus {
    /// Convert to a string representation for the CSV file
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Pending => "pending",
            JobStatus::InProgress => "in_progress",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Skipped => "skipped",
        }
    }

    /// Parse from a string
    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(JobStatus::Pending),
            "in_progress" => Ok(JobStatus::InProgress),
            "completed" => Ok(JobStatus::Completed),
            "failed" => Ok(JobStatus::Failed),
            "skipped" => Ok(JobStatus::Skipped),
            _ => Err(format!("Unknown job status: {}", s)),
        }
    }
}

/// A job to be processed with enhanced tracking capabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonJob {
    // Original fields
    pub source_db_path: PathBuf,
    pub target_db_path: PathBuf,
    pub output_path: PathBuf,
    pub comparison_number: usize,
    
    // Status tracking
    pub status: JobStatus,
    
    // Timestamps
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<SystemTime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<SystemTime>,
    
    // Performance metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_mb: Option<u64>,
    
    // Results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub early_escape_detected: Option<bool>,
    
    // Error information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    
    // Checkpoint information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_checkpoint: Option<SystemTime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_processed_position: Option<u64>,
    
}

impl ComparisonJob {
    /// Create a new pending job
    pub fn new(
        source_db_path: PathBuf,
        target_db_path: PathBuf,
        output_path: PathBuf,
        comparison_number: usize,
    ) -> Self {
        Self {
            source_db_path,
            target_db_path,
            output_path,
            comparison_number,
            status: JobStatus::Pending,
            start_time: None,
            end_time: None,
            duration_seconds: None,
            peak_memory_mb: None,
            match_count: None,
            early_escape_detected: None,
            error_message: None,
            last_checkpoint: None,
            processed_chunks: None,
            total_chunks: None,
            last_processed_position: None,
        }
    }
    
    /// Mark the job as in progress and record start time
    pub fn mark_started(&mut self) {
        self.status = JobStatus::InProgress;
        self.start_time = Some(SystemTime::now());
    }
    
    /// Mark the job as completed and record end time and results
    pub fn mark_completed(&mut self, match_count: usize, early_escape: bool) {
        self.status = JobStatus::Completed;
        self.end_time = Some(SystemTime::now());
        self.match_count = Some(match_count);
        self.early_escape_detected = Some(early_escape);
        
        // Calculate duration if possible
        if let Some(start) = self.start_time {
            if let Ok(duration) = self.end_time.unwrap().duration_since(start) {
                self.duration_seconds = Some(duration.as_secs_f64());
            }
        }
        
        // Get peak memory if available
        self.peak_memory_mb = Self::get_current_memory_usage();
    }
    
    /// Mark the job as failed and record error information
    pub fn mark_failed(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.end_time = Some(SystemTime::now());
        self.error_message = Some(error);
        
        // Calculate duration if possible
        if let Some(start) = self.start_time {
            if let Ok(duration) = self.end_time.unwrap().duration_since(start) {
                self.duration_seconds = Some(duration.as_secs_f64());
            }
        }
        
        // Get peak memory if available
        self.peak_memory_mb = Self::get_current_memory_usage();
    }
    
    /// Mark the job as skipped (for already processed jobs)
    pub fn mark_skipped(&mut self, reason: Option<String>) {
        self.status = JobStatus::Skipped;
        if let Some(r) = reason {
            self.error_message = Some(r);
        }
    }
    
    /// Get current memory usage
    pub fn get_current_memory_usage() -> Option<u64> {
        if let Ok(mem_info) = sys_info::mem_info() {
            let used_mem_mb = (mem_info.total - mem_info.avail) / 1024;
            Some(used_mem_mb)
        } else {
            None
        }
    }
    
    /// Get display string for CSV
    pub fn get_source_display(&self) -> String {
        self.source_db_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }
    
    /// Get display string for CSV
    pub fn get_target_display(&self) -> String {
        self.target_db_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }
    /// Record a checkpoint for this job
    pub fn record_checkpoint(&mut self, chunks_processed: usize, total_chunks: usize, last_position: u64) {
        self.last_checkpoint = Some(SystemTime::now());
        self.processed_chunks = Some(chunks_processed);
        self.total_chunks = Some(total_chunks);
        self.last_processed_position = Some(last_position);
    }
    
    /// Check if this job appears to be interrupted
    pub fn is_interrupted(&self) -> bool {
        // A job is considered interrupted if:
        // 1. It's marked as in_progress
        // 2. It has a checkpoint recorded
        // 3. It doesn't have an end_time
        self.status == JobStatus::InProgress && 
        self.last_checkpoint.is_some() && 
        self.end_time.is_none()
    }
    
    /// Get the progress percentage based on checkpoint data
    pub fn get_progress_percentage(&self) -> Option<f64> {
        if let (Some(processed), Some(total)) = (self.processed_chunks, self.total_chunks) {
            if total > 0 {
                return Some((processed as f64 / total as f64) * 100.0);
            }
        }
        None
    }
    
    /// Get human-readable checkpoint information
    pub fn get_checkpoint_info(&self) -> String {
        if let Some(checkpoint_time) = self.last_checkpoint {
            let time_str = format_system_time(checkpoint_time);
            
            if let (Some(processed), Some(total)) = (self.processed_chunks, self.total_chunks) {
                let percentage = (processed as f64 / total as f64) * 100.0;
                return format!("Checkpoint at {} - Progress: {}/{} chunks ({:.1}%)", 
                               time_str, processed, total, percentage);
            } else {
                return format!("Checkpoint at {} - No progress data", time_str);
            }
        } else {
            "No checkpoint data".to_string()
        }
    }
    
    /// Get the time elapsed since the last checkpoint
    pub fn time_since_checkpoint(&self) -> Option<Duration> {
        if let Some(checkpoint) = self.last_checkpoint {
            SystemTime::now().duration_since(checkpoint).ok()
        } else {
            None
        }
    }
}

/// Helper function to format SystemTime as a human-readable string
fn format_system_time(time: SystemTime) -> String {
    // Check if time is valid (after UNIX epoch) without capturing the unused duration
    if time.duration_since(UNIX_EPOCH).is_ok() {
        // Convert to DateTime and format
        let datetime = chrono::DateTime::<chrono::Local>::from(time);
        datetime.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        "Invalid time".to_string()
    }
}
/// Result type for JobTracker operations
pub type JobResult<T> = std::result::Result<T, Box<dyn StdError>>;

/// JobTracker manages a collection of comparison jobs and their CSV representation
pub struct JobTracker {
    /// Path to the CSV file
    pub csv_path: PathBuf,
    /// List of all jobs
    pub jobs: Vec<ComparisonJob>,
}

impl JobTracker {
    /// Create a new JobTracker with a specified path or generate a timestamped one
    pub fn new(path: Option<&Path>) -> JobResult<Self> {
        let csv_path = match path {
            Some(p) => p.to_path_buf(),
            None => Self::generate_timestamped_path(),
        };
        
        // Ensure directory exists
        if let Some(parent) = csv_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        
        Ok(Self {
            csv_path,
            jobs: Vec::new(),
        })
    }
    
    /// Find and load any existing job CSV file in the specified directory
    pub fn load_existing(dir: &Path) -> JobResult<Option<Self>> {
        // Find CSV files in the directory
        if !dir.exists() {
            return Ok(None);
        }
        
        let mut newest_file = None;
        let mut newest_time = SystemTime::UNIX_EPOCH;
        
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            // Check if it's a CSV file
            if path.extension().map_or(false, |ext| ext == "csv") {
                // Check if it's a job tracking file (simple check for now)
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > newest_time {
                            newest_time = modified;
                            newest_file = Some(path);
                        }
                    }
                }
            }
        }
        
        // If found, load it
        if let Some(path) = newest_file {
            let mut tracker = JobTracker::new(Some(&path))?;
            tracker.load()?;
            Ok(Some(tracker))
        } else {
            Ok(None)
        }
    }
    
    /// Load jobs from the CSV file
    pub fn load(&mut self) -> JobResult<()> {
        if !self.csv_path.exists() {
            return Ok(());
        }
        
        match load_jobs_from_csv(&self.csv_path) {
            Ok(jobs) => {
                self.jobs = jobs;
                Ok(())
            },
            Err(e) => {
                Err(Box::new(e))
            }
        }
    }
    
    /// Save current jobs to the CSV file with file locking and atomic update
    pub fn save(&self) -> JobResult<()> {
        // Create a backup of the current CSV file
        self.create_backup()?;
        
        // Use our enhanced save_jobs_to_csv function
        match save_jobs_to_csv(&self.jobs, &self.csv_path) {
            Ok(_) => Ok(()),
            Err(e) => {
                warn!("Error saving job tracker: {}", e);
                Err(Box::new(e))
            }
        }
    }

    /// Create a backup of the current CSV file
    fn create_backup(&self) -> JobResult<()> {
        // Skip if the file doesn't exist yet
        if !self.csv_path.exists() {
            return Ok(());
        }
        
        // Create a timestamped backup name
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let backup_path = self.csv_path.with_file_name(
            format!("{}.{}.bak", 
                self.csv_path.file_stem().unwrap_or_default().to_string_lossy(),
                timestamp
            )
        );
        
        // Copy the file
        match fs::copy(&self.csv_path, &backup_path) {
            Ok(_) => {
                info!("Created timestamped backup at {}", backup_path.display());
                Ok(())
            },
            Err(e) => {
                warn!("Failed to create timestamped backup: {}", e);
                Err(Box::new(e))
            }
        }
    }
    /// Mark all interrupted jobs as failed
    pub fn mark_interrupted_jobs_failed(&mut self) -> JobResult<()> {
        // First collect just the indices to avoid borrowing issues
        let job_indices: Vec<usize> = self.check_for_interrupted_jobs()
            .into_iter()
            .map(|(index, _)| index)
            .collect();
        
        // Now use the collected indices to mark jobs as failed
        for job_index in job_indices {
            if let Some(job) = self.get_job_mut(job_index) {
                job.mark_failed("Job was interrupted and not resumed".to_string());
            }
        }
        
        self.save()
    }
    /// Save a checkpoint update for a specific job
    pub fn checkpoint_job(&mut self, job_index: usize, chunks_processed: usize, 
            total_chunks: usize, last_position: u64) -> JobResult<()> {
    if job_index >= self.jobs.len() {
    return Err(Box::new(FileError::Other(
    format!("Invalid job index: {} (max: {})", job_index, self.jobs.len())
    )));
    }

    // Update the job with checkpoint information
    if let Some(job) = self.jobs.get_mut(job_index) {
    job.record_checkpoint(chunks_processed, total_chunks, last_position);

    // Log checkpoint information
    info!("Checkpoint saved for job {} - {}", 
    job.comparison_number, job.get_checkpoint_info());

    // Save immediately to ensure persistence
    self.save()?;

    Ok(())
    } else {
    Err(Box::new(FileError::Other(format!("Failed to access job at index {}", job_index))))
    }
    }

    /// Check for and recover interrupted jobs
    pub fn check_for_interrupted_jobs(&self) -> Vec<(usize, &ComparisonJob)> {
    let mut interrupted = Vec::new();

    for (index, job) in self.jobs.iter().enumerate() {
    if job.is_interrupted() {
    interrupted.push((index, job));
    }
    }

    if !interrupted.is_empty() {
    info!("Found {} interrupted jobs that can be resumed", interrupted.len());

    // Log details about each interrupted job
    for (idx, (_, job)) in interrupted.iter().enumerate() {
    info!("Interrupted job #{}: {} vs {} - {}",
        idx + 1,
        job.source_db_path.file_stem().unwrap_or_default().to_string_lossy(),
        job.target_db_path.file_stem().unwrap_or_default().to_string_lossy(),
        job.get_checkpoint_info());
    }
    }

    interrupted
    }

    /// Resume an interrupted job
    pub fn prepare_job_for_resume(&mut self, job_index: usize) -> JobResult<u64> {
        if job_index >= self.jobs.len() {
            return Err(Box::new(FileError::Other(
                format!("Invalid job index: {} (max: {})", job_index, self.jobs.len())
            )));
        }
        
        // Get information from the job first, then modify it
        let resume_position;
        let job_number;
        {
            let job = &mut self.jobs[job_index];
            
            if !job.is_interrupted() {
                return Err(Box::new(FileError::Other(
                    format!("Job {} is not in an interrupted state", job.comparison_number)
                )));
            }
            
            // Get the last processed position to resume from
            resume_position = job.last_processed_position.unwrap_or(0);
            job_number = job.comparison_number;
            
            // Update job to reflect it's being resumed
            job.error_message = Some(format!(
                "Job was interrupted at {} and resumed at {}", 
                format_system_time(job.last_checkpoint.unwrap()),
                format_system_time(SystemTime::now())
            ));
        }
        
        // Now save the changes to the job tracker
        self.save()?;
        
        info!("Prepared job {} for resuming from position {}", job_number, resume_position);
        
        Ok(resume_position)
    }
    /// Find a job by source and target paths
    pub fn find_job(&self, source: &Path, target: &Path) -> Option<&ComparisonJob> {
        self.jobs.iter().find(|job| {
            job.source_db_path == source && job.target_db_path == target
        })
    }
    
    /// Find a job by its index
    pub fn get_job(&self, index: usize) -> Option<&ComparisonJob> {
        self.jobs.get(index)
    }
    
    /// Get a mutable reference to a job by its index
    pub fn get_job_mut(&mut self, index: usize) -> Option<&mut ComparisonJob> {
        self.jobs.get_mut(index)
    }
    
    /// Update a job's status by index
    pub fn update_job_status(&mut self, index: usize, status: JobStatus) -> JobResult<()> {
        if let Some(job) = self.jobs.get_mut(index) {
            job.status = status;
            self.save()?;
        }
        Ok(())
    }
    
    /// Add a new job
    pub fn add_job(&mut self, job: ComparisonJob) {
        self.jobs.push(job);
    }
    
    /// Generate a report of all jobs
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Job Tracker Report - {}\n", Local::now().format("%Y-%m-%d %H:%M:%S")));
        report.push_str(&format!("Total Jobs: {}\n\n", self.jobs.len()));
        
        // Count jobs by status
        let mut pending_count = 0;
        let mut in_progress_count = 0;
        let mut completed_count = 0;
        let mut failed_count = 0;
        let mut skipped_count = 0;
        
        for job in &self.jobs {
            match job.status {
                JobStatus::Pending => pending_count += 1,
                JobStatus::InProgress => in_progress_count += 1,
                JobStatus::Completed => completed_count += 1,
                JobStatus::Failed => failed_count += 1,
                JobStatus::Skipped => skipped_count += 1,
            }
        }
        
        report.push_str("Status Summary:\n");
        report.push_str(&format!("  Pending: {}\n", pending_count));
        report.push_str(&format!("  In Progress: {}\n", in_progress_count));
        report.push_str(&format!("  Completed: {}\n", completed_count));
        report.push_str(&format!("  Failed: {}\n", failed_count));
        report.push_str(&format!("  Skipped: {}\n\n", skipped_count));
        
        // Calculate averages for completed jobs
        let mut total_duration = 0.0;
        let mut total_memory: u64 = 0;
        let mut total_matches: usize = 0;
        let mut job_count: u64 = 0;
        
        for job in &self.jobs {
            if job.status == JobStatus::Completed {
                if let Some(duration) = job.duration_seconds {
                    total_duration += duration;
                }
                if let Some(memory) = job.peak_memory_mb {
                    total_memory += memory;
                }
                if let Some(matches) = job.match_count {
                    total_matches += matches;
                }
                job_count += 1;
            }
        }
        
        if job_count > 0 {
            report.push_str("Completed Job Statistics:\n");
            report.push_str(&format!("  Average Duration: {:.2} seconds\n", total_duration / job_count as f64));
            report.push_str(&format!("  Average Memory Usage: {} MB\n", total_memory / job_count));
            report.push_str(&format!("  Average Matches Found: {}\n\n", total_matches as u64 / job_count));
        }
        
        // Recent failed jobs
        let failed_jobs: Vec<&ComparisonJob> = self.jobs.iter()
            .filter(|j| j.status == JobStatus::Failed)
            .collect();
        
        if !failed_jobs.is_empty() {
            report.push_str("Recent Failed Jobs:\n");
            for (i, job) in failed_jobs.iter().take(5).enumerate() {
                report.push_str(&format!("  {}. {} vs {} - Error: {}\n", 
                    i + 1, 
                    job.get_source_display(), 
                    job.get_target_display(),
                    job.error_message.as_ref().unwrap_or(&"Unknown error".to_string())
                ));
            }
            report.push_str("\n");
        }
        
        report
    }
    /// Export job data to a CSV file with detailed statistics for analysis
    pub fn export_detailed_csv(&self, path: &Path) -> FileResult<()> {
        // Create the file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write CSV header with descriptive column names
        writeln!(writer, 
            "Comparison #,Source DB,Target DB,Status,Match Count,Duration (sec),Memory (MB),Early Escape,Start Time,End Time,Success Rate,Error Message"
        )?;
        
        // Write each job as a row with properly formatted fields
        for job in &self.jobs {
            // Get display names for source and target
            let source_name = job.get_source_display();
            let target_name = job.get_target_display();
            
            // Format status
            let status = job.status.as_str();
            
            // Format optional numeric fields
            let match_count = job.match_count.map_or("".to_string(), |c| c.to_string());
            let duration = job.duration_seconds.map_or("".to_string(), |d| format!("{:.2}", d));
            let memory = job.peak_memory_mb.map_or("".to_string(), |m| m.to_string());
            let early_escape = job.early_escape_detected.map_or("".to_string(), |e| e.to_string());
            
            // Calculate success rate (matches per second) if both values exist
            let success_rate = if let (Some(count), Some(dur)) = (job.match_count, job.duration_seconds) {
                if dur > 0.0 {
                    format!("{:.2}", count as f64 / dur)
                } else {
                    "".to_string()
                }
            } else {
                "".to_string()
            };
            
            // Format timestamps as human-readable dates
            let start_time = job.start_time.map_or("".to_string(), |t| {
                chrono::DateTime::<chrono::Local>::from(t)
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string()
            });
            
            let end_time = job.end_time.map_or("".to_string(), |t| {
                chrono::DateTime::<chrono::Local>::from(t)
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string()
            });
            
            // Clean up error message for CSV compatibility
            let error = job.error_message.clone().unwrap_or_default()
                .replace('\n', " ")   // Replace newlines with spaces
                .replace('"', "'");   // Replace double quotes with single quotes
            
            // Write the formatted row
            writeln!(writer, 
                "{},\"{}\",\"{}\",{},{},{},{},{},{},{},{},\"{}\"",
                job.comparison_number,
                source_name,
                target_name,
                status,
                match_count,
                duration,
                memory,
                early_escape,
                start_time,
                end_time,
                success_rate,
                error
            )?;
        }
        
        // Make sure all data is written
        writer.flush()?;
        
        Ok(())
    }
    
    /// Generate a timestamped CSV path in the state directory
    fn generate_timestamped_path() -> PathBuf {
        // Use the state directory from configuration if available
        let state_dir = if let Ok(config) = KuttabConfig::from_ini("default.ini") {
            PathBuf::from(&config.files.state_dir)
        } else {
            PathBuf::from("/data/state") // Fallback
        };
        
        // Create directory if it doesn't exist
        if !state_dir.exists() {
            let _ = std::fs::create_dir_all(&state_dir);
        }
        
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        state_dir.join(format!("jobs_{}.csv", timestamp))
    }
}

/// Find all database directories in the specified path
pub fn get_db_dirs(dir: &Path) -> FileResult<Vec<PathBuf>> {
    let mut db_paths = Vec::new();
    
    info!("Searching for databases in directory: {:?}", dir);
    if dir.exists() && dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            // Check for LMDB database directory structure:
            // 1. Must be a directory
            // 2. Must end with ".db" and not be "metadata.db"
            // 3. Must contain "data.mdb" (standard LMDB file)
            if path.is_dir() && 
               path.file_name()
                   .and_then(|s| s.to_str())
                   .map(|s| s.ends_with(".db") && s != "metadata.db")
                   .unwrap_or(false) &&
               path.join("data.mdb").exists() {
                info!("Found LMDB database: {:?}", path);
                db_paths.push(path);
            }
        }
    } else {
        warn!("Directory not found or not accessible: {:?}", dir);
    }
    
    db_paths.sort();
    info!("Total databases found: {}", db_paths.len());
    Ok(db_paths)
}

/// Generate an output path for a pair of databases based on their sizes
/// Always puts the larger database name first in the filename
pub fn generate_output_path(output_dir: &Path, source_db: &Path, target_db: &Path) -> PathBuf {
    let source_name = source_db.file_stem().unwrap_or_default().to_string_lossy();
    let target_name = target_db.file_stem().unwrap_or_default().to_string_lossy();
    
    // Calculate database sizes by checking the data.mdb file size, which is a good proxy for ngram count
    let source_size = get_db_size(source_db).unwrap_or(0);
    let target_size = get_db_size(target_db).unwrap_or(0);
    
    info!("Database sizes: {} = {}MB, {} = {}MB", 
          source_name, source_size / (1024*1024), 
          target_name, target_size / (1024*1024));
    
    // Put the larger database first in the filename
    let (first_name, second_name) = if source_size >= target_size {
        (source_name, target_name)
    } else {
        (target_name, source_name)
    };
    
    output_dir.join(format!("matches_{}_{}.jsonl", first_name, second_name))
}

/// Get the size of an LMDB database by checking its data.mdb file
fn get_db_size(db_path: &Path) -> std::io::Result<u64> {
    let data_path = db_path.join("data.mdb");
    if data_path.exists() {
        let metadata = std::fs::metadata(data_path)?;
        Ok(metadata.len())
    } else {
        // If data.mdb doesn't exist, return 0 as the size
        Ok(0)
    }
}

/// Create a unique comparison ID based on the job's position
pub fn create_job_id(source_idx: usize, target_idx: usize, target_count: usize) -> usize {
    source_idx * target_count + target_idx + 1
}

/// Check if a job already exists and has completed
pub fn is_job_already_completed(output_path: &Path) -> bool {
    if output_path.exists() {
        // Check if the file exists and has content
        match fs::metadata(&output_path) {
            Ok(metadata) => metadata.len() > 0,
            Err(_) => false,
        }
    } else {
        false
    }
}

/// Discover all comparison jobs between source and target databases
pub fn discover_jobs(
    source_dir: &Path, 
    target_dir: &Path, 
    output_dir: &Path, 
    self_comparison: bool
) -> FileResult<Vec<ComparisonJob>> {
    // Ensure output directory exists
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Find source databases
    let source_dbs = get_db_dirs(source_dir)?;
    if source_dbs.is_empty() {
        return Err(FileError::Other("No databases found in source directory".to_string()));
    }
    
    // Check if we're in self-comparison mode (explicit flag)
    if self_comparison {
        if source_dbs.len() < 2 {
            return Err(FileError::Other(
                "Self-comparison mode requires at least 2 source databases".to_string()
            ));
        }
        info!("Running in self-comparison mode with {} source databases", source_dbs.len());
        return discover_self_comparison_jobs(&source_dbs, output_dir);
    }
    
    // Check target directory for databases
    let target_dbs = get_db_dirs(target_dir)?;
    
    // If target directory is empty, automatically fall back to self-comparison mode
    if target_dbs.is_empty() {
        info!("No databases found in target directory - automatically falling back to self-comparison mode");
        if source_dbs.len() < 2 {
            return Err(FileError::Other(
                "Self-comparison requires at least 2 source databases".to_string()
            ));
        }
        info!("Running in self-comparison mode with {} source databases", source_dbs.len());
        return discover_self_comparison_jobs(&source_dbs, output_dir);
    }
    
    // Standard comparison mode - using target databases
    info!("Discovered {} source and {} target databases for comparison", 
         source_dbs.len(), target_dbs.len());
    
    // Create comparison jobs
    let mut jobs = Vec::new();
    let target_len = target_dbs.len();
    
    for (source_idx, source_db) in source_dbs.iter().enumerate() {
        for (target_idx, target_db) in target_dbs.iter().enumerate() {
            // Generate output file path
            let output_path = generate_output_path(output_dir, source_db, target_db);
            
            // Create unique job ID
            let comparison_number = create_job_id(source_idx, target_idx, target_len);
            
            // Create job
            let job = ComparisonJob::new(
                source_db.clone(),
                target_db.clone(),
                output_path,
                comparison_number,
            );
            
            jobs.push(job);
        }
    }
    
    info!("Created {} comparison jobs", jobs.len());
    Ok(jobs)
}

/// Discover jobs for self-comparison mode (comparing source databases against each other)
pub fn discover_self_comparison_jobs(
    source_dbs: &[PathBuf],
    output_dir: &Path
) -> FileResult<Vec<ComparisonJob>> {
    let mut jobs = Vec::new();
    let mut comparison_number = 1;
    
    // Create jobs avoiding redundant comparisons (each pair only once)
    for (i, source_db) in source_dbs.iter().enumerate() {
        for (j, target_db) in source_dbs.iter().enumerate() {
            // Skip self comparison and redundant comparisons (where i >= j)
            if i == j || i > j {
                continue;
            }
            
            // Generate output file path
            let output_path = generate_output_path(output_dir, source_db, target_db);
            
            // Create job
            let job = ComparisonJob::new(
                source_db.clone(),
                target_db.clone(),
                output_path,
                comparison_number,
            );
            
            jobs.push(job);
            comparison_number += 1;
        }
    }
    
    info!("Created {} self-comparison jobs", jobs.len());
    Ok(jobs)
}

/// Filter jobs by status (returns jobs that match any of the given statuses)
pub fn filter_jobs_by_status<'a>(jobs: &'a [ComparisonJob], statuses: &[JobStatus]) -> Vec<&'a ComparisonJob> {
    jobs.iter()
        .filter(|job| statuses.contains(&job.status))
        .collect()
}

/// Find pending jobs (including those that have failed previously)
pub fn find_pending_jobs<'a>(jobs: &'a [ComparisonJob]) -> Vec<&'a ComparisonJob> {
    filter_jobs_by_status(jobs, &[JobStatus::Pending, JobStatus::Failed])
}

/// Mark already completed jobs as skipped
pub fn mark_completed_jobs_as_skipped(jobs: &mut [ComparisonJob]) -> usize {
    let mut skipped_count = 0;
    
    for job in jobs.iter_mut() {
        if job.status == JobStatus::Pending && is_job_already_completed(&job.output_path) {
            job.mark_skipped(Some("Output file already exists".to_string()));
            skipped_count += 1;
        }
    }
    
    skipped_count
}

/// Merge existing jobs with newly discovered jobs
pub fn merge_jobs(
    existing_jobs: &[ComparisonJob], 
    new_jobs: &[ComparisonJob]
) -> Vec<ComparisonJob> {
    let mut merged_jobs = existing_jobs.to_vec();
    let mut added_count = 0;
    
    // Create a map of existing jobs for faster lookup
    let existing_job_map: std::collections::HashMap<(String, String), bool> = existing_jobs.iter()
        .map(|job| {
            let source = job.get_source_display();
            let target = job.get_target_display();
            ((source, target), true)
        })
        .collect();
    
    // Add new jobs that don't exist yet
    for job in new_jobs {
        let source = job.get_source_display();
        let target = job.get_target_display();
        
        if !existing_job_map.contains_key(&(source.clone(), target.clone())) {
            merged_jobs.push(job.clone());
            added_count += 1;
        }
    }
    
    info!("Merged {} existing jobs with {} new jobs (added {} new jobs)",
         existing_jobs.len(), new_jobs.len(), added_count);
    
    merged_jobs
}
