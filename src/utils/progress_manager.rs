// src/utils/progress_manager.rs
//! Centralized progress management for all comparison operations
//! 
//! This module provides a thread-safe, hierarchical progress tracking system
//! that manages all progress bars throughout the application.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use std::fmt;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle, ProgressDrawTarget};
use log::{debug, error, info};

use crate::error::Result;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Maximum number of detailed job displays in parallel mode
const MAX_PARALLEL_DISPLAY: usize = 4;

/// Template for the overall progress bar
const OVERALL_TEMPLATE: &str = "[{elapsed_precise}] {bar:40.red/blue} {pos}/{len} jobs | {msg} | ETA: {eta_precise}";

/// Template for job progress bars in sequential mode  
const JOB_TEMPLATE_SEQUENTIAL: &str = "Job {prefix} | {bar:40.white/blue} {percent}% | {elapsed_precise} | {msg}";

/// Template for job progress bars in parallel mode (compact)
const JOB_TEMPLATE_PARALLEL: &str = " {prefix} [{bar:30.white/red}] {percent}% | {msg}";

/// Template for phase progress bars
const PHASE_TEMPLATE: &str = "    {prefix} [{bar:35.yellow/blue}] {pos}/{len} | {msg} | ETA: {eta}";

/// Template for indeterminate operations
const SPINNER_TEMPLATE: &str = "    {prefix} {spinner:.cyan} {msg}";

// ============================================================================
// ENUMERATIONS
// ============================================================================

/// Execution mode for the application
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    Sequential,
    Parallel { max_concurrent: usize },
}

/// Different phases of job processing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProgressPhase {
    Initialization,
    GridProcessing,
    Sorting,
    Clustering,
    Validation,
    FinalMerge,
    Complete,
}

impl fmt::Display for ProgressPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization => write!(f, "Initializing"),
            Self::GridProcessing => write!(f, "Grid processing"),
            Self::Sorting => write!(f, "Sorting"),
            Self::Clustering => write!(f, "Clustering"),
            Self::Validation => write!(f, "Validation"),
            Self::FinalMerge => write!(f, "Final merge"),
            Self::Complete => write!(f, "Complete"),
        }
    }
}

/// Status of a job for display purposes
#[derive(Debug, Clone)]
pub enum JobStatus {
    Pending,
    Running(ProgressPhase),
    Completed { matches: usize, duration: Duration },
    Failed { error: String },
    EarlyEscape { reason: String },
}

// ============================================================================
// JOB PROGRESS TRACKING
// ============================================================================

/// Internal structure tracking progress for a single job
struct JobProgress {
    
    /// Display name for the job (e.g., "source1 vs target1")
    job_name: String,
    
    /// Main progress bar for the job
    main_bar: ProgressBar,
    
    /// Current phase-specific progress bar (if any)
    phase_bar: Option<ProgressBar>,
    
    /// Current processing phase
    current_phase: ProgressPhase,
    
    /// When the job started
    start_time: Instant,
    
    /// Current status
    status: JobStatus,
    
    /// Whether this job has a detailed display slot
    has_display_slot: bool,
    
    /// Position in the MultiProgress (for removal)
    display_position: Option<usize>,
    
    /// Match count (updated during processing)
    match_count: usize,

    bar_added_to_multi: bool,
}

impl JobProgress {
    /// Create a new job progress tracker
    fn new(job_name: String, main_bar: ProgressBar) -> Self {
        Self {
            job_name,
            main_bar,
            phase_bar: None,
            current_phase: ProgressPhase::Initialization,
            start_time: Instant::now(),
            status: JobStatus::Pending,
            has_display_slot: false,
            display_position: None,
            match_count: 0,
            bar_added_to_multi: false,
        }
    }
    
    /// Clean up progress bars when job completes
    fn cleanup(&self) {
        self.main_bar.finish_and_clear();
        if let Some(ref bar) = self.phase_bar {
            bar.finish_and_clear();
        }
    }
}

// ============================================================================
// MAIN PROGRESS MANAGER
// ============================================================================

/// Centralized progress manager for all operations
pub struct ProgressManager {
    /// Root MultiProgress instance
    multi: Arc<MultiProgress>,
    
    /// Overall progress bar (always visible)
    overall_bar: ProgressBar,
    
    /// Registry of active jobs
    active_jobs: Arc<RwLock<HashMap<usize, JobProgress>>>,
    
    /// Execution mode
    mode: ExecutionMode,
    
    /// Display slots for parallel mode (limited number)
    display_slots: Arc<Mutex<Vec<Option<usize>>>>,
    
    /// Counter for display positions
    next_position: Arc<Mutex<usize>>,
    
    /// Total number of jobs to process
    total_jobs: usize,
    
    /// Number of completed jobs
    completed_jobs: Arc<Mutex<usize>>,
    
    /// Whether progress display is suspended (for logging)
    suspended: Arc<Mutex<bool>>,

    /// Maps job_id (comparison_number) to display position (1, 2, 3...)
    job_position_map: Arc<Mutex<HashMap<usize, usize>>>,
    
    /// Next position to allocate
    next_display_position: Arc<Mutex<usize>>,
    
    /// Grand total of all jobs (including already completed)
    total_original_jobs: usize,

    /// Lock for overall bar updates to prevent concurrent redraws
    overall_bar_lock: Arc<Mutex<()>>,
    
}

impl ProgressManager {
pub fn new(
    total_jobs: usize,
    previously_completed: usize,
    mode: ExecutionMode
) -> Arc<Self> {
    let multi = MultiProgress::new();
    let total_original_jobs = total_jobs;
    
    // Calculate initial state
    let jobs_remaining = total_jobs - previously_completed;
    let initial_message = if previously_completed > 0 {
        format!("Resuming: {} complete, {} to process", previously_completed, jobs_remaining)
    } else {
        "Starting comparisons".to_string()
    };
    
    // Create overall progress bar with TOTAL job count
    let overall_bar = ProgressBar::new(total_original_jobs as u64);
    overall_bar.set_style(
        ProgressStyle::default_bar()
            .template(OVERALL_TEMPLATE)
            .unwrap()
            .progress_chars("=>-")
    );
    
    // ✅ ADD to MultiProgress FIRST (before setting position/message)
    let overall_bar = multi.add(overall_bar);
    
    // ✅ THEN configure initial state (now it will update in place)
    overall_bar.set_position(previously_completed as u64);
    overall_bar.set_message(initial_message);
    
    Arc::new(Self {
        multi: Arc::new(multi),
        overall_bar,
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            mode,
            display_slots: Arc::new(Mutex::new(vec![None; MAX_PARALLEL_DISPLAY])),
            next_position: Arc::new(Mutex::new(1)),
            total_jobs: total_jobs - previously_completed,
            completed_jobs: Arc::new(Mutex::new(previously_completed)),
            suspended: Arc::new(Mutex::new(false)),
            job_position_map: Arc::new(Mutex::new(HashMap::new())),
            next_display_position: Arc::new(Mutex::new(previously_completed + 1)),
            total_original_jobs,
            overall_bar_lock: Arc::new(Mutex::new(())),
        })
    }

    
    /// Start tracking a new job
    pub fn start_job(&self, job_id: usize, name: String) -> Result<()> {
        
        let display_position = self.allocate_position(job_id);
                
        // Create bar with hidden draw target to prevent premature rendering
        let main_bar = ProgressBar::new(100);
        main_bar.set_draw_target(ProgressDrawTarget::hidden()); // ← ADD THIS
        
        main_bar.set_style(
            ProgressStyle::default_bar()
                .template(match self.mode {
                    ExecutionMode::Sequential => JOB_TEMPLATE_SEQUENTIAL,
                    ExecutionMode::Parallel { .. } => JOB_TEMPLATE_PARALLEL,
                })
                .unwrap()
                .progress_chars("=>-")
        );
        
        let prefix = format!("{:>4}/{}", display_position, self.total_original_jobs);
        main_bar.set_prefix(prefix);
        
        main_bar.set_position(0);
        
        let short_name = if name.chars().count() > 60 {
            name.chars().take(57).collect::<String>() + "..."
        } else {
            name.clone()
        };
        main_bar.set_message(short_name.clone());
        
        // NOW add to MultiProgress (it will manage the draw target)
        let main_bar = self.multi.add(main_bar);
        
        // Enable steady tick for elapsed time updates
        main_bar.enable_steady_tick(Duration::from_millis(100));

        let mut job_progress = JobProgress::new(short_name, main_bar);
        job_progress.bar_added_to_multi = true;  
                    
        job_progress.status = JobStatus::Running(ProgressPhase::Initialization);
        
        if let ExecutionMode::Parallel { .. } = self.mode {
            if let Ok(mut slots) = self.display_slots.lock() {
                for (i, slot) in slots.iter_mut().enumerate() {
                    if slot.is_none() {
                        *slot = Some(job_id);
                        job_progress.has_display_slot = true;
                        debug!("Allocated display slot {} to job {}", i, job_id);
                        break;
                    }
                }
            }
        } else {
            job_progress.has_display_slot = true;
        }
        
        if let Ok(mut pos) = self.next_position.lock() {
            job_progress.display_position = Some(*pos);
            *pos += 1;
        }
        
        if let Ok(mut jobs) = self.active_jobs.write() {
            jobs.insert(job_id, job_progress);
        }
        
        Ok(())
    }
    
    /// Transition a job to a new phase
    pub fn enter_phase(&self, job_id: usize, phase: ProgressPhase) -> Result<()> {
        debug!("Job {} entering phase: {}", job_id, phase);
        
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(job) = jobs.get_mut(&job_id) {
                // Clear previous phase bar if exists
                if let Some(bar) = job.phase_bar.take() {
                    bar.finish_and_clear();
                }
                
                // Update phase and status
                job.current_phase = phase.clone();
                job.status = JobStatus::Running(phase.clone());
                
                // Update main bar message
                job.main_bar.set_message(format!("{}", job.job_name));
                
                // Create spinner for phases without progress bars (Sorting, Validation, FinalMerge)
                if job.has_display_slot {
                    let needs_spinner = matches!(
                        phase, 
                        ProgressPhase::Sorting | ProgressPhase::Validation | ProgressPhase::FinalMerge
                    );
                    
                    if needs_spinner {
                        let spinner = self.multi.insert_after(&job.main_bar, ProgressBar::new_spinner());
                        spinner.set_style(
                            ProgressStyle::default_spinner()
                                .template(SPINNER_TEMPLATE)
                                .unwrap()
                        );
                        spinner.set_prefix(format!("└─ {}", phase));
                        spinner.set_message("Processing...".to_string());
                        spinner.enable_steady_tick(Duration::from_millis(100));
                        job.phase_bar = Some(spinner);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub fn update_phase_progress(&self, job_id: usize, current: u64, total: u64, message: Option<String>) -> Result<()> {
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(job) = jobs.get_mut(&job_id) {
                // Create phase bar lazily if it doesn't exist yet
                if job.phase_bar.is_none() && job.has_display_slot {
                    let phase_bar = self.multi.insert_after(&job.main_bar, ProgressBar::new(total));
                    phase_bar.set_style(
                        ProgressStyle::default_bar()
                            .template(PHASE_TEMPLATE)
                            .unwrap()
                            .progress_chars("=>-")
                    );
                    phase_bar.set_prefix(format!("└─ {}", job.current_phase));
                    job.phase_bar = Some(phase_bar);
                }
                
                // Update phase bar if it exists
                if let Some(ref phase_bar) = job.phase_bar {
                    phase_bar.set_length(total);
                    phase_bar.set_position(current);
                    
                    if let Some(msg) = message {
                        phase_bar.set_message(msg);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Set a message for a job
    pub fn set_message(&self, job_id: usize, message: String) -> Result<()> {
        if let Ok(jobs) = self.active_jobs.read() {
            if let Some(job) = jobs.get(&job_id) {
                job.main_bar.set_message(message);
            }
        }
        Ok(())
    }
    
    /// Update match count for a job
    pub fn update_match_count(&self, job_id: usize, matches: usize) -> Result<()> {
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.match_count = matches;
            }
        }
        Ok(())
    }
    
    pub fn complete_job(&self, job_id: usize, matches: usize) -> Result<()> {
        info!("Completing job {} with {} matches", job_id, matches);
        
        // Update completed count
        let completed = {
            let mut completed = self.completed_jobs.lock().unwrap();
            *completed += 1;
            *completed
        };
        
        // Update overall progress bar (with lock to prevent duplicate redraws)
        {
            let _lock = self.overall_bar_lock.lock().unwrap();
            self.overall_bar.set_position(completed as u64);
            self.overall_bar.set_message(format!(
                "{}/{} jobs | {} complete ({} pending)", 
                completed,
                self.total_original_jobs,
                completed,
                self.total_original_jobs - completed
            ));
        }
                
        // Clean up job progress bars
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(mut job) = jobs.remove(&job_id) {
                // Calculate duration for logging
                let duration = job.start_time.elapsed();
                info!("Job {} completed: {} matches in {:.1}s", 
                    job_id, matches, duration.as_secs_f64());
                
                // Clear the main bar completely
                job.main_bar.finish_and_clear();
                
                // Clean up phase bar
                if let Some(phase_bar) = job.phase_bar.take() {
                    phase_bar.finish_and_clear();
                }
                                
                // Release display slot
                if let ExecutionMode::Parallel { .. } = self.mode {
                    if let Ok(mut slots) = self.display_slots.lock() {
                        for slot in slots.iter_mut() {
                            if *slot == Some(job_id) {
                                *slot = None;
                                debug!("Released display slot for job {}", job_id);
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Mark a job as failed
    pub fn fail_job(&self, job_id: usize, error: String) -> Result<()> {
        error!("Job {} failed: {}", job_id, error);
        
        // Update completed count (failed jobs count as completed)
        let completed = {
            let mut completed = self.completed_jobs.lock().unwrap();
            *completed += 1;
            *completed
        };
        
        // Update overall progress bar (with lock to prevent duplicate redraws)
        {
            let _lock = self.overall_bar_lock.lock().unwrap();
            self.overall_bar.set_position(completed as u64);
            self.overall_bar.set_message(format!(
                "Job {}/{} failed | {}/{} complete ({} pending)", 
                job_id, self.total_jobs, completed,
                self.total_original_jobs, self.total_original_jobs - completed
            ));
        }
        
        // Clean up job progress bars
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(mut job) = jobs.remove(&job_id) {
                job.status = JobStatus::Failed { error: error.clone() };
                
                // Set failure message
                job.main_bar.set_message(format!("✗ Failed: {}", error));
                job.main_bar.abandon_with_message(format!("✗ FAILED"));
                
                // Clean up
                job.cleanup();
                
                // Release display slot
                if let ExecutionMode::Parallel { .. } = self.mode {
                    if let Ok(mut slots) = self.display_slots.lock() {
                        for slot in slots.iter_mut() {
                            if *slot == Some(job_id) {
                                *slot = None;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Mark a job as early escaped (e.g., duplicate detection)
    pub fn early_escape_job(&self, job_id: usize, reason: String) -> Result<()> {
        info!("Job {} early escape: {}", job_id, reason);
        
        // Update completed count
        let completed = {
            let mut completed = self.completed_jobs.lock().unwrap();
            *completed += 1;
            *completed
        };
        
        // Update overall progress bar (with lock to prevent duplicate redraws)
        {
            let _lock = self.overall_bar_lock.lock().unwrap();
            self.overall_bar.set_position(completed as u64);
            self.overall_bar.set_message(format!(
                "Job {}/{} early escape | {}/{} complete ({} pending)",
                job_id, self.total_jobs, completed,
                self.total_original_jobs, self.total_original_jobs - completed
            ));
        }
        
        // Clean up job progress bars
        if let Ok(mut jobs) = self.active_jobs.write() {
            if let Some(mut job) = jobs.remove(&job_id) {
                job.status = JobStatus::EarlyEscape { reason: reason.clone() };
                
                job.main_bar.set_message(format!("⚡ Early escape: {}", reason));
                job.main_bar.finish_with_message("⚡ EARLY ESCAPE");
                
                job.cleanup();
                
                // Release display slot
                if let ExecutionMode::Parallel { .. } = self.mode {
                    if let Ok(mut slots) = self.display_slots.lock() {
                        for slot in slots.iter_mut() {
                            if *slot == Some(job_id) {
                                *slot = None;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Update the overall progress message
    pub fn update_overall_message(&self, message: String) {
        let _lock = self.overall_bar_lock.lock().unwrap();
        self.overall_bar.set_message(message);
    }
    
    /// Suspend progress display (for clean logging output)
    pub fn suspend(&self) {
        if let Ok(mut suspended) = self.suspended.lock() {
            *suspended = true;
            self.multi.set_draw_target(ProgressDrawTarget::hidden());
        }
    }
    
    /// Resume progress display
    pub fn resume(&self) {
        if let Ok(mut suspended) = self.suspended.lock() {
            *suspended = false;
            self.multi.set_draw_target(ProgressDrawTarget::stderr());
        }
    }
    
    /// Force a refresh of all progress bars
    pub fn refresh(&self) -> Result<()> {
        // MultiProgress handles refresh internally
        Ok(())
    }
    
    /// Finish all progress bars and clean up
    pub fn finish(self: Arc<Self>) {
        // Clean up any remaining active jobs
        if let Ok(mut jobs) = self.active_jobs.write() {
            for (_, job) in jobs.drain() {
                job.cleanup();
            }
        }
        
        // ✅ Finish overall bar WITH LOCK
        {
            let _lock = self.overall_bar_lock.lock().unwrap();
            self.overall_bar.finish_with_message("All jobs completed");
        }
    }

    /// Allocate a display position for a job
    fn allocate_position(&self, job_id: usize) -> usize {
        let mut map = self.job_position_map.lock().unwrap();
        
        // Check if already allocated
        if let Some(&pos) = map.get(&job_id) {
            return pos;
        }
        
        // Allocate new position
        let mut next_pos = self.next_display_position.lock().unwrap();
        let position = *next_pos;
        *next_pos += 1;
        
        map.insert(job_id, position);
        position
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let completed = *self.completed_jobs.lock().unwrap();
        let active = self.active_jobs.read().unwrap().len();
        (completed, active)
    }
    
    /// Check if all jobs are complete
    pub fn is_complete(&self) -> bool {
        *self.completed_jobs.lock().unwrap() >= self.total_jobs
    }
}

// Add this to src/utils/progress_manager.rs (at the end of the file, before tests)

// ============================================================================
// PROGRESS CONTEXT
// ============================================================================

/// Context for passing progress tracking through the call chain
#[derive(Clone)]
pub struct ProgressContext {
    manager: Arc<ProgressManager>,
    job_id: usize,
    phase_start_time: Arc<Mutex<Option<Instant>>>,
    phase_items_processed: Arc<Mutex<usize>>,
}

impl ProgressContext {
    /// Create a new progress context for a job
    pub fn new(manager: Arc<ProgressManager>, job_id: usize) -> Self {
        Self {
            manager,
            job_id,
            phase_start_time: Arc::new(Mutex::new(None)),
            phase_items_processed: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn enter_phase(&self, phase: ProgressPhase) {
        self.manager.enter_phase(self.job_id, phase.clone()).ok();
        
        // Reset phase tracking
        *self.phase_start_time.lock().unwrap() = Some(Instant::now());
        *self.phase_items_processed.lock().unwrap() = 0;
        
        // Update overall job progress based on phase START (not the phase's internal progress)
        let overall_percent = match phase {
            ProgressPhase::Initialization => 0,
            ProgressPhase::GridProcessing => 0,    // Grid starts at 0%
            ProgressPhase::Sorting => 50,          // Sorting starts at 50%
            ProgressPhase::Clustering => 60,       // Clustering starts at 60%
            ProgressPhase::Validation => 70,       // Validation starts at 70%
            ProgressPhase::FinalMerge => 80,       // Merge starts at 80%
            ProgressPhase::Complete => 100,
        };
        
        // This should update the MAIN job bar to the phase start percentage
        // NOT update it during phase processing
        if let Ok(jobs) = self.manager.active_jobs.read() {
            if let Some(job) = jobs.get(&self.job_id) {
                job.main_bar.set_position(overall_percent);
            }
        }
    }
    
    /// Update progress within current phase with ETA calculation
    pub fn update_progress(&self, current: u64, total: u64, message: Option<String>) {
    // ONLY update the phase progress bar, not the main job bar
    self.manager.update_phase_progress(
        self.job_id, 
        current, 
        total,
        message
    ).ok();
    }
    
    /// Update progress for grid processing with automatic frequency limiting
    pub fn update_grid_progress(&self, current_cell: usize, total_cells: usize, matches_so_far: usize) {
        // OPTIMIZATION: Combine both updates into a SINGLE lock acquisition
        // This reduces lock contention from 2 locks per update to 1 lock per update
        if let Ok(mut jobs) = self.manager.active_jobs.write() {
            if let Some(job) = jobs.get_mut(&self.job_id) {
                // Update match count directly (no extra lock needed)
                job.match_count = matches_so_far;
                
                // Update phase bar if it exists
                if job.phase_bar.is_none() && job.has_display_slot {
                    // Create phase bar lazily if needed
                    let phase_bar = self.manager.multi.add(ProgressBar::new(total_cells as u64));
                    phase_bar.set_style(
                        ProgressStyle::default_bar()
                            .template(PHASE_TEMPLATE)
                            .unwrap()
                            .progress_chars("=>-")
                    );
                    phase_bar.set_prefix(format!("└─ {}", job.current_phase));
                    job.phase_bar = Some(phase_bar);
                }
                
                if let Some(ref phase_bar) = job.phase_bar {
                    // Format message and update bar in one go
                    let message = format!(
                        "Cells processed | {} matches found",
                        matches_so_far
                    );
                    phase_bar.set_length(total_cells as u64);
                    phase_bar.set_position(current_cell as u64);
                    phase_bar.set_message(message);
                }
            }
        }
    }
    
    /// Update match count for the job
    pub fn update_matches(&self, count: usize) {
        self.manager.update_match_count(self.job_id, count).ok();
    }
    
    /// Update progress for clustering chunks
    pub fn update_cluster_progress(&self, current_chunk: usize, total_chunks: usize, clusters_so_far: usize) {
        let message = format!(
            "Processing chunk {}/{} | {} clusters so far",
            current_chunk, total_chunks, clusters_so_far
        );
        self.update_progress(current_chunk as u64, total_chunks as u64, Some(message));
    }
    
    /// Update progress for validation batches
    pub fn update_validation_progress(&self, current_batch: usize, total_batches: usize, validated_so_far: usize) {
        let message = format!(
            "Batch {} | {} validated",
            current_batch, validated_so_far
        );
        self.update_progress(current_batch as u64, total_batches as u64, Some(message));
    }
    
    /// Map phase progress to overall job progress
    pub fn map_phase_to_overall(&self, phase: &ProgressPhase, phase_percent: f64) -> u64 {
        let (base, range) = match phase {
            ProgressPhase::Initialization => (0, 0),
            ProgressPhase::GridProcessing => (0, 25),
            ProgressPhase::Sorting => (25, 15),
            ProgressPhase::Clustering => (40, 20),
            ProgressPhase::Validation => (60, 20),
            ProgressPhase::FinalMerge => (80, 20),
            ProgressPhase::Complete => (100, 0),
        };
        
        let overall = base + ((phase_percent / 100.0) * range as f64) as u64;
        overall.min(100)
    }
    
    /// Update overall job progress based on phase progress
    pub fn update_overall_from_phase(&self, phase: &ProgressPhase, current: u64, total: u64) {
        if total > 0 {
            let phase_percent = current as f64 / total as f64 * 100.0;
            let overall_percent = self.map_phase_to_overall(phase, phase_percent);
            
            // Update the main job bar
            if let Ok(jobs) = self.manager.active_jobs.read() {
                if let Some(job) = jobs.get(&self.job_id) {
                    job.main_bar.set_position(overall_percent);
                }
            }
        }
    }
}
// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a spinner for indeterminate operations
pub fn create_spinner(multi: &MultiProgress, message: &str) -> ProgressBar {
    let spinner = multi.add(ProgressBar::new_spinner());
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template(SPINNER_TEMPLATE)
            .unwrap()
    );
    spinner.set_message(message.to_string());
    spinner
}

/// Format duration for display
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}