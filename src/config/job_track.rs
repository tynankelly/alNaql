// config/job_track.rs
//! Job tracking configuration module
//! 
//! This module contains the configuration structure and logic for job processing,
//! including options for reprocessing, retrying, and filtering jobs by status.

use std::path::{Path, PathBuf};
use crate::utils::job_tracker::JobStatus;

/// Configuration for job processing
#[derive(Debug, Clone)]
pub struct JobConfig {
    /// Path to specific job file (if provided)
    pub job_file: Option<PathBuf>,
    /// Whether to reprocess completed jobs
    pub reprocess_completed: bool,
    /// Whether to only retry failed jobs
    pub retry_failed: bool,
    /// Whether to skip jobs with existing output files
    pub skip_existing: bool,
    /// Whether to generate a detailed report
    pub generate_report: bool,
    /// Whether to list all jobs and their status
    pub list_jobs: bool,
    /// Use Rayon for parallel processing
    pub use_rayon: bool,
    /// Run in self-comparison mode
    pub self_comparison: bool,
    /// Path to configuration file
    pub config_file: Option<String>,
    /// Whether to reset state and output directories
    pub reset: bool,
    /// Resume interrupted grid streaming runs from last completed stage
    pub resume: bool,
}

impl JobConfig {
    /// Create a new JobConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration for consistency
    /// Returns an error if the configuration is invalid
    pub fn validate(&self) -> Result<(), String> {
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
        
        // Reset is incompatible with retry/reprocess operations
        if self.reset && (self.reprocess_completed || self.retry_failed) {
            return Err("--reset cannot be used with --reprocess-completed or --retry-failed".to_string());
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

    /// Print help information about command line options
    pub fn print_help() {
        println!("alNaql Match Finder - Command Line Options:");
        println!("  --job-file <path>        Use specific job tracking file");
        println!("  --reprocess-completed    Reprocess jobs that were already completed");
        println!("  --retry-failed           Only process jobs that previously failed");
        println!("  --skip-existing          Skip jobs with existing output files (default)");
        println!("  --no-skip-existing       Process all jobs even if output exists");
        println!("  --generate-report        Generate a detailed CSV report of all jobs");
        println!("  --list-jobs              Display all jobs and their status");
        println!("  --rayon                  Use Rayon for parallel processing");
        println!("  --self-comparison        Compare source databases against each other");
        println!("  --reset                  Delete state and output directories for a fresh start");
        println!("      --resume                Resume interrupted grid streaming runs");
        println!("  <file.ini>               Use specified INI file for configuration");
        println!();
    }

    /// Check if only information display is requested (no actual processing)
    pub fn is_display_only(&self) -> bool {
        self.list_jobs || self.generate_report
    }

    /// Check if job processing should occur
    pub fn should_process_jobs(&self) -> bool {
        !self.list_jobs || !self.generate_report
    }

}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            job_file: None,
            reprocess_completed: false,
            retry_failed: false,
            skip_existing: true,
            generate_report: false,
            list_jobs: false,
            use_rayon: false,
            self_comparison: false,
            config_file: None,
            reset: false,
            resume: false,
        }
    }
}

/// Builder pattern for JobConfig to allow programmatic configuration
pub struct JobConfigBuilder {
    config: JobConfig,
}

impl JobConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: JobConfig::default(),
        }
    }

    /// Set the job file path
    pub fn job_file(mut self, path: PathBuf) -> Self {
        self.config.job_file = Some(path);
        self
    }

    /// Enable reprocessing of completed jobs
    pub fn reprocess_completed(mut self, enable: bool) -> Self {
        self.config.reprocess_completed = enable;
        self
    }

    /// Enable retrying of failed jobs only
    pub fn retry_failed(mut self, enable: bool) -> Self {
        self.config.retry_failed = enable;
        self
    }

    /// Set whether to skip existing output files
    pub fn skip_existing(mut self, enable: bool) -> Self {
        self.config.skip_existing = enable;
        self
    }

    /// Enable report generation
    pub fn generate_report(mut self, enable: bool) -> Self {
        self.config.generate_report = enable;
        self
    }

    /// Enable job listing
    pub fn list_jobs(mut self, enable: bool) -> Self {
        self.config.list_jobs = enable;
        self
    }

    /// Enable Rayon parallel processing
    pub fn use_rayon(mut self, enable: bool) -> Self {
        self.config.use_rayon = enable;
        self
    }

    /// Enable self-comparison mode
    pub fn self_comparison(mut self, enable: bool) -> Self {
        self.config.self_comparison = enable;
        self
    }

    /// Set configuration file path
    pub fn config_file(mut self, path: String) -> Self {
        self.config.config_file = Some(path);
        self
    }

    /// Enable reset mode
    pub fn reset(mut self, enable: bool) -> Self {
        self.config.reset = enable;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> Result<JobConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}