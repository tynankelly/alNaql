// config/from_args.rs
//! Command-line argument parsing for job configuration
//!
//! This module handles parsing command-line arguments into JobConfig structures,
//! separating the CLI parsing logic from the configuration structure itself.

use std::path::PathBuf;
use super::job_track::JobConfig;

/// Parse command-line arguments into a JobConfig
/// 
/// # Returns
/// A validated JobConfig based on the provided command-line arguments
/// 
/// # Exits
/// - Exits with code 0 if --help is requested
/// - Exits with code 1 if validation fails
pub fn parse_job_config() -> JobConfig {
    let args: Vec<String> = std::env::args().collect();
    
    // Check for help flag first
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        JobConfig::print_help();
        std::process::exit(0);
    }
    
    // Parse arguments into config
    let config = parse_args_to_config(&args);
    
    // Validate the configuration
    if let Err(e) = config.validate() {
        eprintln!("Configuration error: {}", e);
        eprintln!("\nUse --help for usage information");
        std::process::exit(1);
    }
    
    config
}

/// Parse command-line arguments into a JobConfig without validation
/// 
/// This is useful for testing or when you want to handle validation separately
pub fn parse_args_to_config(args: &[String]) -> JobConfig {
    // Initialize with defaults
    let mut config = JobConfig::default();
    
    // Parse arguments (skip program name at index 0)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--job-file" => {
                if i + 1 < args.len() {
                    config.job_file = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                } else {
                    eprintln!("Warning: --job-file requires a path argument");
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
            "--reset" => {
                config.reset = true;
            },
            "--resume" => {  // ADD THIS CASE
                config.resume = true;
            },
            arg if arg.ends_with(".ini") => {
                // Assume any .ini file is a config file
                config.config_file = Some(arg.to_string());
            },
            arg if arg.starts_with("--") => {
                // Unknown flag - warn but don't fail
                eprintln!("Warning: Unknown option '{}' (ignored)", arg);
            },
            _ => {
                // Could be a positional argument or value for a previous flag
                // Currently we don't use positional arguments, so ignore
            }
        }
        i += 1;
    }
    
    config
}

/// Parse arguments and return a validated JobConfig or error message
/// 
/// This version returns a Result instead of exiting on error
pub fn try_parse_job_config() -> Result<JobConfig, String> {
    let args: Vec<String> = std::env::args().collect();
    
    // Check for help flag
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err("Help requested".to_string());
    }
    
    let config = parse_args_to_config(&args);
    config.validate()?;
    Ok(config)
}

/// Extract just the config file path from arguments without full parsing
/// 
/// This is useful when you need to load the INI file before full argument parsing
pub fn extract_config_file_from_args() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    
    for arg in &args[1..] {
        if arg.ends_with(".ini") {
            return Some(arg.clone());
        }
    }
    
    None
}

/// Check if help was requested without parsing all arguments
pub fn is_help_requested() -> bool {
    std::env::args().any(|arg| arg == "--help" || arg == "-h")
}