// config/debug.rs - Debug and logging configuration

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::{Error, Result};
use crate::config::loader::Configurable;

/// Configuration for debugging and diagnostic logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Log level for n-gram operations ("error", "warn", "info", "debug", "trace", "none")
    pub ngram_log: String,
    /// Whether to log discarded matches for analysis
    pub log_discarded_matches: bool,
    
    /// Path to the file where discarded matches are logged
    pub discarded_log_path: String,
    
    /// Phrase to trace through the matching pipeline for debugging
    pub traced_phrase: String,

    /// Keep temp files from grid matching
    pub keep_temp_files: Option<bool>,

    /// Keep temp files only when errors occur (useful for debugging failures)
    pub keep_temp_files_on_error: Option<bool>,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            ngram_log: "info".to_string(),
            log_discarded_matches: false,
            discarded_log_path: "logs/discarded_matches.log".to_string(),
            traced_phrase: String::new(),
            keep_temp_files: Some(false),
            keep_temp_files_on_error: Some(false),
        }
    }
}

impl Configurable for DebugConfig {
    fn section_name() -> &'static str {
        "debug"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            "ngram_log" => {
                self.ngram_log = crate::config::loader::parse::string(value)?;
                Ok(true)
            },
            "log_discarded_matches" => {
                self.log_discarded_matches = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "discarded_log_path" => {
                self.discarded_log_path = crate::config::loader::parse::string(value)?;
                Ok(true)
            },
            "traced_phrase" => {
                self.traced_phrase = crate::config::loader::parse::string(value)?;
                Ok(true)
            },
            "keep_temp_files" => {
                self.keep_temp_files = Some(crate::config::loader::parse::bool(value)?);
                Ok(true)
            },
            "keep_temp_files_on_error" => {
                self.keep_temp_files_on_error = Some(crate::config::loader::parse::bool(value)?);
                Ok(true)
            },
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
        // If logging is enabled, ensure the log directory exists
        if self.log_discarded_matches {
            let log_path = PathBuf::from(&self.discarded_log_path);
            if let Some(parent) = log_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| Error::Config(
                    format!("Failed to create log directory for discarded matches: {}", e)
                ))?;
            }
            
            // Warn if the traced phrase is very long
            if self.traced_phrase.len() > 100 {
                log::warn!("traced_phrase is very long ({} chars), this may impact performance", 
                    self.traced_phrase.len());
            }
        }
        
        Ok(())
    }
}

impl DebugConfig {
    /// Check if any debug features are enabled
    pub fn is_debugging_enabled(&self) -> bool {
        self.log_discarded_matches || !self.traced_phrase.is_empty()
    }
    
    /// Check if phrase tracing is enabled
    pub fn is_tracing_enabled(&self) -> bool {
        !self.traced_phrase.is_empty()
    }
    
    /// Check if a text contains the traced phrase
    pub fn contains_traced_phrase(&self, text: &str) -> bool {
        if self.traced_phrase.is_empty() {
            false
        } else {
            text.contains(&self.traced_phrase)
        }
    }
    
    /// Get the discarded log path as a PathBuf
    pub fn get_discarded_log_path(&self) -> PathBuf {
        PathBuf::from(&self.discarded_log_path)
    }
    
    /// Create a debug context string for logging
    pub fn debug_context(&self) -> String {
        let mut context = Vec::new();
        
        if self.log_discarded_matches {
            context.push(format!("logging discarded to {}", self.discarded_log_path));
        }
        
        if !self.traced_phrase.is_empty() {
            context.push(format!("tracing phrase: '{}'", 
                if self.traced_phrase.len() > 20 {
                    format!("{}...", &self.traced_phrase[..20])
                } else {
                    self.traced_phrase.clone()
                }
            ));
        }
        
        if self.keep_temp_files.unwrap_or(false) {
            context.push("keeping temp files".to_string());
        }

        if self.keep_temp_files_on_error.unwrap_or(false) {
            context.push("keeping temp files on error".to_string());
        }

        if context.is_empty() {
            "debug disabled".to_string()
        } else {
            format!("Debug: {}", context.join(", "))
        }
    }
}