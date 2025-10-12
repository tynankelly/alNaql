// config/paths.rs - File and directory path configuration

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::{Error, Result};
use crate::config::loader::Configurable;
use crate::config_fields;

/// Configuration for all file and directory paths used by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    /// Directory containing source text files
    pub source_dir: PathBuf,
    
    /// Directory containing target text files for comparison
    pub target_dir: PathBuf,
    
    /// Directory for storing source n-grams
    pub ngrams_source_dir: PathBuf,
    
    /// Directory for storing target n-grams
    pub ngrams_target_dir: PathBuf,
    
    /// Directory for match output files
    pub output_dir: PathBuf,
    
    /// Directory for temporary files during processing
    pub temp_dir: PathBuf,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            source_dir: PathBuf::from("data/source"),
            target_dir: PathBuf::from("data/target"),
            ngrams_source_dir: PathBuf::from("ngrams/source"),
            ngrams_target_dir: PathBuf::from("ngrams/target"),
            output_dir: PathBuf::from("output/matches"),
            temp_dir: PathBuf::from("output/temp"),
        }
    }
}

impl Configurable for PathsConfig {
    fn section_name() -> &'static str {
        "paths"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        // Note: We're keeping the exact same field names as in FileConfig
        config_fields!(self, key, value, {
            source_dir => path,
            target_dir => path,
            ngrams_source_dir => path,
            ngrams_target_dir => path,
            output_dir => path,
            temp_dir => path,
        })
    }
    
    fn validate(&self) -> Result<()> {
        // Check that required input directories exist
        if !self.source_dir.exists() {
            return Err(Error::Config(
                format!("Source directory does not exist: {:?}", self.source_dir)
            ));
        }
        
        if !self.target_dir.exists() {
            return Err(Error::Config(
                format!("Target directory does not exist: {:?}", self.target_dir)
            ));
        }
        
        // Create output directories if they don't exist
        // This matches the behavior in the original FileConfig::validate()
        std::fs::create_dir_all(&self.ngrams_source_dir)
            .map_err(|e| Error::Config(
                format!("Failed to create ngrams source directory: {}", e)
            ))?;
            
        std::fs::create_dir_all(&self.ngrams_target_dir)
            .map_err(|e| Error::Config(
                format!("Failed to create ngrams target directory: {}", e)
            ))?;
            
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| Error::Config(
                format!("Failed to create output directory: {}", e)
            ))?;
            
        std::fs::create_dir_all(&self.temp_dir)
            .map_err(|e| Error::Config(
                format!("Failed to create temp directory: {}", e)
            ))?;
        
        Ok(())
    }
}

impl PathsConfig {
    /// Get the source directory path
    pub fn source_dir(&self) -> &PathBuf {
        &self.source_dir
    }
    
    /// Get the target directory path
    pub fn target_dir(&self) -> &PathBuf {
        &self.target_dir
    }
    
    /// Get the path for a specific source n-grams file
    pub fn source_ngrams_path(&self, filename: &str) -> PathBuf {
        self.ngrams_source_dir.join(filename)
    }
    
    /// Get the path for a specific target n-grams file
    pub fn target_ngrams_path(&self, filename: &str) -> PathBuf {
        self.ngrams_target_dir.join(filename)
    }
    
    /// Get the path for an output file
    pub fn output_path(&self, filename: &str) -> PathBuf {
        self.output_dir.join(filename)
    }
    
    /// Get the path for a temporary file
    pub fn temp_path(&self, filename: &str) -> PathBuf {
        self.temp_dir.join(filename)
    }
    
    /// Check if all required directories are accessible
    pub fn check_accessibility(&self) -> Result<()> {
        // Check read access for input directories
        std::fs::read_dir(&self.source_dir)
            .map_err(|e| Error::Config(
                format!("Cannot read source directory: {}", e)
            ))?;
            
        std::fs::read_dir(&self.target_dir)
            .map_err(|e| Error::Config(
                format!("Cannot read target directory: {}", e)
            ))?;
        
        // Check write access for output directories
        let test_file = self.temp_dir.join(".write_test");
        std::fs::write(&test_file, b"test")
            .map_err(|e| Error::Config(
                format!("Cannot write to temp directory: {}", e)
            ))?;
        std::fs::remove_file(test_file).ok();
        
        Ok(())
    }
}