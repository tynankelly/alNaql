// config/mod.rs - Main configuration module

pub mod binary_io;
pub mod debug;
pub mod execution;
pub mod from_args;
pub mod generation;
pub mod job_track;
pub mod loader;
pub mod matching;
pub mod paths;
pub mod parser;
pub mod storage;

use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::fs;
use crate::error::Result;
use log::{info, warn, trace};

use self::loader::Configurable;

// Re-export commonly used types
pub use binary_io::{BinaryIOConfig, OutputFormat};
pub use job_track::{JobConfig, JobConfigBuilder};
pub use from_args::parse_job_config;
pub use self::generation::NGramType;
pub use self::matching::{SimilarityMetric, ProcessStrategy, NGramMatchingStrategy, ClusteringMode, MatchingConfig};

/// Main configuration structure for AlNaql
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlNaqlConfig {
    // Keep exact field names for compatibility
    /// File paths configuration (was FileConfig)
    pub files: paths::PathsConfig,
    
    /// Text processing configuration (was ParserConfig)
    pub parser: parser::ParserConfig,
    
    /// N-gram generation configuration (was GeneratorConfig)
    pub generator: generation::GenerationConfig,
    
    /// Matching algorithm configuration (was MatcherConfig)
    pub matcher: matching::MatchingConfig,
    
    /// Storage backend configuration (was StorageConfig)
    pub storage: storage::StorageConfig,
    
    /// Execution strategy configuration (was ProcessorConfig)
    pub execution: execution::ExecutionConfig,

    /// Binary I/O and compression configuration
    pub binary_io: binary_io::BinaryIOConfig,
        
    /// Debug and logging configuration (new, from ProcessorConfig + MatcherConfig)
    pub debug: debug::DebugConfig,

}

impl Default for AlNaqlConfig {
    fn default() -> Self {
        Self {
            files: paths::PathsConfig::default(),
            parser: parser::ParserConfig::default(),
            generator: generation::GenerationConfig::default(),
            matcher: matching::MatchingConfig::default(),
            storage: storage::StorageConfig::default(),
            execution: execution::ExecutionConfig::default(),
            binary_io: binary_io::BinaryIOConfig::default(),
            debug: debug::DebugConfig::default(),
        }
    }
}

impl AlNaqlConfig {
    /// Load configuration from an INI file
    pub fn from_ini<P: AsRef<Path>>(path: P) -> Result<Self> {
        let absolute_path = std::fs::canonicalize(&path)
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        
        info!("Loading configuration from: {:?}", absolute_path);
        
        let content = fs::read_to_string(&path)?;
        
        let mut config = Self::default();
        let mut current_section = String::new();
        let mut sections_found = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            // Handle section headers
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len()-1].to_string();
                sections_found.push(current_section.clone());
                trace!("Line {}: Found section [{}]", line_num + 1, current_section);
                continue;
            }

            // Parse key-value pairs
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                
                trace!("  Line {}: {}={} in section [{}]", line_num + 1, key, value, current_section);
                
                // Route to appropriate config based on section
                let handled = match current_section.as_str() {
                    // Map old section names to new configs for compatibility
                    "paths" => {
                        config.files.set_field(key, value)?
                    },
                    "parser" => {
                        config.parser.set_field(key, value)?
                    },
                    "generator" | "generation" => {
                        config.generator.set_field(key, value)?
                    },
                    "matcher" | "matching" => {
                        config.matcher.set_field(key, value)?
                    },
                    "storage" => {
                        config.storage.set_field(key, value)?
                    },
                    "execution" => {
                        config.execution.set_field(key, value)?
                    },
                    "binary_io" | "binary" => {
                        config.binary_io.set_field(key, value)?
                    },
                    "debug" => {
                        config.debug.set_field(key, value)?
                    },

                    _ => {
                        // Check for nested sections (e.g., "matcher.algorithm")
                        if current_section.starts_with("matcher.") {
                            config.matcher.set_field(key, value)?
                        } else if current_section.starts_with("storage.") {
                            config.storage.set_field(key, value)?
                        } else {
                            warn!("Unknown configuration section: [{}]", current_section);
                            false
                        }
                    }
                };
                
                if !handled {
                    warn!("Unknown configuration key '{}' in section [{}]", key, current_section);
                }
            } else {
                warn!("Line {}: Invalid configuration line (expected key=value): {}", 
                    line_num + 1, line);
            }
        }
        
        // Log which sections were found
        if !sections_found.is_empty() {
            info!("Configuration sections loaded: {}", sections_found.join(", "));
        }
        
        // Validate all subsystems
        config.validate()?;
        
        info!("Configuration loaded successfully");
        Ok(config)
    }
    
    /// Validate the entire configuration
    pub fn validate(&self) -> Result<()> {
        // Validate each subsystem
        self.files.validate()?;
        self.parser.validate()?;
        self.generator.validate()?;
        self.matcher.validate()?;
        self.storage.validate()?;
        self.execution.validate()?;
        self.binary_io.validate()?;
        self.debug.validate()?;        
        // Cross-configuration validation
        self.validate_cross_config()?;
        
        Ok(())
    }
    
    /// Validate cross-configuration dependencies and consistency
    fn validate_cross_config(&self) -> Result<()> {
                
        // Warn if debugging is enabled in production-like settings
        if self.debug.is_debugging_enabled() && self.execution.processing_mode == "parallel" {
            warn!("Debug features are enabled with parallel processing - this may impact performance");
        }
                
        Ok(())
    }
    
    /// Save configuration to an INI file
    pub fn to_ini<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut content = String::new();
        
        // Header
        content.push_str("# AlNaql Configuration File\n");
        content.push_str("# Generated automatically - do not edit while system is running\n\n");
        
        // Paths section
        content.push_str("[paths]\n");
        content.push_str(&format!("source_dir = \"{}\"\n", self.files.source_dir.display()));
        content.push_str(&format!("target_dir = \"{}\"\n", self.files.target_dir.display()));
        content.push_str(&format!("ngrams_source_dir = \"{}\"\n", self.files.ngrams_source_dir.display()));
        content.push_str(&format!("ngrams_target_dir = \"{}\"\n", self.files.ngrams_target_dir.display()));
        content.push_str(&format!("output_dir = \"{}\"\n", self.files.output_dir.display()));
        content.push_str(&format!("temp_dir = \"{}\"\n", self.files.temp_dir.display()));
        content.push_str("\n");
        
        // Processing section
        content.push_str("[processing]\n");
        content.push_str(&format!("remove_numbers = {}\n", self.parser.remove_numbers));
        content.push_str(&format!("remove_diacritics = {}\n", self.parser.remove_diacritics));
        content.push_str(&format!("remove_tatweel = {}\n", self.parser.remove_tatweel));
        content.push_str(&format!("normalize_arabic = {}\n", self.parser.normalize_arabic));
        content.push_str(&format!("preserve_punctuation = {}\n", self.parser.preserve_punctuation));
        if let Some(ref path) = self.parser.stop_words_file {
            content.push_str(&format!("stop_words_file = \"{}\"\n", path.display()));
        }
        content.push_str("\n");
        
        // Generation section
        content.push_str("[generation]\n");
        content.push_str(&format!("ngram_size = {}\n", self.generator.ngram_size));
        content.push_str(&format!("ngram_type = \"{}\"\n", self.generator.ngram_type.as_str()));
        content.push_str(&format!("compute_similarity_features = {}\n", self.generator.compute_similarity_features));
        content.push_str(&format!("stride = {}\n", self.generator.stride));
        content.push_str(&format!("prefix_index_depth = {}\n", self.generator.prefix_index_depth));
        content.push_str(&format!("use_parallel = {}\n", self.generator.use_parallel));
        content.push_str(&format!("thread_count = {}\n", self.generator.thread_count));
        content.push_str("\n");
        
        // Add more sections as needed...
        // (Abbreviated for brevity - would include all sections)
        
        fs::write(path, content)?;
        Ok(())
    }
    
    /// Get a summary of the configuration
    pub fn summary(&self) -> String {
        format!(
            "AlNaql Configuration:\n\
            - Processing mode: {}\n\
            - N-gram type: {:?} (size: {})\n\
            - Similarity metric: {:?}\n\
            - Storage: {:?}\n\
            - Debug: {}",
            self.execution.processing_mode,
            self.generator.ngram_type,
            self.generator.ngram_size,
            self.matcher.ngram_similarity_metric,
            self.storage.db_path.display(),
            self.debug.debug_context()
        )
    }
    /// Get the output format for final results
    pub fn output_format(&self) -> OutputFormat {
        self.binary_io.output_format
    }
    
    /// Check if temporary files should be compressed (always true)
    pub fn compress_temp_files(&self) -> bool {
        self.binary_io.compress_temp_files()
    }
    
    /// Check if final output should be compressed
    pub fn compress_final_output(&self) -> bool {
        self.binary_io.should_compress_final()
    }
    /// Check if binary output should be generated
    pub fn should_output_binary(&self) -> bool {
        self.binary_io.output_format.includes_binary()
    }
    
    /// Check if JSON output should be generated  
    pub fn should_output_json(&self) -> bool {
        self.binary_io.output_format.includes_json()
    }
    
    /// Get all output paths for the given base path
    pub fn get_output_paths(&self, base_path: &Path) -> Vec<PathBuf> {
        self.binary_io.generate_output_paths(base_path)
    }
}

