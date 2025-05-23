// src/config/subsystems/generator.rs

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::FromIni;
use log::LevelFilter;

// Add the NGramType enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NGramType {
    Character,
    Word,
}

impl NGramType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NGramType::Character => "character",
            NGramType::Word => "word",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "character" | "char" => Some(Self::Character),
            "word" => Some(Self::Word),
            _ => None,
        }
    }
}

impl Default for NGramType {
    fn default() -> Self {
        Self::Word  // Make Word the default
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    // Size of n-grams to generate
    pub ngram_size: usize,

    // Type of n-grams to generate (character or word)
    pub ngram_type: NGramType,

    // Stride parameter for n-gram generation (how many units to advance)
    pub stride: usize,

    // Parallelization settings
    pub use_parallel: bool,
    pub thread_count: usize,

    // I/O settings
    pub buffer_size: usize,
    pub chunk_size: usize,

    // Log level
    pub ngram_log: String,
    #[serde(skip)]
    level_filter: Option<LevelFilter>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            ngram_size: 6,
            ngram_type: NGramType::default(),  // Use Word as default
            stride: 1,
            use_parallel: true,
            thread_count: 4,
            buffer_size: 4096,     // 4KB buffer
            chunk_size: 1024,      // 1KB chunks
            ngram_log: "info".to_string(),
            level_filter: Some(LevelFilter::Info),
        }
    }
}

impl FromIni for GeneratorConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        if section_name != "generator" {
            return None;
        }

        match key {
            "ngram_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.ngram_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid ngram_size (must be > 0): {}", value)
                    ))),
                }
            },
            "ngram_type" => {
                match NGramType::from_str(value) {
                    Some(ngram_type) => {
                        self.ngram_type = ngram_type;
                        Some(Ok(()))
                    },
                    None => Some(Err(Error::Config(
                        format!("Invalid ngram_type (must be 'character' or 'word'): {}", value)
                    ))),
                }
            },
            "stride" => {
                match value.parse() {
                    Ok(stride) if stride > 0 => {
                        self.stride = stride;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid stride (must be > 0): {}", value)
                    ))),
                }
            },
            "use_parallel" => {
                match value.parse() {
                    Ok(flag) => {
                        self.use_parallel = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid use_parallel value (must be true/false): {}", value)
                    ))),
                }
            },
            "thread_count" => {
                match value.parse() {
                    Ok(count) if count > 0 => {
                        self.thread_count = count;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid thread_count (must be > 0): {}", value)
                    ))),
                }
            },
            "buffer_size" => {
                match value.parse() {
                    Ok(size) if size >= 1024 => {  // Minimum 1KB buffer
                        self.buffer_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid buffer_size (must be >= 1024): {}", value)
                    ))),
                }
            },
            "chunk_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.chunk_size = size;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid chunk_size (must be > 0): {}", value)
                    ))),
                }
            },
            "ngram_log" => {
                let level_str = value.trim().to_lowercase();
                Some(
                    match level_str.as_str() {
                        "error" | "warn" | "info" | "debug" | "trace" | "none" => {
                            self.ngram_log = level_str.clone();
                            // Update the cached level filter
                            self.level_filter = Some(match level_str.as_str() {
                                "error" => LevelFilter::Error,
                                "warn" => LevelFilter::Warn,
                                "info" => LevelFilter::Info,
                                "debug" => LevelFilter::Debug,
                                "trace" => LevelFilter::Trace,
                                "none" => LevelFilter::Off,
                                _ => unreachable!(),
                            });
                            Ok(())
                        },
                        _ => Err(Error::Config(
                            format!("Invalid log level '{}'. Must be one of: none, error, warn, info, debug, trace", value)
                        ))
                    }
                )
            }
            _ => None,
        }
    }
}

impl GeneratorConfig {
    pub fn get_log_level(&self) -> LevelFilter {
        // If we have a cached level, return it
        if let Some(level) = self.level_filter {
            return level;
        }
        
        // Otherwise parse from the string
        match self.ngram_log.trim().to_lowercase().as_str() {
            "error" => LevelFilter::Error,
            "warn" => LevelFilter::Warn,
            "info" => LevelFilter::Info,
            "debug" => LevelFilter::Debug,
            "trace" => LevelFilter::Trace,
            "none" => LevelFilter::Off,
            _ => LevelFilter::Info, // Default to Info if invalid
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.ngram_size == 0 {
            return Err(Error::Config(
                "ngram_size must be greater than 0".to_string()
            ));
        }
        if self.stride == 0 {
            return Err(Error::Config(
                "stride must be greater than 0".to_string()
            ));
        }
        
        if self.stride > self.ngram_size {
            return Err(Error::Config(
                format!("stride ({}) cannot be greater than ngram_size ({})", 
                        self.stride, self.ngram_size)
            ));
        }

        // For word-based ngrams, we might want different validation rules
        if self.ngram_type == NGramType::Word && self.ngram_size > 500 {
            return Err(Error::Config(
                "For word-based ngrams, ngram_size must not exceed 500".to_string()
            ));
        }

        if self.thread_count == 0 {
            return Err(Error::Config(
                "thread_count must be greater than 0".to_string()
            ));
        }

        if self.buffer_size < 1024 {
            return Err(Error::Config(
                "buffer_size must be at least 1024 bytes".to_string()
            ));
        }

        if self.chunk_size == 0 {
            return Err(Error::Config(
                "chunk_size must be greater than 0".to_string()
            ));
        }

        // Warning if chunk_size > buffer_size
        if self.chunk_size > self.buffer_size {
            log::warn!(
                "chunk_size ({}) is larger than buffer_size ({}), which may impact performance",
                self.chunk_size,
                self.buffer_size
            );
        }

        Ok(())
    }

    /// Adjust thread count based on available system resources
    pub fn adjust_thread_count(&mut self) {
        if self.use_parallel {
            let system_threads = num_cpus::get();
            // Use at most 75% of available threads
            self.thread_count = std::cmp::min(
                self.thread_count,
                std::cmp::max(1, (system_threads as f32 * 0.75) as usize)
            );
        } else {
            self.thread_count = 1;
        }
    }

    /// Calculate optimal chunk size based on buffer size
    pub fn get_optimal_chunk_size(&self) -> usize {
        if self.chunk_size > self.buffer_size {
            self.buffer_size
        } else {
            // Round down to nearest multiple of 1KB
            (self.chunk_size / 1024) * 1024
        }
    }

    /// Returns a human-readable description of the configuration
    pub fn describe(&self) -> String {
        format!(
            "NGram generator configuration:\n\
             - NGram type: {}\n\
             - NGram size: {} {}\n\
             - Parallel processing: {}\n\
             - Thread count: {}\n\
             - Buffer size: {} bytes\n\
             - Chunk size: {} bytes",
            self.ngram_type.as_str(),
            self.ngram_size,
            match self.ngram_type {
                NGramType::Character => "characters",
                NGramType::Word => "words",
            },
            if self.use_parallel { "enabled" } else { "disabled" },
            self.thread_count,
            self.buffer_size,
            self.chunk_size
        )
    }
}