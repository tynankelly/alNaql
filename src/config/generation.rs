// config/generation.rs - N-gram generation configuration

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::loader::Configurable;

/// Type of n-grams to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NGramType {
    Character,
    Word,
}

impl NGramType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim_matches('"').to_lowercase().as_str() {
            "character" | "char" => Some(Self::Character),
            "word" => Some(Self::Word),
            _ => None,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Character => "character",
            Self::Word => "word",
        }
    }
}

impl Default for NGramType {
    fn default() -> Self {
        Self::Character
    }
}

/// Configuration for n-gram generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Size of n-grams to generate (e.g., 3 for trigrams)
    pub ngram_size: usize,
    
    /// Type of n-grams (character or word)
    pub ngram_type: NGramType,
    
    /// Whether to compute similarity features during generation
    pub compute_similarity_features: bool,
    
    /// Stride for n-gram generation (1 = overlapping, n = skip n-1)
    pub stride: usize,
    
    /// Depth for prefix indexing (minimum 2)
    pub prefix_index_depth: usize,
    
    /// Whether to use parallel processing for generation
    pub use_parallel: bool,
    
    /// Number of threads for parallel processing (0 = auto)
    pub thread_count: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            ngram_size: 3,
            ngram_type: NGramType::Character,
            compute_similarity_features: true,
            stride: 1,
            prefix_index_depth: 3,
            use_parallel: true,
            thread_count: 0, // 0 means auto-detect
        }
    }
}

impl Configurable for GenerationConfig {
    fn section_name() -> &'static str {
        "generation"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            "ngram_size" => {
                match value.parse() {
                    Ok(size) if size > 0 => {
                        self.ngram_size = size;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid ngram_size (must be > 0): {}", value)
                    )),
                }
            },
            "ngram_type" => {
                match NGramType::from_str(value) {
                    Some(ngram_type) => {
                        self.ngram_type = ngram_type;
                        Ok(true)
                    },
                    None => Err(Error::Config(
                        format!("Invalid ngram_type (must be 'character' or 'word'): {}", value)
                    )),
                }
            },
            "compute_similarity_features" => {
                match value.parse() {
                    Ok(flag) => {
                        self.compute_similarity_features = flag;
                        Ok(true)
                    },
                    Err(_) => Err(Error::Config(
                        format!("Invalid compute_similarity_features value (must be true/false): {}", value)
                    )),
                }
            },
            "stride" => {
                match value.parse() {
                    Ok(stride) if stride > 0 => {
                        self.stride = stride;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid stride (must be > 0): {}", value)
                    )),
                }
            },
            "prefix_index_depth" => {
                match value.parse::<usize>() {
                    Ok(depth) if depth >= 2 => {
                        self.prefix_index_depth = depth;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid prefix_index_depth (must be >= 2): {}", value)
                    )),
                }
            },
            "use_parallel" => {
                match value.parse() {
                    Ok(flag) => {
                        self.use_parallel = flag;
                        Ok(true)
                    },
                    Err(_) => Err(Error::Config(
                        format!("Invalid use_parallel value (must be true/false): {}", value)
                    )),
                }
            },
            "thread_count" => {
                match value.parse() {
                    Ok(count) => {
                        self.thread_count = count;
                        Ok(true)
                    },
                    _ => Err(Error::Config(
                        format!("Invalid thread_count (must be >= 0): {}", value)
                    )),
                }
            },
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Validate ngram_size
        if self.ngram_size == 0 {
            return Err(Error::Config(
                "ngram_size must be greater than 0".to_string()
            ));
        }
        
        // Validate stride
        if self.stride == 0 {
            return Err(Error::Config(
                "stride must be greater than 0".to_string()
            ));
        }
        
        // Validate prefix_index_depth
        if self.prefix_index_depth < 2 {
            return Err(Error::Config(
                "prefix_index_depth must be at least 2".to_string()
            ));
        }
                
        // If thread_count is 0, it means auto-detect, which is valid
        // If thread_count is specified, validate it's reasonable
        if self.thread_count > 256 {
            log::warn!("thread_count {} seems unusually high", self.thread_count);
        }
        
        Ok(())
    }
}

impl GenerationConfig {
    /// Get the effective thread count (resolves 0 to actual CPU count)
    pub fn effective_thread_count(&self) -> usize {
        if self.thread_count == 0 {
            num_cpus::get()
        } else {
            self.thread_count
        }
    }
    
    /// Check if configuration is for character n-grams
    pub fn is_character_ngrams(&self) -> bool {
        matches!(self.ngram_type, NGramType::Character)
    }
    
    /// Check if configuration is for word n-grams
    pub fn is_word_ngrams(&self) -> bool {
        matches!(self.ngram_type, NGramType::Word)
    }
}