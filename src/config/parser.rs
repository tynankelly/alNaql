// config/processing.rs - Text processing and parsing configuration

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::Result;
use crate::config::loader::Configurable;
use crate::config_fields;

/// Configuration for text processing and normalization
/// Merges the previous ParserConfig and TextProcessingConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    /// Whether to remove numeric characters from text
    pub remove_numbers: bool,
    
    /// Whether to remove Arabic diacritical marks
    pub remove_diacritics: bool,
    
    /// Whether to remove tatweel (Arabic text elongation character)
    pub remove_tatweel: bool,
    
    /// Whether to normalize Arabic character variants
    pub normalize_arabic: bool,
    
    /// Whether to preserve punctuation in processed text
    pub preserve_punctuation: bool,
    
    /// Optional path to a file containing stop words to filter
    pub stop_words_file: Option<PathBuf>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            // These defaults match the original ParserConfig/TextProcessingConfig
            remove_numbers: true,
            remove_diacritics: true,
            remove_tatweel: true,
            normalize_arabic: true,
            preserve_punctuation: false,
            stop_words_file: None,
        }
    }
}

impl Configurable for ParserConfig {
    fn section_name() -> &'static str {
        // This will map to [processing] in the INI file
        // but we'll also accept [text_processing] for compatibility
        "processing"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        config_fields!(self, key, value, {
            remove_numbers => bool,
            remove_diacritics => bool,
            remove_tatweel => bool,
            normalize_arabic => bool,
            preserve_punctuation => bool,
            stop_words_file => optional_path,
        })
    }
    
    fn validate(&self) -> Result<()> {
        // Validate stop_words_file if it's specified
        if let Some(path) = &self.stop_words_file {
            // Only warn if file doesn't exist, don't fail
            // This matches the original ParserConfig behavior
            if !path.exists() {
                log::warn!("Stop words file not found: {:?}", path);
            }
        }
        
        // No other validation needed for boolean flags
        Ok(())
    }
}

impl ParserConfig {
    /// Returns true if any text processing options are enabled
    /// (This method exists in the original TextProcessingConfig)
    pub fn has_processing_enabled(&self) -> bool {
        self.remove_numbers || 
        self.remove_diacritics || 
        self.remove_tatweel || 
        self.normalize_arabic || 
        !self.preserve_punctuation
    }
    
    /// Returns a description of which processing options are enabled
    /// (This method exists in the original TextProcessingConfig)
    pub fn describe(&self) -> String {
        let mut enabled = Vec::new();
        
        if self.remove_numbers {
            enabled.push("removing numbers");
        }
        if self.remove_diacritics {
            enabled.push("removing diacritics");
        }
        if self.remove_tatweel {
            enabled.push("removing tatweel");
        }
        if self.normalize_arabic {
            enabled.push("normalizing Arabic");
        }
        if !self.preserve_punctuation {
            enabled.push("removing punctuation");
        }
        
        if enabled.is_empty() {
            "no text processing enabled".to_string()
        } else {
            enabled.join(", ")
        }
    }
}
