// src/config/subsystems/parser.rs

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::{Error, Result};
use crate::config::FromIni;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    // Text cleaning settings
    pub remove_numbers: bool,
    pub remove_diacritics: bool,
    pub remove_tatweel: bool,
    pub normalize_arabic: bool,
    pub preserve_punctuation: bool,
    
    // New field for stop words
    pub stop_words_file: Option<PathBuf>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            // From default.ini text_processing section
            remove_numbers: true,
            remove_diacritics: true,
            remove_tatweel: true,
            normalize_arabic: true,
            preserve_punctuation: false,
            
            // Default to no stop words file
            stop_words_file: None,
        }
    }
}

impl FromIni for ParserConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        // Match both root and text_processing section for flexibility
        if section_name != "text_processing" {
            return None;
        }

        match key {
            "remove_numbers" => {
                match value.parse() {
                    Ok(flag) => {
                        self.remove_numbers = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid remove_numbers value (must be true/false): {}", value)
                    ))),
                }
            },
            "remove_diacritics" => {
                match value.parse() {
                    Ok(flag) => {
                        self.remove_diacritics = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid remove_diacritics value (must be true/false): {}", value)
                    ))),
                }
            },
            "remove_tatweel" => {
                match value.parse() {
                    Ok(flag) => {
                        self.remove_tatweel = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid remove_tatweel value (must be true/false): {}", value)
                    ))),
                }
            },
            "normalize_arabic" => {
                match value.parse() {
                    Ok(flag) => {
                        self.normalize_arabic = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid normalize_arabic value (must be true/false): {}", value)
                    ))),
                }
            },
            "preserve_punctuation" => {
                match value.parse() {
                    Ok(flag) => {
                        self.preserve_punctuation = flag;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid preserve_punctuation value (must be true/false): {}", value)
                    ))),
                }
            },
            "stop_words_file" => {
                // Handle the new stop_words_file setting
                let file_path = PathBuf::from(value.trim_matches('"'));
                self.stop_words_file = Some(file_path);
                Some(Ok(()))
            },
            _ => None,
        }
    }
}

impl ParserConfig {
    pub fn validate(&self) -> Result<()> {
        // Validate stop_words_file if it's specified
        if let Some(path) = &self.stop_words_file {
            // Only warn if file doesn't exist, don't fail
            if !path.exists() {
                log::warn!("Stop words file not found: {:?}", path);
            }
        }
        
        // No validation needed for other boolean flags
        Ok(())
    }

    /// Returns a description of the current text processing configuration
    pub fn describe(&self) -> String {
        let mut description = Vec::new();
        
        if self.remove_numbers {
            description.push("removing numeric characters".to_string());
        }
        if self.remove_diacritics {
            description.push("removing diacritical marks".to_string());
        }
        if self.remove_tatweel {
            description.push("removing tatweel".to_string());
        }
        if self.normalize_arabic {
            description.push("normalizing Arabic characters".to_string());
        }
        if !self.preserve_punctuation {
            description.push("removing punctuation".to_string());
        }
        if let Some(path) = &self.stop_words_file {
            description.push(format!("using stop words from {:?}", path));
        }

        if description.is_empty() {
            "no text processing applied".to_string()
        } else {
            description.join(", ")
        }
    }

    /// Helper method to check if any text processing is enabled
    pub fn has_processing_enabled(&self) -> bool {
        self.remove_numbers || 
        self.remove_diacritics || 
        self.remove_tatweel || 
        self.normalize_arabic || 
        !self.preserve_punctuation ||
        self.stop_words_file.is_some()
    }
}