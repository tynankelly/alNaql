// src/config/subsystems/text_processing.rs

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::FromIni;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    #[serde(default = "default_remove_numbers")]
    pub remove_numbers: bool,
    
    #[serde(default = "default_remove_diacritics")]
    pub remove_diacritics: bool,
    
    #[serde(default = "default_remove_tatweel")]
    pub remove_tatweel: bool,
    
    #[serde(default = "default_normalize_arabic")]
    pub normalize_arabic: bool,
    
    #[serde(default = "default_preserve_punctuation")]
    pub preserve_punctuation: bool,
}

// Default functions
fn default_remove_numbers() -> bool { true }
fn default_remove_diacritics() -> bool { true }
fn default_remove_tatweel() -> bool { true }
fn default_normalize_arabic() -> bool { true }
fn default_preserve_punctuation() -> bool { false }

impl Default for TextProcessingConfig {
    fn default() -> Self {
        Self {
            remove_numbers: default_remove_numbers(),
            remove_diacritics: default_remove_diacritics(),
            remove_tatweel: default_remove_tatweel(),
            normalize_arabic: default_normalize_arabic(),
            preserve_punctuation: default_preserve_punctuation(),
        }
    }
}

impl FromIni for TextProcessingConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
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
            _ => None,
        }
    }
}

impl TextProcessingConfig {
    pub fn validate(&self) -> Result<()> {
        // Currently no validation needed for boolean flags
        Ok(())
    }

    /// Returns true if any text processing options are enabled
    pub fn has_processing_enabled(&self) -> bool {
        self.remove_numbers || 
        self.remove_diacritics || 
        self.remove_tatweel || 
        self.normalize_arabic || 
        !self.preserve_punctuation
    }

    /// Returns a description of which processing options are enabled
    pub fn describe(&self) -> String {
        let mut enabled = Vec::new();
        
        if self.remove_numbers { enabled.push("removing numbers"); }
        if self.remove_diacritics { enabled.push("removing diacritics"); }
        if self.remove_tatweel { enabled.push("removing tatweel"); }
        if self.normalize_arabic { enabled.push("normalizing Arabic"); }
        if !self.preserve_punctuation { enabled.push("removing punctuation"); }

        if enabled.is_empty() {
            "no text processing enabled".to_string()
        } else {
            enabled.join(", ")
        }
    }
}