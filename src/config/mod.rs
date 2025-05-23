pub mod file;
pub mod subsystems;

use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs;
use crate::error::Result;
use log::{info, warn, trace};

pub trait FromIni {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuttabConfig {
    // File paths
    pub files: file::FileConfig,
    
    // Subsystem configs
    pub parser: subsystems::ParserConfig,
    pub generator: subsystems::GeneratorConfig,
    pub matcher: subsystems::MatcherConfig,
    pub storage: subsystems::StorageConfig,
    pub processor: subsystems::ProcessorConfig,
}

impl KuttabConfig {
    pub fn validate(&self) -> Result<()> {
        self.files.validate()?;
        self.parser.validate()?;
        self.generator.validate()?;
        self.matcher.validate()?;
        self.storage.validate()?;
        self.processor.validate()?;
        Ok(())
    }

    pub fn from_ini<P: AsRef<Path>>(path: P) -> Result<Self> {
        let absolute_path = std::fs::canonicalize(&path)
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        
        trace!("Loading configuration from: {:?}", absolute_path);
        
        let content = fs::read_to_string(&path)?;
        
        let mut config = Self::default();
        let mut current_section = String::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len()-1].to_string();
                trace!("  Line {}: Found section: [{}]", line_num + 1, current_section);
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                
                
                // Delegate to appropriate subsystem config
                if let Some(result) = match current_section.as_str() {
                    "file" => config.files.from_ini_section(&current_section, key, value),
                    "text_processing" => {
                        info!("    Processing text_processing config: {}={}", key, value);
                        config.parser.from_ini_section(&current_section, key, value)
                    },
                    "generator" => config.generator.from_ini_section(&current_section, key, value),
                    "matcher" => config.matcher.from_ini_section(&current_section, key, value),
                    "storage" => config.storage.from_ini_section(&current_section, key, value),
                    "processor" => config.processor.from_ini_section(&current_section, key, value),
                    // Handle nested sections
                    s if s.starts_with("matcher.") => config.matcher.from_ini_section(&current_section, key, value),
                    s if s.starts_with("storage.") => config.storage.from_ini_section(&current_section, key, value),
                    _ => None,
                } {
                    if let Err(e) = result {
                        warn!("Error processing config key {}={}: {}", key, value, e);
                    } else {
                    }
                } else {
                    warn!("Unrecognized config key: {}={} in section [{}]", key, value, current_section);
                }
            }
        }

        config.validate()?;
        Ok(config)
    }
}

impl Default for KuttabConfig {
    fn default() -> Self {
        Self {
            files: file::FileConfig::default(),
            parser: subsystems::ParserConfig::default(),
            generator: subsystems::GeneratorConfig::default(),
            matcher: subsystems::MatcherConfig::default(),
            storage: subsystems::StorageConfig::default(),
            processor: subsystems::ProcessorConfig::default(),
        }
    }
}