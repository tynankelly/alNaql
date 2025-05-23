// src/config/file.rs

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use crate::error::Result;
use super::FromIni;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    pub source_dir: PathBuf,
    pub target_dir: PathBuf,
    pub ngrams_source_dir: PathBuf,
    pub ngrams_target_dir: PathBuf,
    pub output_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub state_dir: PathBuf,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            source_dir: PathBuf::from("data/source"),
            target_dir: PathBuf::from("data/target"),
            ngrams_source_dir: PathBuf::from("ngrams/source"),
            ngrams_target_dir: PathBuf::from("ngrams/target"),
            output_dir: PathBuf::from("data/matches"),
            temp_dir: PathBuf::from("data/temp"),
            state_dir: PathBuf::from("data/state"),
        }
    }
}

impl FromIni for FileConfig {
    fn from_ini_section(&mut self, _section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        match key {
            "source_dir" => {
                self.source_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "target_dir" => {
                self.target_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "ngrams_source_dir" => {
                self.ngrams_source_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "ngrams_target_dir" => {
                self.ngrams_target_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "output_dir" => {
                self.output_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "temp_dir" => {
                self.temp_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            "state_dir" => {
                self.state_dir = PathBuf::from(value.trim_matches('"'));
                Some(Ok(()))
            },
            _ => None,
        }
    }
}

impl FileConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.source_dir.exists() {
            return Err(crate::error::Error::Config(
                format!("Source directory does not exist: {:?}", self.source_dir)
            ));
        }
        if !self.target_dir.exists() {
            return Err(crate::error::Error::Config(
                format!("Target directory does not exist: {:?}", self.target_dir)
            ));
        }
        
        // Create ngram directories if they don't exist (instead of checking)
        std::fs::create_dir_all(&self.ngrams_source_dir)?;
        std::fs::create_dir_all(&self.ngrams_target_dir)?;
        
        // Create output and temp directories if they don't exist
        std::fs::create_dir_all(&self.output_dir)?;
        std::fs::create_dir_all(&self.temp_dir)?;
        std::fs::create_dir_all(&self.state_dir)?;
        
        Ok(())
    }
}