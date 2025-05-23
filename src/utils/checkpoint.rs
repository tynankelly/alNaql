use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::Write;
use std::time::{SystemTime, Duration};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCheckpoint {
    pub file_path: PathBuf,
    pub processed_bytes: u64,
    pub processed_ngrams: u64,
    pub last_sequence_number: u64,
    pub temporary_files: Vec<PathBuf>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub checkpoints: Vec<ProcessingCheckpoint>,
    pub total_files: usize,
    pub completed_files: usize,
    pub start_time: SystemTime,
    pub last_update: SystemTime,
}

pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    metadata: CheckpointMetadata,
    checkpoint_interval: Duration,
    last_checkpoint: SystemTime,
}

impl CheckpointManager {
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, checkpoint_interval_secs: u64) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir)?;
        
        let metadata = Self::load_or_create_metadata(&checkpoint_dir)?;
        
        Ok(Self {
            checkpoint_dir,
            metadata,
            checkpoint_interval: Duration::from_secs(checkpoint_interval_secs),
            last_checkpoint: SystemTime::now(),
        })
    }

    fn load_or_create_metadata(dir: &Path) -> Result<CheckpointMetadata> {
        let metadata_path = dir.join("metadata.json");
        
        if metadata_path.exists() {
            let file = fs::File::open(metadata_path)?;
            let metadata: CheckpointMetadata = serde_json::from_reader(file)
                .map_err(|e| Error::Config(format!("Failed to parse checkpoint metadata: {}", e)))?;
            Ok(metadata)
        } else {
            Ok(CheckpointMetadata {
                checkpoints: Vec::new(),
                total_files: 0,
                completed_files: 0,
                start_time: SystemTime::now(),
                last_update: SystemTime::now(),
            })
        }
    }

    pub fn should_checkpoint(&self) -> bool {
        self.last_checkpoint.elapsed().unwrap_or(Duration::from_secs(0)) >= self.checkpoint_interval
    }

    pub fn save_checkpoint(&mut self, checkpoint: ProcessingCheckpoint) -> Result<()> {
        // Update metadata
        if let Some(existing) = self.metadata.checkpoints.iter_mut()
            .find(|c| c.file_path == checkpoint.file_path) {
            *existing = checkpoint.clone();
        } else {
            self.metadata.checkpoints.push(checkpoint.clone());
        }
        
        self.metadata.last_update = SystemTime::now();
        
        // Save metadata
        let metadata_path = self.checkpoint_dir.join("metadata.json");
        let temp_path = metadata_path.with_extension("tmp");
        
        {
            let mut file = fs::File::create(&temp_path)?;
            serde_json::to_writer_pretty(&mut file, &self.metadata)
                .map_err(|e| Error::Config(format!("Failed to write checkpoint metadata: {}", e)))?;
            file.flush()?;
        }
        
        fs::rename(temp_path, metadata_path)?;
        self.last_checkpoint = SystemTime::now();
        
        Ok(())
    }

    pub fn get_checkpoint<P: AsRef<Path>>(&self, file_path: P) -> Option<&ProcessingCheckpoint> {
        self.metadata.checkpoints.iter()
            .find(|c| c.file_path == file_path.as_ref())
    }

    pub fn cleanup_temporary_files(&self) -> Result<()> {
        for checkpoint in &self.metadata.checkpoints {
            for temp_file in &checkpoint.temporary_files {
                if temp_file.exists() {
                    fs::remove_file(temp_file)?;
                }
            }
        }
        Ok(())
    }

    pub fn mark_file_completed<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        if let Some(idx) = self.metadata.checkpoints.iter()
            .position(|c| c.file_path == file_path.as_ref()) {
            self.metadata.checkpoints.remove(idx);
            self.metadata.completed_files += 1;
            self.save_metadata()?;
        }
        Ok(())
    }

    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.checkpoint_dir.join("metadata.json");
        let file = fs::File::create(metadata_path)?;
        serde_json::to_writer_pretty(file, &self.metadata)
            .map_err(|e| Error::Config(format!("Failed to write metadata: {}", e)))?;
        Ok(())
    }

    pub fn get_progress(&self) -> (usize, usize) {
        (self.metadata.completed_files, self.metadata.total_files)
    }
}