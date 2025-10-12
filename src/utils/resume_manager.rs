// utils/resume_manager.rs
//! Resume management for interrupted grid streaming processes
//! 
//! This module provides functionality to save and restore the state of
//! grid streaming pipeline execution, allowing jobs to be resumed from
//! the last successfully completed stage after interruption.

use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{info, warn, debug};
use crate::error::{Result, Error};

// ============================================================================
// CORE TYPES
// ============================================================================

/// Represents the stages of the grid streaming pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Initial grid matching stage - produces grid_matches.tmp
    GridMatching,
    /// Sorting and deduplication - produces grid_matches_sorted.tmp  
    Sorting,
    /// Clustering matches - produces clusters.tmp
    Clustering,
    /// Validation of clusters - produces validated.tmp
    Validation,
    /// Final merge and output - produces final output file
    FinalMerge,
    /// Pipeline fully completed
    Complete,
}

impl PipelineStage {
    /// Get the expected output filename for this stage
    pub fn output_filename(&self) -> &'static str {
        match self {
            PipelineStage::GridMatching => "grid_matches.tmp",
            PipelineStage::Sorting => "grid_matches_sorted.tmp",
            PipelineStage::Clustering => "clusters.tmp",
            PipelineStage::Validation => "validated.tmp",
            PipelineStage::FinalMerge => "final_output.tmp",
            PipelineStage::Complete => "",
        }
    }
    
    /// Get the next stage in the pipeline
    pub fn next_stage(&self) -> Option<PipelineStage> {
        match self {
            PipelineStage::GridMatching => Some(PipelineStage::Sorting),
            PipelineStage::Sorting => Some(PipelineStage::Clustering),
            PipelineStage::Clustering => Some(PipelineStage::Validation),
            PipelineStage::Validation => Some(PipelineStage::FinalMerge),
            PipelineStage::FinalMerge => Some(PipelineStage::Complete),
            PipelineStage::Complete => None,
        }
    }
    
    /// Get progress bar position for this stage (0-100)
    pub fn progress_position(&self) -> u64 {
        match self {
            PipelineStage::GridMatching => 40,
            PipelineStage::Sorting => 45,
            PipelineStage::Clustering => 50,
            PipelineStage::Validation => 70,
            PipelineStage::FinalMerge => 85,
            PipelineStage::Complete => 100,
        }
    }
}

/// Metadata about a pipeline run, persisted to disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Source database path (for verification)
    pub source_db: String,
    /// Target database path (for verification)
    pub target_db: String,
    /// Timestamp when run was started (ISO 8601)
    pub started_at: String,
    /// Last update timestamp (ISO 8601)
    pub updated_at: String,
    /// Completion status of each stage
    pub stages_completed: HashMap<String, bool>,
    /// Output files produced by each stage
    pub stage_outputs: HashMap<String, String>,
    /// Number of items processed at each stage (for reporting)
    pub stage_counts: HashMap<String, usize>,
}

impl RunMetadata {
    /// Create new metadata for a fresh run
    pub fn new(source_db: &Path, target_db: &Path) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        
        let mut stages_completed = HashMap::new();
        stages_completed.insert("grid_matching".to_string(), false);
        stages_completed.insert("sorting".to_string(), false);
        stages_completed.insert("clustering".to_string(), false);
        stages_completed.insert("validation".to_string(), false);
        stages_completed.insert("final_merge".to_string(), false);
        
        RunMetadata {
            source_db: source_db.to_string_lossy().to_string(),
            target_db: target_db.to_string_lossy().to_string(),
            started_at: now.clone(),
            updated_at: now,
            stages_completed,
            stage_outputs: HashMap::new(),
            stage_counts: HashMap::new(),
        }
    }
    
    /// Mark a stage as completed and record its output
    pub fn mark_stage_complete(
        &mut self, 
        stage: PipelineStage, 
        output_file: &str,
        item_count: usize
    ) {
        let stage_key = self.stage_to_key(stage);
        self.stages_completed.insert(stage_key.clone(), true);
        self.stage_outputs.insert(stage_key.clone(), output_file.to_string());
        self.stage_counts.insert(stage_key, item_count);
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
    
    /// Check if a stage is completed
    pub fn is_stage_complete(&self, stage: PipelineStage) -> bool {
        let stage_key = self.stage_to_key(stage);
        *self.stages_completed.get(&stage_key).unwrap_or(&false)
    }
    
    /// Get the output file for a completed stage
    pub fn get_stage_output(&self, stage: PipelineStage) -> Option<&str> {
        let stage_key = self.stage_to_key(stage);
        self.stage_outputs.get(&stage_key).map(|s| s.as_str())
    }
    
    /// Get the last completed stage
    pub fn last_completed_stage(&self) -> Option<PipelineStage> {
        // Check stages in reverse order to find the last completed one
        if self.is_stage_complete(PipelineStage::FinalMerge) {
            Some(PipelineStage::FinalMerge)
        } else if self.is_stage_complete(PipelineStage::Validation) {
            Some(PipelineStage::Validation)
        } else if self.is_stage_complete(PipelineStage::Clustering) {
            Some(PipelineStage::Clustering)
        } else if self.is_stage_complete(PipelineStage::Sorting) {
            Some(PipelineStage::Sorting)
        } else if self.is_stage_complete(PipelineStage::GridMatching) {
            Some(PipelineStage::GridMatching)
        } else {
            None
        }
    }
    
    /// Convert stage enum to string key for hashmap
    fn stage_to_key(&self, stage: PipelineStage) -> String {
        match stage {
            PipelineStage::GridMatching => "grid_matching",
            PipelineStage::Sorting => "sorting",
            PipelineStage::Clustering => "clustering",
            PipelineStage::Validation => "validation",
            PipelineStage::FinalMerge => "final_merge",
            PipelineStage::Complete => "complete",
        }.to_string()
    }
}

/// State information for resuming an interrupted run
#[derive(Debug, Clone)]
pub struct ResumeState {
    /// Directory containing the interrupted run
    pub run_directory: PathBuf,
    /// Metadata about the run
    pub metadata: RunMetadata,
    /// Stage to resume from
    pub resume_from_stage: PipelineStage,
    /// Input file for the resume stage (output from previous stage)
    pub input_file: Option<PathBuf>,
}

impl ResumeState {
    /// Create a resume state from metadata
    pub fn from_metadata(run_dir: PathBuf, metadata: RunMetadata) -> Result<Self> {
        // Determine what stage to resume from
        let (resume_from_stage, input_file) = match metadata.last_completed_stage() {
            Some(PipelineStage::GridMatching) => {
                let input = run_dir.join(PipelineStage::GridMatching.output_filename());
                (PipelineStage::Sorting, Some(input))
            },
            Some(PipelineStage::Sorting) => {
                let input = run_dir.join(PipelineStage::Sorting.output_filename());
                (PipelineStage::Clustering, Some(input))
            },
            Some(PipelineStage::Clustering) => {
                let input = run_dir.join(PipelineStage::Clustering.output_filename());
                (PipelineStage::Validation, Some(input))
            },
            Some(PipelineStage::Validation) => {
                let input = run_dir.join(PipelineStage::Validation.output_filename());
                (PipelineStage::FinalMerge, Some(input))
            },
            Some(PipelineStage::FinalMerge) => {
                // Already complete
                (PipelineStage::Complete, None)
            },
            Some(PipelineStage::Complete) => {
                // Already complete
                (PipelineStage::Complete, None)
            },
            None => {
                // No stages completed, start from beginning
                (PipelineStage::GridMatching, None)
            }
        };
        
        // Verify input file exists if required
        if let Some(ref input) = input_file {
            if !input.exists() {
                return Err(Error::Storage(format!(
                    "Resume input file not found: {:?}", input
                )));
            }
        }
        
        Ok(ResumeState {
            run_directory: run_dir,
            metadata,
            resume_from_stage,
            input_file,
        })
    }
    
    /// Get a summary of what will be resumed
    pub fn summary(&self) -> String {
        let completed_count = self.metadata.stages_completed
            .values()
            .filter(|&&v| v)
            .count();
        
        format!(
            "Resuming from stage: {:?}\n\
             Completed stages: {}/5\n\
             Run directory: {:?}\n\
             Started: {}\n\
             Last updated: {}",
            self.resume_from_stage,
            completed_count,
            self.run_directory,
            self.metadata.started_at,
            self.metadata.updated_at
        )
    }
}

// ============================================================================
// RESUME MANAGER
// ============================================================================

/// Manages resume operations for grid streaming pipeline
pub struct ResumeManager;

impl ResumeManager {
    /// Generate a deterministic run directory name from database paths
    pub fn generate_run_name(source_db: &Path, target_db: &Path) -> String {
        // Extract just the filenames without extensions
        let source_name = source_db
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        let target_name = target_db
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        // Sanitize names to be filesystem-safe
        let source_safe = Self::sanitize_filename(&source_name);
        let target_safe = Self::sanitize_filename(&target_name);
        
        format!("run_{}_{}", source_safe, target_safe)
    }
    
    /// Get the path to the run directory
    pub fn get_run_directory(source_db: &Path, target_db: &Path) -> PathBuf {
        let run_name = Self::generate_run_name(source_db, target_db);
        PathBuf::from("temp/streaming").join(run_name)
    }
    
    /// Get the metadata file path for a run directory
    pub fn metadata_path(run_dir: &Path) -> PathBuf {
        run_dir.join("run_metadata.json")
    }
    
    /// Check if a resumable run exists for the given database pair
    pub fn find_resumable_run(source_db: &Path, target_db: &Path) -> Result<Option<ResumeState>> {
        let run_dir = Self::get_run_directory(source_db, target_db);
        
        if !run_dir.exists() {
            debug!("No existing run directory found at {:?}", run_dir);
            return Ok(None);
        }
        
        let metadata_file = Self::metadata_path(&run_dir);
        if !metadata_file.exists() {
            warn!("Run directory exists but no metadata found at {:?}", metadata_file);
            return Ok(None);
        }
        
        // Load and validate metadata
        let metadata = Self::load_metadata(&metadata_file)?;
        
        // Verify database paths match
        let source_str = source_db.to_string_lossy().to_string();
        let target_str = target_db.to_string_lossy().to_string();
        
        if metadata.source_db != source_str || metadata.target_db != target_str {
            warn!("Metadata database paths don't match current paths");
            return Ok(None);
        }
        
        // Check if already complete
        if metadata.is_stage_complete(PipelineStage::Complete) 
            || metadata.is_stage_complete(PipelineStage::FinalMerge) {
            info!("Found completed run, no resume needed");
            return Ok(None);
        }
        
        // Create resume state
        let resume_state = ResumeState::from_metadata(run_dir, metadata)?;
        
        info!("Found resumable run: {}", resume_state.summary());
        Ok(Some(resume_state))
    }
    
    /// Save metadata to disk
    pub fn save_metadata(run_dir: &Path, metadata: &RunMetadata) -> Result<()> {
        let metadata_file = Self::metadata_path(run_dir);
        
        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| Error::Storage(format!("Failed to serialize metadata: {}", e)))?;
        
        fs::write(&metadata_file, json)
            .map_err(|e| Error::Storage(format!("Failed to write metadata: {}", e)))?;
        
        debug!("Saved metadata to {:?}", metadata_file);
        Ok(())
    }
    
    /// Load metadata from disk
    pub fn load_metadata(metadata_file: &Path) -> Result<RunMetadata> {
        let json = fs::read_to_string(metadata_file)
            .map_err(|e| Error::Storage(format!("Failed to read metadata: {}", e)))?;
        
        let metadata = serde_json::from_str(&json)
            .map_err(|e| Error::Storage(format!("Failed to parse metadata: {}", e)))?;
        
        Ok(metadata)
    }
    
    /// Initialize a new run directory with metadata
    pub fn initialize_run(source_db: &Path, target_db: &Path) -> Result<PathBuf> {
        let run_dir = Self::get_run_directory(source_db, target_db);
        
        // Create directory if it doesn't exist
        fs::create_dir_all(&run_dir)
            .map_err(|e| Error::Storage(format!("Failed to create run directory: {}", e)))?;
        
        // Create initial metadata
        let metadata = RunMetadata::new(source_db, target_db);
        Self::save_metadata(&run_dir, &metadata)?;
        
        info!("Initialized new run at {:?}", run_dir);
        Ok(run_dir)
    }
    
    /// Update metadata after completing a stage
    pub fn update_stage_completion(
        run_dir: &Path,
        stage: PipelineStage,
        output_file: &str,
        item_count: usize
    ) -> Result<()> {
        let metadata_file = Self::metadata_path(run_dir);
        
        // Load existing metadata
        let mut metadata = Self::load_metadata(&metadata_file)?;
        
        // Update with stage completion
        metadata.mark_stage_complete(stage, output_file, item_count);
        
        // Save back to disk
        Self::save_metadata(run_dir, &metadata)?;
        
        info!("Marked stage {:?} as complete with {} items", stage, item_count);
        Ok(())
    }
    
    /// Clean up an incomplete or corrupted run
    pub fn cleanup_run(source_db: &Path, target_db: &Path) -> Result<()> {
        let run_dir = Self::get_run_directory(source_db, target_db);
        
        if run_dir.exists() {
            fs::remove_dir_all(&run_dir)
                .map_err(|e| Error::Storage(format!("Failed to cleanup run directory: {}", e)))?;
            info!("Cleaned up run directory at {:?}", run_dir);
        }
        
        Ok(())
    }
    
    /// List all incomplete runs in the streaming directory
    pub fn list_incomplete_runs() -> Result<Vec<(PathBuf, RunMetadata)>> {
        let streaming_dir = PathBuf::from("temp/streaming");
        
        if !streaming_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut incomplete_runs = Vec::new();
        
        for entry in fs::read_dir(&streaming_dir)
            .map_err(|e| Error::Storage(format!("Failed to read streaming directory: {}", e)))? 
        {
            let entry = entry.map_err(|e| Error::Storage(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();
            
            if path.is_dir() {
                let metadata_file = Self::metadata_path(&path);
                if metadata_file.exists() {
                    if let Ok(metadata) = Self::load_metadata(&metadata_file) {
                        // Check if incomplete
                        if !metadata.is_stage_complete(PipelineStage::Complete) 
                            && !metadata.is_stage_complete(PipelineStage::FinalMerge) {
                            incomplete_runs.push((path, metadata));
                        }
                    }
                }
            }
        }
        
        Ok(incomplete_runs)
    }
    
    /// Sanitize a filename to be filesystem-safe
    fn sanitize_filename(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}
