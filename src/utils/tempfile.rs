// utils/tempfile.rs - Updated with binary I/O support

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use log::{debug, warn, info};

// IMPORTS for binary I/O
use crate::utils::binary_io::{BinaryWriter, BinaryReader, DataType, BinaryIOError};
use crate::matcher::types::ReconstructedSegment;
use crate::error::Result as AlnaqlResult;
use crate::error::Error as AlnaqlError;

// Add conversion from BinaryIOError to AlnaqlError
impl From<BinaryIOError> for AlnaqlError {
    fn from(err: BinaryIOError) -> Self {
        match err {
            BinaryIOError::Io(e) => AlnaqlError::Io(e),
            other => AlnaqlError::storage(format!("Binary I/O error: {}", other)),
        }
    }
}

/// Manages temporary files for storing intermediate processing results
pub struct TempFileManager {
    // Directory where temporary files will be stored
    temp_dir: PathBuf,
    // Prefix for temporary filenames
    prefix: String,
    // Track all created temp files
    temp_files: Arc<Mutex<Vec<PathBuf>>>,
    // Counter for unique file IDs
    counter: AtomicUsize,
    // Whether to keep files
    keep_files: bool,
}

// Implement Clone for TempFileManager
impl Clone for TempFileManager {
    fn clone(&self) -> Self {
        Self {
            temp_dir: self.temp_dir.clone(),
            prefix: self.prefix.clone(),
            temp_files: Arc::clone(&self.temp_files),
            counter: AtomicUsize::new(self.counter.load(Ordering::SeqCst)),
            keep_files: self.keep_files,  // Add this line
        }
    }
}

impl TempFileManager {
    /// Create a new TempFileManager
    ///
    /// # Arguments
    /// * `temp_dir` - Directory where temporary files will be stored
    /// * `prefix` - Prefix for temporary filenames
    ///
    /// # Returns
    /// A Result containing a new TempFileManager or an io::Error
    /// Create a new TempFileManager
    pub fn new<P: AsRef<Path>>(temp_dir: P, prefix: &str) -> io::Result<Self> {
        let temp_dir = temp_dir.as_ref();  // Convert to &Path
        
        // Create temp directory if it doesn't exist
        if !temp_dir.exists() {
            debug!("Creating temporary directory: {:?}", temp_dir);
            fs::create_dir_all(temp_dir)?;
        }
        
        Ok(Self {
            temp_dir: temp_dir.to_path_buf(),
            prefix: prefix.to_string(),
            temp_files: Arc::new(Mutex::new(Vec::new())),
            counter: AtomicUsize::new(0),
            keep_files: false,
        })
    }
    
    /// Set whether to keep files on cleanup (call after creation)
    pub fn set_keep_files(&mut self, keep: bool) {
        self.keep_files = keep;
    }
    
    /// Create a new temporary file with a unique name
    ///
    /// # Arguments
    /// * `chunk_id` - ID of the chunk being processed
    ///
    /// # Returns
    /// A tuple containing the file path and a File handle
    pub fn create_temp_file(&self, chunk_id: usize) -> io::Result<(PathBuf, File)> {
        // Get a timestamp for uniqueness
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Increment counter
        let counter = self.counter.fetch_add(1, Ordering::SeqCst);
        
        // Use .bin extension for binary files
        let filename = format!(
            "{}_chunk_{}_{}_{}.tmp.bin",
            self.prefix, chunk_id, timestamp, counter
        );
        
        let file_path = self.temp_dir.join(filename);
        debug!("Creating temporary file: {:?}", file_path);
        info!("DEBUG: Creating temporary file at: {:?}", file_path);
        info!("DEBUG: Parent directory exists: {}", file_path.parent().map_or(false, |p| p.exists()));
        
        // Create and open the file
        let file = File::create(&file_path)?;
        
        // Track the file
        if let Ok(mut files) = self.temp_files.lock() {
            files.push(file_path.clone());
        } else {
            warn!("Failed to lock temp_files mutex when creating {:?}", file_path);
        }
        
        Ok((file_path, file))
    }
    
    /// Write a collection of ReconstructedSegment objects to a temporary file
    /// UPDATED: Uses BinaryWriter with compression
    ///
    /// # Arguments
    /// * `matches` - Collection of ReconstructedSegment objects to write
    /// * `chunk_id` - ID of the chunk being processed
    ///
    /// # Returns
    /// The path to the created temporary file
    pub fn write_matches(&self, matches: &[ReconstructedSegment], chunk_id: usize) -> AlnaqlResult<PathBuf> {
        // Create a new temporary file
        let (file_path, file) = self.create_temp_file(chunk_id)
            .map_err(|e| AlnaqlError::Io(e))?;
        
        info!("Writing {} matches to compressed binary file: {:?}", matches.len(), file_path);
        
        // Use BinaryWriter with compression (always true for temp files)
        let mut writer = BinaryWriter::new(
            file, 
            true,  // Always compress temp files
            DataType::ReconstructedSegment
        )?;
        
        // Write all matches using binary serialization
        for segment in matches {
            writer.write_item(segment)?;
        }
        
        // Ensure everything is written
        writer.flush()?;
        
        info!("Successfully wrote {} matches to binary temp file", matches.len());
        
        Ok(file_path)
    }
    
    /// Read matches from a binary temporary file
    ///
    /// # Arguments
    /// * `path` - Path to the binary file to read
    ///
    /// # Returns
    /// A vector of ReconstructedSegment objects
    pub fn read_matches(&self, path: &Path) -> AlnaqlResult<Vec<ReconstructedSegment>> {
        info!("Reading matches from binary file: {:?}", path);
        
        let file = File::open(path)
            .map_err(|e| AlnaqlError::Io(e))?;
        
        // Create BinaryReader - it will auto-detect compression from header
        let mut reader = BinaryReader::new(file)?;
        
        // Verify we're reading the right type
        if reader.data_type() != DataType::ReconstructedSegment {
            return Err(AlnaqlError::storage(
                format!("Expected ReconstructedSegment data, found {:?}", reader.data_type())
            ));
        }
        
        // Read all matches
        let matches = reader.read_all()?;
        
        info!("Successfully read {} matches from binary file", matches.len());
        
        Ok(matches)
    }
    
    /// NEW METHOD: Stream matches from a binary file without loading all into memory
    ///
    /// # Arguments
    /// * `path` - Path to the binary file to stream
    ///
    /// # Returns
    /// An iterator over ReconstructedSegment objects
    pub fn stream_matches(&self, path: &Path) -> AlnaqlResult<impl Iterator<Item = AlnaqlResult<ReconstructedSegment>>> {
        let file = File::open(path)
            .map_err(|e| AlnaqlError::Io(e))?;
        
        let reader = BinaryReader::new(file)?;
        
        // Verify data type
        if reader.data_type() != DataType::ReconstructedSegment {
            return Err(AlnaqlError::storage(
                format!("Expected ReconstructedSegment data, found {:?}", reader.data_type())
            ));
        }
        
        // Return iterator that converts BinaryIOError to AlnaqlError
        Ok(reader.into_iter::<ReconstructedSegment>()
            .map(|result| result.map_err(|e| e.into())))
    }

    /// Get all temporary file paths
    ///
    /// # Returns
    /// A vector of paths to all temporary files created by this manager
    pub fn get_temp_files(&self) -> Vec<PathBuf> {
        if let Ok(files) = self.temp_files.lock() {
            files.clone()
        } else {
            warn!("Failed to lock temp_files mutex when getting temp files");
            Vec::new()
        }
    }

    /// Get the temporary directory path
    pub fn get_temp_dir(&self) -> PathBuf {
        self.temp_dir.clone()
    }
    
    /// Clean up all temporary files (respects keep_files flag)
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn cleanup(&self) -> io::Result<()> {
        if self.keep_files {
            info!("Keeping temporary files for debugging at: {:?}", self.temp_dir);
            
            // Log all temp files that were created
            if let Ok(files) = self.temp_files.lock() {
                if !files.is_empty() {
                    info!("Temporary files preserved ({} total):", files.len());
                    for (i, file) in files.iter().enumerate() {
                        if file.exists() {
                            // Get file size for info
                            if let Ok(metadata) = fs::metadata(file) {
                                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                                info!("  [{}] {:?} ({:.2} MB)", i + 1, file.file_name().unwrap_or_default(), size_mb);
                            } else {
                                info!("  [{}] {:?}", i + 1, file.file_name().unwrap_or_default());
                            }
                        } else {
                            info!("  [{}] {:?} (already deleted)", i + 1, file.file_name().unwrap_or_default());
                        }
                    }
                    info!("To manually clean up, run: rm -rf {:?}", self.temp_dir);
                }
            }
            
            return Ok(());
        }
        
        // Original cleanup code
        if let Ok(mut files) = self.temp_files.lock() {
            for file_path in files.iter() {
                if file_path.exists() {
                    debug!("Removing temporary file: {:?}", file_path);
                    if let Err(e) = fs::remove_file(file_path) {
                        warn!("Failed to remove temporary file {:?}: {}", file_path, e);
                    }
                }
            }
            files.clear();
        } else {
            warn!("Failed to lock temp_files mutex during cleanup");
        }
        Ok(())
    }
}