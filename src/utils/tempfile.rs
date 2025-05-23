// utils/tempfile.rs
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{self, Write, BufReader, BufWriter, Lines};
use std::io::BufRead; // Added BufRead trait
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use log::{debug, warn, info};
use crate::matcher::MatchResult;
use crate::error::Result as KuttabResult;
use crate::error::Error as KuttabError;

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
}

// Implement Clone for TempFileManager 
impl Clone for TempFileManager {
    fn clone(&self) -> Self {
        Self {
            temp_dir: self.temp_dir.clone(),
            prefix: self.prefix.clone(),
            temp_files: Arc::clone(&self.temp_files),
            counter: AtomicUsize::new(self.counter.load(Ordering::SeqCst)),
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
    pub fn new<P: AsRef<Path>>(temp_dir: P, prefix: &str) -> io::Result<Self> {
        // Create the directory if it doesn't exist
        fs::create_dir_all(&temp_dir)?;

        info!("DEBUG: Temporary directory created at: {:?}", temp_dir.as_ref());
        let dir_exists = Path::new(temp_dir.as_ref()).exists();
        info!("DEBUG: Directory exists check: {}", dir_exists);
        
        let temp_dir_path = PathBuf::from(temp_dir.as_ref());
        
        Ok(Self {
            temp_dir: temp_dir_path,
            prefix: prefix.to_string(),
            temp_files: Arc::new(Mutex::new(Vec::new())),
            counter: AtomicUsize::new(0),
        })
    }
    
    /// Create a new temporary file with a unique name
    ///
    /// # Arguments
    /// * `chunk_id` - ID of the chunk being processed
    ///
    /// # Returns
    /// A tuple containing the file path and a BufWriter to the file
    pub fn create_temp_file(&self, chunk_id: usize) -> io::Result<(PathBuf, BufWriter<File>)> {
        // Get a timestamp for uniqueness
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // Increment counter
        let counter = self.counter.fetch_add(1, Ordering::SeqCst);
        
        // Create a unique filename
        let filename = format!(
            "{}_chunk_{}_{}_{}.tmp.jsonl",
            self.prefix, chunk_id, timestamp, counter
        );
        
        let file_path = self.temp_dir.join(filename);
        debug!("Creating temporary file: {:?}", file_path);
        info!("DEBUG: Creating temporary file at: {:?}", file_path);
        info!("DEBUG: Parent directory exists: {}", file_path.parent().map_or(false, |p| p.exists()));
        
        // Create and open the file
        let file = File::create(&file_path)?;
        let writer = BufWriter::new(file);
        
        // Track the file
        if let Ok(mut files) = self.temp_files.lock() {
            files.push(file_path.clone());
        } else {
            warn!("Failed to lock temp_files mutex when creating {:?}", file_path);
        }
        
        Ok((file_path, writer))
    }
    
    /// Write a collection of MatchResult objects to a temporary file
    ///
    /// # Arguments
    /// * `matches` - Collection of MatchResult objects to write
    /// * `chunk_id` - ID of the chunk being processed
    ///
    /// # Returns
    /// The path to the created temporary file
    pub fn write_matches(&self, matches: &[MatchResult], chunk_id: usize) -> KuttabResult<PathBuf> {
        info!("DEBUG: About to write {} matches to temporary file for chunk {}", matches.len(), chunk_id);
        // Create a new temporary file
        let (file_path, mut writer) = self.create_temp_file(chunk_id)
            .map_err(|e| KuttabError::Io(e))?;
        
        info!("Writing {} matches to temporary file: {:?}", matches.len(), file_path);
        
        // Write each match as a JSON line
        for match_result in matches {
            let json = serde_json::to_string(&match_result)
                .map_err(|e| KuttabError::Json(e))?;
            writeln!(writer, "{}", json)
                .map_err(|e| KuttabError::Io(e))?;
        }
        
        // Ensure everything is written
        writer.flush()
            .map_err(|e| KuttabError::Io(e))?;
        
        info!("DEBUG: Successfully wrote matches to temporary file: {:?}", file_path);
        info!("DEBUG: File exists check: {}", file_path.exists());

        Ok(file_path)
    }
    
    /// Read all matches from a temporary file
    ///
    /// # Arguments
    /// * `file_path` - Path to the temporary file
    ///
    /// # Returns
    /// A vector of MatchResult objects
    pub fn read_matches(&self, file_path: &Path) -> KuttabResult<Vec<MatchResult>> {
        debug!("Reading all matches from file: {:?}", file_path);
        
        let file = File::open(file_path)
            .map_err(|e| KuttabError::Io(e))?;
        let reader = BufReader::new(file);
        
        let mut matches = Vec::new();
        
        for line in reader.lines() {
            let line = line.map_err(|e| KuttabError::Io(e))?;
            let match_result = serde_json::from_str::<MatchResult>(&line)
                .map_err(|e| KuttabError::Json(e))?;
            matches.push(match_result);
        }
        
        debug!("Read {} matches from {:?}", matches.len(), file_path);
        Ok(matches)
    }
    
    /// Read matches from a temporary file in batches and process them
    ///
    /// # Arguments
    /// * `file_path` - Path to the temporary file
    /// * `batch_size` - Number of matches to read in each batch
    /// * `callback` - Function to call with each batch of matches
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn read_matches_batched<F>(&self, file_path: &Path, batch_size: usize, mut callback: F) -> KuttabResult<()>
    where
        F: FnMut(Vec<MatchResult>) -> KuttabResult<()>,
    {
        debug!("Reading matches in batches of {} from {:?}", batch_size, file_path);
        
        let file = File::open(file_path)
            .map_err(|e| KuttabError::Io(e))?;
        let reader = BufReader::new(file);
        
        let mut batch = Vec::with_capacity(batch_size);
        let mut total_read = 0;
        
        for line in reader.lines() {
            let line = line.map_err(|e| KuttabError::Io(e))?;
            let match_result = serde_json::from_str::<MatchResult>(&line)
                .map_err(|e| KuttabError::Json(e))?;
            
            batch.push(match_result);
            total_read += 1;
            
            if batch.len() >= batch_size {
                debug!("Processing batch of {} matches", batch.len());
                callback(batch)?;
                batch = Vec::with_capacity(batch_size);
            }
        }
        
        // Process any remaining matches
        if !batch.is_empty() {
            debug!("Processing final batch of {} matches", batch.len());
            callback(batch)?;
        }
        
        debug!("Completed batch processing of {} total matches from {:?}", total_read, file_path);
        Ok(())
    }
    
    /// Create a line iterator for a temporary file that yields MatchResult objects
    ///
    /// # Arguments
    /// * `file_path` - Path to the temporary file
    ///
    /// # Returns
    /// A MatchIterator that yields MatchResult objects
    pub fn match_iterator(&self, file_path: &Path) -> KuttabResult<MatchIterator> {
        let file = File::open(file_path)
            .map_err(|e| KuttabError::Io(e))?;
        let reader = BufReader::new(file);
        
        Ok(MatchIterator {
            lines: reader.lines(),
        })
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

    pub fn get_temp_dir(&self) -> PathBuf {
        self.temp_dir.clone()
    }
    
    /// Clean up all temporary files
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn cleanup(&self) -> io::Result<()> {
        if let Ok(mut files) = self.temp_files.lock() {
            for file_path in files.iter() {
                if file_path.exists() {
                    debug!("Removing temporary file: {:?}", file_path);
                    if let Err(e) = fs::remove_file(file_path) {
                        warn!("Failed to remove temporary file {:?}: {}", file_path, e);
                    }
                }
            }
            
            // Clear the list
            files.clear();
            Ok(())
        } else {
            warn!("Failed to lock temp_files mutex during cleanup");
            Err(io::Error::new(io::ErrorKind::Other, "Failed to lock temp_files mutex"))
        }
    }
    
    /// Clean up a specific temporary file
    ///
    /// # Arguments
    /// * `file_path` - Path to the temporary file to clean up
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn cleanup_file(&self, file_path: &Path) -> io::Result<()> {
        if let Ok(mut files) = self.temp_files.lock() {
            if file_path.exists() {
                debug!("Removing temporary file: {:?}", file_path);
                if let Err(e) = fs::remove_file(file_path) {
                    warn!("Failed to remove temporary file {:?}: {}", file_path, e);
                    return Err(e);
                }
            }
            
            // Remove from the list
            files.retain(|p| p != file_path);
            Ok(())
        } else {
            warn!("Failed to lock temp_files mutex when cleaning up file {:?}", file_path);
            Err(io::Error::new(io::ErrorKind::Other, "Failed to lock temp_files mutex"))
        }
    }
}

/// Iterator that yields MatchResult objects from a file
pub struct MatchIterator {
    lines: Lines<BufReader<File>>,
}

impl Iterator for MatchIterator {
    type Item = KuttabResult<MatchResult>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.lines.next() {
            Some(Ok(line)) => {
                match serde_json::from_str::<MatchResult>(&line) {
                    Ok(match_result) => Some(Ok(match_result)),
                    Err(e) => Some(Err(KuttabError::Json(e))),
                }
            },
            Some(Err(e)) => Some(Err(KuttabError::Io(e))),
            None => None,
        }
    }
}

// Implement Drop to ensure cleanup on drop
impl Drop for TempFileManager {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            warn!("Error during TempFileManager cleanup: {}", e);
        }
    }
}
