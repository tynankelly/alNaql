// boundary_handler.rs - Simplified version for memory-mapped files
use std::str;
use log::{debug, warn};
use crate::config::NGramType;
use crate::error::{Error, Result};

/// Handles finding safe chunk boundaries in memory-mapped files
/// Ensures chunks end at valid UTF-8 and optional word boundaries
pub struct BoundaryHandler {
    /// Ngram type (Character or Word) to determine boundary handling strategy
    ngram_type: NGramType,
    /// Track total bytes processed for debugging/validation
    bytes_processed: usize,
}

impl BoundaryHandler {
    /// Create a new boundary handler
    pub fn new(ngram_type: NGramType) -> Self {
        debug!("Creating BoundaryHandler for {:?} ngrams", ngram_type);
        
        Self {
            ngram_type,
            bytes_processed: 0,
        }
    }
    
    /// Find a safe ending position for a chunk in the memory-mapped data
    /// Returns the byte position where this chunk should end
    pub fn find_chunk_boundary(
        &mut self,
        file_data: &[u8],
        start_pos: usize,
        desired_chunk_size: usize,
    ) -> Result<usize> {
        // Sanity check
        if start_pos >= file_data.len() {
            return Ok(file_data.len());
        }
        
        // Calculate desired end position
        let mut end_pos = (start_pos + desired_chunk_size).min(file_data.len());
        
        // If we're at the end of the file, just return it
        if end_pos == file_data.len() {
            debug!("Chunk ends at file boundary: {}", end_pos);
            return Ok(end_pos);
        }
        
        // Step 1: Find a valid UTF-8 character boundary
        end_pos = self.find_utf8_boundary(file_data, end_pos, start_pos)?;
        
        // Step 2: For word ngrams, also find a word boundary
        if self.ngram_type == NGramType::Word {
            end_pos = self.find_word_boundary(file_data, start_pos, end_pos)?;
        }
        
        debug!("Chunk boundary: start={}, end={}, size={}", 
               start_pos, end_pos, end_pos - start_pos);
        
        // Update tracking
        self.bytes_processed = end_pos;
        
        Ok(end_pos)
    }
    
    /// Find the nearest valid UTF-8 character boundary at or before the target position
    fn find_utf8_boundary(
        &self,
        data: &[u8],
        target_pos: usize,
        min_pos: usize,
    ) -> Result<usize> {
        let mut pos = target_pos;
        
        // UTF-8 continuation bytes start with 0b10xxxxxx
        // We need to find a byte that starts a character (not 0b10xxxxxx)
        while pos > min_pos {
            // Check if this is the start of a UTF-8 character
            if data[pos] & 0b11000000 != 0b10000000 {
                // This is either ASCII (0xxxxxxx) or a multi-byte starter (11xxxxxx)
                debug!("Found UTF-8 boundary at position {}", pos);
                return Ok(pos);
            }
            pos -= 1;
        }
        
        // If we couldn't find a boundary (shouldn't happen with reasonable chunk sizes)
        Err(Error::text(format!(
            "Could not find UTF-8 boundary between {} and {}", 
            min_pos, target_pos
        )))
    }
    
    /// Find the nearest word boundary at or before the target position
    fn find_word_boundary(
        &self,
        data: &[u8],
        start_pos: usize,
        end_pos: usize,
    ) -> Result<usize> {
        // Extract the chunk as a string to find word boundaries
        let chunk_data = &data[start_pos..end_pos];
        
        // Convert to string (should always work since we found UTF-8 boundary)
        let text = str::from_utf8(chunk_data)
            .map_err(|e| Error::text(format!("UTF-8 conversion failed: {}", e)))?;
        
        // Find the last word boundary (whitespace or punctuation)
        if let Some(last_boundary_pos) = text.rfind(|c: char| {
            c.is_whitespace() || 
            // Common punctuation
            c == '.' || c == '!' || c == '?' || c == ',' || c == ';' || c == ':' ||
            // Arabic punctuation
            c == '،' || c == '؛' || c == '؟' || c == '۔' || c == '؍' ||
            // XML markers (if you want to break at tags)
            c == '>' || c == '<'
        }) {
            let boundary_byte_pos = start_pos + text[..last_boundary_pos + 1].len();
            debug!("Found word boundary at position {}", boundary_byte_pos);
            Ok(boundary_byte_pos)
        } else {
            // No word boundary found - this might happen with very long words
            // or continuous text without breaks
            warn!("No word boundary found in chunk, using UTF-8 boundary");
            Ok(end_pos)
        }
    }
    
    /// Get total bytes processed so far
    pub fn bytes_processed(&self) -> usize {
        self.bytes_processed
    }
    
    /// Reset the handler for processing a new file
    pub fn reset(&mut self) {
        self.bytes_processed = 0;
        debug!("BoundaryHandler reset");
    }
}

/// Helper function to process an entire memory-mapped file
impl BoundaryHandler {
    /// Process a complete file using memory-mapped data
    /// This is a convenience method showing how to use the boundary handler
    pub fn process_file<F>(
        &mut self,
        file_data: &[u8],
        chunk_size: usize,
        mut process_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(&str, usize) -> Result<()>,
    {
        let file_size = file_data.len();
        let mut position = 0;
        let mut chunk_count = 0;
        
        debug!("Processing file of {} bytes with chunk size {}", file_size, chunk_size);
        
        while position < file_size {
            // Find safe boundary for this chunk
            let chunk_end = self.find_chunk_boundary(
                file_data,
                position,
                chunk_size
            )?;
            
            // Extract chunk data
            let chunk_bytes = &file_data[position..chunk_end];
            
            // Convert to string (safe because we found proper boundaries)
            let text = str::from_utf8(chunk_bytes)
                .map_err(|e| Error::text(format!("UTF-8 conversion failed: {}", e)))?;
            
            // Process this chunk
            chunk_count += 1;
            debug!("Processing chunk {} at position {}, size {}", 
                   chunk_count, position, chunk_bytes.len());
            
            process_chunk(text, position)?;
            
            // Move to next chunk
            position = chunk_end;
            
            // Break if we've reached the end
            if position >= file_size {
                break;
            }
        }
        
        debug!("Completed processing {} chunks, {} total bytes", 
               chunk_count, self.bytes_processed);
        
        Ok(())
    }
}
