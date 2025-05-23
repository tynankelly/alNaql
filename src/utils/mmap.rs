use std::fs::File;
use std::path::{Path, PathBuf};
use memmap2::Mmap;
use log::debug;
use crate::error::Result;

pub struct MmapFileHandler {
    mapped_file: Mmap,
    file_size: usize,
    file_path: PathBuf,
    default_chunk_size: usize,
    max_chunks: usize,
}

impl MmapFileHandler {
    pub fn new<P: AsRef<Path>>(path: P, default_chunk_size: usize, max_chunks: usize) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)?;
        let file_size = file.metadata()?.len() as usize;
        
        // Create memory mapping of the file
        let mapped_file = unsafe { Mmap::map(&file)? };
        
        debug!("Created memory-mapped file handler for {:?} ({} bytes)", file_path, file_size);
        
        Ok(Self {
            mapped_file,
            file_size,
            file_path,
            default_chunk_size: default_chunk_size.max(1024), // Minimum 1KB
            max_chunks, // 0 means no limit
        })
    }
    
    /// Get a reference to the entire file content
    pub fn get_content(&self) -> &[u8] {
        &self.mapped_file
    }
    
    /// Get the file size
    pub fn get_size(&self) -> usize {
        self.file_size
    }
    
    /// Get the file path
    pub fn get_path(&self) -> &Path {
        &self.file_path
    }
    
    /// Get iterator over file chunks with their absolute positions
    /// Returns pairs of (chunk, absolute_position)
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[u8]> + '_ {
        // Use the provided chunk size or fall back to default
        let effective_chunk_size = if chunk_size > 0 { 
            chunk_size 
        } else { 
            self.default_chunk_size 
        };
        
        struct ChunkIterator<'a> {
            file_data: &'a [u8],
            chunk_size: usize,
            current_position: usize,
            file_size: usize,
            max_chunks: usize,
            chunks_returned: usize,
        }
        
        impl<'a> Iterator for ChunkIterator<'a> {
            type Item = &'a [u8];
            
            fn next(&mut self) -> Option<Self::Item> {
                // Check if we've reached the end of the file
                if self.current_position >= self.file_size {
                    return None;
                }
                
                // Check if we've hit the maximum chunk limit
                if self.max_chunks > 0 && self.chunks_returned >= self.max_chunks {
                    return None;
                }
                
                // Calculate end of this chunk
                let end = (self.current_position + self.chunk_size).min(self.file_size);
                let chunk = &self.file_data[self.current_position..end];
                
                // Update state for next iteration
                self.current_position = end;
                self.chunks_returned += 1;
                
                Some(chunk)
            }
        }
        
        ChunkIterator {
            file_data: &self.mapped_file,
            chunk_size: effective_chunk_size,
            current_position: 0,
            file_size: self.file_size,
            max_chunks: self.max_chunks,
            chunks_returned: 0,
        }
    }
    
    /// NEW METHOD: Get iterator over file chunks with their absolute positions
    /// Returns pairs of (chunk, absolute_position)
    pub fn chunks_with_position(&self, chunk_size: usize) -> impl Iterator<Item = (&[u8], usize)> + '_ {
        // Use the provided chunk size or fall back to default
        let effective_chunk_size = if chunk_size > 0 { 
            chunk_size 
        } else { 
            self.default_chunk_size 
        };
        
        struct PositionedChunkIterator<'a> {
            file_data: &'a [u8],
            chunk_size: usize,
            current_position: usize,
            file_size: usize,
            max_chunks: usize,
            chunks_returned: usize,
        }
        
        impl<'a> Iterator for PositionedChunkIterator<'a> {
            type Item = (&'a [u8], usize);
            
            fn next(&mut self) -> Option<Self::Item> {
                // Check if we've reached the end of the file
                if self.current_position >= self.file_size {
                    return None;
                }
                
                // Check if we've hit the maximum chunk limit
                if self.max_chunks > 0 && self.chunks_returned >= self.max_chunks {
                    return None;
                }
                
                // Calculate end of this chunk
                let end = (self.current_position + self.chunk_size).min(self.file_size);
                let chunk = &self.file_data[self.current_position..end];
                
                // Store absolute position for this chunk
                let absolute_position = self.current_position;
                
                // Update state for next iteration
                self.current_position = end;
                self.chunks_returned += 1;
                
                Some((chunk, absolute_position))
            }
        }
        
        PositionedChunkIterator {
            file_data: &self.mapped_file,
            chunk_size: effective_chunk_size,
            current_position: 0,
            file_size: self.file_size,
            max_chunks: self.max_chunks,
            chunks_returned: 0,
        }
    }
}