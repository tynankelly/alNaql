// boundary_handler.rs with completely revised UTF-8 handling
use std::str;
use log::{debug, warn, error};
use crate::config::subsystems::generator::NGramType;

/// Handles text boundaries for streaming processing
pub struct BoundaryHandler {
    /// Buffer for incomplete UTF-8 sequences
    utf8_buffer: Vec<u8>,
    /// Buffer for overlapping text between chunks
    overlap_buffer: String,
    /// Size of overlap to maintain between chunks
    overlap_size: usize,
    /// Last complete boundary position
    last_complete_position: usize,
    /// Current absolute position in the file
    current_absolute_position: usize,
    /// Absolute position of text in the overlap buffer 
    overlap_start_position: usize,
    /// Ngram type (Character or Word) to determine boundary handling strategy
    ngram_type: NGramType,
}

impl BoundaryHandler {
    pub fn new(overlap_size: usize, ngram_type: NGramType) -> Self {
        // Log the creation parameters
        debug!("Creating BoundaryHandler with overlap_size={} and ngram_type={:?}", 
               overlap_size, ngram_type);
        
        let max_overlap_size = overlap_size * 2;
        Self {
            utf8_buffer: Vec::with_capacity(1024),
            overlap_buffer: String::with_capacity(max_overlap_size),
            overlap_size,
            last_complete_position: 0,
            current_absolute_position: 0,
            overlap_start_position: 0,
            ngram_type,
        }
    }
    
    /// Process a chunk with its absolute position in the file
    /// Returns (combined_text, absolute_position, overlap_bytes) if successful
    pub fn process_chunk(&mut self, chunk: &[u8], chunk_position: usize) -> Option<(String, usize, usize)> {
        debug!("Processing chunk at position {} with size {}", chunk_position, chunk.len());
        
        // Update absolute position tracking
        self.current_absolute_position = chunk_position;
        
        // Append the new chunk to any incomplete UTF-8 sequence from previous chunk
        self.utf8_buffer.extend_from_slice(chunk);
        
        // Try to convert the buffer to a UTF-8 string
        match std::str::from_utf8(&self.utf8_buffer) {
            Ok(text) => {
                // Successfully converted entire buffer
                debug!("Successfully converted chunk to UTF-8, length: {}", text.len());
                
                // Calculate overlap size in bytes before combining
                let overlap_bytes = self.overlap_buffer.len();
                debug!("Current overlap buffer size: {} bytes", overlap_bytes);
                
                // Combine overlap buffer with new text
                let combined = format!("{}{}", self.overlap_buffer, text);
                debug!("Combined text length: {} bytes", combined.len());
                
                // The absolute position of the combined text is the position of the overlap
                // If no overlap (first chunk), it's the chunk position
                let text_absolute_position = if overlap_bytes > 0 {
                    self.overlap_start_position
                } else {
                    chunk_position
                };

                debug!("BOUNDARY_DEBUG: Chunk at pos={}, overlap_buffer size={}, text_absolute_position={}", 
                    chunk_position, overlap_bytes, text_absolute_position);
                    debug!("BOUNDARY_DEBUG: First 30 chars of chunk: \"{}\"", 
                    std::str::from_utf8(&chunk[..std::cmp::min(30, chunk.len())])
                        .unwrap_or("<invalid UTF-8>")
                        .chars()
                        .take(30)
                        .collect::<String>()
                        .replace('\n', "\\n"));
                
                // Calculate overlap for next chunk - SAFELY
                if combined.len() > self.overlap_size {
                    debug!("Computing overlap for next chunk (overlap_size: {})", self.overlap_size);
                    
                    // COMPLETELY REVISED APPROACH TO ENSURE SAFE BOUNDARIES
                    // Instead of trying to calculate boundaries directly, we'll work with character indices
                    
                    // Convert to a vector of characters with their byte indices
                    let char_indices: Vec<(usize, char)> = combined.char_indices().collect();
                    
                    if char_indices.is_empty() {
                        // Empty string case
                        debug!("Combined text is empty");
                        self.overlap_buffer.clear();
                        self.overlap_start_position = text_absolute_position;
                    } else {
                        // Get the total character count
                        let char_count = char_indices.len();
                        
                        // Calculate number of characters to keep in overlap
                        let chars_to_keep = match self.ngram_type {
                            NGramType::Character => self.overlap_size,
                            NGramType::Word => self.overlap_size * 2, // Keep more for word ngrams
                        };
                        
                        if chars_to_keep >= char_count {
                            // If we need to keep more characters than we have, keep everything
                            debug!("chars_to_keep ({}) >= char_count ({}), keeping entire combined text", 
                                   chars_to_keep, char_count);
                            self.overlap_buffer = combined.clone();
                            self.overlap_start_position = text_absolute_position;
                        } else {
                            // We have enough characters to work with
                            
                            // Start from the desired number of characters from the end
                            let target_char_idx = char_count - chars_to_keep;
                            
                            // Get byte position for this character
                            let (byte_pos, _) = char_indices[target_char_idx];
                            
                            // For word ngrams, try to align to a word boundary
                            let safe_byte_pos = if self.ngram_type == NGramType::Word {
                                // Find a word boundary near this position
                                self.find_word_boundary_by_chars(&combined, &char_indices, target_char_idx)
                            } else {
                                // Character ngrams can use character boundaries directly
                                byte_pos
                            };
                            
                            debug!("Using safe byte position {} for overlap", safe_byte_pos);
                            
                            // Verify this is a valid boundary (should always be true since we used char_indices)
                            if !combined.is_char_boundary(safe_byte_pos) {
                                // This should never happen, but if it does, find a valid boundary
                                error!("CRITICAL: Position {} is not a valid char boundary!", safe_byte_pos);
                                
                                // Find closest valid boundary
                                let mut valid_pos = safe_byte_pos;
                                while valid_pos < combined.len() && !combined.is_char_boundary(valid_pos) {
                                    valid_pos += 1;
                                }
                                if valid_pos >= combined.len() {
                                    valid_pos = safe_byte_pos;
                                    while valid_pos > 0 && !combined.is_char_boundary(valid_pos) {
                                        valid_pos -= 1;
                                    }
                                }
                                
                                debug!("Emergency adjusted to valid boundary at {}", valid_pos);
                                
                                // Create overlap buffer from this valid position
                                self.overlap_buffer = combined[valid_pos..].to_string();
                                self.overlap_start_position = text_absolute_position + valid_pos;
                            } else {
                                // Normal case - safe_byte_pos is a valid boundary
                                // Create overlap buffer
                                self.overlap_buffer = combined[safe_byte_pos..].to_string();
                                self.overlap_start_position = text_absolute_position + safe_byte_pos;
                                debug!("BOUNDARY_DEBUG: New overlap starts at abs_pos={}, rel_pos={}, text=\"{}\"", 
                                    self.overlap_start_position, 
                                    safe_byte_pos, 
                                    self.overlap_buffer.chars().take(20).collect::<String>()
                                        .replace('\n', "\\n"));
                            }
                            
                            debug!("Created new overlap buffer, length: {} bytes, starting at absolute position: {}", 
                                   self.overlap_buffer.len(), self.overlap_start_position);
                        }
                    }
                } else {
                    // Combined text is shorter than the desired overlap size
                    self.overlap_buffer = combined.clone();
                    self.overlap_start_position = text_absolute_position;
                    debug!("Text shorter than overlap size, keeping everything");
                }
                
                // Clear UTF-8 buffer since we processed everything
                self.utf8_buffer.clear();
                
                // Return the combined text, absolute position, and overlap size
                Some((combined, text_absolute_position, overlap_bytes))
            },
            Err(e) => {
                // Handle UTF-8 conversion errors
                warn!("UTF-8 conversion error: {}", e);
                let valid_up_to = e.valid_up_to();
                
                if valid_up_to > 0 {
                    // We have some valid UTF-8 at the beginning
                    debug!("Partial UTF-8 conversion up to position {}", valid_up_to);
                    let valid_text = std::str::from_utf8(&self.utf8_buffer[..valid_up_to]).unwrap();
                    
                    // Calculate overlap size in bytes before combining
                    let overlap_bytes = self.overlap_buffer.len();
                    
                    let combined = format!("{}{}", self.overlap_buffer, valid_text);
                    debug!("Combined text length (partial conversion): {} bytes", combined.len());
                    
                    // The absolute position of the combined text
                    let text_absolute_position = if overlap_bytes > 0 {
                        self.overlap_start_position
                    } else {
                        chunk_position
                    };
                    
                    // Calculate overlap (same approach as above for consistency)
                    if combined.len() > self.overlap_size {
                        debug!("Computing overlap for next chunk (partial conversion)");
                        
                        // Use the char_indices approach for consistency
                        let char_indices: Vec<(usize, char)> = combined.char_indices().collect();
                        
                        if char_indices.is_empty() {
                            self.overlap_buffer.clear();
                            self.overlap_start_position = text_absolute_position;
                        } else {
                            let char_count = char_indices.len();
                            
                            // Calculate overlap size in characters
                            let chars_to_keep = match self.ngram_type {
                                NGramType::Character => self.overlap_size,
                                NGramType::Word => self.overlap_size * 2,
                            };
                            
                            if chars_to_keep >= char_count {
                                self.overlap_buffer = combined.clone();
                                self.overlap_start_position = text_absolute_position;
                            } else {
                                let target_char_idx = char_count - chars_to_keep;
                                let (byte_pos, _) = char_indices[target_char_idx];
                                
                                let safe_byte_pos = if self.ngram_type == NGramType::Word {
                                    self.find_word_boundary_by_chars(&combined, &char_indices, target_char_idx)
                                } else {
                                    byte_pos
                                };
                                
                                debug!("Using safe byte position {} for overlap (partial)", safe_byte_pos);
                                
                                // Verify valid boundary
                                if !combined.is_char_boundary(safe_byte_pos) {
                                    error!("CRITICAL: Position {} is not a valid char boundary! (partial)", safe_byte_pos);
                                    
                                    // Find closest valid boundary
                                    let mut valid_pos = safe_byte_pos;
                                    while valid_pos < combined.len() && !combined.is_char_boundary(valid_pos) {
                                        valid_pos += 1;
                                    }
                                    if valid_pos >= combined.len() {
                                        valid_pos = safe_byte_pos;
                                        while valid_pos > 0 && !combined.is_char_boundary(valid_pos) {
                                            valid_pos -= 1;
                                        }
                                    }
                                    
                                    debug!("Emergency adjusted to valid boundary at {} (partial)", valid_pos);
                                    
                                    self.overlap_buffer = combined[valid_pos..].to_string();
                                    self.overlap_start_position = text_absolute_position + valid_pos;
                                } else {
                                    self.overlap_buffer = combined[safe_byte_pos..].to_string();
                                    self.overlap_start_position = text_absolute_position + safe_byte_pos;
                                    debug!("BOUNDARY_DEBUG: New overlap starts at abs_pos={}, rel_pos={}, text=\"{}\"", 
                                        self.overlap_start_position, 
                                        safe_byte_pos, 
                                        self.overlap_buffer.chars().take(20).collect::<String>()
                                            .replace('\n', "\\n"));
                                }
                            }
                        }
                    } else {
                        self.overlap_buffer = combined.clone();
                        self.overlap_start_position = text_absolute_position;
                    }
                    
                    // Keep remaining bytes for next chunk
                    self.utf8_buffer = self.utf8_buffer[valid_up_to..].to_vec();
                    debug!("Keeping {} invalid bytes for next chunk", self.utf8_buffer.len());
                    
                    // Return the combined text, absolute position, and overlap size
                    Some((combined, text_absolute_position, overlap_bytes))
                } else {
                    // No valid UTF-8 at all, buffer for next chunk
                    warn!("No valid UTF-8 found in chunk, buffering for next chunk");
                    None
                }
            }
        }
    }
    
    /// Find a word boundary near a character position using character indices
    /// This is a more robust approach than working with byte positions directly
    fn find_word_boundary_by_chars(&self, _text: &str, char_indices: &[(usize, char)], target_char_idx: usize) -> usize {
        debug!("Finding word boundary near character index {}", target_char_idx);
        
        if char_indices.is_empty() || target_char_idx >= char_indices.len() {
            debug!("Invalid target character index");
            return 0;
        }
        
        // Get the byte position for this character
        let (byte_pos, _) = char_indices[target_char_idx];
        
        // First, check if we're already at a good boundary (like a space)
        if target_char_idx < char_indices.len() - 1 {
            let (_, current_char) = char_indices[target_char_idx];
            if current_char.is_whitespace() {
                debug!("Already at a space boundary at char index {}", target_char_idx);
                return byte_pos;
            }
        }
        
        // Look for a space before this position
        let mut space_before_idx = target_char_idx;
        while space_before_idx > 0 {
            let (_, c) = char_indices[space_before_idx];
            if c.is_whitespace() {
                // Found a space - use the position after it
                let (pos, _) = char_indices[space_before_idx + 1];
                debug!("Found space before at char index {}, using position {}", space_before_idx, pos);
                return pos;
            }
            space_before_idx -= 1;
        }
        
        // Look for a space after this position
        let mut space_after_idx = target_char_idx;
        while space_after_idx < char_indices.len() - 1 {
            let (_, c) = char_indices[space_after_idx];
            if c.is_whitespace() {
                // Found a space - use its position
                let (pos, _) = char_indices[space_after_idx];
                debug!("Found space after at char index {}, using position {}", space_after_idx, pos);
                return pos;
            }
            space_after_idx += 1;
        }
        
        // No space found - use the original character boundary
        debug!("No word boundary found, using original position {}", byte_pos);
        byte_pos
    }
    
    /// Get the absolute position of the current overlap buffer
    pub fn get_overlap_position(&self) -> usize {
        self.overlap_start_position
    }
    
    /// Get the current overlap buffer content
    pub fn get_overlap(&self) -> &str {
        &self.overlap_buffer
    }
    
    /// Reset the handler state
    pub fn reset(&mut self) {
        debug!("Resetting boundary handler state");
        self.utf8_buffer.clear();
        self.overlap_buffer.clear();
        self.last_complete_position = 0;
        self.current_absolute_position = 0;
        self.overlap_start_position = 0;
    }
}