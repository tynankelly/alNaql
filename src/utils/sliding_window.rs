// sliding_window.rs - Add this as a new file in the utils directory

use std::collections::VecDeque;
use crate::config::subsystems::generator::NGramType;
use crate::types::NGramEntry;

/// Handles sliding window processing for streaming ngram generation
pub struct SlidingWindow {
    /// The type of ngrams being generated
    ngram_type: NGramType,
    /// The size of ngrams to generate
    ngram_size: usize,
    /// The buffer for maintaining context across chunks
    buffer: VecDeque<String>,
    /// Maximum buffer size (to prevent unbounded growth)
    max_buffer_size: usize,
    /// The current sequence number
    current_sequence: u64,
    /// File path for generated ngrams
    file_path: String,
}

impl SlidingWindow {
    /// Create a new SlidingWindow with the specified ngram type and size
    pub fn new(ngram_type: NGramType, ngram_size: usize) -> Self {
        // For word ngrams, we need a larger buffer
        let max_buffer_size = match ngram_type {
            NGramType::Character => ngram_size * 10,    // Store 10x the character ngram size
            NGramType::Word => ngram_size * 50,         // Store 50x the word ngram size
        };
        
        Self {
            ngram_type,
            ngram_size,
            buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            current_sequence: 1,
            file_path: String::new(),
        }
    }
    
    /// Set the file path for generated ngrams
    pub fn set_file_path(&mut self, path: &str) {
        self.file_path = path.to_string();
    }
    
    /// Reset the sequence counter
    pub fn reset_sequence(&mut self) {
        self.current_sequence = 1;
    }
    
    /// Process a chunk of text and generate ngrams
    pub fn process_chunk(&mut self, text: &str) -> Vec<NGramEntry> {
        match self.ngram_type {
            NGramType::Character => self.process_character_chunk(text),
            NGramType::Word => self.process_word_chunk(text),
        }
    }
    
    /// Process a chunk for character-based ngrams
    fn process_character_chunk(&mut self, text: &str) -> Vec<NGramEntry> {
        let mut ngrams = Vec::new();
        
        // Add characters to the buffer
        for c in text.chars() {
            self.buffer.push_back(c.to_string());
            
            // Keep buffer size within bounds
            while self.buffer.len() > self.max_buffer_size {
                self.buffer.pop_front();
            }
            
            // Generate ngrams when buffer reaches required size
            if self.buffer.len() >= self.ngram_size {
                if let Some(ngram) = self.create_character_ngram() {
                    ngrams.push(ngram);
                    self.current_sequence += 1;
                }
            }
        }
        
        ngrams
    }
    
    /// Process a chunk for word-based ngrams
    fn process_word_chunk(&mut self, text: &str) -> Vec<NGramEntry> {
        let mut ngrams = Vec::new();
        
        // Split text into words
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Add words to the buffer
        for word in words {
            self.buffer.push_back(word.to_string());
            
            // Keep buffer size within bounds
            while self.buffer.len() > self.max_buffer_size {
                self.buffer.pop_front();
            }
            
            // Generate ngrams when buffer reaches required size
            if self.buffer.len() >= self.ngram_size {
                if let Some(ngram) = self.create_word_ngram() {
                    ngrams.push(ngram);
                    self.current_sequence += 1;
                }
            }
        }
        
        ngrams
    }
    
    /// Create a character-based ngram from the current buffer
    fn create_character_ngram(&self) -> Option<NGramEntry> {
        if self.buffer.len() < self.ngram_size {
            return None;
        }
        
        // Take the last `ngram_size` characters from the buffer
        let start_idx = self.buffer.len() - self.ngram_size;
        let ngram_text: String = self.buffer
            .iter()
            .skip(start_idx)
            .take(self.ngram_size)
            .cloned()
            .collect();
        
        // Create the NGramEntry
        // Note: In a real implementation, you'd need to track actual text positions
        Some(NGramEntry {
            sequence_number: self.current_sequence,
            file_path: self.file_path.clone(),
            text_position: crate::types::TextPosition {
                start_byte: 0,  // Placeholder - would need actual position tracking
                end_byte: 0,    // Placeholder - would need actual position tracking
                line_number: 0, // Placeholder - would need actual position tracking
            },
            text_content: crate::types::NGramText {
                original_slice: ngram_text.clone(),
                cleaned_text: ngram_text,
            },
        })
    }
    
    /// Create a word-based ngram from the current buffer
    fn create_word_ngram(&self) -> Option<NGramEntry> {
        if self.buffer.len() < self.ngram_size {
            return None;
        }
        
        // Take the last `ngram_size` words from the buffer
        let start_idx = self.buffer.len() - self.ngram_size;
        let ngram_text: String = self.buffer
            .iter()
            .skip(start_idx)
            .take(self.ngram_size)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        
        // Create the NGramEntry
        // Note: In a real implementation, you'd need to track actual text positions
        Some(NGramEntry {
            sequence_number: self.current_sequence,
            file_path: self.file_path.clone(),
            text_position: crate::types::TextPosition {
                start_byte: 0,  // Placeholder - would need actual position tracking
                end_byte: 0,    // Placeholder - would need actual position tracking
                line_number: 0, // Placeholder - would need actual position tracking
            },
            text_content: crate::types::NGramText {
                original_slice: ngram_text.clone(),
                cleaned_text: ngram_text,
            },
        })
    }
    
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    
    /// Get the current sequence number
    pub fn get_sequence(&self) -> u64 {
        self.current_sequence
    }
}