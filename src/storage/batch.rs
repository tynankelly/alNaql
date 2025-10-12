// storage/batch.rs - Complete file with SIMD optimizations

use log::{debug, trace, warn, error};
use lmdb_rkv::WriteFlags;
use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;

use crate::error::{Error, Result};
use crate::types::{NGramEntry, PositionData};
use crate::ngram::features::{generate_ngram_metadata, extract_prefix_bytes};
use crate::types::CompactCharFrequencies;
use super::LMDBStorage;

//==============================================================================
// CONSTANTS
//==============================================================================

const MAX_RETRY_ATTEMPTS: usize = 5;
const BASE_RETRY_DELAY_MS: u64 = 10;

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

fn should_retry_lmdb_error(error_msg: &str) -> bool {
    error_msg.contains("MDB_MAP_FULL") ||
    error_msg.contains("MDB_TXN_FULL") ||
    error_msg.contains("EAGAIN") ||
    error_msg.contains("deadlock")
}

fn apply_retry_backoff(attempt: usize) {
    let delay = Duration::from_millis(BASE_RETRY_DELAY_MS * (1 << attempt));
    std::thread::sleep(delay);
}

//==============================================================================
// BATCH STORAGE IMPLEMENTATION
//==============================================================================

impl LMDBStorage {
    /// Public batch storage method
    /// Receives NGramEntry array and processes in adaptive chunks
    pub fn store_ngrams_batch(&mut self, entries: &[NGramEntry]) -> Result<()> {
        if entries.is_empty() {
            debug!("Empty batch provided for storage");
            return Ok(());
        }
        
        trace!("Storing batch of {} NGram entries", entries.len());
        let start_time = Instant::now();
        
        // Get adaptive batch size
        let chunk_size = self.adaptive_batch_size.load(Ordering::Relaxed);
        let total_chunks = (entries.len() + chunk_size - 1) / chunk_size;
        
        // Process entries in optimally-sized chunks
        for (chunk_idx, chunk) in entries.chunks(chunk_size).enumerate() {
            // Progress logging
            if chunk_idx % 10 == 0 || chunk_idx == total_chunks - 1 {
                trace!("Processing chunk {}/{} ({} entries)", 
                    chunk_idx + 1, total_chunks, chunk.len());
            }
            
            // Store this chunk using column storage
            self.store_ngram_batch_internal(chunk)?;
        }
        
        let duration = start_time.elapsed();
        trace!("Successfully stored {} entries in {:?}", entries.len(), duration);
        
        // Update metrics
        self.metrics.increment_writes();
        self.metrics.record_bytes_written((entries.len() * 96) as u64); // Approximate
        
        Ok(())
    }
    
    /// Internal batch storage method - processes a single chunk
    /// No longer uses CompactFeatures or features_compact_db
    pub(crate) fn store_ngram_batch_internal(&mut self, chunk: &[NGramEntry]) -> Result<()> {
        let mut attempt = 0;
        let start_time = Instant::now();
        
        // DEBUG: Log initial batch info
        debug!("=== BATCH STORAGE START ===");
        debug!("Storing batch of {} NGram entries", chunk.len());
        if !chunk.is_empty() {
            debug!("First entry sequence: {}", chunk[0].sequence_number);
            debug!("Last entry sequence: {}", chunk[chunk.len()-1].sequence_number);
        }
        
        // PRE-CALCULATE everything that needs self BEFORE the write scope
        let bulk_loading = self.is_bulk_loading.load(Ordering::Relaxed);
        let prefix_index_depth = self.prefix_index_depth;
        
        debug!("Bulk loading mode: {}", bulk_loading);
        debug!("Prefix index depth: {}", prefix_index_depth);
        
        // Retry loop for transient LMDB errors
        while attempt < MAX_RETRY_ATTEMPTS {
            attempt += 1;
            debug!("Storage attempt {}/{}", attempt, MAX_RETRY_ATTEMPTS);
            
            // Pre-process all entries outside the transaction
            // This reduces time spent in the write transaction
            let preprocessed: Vec<_> = chunk.iter().enumerate().map(|(idx, entry)| {
                let sequence = entry.sequence_number;
                let key = Self::create_ngram_key(sequence);
                let text = &entry.text_content.cleaned_text;
                let text_bytes = text.as_bytes();
                
                // Debug logging for first few entries
                if idx < 3 {
                    debug!("Processing entry {}:", idx);
                    debug!("  Sequence: {}", sequence);
                    debug!("  Text length: {} chars, {} bytes", text.chars().count(), text_bytes.len());
                }
                
                // === EXTRACT PREFIX ===
                // Using the new function from features.rs
                let prefix_bytes = if !text.is_empty() {
                    extract_prefix_bytes(text, prefix_index_depth)
                } else {
                    Vec::new()
                };
                
                if idx < 3 && !prefix_bytes.is_empty() {
                    debug!("  Prefix extracted: {:?} ({} bytes)", 
                        String::from_utf8_lossy(&prefix_bytes), 
                        prefix_bytes.len());
                }
                
                // === PACK CHARACTER FREQUENCIES ===
                // Access char_frequencies directly from NGramEntry (no wrapper)
                let char_freq_bytes = Self::pack_char_frequencies(&entry.char_frequencies)?;
                
                if idx < 3 {
                    debug!("  Char frequencies serialized: {} bytes", char_freq_bytes.len());
                    debug!("    - total_chars: {}", entry.char_frequencies.total_chars);
                    debug!("    - unique_chars: {}", entry.char_frequencies.unique_chars);
                    debug!("    - magnitude_squared: {}", entry.char_frequencies.magnitude_squared);
                }
                
                // === PACK POSITION DATA ===
                let position_data = Self::pack_position_from_entry(entry);
                let position_bytes = Self::pack_position(&position_data)?;
                
                // === GENERATE METADATA ===
                // Using the new function from features.rs
                let metadata = generate_ngram_metadata(sequence, text);
                let metadata_bytes = bincode::serialize(&metadata)
                    .map_err(|e| Error::Serialization(format!("Failed to serialize metadata: {}", e)))?;
                
                if idx < 3 {
                    debug!("  Metadata generated:");
                    debug!("    - text_hash: {}", metadata.text_hash);
                    debug!("    - text_length: {}", metadata.text_length);
                }
                
                Ok((
                    idx,
                    sequence,
                    key,
                    text_bytes.to_vec(),
                    prefix_bytes,
                    char_freq_bytes,
                    position_bytes,
                    metadata_bytes,
                ))
            }).collect::<Result<Vec<_>>>()?;
            
            debug!("Pre-processing complete for {} entries", preprocessed.len());
            
            // Perform the actual storage in a transaction
            let operation_result = self.with_write_scope(|mut scope| {
                // Get database handles
                let sequences_db = scope.storage().sequences_db;
                let text_bytes_db = scope.storage().text_bytes_db;
                let char_frequencies_db = scope.storage().char_frequencies_db;
                let position_db = scope.storage().position_db;
                let prefix_bytes_db = scope.storage().prefix_bytes_db;
                let ngram_metadata_db = scope.storage().ngram_metadata_db;
                // NOTE: features_compact_db is removed - no longer used
                
                let flags = if bulk_loading {
                    WriteFlags::NO_OVERWRITE | WriteFlags::APPEND
                } else {
                    WriteFlags::empty()
                };
                
                // Track what we successfully store
                let mut stored_sequences = 0;
                let mut stored_frequencies = 0;
                let mut stored_metadata = 0;
                let mut stored_prefixes = 0;
                let mut storage_errors = Vec::new();
                
                // Store each entry's data in the appropriate columns
                for (_idx, sequence, key, text_bytes, prefix_bytes, char_freq_bytes, position_bytes, metadata_bytes) 
                    in &preprocessed 
                {
                    // Store sequence number (as a registry of valid sequences)
                    match scope.txn().put(sequences_db, &key, &key, flags) {
                        Ok(_) => stored_sequences += 1,
                        Err(e) => {
                            storage_errors.push(format!("Seq {}: {}", sequence, e));
                            continue; // Skip this entry if we can't store the sequence
                        }
                    }
                    
                    // Store text bytes
                    if let Err(e) = scope.txn().put(text_bytes_db, key, text_bytes, flags) {
                        storage_errors.push(format!("Text {}: {}", sequence, e));
                    }
                    
                    // Store character frequencies (the SIMD-optimized 96-byte structure)
                    match scope.txn().put(char_frequencies_db, key, char_freq_bytes, flags) {
                        Ok(_) => stored_frequencies += 1,
                        Err(e) => storage_errors.push(format!("CharFreq {}: {}", sequence, e)),
                    }
                    
                    // Store position data
                    if let Err(e) = scope.txn().put(position_db, key, position_bytes, flags) {
                        storage_errors.push(format!("Position {}: {}", sequence, e));
                    }
                    
                    // Store prefix (if not empty)
                    if !prefix_bytes.is_empty() {
                        match scope.txn().put(prefix_bytes_db, key, prefix_bytes, flags) {
                            Ok(_) => stored_prefixes += 1,
                            Err(e) => storage_errors.push(format!("Prefix {}: {}", sequence, e)),
                        }
                    }
                    
                    // Store metadata
                    match scope.txn().put(ngram_metadata_db, key, metadata_bytes, flags) {
                        Ok(_) => {
                            stored_metadata += 1;
                            // ADD THIS DEBUG CHECK
                            trace!("Successfully stored metadata for sequence {}", sequence);
                        },
                        Err(e) => {
                            storage_errors.push(format!("Metadata {}: {}", sequence, e));
                            // ADD THIS ERROR LOG
                            error!("CRITICAL: Failed to store metadata for sequence {}: {}", sequence, e);
                        }
                    }
                }
                
                // Log summary before commit
                debug!("=== PRE-COMMIT SUMMARY ===");
                debug!("Sequences stored: {}/{}", stored_sequences, preprocessed.len());
                debug!("Frequencies stored: {}/{}", stored_frequencies, preprocessed.len());
                debug!("Metadata stored: {}/{}", stored_metadata, preprocessed.len());
                debug!("Prefixes stored: {}/{}", stored_prefixes, preprocessed.len());

                // ADD THE DEBUG CODE HERE
                if stored_metadata == 0 && !preprocessed.is_empty() {
                    error!("CRITICAL WARNING: No metadata was stored for {} entries!", preprocessed.len());
                }
                
                if !storage_errors.is_empty() {
                    warn!("Storage errors encountered: {} errors", storage_errors.len());
                    for error in storage_errors.iter().take(5) {
                        warn!("  - {}", error);
                    }
                }
                
                // Commit the transaction
                scope.commit()
            });
            
            // Handle the result of this storage attempt
            match operation_result {
                Ok(_) => {
                    let duration = start_time.elapsed();
                    debug!("=== BATCH STORAGE SUCCESS ===");
                    debug!("Successfully stored {} entries in {:?} (attempt {})", 
                        chunk.len(), duration, attempt);
                    
                    return Ok(());
                },
                Err(e) => {
                    debug!("Storage attempt {} failed: {}", attempt, e);
                    
                    // Check if we should retry
                    if should_retry_lmdb_error(&e.to_string()) && attempt < MAX_RETRY_ATTEMPTS {
                        warn!("Retryable error encountered, will retry after backoff...");
                        apply_retry_backoff(attempt);
                        continue;
                    }
                    
                    // Non-retryable error or max attempts reached
                    debug!("=== BATCH STORAGE FAILED ===");
                    return Err(e);
                }
            }
        }
        
        // If we get here, we've exhausted all retry attempts
        Err(Error::Database(format!("Failed to store batch after {} attempts", MAX_RETRY_ATTEMPTS)))
    }
    
    /// Helper function to pack position data from NGramEntry
    fn pack_position_from_entry(entry: &NGramEntry) -> PositionData {
        PositionData {
            file_path: entry.file_path.clone(),
            start_byte: entry.text_position.start_byte,
            end_byte: entry.text_position.end_byte,
            line_number: entry.text_position.line_number as u32,
        }
    }
}

//==============================================================================
// COLUMN PACKING HELPERS - UPDATED FOR SIMD
//==============================================================================

impl LMDBStorage {
    /// Pack character frequencies for column storage - SIMD optimized version
    /// 
    /// Since our new CompactCharFrequencies is exactly 96 bytes with fixed layout,
    /// we can serialize it more efficiently than using bincode.
    pub fn pack_char_frequencies(freq: &crate::types::CompactCharFrequencies) -> Result<Vec<u8>> {
        // Direct byte serialization (most efficient for SIMD)
        // This is faster and produces exactly 96 bytes every time
        let mut bytes = Vec::with_capacity(96);
        
        // Write the counts array (64 bytes)
        bytes.extend_from_slice(&freq.counts);
        
        // Write metadata in little-endian format for consistency
        bytes.extend_from_slice(&freq.total_chars.to_le_bytes());     // 4 bytes
        bytes.extend_from_slice(&freq.unique_chars.to_le_bytes());     // 2 bytes  
        bytes.extend_from_slice(&freq.magnitude_squared.to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&freq.char_bitmap.to_le_bytes());       // 8 bytes
        bytes.extend_from_slice(&freq.padding);                         // 22 bytes
        
        debug_assert_eq!(bytes.len(), 96, "Serialized CompactCharFrequencies must be exactly 96 bytes");
        
        Ok(bytes)
    }
    
    /// Unpack character frequencies from column storage - SIMD optimized version
    /// 
    /// Expects exactly 96 bytes for the new fixed-size structure
    pub fn unpack_char_frequencies(bytes: &[u8]) -> Result<CompactCharFrequencies> {
        if bytes.len() != 96 {
            return Err(Error::Serialization(
                format!("Expected 96 bytes, got {}", bytes.len())
            ));
        }
        
        let mut freqs = CompactCharFrequencies::new();
        let mut offset = 0;
        
        // Read the counts array (64 bytes)
        freqs.counts.copy_from_slice(&bytes[offset..offset+64]);
        offset += 64;
        
        // Read metadata (keeping existing error handling)
        freqs.total_chars = u32::from_le_bytes(
            bytes[offset..offset+4].try_into()
                .map_err(|_| Error::Serialization("Invalid total_chars".to_string()))?
        );
        offset += 4;
        
        freqs.unique_chars = u16::from_le_bytes(
            bytes[offset..offset+2].try_into()
                .map_err(|_| Error::Serialization("Invalid unique_chars".to_string()))?
        );
        offset += 2;
        
        freqs.magnitude_squared = f32::from_le_bytes(
            bytes[offset..offset+4].try_into()
                .map_err(|_| Error::Serialization("Invalid magnitude_squared".to_string()))?
        );
        offset += 4;
        
        // Read bitmap with same error handling pattern
        freqs.char_bitmap = u64::from_le_bytes(
            bytes[offset..offset+8].try_into()
                .map_err(|_| Error::Serialization("Invalid char_bitmap".to_string()))?
        );
        // offset += 8; // Not needed since we don't read padding
        
        // The padding is already initialized to zeros in new()
        
        Ok(freqs)
    }
        
    /// Pack position data for column storage
    /// (This remains unchanged as PositionData is still variable-length)
    pub fn pack_position(pos: &PositionData) -> Result<Vec<u8>> {
        bincode::serialize(pos)
            .map_err(|e| Error::Serialization(format!("Failed to pack position data: {}", e)))
    }
    
    /// Unpack position data from column storage  
    /// (This remains unchanged as PositionData is still variable-length)
    pub fn unpack_position(bytes: &[u8]) -> Result<PositionData> {
        bincode::deserialize(bytes)
            .map_err(|e| Error::Serialization(format!("Failed to unpack position data: {}", e)))
    }
    
    /// Helper method to verify that a CompactCharFrequencies is properly aligned in memory
    /// This is useful for debugging SIMD alignment issues
    #[cfg(debug_assertions)]
    pub fn verify_frequency_alignment(freq: &crate::types::CompactCharFrequencies) -> bool {
        let ptr = freq as *const _ as usize;
        let aligned = ptr % 32 == 0;
        if !aligned {
            log::warn!("CompactCharFrequencies at address {:x} is not 32-byte aligned!", ptr);
        }
        aligned
    }
}

//==============================================================================
// BATCH SERIALIZATION HELPERS
//==============================================================================

impl LMDBStorage {
    /// Efficiently serialize multiple character frequencies in batch
    /// Returns a vector of serialized byte arrays
    pub fn pack_char_frequencies_batch(
        frequencies: &[crate::types::CompactCharFrequencies]
    ) -> Result<Vec<Vec<u8>>> {
        frequencies.iter()
            .map(|freq| Self::pack_char_frequencies(freq))
            .collect()
    }
    
    /// Efficiently deserialize multiple character frequencies in batch
    pub fn unpack_char_frequencies_batch(
        byte_arrays: &[Vec<u8>]
    ) -> Result<Vec<crate::types::CompactCharFrequencies>> {
        byte_arrays.iter()
            .map(|bytes| Self::unpack_char_frequencies(bytes))
            .collect()
    }
}