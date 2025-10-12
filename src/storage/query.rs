// storage/lmdb/query.rs

use std::ops::Range;
use log::{debug, warn, trace, info};
use lmdb_rkv::{Transaction, Cursor};
use ahash::AHashMap;
use crate::error::{Error, Result};
use crate::types::PositionData;
use super::LMDBStorage;
use crate::storage::StorageStats;

impl LMDBStorage {
    
    /// Get all sequence numbers in the database
    pub fn get_all_sequences(&self) -> Result<Vec<u64>> {
        self.with_read_scope(|scope| {
            debug!("Loading all sequence numbers");
            
            let mut cursor = scope.txn().open_ro_cursor(self.sequences_db)
                .map_err(|e| Error::Database(format!("Failed to create cursor: {}", e)))?;
            
            let mut sequences = Vec::new();
            
            // Iterate through all entries in sequences_db
            for result in cursor.iter_start() {
                if let Ok((key_bytes, _)) = result {
                    if key_bytes.len() == 8 {
                        let sequence = u64::from_be_bytes([
                            key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3],
                            key_bytes[4], key_bytes[5], key_bytes[6], key_bytes[7],
                        ]);
                        sequences.push(sequence);
                    }
                }
            }
            
            debug!("Found {} sequences in database", sequences.len());
            Ok(sequences)
        })
    }
    
    
    /// Get database information - USES SCOPED TRANSACTIONS WITH PROPER LMDB STATS
    pub fn get_db_info(&self) -> Result<StorageStats> {
        self.with_read_scope(|scope| {
            debug!("Retrieving database statistics using LMDB native stats");
            
            // Get database-specific statistics (not environment-wide)
            let db_stat = scope.txn().stat(self.sequences_db)
                .map_err(|e| Error::Database(format!("Failed to get database statistics: {}", e)))?;
            
            // LMDB provides exact counts - no manual iteration needed!
            let total_ngrams = db_stat.entries() as u64;
            
            // Calculate approximate total bytes using LMDB page information
            // This is more accurate than manual counting for large databases
            let page_size = db_stat.page_size() as u64;  // FIXED: page_size() not psize()
            let leaf_pages = db_stat.leaf_pages() as u64;
            let overflow_pages = db_stat.overflow_pages() as u64;
            
            // Estimate total bytes: (leaf pages + overflow pages) * page size
            // This gives us storage usage rather than raw data size
            let total_bytes = (leaf_pages + overflow_pages) * page_size;
            
            debug!("Database stats: {} entries, {} bytes ({} leaf pages, {} overflow pages, {} bytes/page)",
                total_ngrams, total_bytes, leaf_pages, overflow_pages, page_size);
            
            // For very large databases, we can log additional diagnostics
            if total_ngrams > 1_000_000 {
                let branch_pages = db_stat.branch_pages() as u64;
                let depth = db_stat.depth() as u64;
                info!("Large database detected: depth={}, branch_pages={}, storage_efficiency={:.2} entries/page",
                    depth, branch_pages, total_ngrams as f64 / (leaf_pages + branch_pages) as f64);
            }
            
            Ok(StorageStats {
                total_ngrams,
                total_bytes,
                unique_sequences: total_ngrams, // For LMDB, sequence numbers are unique keys
                unique_cleaned_texts: 0, // Would need separate analysis - not available from LMDB stats
                compression_ratio: 1.0, // LMDB doesn't compress data like some other databases
            })
        })
    }
    
    /// Check if a specific sequence exists - USES SCOPED TRANSACTIONS
    pub fn sequence_exists(&self, sequence: u64) -> Result<bool> {
        self.with_read_scope(|scope| {
            let key = Self::create_ngram_key(sequence);
            trace!("Checking existence of sequence {}", sequence);
            
            // Use ReadScope convenience method
            let exists = scope.exists(self.sequences_db, &key)?;
            
            trace!("Sequence {} exists: {}", sequence, exists);
            Ok(exists)
        })
    }
    
    /// Get the highest sequence number in the database - USES SCOPED TRANSACTIONS
    pub fn get_max_sequence(&self) -> Result<Option<u64>> {
        self.with_read_scope(|scope| {
            debug!("Finding maximum sequence number");
            
            // Find last key efficiently by iterating to the end
            let max_sequence = {
                let mut cursor = scope.txn().open_ro_cursor(self.sequences_db)
                    .map_err(|e| Error::Database(format!("Failed to create cursor: {}", e)))?;
                
                let mut last_sequence = None;
                
                // Iterate through all entries to find the last one
                for result in cursor.iter_start() {
                    if let Ok((key_bytes, _)) = result {
                        if key_bytes.len() == 8 {
                            let sequence = u64::from_be_bytes([
                                key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3],
                                key_bytes[4], key_bytes[5], key_bytes[6], key_bytes[7],
                            ]);
                            last_sequence = Some(sequence);
                        }
                    }
                }
                
                last_sequence
            };
            
            debug!("Maximum sequence number: {:?}", max_sequence);
            Ok(max_sequence)
        })
    }
    
    /// Count entries in a sequence range efficiently - USES SCOPED TRANSACTIONS
    pub fn count_entries_in_range(&self, range: Range<u64>) -> Result<u64> {
        if range.start >= range.end {
            return Ok(0);
        }
        
        self.with_read_scope(|scope| {
            debug!("Counting entries in range {}..{}", range.start, range.end);
            
            // Count efficiently with cursor
            let count = {
                let mut cursor = scope.txn().open_ro_cursor(self.sequences_db)
                    .map_err(|e| Error::Database(format!("Failed to create cursor: {}", e)))?;
                
                let mut entry_count = 0u64;
                
                // Iterate through all entries and count those in range
                for result in cursor.iter_start() {
                    if let Ok((key_bytes, _)) = result {
                        if key_bytes.len() == 8 {
                            let sequence = u64::from_be_bytes([
                                key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3],
                                key_bytes[4], key_bytes[5], key_bytes[6], key_bytes[7],
                            ]);
                            
                            if sequence >= range.start && sequence < range.end {
                                entry_count += 1;
                            }
                            
                            // Early termination if we've passed the range
                            if sequence >= range.end {
                                break;
                            }
                        }
                    }
                }
                
                entry_count
            };
            
            debug!("Found {} entries in range {}..{}", count, range.start, range.end);
            Ok(count)
        })
    }

    //==============================================================================
    // UTILITY METHODS - Available for ReadScope implementations in scopes.rs
    //==============================================================================
    
    /// Create a sequence key from a u64 sequence number
    pub(super) fn create_ngram_key(sequence: u64) -> [u8; 8] {
        sequence.to_be_bytes()
    }

//==============================================================================
// COLUMN READING METHODS
//==============================================================================

    /// Load only text bytes for a batch of sequences
    /// Returns HashMap for O(1) lookups during matching
    pub fn get_text_bytes_batch(&self, sequences: &[u64]) -> Result<AHashMap<u64, Vec<u8>>> {
        if sequences.is_empty() {
            return Ok(AHashMap::new());
        }
        
        trace!("Loading text bytes for {} sequences", sequences.len());
        let mut results = AHashMap::with_capacity(sequences.len());
        
        self.with_read_scope(|scope| {
            let text_bytes_db = scope.storage().text_bytes_db;
            
            for &sequence in sequences {
                let key = Self::create_ngram_key(sequence);
                
                match scope.txn().get(text_bytes_db, &key) {
                    Ok(bytes) => {
                        // Store as owned Vec<u8>
                        results.insert(sequence, bytes.to_vec());
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        trace!("Text bytes not found for sequence {}", sequence);
                    },
                    Err(e) => {
                        warn!("Error reading text bytes for sequence {}: {}", sequence, e);
                    }
                }
            }
            
            Ok(())
        })?;
        
        self.metrics.increment_reads();
        trace!("Loaded text bytes for {}/{} sequences", results.len(), sequences.len());
        Ok(results)
    }
    
    
    /// Load character frequencies for a batch of sequences
    /// Used for detailed similarity calculations
    pub fn get_char_frequencies_batch(&self, sequences: &[u64]) -> Result<AHashMap<u64, crate::types::CompactCharFrequencies>> {
        if sequences.is_empty() {
            return Ok(AHashMap::new());
        }
        
        trace!("Loading character frequencies for {} sequences", sequences.len());
        let mut results = AHashMap::with_capacity(sequences.len());
        
        self.with_read_scope(|scope| {
            let char_frequencies_db = scope.storage().char_frequencies_db;
            
            for &sequence in sequences {
                let key = Self::create_ngram_key(sequence);
                
                match scope.txn().get(char_frequencies_db, &key) {
                    Ok(bytes) => {
                        // Deserialize character frequencies
                        match Self::unpack_char_frequencies(bytes) {
                            Ok(frequencies) => {
                                results.insert(sequence, frequencies);
                            },
                            Err(e) => {
                                warn!("Failed to unpack char frequencies for sequence {}: {}", sequence, e);
                            }
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        trace!("Character frequencies not found for sequence {}", sequence);
                    },
                    Err(e) => {
                        warn!("Error reading char frequencies for sequence {}: {}", sequence, e);
                    }
                }
            }
            
            Ok(())
        })?;
        
        self.metrics.increment_reads();
        trace!("Loaded char frequencies for {}/{} sequences", results.len(), sequences.len());
        Ok(results)
    }
    
    /// Load position data for a batch of sequences
    /// Used for output generation and XML mapping
    pub fn get_position_batch(&self, sequences: &[u64]) -> Result<AHashMap<u64, PositionData>> {
        if sequences.is_empty() {
            return Ok(AHashMap::new());
        }
        
        trace!("Loading position data for {} sequences", sequences.len());
        let mut results = AHashMap::with_capacity(sequences.len());
        
        self.with_read_scope(|scope| {
            let position_db = scope.storage().position_db;
            
            for &sequence in sequences {
                let key = Self::create_ngram_key(sequence);
                
                match scope.txn().get(position_db, &key) {
    Ok(bytes) => {
        match Self::unpack_position(bytes) {
            Ok(position) => {
                // ADD THIS DEBUG (only for sequences that seem problematic):
                if position.end_byte - position.start_byte > 100_000 {
                    info!("SUSPICIOUS_POSITION: Sequence {} has huge span: {}..{} ({} bytes)",
                        sequence, position.start_byte, position.end_byte,
                        position.end_byte - position.start_byte);
                }
                results.insert(sequence, position);
            },
                            Err(e) => {
                                warn!("Failed to unpack position for sequence {}: {}", sequence, e);
                            }
                        }
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        trace!("Position data not found for sequence {}", sequence);
                    },
                    Err(e) => {
                        warn!("Error reading position for sequence {}: {}", sequence, e);
                    }
                }
            }
            
            Ok(())
        })?;
        
        self.metrics.increment_reads();
        trace!("Loaded position data for {}/{} sequences", results.len(), sequences.len());
        Ok(results)
    }
    
    /// Check which sequences exist in the database
    /// Useful for validation and filtering
    pub fn check_sequences_exist(&self, sequences: &[u64]) -> Result<Vec<u64>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        
        trace!("Checking existence of {} sequences", sequences.len());
        let mut existing = Vec::new();
        
        self.with_read_scope(|scope| {
            let sequences_db = scope.storage().sequences_db;
            
            for &sequence in sequences {
                let key = Self::create_ngram_key(sequence);
                
                match scope.txn().get(sequences_db, &key) {
                    Ok(_) => {
                        existing.push(sequence);
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        // Not found, skip
                    },
                    Err(e) => {
                        warn!("Error checking sequence {}: {}", sequence, e);
                    }
                }
            }
            
            Ok(())
        })?;
        
        trace!("Found {}/{} sequences exist", existing.len(), sequences.len());
        Ok(existing)
    }

}