// storage/lmdb/query.rs

use std::ops::Range;
use log::{debug, warn};
use lmdb_rkv::{Transaction, Cursor};

use crate::error::{Error, Result};
use crate::types::NGramEntry;

use super::LMDBStorage;
use super::prefix;

impl LMDBStorage {
    /// Get NGramEntries within a range of sequence numbers
    pub fn get_sequence_range(&self, range: Range<u64>) -> Result<Vec<NGramEntry>> {
        const BATCH_SIZE: usize = 1000;
        
        if range.start >= range.end {
            return Ok(Vec::new());
        }
        
        // Pre-allocate with a reasonable size
        let mut all_entries = Vec::with_capacity((range.end - range.start) as usize);
        
        // Process the range in manageable batches to avoid huge transactions
        for batch_start in (range.start..range.end).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE as u64).min(range.end);
            
            // Start a read transaction
            let txn = self.env.begin_ro_txn()
                .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
            
            {
                // Cursor scope - ensure cursor is dropped before transaction ends
                let mut cursor = txn.open_ro_cursor(self.main_db)
                    .map_err(|e| Error::Database(format!("Failed to create cursor: {}", e)))?;
                
                // Position cursor at first key in range
                let start_key = Self::create_ngram_key(batch_start);
                
                // Create a vector to hold entries from this batch
                let mut batch_entries = Vec::new();
                
                // FIXED: Handle Result from cursor iterator
                for result in cursor.iter_from(start_key) {
                    if let Ok((key, value)) = result {
                        if key.len() == 8 {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(key);
                            let seq = u64::from_be_bytes(bytes);
                            
                            // Check if we're still in range
                            if seq >= batch_end {
                                break;
                            }
                            
                            // Deserialize and store entry
                            match Self::deserialize_entry(value) {
                                Ok(entry) => {
                                    batch_entries.push(entry);
                                },
                                Err(e) => {
                                    warn!("Failed to deserialize entry at sequence {}: {}", seq, e);
                                }
                            }
                        }
                    } else if let Err(e) = result {
                        warn!("Error iterating cursor: {}", e);
                        // Continue despite errors to maximize data retrieval
                    }
                }
                
                // Add entries from this batch to overall result
                all_entries.extend(batch_entries);
            } // Cursor is dropped here
            
            // End transaction
            txn.abort();
        }
        
        // Update metrics
        self.metrics.record_bytes_read(all_entries.len() as u64 * 200); // Rough estimate
        self.metrics.increment_reads();
        
        Ok(all_entries)
    }

    /// Find NGramEntries that match a given prefix
    pub fn find_ngrams_by_prefix(&self, prefix: &[u8]) -> Result<Vec<NGramEntry>> {
        // First get all sequence numbers for this prefix 
        let sequences = self.get_sequences_by_prefix(prefix)?;
        
        // Pre-allocate with the exact capacity we need
        let mut ngrams = Vec::with_capacity(sequences.len());
        
        // Use batch fetching for optimal performance
        if !sequences.is_empty() {
            ngrams = self.get_ngrams_batch(&sequences)?;
        }
        
        // Update metrics
        self.metrics.increment_reads();
        
        Ok(ngrams)
    }

    /// Get sequence numbers for NGrams with a given prefix
    pub fn get_sequences_by_prefix_for_ngram(&self, ngram: &NGramEntry) -> Result<Vec<u64>> {
        let prefix = prefix::extract_two_char_prefix(ngram.text_content.cleaned_text.as_bytes());
        self.get_sequences_by_prefix(prefix)
    }

    /// Get the maximum sequence number currently in the database
    pub fn get_max_sequence_number(&self) -> Result<u64> {
        let txn = self.env.begin_ro_txn()
            .map_err(|e| Error::Database(format!("Failed to start read transaction: {}", e)))?;
        
        let mut max_seq = 0;
        
        {
            // Create cursor in scope
            let mut cursor = txn.open_ro_cursor(self.main_db)
                .map_err(|e| Error::Database(format!("Failed to create cursor: {}", e)))?;
            
            // FIXED: Handle Result from cursor iterator
            for result in cursor.iter() {
                if let Ok((key, _)) = result {
                    if key.len() == 8 {
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(key);
                        let seq = u64::from_be_bytes(bytes);
                        
                        // Update max_seq if this sequence is higher
                        if seq > max_seq {
                            max_seq = seq;
                        }
                    }
                }
            }
        } // Cursor is dropped here
        
        // End transaction
        txn.abort();
        
        Ok(max_seq)
    }

    /// Search for NGramEntries containing text similar to the given query
    pub fn search_text(&self, query: &str, limit: usize) -> Result<Vec<NGramEntry>> {
        // Get prefix from the query
        if query.is_empty() {
            return Ok(Vec::new());
        }
        
        // Extract prefix for lookup
        let query_bytes = query.as_bytes();
        let prefix = prefix::extract_two_char_prefix(query_bytes);
        
        // Get all NGrams with this prefix
        let mut candidates = self.find_ngrams_by_prefix(prefix)?;
        
        debug!("Found {} prefix candidates for query: {}", candidates.len(), query);
        
        // Filter and sort by relevance
        candidates.retain(|ngram| {
            // Simple case-insensitive contains check
            ngram.text_content.cleaned_text.to_lowercase().contains(&query.to_lowercase())
        });
        
        debug!("After filtering, {} candidates match query", candidates.len());
        
        // Sort by relevance (exact matches first, then by sequence number for stability)
        candidates.sort_by(|a, b| {
            let a_exact = a.text_content.cleaned_text.eq_ignore_ascii_case(query);
            let b_exact = b.text_content.cleaned_text.eq_ignore_ascii_case(query);
            
            if a_exact && !b_exact {
                std::cmp::Ordering::Less
            } else if !a_exact && b_exact {
                std::cmp::Ordering::Greater
            } else {
                // Secondary sort by sequence number (newest first)
                b.sequence_number.cmp(&a.sequence_number)
            }
        });
        
        // Limit results if needed
        if candidates.len() > limit {
            candidates.truncate(limit);
        }
        
        debug!("Returning {} results for query: {}", candidates.len(), query);
        Ok(candidates)
    }

    /// Get sequences by prefix using a batch approach
    pub fn get_sequences_by_prefix_batch(&self, prefixes: &[&[u8]]) -> Result<Vec<Vec<u64>>> {
        let mut results = Vec::with_capacity(prefixes.len());
        
        for &prefix in prefixes {
            results.push(self.get_sequences_by_prefix(prefix)?);
        }
        
        Ok(results)
    }
}