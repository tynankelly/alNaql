// storage/prefix.rs - Simplified prefix indexing without partitions

use ahash::AHashMap;
use log::{info, debug, trace, warn};
use lmdb_rkv::{WriteFlags, Transaction, Cursor};

use crate::error::{Error, Result};
use crate::types::NGramEntry;
use super::LMDBStorage;

impl LMDBStorage {
    /// Collect prefix data from a batch of NGram entries
    /// Returns a simple map of prefix_bytes → sequence_numbers
    pub fn collect_prefix_data(&self, entries: &[NGramEntry]) -> Result<AHashMap<Vec<u8>, Vec<u64>>> {
        let mut prefix_map: AHashMap<Vec<u8>, Vec<u64>> = AHashMap::new();
        
        for entry in entries {
            let text = &entry.text_content.cleaned_text;
            
            // Extract prefix as string (respects Unicode boundaries)
            let prefix_str: String = text.chars()
                .take(self.prefix_index_depth)
                .collect();
            
            // Skip if text is too short for meaningful prefix
            if prefix_str.is_empty() {
                trace!("Skipping empty prefix for sequence {}", entry.sequence_number);
                continue;
            }
            
            // Convert to bytes for storage
            let prefix_bytes = prefix_str.into_bytes();
            
            // Add this sequence to the prefix's list
            prefix_map
                .entry(prefix_bytes)
                .or_insert_with(Vec::new)
                .push(entry.sequence_number);
        }
        
        debug!("Collected {} unique prefixes from {} entries", 
               prefix_map.len(), entries.len());
        
        Ok(prefix_map)
    }

    /// Store prefix data into the prefix index database
    /// Takes a map of prefix_bytes → sequence_numbers and stores it
    pub fn store_prefix_data(&mut self, prefix_data: AHashMap<Vec<u8>, Vec<u64>>) -> Result<()> {
        if prefix_data.is_empty() {
            debug!("No prefix data to store");
            return Ok(());
        }
        
        let start_time = std::time::Instant::now();
        info!("Storing {} unique prefixes", prefix_data.len());
        
        self.with_write_scope(|mut scope| {
            let prefix_index_db = scope.storage().prefix_index_db;
            let mut stored_count = 0;
            
            for (prefix_bytes, mut sequences) in prefix_data {
                // Deduplicate and sort sequences for this prefix
                sequences.sort_unstable();
                sequences.dedup();
                
                // Serialize the sequence list
                let packed_sequences = bincode::serialize(&sequences)
                    .map_err(|e| Error::Serialization(
                        format!("Failed to serialize sequences: {}", e)
                    ))?;
                
                // Store in the database
                scope.txn().put(
                    prefix_index_db,
                    &prefix_bytes,
                    &packed_sequences,
                    WriteFlags::empty()
                ).map_err(|e| Error::Database(
                    format!("Failed to store prefix: {}", e)
                ))?;
                
                stored_count += 1;
                
                if stored_count % 1000 == 0 {
                    trace!("Stored {} prefixes so far", stored_count);
                }
            }
            
            scope.commit()?;
            
            info!("Successfully stored {} prefixes in {:?}", 
                  stored_count, start_time.elapsed());
            Ok(())
        })
    }


    /// Store prefix bytes for a single sequence in the prefix_bytes column
    pub fn store_prefix_for_sequence(&mut self, sequence: u64, prefix_bytes: &[u8]) -> Result<()> {
        self.with_write_scope(|mut scope| {
            let prefix_bytes_db = scope.storage().prefix_bytes_db;
            let key = Self::create_ngram_key(sequence);
            
            scope.txn().put(
                prefix_bytes_db,
                &key,
                &prefix_bytes,  // ADD THE & HERE - pass reference to the slice
                WriteFlags::empty()
            ).map_err(|e| Error::Database(
                format!("Failed to store prefix for sequence {}: {}", sequence, e)
            ))?;
            
            scope.commit()
        })
    }
    
    /// Batch store prefixes for multiple sequences
    pub fn store_prefixes_batch(&mut self, sequence_prefix_pairs: &[(u64, Vec<u8>)]) -> Result<()> {
        self.with_write_scope(|mut scope| {
            let prefix_bytes_db = scope.storage().prefix_bytes_db;
            
            for (sequence, prefix_bytes) in sequence_prefix_pairs {
                let key = Self::create_ngram_key(*sequence);
                
                scope.txn().put(
                    prefix_bytes_db,
                    &key,
                    &prefix_bytes,  // ADD THE & HERE TOO
                    WriteFlags::empty()
                ).map_err(|e| Error::Database(
                    format!("Failed to store prefix for sequence {}: {}", sequence, e)
                ))?;
            }
            
            scope.commit()
        })
    }

    /// Retrieve the stored prefix bytes for a given sequence
    /// This eliminates prefix extraction during matching
    pub fn get_prefix_bytes(&self, sequence: u64) -> Result<Vec<u8>> {
        self.with_read_scope(|scope| {
            let prefix_bytes_db = scope.storage().prefix_bytes_db;
            let key = Self::create_ngram_key(sequence);
            
            match scope.txn().get(prefix_bytes_db, &key) {
                Ok(bytes) => Ok(bytes.to_vec()),
                Err(lmdb_rkv::Error::NotFound) => {
                    Err(Error::Database(format!("No prefix found for sequence {}", sequence)))
                },
                Err(e) => {
                    Err(Error::Database(format!("Failed to retrieve prefix for sequence {}: {}", sequence, e)))
                }
            }
        })
    }
    
    /// Query sequences by prefix bytes from the inverted index
    /// This is the core matching operation
    pub fn get_sequences_by_prefix_bytes(&self, prefix_bytes: &[u8]) -> Result<Vec<u64>> {
        self.with_read_scope(|scope| {
            let prefix_index_db = scope.storage().prefix_index_db;
            
            match scope.txn().get(prefix_index_db, &prefix_bytes) {
                Ok(data) => {
                    let sequences: Vec<u64> = bincode::deserialize(data)
                        .map_err(|e| Error::Serialization(
                            format!("Failed to deserialize sequences: {}", e)
                        ))?;
                    Ok(sequences)
                },
                Err(lmdb_rkv::Error::NotFound) => {
                    trace!("No sequences found for prefix");
                    Ok(Vec::new())
                },
                Err(e) => {
                    Err(Error::Database(format!("Failed to query prefix index: {}", e)))
                }
            }
        })
    }
    
    /// Convenience method that extracts prefix from NGramEntry and queries
    /// Maintains backward compatibility
    pub fn get_sequences_by_prefix_for_ngram(&self, ngram: &NGramEntry) -> Result<Vec<u64>> {
        let text = &ngram.text_content.cleaned_text;
        
        // Extract prefix (same as in collect_prefix_data)
        let prefix_str: String = text.chars()
            .take(self.prefix_index_depth)
            .collect();
        
        if prefix_str.is_empty() {
            return Ok(Vec::new());
        }
        
        let prefix_bytes = prefix_str.into_bytes();
        self.get_sequences_by_prefix_bytes(&prefix_bytes)
    }
    
    /// Batch retrieve prefixes for multiple sequences
    /// Returns HashMap for efficient lookups during matching
    pub fn get_prefix_bytes_batch(&self, sequences: &[u64]) -> Result<AHashMap<u64, Vec<u8>>> {
        if sequences.is_empty() {
            return Ok(AHashMap::new());
        }
        
        trace!("Loading prefixes for {} sequences", sequences.len());
        let mut results = AHashMap::with_capacity(sequences.len());
        
        self.with_read_scope(|scope| {
            let prefix_bytes_db = scope.storage().prefix_bytes_db;
            
            for &sequence in sequences {
                let key = Self::create_ngram_key(sequence);
                
                match scope.txn().get(prefix_bytes_db, &key) {
                    Ok(bytes) => {
                        results.insert(sequence, bytes.to_vec());
                    },
                    Err(lmdb_rkv::Error::NotFound) => {
                        trace!("No prefix found for sequence {}", sequence);
                    },
                    Err(e) => {
                        warn!("Error reading prefix for sequence {}: {}", sequence, e);
                    }
                }
            }
            
            Ok(())
        })?;
        
        trace!("Loaded prefixes for {}/{} sequences", results.len(), sequences.len());
        Ok(results)
    }

    /// Verify prefix storage integrity
    /// Ensures consistency between prefix_bytes_db and prefix_index_db
    pub fn verify_prefix_storage(&self) -> Result<()> {
        info!("Verifying prefix storage integrity");
        
        self.with_read_scope(|scope| {
            let prefix_bytes_db = scope.storage().prefix_bytes_db;
            let prefix_index_db = scope.storage().prefix_index_db;
            
            // Count sequences in prefix_bytes_db
            let mut sequence_count = 0;
            let mut cursor = scope.txn().open_ro_cursor(prefix_bytes_db)
                .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
            for _ in cursor.iter() {
                sequence_count += 1;
            }
            
            // Count sequences in prefix_index_db (inverted index)
            let mut total_indexed = 0;
            let mut unique_prefixes = 0;
            let mut cursor = scope.txn().open_ro_cursor(prefix_index_db)
                .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
            
            for item in cursor.iter() {
                let (_, value) = item
                    .map_err(|e| Error::Database(format!("Cursor iteration error: {}", e)))?;
                unique_prefixes += 1;
                
                let sequences: Vec<u64> = bincode::deserialize(value)
                    .map_err(|e| Error::Serialization(
                        format!("Failed to deserialize sequences during verification: {}", e)
                    ))?;
                total_indexed += sequences.len();
            }
            
            info!("Prefix storage verification:");
            info!("  Sequences with stored prefixes: {}", sequence_count);
            info!("  Unique prefixes in index: {}", unique_prefixes);
            info!("  Total sequence references in index: {}", total_indexed);
            
            // Basic consistency check
            if sequence_count != total_indexed {
                warn!("Mismatch: {} sequences have prefixes but {} are indexed", 
                      sequence_count, total_indexed);
            }
            
            Ok(())
        })
    }
}
