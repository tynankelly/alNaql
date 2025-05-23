// storage/lmdb/prefix.rs

use log::{debug, info, trace, warn};
use std::collections::{HashMap, HashSet};
use lazy_static::lazy_static;
use std::time::Instant;
use lmdb_rkv::{Cursor, Transaction, Error as LmdbError, WriteFlags};
use crate::utils::simd_prefix::extract_first_two_chars;
use crate::utils::simd_deserialize::deserialize_u64_vec;
use crate::error::{Error, Result};
use crate::types::NGramEntry;
use super::LMDBStorage;

// Prefix-related constants
const MAX_RETRY_ATTEMPTS: usize = 3;
const RETRY_DELAY_MS: u64 = 10;

// Define prefix partitions similar to RocksDB implementation
pub struct PrefixPartition {
    pub db_name: String,
    pub characters: Vec<char>,
}

// Lazy static definitions for prefix partitioning
lazy_static! {
    // Map of first characters to their prefix partition details
    pub static ref PREFIX_PARTITIONS: Vec<PrefixPartition> = {
        let mut partitions = Vec::new();
        
        // Add space/whitespace partition first since it's a common case
        partitions.push(PrefixPartition {
            db_name: "prefix_space".to_string(),
            characters: vec![' '],
        });

        // Add Arabic letter partitions
        // Note: These are organized by Arabic alphabet order for easier maintenance
        let arabic_groups = [
        // Base letter groups with their normalized variants
        (vec!['ا', 'أ', 'إ', 'آ'], "prefix_alef"),
        (vec!['ب'], "prefix_ba"),
        (vec!['ت'], "prefix_ta"),
        (vec!['ث'], "prefix_tha"),
        (vec!['ج'], "prefix_jim"),
        (vec!['ح'], "prefix_ha"),
        (vec!['خ'], "prefix_kha"),
        (vec!['د'], "prefix_dal"),
        (vec!['ذ'], "prefix_thal"),
        (vec!['ر'], "prefix_ra"),
        (vec!['ز'], "prefix_zain"),
        (vec!['س'], "prefix_sin"),
        (vec!['ش'], "prefix_shin"),
        (vec!['ص'], "prefix_sad"),
        (vec!['ض'], "prefix_dad"),
        (vec!['ط'], "prefix_ta_marbuta"),
        (vec!['ظ'], "prefix_dha"),
        (vec!['ع'], "prefix_ain"),
        (vec!['غ'], "prefix_ghain"),
        (vec!['ف'], "prefix_fa"),
        (vec!['ق'], "prefix_qaf"),
        (vec!['ك'], "prefix_kaf"),
        (vec!['ل'], "prefix_lam"),
        (vec!['م'], "prefix_mim"),
        (vec!['ن'], "prefix_nun"),
        (vec!['ه', 'ة'], "prefix_ha_final"),
        (vec!['و', 'ؤ'], "prefix_waw"),
        (vec!['ي', 'ى', 'ئ'], "prefix_ya"),
        (vec!['ء'], "prefix_hamza"),
        ];

        // Add each Arabic letter group to our partitions
        for (chars, name) in arabic_groups.iter() {
            partitions.push(PrefixPartition {
                db_name: name.to_string(),
                characters: chars.clone(),
            });
        }

        partitions
    };

    // Character to database name mapping for quick prefix lookups
    pub static ref CHAR_TO_DB: HashMap<char, String> = {
        let mut map = HashMap::new();
        for partition in PREFIX_PARTITIONS.iter() {
            for &c in partition.characters.iter() {
                map.insert(c, partition.db_name.clone());
            }
        }
        map
    };
}

// In prefix.rs
pub fn extract_two_char_prefix(text: &[u8]) -> &[u8] {
    // Use SIMD internally but keep the same return type
    let (first, second) = extract_first_two_chars(text);
    
    // If no characters, return empty slice
    if first.is_none() {
        return &[];
    }
    
    // Calculate prefix length based on characters found
    let first_char = first.unwrap();
    let first_len = first_char.len_utf8();
    
    if second.is_none() {
        // Only first character available
        return &text[0..first_len.min(text.len())];
    }
    
    // Both characters available
    let second_char = second.unwrap();
    let second_len = second_char.len_utf8();
    let total_len = (first_len + second_len).min(text.len());
    
    &text[0..total_len]
}

impl LMDBStorage {
    /// Collect prefix data from a batch of NGram entries
    pub fn collect_prefix_data(&self, chunk: &[NGramEntry]) -> Result<HashMap<String, HashMap<char, Vec<u64>>>> {
        // Create a map from DB name to sequence numbers
        let mut partition_groups: HashMap<String, HashMap<char, Vec<u64>>> = HashMap::new();      
        let mut processed_count = 0;
        let mut skipped_count = 0;
        
        // First pass: organize by partition and second character
        for entry in chunk {
            // Extract first and second characters
            let text = &entry.text_content.cleaned_text;
            let mut chars = text.chars();
            
            if let Some(first_char) = chars.next() {
                // Look up which DB this character belongs to
                if let Some(db_name) = CHAR_TO_DB.get(&first_char) {
                    // Get second character (or space if there isn't one)
                    let second_char = chars.next().unwrap_or(' ');
                    
                    // Add sequence to the appropriate partition and second-char group
                    partition_groups
                        .entry(db_name.clone())
                        .or_insert_with(HashMap::new)
                        .entry(second_char)
                        .or_insert_with(Vec::new)
                        .push(entry.sequence_number);
                    
                    processed_count += 1;
                } else {
                    warn!("No partition found for character: {} in ngram: {} (seq: {})", 
                          first_char, text, entry.sequence_number);
                    skipped_count += 1;
                }
            } else {
                warn!("Empty text in ngram: {} (seq: {})", 
                      text, entry.sequence_number);
                skipped_count += 1;
            }
        }
        
        // Second pass: deduplicate sequences in each group
        let mut total_duplicates = 0;
        for (_, char_map) in partition_groups.iter_mut() {
            for (_, sequences) in char_map.iter_mut() {
                let original_count = sequences.len();
                sequences.sort_unstable();
                sequences.dedup();
                let new_count = sequences.len();
                total_duplicates += original_count - new_count;
            }
        }
        
        // Log statistics
        let total_sequences: usize = partition_groups
            .values()
            .flat_map(|char_map| char_map.values())
            .map(|seqs| seqs.len())
            .sum();
        
        info!("Prefix collection complete:");
        info!("  Ngrams processed: {}/{}", processed_count, chunk.len());
        info!("  Ngrams skipped: {}", skipped_count);
        info!("  Total sequences collected: {}", total_sequences);
        if total_duplicates > 0 {
            info!("  Duplicates removed during collection: {}", total_duplicates);
        }
        info!("  Partition groups: {}", partition_groups.len());
        
        Ok(partition_groups)
    }

    /// Stores prefix data with sequence merging for existing entries
    pub fn store_prefix_data(&mut self, partitions: HashMap<String, HashMap<char, Vec<u64>>>) -> Result<()> {
        let mut total_sequences = 0;
        let mut unique_sequences = HashSet::new();
        
        // Count sequences for logging
        for (_, second_char_map) in &partitions {
            for (_, sequences) in second_char_map {
                total_sequences += sequences.len();
                for &seq in sequences {
                    unique_sequences.insert(seq);
                }
            }
        }
        
        info!("Preparing to store prefix data:");
        info!("  Total sequences to store: {}", total_sequences);
        info!("  Unique sequence numbers: {}", unique_sequences.len());
        
        // Track metrics
        let mut successful_partitions = 0;
        let mut failed_partitions = 0;
        
        // Process each partition
        for (db_name, second_char_map) in partitions {
            // Get database handle
            let db = match self.prefix_dbs.get(&db_name) {
                Some(&db) => db,
                None => {
                    warn!("Database not found: {}", db_name);
                    failed_partitions += 1;
                    continue;
                }
            };
            
            // Use a separate transaction for each partition
            let mut attempt = 0;
            let mut partition_success = false;
            
            while attempt < MAX_RETRY_ATTEMPTS && !partition_success {
                attempt += 1;
                let start = Instant::now();
                
                // Start a write transaction
                let mut txn = match self.env.begin_rw_txn() {
                    Ok(txn) => txn,
                    Err(e) => {
                        if attempt < MAX_RETRY_ATTEMPTS {
                            warn!("Failed to start transaction (attempt {}): {}", attempt, e);
                            std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
                            continue;
                        }
                        return Err(Error::Database(format!("Failed to start transaction after {} attempts: {}", 
                                                        attempt, e)));
                    }
                };
                
                // Process each second-level character key
                let mut key_success_count = 0;
                let total_keys = second_char_map.len();
                
                for (second_char, sequences) in &second_char_map {
                    if sequences.is_empty() {
                        key_success_count += 1;
                        continue;
                    }
                    
                    // Key for this prefix (second char as string)
                    let key = second_char.to_string().into_bytes();
                    
                    // First try to get existing sequences
                    let mut merged_sequences = match txn.get(db, &key) {
                        Ok(existing_value) => {
                            // Successfully found existing data, try to deserialize it
                            match bincode::deserialize::<Vec<u64>>(existing_value) {
                                Ok(existing_sequences) => {
                                    trace!("Found {} existing sequences for {}.{}", 
                                        existing_sequences.len(), db_name, second_char);
                                    existing_sequences
                                },
                                Err(e) => {
                                    warn!("Failed to deserialize existing sequences for {}.{}: {}", 
                                        db_name, second_char, e);
                                    Vec::new() // Start fresh if deserialization fails
                                }
                            }
                        },
                        Err(e) if e.to_string().contains("NOTFOUND") => {
                            // Key doesn't exist yet, which is normal
                            trace!("No existing key found for {}.{}", db_name, second_char);
                            Vec::new()
                        },
                        Err(e) => {
                            // Other error occurred
                            warn!("Error reading existing key {}.{}: {}", db_name, second_char, e);
                            Vec::new() // Proceed with empty list in case of errors
                        }
                    };
                    
                    // Add the new sequences to the existing ones
                    let original_count = merged_sequences.len();
                    merged_sequences.extend(sequences);
                    
                    // Sort and deduplicate
                    merged_sequences.sort_unstable();
                    merged_sequences.dedup();
                    
                    // Log the merge statistics
                    if original_count > 0 {
                        trace!("Merged sequences for {}.{}: {} + {} = {} (after deduplication)",
                            db_name, second_char, original_count, sequences.len(), merged_sequences.len());
                    }
                    
                    // Serialize the merged sequences
                    let serialized = match bincode::serialize(&merged_sequences) {
                        Ok(data) => data,
                        Err(e) => {
                            warn!("Failed to serialize merged sequences for {}.{}: {}", db_name, second_char, e);
                            continue;
                        }
                    };
                    
                    // Store the merged sequences
                    if let Err(e) = txn.put(db, &key, &serialized, WriteFlags::empty()) {
                        warn!("Failed to store merged sequences for {}.{}: {}", db_name, second_char, e);
                    } else {
                        // Log success at debug level for the first few keys in each partition
                        if key_success_count < 5 {
                            debug!("Stored {} sequences for {}.{} (including {} new)", 
                                merged_sequences.len(), db_name, second_char, sequences.len());
                        }
                        key_success_count += 1;
                    }
                }
                
                // Commit the transaction if all keys were processed
                if key_success_count == total_keys {
                    match txn.commit() {
                        Ok(_) => {
                            partition_success = true;
                            successful_partitions += 1;
                            debug!("Successfully stored prefix partition {} with {} keys in {:?}", 
                                db_name, total_keys, start.elapsed());
                        },
                        Err(e) => {
                            if attempt < MAX_RETRY_ATTEMPTS {
                                warn!("Transaction commit failed for {} (attempt {}): {}", 
                                    db_name, attempt, e);
                                std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
                            } else {
                                warn!("Failed to commit transaction for {} after {} attempts: {}", 
                                    db_name, attempt, e);
                                failed_partitions += 1;
                            }
                        }
                    }
                } else {
                    // Not all keys were processed successfully, abort this transaction
                    txn.abort();
                    
                    if attempt < MAX_RETRY_ATTEMPTS {
                        warn!("Not all keys in {} were processed successfully, retrying (attempt {})", 
                            db_name, attempt);
                        std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
                    } else {
                        warn!("Failed to process all keys in {} after {} attempts", 
                            db_name, attempt);
                        failed_partitions += 1;
                    }
                }
            }
        }
        
        info!("Prefix data storage summary:");
        info!("  Successful partition writes: {}", successful_partitions);
        info!("  Failed partition writes: {}", failed_partitions);
        
        // If we have failures, make that more prominent in the logs
        if failed_partitions > 0 {
            warn!("ATTENTION: {} partition writes failed during prefix data storage", failed_partitions);
        } else {
            info!("All prefix partitions were successfully stored");
        }
        
        // Verify storage after writing
        self.verify_prefix_storage()?;
        
        info!("=== Completed prefix data storage ===");
        Ok(())
    }

    // Then update lookup_sequences_in_db to deserialize the value
    fn lookup_sequences_in_db(&self, db_name: &str, prefix: &[u8]) -> Result<Vec<u64>> {
        // Get the database handle (keep this part unchanged)
        let db = match self.prefix_dbs.get(db_name) {
            Some(&db) => db,
            None => {
                return Err(Error::Database(format!("Prefix database not found: {}", db_name)));
            }
        };
        
        // Parse prefix to get second character (keep this part unchanged)
        let mut second_char = Vec::new();
        let (_, second_char_opt) = extract_first_two_chars(prefix);
        if let Some(c) = second_char_opt {
            second_char = c.to_string().into_bytes();
        }
        
        // If we don't have a second character, return empty results (unchanged)
        if second_char.is_empty() {
            return Ok(Vec::new());
        }
        
        // Collect results with retry logic (unchanged)
        let mut sequences = Vec::new();
        
        // Retry loop (keep most of this part unchanged)
        for attempt in 1..=MAX_RETRY_ATTEMPTS {
            // Start a read transaction (unchanged)
            let txn = match self.env.begin_ro_txn() {
                Ok(txn) => txn,
                Err(e) => {
                    if attempt < MAX_RETRY_ATTEMPTS {
                        warn!("Failed to start transaction (attempt {}): {}", attempt, e);
                        std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
                        continue;
                    }
                    return Err(Error::Database(format!("Failed to start transaction: {}", e)));
                }
            };
            
            // Get the value for the key (unchanged)
            match txn.get(db, &second_char) {
                Ok(value) => {
                    // Here's where we modify the code to use our SIMD deserializer
                    let deserialize_start = Instant::now();
                    
                    // Try our SIMD-optimized deserializer first
                    match deserialize_u64_vec(value) {
                        Ok(seqs) => {
                            sequences = seqs;
                            let elapsed = deserialize_start.elapsed();
                            
                            // Log performance metrics if there are many sequences
                            if sequences.len() > 1000 && log::log_enabled!(log::Level::Debug) {
                                let throughput = (sequences.len() as f64) / elapsed.as_secs_f64() / 1_000_000.0;
                                debug!(
                                    "SIMD: Deserialized {} sequences for prefix in {}: {:?} ({:.2} M/sec)", 
                                    sequences.len(), db_name, elapsed, throughput
                                );
                            } else {
                                trace!(
                                    "SIMD: Successfully deserialized {} sequences for prefix in {}", 
                                    sequences.len(), db_name
                                );
                            }
                        },
                        Err(e) => {
                            // Fall back to standard bincode if SIMD deserializer fails
                            warn!(
                                "SIMD deserializer failed for prefix in {}: {}. Falling back to standard deserializer", 
                                db_name, e
                            );
                            
                            match bincode::deserialize::<Vec<u64>>(value) {
                                Ok(seqs) => {
                                    sequences = seqs;
                                    trace!(
                                        "Fallback: Successfully deserialized {} sequences for prefix in {}", 
                                        sequences.len(), db_name
                                    );
                                },
                                Err(e) => {
                                    warn!("Failed to deserialize sequences for prefix in {}: {}", db_name, e);
                                }
                            }
                        }
                    }
                },
                Err(lmdb_rkv::Error::NotFound) => {
                    // Key doesn't exist, this is normal (unchanged)
                    trace!("No entries found for key in database {}", db_name);
                },
                Err(e) => {
                    // Other error (unchanged)
                    warn!("Error getting key from database {}: {}", db_name, e);
                }
            }
            
            // End transaction (unchanged)
            txn.abort();
            break;
        }
        
        // Return the sequences (unchanged)
        trace!("Found {} sequences for prefix in {}", sequences.len(), db_name);
        Ok(sequences)
    }

    /// Verify that prefix data was correctly stored
    pub fn verify_prefix_storage(&self) -> Result<()> {
        debug!("Verifying prefix storage...");
        
        let start = Instant::now();
        let mut total_verified = 0;
        let mut database_counts = HashMap::new();
        
        // Start a read transaction
        let txn = match self.env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => return Err(Error::Database(format!("Failed to start transaction: {}", e))),
        };
        
        // Check each prefix database
        for partition in PREFIX_PARTITIONS.iter() {
            let db_name = &partition.db_name;
            
            // Get database handle
            let db = match self.prefix_dbs.get(db_name) {
                Some(db) => *db,
                None => {
                    debug!("Database not found for verification: {}", db_name);
                    continue;
                }
            };
            
            // Sample some entries from this database to verify storage
            let mut cursor = match txn.open_ro_cursor(db) {
                Ok(cursor) => cursor,
                Err(e) => {
                    debug!("Failed to open cursor for {}: {}", db_name, e);
                    continue;
                }
            };
            
            // Count entries using the iterator
            let mut db_count = 0;
            let mut sample_count = 0;
            
            // Use cursor.iter() to get all key-value pairs
            for result in cursor.iter() {
                // First handle the Result to get the key-value pair
                if let Ok((key, value)) = result {
                    db_count += 1;
                    
                    // Look at a few samples (first 2 entries) to verify structure
                    if sample_count < 2 {
                        // Try to deserialize the value to verify it's correctly formatted
                        match bincode::deserialize::<Vec<u64>>(value) {
                            Ok(sequences) => {
                                // Successfully deserialized
                                let key_str = std::str::from_utf8(key).unwrap_or("(invalid UTF-8)");
                                info!("Sample entry from {}.{}: {} sequences stored", 
                                    db_name, key_str, sequences.len());
                                
                                // Show first few sequence numbers
                                if !sequences.is_empty() {
                                    let display_count = sequences.len().min(5);
                                    info!("  First {} sequences: {:?}", 
                                        display_count, &sequences[..display_count]);
                                }
                            },
                            Err(e) => {
                                warn!("Failed to deserialize value from {}: {}", db_name, e);
                            }
                        }
                        sample_count += 1;
                    }
                    
                    // Limit checking to 1000 entries per database
                    if db_count >= 1000 {
                        info!("Limiting verification to 1000 entries for database {}", db_name);
                        break;
                    }
                } else {
                    // Optionally handle cursor iteration errors
                    warn!("Error while iterating through database {}", db_name);
                }
            }
            
            // Store counts
            database_counts.insert(db_name.clone(), db_count);
            total_verified += db_count;
            
            // Drop cursor
            drop(cursor);
        }
        
        // End transaction
        txn.abort();
        
        // Log verification results
        info!("Prefix storage verification summary:");
        info!("  Total verified entries: {}", total_verified);
        info!("  Verification time: {:?}", start.elapsed());
        
        // Log which databases have data
        let populated_dbs = database_counts.iter()
            .filter(|(_, &count)| count > 0)
            .map(|(name, count)| format!("{}: {}", name, count))
            .collect::<Vec<_>>()
            .join(", ");
        
        info!("  Populated databases: {}", populated_dbs);
        
        Ok(())
    }

    /// Get sequences by prefix, using cache when available
    pub fn get_sequences_by_prefix(&self, prefix: &[u8]) -> Result<Vec<u64>> {
        // Early return for empty prefix (unchanged)
        if prefix.is_empty() {
            return Ok(Vec::new());
        }
        
        // Cache check (unchanged)
        {
            let mut cache = self.prefix_cache.write();
            if let Some(sequences) = cache.get(&prefix.to_vec()) {
                return Ok(sequences.clone());
            }
        }
        
        // Use SIMD to extract the first character for database selection
        let (first_char_opt, _) = extract_first_two_chars(prefix);
        
        let result = match first_char_opt {
            Some(first_char) => {
                if let Some(db_name) = CHAR_TO_DB.get(&first_char) {
                    self.lookup_sequences_in_db(db_name, prefix)?
                } else {
                    debug!("No prefix database for character: {}", first_char);
                    Vec::new()
                }
            },
            None => {
                debug!("No valid first character in prefix");
                Vec::new()
            }
        };
        
        // Cache update (unchanged)
        {
            let mut cache = self.prefix_cache.write();
            cache.put(prefix.to_vec(), result.clone());
        }
        
        Ok(result)
    }
    
    /// Get statistics about the prefix cache
    pub fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.prefix_cache.read();
        let current_size = cache.len();
        let max_capacity = cache.cap().get();
        (current_size, max_capacity)
    }

    /// Clear the prefix cache
    pub fn clear_prefix_cache(&mut self) {
        self.prefix_cache.write().clear();
    }

    /// Trace where specific sequences are located in prefix databases
    pub fn trace_sequence_locations(&self, sequences_to_trace: &[u64]) -> Result<()> {
        trace!("Tracing {} specific sequence numbers", sequences_to_trace.len());
        
        // Start a read transaction
        let txn = match self.env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => return Err(Error::Database(format!("Failed to start transaction: {}", e))),
        };
        
        for &seq in sequences_to_trace {
            trace!("Looking for sequence {} in prefix databases", seq);
            let mut found_locations = Vec::new();
            
            // Check each prefix database
            for partition in PREFIX_PARTITIONS.iter() {
                let db_name = &partition.db_name;
                
                // Get database handle
                let db = match self.prefix_dbs.get(db_name) {
                    Some(db) => *db,
                    None => {
                        continue;
                    }
                };
                
                // Open a cursor to the database
                let mut cursor = match txn.open_ro_cursor(db) {
                    Ok(cursor) => cursor,
                    Err(_) => {
                        continue;
                    }
                };
                
                // Search for the sequence value across all keys
                for entry in cursor.iter() {
                    if let Ok((key, value)) = entry {
                        // Deserialize the value as a Vec<u64>
                        match bincode::deserialize::<Vec<u64>>(value) {
                            Ok(sequences) => {
                                // Check if our sequence is in this vector
                                if sequences.contains(&seq) {
                                    // Convert key to a string for logging
                                    let key_str = String::from_utf8_lossy(key).to_string();
                                    found_locations.push(format!("{}:{}", db_name, key_str));
                                    break; // Found in this database, move to next
                                }
                            },
                            Err(e) => {
                                // Log error but continue processing
                                trace!("Failed to deserialize sequence vector in {}: {}", db_name, e);
                            }
                        }
                    }
                }
                
                // Drop cursor
                drop(cursor);
            }
            
            if found_locations.is_empty() {
                trace!("  Sequence {} not found in any prefix database", seq);
            } else {
                trace!("  Sequence {} found in {} locations: {:?}", 
                    seq, found_locations.len(), found_locations);
            }
        }
        
        // End transaction
        txn.abort();
        
        Ok(())
    }

    /// Validate the prefix index against NGram entries
    pub fn validate_prefix_index(&self, expected_ngram_count: usize) -> Result<()> {
        trace!("Validating prefix index against {} expected ngrams", expected_ngram_count);
        
        // Count sequences in prefix databases
        let (total_sequences, sequence_counts) = self.count_all_prefix_sequences()?;
        
        // Number of unique sequence numbers
        let unique_sequences = sequence_counts.len();
        
        // Check numbers
        trace!("Validation results:");
        trace!("  Expected ngram count: {}", expected_ngram_count);
        trace!("  Total sequences in prefix index: {}", total_sequences);
        trace!("  Unique sequence numbers in prefix index: {}", unique_sequences);
        
        // Analyze missing sequences
        if unique_sequences < expected_ngram_count {
            let missing_count = expected_ngram_count - unique_sequences;
            trace!("  MISSING SEQUENCES: {} ({:.2}% of expected)",
                 missing_count, 
                 (missing_count as f64 / expected_ngram_count as f64) * 100.0);
            
            // Find some examples of missing sequences
            let mut missing_examples = Vec::new();
            for seq in 1..=expected_ngram_count as u64 {
                if !sequence_counts.contains_key(&seq) {
                    missing_examples.push(seq);
                    if missing_examples.len() >= 5 {
                        break;
                    }
                }
            }
            
            if !missing_examples.is_empty() {
                trace!("  Sample missing sequences: {:?}", missing_examples);
                
                // Trace these specific sequences in the main DB to verify they exist
                self.verify_sequences_exist_in_main(&missing_examples)?;
            }
        }
        
        // Analyze duplicate sequences
        let duplicate_count = sequence_counts.values().filter(|&&count| count > 1).count();
        if duplicate_count > 0 {
            trace!("  DUPLICATE SEQUENCES: {} ({:.2}% of expected)", 
                 duplicate_count,
                 (duplicate_count as f64 / expected_ngram_count as f64) * 100.0);
            
            // Find some examples of duplicated sequences
            let mut duplicate_examples = Vec::new();
            for (seq, count) in &sequence_counts {
                if *count > 1 {
                    duplicate_examples.push((*seq, *count));
                    if duplicate_examples.len() >= 5 {
                        break;
                    }
                }
            }
            
            if !duplicate_examples.is_empty() {
                trace!("  Sample duplicates: {:?}", duplicate_examples);
                
                // Trace these specific sequences
                let seq_numbers: Vec<u64> = duplicate_examples.iter().map(|(seq, _)| *seq).collect();
                self.trace_sequence_locations(&seq_numbers)?;
            }
        }
        
        Ok(())
    }

    /// Count sequences in all prefix databases
    pub fn count_all_prefix_sequences(&self) -> Result<(usize, HashMap<u64, usize>)> {
        let mut total_count = 0;
        let mut sequence_counts: HashMap<u64, usize> = HashMap::new();
        
        trace!("Starting verification of all prefix sequences...");
        
        // Start a read transaction
        let txn = match self.env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => return Err(Error::Database(format!("Failed to start transaction: {}", e))),
        };
        
        // Iterate through all prefix databases
        for partition in PREFIX_PARTITIONS.iter() {
            let db_name = &partition.db_name;
            
            // Get database handle
            let db = match self.prefix_dbs.get(db_name) {
                Some(db) => *db,
                None => {
                    debug!("Database not found for counting: {}", db_name);
                    continue;
                }
            };
            
            // Open a cursor to the database
            let mut cursor = match txn.open_ro_cursor(db) {
                Ok(cursor) => cursor,
                Err(e) => {
                    debug!("Failed to open cursor for {}: {}", db_name, e);
                    continue;
                }
            };
            
            // Iterate through all entries
            for entry in cursor.iter() {
                if let Ok((_key, value)) = entry {
                    // Deserialize the value as a Vec<u64> - this is the key fix
                    match bincode::deserialize::<Vec<u64>>(value) {
                        Ok(sequences) => {
                            // Process each sequence in the vector
                            for &seq in &sequences {
                                // Count total and track occurrence counts
                                total_count += 1;
                                *sequence_counts.entry(seq).or_insert(0) += 1;
                            }
                        },
                        Err(e) => {
                            // Log error but continue processing
                            trace!("Failed to deserialize sequence vector in {}: {}", db_name, e);
                        }
                    }
                }
            }
            
            // Drop cursor
            drop(cursor);
        }
        
        // End transaction
        txn.abort();
        
        // Log summary statistics
        let unique_sequences = sequence_counts.len();
        let duplicate_count = sequence_counts.values().filter(|&&count| count > 1).count();
        let max_duplication = sequence_counts.values().max().unwrap_or(&0);
        
        trace!("Prefix sequence validation complete:");
        trace!("  Total sequences in prefix index: {}", total_count);
        trace!("  Unique sequence numbers: {}", unique_sequences);
        trace!("  Sequences with duplicates: {}", duplicate_count);
        trace!("  Maximum duplication count: {}", max_duplication);
        
        Ok((total_count, sequence_counts))
    }

    /// Verify sequences exist in the main database
    fn verify_sequences_exist_in_main(&self, sequences: &[u64]) -> Result<()> {
        trace!("Verifying {} sequences exist in main DB", sequences.len());
        
        // Start a read transaction
        let txn = match self.env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => return Err(Error::Database(format!("Failed to start transaction: {}", e))),
        };
        
        for &seq in sequences {
            let key = Self::create_ngram_key(seq);
            
            match txn.get(self.main_db, &key) {
                Ok(_) => {
                    trace!("  Sequence {} exists in main DB but is missing from prefix index", seq);
                },
                Err(LmdbError::NotFound) => {
                    trace!("  Sequence {} doesn't exist in main DB either", seq);
                },
                Err(e) => {
                    trace!("  Error checking sequence {}: {}", seq, e);
                }
            }
        }
        
        // End transaction
        txn.abort();
        
        Ok(())
    }
}