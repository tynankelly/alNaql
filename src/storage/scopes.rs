// storage/lmdb/scopes.rs
// Operation-Scoped Transaction Architecture - Minimal wrapping with value-added convenience

use std::sync::atomic::Ordering;
use lmdb_rkv::{Database, Transaction, Cursor, WriteFlags};

use crate::error::{Error, Result};
use crate::types::NGramEntry;
use crate::storage::NGramMetadata;
use super::LMDBStorage;

//==============================================================================
// CORE SCOPE STRUCTURES - TRANSACTION LIFECYCLE MANAGEMENT
//==============================================================================

/// Read-only transaction scope with proper lifetime management
/// Provides scoped access to LMDB read transactions with automatic cleanup
pub struct ReadScope<'storage> {
    /// Owned lmdb_rkv read-only transaction
    txn: lmdb_rkv::RoTransaction<'storage>,
    /// Reference to storage backend for database access and configuration
    storage: &'storage LMDBStorage,
}

/// Read-write transaction scope with proper lifetime management
/// Provides scoped access to LMDB write transactions with automatic cleanup
pub struct WriteScope<'storage> {
    /// Owned lmdb_rkv read-write transaction
    txn: lmdb_rkv::RwTransaction<'storage>,
    /// Reference to storage backend for database access and configuration
    storage: &'storage LMDBStorage,
    /// Track whether transaction has been committed for proper cleanup
    committed: bool,
}

//==============================================================================
// DIRECT TRANSACTION ACCESS - NO WRAPPING NEEDED
//==============================================================================

impl<'storage> ReadScope<'storage> {
    /// Create a new read scope from an owned lmdb_rkv read-only transaction
    #[inline(always)]
    pub(super) fn new(txn: lmdb_rkv::RoTransaction<'storage>, storage: &'storage LMDBStorage) -> Self {
        Self { txn, storage }
    }
    
    /// Direct access to lmdb_rkv transaction - use this for all basic operations
    /// This provides zero-overhead access to the underlying transaction
    #[inline(always)]
    pub fn txn(&self) -> &lmdb_rkv::RoTransaction<'storage> {
        &self.txn
    }
    
    /// Access to storage backend for database handles and configuration
    /// Use this to get database references and check storage state
    #[inline(always)]
    pub fn storage(&self) -> &LMDBStorage {
        self.storage
    }
}

impl<'storage> WriteScope<'storage> {
    /// Create a new write scope from an owned lmdb_rkv read-write transaction
    #[inline(always)]
    pub(super) fn new(txn: lmdb_rkv::RwTransaction<'storage>, storage: &'storage LMDBStorage) -> Self {
        Self { 
            txn, 
            storage, 
            committed: false 
        }
    }
    
    /// Direct access to lmdb_rkv transaction - use this for all basic operations
    /// This provides zero-overhead access to the underlying transaction
    #[inline(always)]
    pub fn txn(&mut self) -> &mut lmdb_rkv::RwTransaction<'storage> {
        &mut self.txn
    }
    
    /// Immutable access to transaction for read operations within write scope
    #[inline(always)]
    pub fn txn_ref(&self) -> &lmdb_rkv::RwTransaction<'storage> {
        &self.txn
    }
    
    /// Access to storage backend for database handles and configuration
    /// Use this to get database references and check storage state
    #[inline(always)]
    pub fn storage(&self) -> &LMDBStorage {
        self.storage
    }
    
    /// Commit transaction - manages committed state for proper cleanup
    /// This is the only way to successfully complete a write transaction
    #[inline(always)]
    pub fn commit(mut self) -> Result<()> {
        self.committed = true;
        // Use consume pattern to avoid move-out-of-Drop issue
        let txn = unsafe { std::ptr::read(&self.txn) };
        std::mem::forget(self); // Prevent Drop from running
        txn.commit()
            .map_err(|e| Error::Database(format!("Failed to commit transaction: {}", e)))
    }
}

//==============================================================================
// VALUE-ADDED CONVENIENCE METHODS - ONLY WHEN THEY ADD VALUE
//==============================================================================

impl ReadScope<'_> {
    /// Check if key exists - convenience method (lmdb_rkv doesn't have this)
    /// Returns true if key exists, false if not found, error for other issues
    #[inline(always)]
    pub fn exists<K>(&self, db: Database, key: &K) -> Result<bool> 
    where 
        K: AsRef<[u8]>
    {
        match self.txn.get(db, key) {
            Ok(_) => Ok(true),
            Err(lmdb_rkv::Error::NotFound) => Ok(false),
            Err(e) => Err(Error::Database(format!("Failed to check key existence: {}", e))),
        }
    }
    
    /// Iterator with positioning - adds convenience over raw cursor
    /// Allows starting iteration from a specific key position
    #[inline(always)]
    pub fn iter_from<K>(&self, db: Database, start_key: Option<&K>) -> Result<ScopeIterator<'_>> 
    where 
        K: AsRef<[u8]>
    {
        let mut cursor = self.txn.open_ro_cursor(db)
            .map_err(|e| Error::Database(format!("Failed to open cursor: {}", e)))?;
        
        let positioned = if let Some(key) = start_key {
            // Position cursor at or after the start key
            cursor.iter_from(key).next().is_some()
        } else {
            // Position at first key
            cursor.iter_start().next().is_some()
        };
        
        Ok(ScopeIterator::new(cursor, positioned))
    }
    
    /// Get with our error handling - converts lmdb_rkv errors to our Error type
    /// Provides consistent error handling across the application
    #[inline(always)]
    pub fn get<K>(&self, db: Database, key: &K) -> Result<Option<Vec<u8>>> 
    where 
        K: AsRef<[u8]>
    {
        match self.txn.get(db, key) {
            Ok(value) => Ok(Some(value.to_vec())),
            Err(lmdb_rkv::Error::NotFound) => Ok(None),
            Err(e) => Err(Error::Database(format!("Failed to get value: {}", e))),
        }
    }
    
    /// Get database statistics - adds error conversion
    #[inline(always)]
    pub fn get_db_stat(&self, db: Database) -> Result<lmdb_rkv::Stat> {
        self.txn.stat(db)
            .map_err(|e| Error::Database(format!("Failed to get database statistics: {}", e)))
    }
}

impl WriteScope<'_> {
    /// Put with bulk loading optimization - adds conditional logic based on storage state
    /// Uses NO_OVERWRITE flag during bulk loading for better performance
    #[inline(always)]
    pub fn put_optimized<K, V>(&mut self, db: Database, key: &K, value: &V) -> Result<()> 
    where 
        K: AsRef<[u8]>,
        V: AsRef<[u8]>
    {
        let flags = if self.storage.is_bulk_loading.load(Ordering::Relaxed) {
            WriteFlags::NO_OVERWRITE  // Faster for bulk operations
        } else {
            WriteFlags::empty()       // Safe for normal operations
        };
        
        self.txn.put(db, key, value, flags)
            .map_err(|e| Error::Database(format!("Failed to put optimized: {}", e)))
    }
    
    /// Delete with our error handling - handles NotFound gracefully
    /// Treats deletion of non-existent keys as success (idempotent)
    #[inline(always)]
    pub fn delete<K>(&mut self, db: Database, key: &K) -> Result<()> 
    where 
        K: AsRef<[u8]>
    {
        match self.txn.del(db, key, None) {
            Ok(_) => Ok(()),
            Err(lmdb_rkv::Error::NotFound) => Ok(()), // Deletion of non-existent key is OK
            Err(e) => Err(Error::Database(format!("Failed to delete key: {}", e))),
        }
    }
    
    /// Batch put operation with optimized flags
    /// Efficiently stores multiple key-value pairs in a single transaction
    #[inline(always)]
    pub fn put_batch<K, V>(&mut self, db: Database, entries: &[(&K, &V)]) -> Result<()> 
    where 
        K: AsRef<[u8]>,
        V: AsRef<[u8]>
    {
        let flags = if self.storage.is_bulk_loading.load(Ordering::Relaxed) {
            WriteFlags::NO_OVERWRITE
        } else {
            WriteFlags::empty()
        };
        
        for (key, value) in entries {
            self.txn.put(db, key, value, flags)
                .map_err(|e| Error::Database(format!("Failed to put batch entry: {}", e)))?;
        }
        Ok(())
    }
    
    /// Get database statistics for write transactions
    #[inline(always)]
    pub fn get_db_stat(&self, db: Database) -> Result<lmdb_rkv::Stat> {
        self.txn.stat(db)
            .map_err(|e| Error::Database(format!("Failed to get database statistics: {}", e)))
    }
}

//==============================================================================
// TYPE-SPECIFIC SERIALIZATION HELPERS - OUR DOMAIN TYPES
//==============================================================================

impl ReadScope<'_> {
    /// Deserialize our NGramEntry type - domain-specific helper
    #[inline(always)]
    pub fn deserialize_entry(value: &[u8]) -> Result<NGramEntry> {
        bincode::deserialize(value)
            .map_err(|e| Error::Serialization(format!("Failed to deserialize NGramEntry: {}", e)))
    }
    
    /// Deserialize our NGramMetadata type - domain-specific helper
    #[inline(always)]
    pub fn deserialize_metadata(value: &[u8]) -> Result<NGramMetadata> {
        bincode::deserialize(value)
            .map_err(|e| Error::Serialization(format!("Failed to deserialize NGramMetadata: {}", e)))
    }
    
    /// Deserialize sequence list - domain-specific helper for prefix operations
    #[inline(always)]
    pub fn deserialize_sequence_list(value: &[u8]) -> Result<Vec<u64>> {
        bincode::deserialize(value)
            .map_err(|e| Error::Serialization(format!("Failed to deserialize sequence list: {}", e)))
    }
    
    /// Deserialize u64 from big-endian bytes - common pattern for sequence numbers
    #[inline(always)]
    pub fn deserialize_u64(value: &[u8]) -> Result<u64> {
        if value.len() >= 8 {
            let bytes: [u8; 8] = value[0..8].try_into()
                .map_err(|_| Error::Serialization("Invalid u64 bytes".to_string()))?;
            Ok(u64::from_be_bytes(bytes))
        } else {
            Err(Error::Serialization("Value too short for u64".to_string()))
        }
    }
}

impl WriteScope<'_> {
    /// Serialize our NGramEntry type - domain-specific helper
    #[inline(always)]
    pub fn serialize_entry(&self, entry: &NGramEntry) -> Result<Vec<u8>> {
        bincode::serialize(entry)
            .map_err(|e| Error::Serialization(format!("Failed to serialize NGramEntry: {}", e)))
    }
    
    /// Serialize our NGramMetadata type - domain-specific helper
    #[inline(always)]
    pub fn serialize_metadata(&self, metadata: &NGramMetadata) -> Result<Vec<u8>> {
        bincode::serialize(metadata)
            .map_err(|e| Error::Serialization(format!("Failed to serialize NGramMetadata: {}", e)))
    }
    
    /// Serialize sequence list - domain-specific helper for prefix operations
    #[inline(always)]
    pub fn serialize_sequence_list(&self, sequences: &[u64]) -> Result<Vec<u8>> {
        bincode::serialize(sequences)
            .map_err(|e| Error::Serialization(format!("Failed to serialize sequence list: {}", e)))
    }
}

//==============================================================================
// KEY CREATION UTILITIES - SHARED PATTERNS
//==============================================================================

impl ReadScope<'_> {
    /// Create sequence key - shared pattern across modules
    #[inline(always)]
    pub fn create_sequence_key(sequence: u64) -> [u8; 8] {
        sequence.to_be_bytes()
    }
    
    /// Create prefix key - shared pattern for prefix operations
    #[inline(always)]
    pub fn create_prefix_key(second_char: u8) -> [u8; 1] {
        [second_char]
    }
    
    /// Create file path key - shared pattern for metadata operations
    #[inline(always)]
    pub fn create_file_key(file_path: &str) -> Vec<u8> {
        file_path.as_bytes().to_vec()
    }
    
    /// Create metadata summary key - constant key for database summary
    #[inline(always)]
    pub fn create_summary_key() -> &'static [u8] {
        b"summary"
    }
}

// WriteScope inherits key creation methods from ReadScope
impl WriteScope<'_> {
    /// Create sequence key - delegates to ReadScope
    #[inline(always)]
    pub fn create_sequence_key(sequence: u64) -> [u8; 8] {
        ReadScope::create_sequence_key(sequence)
    }
    
    /// Create prefix key - delegates to ReadScope
    #[inline(always)]
    pub fn create_prefix_key(second_char: u8) -> [u8; 1] {
        ReadScope::create_prefix_key(second_char)
    }
    
    /// Create file path key - delegates to ReadScope
    #[inline(always)]
    pub fn create_file_key(file_path: &str) -> Vec<u8> {
        ReadScope::create_file_key(file_path)
    }
    
    /// Create metadata summary key - delegates to ReadScope
    #[inline(always)]
    pub fn create_summary_key() -> &'static [u8] {
        ReadScope::create_summary_key()
    }
}

//==============================================================================
// ITERATOR WRAPPER - ADDS POSITIONING VALUE
//==============================================================================

/// Iterator wrapper that adds positioning convenience over raw LMDB cursors
/// Iterator wrapper that adds positioning convenience over raw LMDB cursors

pub struct ScopeIterator<'cursor> {
    /// Cursor must be kept alive for iterator lifetime (not accessed directly)
    _cursor: lmdb_rkv::RoCursor<'cursor>,
    iter: lmdb_rkv::Iter<'cursor>,
}

impl<'cursor> ScopeIterator<'cursor> {
    /// Create new iterator positioned appropriately
    #[inline(always)]
    fn new(mut cursor: lmdb_rkv::RoCursor<'cursor>, positioned: bool) -> Self {
        let iter = if positioned {
            cursor.iter()
        } else {
            cursor.iter_start()
        };
        
        Self { 
            _cursor: cursor,  // Underscore prefix indicates intentionally unused
            iter 
        }
    }
}

impl<'cursor> Iterator for ScopeIterator<'cursor> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(Ok((key, value))) => Some(Ok((key.to_vec(), value.to_vec()))),
            Some(Err(e)) => Some(Err(Error::Database(format!("Iterator error: {}", e)))),
            None => None,
        }
    }
}

//==============================================================================
// AUTOMATIC CLEANUP - TRANSACTION LIFECYCLE MANAGEMENT
//==============================================================================

impl Drop for ReadScope<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        // Read transactions auto-abort on drop - no explicit action needed
        // LMDB will automatically clean up the transaction
    }
}

impl Drop for WriteScope<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.committed {
            // Write transaction will auto-abort on drop
        } else {
        }
    }
}

//==============================================================================
// THREAD SAFETY MARKERS
//==============================================================================

// ReadScope is Send + Sync because lmdb_rkv read transactions are thread-safe
// when each thread has its own transaction
unsafe impl Send for ReadScope<'_> {}
unsafe impl Sync for ReadScope<'_> {}

// WriteScope is Send but not Sync (write transactions are exclusive)
unsafe impl Send for WriteScope<'_> {}
