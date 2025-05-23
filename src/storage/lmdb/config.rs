// storage/lmdb/config.rs

use log::debug;
use lmdb_rkv::{EnvironmentFlags, DatabaseFlags};
use crate::config::subsystems::storage::StorageConfig;

// Database names/identifiers
pub const CF_MAIN: &str = "main";            // Stores NGramEntry data
pub const CF_METADATA: &str = "metadata";     // Stores database statistics and file metadata
pub const CF_NGRAM_METADATA: &str = "ngram_metadata"; // Stores ngram metadata
pub const CF_PREFIX_BASE: &str = "prefix_";   // Base name for prefix partition DBs

// Memory and batch configuration constants
pub const MEMORY_LIMIT_PERCENTAGE: f64 = 0.7; // Use up to 70% of available memory
pub const MIN_BATCH_SIZE: usize = 1000;
pub const MAX_BATCH_SIZE: usize = 100_000;

// Default sizes for memory allocation
pub const DEFAULT_MAP_SIZE: usize = 10 * 1024 * 1024 * 1024;  // 10GB default map size
pub const DEFAULT_MAX_READERS: u32 = 126;  // Default max readers
pub const DEFAULT_MAX_DBS: u32 = 100;      // Default max databases

// Default flags
pub fn default_env_flags() -> EnvironmentFlags {
    EnvironmentFlags::NO_TLS |
    EnvironmentFlags::NO_READAHEAD
}

// Create default environment options
pub fn create_env_options(config: &StorageConfig) -> (EnvironmentFlags, u32, u32, usize) {
    let mut flags = default_env_flags();
    
    // Configure environment flags based on config
    if config.use_fsync {
        // Don't add NO_SYNC if fsync is enabled
    } else {
        flags |= EnvironmentFlags::NO_SYNC;
    }
    
    if config.allow_mmap_writes {
        // This is default for LMDB
    } else {
        // No equivalent in LMDB, it always uses memory mapping
    }
    
    if config.allow_mmap_reads {
        // This is default for LMDB
    } else {
        // No equivalent in LMDB, it always uses memory mapping
    }
    
    // Configure max readers and max dbs
    let max_readers = config.lmdb_max_readers.unwrap_or(DEFAULT_MAX_READERS);
    let max_dbs = config.lmdb_max_dbs.unwrap_or(DEFAULT_MAX_DBS);
    
    // Configure map size
    let map_size = config.lmdb_map_size_mb.unwrap_or(DEFAULT_MAP_SIZE / (1024 * 1024)) * 1024 * 1024;
    
    debug!("Created LMDB environment options:");
    debug!("  Flags: {:?}", flags);
    debug!("  Max readers: {}", max_readers);
    debug!("  Max DBs: {}", max_dbs);
    debug!("  Map size: {} bytes", map_size);
    
    (flags, max_readers, max_dbs, map_size)
}

// Configure database flags
pub fn create_db_flags(is_dup_sort: bool) -> DatabaseFlags {
    let mut flags = DatabaseFlags::empty();
    
    if is_dup_sort {
        flags |= DatabaseFlags::DUP_SORT;
    }
    
    flags
}

// Create database flags for each database type
pub fn get_db_flags(db_name: &str) -> DatabaseFlags {
    if db_name == CF_MAIN || db_name == CF_METADATA || db_name == CF_NGRAM_METADATA {
        // These databases have unique keys
        DatabaseFlags::empty()
    } else if db_name.starts_with(CF_PREFIX_BASE) {
        // Prefix databases need duplicate keys
        DatabaseFlags::DUP_SORT
    } else {
        // Default to unique keys
        DatabaseFlags::empty()
    }
}

// Calculate the optimal batch size based on available memory
pub fn calculate_batch_size(available_memory: usize) -> usize {
    let estimated_entry_size = 1024; // Estimate average NGramEntry size in bytes
    let calculated_size = available_memory / (estimated_entry_size * 4);
    calculated_size.clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
}

// Helper to get all required database names
pub fn get_all_db_names(include_prefix_dbs: bool) -> Vec<String> {
    let mut names = vec![CF_MAIN.to_string(), CF_METADATA.to_string(), CF_NGRAM_METADATA.to_string()];
    
    if include_prefix_dbs {
        names.push(format!("{}space", CF_PREFIX_BASE));
        // Add other prefix databases as needed
    }
    
    names
}
/// Calculate appropriate LMDB map size based on input file size
/// Returns map size in bytes
pub fn calculate_map_size_from_file_size(file_size: u64) -> usize {
    // Using a progressive scale with generous overestimation
    // to accommodate any script and ngram density
    let base_map_size = match file_size {
        // Small files (<1MB): 2GB base
        size if size < 1_000_000 => 2_u64 << 30,
        
        // Medium files (1-10MB): 6GB base
        size if size < 10_000_000 => 6_u64 << 30,
        
        // Large files (10-50MB): 12GB base
        size if size < 50_000_000 => 12_u64 << 30,
        
        // Very large files (>50MB): 20GB base
        _ => 20_u64 << 30,
    };
    
    // Ensure the size is a multiple of the page size (typically 4KB)
    let page_size = 4096;
    let remainder = base_map_size % page_size;
    let map_size = if remainder == 0 {
        base_map_size
    } else {
        base_map_size + (page_size - remainder)
    };
    
    debug!("Calculated LMDB map size: {} MB for file size: {} bytes",
        map_size / (1024 * 1024), file_size);
    
    map_size as usize
}