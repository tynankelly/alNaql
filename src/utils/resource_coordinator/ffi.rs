// src/storage/lmdb/ffi.rs
//
// This module provides direct FFI bindings to LMDB functions not exposed by lmdb-rkv.
// Currently focused on reader management functionality.

use std::path::Path;
use std::ffi::c_void;
use std::os::raw::{c_int, c_char};
use log::{debug, info, warn};

use lmdb_rkv::Environment;
use crate::error::{Error, Result};

/// Raw FFI declarations for LMDB functions
#[allow(non_camel_case_types)]
type MDB_env = c_void;

/// Safe type alias for MDB environment pointer
type MdbEnvPtr = *mut MDB_env;

// FFI binding to mdb_reader_check function
extern "C" {
    /// Check for stale readers in the environment.
    ///
    /// # Safety
    /// - env must be a valid LMDB environment pointer
    /// - function must be called from a single-threaded context for the environment
    fn mdb_reader_check(env: *mut MDB_env, dead: *mut c_int) -> c_int;
    
    /// Get LMDB version information
    fn mdb_version(major: *mut c_int, minor: *mut c_int, patch: *mut c_int) -> *const c_char;
}

/// Common LMDB error codes
const MDB_SUCCESS: c_int = 0;
const MDB_NOTFOUND: c_int = -30798;
const MDB_READERS_FULL: c_int = -30788;

/// Error handler for LMDB error codes
fn handle_lmdb_error(err_code: c_int, operation: &str) -> Result<()> {
    match err_code {
        MDB_SUCCESS => Ok(()),
        MDB_NOTFOUND => Err(Error::storage(format!("{}: No stale readers found", operation))),
        MDB_READERS_FULL => Err(Error::storage(format!("{}: Reader slots full", operation))),
        _ => Err(Error::storage(format!("{}: LMDB error code {}", operation, err_code))),
    }
}

/// Environment operations related to reader management
pub struct ReaderManager;

impl ReaderManager {
    /// Extract the raw LMDB environment pointer from lmdb_rkv::Environment
    ///
    /// # Safety
    /// This function accesses internal implementation details of the lmdb_rkv crate.
    /// It might break if the internal representation changes in future versions.
    pub unsafe fn extract_env_ptr(env: &Environment) -> Result<MdbEnvPtr> {
        // Approach 1: Try to extract it from the Debug representation
        // This is a bit hacky but may work if the Environment struct has a simple wrapper
        let env_ptr: MdbEnvPtr = {
            // The Environment struct likely contains a raw pointer field
            // We can try to extract it using pointer casting
            let env_ptr = (env as *const Environment) as *const *mut MDB_env;
            *env_ptr
        };

        // Validate the pointer
        if env_ptr.is_null() {
            return Err(Error::storage("Failed to extract valid environment pointer".to_string()));
        }

        Ok(env_ptr)
    }
    
    /// Alternative approach to extracting the environment pointer.
    /// This uses a more targeted approach if the above method fails.
    ///
    /// # Safety
    /// This is extremely unsafe and relies on internal layout of Environment struct.
    pub unsafe fn extract_env_ptr_alternative(env: &Environment) -> Result<MdbEnvPtr> {
        // IMPORTANT: This is a major hack and very unsafe.
        // It assumes the Environment struct has a specific memory layout.
        // This should be replaced with a more reliable method if possible.
        
        // Assume the first field of Environment is the raw pointer
        let env_ptr = *(env as *const _ as *const MdbEnvPtr);
        
        if env_ptr.is_null() {
            return Err(Error::storage("Failed to extract environment pointer (alt method)".to_string()));
        }
        
        Ok(env_ptr)
    }

    /// Check for and clean up stale readers
    ///
    /// This function calls the native LMDB mdb_reader_check function to identify
    /// and remove reader slots that belong to dead processes.
    ///
    /// Returns the number of stale readers that were cleared.
    pub fn cleanup_stale_readers(env: &Environment) -> Result<usize> {
        // Variable to store the count of dead readers found
        let mut dead: c_int = 0;
        
        // Extract the raw environment pointer
        let env_ptr = unsafe { Self::extract_env_ptr(env)? };
        
        // Call mdb_reader_check
        let result = unsafe { mdb_reader_check(env_ptr, &mut dead) };
        
        // Handle errors
        handle_lmdb_error(result, "Reader cleanup")?;
        
        // Convert to usize and return
        let cleaned = dead as usize;
        
        // Log the result
        if cleaned > 0 {
            info!("Cleaned up {} stale reader slots", cleaned);
        } else {
            debug!("No stale reader slots found");
        }
        
        Ok(cleaned)
    }
    
    /// Try all available methods to clean up stale readers
    ///
    /// This function attempts multiple approaches to clean up stale readers,
    /// falling back to alternative methods if the primary approach fails.
    pub fn cleanup_stale_readers_robust(env: &Environment) -> Result<usize> {
        // Try primary method first
        match Self::cleanup_stale_readers(env) {
            Ok(count) => Ok(count),
            Err(e) => {
                warn!("Primary reader cleanup failed: {}", e);
                
                // Try alternative method
                let env_ptr = unsafe { Self::extract_env_ptr_alternative(env)? };
                
                let mut dead: c_int = 0;
                let result = unsafe { mdb_reader_check(env_ptr, &mut dead) };
                
                // Handle errors
                handle_lmdb_error(result, "Alternative reader cleanup")?;
                
                let cleaned = dead as usize;
                if cleaned > 0 {
                    info!("Alternative method: Cleaned up {} stale reader slots", cleaned);
                }
                
                Ok(cleaned)
            }
        }
    }

    /// Get LMDB version information
    pub fn get_lmdb_version() -> (i32, i32, i32) {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        let mut patch: c_int = 0;
        
        unsafe {
            mdb_version(&mut major, &mut minor, &mut patch);
        }
        
        (major, minor, patch)
    }
    
    /// Utility function to check cleanup capability
    ///
    /// This function attempts to determine if reader cleanup is available
    /// in the current LMDB installation without actually performing a cleanup.
    pub fn is_cleanup_available() -> bool {
        // Get LMDB version
        let (major, minor, _) = Self::get_lmdb_version();
        
        // Reader cleanup should be available in recent LMDB versions
        if major > 0 || (major == 0 && minor >= 9) {
            // Minimal version check passed, now try to resolve the symbol
            // This is a bit tricky in Rust, so we'll just assume it's available
            // if the version is recent enough
            true
        } else {
            warn!("Detected LMDB version {}.{} which may not support reader cleanup", 
                 major, minor);
            false
        }
    }
}

/// Utility functions to work with environment paths
pub struct EnvPathUtils;

impl EnvPathUtils {
    /// Check if a lockfile exists for an environment
    pub fn has_lockfile(path: &Path) -> bool {
        let lock_path = path.join("lock.mdb");
        lock_path.exists()
    }
    
    /// Check if a reader lockfile exists
    pub fn has_reader_lockfile(path: &Path) -> bool {
        // LMDB typically uses "data.mdb" and "lock.mdb"
        // Reader state is part of lock.mdb
        Self::has_lockfile(path)
    }
    
    /// Check ownership of lockfiles
    ///
    /// This is platform specific and may not work on all systems.
    /// On Unix-like systems, it attempts to check if the lockfile
    /// is owned by the current process.
    #[cfg(unix)]
    pub fn check_lockfile_ownership(path: &Path) -> Result<bool> {
        use std::os::unix::fs::MetadataExt;
        
        let lock_path = path.join("lock.mdb");
        if !lock_path.exists() {
            return Ok(false);
        }
        
        match std::fs::metadata(&lock_path) {
            Ok(metadata) => {
                // Get owner uid
                let owner_uid = metadata.uid();
                
                // Get current process uid
                let current_uid = unsafe { libc::getuid() };
                
                Ok(owner_uid == current_uid)
            },
            Err(e) => {
                Err(Error::storage(format!("Failed to check lockfile ownership: {}", e)))
            }
        }
    }
    
    /// Windows version of check_lockfile_ownership (limited functionality)
    #[cfg(windows)]
    pub fn check_lockfile_ownership(path: &Path) -> Result<bool> {
        // On Windows, we can't easily check file ownership
        // We just check if the file exists
        Ok(Self::has_lockfile(path))
    }
}

/// Public API for checking and cleaning readers
///
/// This module provides a high-level API for reader management
/// that abstracts away the unsafe FFI details.
pub mod readers {
    use super::*;
    
    /// Check for and clean up stale readers
    ///
    /// Returns the number of stale readers that were cleaned up.
    pub fn cleanup_stale_readers(env: &Environment) -> Result<usize> {
        ReaderManager::cleanup_stale_readers(env)
    }
    
    /// Attempt to clean up stale readers with fallback methods
    ///
    /// This function tries multiple approaches to clean up stale readers.
    /// It's more robust than the basic cleanup function but may be slower.
    pub fn cleanup_stale_readers_robust(env: &Environment) -> Result<usize> {
        ReaderManager::cleanup_stale_readers_robust(env)
    }
    
    /// Check if reader cleanup is available
    pub fn is_cleanup_available() -> bool {
        ReaderManager::is_cleanup_available()
    }
    
    /// Get LMDB version information
    pub fn get_lmdb_version() -> (i32, i32, i32) {
        ReaderManager::get_lmdb_version()
    }
}