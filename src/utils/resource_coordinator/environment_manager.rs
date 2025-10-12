// src/utils/resource_coordinator/environment_manager.rs
// Environment manager with direct handle returns and full coordinator support
// Storage module creates databases, we only manage environments
// Updates shared state atomics for reader slot monitoring

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use log::{info, debug, warn};
use lmdb_rkv::{Environment, EnvironmentFlags};
use super::ffi;
use crate::error::{Error, Result};
use lazy_static::lazy_static;

// PHASE 2 ADDITION: Import shared state
use super::SHARED_STATE;

//==============================================================================
// ENVIRONMENT CONFIGURATION
//==============================================================================

/// Configuration for LMDB environment (simpler than storage::lmdb::config)
#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    pub flags: EnvironmentFlags,
    pub max_readers: u32,
    pub max_dbs: u32,
    pub map_size: usize,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            flags: EnvironmentFlags::empty(),
            max_readers: 126,
            max_dbs: 100,
            map_size: 10 * 1024 * 1024 * 1024, // 10GB default
        }
    }
}

//==============================================================================
// ENVIRONMENT METRICS
//==============================================================================

/// Metrics tracking for a single environment
#[derive(Debug)]
pub struct EnvironmentMetrics {
    creation_time: Instant,
    last_accessed: AtomicU64, // Unix timestamp for atomic updates
    transaction_count: AtomicUsize,
}

impl EnvironmentMetrics {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            creation_time: Instant::now(),
            last_accessed: AtomicU64::new(now),
            transaction_count: AtomicUsize::new(0),
        }
    }
    
    pub fn record_transaction(&self) {
        self.transaction_count.fetch_add(1, Ordering::Relaxed);
        self.update_access_time();
    }
    
    pub fn update_access_time(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_accessed.store(now, Ordering::Relaxed);
    }
    
    pub fn get_transaction_count(&self) -> u64 {
        self.transaction_count.load(Ordering::Relaxed) as u64
    }
    
    pub fn age(&self) -> Duration {
        self.creation_time.elapsed()
    }
    
    pub fn last_accessed_age(&self) -> Duration {
        let last_timestamp = self.last_accessed.load(Ordering::Relaxed);
        let current_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Duration::from_secs(current_timestamp.saturating_sub(last_timestamp))
    }
}

//==============================================================================
// MINIMAL ENVIRONMENT HANDLE (Exactly from Plan)
//==============================================================================

/// Minimal environment handle - ONLY provides environment access
/// Database creation and caching handled by storage layer (init.rs)
#[derive(Debug, Clone)]
pub struct EnvironmentHandle {
    /// Strong reference to LMDB environment (never fails to access)
    environment: Arc<Environment>,
    /// Database path for identification
    path: PathBuf,
    /// Simple metrics for coordinator integration
    metrics: Arc<EnvironmentMetrics>,
}

impl EnvironmentHandle {
    /// Create new environment handle with strong reference
    pub fn new(environment: Arc<Environment>, path: PathBuf) -> Self {
        debug!("Creating new EnvironmentHandle for {:?}", path);
        Self {
            environment,
            path,
            metrics: Arc::new(EnvironmentMetrics::new()),
        }
    }
    
    //==========================================================================
    // DIRECT ENVIRONMENT ACCESS (For scopes.rs integration)
    //==========================================================================
    
    /// Get direct reference to environment (for scopes.rs transaction creation)
    /// This is the ONLY method scopes.rs needs
    pub fn environment(&self) -> &Arc<Environment> {
        self.metrics.update_access_time();
        &self.environment
    }
    
    /// Get environment path
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    //==========================================================================
    // METRICS AND MONITORING (Environment-level only)
    //==========================================================================
    
    /// Get metrics for this environment
    pub fn get_metrics(&self) -> &Arc<EnvironmentMetrics> {
        &self.metrics
    }
    
    /// Check environment health (reader usage only)
    pub fn check_reader_health(&self) -> Result<f64> {
        match self.environment.info() {
            Ok(info) => {
                let max_readers = info.max_readers();
                let used_readers = info.num_readers();
                let health_ratio = (used_readers as f64 / max_readers as f64) * 100.0;
                
                // PHASE 2 ADDITION: Update shared state with available reader slots
                let available_slots = max_readers.saturating_sub(used_readers);
                SHARED_STATE.reader_slots_available.store(available_slots as u16, Ordering::Relaxed);
                
                if health_ratio > 90.0 {
                    warn!("High reader usage for {:?}: {:.1}% ({}/{})", 
                          self.path, health_ratio, used_readers, max_readers);
                }
                
                Ok(health_ratio)
            },
            Err(e) => {
                warn!("Failed to get environment info for {:?}: {}", self.path, e);
                Ok(0.0) // Return safe default
            }
        }
    }
    
    // Clean up stale readers for this environment
    pub fn cleanup_stale_readers(&self) -> Result<usize> {
        // Use the FFI wrapper to call LMDB's mdb_reader_check
        match ffi::readers::cleanup_stale_readers_robust(&self.environment) {
            Ok(count) => {
                if count > 0 {
                    info!("Cleaned up {} stale readers for environment at {:?}", 
                          count, self.path);
                } else {
                    debug!("No stale readers found for environment at {:?}", self.path);
                }
                Ok(count)
            },
            Err(e) => {
                warn!("Failed to clean up stale readers for {:?}: {}", self.path, e);
                Err(Error::Database(format!(
                    "Reader cleanup failed for {:?}: {}", self.path, e
                )))
            }
        }
    }
}

//==============================================================================
// ENVIRONMENT MANAGER (Minimal Global Registry)
//==============================================================================

/// Minimal environment manager - ONLY manages environments
#[derive(Debug)]
pub struct EnvironmentManager {
    environments: HashMap<PathBuf, EnvironmentHandle>,
}

impl EnvironmentManager {
    fn new() -> Self {
        Self {
            environments: HashMap::new(),
        }
    }
    
    /// Get or create environment handle - ONLY environment creation
    pub fn get_or_create_environment(&mut self, path: &Path, config: EnvironmentConfig) -> Result<EnvironmentHandle> {
        let path_buf = path.to_path_buf();
        
        // Return existing if available
        if let Some(handle) = self.environments.get(&path_buf) {
            debug!("Reusing existing environment for {:?}", path);
            return Ok(handle.clone());
        }
        
        // PHASE 2 ADDITION: If this is the first environment, initialize reader slots
        if self.environments.is_empty() {
            SHARED_STATE.reader_slots_available.store(config.max_readers as u16, Ordering::Relaxed);
        }
        
        // Create new environment
        debug!("Creating new LMDB environment at {:?}", path);
        
        let mut env_builder = Environment::new();
        env_builder
            .set_flags(config.flags)
            .set_max_readers(config.max_readers)
            .set_max_dbs(config.max_dbs)
            .set_map_size(config.map_size);
        
        let environment = env_builder
            .open(path)
            .map_err(|e| Error::storage(format!("Failed to open LMDB environment at {:?}: {}", path, e)))?;
        
        let environment = Arc::new(environment);
        let handle = EnvironmentHandle::new(environment.clone(), path_buf.clone());
        
        // Store in registry
        self.environments.insert(path_buf, handle.clone());
        info!("Created new LMDB environment at {:?}", path);
        
        Ok(handle)
    }
    
    /// Close environment by path (removes from registry)
    pub fn close_environment(&mut self, path: &Path) -> Result<bool> {
        let path_buf = path.to_path_buf();
        
        if let Some(handle) = self.environments.remove(&path_buf) {
            // Sync before closing (best effort)
            if let Err(e) = handle.environment.sync(true) {
                warn!("Failed to sync environment before closing {:?}: {}", path, e);
            } else {
                debug!("Successfully synced environment {:?} to disk", path);
            }
            
            // PHASE 2 ADDITION: Recalculate available reader slots after closing
            let _ = self.get_overall_reader_health(); // This will update the atomic
            
            info!("Successfully closed environment at {:?}", path);
            Ok(true)
        } else {
            debug!("No environment found to close at {:?}", path);
            Ok(false)
        }
    }
    
    /// Get statistics for all environments
    pub fn get_stats(&self) -> EnvironmentStats {
        let total_environments = self.environments.len();
        let total_transactions: u64 = self.environments
            .values()
            .map(|handle| handle.get_metrics().get_transaction_count())
            .sum();
        
        EnvironmentStats {
            total_environments,
            total_transactions,
        }
    }
    
    /// Clear all cached environments (for testing)
    pub fn clear_cache(&mut self) -> Result<usize> {
        let count = self.environments.len();
        self.environments.clear();
        info!("Cleared {} cached environments", count);
        Ok(count)
    }
    
    /// Get overall reader health across all environments
    pub fn get_overall_reader_health(&self) -> Result<f64> {
        if self.environments.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_health = 0.0;
        let mut successful_checks = 0;
        
        for handle in self.environments.values() {
            if let Ok(health) = handle.check_reader_health() {
                total_health += health;
                successful_checks += 1;
            }
        }
        
        // PHASE 2 ADDITION: Calculate total available reader slots across all environments
        let mut total_available_slots = 0u16;
        for handle in self.environments.values() {
            if let Ok(info) = handle.environment.info() {
                let available = info.max_readers().saturating_sub(info.num_readers());
                total_available_slots = total_available_slots.saturating_add(available as u16);
            }
        }
        SHARED_STATE.reader_slots_available.store(total_available_slots, Ordering::Relaxed);
        
        if successful_checks > 0 {
            Ok(total_health / successful_checks as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Cleanup stale readers across all environments (coordination only)
    pub fn cleanup_all_stale_readers(&self) -> Result<u32> {
        let mut total_cleaned = 0;
        
        for handle in self.environments.values() {
            // Use FFI for actual cleanup
            match ffi::readers::cleanup_stale_readers(&*handle.environment) {
                Ok(cleaned) => {
                    total_cleaned += cleaned as u32;
                },
                Err(e) => {
                    warn!("Failed to cleanup stale readers for {:?}: {}", handle.path, e);
                    // Continue with other environments
                }
            }
        }
        
        if total_cleaned > 0 {
            info!("Cleaned up {} total stale readers across all environments", total_cleaned);
        }
        
        Ok(total_cleaned)
    }
}

//==============================================================================
// STATISTICS
//==============================================================================

#[derive(Debug, Clone)]
pub struct EnvironmentStats {
    pub total_environments: usize,
    pub total_transactions: u64,
}

//==============================================================================
// GLOBAL ENVIRONMENT MANAGER INSTANCE
//==============================================================================

lazy_static! {
    pub static ref ENVIRONMENT_MANAGER: RwLock<EnvironmentManager> = 
        RwLock::new(EnvironmentManager::new());
}

//==============================================================================
// PUBLIC API FUNCTIONS (Minimal interface for storage module)
//==============================================================================

/// Get or create environment - main API function that storage module calls
pub fn get_or_create_environment(path: &Path, config: EnvironmentConfig) -> Result<EnvironmentHandle> {
    let mut manager = ENVIRONMENT_MANAGER.write().unwrap();
    manager.get_or_create_environment(path, config)
}

/// Close environment by path (utility function)
pub fn close_environment(path: &Path) -> Result<bool> {
    let mut manager = ENVIRONMENT_MANAGER.write().unwrap();
    manager.close_environment(path)
}

/// Get environment manager statistics (for monitoring)
pub fn get_environment_stats() -> EnvironmentStats {
    let manager = ENVIRONMENT_MANAGER.read().unwrap();
    manager.get_stats()
}

/// Clear environment cache (for testing)
pub fn clear_environment_cache() -> Result<usize> {
    let mut manager = ENVIRONMENT_MANAGER.write().unwrap();
    manager.clear_cache()
}

/// Simple maintenance function (for API compatibility with storage)
pub fn check_and_perform_maintenance() -> Result<bool> {
    debug!("Performing environment maintenance check");
    
    let manager = ENVIRONMENT_MANAGER.read().unwrap();
    let overall_health = manager.get_overall_reader_health()?;
    
    if overall_health > 80.0 {
        // High reader usage - perform cleanup
        let cleaned = manager.cleanup_all_stale_readers()?;
        info!("Maintenance cleanup: {} stale readers cleaned", cleaned);
        Ok(cleaned > 0)
    } else {
        debug!("Environment health is good: {:.1}% reader usage", overall_health);
        Ok(false)
    }
}

/// Clean up stale readers for a specific environment path (best effort)
pub fn cleanup_stale_readers(path: &Path) -> Result<u32> {
    debug!("Attempting to cleanup stale readers for {:?}", path);
    
    // Try to get environment from manager
    let manager = ENVIRONMENT_MANAGER.read().unwrap();
    if let Some(handle) = manager.environments.get(&path.to_path_buf()) {
        // Use FFI cleanup 
        match ffi::readers::cleanup_stale_readers(&*handle.environment) {
            Ok(count) => {
                if count > 0 {
                    info!("Cleaned up {} stale readers for {:?}", count, path);
                }
                Ok(count as u32)
            },
            Err(e) => {
                warn!("FFI cleanup failed for {:?}: {}", path, e);
                // Return 0 instead of failing - cleanup is best-effort
                Ok(0)
            }
        }
    } else {
        debug!("No active environment found for path {:?}", path);
        Ok(0)
    }
}

//==============================================================================
// READER HEALTH STATUS (For coordinator integration)
//==============================================================================

/// Reader health status for coordinator decision making
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderHealthStatus {
    /// Less than 70% reader slots used
    Healthy,
    /// 70-90% reader slots used
    Warning,
    /// Over 90% reader slots used
    Critical,
}

/// Get reader health status for a specific environment
pub fn get_reader_health(path: &Path) -> Result<ReaderHealthStatus> {
    let manager = ENVIRONMENT_MANAGER.read().unwrap();
    
    if let Some(handle) = manager.environments.get(&path.to_path_buf()) {
        let health_ratio = handle.check_reader_health()?;
        
        Ok(match health_ratio {
            x if x > 90.0 => ReaderHealthStatus::Critical,
            x if x > 70.0 => ReaderHealthStatus::Warning,
            _ => ReaderHealthStatus::Healthy,
        })
    } else {
        // If environment doesn't exist, assume healthy
        Ok(ReaderHealthStatus::Healthy)
    }
}

//==============================================================================
// ENVIRONMENT STATUS CHECKING (For verification)
//==============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentStatus {
    /// Path doesn't exist
    NotFound,
    /// Directory exists but no LMDB files
    DirectoryExists,
    /// Has valid LMDB data file
    HasDataFile,
    /// Currently open and tracked
    OpenAndTracked,
    /// Marked for closure but still has references
    ClosurePending,
}

/// Deep verification of environment status (for cleanup verification)
pub fn deep_verify_environment(path: &Path) -> Result<EnvironmentStatus> {
    // Check if path exists
    if !path.exists() {
        return Ok(EnvironmentStatus::NotFound);
    }
    
    // Check if it's a directory
    if !path.is_dir() {
        return Err(Error::storage(format!("Path {:?} is not a directory", path)));
    }
    
    // Check for LMDB data file
    let data_file = path.join("data.mdb");
    if !data_file.exists() {
        return Ok(EnvironmentStatus::DirectoryExists);
    }
    
    // Check if currently tracked
    let manager = ENVIRONMENT_MANAGER.read().unwrap();
    if manager.environments.contains_key(&path.to_path_buf()) {
        return Ok(EnvironmentStatus::OpenAndTracked);
    }
    
    // Has data file but not tracked
    Ok(EnvironmentStatus::HasDataFile)
}