// src/utils/resource_coordinator/mod.rs

//! # Resource Coordinator Module
//! 
//! Provides centralized resource management through a lock-free shared state architecture.
//! 
//! ## Architecture
//! 
//! The module uses atomic variables for zero-overhead coordination between resource managers
//! and business logic. Managers update their state atomically, and hot paths read these
//! values directly without function call overhead.
//! 
//! ## Usage
//! ```rust
//! use kuttab::utils::resource_coordinator::SHARED_STATE;
//! use std::sync::atomic::Ordering;
//! 
//! // Read current memory pressure
//! let pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
//! 
//! // Check if under pressure
//! if SHARED_STATE.is_under_pressure() {
//!     // Reduce batch size or apply backpressure
//! }
//! ```

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU16, AtomicUsize, AtomicBool, Ordering};
use lazy_static::lazy_static;
use log::{info, debug};
use crate::error::Result;

//==============================================================================
// MODULE DECLARATIONS
//==============================================================================

pub mod memory_manager;
pub mod environment_manager;
pub mod ffi;

//==============================================================================
// SIMPLE RE-EXPORTS
//==============================================================================

// Global instances
pub use environment_manager::ENVIRONMENT_MANAGER;

// Core types
pub use memory_manager::{MemoryManager, MemoryState};
pub use environment_manager::{EnvironmentManager, EnvironmentHandle, EnvironmentConfig};
pub use ffi::readers;

/// Initialize the resource management system
/// 
/// This starts the resource managers which will monitor system state
/// and update the shared atomics that components read directly.
pub fn initialize_resource_management() -> Result<()> {
    info!("🚀 Initializing resource management system");
    // MemoryManager starts monitoring automatically when created
    // EnvironmentManager tracks LMDB environments as they're created
    
    // Verify the system is working
    let memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
    let thread_budget = SHARED_STATE.thread_budget.load(Ordering::Relaxed);
    
    info!("📊 Resource management initialized:");
    info!("  Memory pressure: {}", memory_pressure);
    info!("  Thread budget: {}", thread_budget);
    info!("  System CPUs: {}", num_cpus::get());
    
    Ok(())
}
//==============================================================================
// SHARED RESOURCE STATE
//==============================================================================

/// Shared atomic state for zero-overhead resource coordination
/// 
/// This struct is cache-line aligned (64 bytes) to prevent false sharing
/// between CPU cores. All fields are atomics for lock-free access.
#[repr(align(64))]
#[derive(Debug)]
pub struct SharedResourceState {
    // Memory pressure from MemoryManager (0-100 scale)
    // 0-25: Low, 26-50: Normal, 51-75: High, 76-100: Critical
    pub memory_pressure: AtomicU8,
    
    // Available thread budget from ThreadManager
    // Decremented on allocation, incremented on deallocation
    pub thread_budget: AtomicU32,
    
    // Free LMDB reader slots from EnvironmentManager
    // Critical for preventing "MDB_READERS_FULL" errors
    pub reader_slots_available: AtomicU16,
    
    // Recommended chunk/batch size (calculated by business logic)
    // Updated based on memory_pressure and thread_budget
    pub chunk_size_limit: AtomicUsize,
    
    // Global backpressure flag (set by business logic)
    // When true, operations should slow down or pause
    pub backpressure: AtomicBool,
    
    /// Number of active comparisons (0-255)
    pub active_comparisons: AtomicU8,
    
    /// Total threads allocated to all comparisons
    pub total_threads_allocated: AtomicU32,
    
    /// Number of databases currently in use
    pub databases_in_use: AtomicU16,
    
    // Update padding size to maintain 64-byte alignment
    _padding: [u8; 29], // Reduced from 37 to account for new fields
}

// Ensure our struct is exactly 64 bytes (one cache line)
const _: () = {
    let size = std::mem::size_of::<SharedResourceState>();
    assert!(size == 64, "SharedResourceState must be exactly 64 bytes");
};

lazy_static! {
    /// Global shared state for resource coordination
    /// 
    /// This is the central point where all resource managers report their state
    /// and business logic reads current system conditions.
    pub static ref SHARED_STATE: SharedResourceState = SharedResourceState {
        memory_pressure: AtomicU8::new(0),
        thread_budget: AtomicU32::new(0),
        reader_slots_available: AtomicU16::new(126),
        chunk_size_limit: AtomicUsize::new(1000),
        backpressure: AtomicBool::new(false),
        active_comparisons: AtomicU8::new(0),
        total_threads_allocated: AtomicU32::new(0),
        databases_in_use: AtomicU16::new(0),
        _padding: [0; 29],
    };
}

impl SharedResourceState {
    /// Get current memory pressure level as enum
    pub fn get_memory_state(&self) -> MemoryState {
        let pressure = self.memory_pressure.load(Ordering::Relaxed);
        match pressure {
            0..=25 => MemoryState::Low,
            26..=50 => MemoryState::Normal,
            51..=75 => MemoryState::AboveNormal,
            76..=100 => MemoryState::High,  // Was Critical, now High
            _ => MemoryState::High,         // Was Critical, now High
        }
    }
    
    /// Check if system is under resource pressure
    pub fn is_under_pressure(&self) -> bool {
        self.memory_pressure.load(Ordering::Relaxed) > 50 ||
        self.thread_budget.load(Ordering::Relaxed) < 4 ||
        self.reader_slots_available.load(Ordering::Relaxed) < 10
    }
    
    /// Get recommended batch size based on current conditions
    pub fn get_recommended_batch_size(&self) -> usize {
        self.chunk_size_limit.load(Ordering::Relaxed)
    }
    
    /// Update chunk size based on current resource state
    /// This is typically called by business logic after reading resource state
    pub fn update_chunk_size_based_on_pressure(&self) {
        let memory_pressure = self.memory_pressure.load(Ordering::Relaxed);
        let thread_budget = self.thread_budget.load(Ordering::Relaxed);
        
        // Base size on memory pressure
        let base_size = match memory_pressure {
            0..=25 => 2000,   // Low pressure
            26..=50 => 1000,  // Normal
            51..=75 => 500,   // High
            _ => 250,         // Critical
        };
        
        // Scale by available threads
        let scaled_size = if thread_budget > 8 {
            base_size
        } else if thread_budget > 4 {
            base_size * 3 / 4
        } else if thread_budget > 2 {
            base_size / 2
        } else {
            base_size / 4
        };
        
        self.chunk_size_limit.store(scaled_size, Ordering::Relaxed);
    }
    
    /// Update backpressure flag based on resource state
    pub fn update_backpressure(&self) {
        let should_backpressure = self.memory_pressure.load(Ordering::Relaxed) > 75 ||
                                  self.thread_budget.load(Ordering::Relaxed) < 2 ||
                                  self.reader_slots_available.load(Ordering::Relaxed) < 5;
        
        self.backpressure.store(should_backpressure, Ordering::Relaxed);
    }
    
    /// Debug helper to print current state
    pub fn debug_state(&self) {
        debug!(
            "ResourceState: memory_pressure={}, threads={}, readers={}, chunk_size={}, backpressure={}",
            self.memory_pressure.load(Ordering::Relaxed),
            self.thread_budget.load(Ordering::Relaxed),
            self.reader_slots_available.load(Ordering::Relaxed),
            self.chunk_size_limit.load(Ordering::Relaxed),
            self.backpressure.load(Ordering::Relaxed)
        );
    }
    /// Check if system can handle another comparison
    pub fn can_add_comparison(&self) -> bool {
        let memory_ok = self.memory_pressure.load(Ordering::Relaxed) < 75;
        let threads_ok = self.total_threads_allocated.load(Ordering::Relaxed) < self.thread_budget.load(Ordering::Relaxed);
        let readers_ok = self.reader_slots_available.load(Ordering::Relaxed) >= 2;
        
        memory_ok && threads_ok && readers_ok
    }
    
    /// Get comparison resource status
    pub fn get_comparison_status(&self) -> String {
        format!(
            "Comparisons: {}, Threads: {}/{}, Memory pressure: {}%, Readers: {}",
            self.active_comparisons.load(Ordering::Relaxed),
            self.total_threads_allocated.load(Ordering::Relaxed),
            self.thread_budget.load(Ordering::Relaxed),
            self.memory_pressure.load(Ordering::Relaxed),
            self.reader_slots_available.load(Ordering::Relaxed)
        )
    }
}

//==============================================================================
// CONVENIENCE FUNCTIONS FOR SHARED STATE
//==============================================================================

/// Get current memory pressure (0-100)
#[inline]
pub fn get_memory_pressure() -> u8 {
    SHARED_STATE.memory_pressure.load(Ordering::Relaxed)
}

/// Get available thread budget
#[inline]
pub fn get_thread_budget() -> u32 {
    SHARED_STATE.thread_budget.load(Ordering::Relaxed)
}

/// Get available LMDB reader slots
#[inline]
pub fn get_reader_slots() -> u16 {
    SHARED_STATE.reader_slots_available.load(Ordering::Relaxed)
}

/// Get recommended chunk size for current conditions
#[inline]
pub fn get_chunk_size() -> usize {
    SHARED_STATE.chunk_size_limit.load(Ordering::Relaxed)
}

/// Check if backpressure should be applied
#[inline]
pub fn should_apply_backpressure() -> bool {
    SHARED_STATE.backpressure.load(Ordering::Relaxed)
}
