// src/utils/mod.rs

//==============================================================================
// NON-RESOURCE-MANAGEMENT UTILITIES
//==============================================================================

pub mod binary_io;
pub mod boundary_handler;
pub mod comparison_coordinator;
pub mod string;
pub mod checkpoint;
pub mod cloud_signal;
pub mod tempfile;
pub mod processing;
pub mod logger;
pub mod mmap;
pub mod discarded_logger;
pub mod job_queue;
pub mod job_scheduler;
pub mod job_tracker;
pub mod parallel_executor;
pub mod progress_manager;
pub mod resume_manager;
pub mod simd;
pub mod simd_deserialize;
pub mod simd_prefix;
pub mod thread_allocation;

pub use comparison_coordinator::{ComparisonCoordinator, JobHandle};
pub use job_queue::{JobQueue, QueueStatus, QueueStatistics, WorkerMessage};
pub use job_scheduler::{JobScheduler, ExecutionBatch, SchedulerStats, optimize_batches};
pub use parallel_executor::{execute_parallel_continuous};
pub use progress_manager::{ProgressManager, ExecutionMode, ProgressPhase};
//==============================================================================
// RESOURCE MANAGEMENT
//==============================================================================

pub mod resource_coordinator;

//==============================================================================
// DIRECT RE-EXPORTS (Non-Resource Management)
//==============================================================================

pub use self::mmap::MmapFileHandler;
pub use self::string::{StringProcessor, FastStringBuilder, compute_text_hash};
pub use self::checkpoint::CheckpointManager;
pub use self::tempfile::TempFileManager;
pub use self::processing::ProcessingManager;
pub use self::logger::Logger;
pub use self::boundary_handler::BoundaryHandler;
pub use discarded_logger::DISCARDED_LOGGER;

//==============================================================================
// RESOURCE MANAGEMENT RE-EXPORTS
//==============================================================================

pub use resource_coordinator::{
    ENVIRONMENT_MANAGER,
    MemoryManager,
    MemoryState,
    EnvironmentManager,
    EnvironmentHandle,
};

//==============================================================================
// SIMPLE FUNCTIONS
//==============================================================================

pub fn initialize_utils() -> crate::error::Result<()> {
    use log::info;
    info!("Utils initialized");
    Ok(())
}