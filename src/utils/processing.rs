use std::path::Path;
use std::time::Duration;
use std::fs;
use std::sync::Arc;
use log::{info, debug};
use tokio;

use std::future::Future;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::config::subsystems::processor::ProcessorConfig;  

use crate::{
    KuttabConfig,
    error::{Error, Result},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingMode {
    Sequential,
    Parallel,
}

pub struct ProcessingManager {
    mode: ProcessingMode,
    max_concurrent: usize,
    target_memory_percent: f64,
}

impl ProcessingManager {
    pub fn new(processing_config: &ProcessorConfig) -> Self {
        let mode = match processing_config.processing_mode.as_str() {
            "parallel" => ProcessingMode::Parallel,
            _ => ProcessingMode::Sequential
        };
    
        let max_concurrent = if mode == ProcessingMode::Parallel {
            if let Ok(mem_info) = sys_info::mem_info() {
                let available_gb = mem_info.avail as f64 / (1024.0 * 1024.0);
                // Calculate more conservative memory-based limit: require 4GB per file instead of 2GB
                let memory_based_limit = (available_gb / 4.0).max(1.0) as usize;
                
                // Use the more conservative of user config and memory-based limit
                if processing_config.max_concurrent_files == 0 {
                    // If user specified 0, use memory-based calculation
                    memory_based_limit
                } else {
                    // Otherwise use the minimum of user config and memory-based limit
                    processing_config.max_concurrent_files.min(memory_based_limit)
                }
            } else {
                // If we can't get memory info, use user config or fallback to a more conservative limit
                processing_config.max_concurrent_files.max(1).min(2)
            }
        } else {
            1
        };
    
        Self {
            mode,
            max_concurrent,
            target_memory_percent: 70.0,
        }
    }

    pub fn mode(&self) -> ProcessingMode {
        self.mode
    }

    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    pub fn should_cleanup(&self) -> bool {
        if let Ok(mem_info) = sys_info::mem_info() {
            let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
            usage_percent > self.target_memory_percent
        } else {
            false
        }
    }

    pub async fn process_directory<F, Fut, T>(
        &self,
        input_dir: &Path,
        output_dir: &Path,
        config: &KuttabConfig,
        m: &MultiProgress,
        process_file: F,
    ) -> Result<Vec<T>>
    where
        F: Fn(std::fs::DirEntry, &Path, &KuttabConfig, &MultiProgress) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let files = fs::read_dir(input_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("xml"))
            .collect::<Vec<_>>();
    
        info!("Found {} files to process in {:?}", files.len(), input_dir);
        info!("Processing mode: {:?}, max concurrent: {}", self.mode, self.max_concurrent);
    
        let total_progress = m.add(ProgressBar::new(files.len() as u64));
        total_progress.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} Overall: [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
            .unwrap());
    
        let mut results = Vec::new();
    
        match self.mode {
            ProcessingMode::Sequential => {
                for entry in files {
                    let result = process_file(entry, output_dir, config, m).await?;
                    results.push(result);
                    total_progress.inc(1);
    
                    // Enhanced memory cleanup after each file
                    if self.should_cleanup() {
                        info!("Performing aggressive memory cleanup between files");
                        
                        // Clear global caches
                        if let Ok(()) = std::panic::catch_unwind(|| {
                            use crate::storage::lmdb::cache::METADATA_CACHE;
                            METADATA_CACHE.clear_thread_local();
                        }) {
                            info!("Successfully cleared metadata cache");
                        }
                        
                        // Force database cleanup
                        crate::storage::clear_storage_cache();
                        
                        // Force garbage collection
                        drop(Vec::<u8>::with_capacity(10 * 1024 * 1024)); // 10MB
                        
                        // Wait for memory to settle
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        
                        // Log current memory status
                        if let Ok(mem_info) = sys_info::mem_info() {
                            let avail_gb = mem_info.avail as f64 / (1024.0 * 1024.0);
                            let total_gb = mem_info.total as f64 / (1024.0 * 1024.0);
                            let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
                            info!("Memory after cleanup: {:.1}% used, {:.2} GB available of {:.2} GB total", 
                                usage_percent, avail_gb, total_gb);
                        }
                    } else {
                        // Even if we don't need aggressive cleanup, still do basic cleanup
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            },
            ProcessingMode::Parallel => {
                let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));
                let mut handles = Vec::new();
            
                for entry in files {
                    let semaphore = Arc::clone(&semaphore);
                    
                    // Enhanced memory check before acquiring new permit
                    // More aggressive waiting if memory is getting high
                    let mut wait_time = 1;
                    while let Ok(mem_info) = sys_info::mem_info() {
                        let usage_percent = 100.0 - (mem_info.avail as f64 / mem_info.total as f64 * 100.0);
                        if usage_percent > self.target_memory_percent {
                            info!("Memory usage at {:.1}%, waiting {} seconds before starting new file...", 
                                usage_percent, wait_time);
                            
                            // If memory is very high, do more aggressive cleanup
                            if usage_percent > 85.0 {
                                info!("Memory usage critically high, forcing global cleanup");
                                crate::storage::clear_storage_cache();
                                drop(Vec::<u8>::with_capacity(50 * 1024 * 1024)); // 50MB
                            }
                            
                            tokio::time::sleep(Duration::from_secs(wait_time)).await;
                            
                            // Increase wait time exponentially for more aggressive backoff
                            wait_time = std::cmp::min(wait_time * 2, 15); // Cap at 15 seconds
                        } else {
                            break;
                        }
                    }
                    
                    let permit = semaphore.acquire_owned().await
                        .map_err(|e| Error::AsyncError(e.to_string()))?;
                    
                    let output_path = output_dir.to_path_buf();
                    let config = config.clone();
                    let m = m.clone();
                    let process_fn = process_file.clone();
            
                    let handle = tokio::spawn(async move {
                        let _permit = permit; // Keep permit alive for the duration of the task
                        let result = process_fn(entry, &output_path, &config, &m).await?;
                        
                        // Enhanced cleanup after processing each file
                        info!("Performing post-file processing cleanup");
                        
                        // Try to clean up metadata cache
                        if let Ok(()) = std::panic::catch_unwind(|| {
                            use crate::storage::lmdb::cache::METADATA_CACHE;
                            METADATA_CACHE.clear_thread_local();
                        }) {
                            debug!("Successfully cleared thread-local metadata cache");
                        }
                        
                        // Force some memory cleanup
                        drop(Vec::<u8>::with_capacity(20 * 1024 * 1024)); // 20MB
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        
                        Ok::<T, Error>(result)
                    });
                    handles.push(handle);
                }
            
                for handle in handles {
                    match handle.await {
                        Ok(Ok(result)) => results.push(result),
                        Ok(Err(e)) => return Err(e),
                        Err(e) => return Err(Error::AsyncError(e.to_string())),
                    }
                    total_progress.inc(1);
                    
                    // Do some cleanup between tasks even in parallel mode
                    if self.should_cleanup() {
                        info!("Cleaning up between task completions");
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        drop(Vec::<u8>::with_capacity(5 * 1024 * 1024)); // 5MB
                    }
                }
            }            
        }
    
        total_progress.finish_with_message("Processing complete");
        Ok(results)
    }
}