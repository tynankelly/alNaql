// src/matcher/concurrent.rs
use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
use std::time::{Instant, Duration};
use log::{info, debug, warn};

// We'll still use a channel-based approach for collecting results, 
// but with Rayon handling the parallel processing
pub struct ResultCollector<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    processed_count: AtomicUsize,
    expected_count: AtomicUsize,
    start_time: Instant,
}

impl<T> ResultCollector<T> {
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = bounded(buffer_size);
        
        ResultCollector {
            sender,
            receiver,
            processed_count: AtomicUsize::new(0),
            expected_count: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }
    
    pub fn sender(&self) -> Sender<T> {
        self.sender.clone()
    }
    
    pub fn increment_processed(&self) {
        self.processed_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn set_expected_count(&self, count: usize) {
        self.expected_count.store(count, Ordering::Relaxed);
    }
    
    pub fn get_processed_count(&self) -> usize {
        self.processed_count.load(Ordering::Relaxed)
    }
    
    pub fn get_expected_count(&self) -> usize {
        self.expected_count.load(Ordering::Relaxed)
    }
    
    pub fn collect_results(&self, timeout: Duration) -> Vec<T> {
        let mut results = Vec::new();
        let start_time = Instant::now();
        let mut collected_count = 0;
        
        // Collect until we've received all expected results or timeout
        while collected_count < self.get_expected_count() && start_time.elapsed() < timeout {
            match self.receiver.recv_timeout(timeout.min(Duration::from_millis(100))) {
                Ok(result) => {
                    debug!("Received result batch with {} items", 1);
                    results.push(result);
                    collected_count += 1;
                },
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Just a short timeout, continue waiting
                    continue;
                },
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // All senders disconnected
                    warn!("Channel disconnected after collecting {} results", collected_count);
                    break;
                }
            }
        }
        
        info!("Collected {} result batches out of {} expected", 
             results.len(), self.get_expected_count());
        
        results
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

// Memory monitoring remains similar
pub struct MemoryMonitor {
    warning_threshold: u64,
    critical_threshold: u64,
    last_check: Mutex<Instant>,
    check_interval: Duration,
}

impl MemoryMonitor {
    pub fn new(warning_mb: u64, critical_mb: u64, check_interval_ms: u64) -> Self {
        MemoryMonitor {
            warning_threshold: warning_mb * 1024 * 1024,
            critical_threshold: critical_mb * 1024 * 1024,
            last_check: Mutex::new(Instant::now()),
            check_interval: Duration::from_millis(check_interval_ms),
        }
    }
    
    pub fn should_check(&self) -> bool {
        let mut last_check = self.last_check.lock().unwrap();
        let elapsed = last_check.elapsed();
        
        if elapsed >= self.check_interval {
            *last_check = Instant::now();
            true
        } else {
            false
        }
    }
    
    pub fn check_memory(&self) -> Option<MemoryPressure> {
        if let Ok(mem_info) = sys_info::mem_info() {
            let used = (mem_info.total - mem_info.avail) * 1024; // KB to bytes
            
            if used > self.critical_threshold {
                Some(MemoryPressure::Critical)
            } else if used > self.warning_threshold {
                Some(MemoryPressure::Warning)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    pub fn get_memory_usage_mb(&self) -> u64 {
        if let Ok(mem_info) = sys_info::mem_info() {
            (mem_info.total - mem_info.avail) / 1024 // MB
        } else {
            0
        }
    }
}

pub enum MemoryPressure {
    Warning,
    Critical,
}