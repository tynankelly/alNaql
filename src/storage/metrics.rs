use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use log::trace;

/// Tracks storage operation metrics for monitoring and optimization
#[derive(Debug)]
pub struct StorageMetrics {
    write_ops: AtomicU64,
    read_ops: AtomicU64,
    batch_ops: AtomicU64,
    failed_ops: AtomicU64,
    total_bytes_written: AtomicU64,
    total_bytes_read: AtomicU64,
    start_time: Instant,
}

impl StorageMetrics {
    pub fn new() -> Self {
        Self {
            write_ops: AtomicU64::new(0),
            read_ops: AtomicU64::new(0),
            batch_ops: AtomicU64::new(0),
            failed_ops: AtomicU64::new(0),
            total_bytes_written: AtomicU64::new(0),
            total_bytes_read: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn increment_writes(&self) {
        self.write_ops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_reads(&self) {
        self.read_ops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_batch_writes(&self, count: usize) {
        self.batch_ops.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn increment_failed_ops(&self) {
        self.failed_ops.fetch_add(1, Ordering::Relaxed);
        trace!("Failed operation recorded. Total failures: {}", self.failed_ops.load(Ordering::Relaxed));
    }

    pub fn record_bytes_written(&self, bytes: u64) {
        self.total_bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_bytes_read(&self, bytes: u64) {
        self.total_bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> StorageMetricsStats {
        let uptime = self.start_time.elapsed();
        StorageMetricsStats {
            write_operations: self.write_ops.load(Ordering::Relaxed),
            read_operations: self.read_ops.load(Ordering::Relaxed),
            batch_operations: self.batch_ops.load(Ordering::Relaxed),
            failed_operations: self.failed_ops.load(Ordering::Relaxed),
            bytes_written: self.total_bytes_written.load(Ordering::Relaxed),
            bytes_read: self.total_bytes_read.load(Ordering::Relaxed),
            uptime_seconds: uptime.as_secs(),
        }
    }

    pub fn reset(&self) {
        self.write_ops.store(0, Ordering::Relaxed);
        self.read_ops.store(0, Ordering::Relaxed);
        self.batch_ops.store(0, Ordering::Relaxed);
        self.failed_ops.store(0, Ordering::Relaxed);
        self.total_bytes_written.store(0, Ordering::Relaxed);
        self.total_bytes_read.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct StorageMetricsStats {
    pub write_operations: u64,
    pub read_operations: u64,
    pub batch_operations: u64,
    pub failed_operations: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub uptime_seconds: u64,
}

impl StorageMetricsStats {
    pub fn operations_per_second(&self) -> f64 {
        let total_ops = self.write_operations + self.read_operations + self.batch_operations;
        if self.uptime_seconds == 0 {
            return 0.0;
        }
        total_ops as f64 / self.uptime_seconds as f64
    }

    pub fn failure_rate(&self) -> f64 {
        let total_ops = self.write_operations + self.read_operations + self.batch_operations;
        if total_ops == 0 {
            return 0.0;
        }
        self.failed_operations as f64 / total_ops as f64
    }

    pub fn throughput_bytes_per_second(&self) -> f64 {
        if self.uptime_seconds == 0 {
            return 0.0;
        }
        (self.bytes_written + self.bytes_read) as f64 / self.uptime_seconds as f64
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}
