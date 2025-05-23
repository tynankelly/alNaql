// Create a new file at src/utils/discarded_logger.rs
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;
use log::{debug, warn};

pub struct DiscardedLogger {
    file: Mutex<Option<File>>,
}

impl DiscardedLogger {
    pub fn new() -> Self {
        Self {
            file: Mutex::new(None),
        }
    }

    pub fn init(&self, path: &Path) -> io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)?;
        
        let mut guard = self.file.lock().unwrap();
        *guard = Some(file);
        
        debug!("Initialized discarded content logger at {:?}", path);
        Ok(())
    }

    pub fn log_discarded(&self, content_type: &str, text: &str, reason: &str, metadata: Option<&str>) -> io::Result<()> {
        let mut guard = self.file.lock().unwrap();
        
        if let Some(file) = guard.as_mut() {
            // Format: [TYPE] | REASON | METADATA | TEXT
            writeln!(
                file, 
                "[{}] | {} | {} | {}", 
                content_type,
                reason,
                metadata.unwrap_or("-"),
                text.replace('\n', " ")
            )?;
            file.flush()?;
            Ok(())
        } else {
            warn!("Attempted to log discarded content but logger not initialized");
            Ok(()) // Return success even if not logging - allows code to continue
        }
    }
}

// Create a global singleton instance
lazy_static::lazy_static! {
    pub static ref DISCARDED_LOGGER: DiscardedLogger = DiscardedLogger::new();
}