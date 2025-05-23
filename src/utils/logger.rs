use std::fs::File;
use std::io::Write;
use std::path::Path;
use crate::error::Result;

pub struct Logger {
    file: Option<File>,
    buffer: Vec<u8>,
}

impl Logger {
    pub fn new() -> Result<Self> {
        Ok(Logger {
            file: None,
            buffer: Vec::with_capacity(64 * 1024), // 64KB buffer
        })
    }

    pub fn with_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Logger {
            file: Some(File::create(path)?),
            buffer: Vec::with_capacity(64 * 1024),
        })
    }

    pub fn log(&mut self, message: &str) -> Result<()> {
        if let Some(ref mut file) = self.file {
            self.buffer.extend_from_slice(message.as_bytes());
            self.buffer.push(b'\n');
            
            if self.buffer.len() >= 64 * 1024 {
                file.write_all(&self.buffer)?;
                file.flush()?;
                self.buffer.clear();
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.file {
            if !self.buffer.is_empty() {
                file.write_all(&self.buffer)?;
                file.flush()?;
                self.buffer.clear();
            }
        }
        Ok(())
    }
}