// config/binary_io.rs - Updated to support dual output

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::loader::Configurable;

/// Output format options for final results
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutputFormat {
    Binary,     // Only binary output
    Json,       // Only JSON output
    Both,       // Both binary AND JSON output
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Both  // Default to both for maximum compatibility
    }
}

impl OutputFormat {
    /// Check if binary output should be generated
    pub fn includes_binary(&self) -> bool {
        matches!(self, OutputFormat::Binary | OutputFormat::Both)
    }
    
    /// Check if JSON output should be generated
    pub fn includes_json(&self) -> bool {
        matches!(self, OutputFormat::Json | OutputFormat::Both)
    }
    
    /// Get file extensions for this format
    pub fn file_extensions(&self) -> Vec<&'static str> {
        match self {
            OutputFormat::Binary => vec!["bin"],
            OutputFormat::Json => vec!["jsonl"],
            OutputFormat::Both => vec!["bin", "jsonl"],
        }
    }
    
    /// Parse from string (for CLI args and config)
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "binary" | "bin" => Ok(OutputFormat::Binary),
            "json" | "jsonl" => Ok(OutputFormat::Json),
            "both" | "all" => Ok(OutputFormat::Both),
            _ => Err(Error::Config(
                format!("Invalid output format: {}. Use 'binary', 'json', or 'both'", s)
            ))
        }
    }
}

/// Binary I/O and compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryIOConfig {
    /// Output format for final results
    pub output_format: OutputFormat,
    
    /// Compress final binary output (only applies to binary format)
    pub compress_final_output: bool,
    
    /// Buffer size in KB for reading
    pub read_buffer_kb: usize,
    
    /// Buffer size in KB for writing  
    pub write_buffer_kb: usize,
    
    /// Compression pool size
    pub compression_pool_size: usize,
}

impl Default for BinaryIOConfig {
    fn default() -> Self {
        BinaryIOConfig {
            output_format: OutputFormat::Both,  // Default to both formats
            compress_final_output: true,
            read_buffer_kb: 256,
            write_buffer_kb: 256,
            compression_pool_size: 16,
        }
    }
}

impl Configurable for BinaryIOConfig {
    fn section_name() -> &'static str {
        "binary_io"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            "output_format" | "format" => {
                self.output_format = OutputFormat::from_str(value)?;
                Ok(true)
            },
            "compress_final_output" | "compress_final" => {
                self.compress_final_output = crate::config::loader::parse::bool(value)?;
                Ok(true)
            },
            "read_buffer_kb" => {
                self.read_buffer_kb = crate::config::loader::parse::usize(value)?;
                Ok(true)
            },
            "write_buffer_kb" => {
                self.write_buffer_kb = crate::config::loader::parse::usize(value)?;
                Ok(true)
            },
            "compression_pool_size" | "pool_size" => {
                self.compression_pool_size = crate::config::loader::parse::usize(value)?;
                Ok(true)
            },
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Validate buffer sizes (min 16KB, max 16MB)
        if self.read_buffer_kb < 16 || self.read_buffer_kb > 16384 {
            return Err(Error::Config(
                format!("read_buffer_kb must be between 16 and 16384, got {}", self.read_buffer_kb)
            ));
        }
        
        if self.write_buffer_kb < 16 || self.write_buffer_kb > 16384 {
            return Err(Error::Config(
                format!("write_buffer_kb must be between 16 and 16384, got {}", self.write_buffer_kb)
            ));
        }
        
        // Validate pool size (min 4, max 128)
        if self.compression_pool_size < 4 || self.compression_pool_size > 128 {
            return Err(Error::Config(
                format!("compression_pool_size must be between 4 and 128, got {}", self.compression_pool_size)
            ));
        }
        
        Ok(())
    }
}

impl BinaryIOConfig {
    /// Check if temporary files should be compressed (always true)
    pub fn compress_temp_files(&self) -> bool {
        true  // Always compress temporary files
    }
    
    /// Check if final binary output should be compressed
    pub fn should_compress_final(&self) -> bool {
        self.compress_final_output && self.output_format.includes_binary()
    }
    
    /// Get buffer size in bytes for reading
    pub fn read_buffer_bytes(&self) -> usize {
        self.read_buffer_kb * 1024
    }
    
    /// Get buffer size in bytes for writing
    pub fn write_buffer_bytes(&self) -> usize {
        self.write_buffer_kb * 1024
    }
    
    /// Generate output paths based on base path and format settings
    pub fn generate_output_paths(&self, base_path: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        
        if self.output_format.includes_binary() {
            let ext = if self.compress_final_output { "bin.lz4" } else { "bin" };
            paths.push(base_path.with_extension(ext));
        }
        
        if self.output_format.includes_json() {
            paths.push(base_path.with_extension("jsonl"));
        }
        
        paths
    }
}