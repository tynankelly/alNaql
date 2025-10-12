// utils/binary_io.rs

use std::io;
use thiserror::Error;
use anyhow::Result;

/// Magic bytes to identify AlNaql binary files
pub const ALNAQL_MAGIC: &[u8; 4] = b"ALNQ";

/// Current binary format version
pub const FORMAT_VERSION: u8 = 1;

/// Error types for binary I/O operations
#[derive(Error, Debug)]
pub enum BinaryIOError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Serialization failed: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Compression failed: {0}")]
    Compression(String),
    
    #[error("Decompression failed: {0}")]
    Decompression(String),
    
    #[error("Corrupted data: expected {expected} bytes, found {actual} bytes")]
    CorruptedData { expected: usize, actual: usize },
    
    #[error("Invalid magic bytes: expected ALNQ, found {found:?}")]
    InvalidMagic { found: [u8; 4] },
    
    #[error("Unsupported format version: {version} (expected {FORMAT_VERSION})")]
    UnsupportedVersion { version: u8 },
    
    #[error("File truncated: attempted to read beyond file size")]
    TruncatedFile,
    
    #[error("Invalid data type marker: {marker}")]
    InvalidDataType { marker: u8 },
}

/// Data type markers for different structs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    SequenceMatch = 1,
    SequenceCluster = 2,
    ReconstructedSegment = 3,
    FinalMatchResult = 4,
}

impl DataType {
    pub fn from_u8(value: u8) -> Result<Self, BinaryIOError> {
        match value {
            1 => Ok(DataType::SequenceMatch),
            2 => Ok(DataType::SequenceCluster),
            3 => Ok(DataType::ReconstructedSegment),
            4 => Ok(DataType::FinalMatchResult),
            _ => Err(BinaryIOError::InvalidDataType { marker: value }),
        }
    }
}

/// File header for all binary files (8 bytes)
#[derive(Debug, Clone, Copy)]
pub struct FileHeader {
    pub magic: [u8; 4],      // "ALNQ"
    pub version: u8,          // Format version
    pub compressed: u8,       // 1=compressed, 0=uncompressed
    pub data_type: DataType,  // Type of data in file
    pub reserved: u8,         // Reserved for future use
}

impl FileHeader {
    /// Create a new file header
    pub fn new(compressed: bool, data_type: DataType) -> Self {
        FileHeader {
            magic: *ALNAQL_MAGIC,
            version: FORMAT_VERSION,
            compressed: if compressed { 1 } else { 0 },
            data_type,
            reserved: 0,
        }
    }
    
    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5] = self.compressed;
        bytes[6] = self.data_type as u8;
        bytes[7] = self.reserved;
        bytes
    }
    
    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BinaryIOError> {
        if bytes.len() < 8 {
            return Err(BinaryIOError::TruncatedFile);
        }
        
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        
        if magic != *ALNAQL_MAGIC {
            return Err(BinaryIOError::InvalidMagic { found: magic });
        }
        
        let version = bytes[4];
        if version != FORMAT_VERSION {
            return Err(BinaryIOError::UnsupportedVersion { version });
        }
        
        Ok(FileHeader {
            magic,
            version,
            compressed: bytes[5],
            data_type: DataType::from_u8(bytes[6])?,
            reserved: bytes[7],
        })
    }
    
    /// Check if this file uses compression
    pub fn is_compressed(&self) -> bool {
        self.compressed == 1
    }
}

pub type BinaryResult<T> = Result<T, BinaryIOError>;

// utils/binary_io.rs (continued)

use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

/// Maximum reasonable size for a pooled buffer (1MB)
const MAX_POOLED_BUFFER_SIZE: usize = 1024 * 1024;

/// Default pool size (number of buffers to keep)
const DEFAULT_POOL_SIZE: usize = 16;

/// A buffer pool for reusing compression/decompression buffers
pub struct CompressionBufferPool {
    buffers: Arc<ArrayQueue<Vec<u8>>>,
    max_buffer_size: usize,
}

impl CompressionBufferPool {
    /// Create a new buffer pool with specified capacity
    pub fn new(pool_size: usize, max_buffer_size: usize) -> Self {
        let buffers = Arc::new(ArrayQueue::new(pool_size));
        
        // Pre-allocate half the pool size to start
        for _ in 0..pool_size/2 {
            let _ = buffers.push(Vec::with_capacity(max_buffer_size / 4));
        }
        
        CompressionBufferPool {
            buffers,
            max_buffer_size,
        }
    }
    
    /// Create a buffer pool with default settings
    pub fn default() -> Self {
        Self::new(DEFAULT_POOL_SIZE, MAX_POOLED_BUFFER_SIZE)
    }
    
    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&self) -> PooledBuffer {
        let buffer = self.buffers.pop()
            .unwrap_or_else(|| Vec::with_capacity(self.max_buffer_size / 4));
        
        PooledBuffer {
            buffer,
            pool: self.buffers.clone(),
            max_size: self.max_buffer_size,
        }
    }
    
    /// Compress data using a pooled buffer
    pub fn compress(&self, data: &[u8]) -> Result<PooledBuffer, BinaryIOError> {
        let mut pooled = self.get_buffer();
        pooled.buffer.clear();
        
        // LZ4 compression with size prepended
        let compressed = lz4_flex::compress_prepend_size(data);
        pooled.buffer.extend_from_slice(&compressed);
        
        Ok(pooled)
    }
    
    /// Decompress data using a pooled buffer  
    pub fn decompress(&self, data: &[u8]) -> Result<PooledBuffer, BinaryIOError> {
        let mut pooled = self.get_buffer();
        pooled.buffer.clear();
        
        // LZ4 decompression with size prepended
        match lz4_flex::decompress_size_prepended(data) {
            Ok(decompressed) => {
                pooled.buffer.extend_from_slice(&decompressed);
                Ok(pooled)
            }
            Err(e) => Err(BinaryIOError::Decompression(e.to_string()))
        }
    }
}

/// A buffer that returns to the pool when dropped
pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: Arc<ArrayQueue<Vec<u8>>>,
    max_size: usize,
}

impl PooledBuffer {
    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
    
    /// Get the buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Take ownership of the inner buffer (won't return to pool)
    pub fn into_vec(mut self) -> Vec<u8> {
        // Set capacity to 0 to prevent returning to pool
        let mut buffer = Vec::new();
        std::mem::swap(&mut buffer, &mut self.buffer);
        buffer
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Only return to pool if buffer is reasonable size
        if !self.buffer.is_empty() && self.buffer.capacity() <= self.max_size {
            self.buffer.clear();
            // Ignore error if pool is full
            let _ = self.pool.push(std::mem::take(&mut self.buffer));
        }
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// Global compression buffer pool (lazy initialized)
use std::sync::OnceLock;

static GLOBAL_BUFFER_POOL: OnceLock<CompressionBufferPool> = OnceLock::new();

/// Get or initialize the global buffer pool
pub fn global_buffer_pool() -> &'static CompressionBufferPool {
    GLOBAL_BUFFER_POOL.get_or_init(|| CompressionBufferPool::default())
}
// utils/binary_io.rs (continued)

use std::io::{Write, BufWriter};
use serde::Serialize;
use byteorder::{LittleEndian, WriteBytesExt};

/// Binary writer with optional compression
pub struct BinaryWriter<W: Write> {
    writer: BufWriter<W>,
    compress: bool,
    buffer_pool: &'static CompressionBufferPool,
    items_written: u64,
}

impl<W: Write> BinaryWriter<W> {
    /// Create a new binary writer
    pub fn new(writer: W, compress: bool, data_type: DataType) -> BinaryResult<Self> {
        let mut writer = BufWriter::with_capacity(256 * 1024, writer);
        
        // Write file header
        let header = FileHeader::new(compress, data_type);
        writer.write_all(&header.to_bytes())?;
        
        Ok(BinaryWriter {
            writer,
            compress,
            buffer_pool: global_buffer_pool(),
            items_written: 0,
        })
    }
    
    /// Write a single item
    pub fn write_item<T: Serialize>(&mut self, item: &T) -> BinaryResult<()> {
        // Serialize to binary
        let serialized = bincode::serialize(item)?;
        
        // Compress if enabled
        let data = if self.compress {
            let compressed = self.buffer_pool.compress(&serialized)?;
            compressed.into_vec()
        } else {
            serialized
        };
        
        // Write length prefix (4 bytes, little endian)
        self.writer.write_u32::<LittleEndian>(data.len() as u32)?;
        
        // Write data
        self.writer.write_all(&data)?;
        
        self.items_written += 1;
        
        // Auto-flush every 1000 items to prevent excessive buffering
        if self.items_written % 1000 == 0 {
            self.writer.flush()?;
        }
        
        Ok(())
    }
    
    /// Write multiple items at once (batch write)
    pub fn write_batch<T: Serialize>(&mut self, items: &[T]) -> BinaryResult<usize> {
        let mut written = 0;
        
        for item in items {
            self.write_item(item)?;
            written += 1;
        }
        
        Ok(written)
    }
    
    /// Flush any buffered data to the underlying writer
    pub fn flush(&mut self) -> BinaryResult<()> {
        self.writer.flush()?;
        Ok(())
    }
    
    /// Get the number of items written
    pub fn items_written(&self) -> u64 {
        self.items_written
    }
    
    /// Finish writing and return the underlying writer
    pub fn finish(mut self) -> BinaryResult<W> {
        self.flush()?;
        self.writer.into_inner()
            .map_err(|e| BinaryIOError::Io(e.into_error()))
    }
    
    /// Get a reference to the inner writer
    pub fn get_ref(&self) -> &W {
        self.writer.get_ref()
    }
    
    /// Get a mutable reference to the inner writer
    pub fn get_mut(&mut self) -> &mut W {
        self.writer.get_mut()
    }
}

/// Convenience builder for BinaryWriter
pub struct BinaryWriterBuilder {
    compress: bool,
    buffer_size: usize,
}

impl BinaryWriterBuilder {
    pub fn new() -> Self {
        BinaryWriterBuilder {
            compress: true,
            buffer_size: 256 * 1024,
        }
    }
    
    pub fn compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }
    
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    pub fn build<W: Write>(self, writer: W, data_type: DataType) -> BinaryResult<BinaryWriter<W>> {
        BinaryWriter::new(writer, self.compress, data_type)
    }
}

impl Default for BinaryWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
// utils/binary_io.rs (continued)

use std::io::{Read, BufReader};
use serde::de::DeserializeOwned;
use byteorder::ReadBytesExt;
use std::marker::PhantomData;

/// Binary reader with optional decompression
pub struct BinaryReader<R: Read> {
    reader: BufReader<R>,
    compress: bool,
    buffer_pool: &'static CompressionBufferPool,
    data_type: DataType,
    items_read: u64,
}

impl<R: Read> BinaryReader<R> {
    /// Create a new binary reader and read/validate header
    pub fn new(mut reader: R) -> BinaryResult<Self> {
        // Read and validate file header
        let mut header_bytes = [0u8; 8];
        reader.read_exact(&mut header_bytes)
            .map_err(|_| BinaryIOError::TruncatedFile)?;
        
        let header = FileHeader::from_bytes(&header_bytes)?;
        
        let reader = BufReader::with_capacity(256 * 1024, reader);
        
        Ok(BinaryReader {
            reader,
            compress: header.is_compressed(),
            buffer_pool: global_buffer_pool(),
            data_type: header.data_type,
            items_read: 0,
        })
    }
    
    /// Read a single item
    pub fn read_item<T: DeserializeOwned>(&mut self) -> BinaryResult<Option<T>> {
        // Read length prefix
        let len = match self.reader.read_u32::<LittleEndian>() {
            Ok(len) => len as usize,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Clean EOF - no more items
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        
        // Validate length is reasonable (max 100MB per item)
        if len > 100 * 1024 * 1024 {
            return Err(BinaryIOError::CorruptedData {
                expected: len,
                actual: 0,
            });
        }
        
        // Read data
        let mut data = vec![0u8; len];
        self.reader.read_exact(&mut data).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                BinaryIOError::TruncatedFile
            } else {
                e.into()
            }
        })?;
        
        // Decompress if needed
        let decompressed = if self.compress {
            let decompressed = self.buffer_pool.decompress(&data)?;
            decompressed.into_vec()
        } else {
            data
        };
        
        // Deserialize
        let item = bincode::deserialize(&decompressed)?;
        
        self.items_read += 1;
        Ok(Some(item))
    }
    
    /// Read all items into a vector
    pub fn read_all<T: DeserializeOwned>(&mut self) -> BinaryResult<Vec<T>> {
        let mut items = Vec::new();
        
        while let Some(item) = self.read_item()? {
            items.push(item);
        }
        
        Ok(items)
    }
    
    /// Read a batch of items (up to count)
    pub fn read_batch<T: DeserializeOwned>(&mut self, count: usize) -> BinaryResult<Vec<T>> {
        let mut items = Vec::with_capacity(count.min(1000));
        
        for _ in 0..count {
            match self.read_item()? {
                Some(item) => items.push(item),
                None => break,
            }
        }
        
        Ok(items)
    }
    
    /// Get the data type this reader expects
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
    
    /// Get the number of items read so far
    pub fn items_read(&self) -> u64 {
        self.items_read
    }
    
    /// Check if this reader uses compression
    pub fn is_compressed(&self) -> bool {
        self.compress
    }
    
    /// Convert reader into an iterator
    pub fn into_iter<T: DeserializeOwned>(self) -> BinaryIterator<R, T> {
        BinaryIterator {
            reader: self,
            _phantom: PhantomData,
        }
    }
}

/// Iterator wrapper for BinaryReader
pub struct BinaryIterator<R: Read, T> {
    reader: BinaryReader<R>,
    _phantom: PhantomData<T>,
}

impl<R: Read, T: DeserializeOwned> Iterator for BinaryIterator<R, T> {
    type Item = BinaryResult<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_item() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Convenience builder for BinaryReader
pub struct BinaryReaderBuilder {
    buffer_size: usize,
}

impl BinaryReaderBuilder {
    pub fn new() -> Self {
        BinaryReaderBuilder {
            buffer_size: 256 * 1024,
        }
    }
    
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    pub fn build<R: Read>(self, reader: R) -> BinaryResult<BinaryReader<R>> {
        BinaryReader::new(reader)
    }
}

impl Default for BinaryReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}