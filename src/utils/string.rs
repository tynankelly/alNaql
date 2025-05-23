use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::{RwLock, Mutex};
use std::collections::VecDeque;
use std::borrow::Cow;
use ahash::AHashMap;
use memchr::memchr;
use log::debug;

// Constants for memory management
const MIN_CHUNK_SIZE: usize = 4 * 1024;      // 4KB minimum
const MAX_CHUNK_SIZE: usize = 1024 * 1024;   // 1MB maximum
const INITIAL_CHUNKS: usize = 4;              // Initial number of chunks in pool
const MAX_POOLED_CHUNKS: usize = 32;         // Maximum number of chunks to keep in pool
const SMALL_STRING_THRESHOLD: usize = 24;     // Threshold for small string optimization
const INITIAL_BUFFER_SIZE: usize = 4096;      // Initial buffer size for string processing

// StringProcessor implementation for text processing
pub struct StringProcessor {
    string_cache: Arc<RwLock<AHashMap<&'static str, String>>>,
    buffer: Vec<u8>,
    builder_buffer: String,
}

impl StringProcessor {
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            string_cache: Arc::new(RwLock::new(AHashMap::with_capacity(cache_capacity))),
            buffer: Vec::with_capacity(INITIAL_BUFFER_SIZE),
            builder_buffer: String::with_capacity(INITIAL_BUFFER_SIZE),
        }
    }

    pub fn process_text<'a>(&mut self, text: &'a str) -> Cow<'a, str> {
        if text.len() < SMALL_STRING_THRESHOLD {
            return Cow::Borrowed(text);
        }
        self.buffer.clear();
        self.buffer.extend_from_slice(text.as_bytes());
        Cow::Owned(String::from_utf8_lossy(&self.buffer).into_owned())
    }

    pub fn split_efficient<'a>(&self, text: &'a str, delimiter: u8) -> impl Iterator<Item = &'a str> {
        let bytes = text.as_bytes();
        let mut last = 0;
        std::iter::from_fn(move || {
            if last >= bytes.len() {
                return None;
            }
            match memchr(delimiter, &bytes[last..]) {
                Some(i) => {
                    let new_last = last + i + 1;
                    let slice = std::str::from_utf8(&bytes[last..last + i]).ok()?;
                    last = new_last;
                    Some(slice)
                }
                None => {
                    let slice = std::str::from_utf8(&bytes[last..]).ok()?;
                    last = bytes.len();
                    Some(slice)
                }
            }
        })
    }

    pub fn intern(&mut self, text: &'static str) -> &'static str {
        if !self.string_cache.read().contains_key(text) {
            self.string_cache.write().insert(text, text.to_string());
        }
        text
    }

    pub fn clear_buffers(&mut self) {
        self.buffer.clear();
        self.builder_buffer.clear();
    }
}

// ChunkPool for memory reuse in FastStringBuilder
struct ChunkPool {
    free_chunks: VecDeque<String>,
    total_pooled_size: usize,
}

impl ChunkPool {
    fn new(chunk_size: usize) -> Self {
        let mut free_chunks = VecDeque::with_capacity(INITIAL_CHUNKS);
        let mut total_size = 0;

        // Pre-allocate initial chunks
        for _ in 0..INITIAL_CHUNKS {
            let chunk = String::with_capacity(chunk_size);
            total_size += chunk.capacity();
            free_chunks.push_back(chunk);
        }

        Self {
            free_chunks,
            total_pooled_size: total_size,
        }
    }

    fn get_chunk(&mut self, required_capacity: usize) -> String {
        if let Some(mut chunk) = self.free_chunks.pop_front() {
            self.total_pooled_size -= chunk.capacity();
            chunk.clear();
            if chunk.capacity() >= required_capacity {
                chunk
            } else {
                String::with_capacity(required_capacity)
            }
        } else {
            String::with_capacity(required_capacity)
        }
    }

    fn return_chunk(&mut self, mut chunk: String) {
        chunk.clear();
        if self.free_chunks.len() < MAX_POOLED_CHUNKS {
            self.total_pooled_size += chunk.capacity();
            self.free_chunks.push_back(chunk);
        }
    }

    fn shrink(&mut self) {
        while self.total_pooled_size > MAX_CHUNK_SIZE * 2 && !self.free_chunks.is_empty() {
            if let Some(chunk) = self.free_chunks.pop_back() {
                self.total_pooled_size -= chunk.capacity();
            }
        }
    }
}

// FastStringBuilder for efficient string concatenation
pub struct FastStringBuilder {
    chunks: Vec<String>,
    current_chunk: String,
    chunk_size: usize,
    total_size: AtomicUsize,
    chunk_pool: Arc<Mutex<ChunkPool>>,
    memory_limit: Option<usize>,
}

impl FastStringBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        let chunk_size = capacity.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        debug!("Creating FastStringBuilder with chunk size: {}", chunk_size);
        Self {
            chunks: Vec::new(),
            current_chunk: String::with_capacity(chunk_size),
            chunk_size,
            total_size: AtomicUsize::new(0),
            chunk_pool: Arc::new(Mutex::new(ChunkPool::new(chunk_size))),
            memory_limit: None,
        }
    }

    pub fn with_memory_limit(capacity: usize, memory_limit_mb: usize) -> Self {
        let mut builder = Self::with_capacity(capacity);
        builder.memory_limit = Some(memory_limit_mb * 1024 * 1024);
        debug!("Set memory limit to {} MB", memory_limit_mb);
        builder
    }

    pub fn push_str(&mut self, s: &str) -> bool {
        debug!("Attempting to push {} chars to builder", s.len());
        
        // Check memory limit if set
        if let Some(limit) = self.memory_limit {
            if self.current_size() + s.len() > limit {
                debug!("Push rejected: would exceed memory limit of {}", limit);
                debug!("Current size: {}, Attempted push: {}", self.current_size(), s.len());
                return false;
            }
        }

        if self.current_chunk.len() + s.len() > self.chunk_size {
            debug!("Current chunk would overflow, creating new chunk");
            // Get new chunk from pool
            let mut pool = self.chunk_pool.lock();
            let new_chunk = pool.get_chunk(self.chunk_size);
            debug!("Created new chunk with capacity: {}", new_chunk.capacity());
            
            // Swap current chunk into storage
            let mut old_chunk = String::with_capacity(self.chunk_size);
            std::mem::swap(&mut self.current_chunk, &mut old_chunk);
            self.chunks.push(old_chunk);
            
            // Use new chunk
            self.current_chunk = new_chunk;
        }

        self.current_chunk.push_str(s);
        self.total_size.fetch_add(s.len(), Ordering::Relaxed);
        debug!("Successfully pushed {} chars, new total: {}", s.len(), self.current_size());
        true
    }

    pub fn build(mut self) -> String {
        let total_len = self.current_size();
        debug!("Building final string with length: {}", total_len);
        let mut result = String::with_capacity(total_len);
        
        // Append all chunks
        for chunk in self.chunks.drain(..) {
            result.push_str(&chunk);
            
            // Return chunk to pool
            let mut pool = self.chunk_pool.lock();
            pool.return_chunk(chunk);
        }
        
        // Append final chunk
        result.push_str(&self.current_chunk);
        
        // Clean up pool
        let mut pool = self.chunk_pool.lock();
        pool.shrink();
        
        debug!("Built string with final length: {}", result.len());
        result
    }

    pub fn current_size(&self) -> usize {
        self.total_size.load(Ordering::Relaxed)
    }

    pub fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn clear(&mut self) {
        let mut pool = self.chunk_pool.lock();
        
        // Return all chunks to pool
        for chunk in self.chunks.drain(..) {
            pool.return_chunk(chunk);
        }
        
        // Clear current chunk but keep its capacity
        self.current_chunk.clear();
        
        // Reset size counter
        self.total_size.store(0, Ordering::Relaxed);
        
        // Shrink pool if needed
        pool.shrink();
        
        debug!("Builder cleared");
    }

    pub fn shrink_to_fit(&mut self) {
        debug!("Shrinking builder to fit");
        // Shrink chunks if they're much larger than their content
        for chunk in &mut self.chunks {
            if chunk.capacity() > chunk.len() * 2 {
                chunk.shrink_to_fit();
            }
        }
        
        // Shrink current chunk if needed
        if self.current_chunk.capacity() > self.current_chunk.len() * 2 {
            self.current_chunk.shrink_to_fit();
        }
        
        // Clean up pool
        self.chunk_pool.lock().shrink();
    }

    pub fn memory_usage(&self) -> usize {
        let chunks_capacity: usize = self.chunks.iter()
            .map(|chunk| chunk.capacity())
            .sum();
        
        chunks_capacity + self.current_chunk.capacity() + 
        self.chunk_pool.lock().total_pooled_size
    }
}

impl Drop for FastStringBuilder {
    fn drop(&mut self) {
        // Return all chunks to pool on drop
        self.clear();
    }
}