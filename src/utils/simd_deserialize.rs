//! SIMD-accelerated deserializers for common data types used in the matching algorithm.

/// Deserializes a vector of u64 values from a Bincode-serialized byte array using SIMD instructions.
/// 
/// # Arguments
/// * `bytes` - The serialized data buffer
/// 
/// # Returns
/// A vector of u64 values, or an error if deserialization fails
pub fn deserialize_u64_vec(bytes: &[u8]) -> Result<Vec<u64>, String> {
    // Early validation - check that we have at least enough bytes for the length prefix
    if bytes.len() < 8 {
        return Err("Input too short for valid Bincode Vec<u64>".to_string());
    }
    
    // Extract length using LittleEndian conversion (bincode default)
    let length = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], 
        bytes[4], bytes[5], bytes[6], bytes[7]
    ]) as usize;
    
    // Validate expected total length
    let expected_bytes = 8 + (length * 8); // 8 bytes for length + 8 bytes per u64
    if bytes.len() < expected_bytes {
        return Err(format!("Input too short: expected {} bytes for {} elements, got {} bytes", 
                          expected_bytes, length, bytes.len()));
    }
    
    // Allocate output vector
    let mut result = Vec::with_capacity(length);
    
    // Choose appropriate SIMD implementation based on CPU features
    // (Similar to your existing SIMD detection pattern)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { deserialize_u64_vec_avx2(bytes, length, &mut result) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { deserialize_u64_vec_sse2(bytes, length, &mut result) }
        } else {
            deserialize_u64_vec_scalar(bytes, length, &mut result)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { deserialize_u64_vec_neon(bytes, length, &mut result) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        deserialize_u64_vec_scalar(bytes, length, &mut result)
    }
    
    Ok(result)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn deserialize_u64_vec_avx2(bytes: &[u8], length: usize, result: &mut Vec<u64>) {
    use std::arch::x86_64::*;
    
    // Start after the length prefix (8 bytes)
    let data_ptr = bytes.as_ptr().add(8);
    
    // Process 4 u64s (32 bytes) at a time
    let mut i = 0;
    while i + 4 <= length {
        // Load 32 bytes (4 u64s)
        let chunk = _mm256_loadu_si256(data_ptr.add(i * 8) as *const __m256i);
        
        // Store directly into our result vector
        // Note: This assumes little-endian is used consistently
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut __m256i, chunk);
        
        i += 4;
    }
    
    // Set the length (we've added elements without changing the length)
    result.set_len(i);
    
    // Process remaining elements individually
    for j in i..length {
        let value = u64::from_le_bytes([
            bytes[8 + j*8], bytes[9 + j*8], bytes[10 + j*8], bytes[11 + j*8],
            bytes[12 + j*8], bytes[13 + j*8], bytes[14 + j*8], bytes[15 + j*8]
        ]);
        result.push(value);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn deserialize_u64_vec_sse2(bytes: &[u8], length: usize, result: &mut Vec<u64>) {
    use std::arch::x86_64::*;
    
    // Start after the length prefix (8 bytes)
    let data_ptr = bytes.as_ptr().add(8);
    
    // Process 2 u64s (16 bytes) at a time
    let mut i = 0;
    while i + 2 <= length {
        // Load 16 bytes (2 u64s)
        let chunk = _mm_loadu_si128(data_ptr.add(i * 8) as *const __m128i);
        
        // Store directly into our result vector
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, chunk);
        
        i += 2;
    }
    
    // Set the length (we've added elements without changing the length)
    result.set_len(i);
    
    // Process remaining elements individually
    for j in i..length {
        let value = u64::from_le_bytes([
            bytes[8 + j*8], bytes[9 + j*8], bytes[10 + j*8], bytes[11 + j*8],
            bytes[12 + j*8], bytes[13 + j*8], bytes[14 + j*8], bytes[15 + j*8]
        ]);
        result.push(value);
    }
}
#[cfg(target_arch = "aarch64")]
unsafe fn deserialize_u64_vec_neon(bytes: &[u8], length: usize, result: &mut Vec<u64>) {
    use std::arch::aarch64::*;
    
    // Start after the length prefix (8 bytes)
    let data_ptr = bytes.as_ptr().add(8);
    
    // Process 2 u64s (16 bytes) at a time
    let mut i = 0;
    while i + 2 <= length {
        // Load 16 bytes (2 u64s)
        let chunk = vld1q_u8(data_ptr.add(i * a8));
        
        // Store directly into our result vector
        vst1q_u8(result.as_mut_ptr().add(i) as *mut u8, chunk);
        
        i += 2;
    }
    
    // Set the length (we've added elements without changing the length)
    result.set_len(i);
    
    // Process remaining elements individually
    for j in i..length {
        let value = u64::from_le_bytes([
            bytes[8 + j*8], bytes[9 + j*8], bytes[10 + j*8], bytes[11 + j*8],
            bytes[12 + j*8], bytes[13 + j*8], bytes[14 + j*8], bytes[15 + j*8]
        ]);
        result.push(value);
    }
}
fn deserialize_u64_vec_scalar(bytes: &[u8], length: usize, result: &mut Vec<u64>) {
    // Start after the length prefix (8 bytes)
    // Process each u64 individually
    for i in 0..length {
        let value = u64::from_le_bytes([
            bytes[8 + i*8], bytes[9 + i*8], bytes[10 + i*8], bytes[11 + i*8],
            bytes[12 + i*8], bytes[13 + i*8], bytes[14 + i*8], bytes[15 + i*8]
        ]);
        result.push(value);
    }
}