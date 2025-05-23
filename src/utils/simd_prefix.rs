use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Extracts the first two Unicode characters from a UTF-8 byte slice.
/// 
/// Uses SIMD acceleration when available, with optimized paths for:
/// - AVX2 on x86_64
/// - SSE2 on x86_64
/// - NEON on ARM64
/// 
/// Falls back to scalar implementation when SIMD is unavailable.
/// 
/// # Arguments
/// * `bytes` - A UTF-8 encoded byte slice
/// 
/// # Returns
/// A tuple containing options for the first and second character.
/// Either value will be None if the corresponding character doesn't exist.
pub fn extract_first_two_chars(bytes: &[u8]) -> (Option<char>, Option<char>) {
    // Skip empty input
    if bytes.is_empty() {
        return (None, None);
    }

    // Runtime feature detection and dispatch
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { extract_first_two_chars_avx2(bytes) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { extract_first_two_chars_sse2(bytes) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { extract_first_two_chars_neon(bytes) };
    }

    // Fallback to scalar implementation
    extract_first_two_chars_scalar(bytes)
}
/// Scalar implementation of first two character extraction
fn extract_first_two_chars_scalar(bytes: &[u8]) -> (Option<char>, Option<char>) {
    // Use the standard library to decode characters safely
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return (None, None), // Invalid UTF-8
    };

    let mut chars = s.chars();
    
    // Get first and second characters
    let first = chars.next();
    let second = chars.next();
    
    (first, second)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extract_first_two_chars_avx2(bytes: &[u8]) -> (Option<char>, Option<char>) {
    // If the slice is too short, use scalar
    if bytes.len() < 4 {
        return extract_first_two_chars_scalar(bytes);
    }
    
    // Load up to 32 bytes
    let data_len = bytes.len().min(32);
    let mut buffer = [0u8; 32];
    buffer[..data_len].copy_from_slice(&bytes[..data_len]);
    
    // Load into AVX2 register
    let data = _mm256_loadu_si256(buffer.as_ptr() as *const __m256i);
    
    // Create masks for UTF-8 pattern detection
    // For ASCII (0xxxxxxx): high bit is 0
    // For 2-byte (110xxxxx): starts with 110
    // For continuation (10xxxxxx): starts with 10
    
    // Create masks
    let ascii_mask = _mm256_set1_epi8(0x7F as i8); // 0x7F = 0111 1111
    let leading_2byte_mask = _mm256_set1_epi8(-32i8); // 0xE0 = 224, which is -32 in i8
    let continuation_mask = _mm256_set1_epi8(-64i8); // 0xC0 = 192, which is -64 in i8

    // Find ASCII bytes (high bit is 0)
    let is_ascii = _mm256_cmpeq_epi8(_mm256_and_si256(data, ascii_mask), data);
    // Find 2-byte leading bytes (110xxxxx)
    let is_2byte_leading = _mm256_cmpeq_epi8(
        _mm256_and_si256(data, leading_2byte_mask),
        _mm256_set1_epi8(-64) // 0xC0 as i8 = -64
    );
    // Find continuation bytes (10xxxxxx)
    let is_continuation = _mm256_cmpeq_epi8(
        _mm256_and_si256(data, continuation_mask),
        _mm256_set1_epi8(-128) // 0x80 as i8 = -128
    );
    
    // Convert to bitmasks
    let ascii_bits = _mm256_movemask_epi8(is_ascii);
    let lead_2byte_bits = _mm256_movemask_epi8(is_2byte_leading);
    let continuation_bits = _mm256_movemask_epi8(is_continuation);
    
    // Process results to find character boundaries
    let mut char_boundaries = [0, 0, 0]; // Start, first boundary, second boundary
    let mut num_chars = 0;
    let mut pos = 0;
    
    // Find up to 2 characters
    while pos < data_len && num_chars < 2 {
        char_boundaries[num_chars] = pos;
        
        // Check if current byte is ASCII
        if ascii_bits & (1 << pos) != 0 {
            // ASCII character (1 byte)
            pos += 1;
        }
        // Check if current byte is 2-byte leading byte
        else if lead_2byte_bits & (1 << pos) != 0 && pos + 1 < data_len && 
                continuation_bits & (1 << (pos + 1)) != 0 {
            // 2-byte character (common for Arabic)
            pos += 2;
        }
        // Must be 3 or 4 byte character, handle with scalar
        else {
            // Extract this substring and process with scalar
            return extract_first_two_chars_scalar(&bytes[pos..]);
        }
        
        num_chars += 1;
    }
    
    // Extract and decode characters
    match num_chars {
        0 => (None, None),
        1 => {
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..pos])
                .ok()
                .and_then(|s| s.chars().next());
            (first, None)
        },
        _ => {
            let first_end = char_boundaries[1];
            let second_end = pos;
            
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..first_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            let second = std::str::from_utf8(&bytes[char_boundaries[1]..second_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            (first, second)
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extract_first_two_chars_sse2(bytes: &[u8]) -> (Option<char>, Option<char>) {
    // If the slice is too short, use scalar
    if bytes.len() < 4 {
        return extract_first_two_chars_scalar(bytes);
    }
    
    // Load up to 16 bytes
    let data_len = bytes.len().min(16);
    let mut buffer = [0u8; 16];
    buffer[..data_len].copy_from_slice(&bytes[..data_len]);
    
    // Load into SSE2 register
    let data = _mm_loadu_si128(buffer.as_ptr() as *const __m128i);
    
    // Create masks for UTF-8 pattern detection
    let ascii_mask = _mm_set1_epi8(0x7F as i8); // 0x7F = 0111 1111
    let leading_2byte_mask = _mm_set1_epi8(-32i8); // 0xE0 as i8
    let continuation_mask = _mm_set1_epi8(-64i8); // 0xC0 as i8

    // Find ASCII bytes (high bit is 0)
    let is_ascii = _mm_cmpeq_epi8(_mm_and_si128(data, ascii_mask), data);
    // Find 2-byte leading bytes (110xxxxx)
    let is_2byte_leading = _mm_cmpeq_epi8(
        _mm_and_si128(data, leading_2byte_mask),
        _mm_set1_epi8(-64) // 0xC0 as i8 = -64
    );
    // Find continuation bytes (10xxxxxx)
    let is_continuation = _mm_cmpeq_epi8(
        _mm_and_si128(data, continuation_mask),
        _mm_set1_epi8(-128) // 0x80 as i8 = -128
    );
    
    // Convert to bitmasks
    let ascii_bits = _mm_movemask_epi8(is_ascii) as u32;
    let lead_2byte_bits = _mm_movemask_epi8(is_2byte_leading) as u32;
    let continuation_bits = _mm_movemask_epi8(is_continuation) as u32;
    
    // Similar logic as AVX2 to find character boundaries
    let mut char_boundaries = [0, 0, 0]; // Start, first boundary, second boundary
    let mut num_chars = 0;
    let mut pos = 0;
    
    while pos < data_len && num_chars < 2 {
        char_boundaries[num_chars] = pos;
        
        // Check if current byte is ASCII
        if ascii_bits & (1 << pos) != 0 {
            // ASCII character (1 byte)
            pos += 1;
        }
        // Check if current byte is 2-byte leading byte
        else if lead_2byte_bits & (1 << pos) != 0 && pos + 1 < data_len && 
                continuation_bits & (1 << (pos + 1)) != 0 {
            // 2-byte character (common for Arabic)
            pos += 2;
        }
        // Must be 3 or 4 byte character, handle with scalar
        else {
            // Extract this substring and process with scalar
            return extract_first_two_chars_scalar(&bytes[pos..]);
        }
        
        num_chars += 1;
    }
    
    // Same extraction logic as AVX2 version
    match num_chars {
        0 => (None, None),
        1 => {
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..pos])
                .ok()
                .and_then(|s| s.chars().next());
            (first, None)
        },
        _ => {
            let first_end = char_boundaries[1];
            let second_end = pos;
            
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..first_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            let second = std::str::from_utf8(&bytes[char_boundaries[1]..second_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            (first, second)
        }
    }
}
#[cfg(target_arch = "aarch64")]
unsafe fn extract_first_two_chars_neon(bytes: &[u8]) -> (Option<char>, Option<char>) {
    // If the slice is too short, use scalar
    if bytes.len() < 4 {
        return extract_first_two_chars_scalar(bytes);
    }
    
    // Load up to 16 bytes
    let data_len = bytes.len().min(16);
    let mut buffer = [0u8; 16];
    buffer[..data_len].copy_from_slice(&bytes[..data_len]);
    
    // Load data into NEON registers
    let data = vld1q_u8(buffer.as_ptr());
    
    // Create masks for UTF-8 pattern detection
    let ascii_mask = vdupq_n_u8(0x80); // 1000 0000
    let leading_2byte_mask = vdupq_n_u8(0xE0); // 1110 0000
    let continuation_mask = vdupq_n_u8(0xC0); // 1100 0000
    
    // Perform comparisons
    // ASCII: (byte & 0x80) == 0
    let is_ascii = vceqq_u8(vandq_u8(data, ascii_mask), vdupq_n_u8(0));
    
    // 2-byte leading: (byte & 0xE0) == 0xC0
    let is_2byte_leading = vceqq_u8(
        vandq_u8(data, leading_2byte_mask),
        vdupq_n_u8(0xC0)
    );
    
    // Continuation: (byte & 0xC0) == 0x80
    let is_continuation = vceqq_u8(
        vandq_u8(data, continuation_mask),
        vdupq_n_u8(0x80)
    );
    
    // Convert to bitmasks using an approach that works on NEON
    // (NEON doesn't have direct movemask equivalent)
    let mut ascii_bits = 0u16;
    let mut lead_2byte_bits = 0u16;
    let mut continuation_bits = 0u16;
    
    // Extract bits from each lane
    let mask = 1u16;
    for i in 0..16 {
        if vgetq_lane_u8(is_ascii, i) != 0 {
            ascii_bits |= mask << i;
        }
        if vgetq_lane_u8(is_2byte_leading, i) != 0 {
            lead_2byte_bits |= mask << i;
        }
        if vgetq_lane_u8(is_continuation, i) != 0 {
            continuation_bits |= mask << i;
        }
    }
    
    // Use the bitmasks to find character boundaries
    let mut char_boundaries = [0, 0, 0]; // Start, first boundary, second boundary
    let mut num_chars = 0;
    let mut pos = 0;
    
    while pos < data_len && num_chars < 2 {
        char_boundaries[num_chars] = pos;
        
        // Check if current byte is ASCII
        if ascii_bits & (1 << pos) != 0 {
            // ASCII character (1 byte)
            pos += 1;
        }
        // Check if current byte is 2-byte leading byte
        else if lead_2byte_bits & (1 << pos) != 0 && pos + 1 < data_len && 
                continuation_bits & (1 << (pos + 1)) != 0 {
            // 2-byte character (common for Arabic)
            pos += 2;
        }
        // Must be 3 or 4 byte character, handle with scalar
        else {
            // Extract this substring and process with scalar
            return extract_first_two_chars_scalar(&bytes[pos..]);
        }
        
        num_chars += 1;
    }
    
    // Same extraction logic as other implementations
    match num_chars {
        0 => (None, None),
        1 => {
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..pos])
                .ok()
                .and_then(|s| s.chars().next());
            (first, None)
        },
        _ => {
            let first_end = char_boundaries[1];
            let second_end = pos;
            
            let first = std::str::from_utf8(&bytes[char_boundaries[0]..first_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            let second = std::str::from_utf8(&bytes[char_boundaries[1]..second_end])
                .ok()
                .and_then(|s| s.chars().next());
                
            (first, second)
        }
    }
}