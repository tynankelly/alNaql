//! SIMD accelerated utility functions for text matching operations.
//!
//! This module provides optimized implementations of common operations
//! that can benefit from SIMD acceleration, with appropriate fallbacks
//! for platforms that don't support specific instruction sets.

use log;

/// Detects the highest SIMD instruction set supported by the current CPU.
///
/// Returns a string identifier for the best available instruction set:
/// - "avx2": Advanced Vector Extensions 2
/// - "sse2": Streaming SIMD Extensions 2
/// - "neon": ARM NEON SIMD instructions
/// - "none": No SIMD support detected or disabled
pub fn detect_best_simd_support() -> &'static str {
    #[cfg(feature = "simd-disabled")]
    {
        return "none";
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "simd-disabled")))]
    {
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        
        if is_x86_feature_detected!("sse2") {
            return "sse2";
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "simd-disabled")))]
    {
        // ARM64 always has NEON in practice
        return "neon";
    }

    // Default when no SIMD support is detected
    "none"
}

/// Logs information about the SIMD support detected on the current system.
/// Call this at application startup to record capabilities in the log.
pub fn log_simd_support() {
    let simd_type = detect_best_simd_support();
    
    log::info!("SIMD support detected: {}", simd_type);
    
    match simd_type {
        "avx2" => log::info!("Using AVX2 SIMD instructions for text matching"),
        "sse2" => log::info!("Using SSE2 SIMD instructions for text matching"),
        "neon" => log::info!("Using NEON SIMD instructions for text matching"),
        _ => log::info!("SIMD optimizations not available, using scalar fallbacks"),
    }
}

/// Compares two byte slices for equality using SIMD acceleration.
///
/// This function performs the same operation as a standard
/// byte-by-byte comparison but uses SIMD instructions to check
/// multiple bytes at once when possible.
///
/// # Arguments
/// * `a` - First byte slice to compare
/// * `b` - Second byte slice to compare
///
/// # Returns
/// `true` if the slices have identical content, `false` otherwise
pub fn simd_memcmp(a: &[u8], b: &[u8]) -> bool {
    // Quick length check for immediate rejection
    if a.len() != b.len() {
        return false;
    }

    let len = a.len();

    // Empty slices are equal
    if len == 0 {
        return true;
    }

    // Special optimized path for very short strings (≤16 bytes)
    if len <= 16 {
        return compare_short_slices(a, b);
    }

    // Dispatch to the appropriate implementation based on runtime CPU feature detection
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compare_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { compare_sse2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64 in practice
        return unsafe { compare_neon(a, b) };
    }

    // Fallback to standard comparison if no SIMD instructions are available
    scalar_compare(a, b)
}

// Optimized path for short strings (16 bytes or less)
#[inline]
fn compare_short_slices(a: &[u8], b: &[u8]) -> bool {
    debug_assert!(a.len() == b.len() && a.len() <= 16);
    
    // For very short slices, direct byte comparison is often fastest
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

// Scalar comparison fallback for when SIMD is unavailable
#[inline]
fn scalar_compare(a: &[u8], b: &[u8]) -> bool {
    debug_assert!(a.len() == b.len());
    
    // Compare in chunks of 8 bytes for better cache utilization
    let chunk_size = std::mem::size_of::<u64>();
    let mut i = 0;
    
    // Process 8-byte chunks
    while i + chunk_size <= a.len() {
        let a_chunk = unsafe {
            std::ptr::read_unaligned(a.as_ptr().add(i) as *const u64)
        };
        let b_chunk = unsafe {
            std::ptr::read_unaligned(b.as_ptr().add(i) as *const u64)
        };
        
        if a_chunk != b_chunk {
            return false;
        }
        
        i += chunk_size;
    }
    
    // Handle remaining bytes
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    
    true
}

// AVX2 implementation (x86_64)
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn compare_avx2(a: &[u8], b: &[u8]) -> bool {
    use std::arch::x86_64::*;
    
    let len = a.len();
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    
    // Process 32 bytes at a time with AVX2
    let mut offset = 0;
    while offset + 32 <= len {
        // Load 32 bytes from each slice
        let a_chunk = _mm256_loadu_si256(ptr_a.add(offset) as *const __m256i);
        let b_chunk = _mm256_loadu_si256(ptr_b.add(offset) as *const __m256i);
        
        // Compare for equality
        let comparison = _mm256_cmpeq_epi8(a_chunk, b_chunk);
        
        // Create a mask where each bit represents a byte comparison result
        let mask = _mm256_movemask_epi8(comparison);
        
        // If not all bytes matched (mask != -1), return false
        if mask != -1i32 {
            return false;
        }
        
        offset += 32;
    }
    
    // Handle remaining bytes (less than 32)
    if offset < len {
        return scalar_compare(&a[offset..], &b[offset..]);
    }
    
    true
}

// SSE2 implementation (x86_64)
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn compare_sse2(a: &[u8], b: &[u8]) -> bool {
    use std::arch::x86_64::*;
    
    let len = a.len();
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    
    // Process 16 bytes at a time with SSE2
    let mut offset = 0;
    while offset + 16 <= len {
        // Load 16 bytes from each slice
        let a_chunk = _mm_loadu_si128(ptr_a.add(offset) as *const __m128i);
        let b_chunk = _mm_loadu_si128(ptr_b.add(offset) as *const __m128i);
        
        // Compare for equality
        let comparison = _mm_cmpeq_epi8(a_chunk, b_chunk);
        
        // Create a mask where each bit represents a byte comparison result
        let mask = _mm_movemask_epi8(comparison);
        
        // If not all bytes matched (mask != 0xFFFF), return false
        if mask != 0xFFFF {
            return false;
        }
        
        offset += 16;
    }
    
    // Handle remaining bytes (less than 16)
    if offset < len {
        return scalar_compare(&a[offset..], &b[offset..]);
    }
    
    true
}

// NEON implementation (ARM)
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn compare_neon(a: &[u8], b: &[u8]) -> bool {
    use std::arch::aarch64::*;
    
    let len = a.len();
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();
    
    // Process 16 bytes at a time with NEON
    let mut offset = 0;
    while offset + 16 <= len {
        // Load 16 bytes from each slice
        let a_chunk = vld1q_u8(ptr_a.add(offset));
        let b_chunk = vld1q_u8(ptr_b.add(offset));
        
        // Compare for equality - returns vector where each byte is either 0xFF or 0x00
        let comparison = vceqq_u8(a_chunk, b_chunk);
        
        // Check if all bytes are equal (converting to 4 32-bit values)
        let test_u32 = vreinterpretq_u32_u8(comparison);
        if vgetq_lane_u32(test_u32, 0) != u32::MAX ||
           vgetq_lane_u32(test_u32, 1) != u32::MAX ||
           vgetq_lane_u32(test_u32, 2) != u32::MAX ||
           vgetq_lane_u32(test_u32, 3) != u32::MAX {
            return false;
        }
        
        offset += 16;
    }
    
    // Handle remaining bytes (less than 16)
    if offset < len {
        return scalar_compare(&a[offset..], &b[offset..]);
    }
    
    true
}
/// Calculates Jaccard similarity between two strings using SIMD acceleration,
/// with full Unicode support for languages like Arabic.
///
/// This implementation handles Unicode code points correctly, ensuring proper
/// treatment of multi-byte characters while still leveraging SIMD acceleration
/// for performance.
///
/// # Arguments
/// * `a` - First string
/// * `b` - Second string
///
/// # Returns
/// A float value between 0.0 (completely different) and 1.0 (identical)
pub fn simd_jaccard(a: &str, b: &str) -> f64 {
    // Early returns for empty strings
    if a.is_empty() && b.is_empty() {
        return 1.0; // Two empty sets have similarity 1.0
    }
    if a.is_empty() || b.is_empty() {
        return 0.0; // If only one is empty, they have no similarity
    }

    // Step 1: Build character sets from both strings
    // This correctly handles Unicode by working with actual characters
    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();

    // Step 2: Create combined set of all unique characters from both texts
    let mut unique_chars = std::collections::HashSet::new();
    for &c in &chars_a {
        unique_chars.insert(c);
    }
    for &c in &chars_b {
        unique_chars.insert(c);
    }

    // If we have too few unique characters, use scalar implementation
    if unique_chars.len() <= 8 {
        return calculate_jaccard_scalar(chars_a, chars_b);
    }

    // Step 3: Create a mapping from characters to indices
    let mut char_to_index = std::collections::HashMap::new();
    let unique_chars_vec: Vec<char> = unique_chars.into_iter().collect();
    for (i, &c) in unique_chars_vec.iter().enumerate() {
        char_to_index.insert(c, i);
    }

    // Step 4: Build frequency arrays using the mapping
    // Resize to the next multiple of 32 for AVX2 (or 16 for SSE2/NEON) 
    // for optimal SIMD processing
    let array_size = round_to_simd_boundary(unique_chars_vec.len());
    let mut freq_a = vec![0u8; array_size];
    let mut freq_b = vec![0u8; array_size];

    // Populate frequency arrays
    for &c in &chars_a {
        let idx = *char_to_index.get(&c).unwrap();
        if freq_a[idx] < 255 { // Prevent overflow
            freq_a[idx] += 1;
        }
    }

    for &c in &chars_b {
        let idx = *char_to_index.get(&c).unwrap();
        if freq_b[idx] < 255 { // Prevent overflow
            freq_b[idx] += 1;
        }
    }

    // Step 5: Calculate intersection and union sizes using SIMD
    let (intersection_size, union_size) = calculate_set_sizes(&freq_a, &freq_b);

    // Step 6: Calculate Jaccard similarity
    if union_size == 0 {
        0.0 // Avoid division by zero
    } else {
        intersection_size as f64 / union_size as f64
    }
}

/// Rounds a size up to the nearest SIMD boundary
/// (32 for AVX2, 16 for SSE2/NEON)
#[inline]
fn round_to_simd_boundary(size: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        (size + 31) & !31 // Round up to multiple of 32
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        (size + 15) & !15 // Round up to multiple of 16
    }
}

/// Fast scalar implementation of Jaccard similarity for small character sets
#[inline]
fn calculate_jaccard_scalar(chars_a: Vec<char>, chars_b: Vec<char>) -> f64 {
    // Build frequency maps
    let mut freq_a = std::collections::HashMap::new();
    let mut freq_b = std::collections::HashMap::new();
    
    for c in chars_a {
        *freq_a.entry(c).or_insert(0) += 1;
    }
    
    for c in chars_b {
        *freq_b.entry(c).or_insert(0) += 1;
    }
    
    // Calculate intersection and union
    let mut intersection_size = 0;
    let mut union_size = 0;
    
    // Get all unique characters from both texts
    let mut all_chars = std::collections::HashSet::new();
    all_chars.extend(freq_a.keys());
    all_chars.extend(freq_b.keys());
    
    // Calculate intersection and union sizes
    for &c in &all_chars {
        let count_a = *freq_a.get(&c).unwrap_or(&0);
        let count_b = *freq_b.get(&c).unwrap_or(&0);
        
        intersection_size += std::cmp::min(count_a, count_b);
        union_size += std::cmp::max(count_a, count_b);
    }
    
    // Calculate similarity
    if union_size == 0 {
        0.0
    } else {
        intersection_size as f64 / union_size as f64
    }
}

/// Calculates intersection and union sizes from two frequency arrays using SIMD.
/// Returns a tuple of (intersection_size, union_size).
#[inline]
fn calculate_set_sizes(freq_a: &[u8], freq_b: &[u8]) -> (usize, usize) {
    // Dispatch to the appropriate SIMD implementation based on hardware support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { calculate_set_sizes_avx2(freq_a, freq_b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { calculate_set_sizes_sse2(freq_a, freq_b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 almost always has NEON
        return unsafe { calculate_set_sizes_neon(freq_a, freq_b) };
    }

    // Fallback to scalar implementation
    calculate_set_sizes_scalar(freq_a, freq_b)
}

/// Scalar fallback implementation for calculating set sizes.
#[inline]
fn calculate_set_sizes_scalar(freq_a: &[u8], freq_b: &[u8]) -> (usize, usize) {
    let mut intersection_size = 0;
    let mut union_size = 0;

    let len = std::cmp::min(freq_a.len(), freq_b.len());
    for i in 0..len {
        let a_count = freq_a[i];
        let b_count = freq_b[i];
        
        // For intersection, take the minimum of the two frequencies
        intersection_size += std::cmp::min(a_count, b_count) as usize;
        
        // For union, take the maximum of the two frequencies
        union_size += std::cmp::max(a_count, b_count) as usize;
    }

    (intersection_size, union_size)
}

/// AVX2 implementation for calculating set sizes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn calculate_set_sizes_avx2(freq_a: &[u8], freq_b: &[u8]) -> (usize, usize) {
    use std::arch::x86_64::*;
    
    let mut intersection_sum = 0;
    let mut union_sum = 0;
    
    // Process 32 bytes at a time with AVX2
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 32) * 32; // Ensure we only process complete 32-byte chunks
    
    for offset in (0..len).step_by(32) {
        // Load 32 bytes from each frequency array
        let a_chunk = _mm256_loadu_si256(ptr_a.add(offset) as *const __m256i);
        let b_chunk = _mm256_loadu_si256(ptr_b.add(offset) as *const __m256i);
        
        // Calculate intersection (minimum of each pair of values)
        let intersection = _mm256_min_epu8(a_chunk, b_chunk);
        
        // Calculate union (maximum of each pair of values)
        let union = _mm256_max_epu8(a_chunk, b_chunk);
        
        // Sum the values (horizontal sum)
        let zero = _mm256_setzero_si256();
        
        // For the intersection: use _mm256_sad_epu8 which gives sum of absolute differences
        let sad_intersection = _mm256_sad_epu8(intersection, zero);
        
        // Extract and sum the four 64-bit values
        let intersection_chunk_sum = 
            _mm256_extract_epi64(sad_intersection, 0) +
            _mm256_extract_epi64(sad_intersection, 1) +
            _mm256_extract_epi64(sad_intersection, 2) +
            _mm256_extract_epi64(sad_intersection, 3);
        
        // Repeat for the union
        let sad_union = _mm256_sad_epu8(union, zero);
        let union_chunk_sum = 
            _mm256_extract_epi64(sad_union, 0) +
            _mm256_extract_epi64(sad_union, 1) +
            _mm256_extract_epi64(sad_union, 2) +
            _mm256_extract_epi64(sad_union, 3);
        
        // Add to running totals
        intersection_sum += intersection_chunk_sum as usize;
        union_sum += union_chunk_sum as usize;
    }
    
    // Handle remaining bytes with scalar implementation
    if len < freq_a.len() {
        let remaining_intersection_sum;
        let remaining_union_sum;
        (remaining_intersection_sum, remaining_union_sum) = 
            calculate_set_sizes_scalar(&freq_a[len..], &freq_b[len..]);
        
        intersection_sum += remaining_intersection_sum;
        union_sum += remaining_union_sum;
    }
    
    (intersection_sum, union_sum)
}

/// SSE2 implementation for calculating set sizes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn calculate_set_sizes_sse2(freq_a: &[u8], freq_b: &[u8]) -> (usize, usize) {
    use std::arch::x86_64::*;
    
    let mut intersection_sum = 0;
    let mut union_sum = 0;
    
    // Process 16 bytes at a time with SSE2
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 16) * 16; // Ensure we only process complete 16-byte chunks
    
    for offset in (0..len).step_by(16) {
        // Load 16 bytes from each frequency array
        let a_chunk = _mm_loadu_si128(ptr_a.add(offset) as *const __m128i);
        let b_chunk = _mm_loadu_si128(ptr_b.add(offset) as *const __m128i);
        
        // Calculate intersection (minimum of each pair of values)
        let intersection = _mm_min_epu8(a_chunk, b_chunk);
        
        // Calculate union (maximum of each pair of values)
        let union = _mm_max_epu8(a_chunk, b_chunk);
        
        // Sum the values (horizontal sum)
        let zero = _mm_setzero_si128();
        
        // For the intersection: use _mm_sad_epu8 which gives sum of absolute differences
        let sad_intersection = _mm_sad_epu8(intersection, zero);
        
        // Extract and sum the two 64-bit values
        let intersection_chunk_sum = 
            _mm_extract_epi64(sad_intersection, 0) +
            _mm_extract_epi64(sad_intersection, 1);
        
        // Repeat for the union
        let sad_union = _mm_sad_epu8(union, zero);
        let union_chunk_sum = 
            _mm_extract_epi64(sad_union, 0) +
            _mm_extract_epi64(sad_union, 1);
        
        // Add to running totals
        intersection_sum += intersection_chunk_sum as usize;
        union_sum += union_chunk_sum as usize;
    }
    
    // Handle remaining bytes with scalar implementation
    if len < freq_a.len() {
        let remaining_intersection_sum;
        let remaining_union_sum;
        (remaining_intersection_sum, remaining_union_sum) = 
            calculate_set_sizes_scalar(&freq_a[len..], &freq_b[len..]);
        
        intersection_sum += remaining_intersection_sum;
        union_sum += remaining_union_sum;
    }
    
    (intersection_sum, union_sum)
}

/// NEON implementation for calculating set sizes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn calculate_set_sizes_neon(freq_a: &[u8], freq_b: &[u8]) -> (usize, usize) {
    use std::arch::aarch64::*;
    
    let mut intersection_sum = 0;
    let mut union_sum = 0;
    
    // Process 16 bytes at a time with NEON
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 16) * 16; // Ensure we only process complete 16-byte chunks
    
    for offset in (0..len).step_by(16) {
        // Load 16 bytes from each frequency array
        let a_chunk = vld1q_u8(ptr_a.add(offset));
        let b_chunk = vld1q_u8(ptr_b.add(offset));
        
        // Calculate intersection (minimum of each pair of values)
        let intersection = vminq_u8(a_chunk, b_chunk);
        
        // Calculate union (maximum of each pair of values)
        let union = vmaxq_u8(a_chunk, b_chunk);
        
        // Sum the values using pairwise addition
        let intersection_sum16 = vpaddlq_u8(intersection); // Sum pairs of u8 -> u16
        let intersection_sum32 = vpaddlq_u16(intersection_sum16); // Sum pairs of u16 -> u32
        let intersection_sum64 = vpaddlq_u32(intersection_sum32); // Sum pairs of u32 -> u64
        
        // Extract and sum the two 64-bit values
        let intersection_chunk_sum = 
            vgetq_lane_u64(intersection_sum64, 0) +
            vgetq_lane_u64(intersection_sum64, 1);
        
        // Repeat for the union
        let union_sum16 = vpaddlq_u8(union);
        let union_sum32 = vpaddlq_u16(union_sum16);
        let union_sum64 = vpaddlq_u32(union_sum32);
        let union_chunk_sum = 
            vgetq_lane_u64(union_sum64, 0) +
            vgetq_lane_u64(union_sum64, 1);
        
        // Add to running totals
        intersection_sum += intersection_chunk_sum as usize;
        union_sum += union_chunk_sum as usize;
    }
    
    // Handle remaining bytes with scalar implementation
    if len < freq_a.len() {
        let remaining_intersection_sum;
        let remaining_union_sum;
        (remaining_intersection_sum, remaining_union_sum) = 
            calculate_set_sizes_scalar(&freq_a[len..], &freq_b[len..]);
        
        intersection_sum += remaining_intersection_sum;
        union_sum += remaining_union_sum;
    }
    
    (intersection_sum, union_sum)
}
/// Calculates cosine similarity between two strings using SIMD acceleration,
/// with full Unicode support for languages like Arabic.
///
/// Cosine similarity measures the cosine of the angle between two non-zero vectors,
/// representing the character frequencies in each string. It ranges from -1 to 1,
/// with 1 meaning identical, 0 meaning orthogonal, and -1 meaning opposite.
/// For text similarity, the range is typically 0 to 1.
///
/// # Arguments
/// * `a` - First string
/// * `b` - Second string
///
/// # Returns
/// A float value between 0.0 (completely different) and 1.0 (identical)
pub fn simd_cosine(a: &str, b: &str) -> f64 {
    // Early returns for empty strings
    if a.is_empty() && b.is_empty() {
        return 1.0; // Two empty strings are considered identical
    }
    if a.is_empty() || b.is_empty() {
        return 0.0; // If only one is empty, they have no similarity
    }

    // Step 1: Extract Unicode characters from both strings
    let chars_a: Vec<char> = a.chars().collect();
    let chars_b: Vec<char> = b.chars().collect();

    // Step 2: Create a unified set of all unique characters
    let mut unique_chars = std::collections::HashSet::new();
    for &c in &chars_a {
        unique_chars.insert(c);
    }
    for &c in &chars_b {
        unique_chars.insert(c);
    }

    // For small character sets, use scalar implementation
    if unique_chars.len() <= 8 {
        return calculate_cosine_scalar(chars_a, chars_b);
    }

    // Step 3: Create mapping from characters to indices
    let mut char_to_index = std::collections::HashMap::new();
    let unique_chars_vec: Vec<char> = unique_chars.into_iter().collect();
    for (i, &c) in unique_chars_vec.iter().enumerate() {
        char_to_index.insert(c, i);
    }

    // Step 4: Build frequency vectors using the mapping
    // Resize to the next multiple of the SIMD width for optimal processing
    let array_size = round_to_simd_boundary(unique_chars_vec.len());
    
    // Use u16 instead of u8 to allow for higher frequencies and to avoid overflow
    // during multiplication when computing dot product
    let mut freq_a = vec![0u16; array_size];
    let mut freq_b = vec![0u16; array_size];

    // Populate frequency arrays
    for &c in &chars_a {
        let idx = *char_to_index.get(&c).unwrap();
        if freq_a[idx] < u16::MAX - 1 { // Prevent overflow
            freq_a[idx] += 1;
        }
    }

    for &c in &chars_b {
        let idx = *char_to_index.get(&c).unwrap();
        if freq_b[idx] < u16::MAX - 1 { // Prevent overflow
            freq_b[idx] += 1;
        }
    }

    // Step 5: Calculate dot product and magnitudes using SIMD
    let (dot_product, magnitude_a, magnitude_b) = calculate_vector_operations(&freq_a, &freq_b);

    // Step 6: Calculate cosine similarity
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0 // Avoid division by zero
    } else {
        // Clamp to [0, 1] to handle any floating-point precision issues
        (dot_product / (magnitude_a * magnitude_b)).clamp(0.0, 1.0)
    }
}

/// Fast scalar implementation of cosine similarity for small character sets
#[inline]
fn calculate_cosine_scalar(chars_a: Vec<char>, chars_b: Vec<char>) -> f64 {
    // Build frequency maps
    let mut freq_a = std::collections::HashMap::new();
    let mut freq_b = std::collections::HashMap::new();
    
    for c in chars_a {
        *freq_a.entry(c).or_insert(0) += 1;
    }
    
    for c in chars_b {
        *freq_b.entry(c).or_insert(0) += 1;
    }
    
    // Calculate dot product and magnitudes
    let mut dot_product = 0;
    let mut magnitude_a_squared = 0;
    let mut magnitude_b_squared = 0;
    
    // Get all unique characters from both texts
    let mut all_chars = std::collections::HashSet::new();
    all_chars.extend(freq_a.keys());
    all_chars.extend(freq_b.keys());
    
    for &c in &all_chars {
        let count_a = *freq_a.get(&c).unwrap_or(&0);
        let count_b = *freq_b.get(&c).unwrap_or(&0);
        
        dot_product += count_a * count_b;
        magnitude_a_squared += count_a * count_a;
        magnitude_b_squared += count_b * count_b;
    }
    
    // Calculate similarity
    if magnitude_a_squared == 0 || magnitude_b_squared == 0 {
        0.0
    } else {
        let magnitude_a = (magnitude_a_squared as f64).sqrt();
        let magnitude_b = (magnitude_b_squared as f64).sqrt();
        (dot_product as f64 / (magnitude_a * magnitude_b)).clamp(0.0, 1.0)
    }
}

/// Calculates dot product and magnitudes for cosine similarity using SIMD.
/// Returns a tuple of (dot_product, magnitude_a, magnitude_b) as f64 values.
#[inline]
fn calculate_vector_operations(freq_a: &[u16], freq_b: &[u16]) -> (f64, f64, f64) {
    // Dispatch to the appropriate SIMD implementation based on hardware support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { calculate_vector_operations_avx2(freq_a, freq_b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { calculate_vector_operations_sse2(freq_a, freq_b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { calculate_vector_operations_neon(freq_a, freq_b) };
    }

    // Fallback to scalar implementation
    calculate_vector_operations_scalar(freq_a, freq_b)
}

/// Scalar fallback implementation for calculating vector operations.
#[inline]
fn calculate_vector_operations_scalar(freq_a: &[u16], freq_b: &[u16]) -> (f64, f64, f64) {
    let mut dot_product: u64 = 0;
    let mut magnitude_a_squared: u64 = 0;
    let mut magnitude_b_squared: u64 = 0;

    for i in 0..freq_a.len() {
        let a_val = freq_a[i] as u64;
        let b_val = freq_b[i] as u64;
        
        dot_product += a_val * b_val;
        magnitude_a_squared += a_val * a_val;
        magnitude_b_squared += b_val * b_val;
    }

    // Convert to floating point for final calculations
    let dot_product_f64 = dot_product as f64;
    let magnitude_a = (magnitude_a_squared as f64).sqrt();
    let magnitude_b = (magnitude_b_squared as f64).sqrt();

    (dot_product_f64, magnitude_a, magnitude_b)
}

/// AVX2 implementation for calculating vector operations.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn calculate_vector_operations_avx2(freq_a: &[u16], freq_b: &[u16]) -> (f64, f64, f64) {
    use std::arch::x86_64::*;
    
    // Initialize accumulators
    let mut dot_product_acc = _mm256_setzero_si256();
    let mut magnitude_a_acc = _mm256_setzero_si256();
    let mut magnitude_b_acc = _mm256_setzero_si256();
    
    // Process 16 elements (16 u16 values = 32 bytes) at a time
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 16) * 16; // Ensure we process complete 16-element chunks
    
    for offset in (0..len).step_by(16) {
        // Load 16 u16 elements (32 bytes) from each array
        let a_chunk = _mm256_loadu_si256(ptr_a.add(offset) as *const __m256i);
        let b_chunk = _mm256_loadu_si256(ptr_b.add(offset) as *const __m256i);
        
        // For dot product: multiply corresponding elements
        // _mm256_mullo_epi16 multiplies 16-bit integers and keeps the lower 16 bits
        let dot_chunk = _mm256_mullo_epi16(a_chunk, b_chunk);
        
        // For magnitudes: square each element
        let a_squared = _mm256_mullo_epi16(a_chunk, a_chunk);
        let b_squared = _mm256_mullo_epi16(b_chunk, b_chunk);
        
        // Accumulate results
        // Convert 16-bit products to 32-bit to avoid overflow during summation
        
        // Process dot product accumulation
        // Convert to 32-bit integers (handling 8 elements at a time)
        let dot_low = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dot_chunk, 0));
        let dot_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dot_chunk, 1));
        // Add to accumulator
        dot_product_acc = _mm256_add_epi32(dot_product_acc, dot_low);
        dot_product_acc = _mm256_add_epi32(dot_product_acc, dot_high);
        
        // Process magnitude A accumulation
        let a_sq_low = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a_squared, 0));
        let a_sq_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a_squared, 1));
        magnitude_a_acc = _mm256_add_epi32(magnitude_a_acc, a_sq_low);
        magnitude_a_acc = _mm256_add_epi32(magnitude_a_acc, a_sq_high);
        
        // Process magnitude B accumulation
        let b_sq_low = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b_squared, 0));
        let b_sq_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b_squared, 1));
        magnitude_b_acc = _mm256_add_epi32(magnitude_b_acc, b_sq_low);
        magnitude_b_acc = _mm256_add_epi32(magnitude_b_acc, b_sq_high);
    }
    
    // Horizontal sum of each accumulator (dot product, magnitude A, magnitude B)
    // Sum across the 8 lanes of each 256-bit register
    
    // Extract 32-bit integers and sum them
    let mut dot_product: u64 = 0;
    let mut magnitude_a: u64 = 0;
    let mut magnitude_b: u64 = 0;
    
    // Store accumulators to memory and sum them
    let mut dp_arr = [0u32; 8];
    let mut ma_arr = [0u32; 8];
    let mut mb_arr = [0u32; 8];
    
    _mm256_storeu_si256(dp_arr.as_mut_ptr() as *mut __m256i, dot_product_acc);
    _mm256_storeu_si256(ma_arr.as_mut_ptr() as *mut __m256i, magnitude_a_acc);
    _mm256_storeu_si256(mb_arr.as_mut_ptr() as *mut __m256i, magnitude_b_acc);
    
    for i in 0..8 {
        dot_product += dp_arr[i] as u64;
        magnitude_a += ma_arr[i] as u64;
        magnitude_b += mb_arr[i] as u64;
    }
    
    // Handle remaining elements with scalar implementation
    if len < freq_a.len() {
        let remaining_dot_product: u64;
        let remaining_magnitude_a: u64;
        let remaining_magnitude_b: u64;
        
        let (remaining_dot_product_f64, remaining_magnitude_a_f64, remaining_magnitude_b_f64) = 
            calculate_vector_operations_scalar(&freq_a[len..], &freq_b[len..]);
        
        remaining_dot_product = remaining_dot_product_f64 as u64;
        remaining_magnitude_a = (remaining_magnitude_a_f64 * remaining_magnitude_a_f64) as u64;
        remaining_magnitude_b = (remaining_magnitude_b_f64 * remaining_magnitude_b_f64) as u64;
        
        dot_product += remaining_dot_product;
        magnitude_a += remaining_magnitude_a;
        magnitude_b += remaining_magnitude_b;
    }
    
    // Convert to floating point for final calculations
    let dot_product_f64 = dot_product as f64;
    let magnitude_a_f64 = (magnitude_a as f64).sqrt();
    let magnitude_b_f64 = (magnitude_b as f64).sqrt();
    
    (dot_product_f64, magnitude_a_f64, magnitude_b_f64)
}

/// SSE2 implementation for calculating vector operations.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn calculate_vector_operations_sse2(freq_a: &[u16], freq_b: &[u16]) -> (f64, f64, f64) {
    use std::arch::x86_64::*;
    
    // Initialize accumulators
    let mut dot_product_acc = _mm_setzero_si128();
    let mut magnitude_a_acc = _mm_setzero_si128();
    let mut magnitude_b_acc = _mm_setzero_si128();
    
    // Process 8 elements (8 u16 values = 16 bytes) at a time
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 8) * 8; // Ensure we process complete 8-element chunks
    
    for offset in (0..len).step_by(8) {
        // Load 8 u16 elements (16 bytes) from each array
        let a_chunk = _mm_loadu_si128(ptr_a.add(offset) as *const __m128i);
        let b_chunk = _mm_loadu_si128(ptr_b.add(offset) as *const __m128i);
        
        // For dot product: multiply corresponding elements
        let dot_chunk = _mm_mullo_epi16(a_chunk, b_chunk);
        
        // For magnitudes: square each element
        let a_squared = _mm_mullo_epi16(a_chunk, a_chunk);
        let b_squared = _mm_mullo_epi16(b_chunk, b_chunk);
        
        // Accumulate results
        // Convert 16-bit products to 32-bit to avoid overflow during summation
        
        // Process dot product accumulation (first 4 elements)
        let dot_low = _mm_cvtepu16_epi32(dot_chunk);
        // Extract high 64 bits and convert to 32-bit integers
        let dot_high = _mm_cvtepu16_epi32(_mm_srli_si128(dot_chunk, 8));
        // Add to accumulator
        dot_product_acc = _mm_add_epi32(dot_product_acc, dot_low);
        dot_product_acc = _mm_add_epi32(dot_product_acc, dot_high);
        
        // Process magnitude A accumulation
        let a_sq_low = _mm_cvtepu16_epi32(a_squared);
        let a_sq_high = _mm_cvtepu16_epi32(_mm_srli_si128(a_squared, 8));
        magnitude_a_acc = _mm_add_epi32(magnitude_a_acc, a_sq_low);
        magnitude_a_acc = _mm_add_epi32(magnitude_a_acc, a_sq_high);
        
        // Process magnitude B accumulation
        let b_sq_low = _mm_cvtepu16_epi32(b_squared);
        let b_sq_high = _mm_cvtepu16_epi32(_mm_srli_si128(b_squared, 8));
        magnitude_b_acc = _mm_add_epi32(magnitude_b_acc, b_sq_low);
        magnitude_b_acc = _mm_add_epi32(magnitude_b_acc, b_sq_high);
    }
    
    // Horizontal sum of each accumulator
    // Store accumulators to memory and sum them
    let mut dp_arr = [0u32; 4];
    let mut ma_arr = [0u32; 4];
    let mut mb_arr = [0u32; 4];
    
    _mm_storeu_si128(dp_arr.as_mut_ptr() as *mut __m128i, dot_product_acc);
    _mm_storeu_si128(ma_arr.as_mut_ptr() as *mut __m128i, magnitude_a_acc);
    _mm_storeu_si128(mb_arr.as_mut_ptr() as *mut __m128i, magnitude_b_acc);
    
    let mut dot_product: u64 = 0;
    let mut magnitude_a: u64 = 0;
    let mut magnitude_b: u64 = 0;
    
    for i in 0..4 {
        dot_product += dp_arr[i] as u64;
        magnitude_a += ma_arr[i] as u64;
        magnitude_b += mb_arr[i] as u64;
    }
    
    // Handle remaining elements with scalar implementation
    if len < freq_a.len() {
        let remaining_dot_product: u64;
        let remaining_magnitude_a: u64;
        let remaining_magnitude_b: u64;
        
        let (remaining_dot_product_f64, remaining_magnitude_a_f64, remaining_magnitude_b_f64) = 
            calculate_vector_operations_scalar(&freq_a[len..], &freq_b[len..]);
        
        remaining_dot_product = remaining_dot_product_f64 as u64;
        remaining_magnitude_a = (remaining_magnitude_a_f64 * remaining_magnitude_a_f64) as u64;
        remaining_magnitude_b = (remaining_magnitude_b_f64 * remaining_magnitude_b_f64) as u64;
        
        dot_product += remaining_dot_product;
        magnitude_a += remaining_magnitude_a;
        magnitude_b += remaining_magnitude_b;
    }
    
    // Convert to floating point for final calculations
    let dot_product_f64 = dot_product as f64;
    let magnitude_a_f64 = (magnitude_a as f64).sqrt();
    let magnitude_b_f64 = (magnitude_b as f64).sqrt();
    
    (dot_product_f64, magnitude_a_f64, magnitude_b_f64)
}

/// NEON implementation for calculating vector operations.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn calculate_vector_operations_neon(freq_a: &[u16], freq_b: &[u16]) -> (f64, f64, f64) {
    use std::arch::aarch64::*;
    
    // Initialize accumulators for 32-bit sums
    let mut dot_acc = vdupq_n_u32(0);
    let mut mag_a_acc = vdupq_n_u32(0);
    let mut mag_b_acc = vdupq_n_u32(0);
    
    // Process 8 elements (8 u16 values = 16 bytes) at a time
    let ptr_a = freq_a.as_ptr();
    let ptr_b = freq_b.as_ptr();
    
    let len = (freq_a.len() / 8) * 8; // Ensure we process complete 8-element chunks
    
    for offset in (0..len).step_by(8) {
        // Load 8 u16 elements from each array
        let a_chunk = vld1q_u16(ptr_a.add(offset));
        let b_chunk = vld1q_u16(ptr_b.add(offset));
        
        // Calculate dot product of 16-bit values
        // vmull_u16 multiplies corresponding elements of two vectors and 
        // produces a 32-bit result, which helps avoid overflow
        
        // Process the first 4 elements
        let dot_low = vmull_u16(vget_low_u16(a_chunk), vget_low_u16(b_chunk));
        // Process the next 4 elements
        let dot_high = vmull_u16(vget_high_u16(a_chunk), vget_high_u16(b_chunk));
        
        // Square terms for magnitudes
        let a_sq_low = vmull_u16(vget_low_u16(a_chunk), vget_low_u16(a_chunk));
        let a_sq_high = vmull_u16(vget_high_u16(a_chunk), vget_high_u16(a_chunk));
        
        let b_sq_low = vmull_u16(vget_low_u16(b_chunk), vget_low_u16(b_chunk));
        let b_sq_high = vmull_u16(vget_high_u16(b_chunk), vget_high_u16(b_chunk));
        
        // Accumulate results
        dot_acc = vaddq_u32(dot_acc, vaddq_u32(dot_low, dot_high));
        mag_a_acc = vaddq_u32(mag_a_acc, vaddq_u32(a_sq_low, a_sq_high));
        mag_b_acc = vaddq_u32(mag_b_acc, vaddq_u32(b_sq_low, b_sq_high));
    }
    
    // Horizontal sum of all lanes in each accumulator
    // Extract and sum the four 32-bit values
    let dot_product: u64 = 
        vgetq_lane_u32(dot_acc, 0) as u64 +
        vgetq_lane_u32(dot_acc, 1) as u64 +
        vgetq_lane_u32(dot_acc, 2) as u64 +
        vgetq_lane_u32(dot_acc, 3) as u64;
        
    let magnitude_a: u64 = 
        vgetq_lane_u32(mag_a_acc, 0) as u64 +
        vgetq_lane_u32(mag_a_acc, 1) as u64 +
        vgetq_lane_u32(mag_a_acc, 2) as u64 +
        vgetq_lane_u32(mag_a_acc, 3) as u64;
        
    let magnitude_b: u64 = 
        vgetq_lane_u32(mag_b_acc, 0) as u64 +
        vgetq_lane_u32(mag_b_acc, 1) as u64 +
        vgetq_lane_u32(mag_b_acc, 2) as u64 +
        vgetq_lane_u32(mag_b_acc, 3) as u64;
    
    // Handle remaining elements with scalar implementation
    let mut final_dot_product = dot_product;
    let mut final_magnitude_a = magnitude_a;
    let mut final_magnitude_b = magnitude_b;
    
    if len < freq_a.len() {
        let (remaining_dot_product_f64, remaining_magnitude_a_f64, remaining_magnitude_b_f64) = 
            calculate_vector_operations_scalar(&freq_a[len..], &freq_b[len..]);
        
        final_dot_product += remaining_dot_product_f64 as u64;
        final_magnitude_a += (remaining_magnitude_a_f64 * remaining_magnitude_a_f64) as u64;
        final_magnitude_b += (remaining_magnitude_b_f64 * remaining_magnitude_b_f64) as u64;
    }
    
    // Convert to floating point for final calculations
    let dot_product_f64 = final_dot_product as f64;
    let magnitude_a_f64 = (final_magnitude_a as f64).sqrt();
    let magnitude_b_f64 = (final_magnitude_b as f64).sqrt();
    
    (dot_product_f64, magnitude_a_f64, magnitude_b_f64)
}