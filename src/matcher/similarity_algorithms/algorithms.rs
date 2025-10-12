use std::arch::x86_64::*;
use crate::utils::simd::{simd_jaccard, simd_cosine};
use crate::error::Result;
use crate::config::MatchingConfig;
use crate::matcher::types::SimilarityMetric;
use crate::types::CompactCharFrequencies;
/// The SimilarityAlgorithm trait defines the interface for comparing text similarity.
/// Each implementation provides a different way to measure how similar two pieces of text are.
/// All similarity scores are normalized between 0.0 (completely different) and 1.0 (identical).
pub trait SimilarityAlgorithm {
    // Entry point for all algorithms
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64>;
}

/// ExactMatcher implements exact string matching with optional storage-based optimization.
/// When storage is available, it can use indexed lookups instead of linear scanning.

pub struct ExactMatcher;

impl ExactMatcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl SimilarityAlgorithm for ExactMatcher {
    // Only implement compare_texts for validation path
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        // For text comparison, we can use direct string comparison
        // or bytes - both are equally fast for exact match
        Ok(if source == target { 1.0 } else { 0.0 })
    }
}

/// JaccardMatcher implements similarity comparison using the Jaccard index,
/// which measures similarity as the size of the intersection divided by the size
/// of the union of character sets.
pub struct JaccardMatcher {
}

impl JaccardMatcher {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Compute Jaccard similarity using SIMD-optimized operations
    /// This is the main entry point that handles CPU feature detection
    #[inline(always)]
    pub fn jaccard_from_features(
        &self, 
        source: &CompactCharFrequencies, 
        target: &CompactCharFrequencies
    ) -> f64 {
        // Ultra-fast rejection: check bitmap overlap (1 CPU instruction)
        if source.char_bitmap & target.char_bitmap == 0 {
            return 0.0;  // No character overlap = Jaccard is 0
        }
        
        // Dispatch to appropriate implementation based on CPU features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { 
                    simd_jaccard_avx2(source, target) as f64 
                }
            } else if is_x86_feature_detected!("sse2") {
                unsafe { 
                    simd_jaccard_sse2(source, target) as f64 
                }
            } else {
                jaccard_scalar(source, target) as f64
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            jaccard_scalar(source, target) as f64
        }
    }
}
/// AVX2 implementation - processes 32 bytes at a time
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn simd_jaccard_avx2(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> f32 {
    // Ensure alignment for optimal performance (debug mode only)
    debug_assert!(source.counts.as_ptr() as usize % 32 == 0);
    debug_assert!(target.counts.as_ptr() as usize % 32 == 0);
    
    // Load first 32 bytes of each frequency array
    let source_vec1 = _mm256_loadu_si256(
        source.counts[0..32].as_ptr() as *const __m256i
    );
    let target_vec1 = _mm256_loadu_si256(
        target.counts[0..32].as_ptr() as *const __m256i
    );
    
    // Load second 32 bytes
    let source_vec2 = _mm256_loadu_si256(
        source.counts[32..64].as_ptr() as *const __m256i
    );
    let target_vec2 = _mm256_loadu_si256(
        target.counts[32..64].as_ptr() as *const __m256i
    );
    
    // Calculate min(source, target) for intersection - 32 operations in 1 instruction!
    let min_vec1 = _mm256_min_epu8(source_vec1, target_vec1);
    let min_vec2 = _mm256_min_epu8(source_vec2, target_vec2);
    
    // Calculate max(source, target) for union
    let max_vec1 = _mm256_max_epu8(source_vec1, target_vec1);
    let max_vec2 = _mm256_max_epu8(source_vec2, target_vec2);
    
    // Sum all bytes using SAD trick
    let intersection = sum_bytes_avx2(min_vec1) + sum_bytes_avx2(min_vec2);
    let union = sum_bytes_avx2(max_vec1) + sum_bytes_avx2(max_vec2);
    
    // Calculate Jaccard similarity
    if union == 0 {
        1.0  // Both empty = identical
    } else {
        intersection as f32 / union as f32
    }
}

/// Helper function to sum all bytes in a 256-bit vector using SAD trick
#[target_feature(enable = "avx2")]
unsafe fn sum_bytes_avx2(vec: __m256i) -> u32 {
    // Create zero vector for SAD operation
    let zeros = _mm256_setzero_si256();
    
    // SAD (Sum of Absolute Differences) against zero = sum of all bytes
    // This produces 4 x 64-bit partial sums
    let sad = _mm256_sad_epu8(vec, zeros);
    
    // Extract the four 64-bit sums and add them
    // Each contains sum of 8 consecutive bytes
    let sum0 = _mm256_extract_epi64(sad, 0) as u32;
    let sum1 = _mm256_extract_epi64(sad, 1) as u32;
    let sum2 = _mm256_extract_epi64(sad, 2) as u32;
    let sum3 = _mm256_extract_epi64(sad, 3) as u32;
    
    sum0 + sum1 + sum2 + sum3
}

/// SSE2 implementation - processes 16 bytes at a time (older CPUs)
#[target_feature(enable = "sse2")]
#[cfg(target_arch = "x86_64")]
unsafe fn simd_jaccard_sse2(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> f32 {
    use std::arch::x86_64::*;
    
    let mut intersection_sum = 0u32;
    let mut union_sum = 0u32;
    
    // Process in 16-byte chunks (4 iterations for 64 bytes)
    for i in 0..4 {
        let offset = i * 16;
        
        // Load 16 bytes from each array
        let source_vec = _mm_loadu_si128(
            source.counts[offset..offset+16].as_ptr() as *const __m128i
        );
        let target_vec = _mm_loadu_si128(
            target.counts[offset..offset+16].as_ptr() as *const __m128i
        );
        
        // Min/Max operations
        let min_vec = _mm_min_epu8(source_vec, target_vec);
        let max_vec = _mm_max_epu8(source_vec, target_vec);
        
        // Sum bytes using SAD
        intersection_sum += sum_bytes_sse2(min_vec);
        union_sum += sum_bytes_sse2(max_vec);
    }
    
    if union_sum == 0 {
        1.0
    } else {
        intersection_sum as f32 / union_sum as f32
    }
}

/// Helper for SSE2 byte summing
#[target_feature(enable = "sse2")]
unsafe fn sum_bytes_sse2(vec: __m128i) -> u32 {
    let zeros = _mm_setzero_si128();
    let sad = _mm_sad_epu8(vec, zeros);
    
    // SSE2 produces 2 x 64-bit sums
    let sum0 = _mm_extract_epi64(sad, 0) as u32;
    let sum1 = _mm_extract_epi64(sad, 1) as u32;
    
    sum0 + sum1
}

/// Scalar fallback for CPUs without SIMD support
#[inline(always)]
fn jaccard_scalar(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> f32 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    
    // Simple loop - compiler may still auto-vectorize this
    for i in 0..64 {
        let s = source.counts[i] as u32;
        let t = target.counts[i] as u32;
        
        intersection += s.min(t);
        union += s.max(t);
    }
    
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

impl SimilarityAlgorithm for JaccardMatcher {

    
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        // Just pass bytes to simd_jaccard
        Ok(simd_jaccard(source.as_bytes(), target.as_bytes()))
    }
}
/// CosineMatcher implements cosine similarity comparison with term frequency weighting.
/// It maintains a cache of term vectors to avoid recomputing frequencies.
pub struct CosineMatcher;

impl CosineMatcher {
    pub fn new() -> Self {
        Self {}
    }
        
    #[inline(always)]
    pub fn cosine_from_features(
        &self,
        source: &CompactCharFrequencies,
        target: &CompactCharFrequencies,
    ) -> f64 {
        // Fast rejection: orthogonal vectors if no character overlap
        if source.char_bitmap & target.char_bitmap == 0 {
            return 0.0;
        }
        
        // Fast rejection: zero magnitude means undefined cosine
        if source.magnitude_squared == 0.0 || target.magnitude_squared == 0.0 {
            return 0.0;
        }
        
        // Calculate dot product using SIMD
        #[cfg(target_arch = "x86_64")]
        {
            let dot_product = if is_x86_feature_detected!("avx2") {
                unsafe { compute_dot_product_avx2(source, target) }
            } else if is_x86_feature_detected!("sse2") {
                unsafe { compute_dot_product_sse2(source, target) }
            } else {
                compute_dot_product_scalar(source, target)
            };
            
            // Cosine = dot_product / sqrt(mag_a * mag_b)
            let magnitude_product = (source.magnitude_squared * target.magnitude_squared) as f64;
            (dot_product as f64 / magnitude_product.sqrt()).min(1.0)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            let dot_product = compute_dot_product_scalar(source, target);
            let magnitude_product = (source.magnitude_squared * target.magnitude_squared) as f64;
            (dot_product as f64 / magnitude_product.sqrt()).min(1.0)
        }
    }
}

/// AVX2 dot product - processes 16 bytes at a time with expansion to 16-bit
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn compute_dot_product_avx2(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> u32 {
    let mut dot_product = 0u32;
    
    // Process in 16-byte chunks (expands to 32 bytes when converting to 16-bit)
    for chunk_idx in 0..4 {
        let offset = chunk_idx * 16;
        
        // Load 16 bytes from each array
        let source_bytes = _mm_loadu_si128(
            source.counts[offset..offset+16].as_ptr() as *const __m128i
        );
        let target_bytes = _mm_loadu_si128(
            target.counts[offset..offset+16].as_ptr() as *const __m128i
        );
        
        // Expand unsigned bytes to 16-bit integers (zero-extension)
        // This prevents overflow when multiplying
        let source_16 = _mm256_cvtepu8_epi16(source_bytes);
        let target_16 = _mm256_cvtepu8_epi16(target_bytes);
        
        // Multiply pairs (16-bit × 16-bit = 16-bit with saturation)
        let products = _mm256_mullo_epi16(source_16, target_16);
        
        // Sum all 16-bit products in this chunk
        dot_product += horizontal_sum_epi16_avx2(products);
    }
    
    dot_product
}

/// Helper to sum all 16-bit values in a 256-bit vector
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_epi16_avx2(vec: __m256i) -> u32 {
    // First, sum pairs of 16-bit values into 32-bit values
    // This prevents overflow during the reduction
    let sum_32 = _mm256_madd_epi16(vec, _mm256_set1_epi16(1));
    
    // Now we have 8 x 32-bit partial sums
    // Extract upper and lower 128-bit halves
    let lower = _mm256_castsi256_si128(sum_32);
    let upper = _mm256_extracti128_si256(sum_32, 1);
    
    // Add the halves
    let sum_128 = _mm_add_epi32(lower, upper);
    
    // Horizontal add the remaining 4 x 32-bit values
    let sum_64 = _mm_hadd_epi32(sum_128, sum_128);
    let sum_final = _mm_hadd_epi32(sum_64, sum_64);
    
    // Extract the final sum
    _mm_extract_epi32(sum_final, 0) as u32
}

/// SSE2 dot product - processes 8 bytes at a time with expansion
#[target_feature(enable = "sse2")]
#[cfg(target_arch = "x86_64")]
unsafe fn compute_dot_product_sse2(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> u32 {
    let mut dot_product = 0u32;
    
    // Process in 8-byte chunks (expands to 16 bytes as 16-bit)
    for chunk_idx in 0..8 {
        let offset = chunk_idx * 8;
        
        // Load 8 bytes from each array  
        let source_bytes = _mm_loadl_epi64(
            source.counts[offset..offset+8].as_ptr() as *const __m128i
        );
        let target_bytes = _mm_loadl_epi64(
            target.counts[offset..offset+8].as_ptr() as *const __m128i
        );
        
        // Unpack bytes to 16-bit (zero extension)
        let source_16 = _mm_unpacklo_epi8(source_bytes, _mm_setzero_si128());
        let target_16 = _mm_unpacklo_epi8(target_bytes, _mm_setzero_si128());
        
        // Multiply pairs
        let products = _mm_mullo_epi16(source_16, target_16);
        
        // Sum products
        dot_product += horizontal_sum_epi16_sse2(products);
    }
    
    dot_product
}

/// Helper for SSE2 horizontal sum
#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum_epi16_sse2(vec: __m128i) -> u32 {
    // Convert to 32-bit to avoid overflow
    let sum_32 = _mm_madd_epi16(vec, _mm_set1_epi16(1));
    
    // Shuffle and add to reduce
    let shuf = _mm_shuffle_epi32(sum_32, 0x0E);  // Swap high/low 64-bits
    let sums = _mm_add_epi32(sum_32, shuf);
    
    let shuf2 = _mm_shuffle_epi32(sums, 0x01);   // Swap within 64-bits  
    let final_sum = _mm_add_epi32(sums, shuf2);
    
    _mm_cvtsi128_si32(final_sum) as u32
}

/// Scalar fallback
#[inline(always)]
fn compute_dot_product_scalar(
    source: &CompactCharFrequencies,
    target: &CompactCharFrequencies
) -> u32 {
    let mut dot_product = 0u32;
    
    for i in 0..64 {
        dot_product += (source.counts[i] as u32) * (target.counts[i] as u32);
    }
    
    dot_product
}

impl SimilarityAlgorithm for CosineMatcher {
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        Ok(simd_cosine(source.as_bytes(), target.as_bytes()))
    }
}

/// LevenshteinMatcher implements similarity based on edit distance.
pub struct LevenshteinMatcher {
    max_distance: usize,
}

impl LevenshteinMatcher {

    pub fn with_config(max_distance: usize) -> Self {
        Self { max_distance }
    }
    
    pub fn levenshtein_distance(&self, source: &[u8], target: &[u8]) -> usize {
        let source_str = unsafe { std::str::from_utf8_unchecked(source) };
        let target_str = unsafe { std::str::from_utf8_unchecked(target) };
        
        let source_chars: Vec<char> = source_str.chars().collect();
        let target_chars: Vec<char> = target_str.chars().collect();
        
        let source_len = source_chars.len();
        let target_len = target_chars.len();
        
        // Quick check
        if source_len.abs_diff(target_len) > self.max_distance {
            return self.max_distance + 1;
        }
        
        // OPTIMIZATION: Only compute diagonal band of width 2*max_distance+1
        let mut prev_row = vec![usize::MAX; target_len + 1];
        let mut curr_row = vec![usize::MAX; target_len + 1];
        
        // Initialize first row within band
        for j in 0..=self.max_distance.min(target_len) {
            prev_row[j] = j;
        }
        
        for i in 1..=source_len {
            // Calculate band boundaries
            let start = i.saturating_sub(self.max_distance).max(1);
            let end = (i + self.max_distance).min(target_len);
            
            // Initialize first column value if in band
            if i <= self.max_distance {
                curr_row[0] = i;
            }
            
            let mut row_min = usize::MAX;
            
            // Only compute values within the diagonal band
            for j in start..=end {
                let cost = if source_chars[i-1] == target_chars[j-1] { 0 } else { 1 };
                
                let delete = if j < target_len { prev_row[j] } else { usize::MAX };
                let insert = if j > 0 { curr_row[j-1] } else { usize::MAX };
                let subst = if j > 0 { prev_row[j-1] } else { usize::MAX };
                
                if delete != usize::MAX || insert != usize::MAX || subst != usize::MAX {
                    curr_row[j] = [
                        delete.saturating_add(1),
                        insert.saturating_add(1),
                        subst.saturating_add(cost)
                    ].into_iter().min().unwrap();
                    
                    row_min = row_min.min(curr_row[j]);
                }
            }
            
            // Early termination
            if row_min > self.max_distance {
                return self.max_distance + 1;
            }
            
            std::mem::swap(&mut prev_row, &mut curr_row);
            curr_row.fill(usize::MAX);  // Reset for next iteration
        }
        
        prev_row[target_len]
    }
}

impl SimilarityAlgorithm for LevenshteinMatcher {
    
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        // Convert to bytes and call our method
        let distance = self.levenshtein_distance(source.as_bytes(), target.as_bytes());
        let max_len = source.len().max(target.len());
        
        if distance > self.max_distance {
            return Ok(0.0);
        }
        
        Ok(1.0 - (distance as f64 / max_len as f64))
    }
}

// algorithms.rs - VSAMatcher

pub struct VSAMatcher {
    min_term_weight: f64,
}

impl VSAMatcher {
    pub fn with_config(min_term_weight: f64) -> Self {
        Self { min_term_weight }
    }
    
    /// VSA using precomputed features with hybrid SIMD/scalar approach
    #[inline(always)]
    pub fn vsa_from_features(
        &self,
        source_freqs: &CompactCharFrequencies,
        target_freqs: &CompactCharFrequencies
    ) -> f64 {
        // Fast rejection: no overlap means orthogonal vectors
        if source_freqs.char_bitmap & target_freqs.char_bitmap == 0 {
            return 0.0;
        }
        
        // Handle empty cases
        let source_total = source_freqs.total_chars as f64;
        let target_total = target_freqs.total_chars as f64;
        
        if source_total == 0.0 || target_total == 0.0 {
            return if source_total == 0.0 && target_total == 0.0 { 1.0 } else { 0.0 };
        }
        
        // Compute weights for all 64 positions using SIMD or scalar
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.vsa_hybrid_avx2(source_freqs, target_freqs, source_total, target_total) }
            } else {
                self.vsa_scalar_optimized(source_freqs, target_freqs, source_total, target_total)
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.vsa_scalar_optimized(source_freqs, target_freqs, source_total, target_total)
        }
    }
    
    /// AVX2 hybrid implementation - vectorized weight calculation, scalar filtering
    #[target_feature(enable = "avx2")]
    unsafe fn vsa_hybrid_avx2(
        &self,
        source_freqs: &CompactCharFrequencies,
        target_freqs: &CompactCharFrequencies,
        source_total: f64,
        target_total: f64
    ) -> f64 {
        // Step 1: Compute all weights using SIMD
        // We'll compute weights for 8 positions at a time using AVX2
        let mut source_weights = [0.0_f64; 64];
        let mut target_weights = [0.0_f64; 64];
        
        // Process 8 bytes at a time, converting to doubles for weight calculation
        for chunk_idx in 0..8 {
            let offset = chunk_idx * 8;
            
            // Load 8 bytes and convert to doubles for precise weight calculation
            let source_bytes = _mm_loadl_epi64(
                source_freqs.counts[offset..offset+8].as_ptr() as *const __m128i
            );
            let target_bytes = _mm_loadl_epi64(
                target_freqs.counts[offset..offset+8].as_ptr() as *const __m128i
            );
            
            // Convert bytes to 32-bit integers first
            let source_32 = _mm256_cvtepu8_epi32(source_bytes);
            let target_32 = _mm256_cvtepu8_epi32(target_bytes);
            
            // Convert to doubles for division
            let source_f64 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(source_32, 0));
            let target_f64 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(target_32, 0));
            
            // Divide by totals to get weights
            let source_total_vec = _mm256_set1_pd(source_total);
            let target_total_vec = _mm256_set1_pd(target_total);
            
            let source_weights_vec = _mm256_div_pd(source_f64, source_total_vec);
            let target_weights_vec = _mm256_div_pd(target_f64, target_total_vec);
            
            // Store first 4 weights
            _mm256_storeu_pd(&mut source_weights[offset], source_weights_vec);
            _mm256_storeu_pd(&mut target_weights[offset], target_weights_vec);
            
            // Process next 4 bytes
            let source_32_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(source_32, 1));
            let target_32_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(target_32, 1));
            
            let source_weights_high = _mm256_div_pd(source_32_high, source_total_vec);
            let target_weights_high = _mm256_div_pd(target_32_high, target_total_vec);
            
            _mm256_storeu_pd(&mut source_weights[offset + 4], source_weights_high);
            _mm256_storeu_pd(&mut target_weights[offset + 4], target_weights_high);
        }
        
        // Step 2: Apply filtering and compute dot product (scalar)
        self.compute_filtered_vsa(&source_weights, &target_weights, source_freqs, target_freqs)
    }
    
    /// Scalar optimized implementation using fixed arrays
    fn vsa_scalar_optimized(
        &self,
        source_freqs: &CompactCharFrequencies,
        target_freqs: &CompactCharFrequencies,
        source_total: f64,
        target_total: f64
    ) -> f64 {
        // Compute weights for all positions
        let mut source_weights = [0.0_f64; 64];
        let mut target_weights = [0.0_f64; 64];
        
        for i in 0..64 {
            if source_freqs.counts[i] > 0 {
                source_weights[i] = source_freqs.counts[i] as f64 / source_total;
            }
            if target_freqs.counts[i] > 0 {
                target_weights[i] = target_freqs.counts[i] as f64 / target_total;
            }
        }
        
        self.compute_filtered_vsa(&source_weights, &target_weights, source_freqs, target_freqs)
    }
    
    /// Common filtering and final calculation
    #[inline(always)]
    fn compute_filtered_vsa(
        &self,
        source_weights: &[f64; 64],
        target_weights: &[f64; 64],
        source_freqs: &CompactCharFrequencies,
        target_freqs: &CompactCharFrequencies
    ) -> f64 {
        let mut dot_product = 0.0;
        let mut source_norm_sq = 0.0;
        let mut target_norm_sq = 0.0;
        
        // Use bitmap to only check positions where at least one text has the character
        let combined_bitmap = source_freqs.char_bitmap | target_freqs.char_bitmap;
        
        // Process each position where characters exist
        for i in 0..64 {
            // Quick check using bitmap
            if combined_bitmap & (1u64 << i) == 0 {
                continue;
            }
            
            let source_weight = source_weights[i];
            let target_weight = target_weights[i];
            
            // Apply min_term_weight filtering
            let source_passes = source_weight >= self.min_term_weight;
            let target_passes = target_weight >= self.min_term_weight;
            
            if source_passes {
                source_norm_sq += source_weight * source_weight;
            }
            
            if target_passes {
                target_norm_sq += target_weight * target_weight;
            }
            
            // Both must pass threshold and exist for dot product
            if source_passes && target_passes && source_weight > 0.0 && target_weight > 0.0 {
                dot_product += source_weight * target_weight;
            }
        }
        
        // Calculate final VSA (cosine similarity on filtered weights)
        let norm = (source_norm_sq * target_norm_sq).sqrt();
        if norm == 0.0 { 
            0.0 
        } else { 
            (dot_product / norm).min(1.0)
        }
    }
/// Direct text comparison - computes VSA from raw text
    /// Used by compare_texts for validation path
    pub fn vsa_from_text(&self, source_text: &str, target_text: &str) -> f64 {
        // Compute character frequencies on-the-fly
        let source_vec = self.text_to_weighted_vector(source_text);
        let target_vec = self.text_to_weighted_vector(target_text);
        
        // Calculate dot product and norms
        let mut dot_product = 0.0;
        let mut source_norm_sq = 0.0;
        let mut target_norm_sq = 0.0;
        
        // Since we filtered during vector creation, all weights pass threshold
        for (i, &source_weight) in source_vec.iter().enumerate() {
            if source_weight > 0.0 {
                source_norm_sq += source_weight * source_weight;
                
                if target_vec[i] > 0.0 {
                    dot_product += source_weight * target_vec[i];
                }
            }
        }
        
        for &target_weight in target_vec.iter() {
            if target_weight > 0.0 {
                target_norm_sq += target_weight * target_weight;
            }
        }
        
        let norm = (source_norm_sq * target_norm_sq).sqrt();
        if norm == 0.0 { 0.0 } else { (dot_product / norm).min(1.0) }
    }
    
    /// Helper to convert text to weighted vector with filtering
    fn text_to_weighted_vector(&self, text: &str) -> [f64; 64] {
        let mut counts = [0u32; 64];
        let mut total = 0u32;
        
        // Count characters using the same mapping as n-gram generation
        for ch in text.chars() {
            if let Some(index) = crate::ngram::arabic_char_mapping::char_to_index(ch) {
                counts[index as usize] += 1;
                total += 1;
            }
        }
        
        // Convert to weights with filtering
        let mut weights = [0.0_f64; 64];
        if total > 0 {
            let total_f64 = total as f64;
            for i in 0..64 {
                if counts[i] > 0 {
                    let weight = counts[i] as f64 / total_f64;
                    // Apply min_term_weight filter
                    if weight >= self.min_term_weight {
                        weights[i] = weight;
                    }
                }
            }
        }
        
        weights
    }
}

impl SimilarityAlgorithm for VSAMatcher {
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        // Use the text-based method for validation
        Ok(self.vsa_from_text(source, target))
    }
}

/// Factory for creating similarity algorithm instances
pub struct SimilarityAlgorithmFactory;

impl SimilarityAlgorithmFactory {
    pub fn create<'a>(
        metric: SimilarityMetric,
        config: &MatchingConfig,
    ) -> Result<Box<dyn SimilarityAlgorithm + 'a>> {
        match metric {
            SimilarityMetric::Exact => Ok(Box::new(ExactMatcher::new())),
            SimilarityMetric::Jaccard => Ok(Box::new(JaccardMatcher::new())),
            SimilarityMetric::Cosine => Ok(Box::new(CosineMatcher::new())),
            SimilarityMetric::Levenshtein => Ok(Box::new(LevenshteinMatcher::with_config(
                config.algorithm_config.levenshtein_max_distance
            ))),
            SimilarityMetric::VSA => Ok(Box::new(VSAMatcher::with_config(
                config.algorithm_config.vsa_min_term_weight
            ))),
        }
    }
}