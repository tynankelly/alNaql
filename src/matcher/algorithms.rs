use ahash::AHashMap;
use crate::utils::simd::{simd_memcmp, simd_jaccard, simd_cosine};
use crate::error::Result;
use crate::types::NGramEntry;
use crate::config::subsystems::matcher::MatcherConfig;
use super::types::{TextSpan, SimilarityMetric, NGramMatch};

/// The SimilarityAlgorithm trait defines the interface for comparing text similarity.
/// Each implementation provides a different way to measure how similar two pieces of text are.
/// All similarity scores are normalized between 0.0 (completely different) and 1.0 (identical).
pub trait SimilarityAlgorithm {
    /// Returns the type of similarity metric this algorithm implements
    fn name(&self) -> SimilarityMetric;

    /// Compares two individual ngrams and returns their similarity score
    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64>;

    /// Compares two spans of text (multiple ngrams) and returns their similarity score
    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64>;

    /// Compares two raw text strings and returns their similarity score
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64>;

    /// Finds all matches for a source ngram in a collection of target ngrams
    /// This method can be optimized by specific implementations (like ExactMatcher)
    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>>;
}

/// ExactMatcher implements exact string matching with optional storage-based optimization.
/// When storage is available, it can use indexed lookups instead of linear scanning.
pub struct ExactMatcher {
}

impl<'a> ExactMatcher {
    /// Creates a new ExactMatcher without storage optimization
    pub fn new() -> Self {
        Self { }
    }
}

impl<'a> SimilarityAlgorithm for ExactMatcher {
    fn name(&self) -> SimilarityMetric {
        SimilarityMetric::Exact
    }

    #[inline]
    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        // For exact matching, texts must match completely
        // Use SIMD-accelerated comparison for better performance
        let source_bytes = source.text_content.cleaned_text.as_bytes();
        let target_bytes = target.text_content.cleaned_text.as_bytes();
        
        Ok(if simd_memcmp(source_bytes, target_bytes) {
            1.0
        } else {
            0.0
        })
    }

    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        // For spans, compare the complete reconstructed text
        Ok(if source.original_text == target.original_text {
            1.0
        } else {
            0.0
        })
    }

    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        Ok(if source == target { 1.0 } else { 0.0 })
    }

    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        // Create a vector to store exact matches
        let mut exact_matches = Vec::new();
        
        // Get the source text bytes for direct comparison
        let source_bytes = source.text_content.cleaned_text.as_bytes();
        
        // Compare each target directly with source using SIMD acceleration
        for target in targets {
            // Use SIMD-accelerated memory comparison for better performance
            if simd_memcmp(target.text_content.cleaned_text.as_bytes(), source_bytes) {
                exact_matches.push(NGramMatch {
                    source_position: source.sequence_number as usize,
                    target_position: target.sequence_number as usize,
                    ngram: source.clone(),
                    similarity: 1.0,
                });
            }
        }
        
        Ok(exact_matches)
    }
}

/// JaccardMatcher implements similarity comparison using the Jaccard index,
/// which measures similarity as the size of the intersection divided by the size
/// of the union of character sets.
pub struct JaccardMatcher;

impl JaccardMatcher {
    pub fn new() -> Self {
        Self
    }
}

impl SimilarityAlgorithm for JaccardMatcher {
    fn name(&self) -> SimilarityMetric {
        SimilarityMetric::Jaccard
    }

    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        // Use Unicode-aware SIMD-accelerated Jaccard similarity calculation
        // This properly handles Arabic and other non-ASCII text
        let source_text = &source.text_content.cleaned_text;
        let target_text = &target.text_content.cleaned_text;
        
        Ok(simd_jaccard(source_text, target_text))
    }

    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        Ok(simd_jaccard(&source.original_text, &target.original_text))
    }

    #[inline]
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        Ok(simd_jaccard(source, target))
    }

    #[inline]
    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        // For Jaccard matcher, we use the default linear search
        // but each individual comparison uses our Unicode-aware SIMD acceleration
        let mut matches = Vec::new();
        for target in targets {
            let similarity = self.compare_ngrams(source, target)?;
            if similarity > 0.0 {
                matches.push(NGramMatch {
                    source_position: source.sequence_number as usize,
                    target_position: target.sequence_number as usize,
                    ngram: source.clone(),
                    similarity,
                });
            }
        }
        Ok(matches)
    }
}
/// CosineMatcher implements cosine similarity comparison with term frequency weighting.
/// It maintains a cache of term vectors to avoid recomputing frequencies.
pub struct CosineMatcher {
}

impl CosineMatcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl SimilarityAlgorithm for CosineMatcher {
    fn name(&self) -> SimilarityMetric {
        SimilarityMetric::Cosine
    }

    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        // Use the SIMD-accelerated cosine similarity function
        // This properly handles Unicode characters including Arabic
        let source_text = &source.text_content.cleaned_text;
        let target_text = &target.text_content.cleaned_text;
        
        Ok(simd_cosine(source_text, target_text))
    }

    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        Ok(simd_cosine(&source.original_text, &target.original_text))
    }

    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        Ok(simd_cosine(source, target))
    }

    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        // For CosineMatcher, we use the default linear search
        // but each individual comparison uses our SIMD acceleration
        let mut matches = Vec::new();
        for target in targets {
            let similarity = self.compare_ngrams(source, target)?;
            if similarity > 0.0 {
                matches.push(NGramMatch {
                    source_position: source.sequence_number as usize,
                    target_position: target.sequence_number as usize,
                    ngram: source.clone(),
                    similarity,
                });
            }
        }
        Ok(matches)
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

    fn levenshtein_distance(&self, source: &str, target: &str) -> usize {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();
        
        let rows = source_chars.len() + 1;
        let cols = target_chars.len() + 1;
        
        let mut matrix = vec![vec![0; cols]; rows];
        
        for i in 0..rows {
            matrix[i][0] = i;
        }
        for j in 0..cols {
            matrix[0][j] = j;
        }
        
        for i in 1..rows {
            for j in 1..cols {
                let cost = if source_chars[i-1] == target_chars[j-1] { 0 } else { 1 };
                matrix[i][j] = *[
                    matrix[i-1][j] + 1,      // deletion
                    matrix[i][j-1] + 1,      // insertion
                    matrix[i-1][j-1] + cost  // substitution
                ].iter().min().unwrap();
            }
        }
        
        matrix[rows-1][cols-1]
    }
}

impl SimilarityAlgorithm for LevenshteinMatcher {
    fn name(&self) -> SimilarityMetric {
        SimilarityMetric::Levenshtein
    }

    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        let distance = self.levenshtein_distance(
            &source.text_content.cleaned_text,
            &target.text_content.cleaned_text
        );
        
        if distance > self.max_distance {
            return Ok(0.0);
        }
        
        let max_len = source.text_content.cleaned_text.len()
            .max(target.text_content.cleaned_text.len());
            
        Ok(1.0 - (distance as f64 / max_len as f64))
    }

    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        let distance = self.levenshtein_distance(&source.original_text, &target.original_text);
        let max_len = source.original_text.len().max(target.original_text.len());
        
        if distance > self.max_distance {
            return Ok(0.0);
        }
        
        Ok(1.0 - (distance as f64 / max_len as f64))
    }

    #[inline]
    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        let distance = self.levenshtein_distance(source, target);
        let max_len = source.len().max(target.len());
        
        if distance > self.max_distance {
            return Ok(0.0);
        }
        
        Ok(1.0 - (distance as f64 / max_len as f64))
    }

    #[inline]
    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        let mut matches = Vec::new();
        for target in targets {
            let similarity = self.compare_ngrams(source, target)?;
            if similarity > 0.0 {
                matches.push(NGramMatch {
                    source_position: source.sequence_number as usize,
                    target_position: target.sequence_number as usize,
                    ngram: source.clone(),
                    similarity,
                });
            }
        }
        Ok(matches)
    }
}

/// VSAMatcher implements Vector Space Analysis with minimum term weight filtering.
pub struct VSAMatcher {
    min_term_weight: f64,
}

impl VSAMatcher {
    pub fn with_config(min_term_weight: f64) -> Self {
        Self { min_term_weight }
    }

    fn text_to_vector(&self, text: &str) -> AHashMap<char, f64> {
        let mut raw_counts = AHashMap::new();
        let total_chars = text.chars().count() as f64;

        // Count raw frequencies
        for c in text.chars() {
            *raw_counts.entry(c).or_insert(0.0) += 1.0;
        }

        // Convert to weighted frequencies
        raw_counts.into_iter()
            .filter_map(|(c, count)| {
                let weight = count / total_chars;
                if weight >= self.min_term_weight {
                    Some((c, weight))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl SimilarityAlgorithm for VSAMatcher {
    fn name(&self) -> SimilarityMetric {
        SimilarityMetric::VSA
    }

    fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        let source_vec = self.text_to_vector(&source.text_content.cleaned_text);
        let target_vec = self.text_to_vector(&target.text_content.cleaned_text);

        let mut dot_product = 0.0;
        let mut source_norm = 0.0;
        let mut target_norm = 0.0;

        for (c, s_val) in &source_vec {
            source_norm += s_val * s_val;
            if let Some(t_val) = target_vec.get(c) {
                dot_product += s_val * t_val;
            }
        }

        for t_val in target_vec.values() {
            target_norm += t_val * t_val;
        }

        let norm = (source_norm * target_norm).sqrt();
        Ok(if norm == 0.0 { 0.0 } else { dot_product / norm })
    }

    fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        self.compare_texts(&source.original_text, &target.original_text)
    }

    fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        let source_vec = self.text_to_vector(source);
        let target_vec = self.text_to_vector(target);

        let mut dot_product = 0.0;
        let mut source_norm = 0.0;
        let mut target_norm = 0.0;

        for (c, s_val) in &source_vec {
            source_norm += s_val * s_val;
            if let Some(t_val) = target_vec.get(c) {
                dot_product += s_val * t_val;
            }
        }

        for t_val in target_vec.values() {
            target_norm += t_val * t_val;
        }

        let norm = (source_norm * target_norm).sqrt();
        Ok(if norm == 0.0 { 0.0 } else { dot_product / norm })
    }

    fn find_matches(&self, source: &NGramEntry, targets: &[NGramEntry]) -> Result<Vec<NGramMatch>> {
        let mut matches = Vec::new();
        for target in targets {
            let similarity = self.compare_ngrams(source, target)?;
            if similarity > 0.0 {
                matches.push(NGramMatch {
                    source_position: source.sequence_number as usize,
                    target_position: target.sequence_number as usize,
                    ngram: source.clone(),
                    similarity,
                });
            }
        }
        Ok(matches)
    }
}

/// Factory for creating similarity algorithm instances
pub struct SimilarityAlgorithmFactory;

impl SimilarityAlgorithmFactory {
    pub fn create<'a>(
        metric: SimilarityMetric,
        config: &MatcherConfig,
    ) -> Result<Box<dyn SimilarityAlgorithm + 'a>> {
        match metric {
            SimilarityMetric::Exact => Ok(Box::new(ExactMatcher::new())),
            SimilarityMetric::Jaccard => Ok(Box::new(JaccardMatcher::new())),
            SimilarityMetric::Cosine => Ok(Box::new(CosineMatcher::new(
            ))),
            SimilarityMetric::Levenshtein => Ok(Box::new(LevenshteinMatcher::with_config(
                config.algorithm_config.levenshtein_max_distance
            ))),
            SimilarityMetric::VSA => Ok(Box::new(VSAMatcher::with_config(
                config.algorithm_config.vsa_min_term_weight
            ))),
        }
    }
}