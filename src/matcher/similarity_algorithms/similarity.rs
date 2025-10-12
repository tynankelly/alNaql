use log::{info, debug, trace, warn};
use ahash::AHashMap;
use std::sync::Arc;
use crate::config::AlNaqlConfig;
use crate::error::{Result, Error};
use crate::storage::LMDBStorage;
use crate::matcher::types::SequenceMatch;
use crate::config::{SimilarityMetric, NGramType};
use crate::matcher::similarity_algorithms::algorithms::{SimilarityAlgorithm, JaccardMatcher, CosineMatcher, LevenshteinMatcher, VSAMatcher, ExactMatcher};
use crate::matcher::clusters::text_classification::TextClassifier;
use crate::types::{CompactCharFrequencies, NGramMetadata};

use crate::utils::simd::{simd_jaccard, simd_cosine};

pub struct SimilarityCalculator {
    config: Arc<AlNaqlConfig>,
}
impl SimilarityCalculator {
    pub fn new(config: Arc<AlNaqlConfig>) -> Result<Self> {
        Ok(Self { 
            config,
        })
    }

    pub fn get_config(&self) -> &AlNaqlConfig {
        &*self.config
    }

    // Entry point for ngram matching
    /// Entry point for ngram matching with optional pre-loaded target data
    pub fn find_matching_ngrams(
        &self,
        source_seq: u64,
        target_sequences: &[u64],
        source_storage: &LMDBStorage,
        target_storage: &LMDBStorage,
        config: &AlNaqlConfig,
        // ✅ NEW: Optional pre-loaded data from cell level
        preloaded_target_metadata: Option<&AHashMap<u64, NGramMetadata>>,
        preloaded_target_freqs: Option<&AHashMap<u64, CompactCharFrequencies>>,
        preloaded_target_text: Option<&AHashMap<u64, Vec<u8>>>,
    ) -> Result<Vec<SequenceMatch>> {
        
        // Early exit if no targets provided
        if target_sequences.is_empty() {
            info!("No target sequences provided for comparison");
            return Ok(Vec::new());
        }
        
        // Check if this is a traced sequence for debugging
        let is_traced = if log::log_enabled!(log::Level::Debug) && !config.debug.traced_phrase.is_empty() {
            // Load source text to check for traced phrase
            let source_text_map = source_storage.get_text_bytes_batch(&[source_seq])?;
            if let Some(bytes) = source_text_map.get(&source_seq) {
                let text = std::str::from_utf8(bytes)?;
                text.contains(&self.config.debug.traced_phrase)
            } else {
                false
            }
        } else {
            false
        };
        
        if is_traced {
            debug!("TRACE [find_matching_ngrams]: Processing traced source sequence: {}", source_seq);
            debug!("TRACE [find_matching_ngrams]: Comparing against {} target sequences", target_sequences.len());
            if preloaded_target_metadata.is_some() {
                debug!("TRACE [find_matching_ngrams]: Using pre-loaded target data (optimized path)");
            }
        }
        
        // Process using the batch processor with optional pre-loaded data
        let matches = self.process_batch(
            source_seq,
            target_sequences,
            source_storage,
            target_storage,
            config,
            preloaded_target_metadata,
            preloaded_target_freqs,
            preloaded_target_text,
        )?;
        
        // Log results for traced sequences
        if is_traced {
            info!("TRACE [find_matching_ngrams]: Found {} matches for traced sequence", matches.len());
            for (i, m) in matches.iter().enumerate().take(5) {
                info!("TRACE [find_matching_ngrams]: Match #{} - source seq: {}, target seq: {}", 
                    i + 1, m.source_sequence, m.target_sequence);
            }
            if matches.len() > 5 {
                info!("TRACE [find_matching_ngrams]: ... and {} more matches", matches.len() - 5);
            }
        }
        
        Ok(matches)
    }

    // Add this helper method if it doesn't exist:
    fn get_algorithm(&self, metric: SimilarityMetric) -> Result<Box<dyn SimilarityAlgorithm>> {
        match metric {
            SimilarityMetric::Jaccard => Ok(Box::new(JaccardMatcher::new())),
            SimilarityMetric::Cosine => Ok(Box::new(CosineMatcher::new())),
            SimilarityMetric::Levenshtein => {
                let max_distance = self.config.matcher.algorithm_config.levenshtein_max_distance;
                Ok(Box::new(LevenshteinMatcher::with_config(max_distance)))
            }
            SimilarityMetric::VSA => {
                let min_term_weight = self.config.matcher.algorithm_config.vsa_min_term_weight;
                Ok(Box::new(VSAMatcher::with_config(min_term_weight)))
            }
            SimilarityMetric::Exact => Ok(Box::new(ExactMatcher::new())),
        }
    }

    fn process_batch(
        &self,
        source_seq: u64,
        target_sequences: &[u64],
        source_storage: &LMDBStorage,
        target_storage: &LMDBStorage,
        config: &AlNaqlConfig,
        preloaded_target_metadata: Option<&AHashMap<u64, NGramMetadata>>,
        preloaded_target_freqs: Option<&AHashMap<u64, CompactCharFrequencies>>,
        preloaded_target_text: Option<&AHashMap<u64, Vec<u8>>>,
    ) -> Result<Vec<SequenceMatch>> {
        let mut matches = Vec::with_capacity(target_sequences.len() / 4);
        
        match self.config.matcher.ngram_similarity_metric {
            SimilarityMetric::Jaccard => {
                // Load source metadata ONCE
                let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
                let source_metadata = source_metadata_vec
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Storage(format!("No metadata for sequence {}", source_seq)))?;
                let source_hash = source_metadata.text_hash;
                
                let jaccard_matcher = JaccardMatcher::new();
                
                // Load source frequencies if using precomputed features
                let source_freqs = if self.config.matcher.use_precomputed_similarity_features {
                    Some(source_storage.get_char_frequencies_batch(&[source_seq])?
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No char frequencies for sequence {}", source_seq)))?
                        .clone())
                } else {
                    None
                };
                
                // Load source bytes if NOT using precomputed features
                let source_bytes = if !self.config.matcher.use_precomputed_similarity_features {
                    let source_text_map = source_storage.get_text_bytes_batch(&[source_seq])?;
                    Some(source_text_map
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No text bytes for sequence {}", source_seq)))?
                        .clone())
                } else {
                    None
                };
                
                // ================================================================
                // ✅ FAST PATH: Use pre-loaded data (no DB calls, no chunking!)
                // ================================================================
                if let Some(preloaded_metadata) = preloaded_target_metadata {
                    trace!("Using pre-loaded target data for {} targets", target_sequences.len());
                    
                    // Process ALL targets directly without chunking
                    for &target_seq in target_sequences {
                        // Get metadata from pre-loaded map
                        let target_meta = match preloaded_metadata.get(&target_seq) {
                            Some(meta) => meta,
                            None => continue,  // Skip if not found
                        };
                        
                        // Check for exact hash match first
                        if source_hash != 0 && target_meta.text_hash == source_hash {
                            matches.push(SequenceMatch {
                                source_sequence: source_seq,
                                target_sequence: target_seq,
                            });
                            continue;
                        }
                        
                        // Calculate similarity based on whether using precomputed features
                        let similarity = if self.config.matcher.use_precomputed_similarity_features {
                            // ✅ CORRECT: Use jaccard_from_features with CompactCharFrequencies
                            if let (Some(src_freqs), Some(preloaded_freqs)) = (&source_freqs, preloaded_target_freqs) {
                                if let Some(target_freqs) = preloaded_freqs.get(&target_seq) {
                                    jaccard_matcher.jaccard_from_features(src_freqs, target_freqs)
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        } else {
                            // ✅ CORRECT: Use simd_jaccard with raw bytes
                            if let (Some(src_bytes), Some(preloaded_text)) = (&source_bytes, preloaded_target_text) {
                                if let Some(target_bytes) = preloaded_text.get(&target_seq) {
                                    simd_jaccard(src_bytes, target_bytes)
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        // Check threshold
                        if similarity >= config.matcher.ngram_min_confidence {
                            matches.push(SequenceMatch {
                                source_sequence: source_seq,
                                target_sequence: target_seq,
                            });
                        }
                    }
                } 
                // ================================================================
                // ❌ LEGACY PATH: No pre-loaded data, use original chunking logic
                // ================================================================
                else {
                    trace!("No pre-loaded data, using chunked DB loading for {} targets", target_sequences.len());
                    
                    // Process targets in chunks (original logic)
                    for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                        // BATCH LOAD 1: Metadata for all targets (for hash check)
                        let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                        
                        // Convert to HashMap for efficient lookup
                        let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                            .into_iter()
                            .map(|m| (m.sequence_number, m))
                            .collect();
                        
                        // Separate exact matches from candidates needing similarity calc
                        let mut exact_matches = Vec::new();
                        let mut candidates = Vec::new();
                        
                        for &target_seq in chunk {
                            if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                if source_hash != 0 && target_meta.text_hash == source_hash {
                                    exact_matches.push(target_seq);
                                } else {
                                    candidates.push(target_seq);
                                }
                            }
                        }
                        
                        // Record exact matches immediately
                        for target_seq in exact_matches {
                            matches.push(SequenceMatch { 
                                source_sequence: source_seq, 
                                target_sequence: target_seq 
                            });
                        }
                        
                        // BATCH LOAD 2: Additional data only for non-exact matches
                        if !candidates.is_empty() {
                            if self.config.matcher.use_precomputed_similarity_features {
                                // ✅ CORRECT: Load char frequencies and use jaccard_from_features
                                let target_freqs_map = target_storage.get_char_frequencies_batch(&candidates)?;
                                
                                if let Some(src_freqs) = &source_freqs {
                                    for &target_seq in &candidates {
                                        if let Some(target_freqs) = target_freqs_map.get(&target_seq) {
                                            // ✅ Use jaccard_from_features, not simd_jaccard
                                            let similarity = jaccard_matcher.jaccard_from_features(src_freqs, target_freqs);
                                            
                                            if similarity >= config.matcher.ngram_min_confidence {
                                                matches.push(SequenceMatch { 
                                                    source_sequence: source_seq, 
                                                    target_sequence: target_seq 
                                                });
                                            }
                                        }
                                    }
                                }
                            } else {
                                // ✅ CORRECT: Load text bytes and use simd_jaccard
                                let target_text_map = target_storage.get_text_bytes_batch(&candidates)?;
                                
                                if let Some(src_bytes) = &source_bytes {
                                    for &target_seq in &candidates {
                                        if let Some(target_bytes) = target_text_map.get(&target_seq) {
                                            // ✅ Use simd_jaccard with raw bytes
                                            let similarity = simd_jaccard(src_bytes, target_bytes);
                                            
                                            if similarity >= config.matcher.ngram_min_confidence {
                                                matches.push(SequenceMatch { 
                                                    source_sequence: source_seq, 
                                                    target_sequence: target_seq 
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            
            SimilarityMetric::Cosine => {
                // Load source metadata ONCE
                let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
                let source_metadata = source_metadata_vec
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Storage(format!("No metadata for sequence {}", source_seq)))?;
                let source_hash = source_metadata.text_hash;
                
                let cosine_matcher = CosineMatcher::new();
                
                // Load source frequencies if using precomputed features
                let source_freqs = if self.config.matcher.use_precomputed_similarity_features {
                    Some(source_storage.get_char_frequencies_batch(&[source_seq])?
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No char frequencies for sequence {}", source_seq)))?
                        .clone())
                } else {
                    None
                };
                
                // Load source bytes if NOT using precomputed features
                let source_bytes = if !self.config.matcher.use_precomputed_similarity_features {
                    let source_text_map = source_storage.get_text_bytes_batch(&[source_seq])?;
                    Some(source_text_map
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No text bytes for sequence {}", source_seq)))?
                        .clone())
                } else {
                    None
                };
                
                // ================================================================
                // ✅ FAST PATH: Use pre-loaded data
                // ================================================================
                if let Some(preloaded_metadata) = preloaded_target_metadata {
                    trace!("Using pre-loaded target data for {} targets (Cosine)", target_sequences.len());
                    
                    for &target_seq in target_sequences {
                        let target_meta = match preloaded_metadata.get(&target_seq) {
                            Some(meta) => meta,
                            None => continue,
                        };
                        
                        // Check for exact hash match first
                        if source_hash != 0 && target_meta.text_hash == source_hash {
                            matches.push(SequenceMatch {
                                source_sequence: source_seq,
                                target_sequence: target_seq,
                            });
                            continue;
                        }
                        
                        // Calculate similarity
                        let similarity = if self.config.matcher.use_precomputed_similarity_features {
                            // ✅ CORRECT: Use cosine_from_features
                            if let (Some(src_freqs), Some(preloaded_freqs)) = (&source_freqs, preloaded_target_freqs) {
                                if let Some(target_freqs) = preloaded_freqs.get(&target_seq) {
                                    cosine_matcher.cosine_from_features(src_freqs, target_freqs)
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        } else {
                            // ✅ CORRECT: Use simd_cosine with raw bytes
                            if let (Some(src_bytes), Some(preloaded_text)) = (&source_bytes, preloaded_target_text) {
                                if let Some(target_bytes) = preloaded_text.get(&target_seq) {
                                    simd_cosine(src_bytes, target_bytes)
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        };
                        
                        if similarity >= config.matcher.ngram_min_confidence {
                            matches.push(SequenceMatch {
                                source_sequence: source_seq,
                                target_sequence: target_seq,
                            });
                        }
                    }
                }
                // ================================================================
                // ❌ LEGACY PATH: Original chunking logic for Cosine
                // ================================================================
                else {
                    trace!("No pre-loaded data, using chunked DB loading for {} targets (Cosine)", target_sequences.len());
                    
                    for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                        let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                        let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                            .into_iter()
                            .map(|m| (m.sequence_number, m))
                            .collect();
                        
                        let mut exact_matches = Vec::new();
                        let mut candidates = Vec::new();
                        
                        for &target_seq in chunk {
                            if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                if source_hash != 0 && target_meta.text_hash == source_hash {
                                    exact_matches.push(target_seq);
                                } else {
                                    candidates.push(target_seq);
                                }
                            }
                        }
                        
                        for target_seq in exact_matches {
                            matches.push(SequenceMatch { 
                                source_sequence: source_seq, 
                                target_sequence: target_seq 
                            });
                        }
                        
                        if !candidates.is_empty() {
                            if self.config.matcher.use_precomputed_similarity_features {
                                // ✅ CORRECT: Use cosine_from_features
                                let target_freqs_map = target_storage.get_char_frequencies_batch(&candidates)?;
                                
                                if let Some(src_freqs) = &source_freqs {
                                    for &target_seq in &candidates {
                                        if let Some(target_freqs) = target_freqs_map.get(&target_seq) {
                                            // ✅ Use cosine_from_features, not simd_cosine
                                            let similarity = cosine_matcher.cosine_from_features(src_freqs, target_freqs);
                                            
                                            if similarity >= config.matcher.ngram_min_confidence {
                                                matches.push(SequenceMatch { 
                                                    source_sequence: source_seq, 
                                                    target_sequence: target_seq 
                                                });
                                            }
                                        }
                                    }
                                }
                            } else {
                                // ✅ CORRECT: Use simd_cosine with raw bytes
                                let target_text_map = target_storage.get_text_bytes_batch(&candidates)?;
                                
                                if let Some(src_bytes) = &source_bytes {
                                    for &target_seq in &candidates {
                                        if let Some(target_bytes) = target_text_map.get(&target_seq) {
                                            // ✅ Use simd_cosine with raw bytes
                                            let similarity = simd_cosine(src_bytes, target_bytes);
                                            
                                            if similarity >= config.matcher.ngram_min_confidence {
                                                matches.push(SequenceMatch { 
                                                    source_sequence: source_seq, 
                                                    target_sequence: target_seq 
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },

            SimilarityMetric::Levenshtein => {
                // Load source metadata for hash check
                let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
                let source_metadata = source_metadata_vec
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Storage(format!("No metadata for sequence {}", source_seq)))?;
                let source_hash = source_metadata.text_hash;
                
                // Load source bytes (Levenshtein always needs raw text, no precomputed features)
                let source_text_map = source_storage.get_text_bytes_batch(&[source_seq])?;
                let source_bytes = source_text_map
                    .get(&source_seq)
                    .ok_or_else(|| Error::Storage(format!("No text bytes for sequence {}", source_seq)))?;
                
                // Create matcher with max distance from config
                let max_distance = self.config.matcher.algorithm_config.levenshtein_max_distance;
                let levenshtein_matcher = LevenshteinMatcher::with_config(max_distance);
                
                // ================================================================
                // ✅ FAST PATH: Use pre-loaded data (no DB calls, no chunking!)
                // ================================================================
                if let Some(preloaded_metadata) = preloaded_target_metadata {
                    trace!("Using pre-loaded target data for {} targets (Levenshtein)", target_sequences.len());
                    
                    // Process ALL targets directly without chunking
                    for &target_seq in target_sequences {
                        // Get metadata from pre-loaded map
                        let target_meta = match preloaded_metadata.get(&target_seq) {
                            Some(meta) => meta,
                            None => continue,  // Skip if not found
                        };
                        
                        // Check for exact hash match first (skip expensive Levenshtein calculation)
                        if source_hash != 0 && target_meta.text_hash == source_hash {
                            matches.push(SequenceMatch {
                                source_sequence: source_seq,
                                target_sequence: target_seq,
                            });
                            continue;
                        }
                        
                        // Get target bytes from pre-loaded map
                        // Note: Levenshtein always needs text bytes, no precomputed feature path
                        if let Some(preloaded_text) = preloaded_target_text {
                            if let Some(target_bytes) = preloaded_text.get(&target_seq) {
                                // Calculate Levenshtein distance directly on bytes
                                let distance = levenshtein_matcher.levenshtein_distance(
                                    source_bytes,  // &Vec<u8>
                                    target_bytes   // &Vec<u8>
                                );
                                
                                // Skip if distance exceeds max threshold
                                if distance > max_distance {
                                    continue;
                                }
                                
                                // Convert distance to normalized similarity score
                                let max_len = source_bytes.len().max(target_bytes.len()).max(1);
                                let similarity = 1.0 - (distance as f64 / max_len as f64);
                                
                                // Check if similarity meets threshold
                                if similarity >= config.matcher.ngram_min_confidence {
                                    matches.push(SequenceMatch {
                                        source_sequence: source_seq,
                                        target_sequence: target_seq,
                                    });
                                }
                            }
                        }
                    }
                }
                // ================================================================
                // ❌ LEGACY PATH: No pre-loaded data, use original chunking logic
                // ================================================================
                else {
                    trace!("No pre-loaded data, using chunked DB loading for {} targets (Levenshtein)", target_sequences.len());
                    
                    // Process targets in chunks (original logic)
                    for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                        // Load metadata for hash check
                        let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                        
                        // Convert to HashMap for efficient lookup
                        let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                            .into_iter()
                            .map(|m| (m.sequence_number, m))
                            .collect();
                        
                        // Separate exact matches from candidates needing distance calculation
                        let mut exact_matches = Vec::new();
                        let mut candidates = Vec::new();
                        
                        for &target_seq in chunk {
                            if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                if source_hash != 0 && target_meta.text_hash == source_hash {
                                    exact_matches.push(target_seq);
                                } else {
                                    candidates.push(target_seq);
                                }
                            }
                        }
                        
                        // Record exact matches immediately
                        for target_seq in exact_matches {
                            matches.push(SequenceMatch { 
                                source_sequence: source_seq, 
                                target_sequence: target_seq 
                            });
                        }
                        
                        // Load text for all candidates and compute distance
                        if !candidates.is_empty() {
                            let target_text_map = target_storage.get_text_bytes_batch(&candidates)?;
                            
                            for &target_seq in &candidates {
                                if let Some(target_bytes) = target_text_map.get(&target_seq) {
                                    // Calculate Levenshtein distance
                                    let distance = levenshtein_matcher.levenshtein_distance(
                                        source_bytes,  // &Vec<u8>
                                        target_bytes   // &Vec<u8>
                                    );
                                    
                                    // Skip if distance exceeds max
                                    if distance > max_distance {
                                        continue;
                                    }
                                    
                                    // Convert to similarity score
                                    let max_len = source_bytes.len().max(target_bytes.len()).max(1);
                                    let similarity = 1.0 - (distance as f64 / max_len as f64);
                                    
                                    if similarity >= config.matcher.ngram_min_confidence {
                                        matches.push(SequenceMatch { 
                                            source_sequence: source_seq, 
                                            target_sequence: target_seq 
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            SimilarityMetric::VSA => {
                // Load source metadata
                let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
                let source_metadata = source_metadata_vec
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Storage(format!("No metadata for sequence {}", source_seq)))?;
                let source_hash = source_metadata.text_hash;
                
                let min_term_weight = self.config.matcher.algorithm_config.vsa_min_term_weight;
                let vsa_matcher = VSAMatcher::with_config(min_term_weight);
                
                // VSA has two different calculation paths depending on configuration
                if self.config.matcher.use_precomputed_similarity_features {
                    // ============================================================
                    // PRECOMPUTED FEATURES PATH
                    // ============================================================
                    let source_freqs = source_storage.get_char_frequencies_batch(&[source_seq])?
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No char frequencies for sequence {}", source_seq)))?
                        .clone();
                    
                    // ================================================================
                    // ✅ FAST PATH: Use pre-loaded data with precomputed features
                    // ================================================================
                    if let Some(preloaded_metadata) = preloaded_target_metadata {
                        trace!("Using pre-loaded target data for {} targets (VSA with features)", target_sequences.len());
                        
                        // Process ALL targets directly without chunking
                        for &target_seq in target_sequences {
                            // Get metadata from pre-loaded map
                            let target_meta = match preloaded_metadata.get(&target_seq) {
                                Some(meta) => meta,
                                None => continue,
                            };
                            
                            // Check for exact hash match first
                            if source_hash != 0 && target_meta.text_hash == source_hash {
                                matches.push(SequenceMatch {
                                    source_sequence: source_seq,
                                    target_sequence: target_seq,
                                });
                                continue;
                            }
                            
                            // Get target frequencies from pre-loaded map
                            if let Some(preloaded_freqs) = preloaded_target_freqs {
                                if let Some(target_freqs) = preloaded_freqs.get(&target_seq) {
                                    // Use precomputed features for VSA calculation
                                    let similarity = vsa_matcher.vsa_from_features(&source_freqs, target_freqs);
                                    
                                    if similarity >= config.matcher.ngram_min_confidence {
                                        matches.push(SequenceMatch {
                                            source_sequence: source_seq,
                                            target_sequence: target_seq,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    // ================================================================
                    // ❌ LEGACY PATH: Chunking with precomputed features
                    // ================================================================
                    else {
                        trace!("No pre-loaded data, using chunked DB loading for {} targets (VSA with features)", target_sequences.len());
                        
                        for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                            // Load metadata for hash check
                            let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                            let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                                .into_iter()
                                .map(|m| (m.sequence_number, m))
                                .collect();
                            
                            // Separate exact matches from candidates
                            let mut exact_matches = Vec::new();
                            let mut candidates = Vec::new();
                            
                            for &target_seq in chunk {
                                if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                    if source_hash != 0 && target_meta.text_hash == source_hash {
                                        exact_matches.push(target_seq);
                                    } else {
                                        candidates.push(target_seq);
                                    }
                                }
                            }
                            
                            // Record exact matches
                            for target_seq in exact_matches {
                                matches.push(SequenceMatch { 
                                    source_sequence: source_seq, 
                                    target_sequence: target_seq 
                                });
                            }
                            
                            // Load frequencies for candidates
                            if !candidates.is_empty() {
                                let target_freqs_map = target_storage.get_char_frequencies_batch(&candidates)?;
                                
                                for &target_seq in &candidates {
                                    if let Some(target_freqs) = target_freqs_map.get(&target_seq) {
                                        // Use precomputed features
                                        let similarity = vsa_matcher.vsa_from_features(&source_freqs, target_freqs);
                                        
                                        if similarity >= config.matcher.ngram_min_confidence {
                                            matches.push(SequenceMatch { 
                                                source_sequence: source_seq, 
                                                target_sequence: target_seq 
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // ============================================================
                    // TEXT-BASED CALCULATION PATH (Fallback when no precomputed features)
                    // ============================================================
                    let source_bytes_map = source_storage.get_text_bytes_batch(&[source_seq])?;
                    let source_bytes = source_bytes_map
                        .get(&source_seq)
                        .ok_or_else(|| Error::Storage(format!("No text bytes for sequence {}", source_seq)))?;
                    
                    // Convert bytes to text for VSA calculation
                    let source_text = std::str::from_utf8(source_bytes)?;
                    
                    // ================================================================
                    // ✅ FAST PATH: Use pre-loaded data with text-based calculation
                    // ================================================================
                    if let Some(preloaded_metadata) = preloaded_target_metadata {
                        trace!("Using pre-loaded target data for {} targets (VSA from text)", target_sequences.len());
                        
                        // Process ALL targets directly without chunking
                        for &target_seq in target_sequences {
                            // Get metadata from pre-loaded map
                            let target_meta = match preloaded_metadata.get(&target_seq) {
                                Some(meta) => meta,
                                None => continue,
                            };
                            
                            // Check for exact hash match first
                            if source_hash != 0 && target_meta.text_hash == source_hash {
                                matches.push(SequenceMatch {
                                    source_sequence: source_seq,
                                    target_sequence: target_seq,
                                });
                                continue;
                            }
                            
                            // Get target text from pre-loaded map
                            if let Some(preloaded_text) = preloaded_target_text {
                                if let Some(target_bytes) = preloaded_text.get(&target_seq) {
                                    // Convert bytes to text and compute VSA
                                    let target_text = std::str::from_utf8(target_bytes)?;
                                    let similarity = vsa_matcher.vsa_from_text(source_text, target_text);
                                    
                                    if similarity >= config.matcher.ngram_min_confidence {
                                        matches.push(SequenceMatch {
                                            source_sequence: source_seq,
                                            target_sequence: target_seq,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    // ================================================================
                    // ❌ LEGACY PATH: Chunking with text-based calculation
                    // ================================================================
                    else {
                        trace!("No pre-loaded data, using chunked DB loading for {} targets (VSA from text)", target_sequences.len());
                        
                        for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                            // Load metadata for hash check
                            let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                            let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                                .into_iter()
                                .map(|m| (m.sequence_number, m))
                                .collect();
                            
                            // Separate exact matches from candidates
                            let mut exact_matches = Vec::new();
                            let mut candidates = Vec::new();
                            
                            for &target_seq in chunk {
                                if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                    if source_hash != 0 && target_meta.text_hash == source_hash {
                                        exact_matches.push(target_seq);
                                    } else {
                                        candidates.push(target_seq);
                                    }
                                }
                            }
                            
                            // Record exact matches
                            for target_seq in exact_matches {
                                matches.push(SequenceMatch { 
                                    source_sequence: source_seq, 
                                    target_sequence: target_seq 
                                });
                            }
                            
                            // Load text for candidates
                            if !candidates.is_empty() {
                                let target_text_map = target_storage.get_text_bytes_batch(&candidates)?;
                                
                                for &target_seq in &candidates {
                                    if let Some(target_bytes) = target_text_map.get(&target_seq) {
                                        // Convert bytes to text and compute VSA
                                        let target_text = std::str::from_utf8(target_bytes)?;
                                        let similarity = vsa_matcher.vsa_from_text(source_text, target_text);
                                        
                                        if similarity >= config.matcher.ngram_min_confidence {
                                            matches.push(SequenceMatch { 
                                                source_sequence: source_seq, 
                                                target_sequence: target_seq 
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            SimilarityMetric::Exact => {
                // Load source metadata for hash
                let source_metadata_vec = source_storage.get_ngram_metadata_batch(&[source_seq])?;
                let source_metadata = source_metadata_vec
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Storage(format!("No metadata for sequence {}", source_seq)))?;
                let source_hash = source_metadata.text_hash;
                
                // Skip if source has no hash (empty or invalid text)
                if source_hash == 0 {
                    return Ok(Vec::new());
                }
                
                // ================================================================
                // ✅ FAST PATH: Use pre-loaded metadata (only needs hash comparison)
                // ================================================================
                if let Some(preloaded_metadata) = preloaded_target_metadata {
                    trace!("Using pre-loaded target data for {} targets (Exact)", target_sequences.len());
                    
                    // Exact matching is the simplest - just compare hash values
                    // No need for frequencies or text bytes at all
                    for &target_seq in target_sequences {
                        if let Some(target_meta) = preloaded_metadata.get(&target_seq) {
                            // For exact matching, hashes must be identical
                            if target_meta.text_hash == source_hash {
                                matches.push(SequenceMatch {
                                    source_sequence: source_seq,
                                    target_sequence: target_seq,
                                });
                            }
                        }
                    }
                }
                // ================================================================
                // ❌ LEGACY PATH: No pre-loaded data, use original chunking logic
                // ================================================================
                else {
                    trace!("No pre-loaded data, using chunked DB loading for {} targets (Exact)", target_sequences.len());
                    
                    // Process targets in chunks
                    for chunk in target_sequences.chunks(config.matcher.similarity_chunk_size) {
                        // Load metadata for all targets in this chunk
                        let target_metadata_vec = target_storage.get_ngram_metadata_batch(chunk)?;
                        
                        // Convert to HashMap for efficient lookup
                        let target_metadata_map: AHashMap<u64, _> = target_metadata_vec
                            .into_iter()
                            .map(|m| (m.sequence_number, m))
                            .collect();
                        
                        // For exact matching, just check if hashes match
                        for &target_seq in chunk {
                            if let Some(target_meta) = target_metadata_map.get(&target_seq) {
                                if target_meta.text_hash == source_hash {
                                    matches.push(SequenceMatch {
                                        source_sequence: source_seq,
                                        target_sequence: target_seq,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        trace!("Completed batch processing: {} matches found from {} targets",
            matches.len(), target_sequences.len());
        
        Ok(matches)
    }


    pub fn compare_texts(
        &self, 
        source: &str, 
        target: &str, 
        text_classifier: Option<&TextClassifier>, 
        config: &AlNaqlConfig,
        ngram_type: NGramType,
        source_sequence_range: Option<(u64, u64)>,
        source_storage: Option<&LMDBStorage>,
    ) -> Result<f64> {
        
        // Get the base algorithm for text comparison
        let algorithm = self.get_algorithm(self.config.matcher.text_similarity_metric)?;
        let base_similarity = algorithm.compare_texts(source, target)?;
        
        match ngram_type {
            NGramType::Character => {
                // Character ngrams: always use simple algorithm comparison
                // No word tokenization makes sense for character fragments
                trace!("Character ngram comparison: using direct algorithm");
                Ok(base_similarity)
            },
            
            NGramType::Word => {
                if !self.config.matcher.apply_text_weighting {
                    // Word ngrams with weighting disabled: use same as character
                    trace!("Word ngram comparison: weighting disabled, using direct algorithm");
                    Ok(base_similarity)
                } else {
                    // Word ngrams with weighting enabled: full logic
                    trace!("Word ngram comparison: applying LCS and banality weighting");
                    
                    // Require text_classifier for weighted comparison
                    let classifier = match text_classifier {
                        Some(c) => c,
                        None => {
                            warn!("Text weighting requested but no TextClassifier provided");
                            return Ok(base_similarity);
                        }
                    };
                    
                    // Calculate word-level LCS similarity
                    let source_words: Vec<&str> = source.split_whitespace().collect();
                    let target_words: Vec<&str> = target.split_whitespace().collect();
                    let lcs_len = self.longest_common_subsequence(&source_words, &target_words);
                    let lcs_similarity = if source_words.len() + target_words.len() > 0 {
                        2.0 * lcs_len as f64 / (source_words.len() + target_words.len()) as f64
                    } else {
                        0.0
                    };

                    trace!("LCS calculation: source_words={}, target_words={}, lcs_len={}, lcs_similarity={:.3}", 
                        source_words.len(), target_words.len(), lcs_len, lcs_similarity);

                    // Get banality assessment
                    let (_is_banal, banality_density, _) = classifier.is_likely_banality(
                        source,
                        config,
                        &self.config.debug.traced_phrase,
                        source_sequence_range,
                        source_storage,
                    );

                    trace!("Banality assessment: density={:.3}, scale_factor={:.3}", 
                        banality_density, config.matcher.banality_scale_factor);

                    // Calculate banality factor
                    let banality_factor = 1.0 - (banality_density * config.matcher.banality_scale_factor);

                    trace!("Banality factor calculation: 1.0 - ({:.3} * {:.3}) = {:.3}", 
                        banality_density, config.matcher.banality_scale_factor, banality_factor);

                    let weighted_lcs = lcs_similarity * banality_factor;

                    trace!("Weighted LCS: {:.3} * {:.3} = {:.3}", 
                        lcs_similarity, banality_factor, weighted_lcs);

                    // Hardcoded mix: 70% word (LCS), 30% algorithm
                    let final_score = (weighted_lcs * config.matcher.lcs_weight) + (base_similarity * config.matcher.algorithm_weight);

                    trace!("Final score: ({:.3} * {:.3}) + ({:.3} * {:.3}) = {:.3}", 
                        weighted_lcs, config.matcher.lcs_weight, 
                        base_similarity, config.matcher.algorithm_weight, 
                        final_score);

                    Ok(final_score)
                }
            }
        }
    }

    fn longest_common_subsequence(&self, source: &[&str], target: &[&str]) -> usize {
        let m = source.len();
        let n = target.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if source[i-1] == target[j-1] {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = dp[i-1][j].max(dp[i][j-1]);
                }
            }
        }
        
        dp[m][n]
    }
    
}