use ahash::{AHashMap, AHashSet};
use std::sync::RwLock;
use log::{info, debug, trace, warn};
use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::types::NGramEntry;
use crate::matcher::algorithms::ExactMatcher;
use crate::config::subsystems::matcher::{MatcherConfig, SimilarityMetric};
use crate::config::subsystems::generator::NGramType;
use crate::matcher::algorithms::SimilarityAlgorithm;
use super::types::{TextSpan, NGramMatch};
use crate::storage::StorageBackend;

// Tunable parameters
const CHAR_FREQ_THRESHOLD: f64 = 0.5;
const MIN_CACHE_SIZE: usize = 50_000;
const MAX_CACHE_SIZE: usize = 100_000;
const EVICTION_PERCENTAGE: f64 = 0.25;

// Type alias for frequency maps
type CharFreqs = AHashMap<char, usize>;
type WordFreqs = AHashMap<String, usize>;


pub struct SimilarityCalculator {
    config: MatcherConfig,
    ngram_type: NGramType,
    char_freq_cache: RwLock<AHashMap<String, CharFreqs>>,
    word_freq_cache: RwLock<AHashMap<String, WordFreqs>>,
    banality_phrases: AHashSet<String>,
    banality_words: AHashSet<String>,
    banality_threshold: f64,
}
fn load_terms_from_file(filename: &str) -> Result<AHashSet<String>> {
    let path = Path::new(filename);
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            warn!("Failed to open file {}: {}", filename, e);
            return Ok(AHashSet::new()); // Return empty set on error
        }
    };
    
    let reader = BufReader::new(file);
    let mut terms = AHashSet::new();
    
    for line in reader.lines() {
        match line {
            Ok(line) => {
                let trimmed = line.trim();
                // Skip empty lines and comments
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    terms.insert(trimmed.to_string());
                }
            },
            Err(e) => warn!("Error reading line: {}", e)
        }
    }
    
    info!("Loaded {} terms from {}", terms.len(), filename);
    Ok(terms)
}

impl SimilarityCalculator {
    pub fn new(config: MatcherConfig, ngram_type: NGramType) -> Result<Self> {
        // Validate confidence thresholds
        if config.ngram_min_confidence <= 0.0 || config.ngram_min_confidence > 1.0 {
            return Err(Error::Config(
                format!("Invalid ngram_min_confidence: {}", config.ngram_min_confidence)
            ));
        }
        
        if config.initial_text_min_confidence <= 0.0 || config.initial_text_min_confidence > 1.0 {
            return Err(Error::Config(
                format!("Invalid text_min_confidence: {}", config.initial_text_min_confidence)
            ));
        }
        if config.merged_text_min_confidence <= 0.0 || config.merged_text_min_confidence > 1.0 {
            return Err(Error::Config(
                format!("Invalid text_min_confidence: {}", config.merged_text_min_confidence)
            ));
        }
    
        // Load banality phrases from file
        let banality_phrases = match load_terms_from_file("filters/banalities_phrases.txt") {
            Ok(phrases) => phrases,
            Err(e) => {
                warn!("Failed to load banality phrases: {}. Using empty set.", e);
                AHashSet::new()
            }
        };
        
        // Load banality words from file
        let banality_words = match load_terms_from_file("filters/banalities_words.txt") {
            Ok(words) => words,
            Err(e) => {
                warn!("Failed to load banality words: {}. Using empty set.", e);
                AHashSet::new()
            }
        };
        
        // Log the loaded data
        info!("Loaded {} banality phrases and {} banality words for similarity calculation", 
              banality_phrases.len(), banality_words.len());
        
        // Default threshold for banality detection - could be made configurable in the future
        let banality_threshold = 0.9;
    
        Ok(Self { 
            config,
            ngram_type,
            char_freq_cache: RwLock::new(AHashMap::with_capacity(MIN_CACHE_SIZE)),
            word_freq_cache: RwLock::new(AHashMap::with_capacity(MIN_CACHE_SIZE)),
            banality_phrases,
            banality_words,
            banality_threshold,
        })
    }

    pub fn get_config(&self) -> &MatcherConfig {
        &self.config
    }

    // In similarity.rs, modify find_matching_ngrams function
    pub fn find_matching_ngrams(
        &self,
        source: &NGramEntry,
        targets: &[NGramEntry],
        storage: Option<&dyn StorageBackend>,
    ) -> Result<Vec<NGramMatch>> {
        // Add tracing for source n-grams
        let is_traced = if log::log_enabled!(log::Level::Info) {
            !self.config.traced_phrase.is_empty() &&
            source.text_content.cleaned_text.contains(&self.config.traced_phrase)
        } else {
            false
        };
        
        // Only log if it's actually traced
        if is_traced {
            info!("TRACE [find_matching_ngrams]: Processing traced source n-gram: '{}' (seq: {})",
            source.text_content.cleaned_text, source.sequence_number);
        }

        // For exact matching, use the simplified ExactMatcher that does direct comparison
        if self.config.ngram_similarity_metric == SimilarityMetric::Exact {
            debug!("Using exact matching with direct comparison");
            
            // Create the ExactMatcher (no storage needed)
            let matcher = ExactMatcher::new();
            let matches = matcher.find_matches(source, targets)?;
            
            // Log the result but don't do any additional work
            if !matches.is_empty() {
                if log::log_enabled!(log::Level::Debug) {
                    debug!("Found {} exact matches with direct comparison", matches.len());
                }
                if is_traced {
                    info!("TRACE [find_matching_ngrams]: Found {} exact matches for traced n-gram", 
                        matches.len());
                }
            } else {
                if log::log_enabled!(log::Level::Debug) {
                    debug!("No exact matches found");
                }
                if is_traced {
                    info!("TRACE [find_matching_ngrams]: No exact matches found for traced n-gram");
                }
            }
            // Always return the matches (which may be empty) for exact matching
            return Ok(matches);
        }

        // If we're using a non-exact similarity metric, we need to compare against the provided targets
        debug!("Using similarity-based comparison with metric: {:?} for {} ngrams", 
            self.config.ngram_similarity_metric,
            self.ngram_type.as_str());
                
        // If we have targets, use them for non-exact matching
        if !targets.is_empty() {
            match self.ngram_type {
                NGramType::Character => {
                    // Use character-based comparison for character ngrams
                    self.process_batch(
                        source,
                        targets,
                        |s, t| {
                            // First check if the character frequencies are similar enough
                            let source_freqs = self.get_char_freqs(&s.text_content.cleaned_text);
                            let target_freqs = self.get_char_freqs(&t.text_content.cleaned_text);
                            
                            if self.char_freqs_similar(&source_freqs, &target_freqs) {
                                self.compare_ngrams(s, t)
                            } else {
 
                                Ok(0.0)
                            }
                        }
                    )
                },
                NGramType::Word => {
                    // Use word-based comparison for word ngrams
                    self.process_batch(
                        source,
                        targets,
                        |s, t| {
                            // Check if word frequencies are similar enough using our new method
                            let source_freqs = self.get_word_freqs(&s.text_content.cleaned_text);
                            let target_freqs = self.get_word_freqs(&t.text_content.cleaned_text);
                            
                            if self.word_freqs_match_threshold(&source_freqs, &target_freqs) {
                                // If words match threshold, proceed with detailed similarity comparison
                                self.compare_ngrams(s, t)
                            } else {
                                
                                Ok(0.0)
                            }
                        }
                    )
                }
            }
        } else if let Some(storage) = storage {
            // For non-exact matching when targets array is empty but storage is available,
            // we need to get potential matches from storage and then apply the appropriate similarity metric

            warn!("Non-exact similarity metric {} being used with empty targets array - this is inefficient",
                self.config.ngram_similarity_metric.as_str());
            
            if is_traced {
                info!("TRACE [find_matching_ngrams]: Using storage to find potential matches for non-exact similarity metric");
            }
            
            // First, get candidates using prefix index (this is just to find potential matches)
            let candidates = match storage.get_sequences_by_prefix_for_ngram(source) {
                Ok(seqs) => seqs,
                Err(e) => {
                    warn!("Failed to get prefix sequences: {}", e);
                    return Ok(Vec::new());
                }
            };
            
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
                        
            // Now get the actual ngrams for these candidate sequences
            let candidate_ngrams = match storage.get_ngrams_batch(&candidates) {
                Ok(ngrams) => ngrams,
                Err(e) => {
                    warn!("Failed to get candidate ngrams: {}", e);
                    return Ok(Vec::new());
                }
            };
                        
            // Now use the process_batch method to apply the appropriate similarity metric
            match self.ngram_type {
                NGramType::Character => {
                    self.process_batch(
                        source,
                        &candidate_ngrams,
                        |s, t| {
                            let source_freqs = self.get_char_freqs(&s.text_content.cleaned_text);
                            let target_freqs = self.get_char_freqs(&t.text_content.cleaned_text);
                            
                            if self.char_freqs_similar(&source_freqs, &target_freqs) {
                                self.compare_ngrams(s, t)
                            } else {
                                Ok(0.0)
                            }
                        }
                    )
                },
                NGramType::Word => {
                    self.process_batch(
                        source,
                        &candidate_ngrams,
                        |s, t| {
                            let source_freqs = self.get_word_freqs(&s.text_content.cleaned_text);
                            let target_freqs = self.get_word_freqs(&t.text_content.cleaned_text);
                            
                            if self.word_freqs_match_threshold(&source_freqs, &target_freqs) {
                                self.compare_ngrams(s, t)
                            } else {
                                Ok(0.0)
                            }
                        }
                    )
                }
            }
        } else {
            // If we have no targets and no storage, we can't do anything
            warn!("No targets and no storage available for non-exact matching");
            Ok(Vec::new())
        }
    }
    fn process_batch<F>(
        &self,
        source: &NGramEntry,
        targets: &[NGramEntry],
        compare: F
    ) -> Result<Vec<NGramMatch>> 
    where
        // F is a function that takes two NGramEntry references and returns a similarity score
        F: Fn(&NGramEntry, &NGramEntry) -> Result<f64>
    {
        // Initialize our results vector. We don't know exactly how many matches we'll find,
        // but we can make a reasonable guess to avoid frequent reallocations
        let mut matches = Vec::with_capacity(targets.len() / 4);  // Assume ~25% might match
        
        // We process targets in chunks to avoid using too much memory at once
        let batch_size = 1000;  // Process 1000 targets at a time
               
        // Iterate over our targets in chunks of batch_size
        for target_batch in targets.chunks(batch_size) {
                        
            // For each target in this batch, calculate its similarity with our source
            for target in target_batch {
                // Use the provided comparison function to determine similarity
                // For exact matches, this will always return 1.0
                // For similarity matching, this does the actual similarity calculation
                match compare(source, target) {
                    Ok(similarity) => {
                        // Only create matches that meet our confidence threshold
                        if similarity >= self.config.ngram_min_confidence {
                                                       
                            matches.push(NGramMatch {
                                // Store positions for both source and target
                                // These are crucial for chain building later
                                source_position: source.sequence_number as usize,
                                target_position: target.sequence_number as usize,
                                // Keep the original ngram for reference
                                ngram: source.clone(),
                                // Store the calculated similarity score
                                similarity,
                            });
                        } else {
                        }
                    },
                    Err(e) => {
                        // If comparison fails, log it but continue processing other matches
                        warn!("Error comparing ngrams: {}. Continuing with next target.", e);
                        continue;
                    }
                }
            }
            
        }
        
        info!("Completed processing. Found {} matches above threshold {}",
            matches.len(),
            self.config.ngram_min_confidence
        );
        
        Ok(matches)
    }

    pub fn compare_ngrams(&self, source: &NGramEntry, target: &NGramEntry) -> Result<f64> {
        let algorithm = self.get_algorithm(self.config.ngram_similarity_metric)?;
        algorithm.compare_ngrams(source, target)
    }

    pub fn compare_spans(&self, source: &TextSpan, target: &TextSpan) -> Result<f64> {
        let algorithm = self.get_algorithm(self.config.text_similarity_metric)?;
        algorithm.compare_spans(source, target)
    }

    pub fn compare_texts(&self, source: &str, target: &str) -> Result<f64> {
        trace!("Comparing texts using {:?} metric with frequency-based approach", 
              self.config.text_similarity_metric);
        
        // Character-level comparison (unchanged)
        let algorithm = self.get_algorithm(self.config.text_similarity_metric)?;
        let char_similarity = algorithm.compare_texts(source, target)?;
        
        // Word-level comparison
        let source_words: Vec<&str> = source.split_whitespace().collect();
        let target_words: Vec<&str> = target.split_whitespace().collect();
        
        let lcs_len = self.longest_common_subsequence(&source_words, &target_words);
        let raw_word_similarity = if source_words.len() + target_words.len() > 0 {
            2.0 * lcs_len as f64 / (source_words.len() + target_words.len()) as f64
        } else {
            0.0
        };
        
        // Check banality of source text
        let (is_banal, banality_density) = self.check_banality(source);
        
        // Calculate banality factor: For banal text, use 0.5, otherwise scale based on density
        let banality_factor = if is_banal {
            0.5  // Half the weight for banal text as requested
        } else {
            // Gradually decrease weight as banality increases, down to 0.7 at threshold
            1.0 - (banality_density / self.banality_threshold * 0.3)
        };
        
        // Apply banality factor to word similarity
        let word_similarity = raw_word_similarity * banality_factor;
        
        // Weighted average (word dominant as before)
        let word_weight = 0.7;
        let similarity = (word_similarity * word_weight) + (char_similarity * (1.0 - word_weight));
       
        Ok(similarity)
    }

    // Helper methods
    fn get_algorithm(&self, metric: SimilarityMetric) -> Result<Box<dyn SimilarityAlgorithm>> {
        super::algorithms::SimilarityAlgorithmFactory::create(metric, &self.config)
    }

    fn get_char_freqs(&self, text: &str) -> CharFreqs {
        // Check cache first
        {
            let cache_read = self.char_freq_cache.read().unwrap();
            if let Some(freqs) = cache_read.get(text) {
                // Trace statement removed
                return freqs.clone();
            }
        }
    
        // Calculate frequencies if not cached
        // Trace statement removed
        let mut freqs = AHashMap::new();
        for c in text.chars() {
            *freqs.entry(c).or_insert(0) += 1;
        }
    
        // Cache result with eviction policy
        let mut cache = self.char_freq_cache.write().unwrap();
        
        // If cache is getting too large, evict oldest entries
        if cache.len() >= MAX_CACHE_SIZE {
            // Trace statement removed
            
            // Get keys sorted by some deterministic order (here we use the strings themselves)
            let mut keys: Vec<String> = cache.keys().cloned().collect();
            keys.sort(); // Sort keys deterministically
            
            // Calculate how many entries to remove (25% of max size)
            let entries_to_remove = (MAX_CACHE_SIZE as f64 * EVICTION_PERCENTAGE) as usize;
            
            // Remove the first N entries
            for key in keys.iter().take(entries_to_remove) {
                cache.remove(key);
            }
            
            // Trace statement removed
        }
        
        // Add the new entry
        cache.insert(text.to_string(), freqs.clone());
        freqs
    }

    fn char_freqs_similar(&self, source: &CharFreqs, target: &CharFreqs) -> bool {
        let mut total_diff = 0;
        let mut total_chars = 0;
    
        let all_chars: AHashSet<_> = source.keys().chain(target.keys()).collect();
        
        for &c in all_chars.iter() {
            let source_count = source.get(&c).copied().unwrap_or(0);
            let target_count = target.get(&c).copied().unwrap_or(0);
            
            total_diff += (source_count as i32 - target_count as i32).abs() as usize;
            total_chars += source_count.max(target_count);
        }
    
        if total_chars == 0 {
            return false;
        }
        
        // Calculate BOTH dissimilarity and similarity for clearer debugging
        let dissimilarity = total_diff as f64 / total_chars as f64;
               
        dissimilarity <= CHAR_FREQ_THRESHOLD
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
    fn get_word_freqs(&self, text: &str) -> WordFreqs {
        // Check cache first
        {
            let cache_read = self.word_freq_cache.read().unwrap();
            if let Some(freqs) = cache_read.get(text) {
                return freqs.clone();
            }
        }
                
        // Split text into words and count frequencies
        let mut freqs = AHashMap::new();
        for word in text.split_whitespace() {
            *freqs.entry(word.to_string()).or_insert(0) += 1;
        }
        
        // Cache result with eviction policy
        let mut cache = self.word_freq_cache.write().unwrap();
        
        // If cache is getting too large, evict oldest entries
        if cache.len() >= MAX_CACHE_SIZE {
            trace!("Word cache reached maximum size ({}), evicting oldest entries", MAX_CACHE_SIZE);
            
            // Get keys sorted by some deterministic order
            let mut keys: Vec<String> = cache.keys().cloned().collect();
            keys.sort();
            
            // Calculate how many entries to remove
            let entries_to_remove = (MAX_CACHE_SIZE as f64 * EVICTION_PERCENTAGE) as usize;
            
            // Remove the first N entries
            for key in keys.iter().take(entries_to_remove) {
                cache.remove(key);
            }
            
        }
        
        // Add the new entry
        cache.insert(text.to_string(), freqs.clone());
        
        freqs
    }
    fn check_banality(&self, text: &str) -> (bool, f64) {
        // Skip empty strings
        if text.is_empty() {
            return (false, 0.0);
        }
        
        // Split text into words
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        
        if word_count == 0 {
            return (false, 0.0);
        }
        
        // Count matched words and phrases
        let mut matching_words = 0;
        let mut covered_positions = vec![false; word_count]; // Track which words are part of phrases
        
        // Check phrases first
        for phrase in &self.banality_phrases {
            if text.contains(phrase) {
                // For each phrase, find and mark all occurrences in the text
                let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                if phrase_words.is_empty() {
                    continue;
                }
                
                // For each potential starting position
                for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                    let mut matches = true;
                    
                    // Check if phrase matches at this position
                    for (offset, phrase_word) in phrase_words.iter().enumerate() {
                        if start_pos + offset >= word_count || words[start_pos + offset] != *phrase_word {
                            matches = false;
                            break;
                        }
                    }
                    
                    // If phrase matches, mark all its words as covered
                    if matches {
                        for offset in 0..phrase_words.len() {
                            if !covered_positions[start_pos + offset] {
                                covered_positions[start_pos + offset] = true;
                                matching_words += 1;
                            }
                        }
                    }
                }
            }
        }
        
        // Then check individual words, only counting those not covered by phrases
        for (pos, word) in words.iter().enumerate() {
            if !covered_positions[pos] && self.banality_words.contains(&word.to_string()) {
                matching_words += 1;
            }
        }
        
        // Calculate density of banal content
        let density = matching_words as f64 / word_count as f64;
        
        // Determine if text is banal based on threshold
        let is_banal = density >= self.banality_threshold;
                   
        (is_banal, density)
    }
    
    // Compare word frequencies for word-based NGrams
    fn word_freqs_match_threshold(&self, source: &WordFreqs, target: &WordFreqs) -> bool {
        // Check if pre-filtering is disabled via configuration
        if self.config.word_match_min_ratio <= 0.0 {
            // If ratio is set to 0, bypass word-level filtering completely
            return true;
        }
        
        // For empty texts, no match
        if source.is_empty() || target.is_empty() {
            return false;
        }
        
        // Find count of words that appear in both
        let mut shared_words = 0;
        for (word, _) in source.iter() {
            if target.contains_key(word) {
                shared_words += 1;
            }
        }
        
        // Get total count of unique words
        let source_word_count = source.len();
        
        // Calculate minimum matching words based on configuration
        let min_ratio = self.config.word_match_min_ratio;
        let min_matching_words = (source_word_count as f64 * min_ratio).ceil() as usize;
        
        let result = shared_words >= min_matching_words;
               
        result
    }
}