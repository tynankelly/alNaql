use std::fs::File;
use std::io::{BufRead, BufReader};
use ahash::{AHashSet, AHashMap};
use log::{info, debug, trace, warn};
use crate::error::Result;
use crate::storage::LMDBStorage;
use crate::config::{MatchingConfig, NGramType};
use crate::config::AlNaqlConfig;
use crate::config::generation::GenerationConfig;

pub struct TextClassifier {
    // Isnad detection fields
    isnad_phrases: AHashSet<String>,
    isnad_words: AHashSet<String>,
    isnad_density_threshold: f64,
    isnad_min_length: usize,
    
    // Banality detection fields
    banality_phrases: AHashSet<String>,
    banality_words: AHashSet<String>,
    banality_density_threshold: f64,
    banality_detection_mode: String,
    
    // Automatic banality detection (passed from matcher level)
    common_ngrams: Option<AutoBanal>,
}

impl TextClassifier {
    pub fn new(
        config: &AlNaqlConfig,
        common_ngrams: Option<AutoBanal>  // Pre-analyzed at matcher level
    ) -> Result<Self> {
        // Load phrase files
        let isnad_phrases = Self::load_terms_from_file(&config.matcher.isnad_phrases_file)
            .unwrap_or_else(|e| {
                warn!("Failed to load isnad phrases: {}. Proceeding with empty set.", e);
                AHashSet::new()
            });
            
        let isnad_words = Self::load_terms_from_file(&config.matcher.isnad_words_file)
            .unwrap_or_else(|e| {
                warn!("Failed to load isnad words: {}. Proceeding with empty set.", e);
                AHashSet::new()
            });
            
        let banality_phrases = Self::load_terms_from_file(&config.matcher.banality_phrases_file)
            .unwrap_or_else(|e| {
                warn!("Failed to load banality phrases: {}. Proceeding with empty set.", e);
                AHashSet::new()
            });
            
        let banality_words = Self::load_terms_from_file(&config.matcher.banality_words_file)
            .unwrap_or_else(|e| {
                warn!("Failed to load banality words: {}. Proceeding with empty set.", e);
                AHashSet::new()
            });
        
        // Add to config at a later point
        let isnad_density_threshold = config.matcher.isnad_density_threshold;
        let isnad_min_length = config.matcher.isnad_min_length;
        let banality_density_threshold = config.matcher.banality_density_threshold;
        
        Ok(Self {
            isnad_phrases,
            isnad_words,
            isnad_density_threshold,
            isnad_min_length,
            banality_phrases,
            banality_words,
            banality_density_threshold,
            banality_detection_mode: config.matcher.banality_detection_mode.clone(),
            common_ngrams,
        })
    }

    fn load_terms_from_file(path: &str) -> Result<AHashSet<String>> {
               
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut terms = AHashSet::new();
        
        for line in reader.lines() {
            let line = line?;
            let term = line.trim().to_string();
            
            // Skip empty lines and comments
            if !term.is_empty() && !term.starts_with('#') {
                terms.insert(term);
            }
        }
        
        Ok(terms)
    }

    pub fn is_likely_isnad(&self, text: &str, config: &GenerationConfig, traced_phrase: &str) 
        -> (bool, bool, f64) {          // Different approach based on ngram type
        match config.ngram_type {
            NGramType::Character => {
                // For character ngrams, use direct character comparison
                let total_chars = text.chars().count();
                
                if total_chars == 0 {
                    return (false, false, 0.0);
                }
                
                // Check if this text contains our traced n-gram - but only if tracing is enabled
                let has_traced = if traced_phrase.is_empty() {
                    false
                } else {
                    text.contains(traced_phrase)
                };
                
                // Check for exact phrases first (direct substring matching)
                let mut matched_chars = 0;
                let mut matched_items = Vec::new();
                
                // Look for isnad phrases in the text
                for phrase in &self.isnad_phrases {
                    if text.contains(phrase) {
                        matched_chars += phrase.chars().count();
                        
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                    }
                }
                
                // Avoid overcounting
                matched_chars = matched_chars.min(total_chars);
                let density = matched_chars as f64 / total_chars as f64;
            let estimated_word_count = text.matches(' ').count() + 1;
            
            let is_isnad = density >= self.isnad_density_threshold;
            let is_short = is_isnad && estimated_word_count < self.isnad_min_length;
            
            if is_isnad {
                trace!("Found isnad by character density: {:.2} ({} of {} chars, ~{} words, short: {})",
                    density, matched_chars, total_chars, estimated_word_count, is_short);
            }
            
            (is_isnad, is_short, density)
        },
            
            NGramType::Word => {
                // Original word-based implementation
                let words: Vec<&str> = text.split_whitespace().collect();
                let word_count = words.len();
                
                if word_count == 0 {
                    return (false, false, 0.0);
                }
                
                // Check if this text contains our traced n-gram - but only if tracing is enabled
                let has_traced = if traced_phrase.is_empty() {
                    false
                } else {
                    text.contains(traced_phrase)
                };
                
                // Track which words have been covered by phrases
                let mut covered_positions = vec![false; word_count];
                let mut matching_words = 0;
                let mut matched_items = Vec::new();
                
                // First check for isnad phrases
                for phrase in &self.isnad_phrases {
                    if text.contains(phrase) {
                        if has_traced {
                            matched_items.push(format!("phrase: '{}'", phrase));
                        }
                        
                        // Find all occurrences of this phrase in the text
                        let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                        
                        // For each potential starting position in the text
                        for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                            let mut matches = true;
                            
                            // Check if phrase matches at this position
                            for (offset, phrase_word) in phrase_words.iter().enumerate() {
                                if start_pos + offset >= word_count || words[start_pos + offset] != *phrase_word {
                                    matches = false;
                                    break;
                                }
                            }
                            
                            // If phrase matches, mark all its positions as covered
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
                
                // Then check for individual words, only counting those not already covered by phrases
                for (pos, word) in words.iter().enumerate() {
                    if !covered_positions[pos] && self.isnad_words.contains(&word.to_string()) {
                        matching_words += 1;
                        covered_positions[pos] = true;
                        
                        if has_traced {
                            matched_items.push(format!("word: '{}'", word));
                        }
                    }
                }
                
                let density = matching_words as f64 / word_count as f64;
            
                let is_isnad = density >= self.isnad_density_threshold;
                let is_short = is_isnad && word_count < self.isnad_min_length;
                
                if is_isnad {
                    trace!("Found isnad by word density: {:.2} ({} of {} words, short: {})",
                        density, matching_words, word_count, is_short);
                }
                
                (is_isnad, is_short, density)
            }
        }
    }

    pub fn is_likely_banality(
        &self, 
        text: &str, 
        config: &AlNaqlConfig, 
        traced_phrase: &str,
        sequence_range: Option<(u64, u64)>,
        storage: Option<&LMDBStorage>,
    ) -> (bool, f64, usize) {
        trace!("BANALITY CHECK: Checking text: '{}'", text);
        
        if text.is_empty() {
            return (false, 0.0, 0);
        }
        
        // Check if this text contains our traced phrase for debugging
        let has_traced = !traced_phrase.is_empty() && text.contains(traced_phrase);
        
        // Choose detection method based on configuration
        match self.banality_detection_mode.as_str() {
            "automatic" => {
                // Use percentile-based automatic detection
                let word_count = text.split_whitespace().count();
                
                if let (Some((start_seq, end_seq)), Some(storage)) = (sequence_range, storage) {
                    // Get all sequences in this range
                    let sequences: Vec<u64> = (start_seq..=end_seq).collect();
                    
                    // Fetch percentiles for all sequences
                    let percentiles = match storage.get_percentiles_batch(&sequences) {
                        Ok(p) => p,
                        Err(e) => {
                            debug!("Failed to fetch percentiles: {}", e);
                            return (false, 0.0, word_count);
                        }
                    };
                    
                    if percentiles.is_empty() {
                        debug!("No percentile data available for sequences {}..{}", start_seq, end_seq);
                        return (false, 0.0, word_count);
                    }
                    
                    // Calculate percentile threshold from banality_auto_proportion
                    // If proportion is 5%, then percentile threshold is 95 (100 - 5)
                    let percentile_threshold = (100.0 - config.matcher.banality_auto_proportion) as u8;
                    
                    // Count how many ngrams are above the percentile threshold
                    let banal_count = percentiles.values()
                        .filter(|&&p| p >= percentile_threshold)
                        .count();
                    
                    // Calculate density (proportion of banal ngrams)
                    let density = banal_count as f64 / sequences.len() as f64;
                    
                    // Check if density exceeds the configured threshold
                    // banality_density_threshold is already 0.0-1.0 in config
                    let is_banal = density >= config.matcher.banality_density_threshold;
                    
                    if has_traced {
                        if is_banal {
                            info!("TRACE [automatic_banality]: Traced phrase detected as banal");
                        }
                        debug!("TRACE: {} of {} ngrams ({:.1}%) are ≥{}th percentile (threshold: {:.1}%)",
                            banal_count, sequences.len(), 
                            density * 100.0,
                            percentile_threshold,
                            config.matcher.banality_density_threshold * 100.0
                        );
                    }
                    
                    trace!("Automatic banality check: banal={}, density={:.2}", is_banal, density);
                    (is_banal, density, word_count)
                } else {
                    // Automatic mode configured but no sequence data available
                    warn!("Automatic banality detection requested but no sequence data available");
                    (false, 0.0, word_count)
                }
            },
            
            "phrase" | "manual" | _ => {
                // [EXACT SAME PHRASE LOGIC AS BEFORE - NO CHANGES]
                match config.generator.ngram_type {
                    NGramType::Character => {
                        // Character-based phrase detection logic
                        let total_chars = text.chars().count();
                        if total_chars == 0 {
                            return (false, 0.0, 0);
                        }
                        
                        let mut matched_chars = 0;
                        let mut matched_items = Vec::new();
                        
                        // Check for banality phrases in the text
                        for phrase in &self.banality_phrases {
                            if text.contains(phrase) {
                                matched_chars += phrase.chars().count();
                                if has_traced {
                                    matched_items.push(format!("phrase: '{}'", phrase));
                                }
                            }
                        }
                        
                        // Prevent overcounting
                        matched_chars = matched_chars.min(total_chars);
                        let density = matched_chars as f64 / total_chars as f64;
                        
                        // Estimate word count for character ngrams
                        let estimated_word_count = text.matches(' ').count() + 1;
                        
                        let is_banal = density >= self.banality_density_threshold;
                        
                        if is_banal && has_traced {
                            info!("TRACE [phrase_banality]: Traced phrase detected as banal with density {:.2}", 
                                density);
                        }
                        
                        trace!("Phrase banality check (char): banal={}, density={:.2}", is_banal, density);
                        (is_banal, density, estimated_word_count)
                    },
                    
                    NGramType::Word => {
                        // Word-based phrase detection logic
                        let words: Vec<&str> = text.split_whitespace().collect();
                        let word_count = words.len();
                        
                        if word_count == 0 {
                            return (false, 0.0, 0);
                        }
                        
                        // Track which words are covered by phrases
                        let mut covered_positions = vec![false; word_count];
                        let mut matching_words = 0;
                        let mut matched_items = Vec::new();
                        
                        // Check for banality phrases first
                        for phrase in &self.banality_phrases {
                            if text.contains(phrase) {
                                if has_traced {
                                    matched_items.push(format!("phrase: '{}'", phrase));
                                }
                                
                                let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                                
                                // Find all occurrences of this phrase
                                for start_pos in 0..=word_count.saturating_sub(phrase_words.len()) {
                                    let mut matches = true;
                                    
                                    for (offset, phrase_word) in phrase_words.iter().enumerate() {
                                        if start_pos + offset >= word_count || 
                                        words[start_pos + offset] != *phrase_word {
                                            matches = false;
                                            break;
                                        }
                                    }
                                    
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
                        
                        // Check individual words not already covered
                        for (pos, word) in words.iter().enumerate() {
                            if !covered_positions[pos] && 
                            self.banality_words.contains(&word.to_string()) {
                                matching_words += 1;
                                covered_positions[pos] = true;
                                if has_traced {
                                    matched_items.push(format!("word: '{}'", word));
                                }
                            }
                        }
                        
                        let density = matching_words as f64 / word_count as f64;
                        let is_banal = density >= self.banality_density_threshold;
                        
                        if is_banal && has_traced {
                            info!("TRACE [phrase_banality]: Traced phrase detected as banal with density {:.2}", 
                                density);
                        }
                        
                        trace!("Phrase banality check (word): banal={}, density={:.2}", is_banal, density);
                        (is_banal, density, word_count)
                    }
                }
            }
        }
    }

    pub fn has_auto_banal(&self) -> bool {
        self.common_ngrams.is_some()
    }
    
    pub fn set_auto_banal(&mut self, auto_banal: Option<AutoBanal>) {
        self.common_ngrams = auto_banal;
    }

}

/// Manages the automatic detection of banalities based on n-gram frequency analysis.
/// 
/// This structure identifies the most common n-grams in a corpus and uses them
/// to detect potentially banal (formulaic or boilerplate) text passages based
/// on the concentration of common n-grams they contain.
pub struct AutoBanal {
    /// Map of common text strings to their sequence numbers
    /// This allows us to quickly check if a piece of text is common
    common_texts: AHashMap<String, Vec<u64>>,
    
    /// Set of all sequence numbers that contain common text
    /// Kept for backward compatibility with existing checks
    common_ngram_ids: AHashSet<u64>,
    
    /// Total number of unique texts analyzed
    total_unique_texts: usize,
    
    /// Total number of ngram sequences processed
    total_sequences: usize,
    
    /// Proportion of most common n-grams being tracked (0-100%)
    proportion: f64,
    
    /// Threshold percentage for determining if a text is banal (0-100%)
    threshold: f64,
}

impl AutoBanal {
    /// Creates a new CommonNgrams instance with the specified parameters.
    /// 
    /// # Arguments
    /// * `proportion` - Percentage of most common n-grams to consider (0-100)
    /// * `threshold` - Threshold percentage for determining if text is banal (0-100)
    pub fn new(proportion: f64, threshold: f64) -> Self {
        Self {
            common_texts: AHashMap::new(),
            common_ngram_ids: AHashSet::new(),
            total_unique_texts: 0,
            total_sequences: 0,
            proportion,
            threshold,
        }
    }

    /// Analyzes both source and target databases to identify the most common n-grams
    pub fn analyze_databases(
        &mut self, 
        source_storage: &LMDBStorage,
        target_storage: &LMDBStorage,
        config: &MatchingConfig,
    ) -> Result<()> {
        info!("Starting automatic banality analysis across both databases");
        
        // We'll build a frequency map of text content
        // The key is the ngram text, the value is all sequences containing that text
        let mut text_frequency: AHashMap<String, Vec<(u64, bool)>> = AHashMap::new();
        // The bool indicates whether it's from source (true) or target (false)
        
        // Process source database
        self.analyze_single_database(
            source_storage, 
            &mut text_frequency, 
            true,  // is_source = true
            "source",
            &config,
        )?;
        
        // Process target database
        self.analyze_single_database(
            target_storage, 
            &mut text_frequency, 
            false,  // is_source = false
            "target",
            &config
        )?;
        
        // Now we have a complete frequency map
        // Let's identify the most common texts
        self.identify_common_texts(text_frequency)?;
        
        info!("Automatic banality detection ready: {} unique common texts covering {} sequences",
            self.common_texts.len(), self.common_ngram_ids.len());
        
        Ok(())
    }
    
    fn analyze_single_database(
        &mut self,
        storage: &LMDBStorage,
        text_frequency: &mut AHashMap<String, Vec<(u64, bool)>>,
        is_source: bool,
        db_name: &str,
        config: &MatchingConfig
    ) -> Result<()> {
        debug!("Analyzing {} database for common ngrams", db_name);
        
        let stats = storage.get_db_info()?;
        let batch_size: u64 = config.auto_banal_batch_size as u64;
        let mut processed: u64 = 0;
        
        for start_seq in (0..stats.total_ngrams).step_by(batch_size as usize) {
            let end_seq = (start_seq + batch_size).min(stats.total_ngrams);
            let sequences: Vec<u64> = (start_seq..end_seq).collect();
            
            // Fetch the text content for this batch
            // This is the key operation - we only fetch the text column
            let text_batch = storage.get_text_bytes_batch(&sequences)?;
            
            // Count frequency of each text
            for (seq_num, text_bytes) in text_batch {
                if let Ok(text) = String::from_utf8(text_bytes) {
                    // Clean the text for comparison (normalize whitespace, etc.)
                    let cleaned_text = text.trim().to_string();
                    
                    if !cleaned_text.is_empty() {
                        text_frequency
                            .entry(cleaned_text)
                            .or_insert_with(Vec::new)
                            .push((seq_num, is_source));
                    }
                }
            }
            
            processed += sequences.len() as u64;
            if processed % 500_000 == 0 {
                debug!("Processed {}/{} ngrams from {} ({:.1}%)", 
                    processed, stats.total_ngrams, db_name,
                    (processed as f64 / stats.total_ngrams as f64) * 100.0);
            }
        }
        
        self.total_sequences += stats.total_ngrams as usize;
        debug!("Completed {} database: {} sequences processed", db_name, stats.total_ngrams);
        
        Ok(())
    }
    
    fn identify_common_texts(
        &mut self, 
        text_frequency: AHashMap<String, Vec<(u64, bool)>>
    ) -> Result<()> {
        // Filter to only texts that appear multiple times
        // A text that appears only once isn't "common"
        let repeated_texts: Vec<(String, Vec<(u64, bool)>)> = text_frequency
            .into_iter()
            .filter(|(_, occurrences)| occurrences.len() > 1)
            .collect();
        
        self.total_unique_texts = repeated_texts.len();
        debug!("Found {} unique texts that appear multiple times", self.total_unique_texts);
        
        // Sort by frequency (most common first)
        let mut frequency_vec = repeated_texts;
        frequency_vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        
        // Log the top 10 most common for debugging
        if log::log_enabled!(log::Level::Debug) {
            debug!("Top 10 most common ngrams:");
            for (text, occurrences) in frequency_vec.iter().take(10) {
                debug!("  '{}' - {} occurrences", 
                    if text.len() > 50 { 
                        format!("{}...", &text[..50]) 
                    } else { 
                        text.clone() 
                    },
                    occurrences.len()
                );
            }
        }
        
        // Calculate how many texts to keep as "common"
        let keep_count = ((self.total_unique_texts as f64 * self.proportion) / 100.0)
            .ceil() as usize;
        
        info!("Keeping top {:.1}% ({} out of {}) most frequent texts as banality indicators",
            self.proportion, keep_count, self.total_unique_texts);
        
        // Clear previous data
        self.common_texts.clear();
        self.common_ngram_ids.clear();
        
        // Store the most common texts and their sequence numbers
        for (text, occurrences) in frequency_vec.into_iter().take(keep_count) {
            // Extract just the sequence numbers, dropping the source/target flag
            let seq_numbers: Vec<u64> = occurrences.iter()
                .map(|(seq_num, _is_source)| *seq_num)
                .collect();
            
            // Store all sequence numbers for this common text
            for seq_num in &seq_numbers {
                self.common_ngram_ids.insert(*seq_num);
            }
            
            // Store the text with just the sequence numbers
            self.common_texts.insert(text, seq_numbers);
        }
        
        Ok(())
    }

    /// Checks if a reconstructed text segment is banal based on common ngram overlap
    pub fn is_text_banal(
        &self, 
        text: &str,
        ngram_type: NGramType,
        ngram_size: usize,
        stride: usize,
    ) -> (bool, f64) {
        if text.is_empty() || self.common_texts.is_empty() {
            return (false, 0.0);
        }
        
        // We need to break the text back into ngrams to check for overlap
        // This mimics what the original ngram generation did
        let text_ngrams = match ngram_type {
            NGramType::Word => {
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.len() < ngram_size {
                    return (false, 0.0);
                }
                
                let mut ngrams = Vec::new();
                for i in (0..=words.len().saturating_sub(ngram_size)).step_by(stride) {
                    let ngram_text = words[i..i+ngram_size].join(" ");
                    ngrams.push(ngram_text);
                }
                ngrams
            },
            NGramType::Character => {
                let chars: Vec<char> = text.chars().collect();
                if chars.len() < ngram_size {
                    return (false, 0.0);
                }
                
                let mut ngrams = Vec::new();
                for i in (0..=chars.len().saturating_sub(ngram_size)).step_by(stride) {
                    let ngram_text: String = chars[i..i+ngram_size].iter().collect();
                    ngrams.push(ngram_text);
                }
                ngrams
            }
        };
        
        // Count how many of these ngrams are in our common set
        let common_count = text_ngrams.iter()
            .filter(|ngram| self.common_texts.contains_key(*ngram))
            .count();
        
        // Calculate density as percentage
        let density = if !text_ngrams.is_empty() {
            (common_count as f64 / text_ngrams.len() as f64) * 100.0
        } else {
            0.0
        };
        
        // Determine if banal based on threshold
        let is_banal = density >= self.threshold;
        
        if is_banal {
            debug!("Text detected as banal: {:.1}% of its ngrams are common (threshold: {:.1}%)",
                density, self.threshold);
        }
        
        (is_banal, density)
    }

    /// Returns the number of unique text patterns identified as common
    /// This tells us how many distinct ngram texts are in our "common" set
    pub fn common_text_count(&self) -> usize {
        self.common_texts.len()
    }
    
    /// Returns the number of sequence IDs that contain common text
    /// This can be larger than common_text_count because one text can appear in many sequences
    pub fn common_ngram_count(&self) -> usize {
        self.common_ngram_ids.len()
    }
    
    /// Returns the total number of unique texts that were analyzed
    /// This includes both common and uncommon texts that appeared more than once
    pub fn unique_texts_analyzed(&self) -> usize {
        self.total_unique_texts
    }
    
    /// Returns the total number of sequences processed during analysis
    /// This is the raw count of all ngrams examined from both databases
    pub fn total_sequences_processed(&self) -> usize {
        self.total_sequences
    }
    
    /// Provides a human-readable summary of the analysis results
    /// Useful for logging and debugging to understand what was found
    pub fn summary(&self) -> String {
        format!(
            "Analyzed {} sequences, found {} unique repeated texts, tracking {} common patterns covering {} sequence IDs",
            self.total_sequences,
            self.total_unique_texts,
            self.common_texts.len(),
            self.common_ngram_ids.len()
        )
    }
}