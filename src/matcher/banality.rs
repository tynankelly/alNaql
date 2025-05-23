use ahash::{AHashMap, AHashSet};
use log::{info, debug, trace}; // Removed unused 'warn'
use crate::error::Result;
// Removed unused NGramEntry import
use crate::storage::StorageBackend;

/// Manages the automatic detection of banalities based on n-gram frequency analysis.
/// 
/// This structure identifies the most common n-grams in a corpus and uses them
/// to detect potentially banal (formulaic or boilerplate) text passages based
/// on the concentration of common n-grams they contain.
pub struct CommonNgrams {
    /// Set of n-gram IDs considered common (potentially banal)
    common_ngrams: AHashSet<u64>,
    
    /// Total number of unique n-grams analyzed
    total_analyzed: usize,
    
    /// Proportion of most common n-grams being tracked (0-100%)
    proportion: f64,
    
    /// Threshold percentage for determining if a text is banal (0-100%)
    threshold: f64,
}

impl CommonNgrams {
    /// Creates a new CommonNgrams instance with the specified parameters.
    /// 
    /// # Arguments
    /// * `proportion` - Percentage of most common n-grams to consider (0-100)
    /// * `threshold` - Threshold percentage for determining if text is banal (0-100)
    pub fn new(proportion: f64, threshold: f64) -> Self {
        Self {
            common_ngrams: AHashSet::new(),
            total_analyzed: 0,
            proportion: proportion,
            threshold: threshold,
        }
    }
    
    /// Analyzes a database to identify the most common n-grams.
    /// 
    /// This method examines the n-grams in the provided storage backend,
    /// counts their frequencies, and identifies the most common ones
    /// based on the configured proportion.
    pub fn analyze_database<S: StorageBackend>(&mut self, storage: &S) -> Result<()> {
        debug!("Starting n-gram frequency analysis for automatic banality detection");
        
        // Get database statistics to understand the scale
        let stats = storage.get_stats()?;
        info!("Analyzing database with {} n-grams", stats.total_ngrams);
        
        // Create a frequency map for counting n-grams
        let mut frequency_map: AHashMap<u64, usize> = AHashMap::new();
        
        // Process n-grams in batches to avoid memory issues
        let batch_size: u64 = 10_000; // Changed to u64 to match the database
        let mut processed: u64 = 0;
        
        // Sequentially process the entire database in chunks
        for start_seq in (0..stats.total_ngrams).step_by(batch_size as usize) {
            let end_seq = (start_seq + batch_size).min(stats.total_ngrams);
            
            // Get a batch of n-grams
            let ngrams = storage.get_sequence_range(start_seq..end_seq)?;
            
            // Count frequencies
            for ngram in &ngrams {
                let ngram_id = ngram.sequence_number;
                *frequency_map.entry(ngram_id).or_insert(0) += 1;
            }
            
            processed += ngrams.len() as u64;
            if processed % 100_000 == 0 {
                debug!("Processed {}/{} n-grams ({:.1}%)", 
                      processed, stats.total_ngrams, 
                      (processed as f64 / stats.total_ngrams as f64) * 100.0);
            }
        }
        
        self.total_analyzed = frequency_map.len();
        info!("Analyzed {} unique n-grams", self.total_analyzed);
        
        // Sort n-grams by frequency (highest first)
        let mut frequency_vec: Vec<(u64, usize)> = frequency_map.into_iter().collect();
        frequency_vec.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Calculate how many n-grams to keep
        let keep_count = ((self.total_analyzed as f64 * self.proportion) / 100.0).ceil() as usize;
        info!("Keeping top {:.1}% ({} out of {}) n-grams as common", 
             self.proportion, keep_count, self.total_analyzed);
        
        // Keep only the most common n-grams
        self.common_ngrams = frequency_vec.into_iter()
            .take(keep_count)
            .map(|(id, _)| id)
            .collect();
        
        debug!("Automatic banality detection ready with {} common n-grams", 
              self.common_ngrams.len());
        
        Ok(())
    }
    
    /// Determines if a text is likely banal based on its n-grams.
    /// 
    /// # Arguments
    /// * `ngram_ids` - Array of n-gram IDs from the text being analyzed
    /// 
    /// # Returns
    /// A tuple containing:
    /// * A boolean indicating if the text is considered banal
    /// * The calculated density (percentage) of common n-grams in the text
    pub fn is_likely_banal(&self, ngram_ids: &[u64]) -> (bool, f64) {
        if ngram_ids.is_empty() || self.common_ngrams.is_empty() {
            return (false, 0.0);
        }
        
        // Count how many n-grams are in the common set
        let common_count = ngram_ids.iter()
            .filter(|&&id| self.common_ngrams.contains(&id))
            .count();
            
        // Calculate density percentage
        let density = (common_count as f64 / ngram_ids.len() as f64) * 100.0;
        
        // Log detailed information for debugging
        trace!("Banality analysis: {}/{} n-grams ({:.2}%) are common (threshold: {:.1}%)",
              common_count, ngram_ids.len(), density, self.threshold);
        
        // Determine if banal based on threshold
        let is_banal = density >= self.threshold;
        
        if is_banal {
            debug!("Text detected as banal with {:.2}% common n-grams (threshold: {:.1}%)",
                 density, self.threshold);
        }
        
        (is_banal, density)
    }
    
    /// Returns true if the specified n-gram is considered common.
    /// 
    /// # Arguments
    /// * `ngram_id` - The ID of the n-gram to check
    pub fn is_common(&self, ngram_id: u64) -> bool {
        self.common_ngrams.contains(&ngram_id)
    }
    
    /// Returns the number of common n-grams being tracked.
    pub fn common_ngram_count(&self) -> usize {
        self.common_ngrams.len()
    }
    
    /// Returns the total number of unique n-grams that were analyzed.
    pub fn total_analyzed_count(&self) -> usize {
        self.total_analyzed
    }
    
    /// Returns the proportion setting used for determining common n-grams.
    pub fn proportion(&self) -> f64 {
        self.proportion
    }
    
    /// Returns the threshold setting used for determining if text is banal.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
    
    /// Provides statistics about the common n-grams collection.
    pub fn stats(&self) -> String {
        format!(
            "CommonNgrams: {} common n-grams ({:.1}% of {} total), threshold: {:.1}%",
            self.common_ngrams.len(),
            if self.total_analyzed > 0 { 
                (self.common_ngrams.len() as f64 / self.total_analyzed as f64) * 100.0
            } else { 
                0.0 
            },
            self.total_analyzed,
            self.threshold
        )
    }
}

impl Clone for CommonNgrams {
    fn clone(&self) -> Self {
        Self {
            common_ngrams: self.common_ngrams.clone(),
            total_analyzed: self.total_analyzed,
            proportion: self.proportion,
            threshold: self.threshold,
        }
    }
}