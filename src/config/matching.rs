// config/matching.rs - Matching algorithm and similarity configuration

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::loader::Configurable;

// ============================================================================
// ENUMS AND TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessStrategy {
    Sequential,
    Parallel,
    Grid,
}

impl Default for ProcessStrategy {
    fn default() -> Self {
        ProcessStrategy::Parallel
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Exact,
    Jaccard,
    Cosine,
    VSA,
    Levenshtein,
}

impl SimilarityMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            SimilarityMetric::Exact => "exact",
            SimilarityMetric::Jaccard => "jaccard",
            SimilarityMetric::Cosine => "cosine",
            SimilarityMetric::VSA => "vsa",
            SimilarityMetric::Levenshtein => "levenshtein",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim_matches('"').to_lowercase().as_str() {
            "exact" => Some(Self::Exact),
            "jaccard" => Some(Self::Jaccard),
            "cosine" => Some(Self::Cosine),
            "vsa" => Some(Self::VSA),
            "levenshtein" => Some(Self::Levenshtein),
            _ => None,
        }
    }
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::Jaccard
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NGramMatchingStrategy {
    Sequential,
    Parallel,
    ParallelOpt,
}

impl Default for NGramMatchingStrategy {
    fn default() -> Self {
        Self::ParallelOpt
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusteringMode {
    Sequential,
    Proximity,
    Both,
}

impl Default for ClusteringMode {
    fn default() -> Self {
        Self::Sequential
    }
}

// ============================================================================
// EMBEDDED CONFIGURATION STRUCTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
   
    // Levenshtein settings
    pub levenshtein_max_distance: usize,
    
    // VSA settings
    pub vsa_min_term_weight: f64,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            levenshtein_max_distance: 10,
            vsa_min_term_weight: 0.01,
        }
    }
}

// ============================================================================
// MAIN CONFIGURATION STRUCT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingConfig {
    // ========== Core Strategy Settings ==========
    pub process_strategy: ProcessStrategy,
    pub ngram_matching_strategy: NGramMatchingStrategy,
    pub clustering_mode: ClusteringMode,
    pub proximity_distance: usize,
    pub max_gap: usize,
    
    // ========== Similarity Metrics & Thresholds ==========
    pub ngram_similarity_metric: SimilarityMetric,
    pub text_similarity_metric: SimilarityMetric,
    pub ngram_min_confidence: f64,
    pub initial_text_min_confidence: f64,
    pub merged_text_min_confidence: f64,
    pub min_density: f64,
    pub adjacent_merge_ratio: f64,
    
    // ========== Filtering Options ==========
    pub use_precomputed_similarity_features: bool,
    pub enable_prefix_filtering: Option<bool>,
    pub enable_metadata_filtering: Option<bool>,
    pub enable_quick_filtering: Option<bool>,
    pub enable_length_filtering: Option<bool>,
    
    // ========== Chain/Length Requirements ==========
    pub min_word_length: usize,
    pub min_chars_length: usize,
    
    // ========== Banality and Isnad Settings ==========
    pub banality_detection_mode: String,
    pub banality_auto_proportion: f64,
    pub banality_auto_threshold: f64,
    pub banality_density_threshold: f64,
    pub auto_banal_batch_size: usize,
    pub banality_phrases_file: String,
    pub banality_words_file: String,
    pub isnad_density_threshold: f64,
    pub isnad_min_length: usize,
    pub isnad_phrases_file: String,
    pub isnad_words_file: String,
    
    // ========== Post-processing ==========
    pub deduplication_strategy: String,
    
    // ========== Algorithm-specific Settings ==========
    pub algorithm_config: AlgorithmConfig,

    // ========== Similarity Weighting Settings ==========
    pub apply_text_weighting: bool,
    pub banality_scale_threshold: f64,
    pub banality_scale_factor: f64,
    pub lcs_weight: f64,  
    pub algorithm_weight: f64,
        
    // ========== Caches and Chunks ==========
    pub cache_capacity: usize,
    pub min_source_chunk: usize,
    pub max_source_chunk: usize,
    pub source_chunks_per_cpu: usize,
    pub parallel_source_chunk_size: usize,
    pub parallel_target_chunk_size: usize,
    pub comparisons_per_cell: usize,
    pub similarity_chunk_size: usize,

    // Buffered parallel processing
    pub use_buffered_parallel: bool,
    pub parallel_buffer_memory_limit: usize,
    pub parallel_buffer_match_limit: usize,
    pub parallel_buffer_time_limit: u64,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            // Core Strategy
            process_strategy: ProcessStrategy::Parallel,
            ngram_matching_strategy: NGramMatchingStrategy::default(),
            clustering_mode: ClusteringMode::Proximity,
            proximity_distance: 8,
            max_gap: 10,
            
            // Similarity Metrics & Thresholds
            ngram_similarity_metric: SimilarityMetric::Exact,
            text_similarity_metric: SimilarityMetric::Jaccard,
            ngram_min_confidence: 0.85,
            initial_text_min_confidence: 0.8,
            merged_text_min_confidence: 0.6,
            min_density: 0.85,
            adjacent_merge_ratio: 0.3,
            
            // Filtering Options
            use_precomputed_similarity_features: true,
            enable_prefix_filtering: None,
            enable_metadata_filtering: None,
            enable_quick_filtering: None,
            enable_length_filtering: None,
            
            // Chain/Length Requirements
            min_word_length: 5,
            min_chars_length: 20,
            
            // Banality and Isnad Settings
            banality_detection_mode: "phrase".to_string(),
            banality_auto_proportion: 5.0,
            banality_auto_threshold: 70.0,
            isnad_density_threshold: 0.65,
            isnad_min_length: 25,
            banality_density_threshold: 0.67,
            isnad_phrases_file: "filters/isnad_phrases.txt".to_string(),
            isnad_words_file: "filters/isnad_words.txt".to_string(),
            banality_phrases_file: "filters/banalities_phrases.txt".to_string(),
            banality_words_file: "filters/banalities_words.txt".to_string(),
            auto_banal_batch_size: 50_000,
            
            // Post-processing
            deduplication_strategy: "keep_both".to_string(),
            
            
            // Algorithm-specific
            algorithm_config: AlgorithmConfig::default(),

            // Word Weighting
            apply_text_weighting: true,
            banality_scale_threshold: 0.5,
            banality_scale_factor: 0.5,
            lcs_weight: 0.7,
            algorithm_weight: 0.3,
            
            // Caches and Chunks
            cache_capacity: 10000,
            min_source_chunk: 1000,
            max_source_chunk: 10000,
            source_chunks_per_cpu: 2,
            comparisons_per_cell: 10000000,
            parallel_source_chunk_size: 500,
            parallel_target_chunk_size: 2000,
            similarity_chunk_size: 1000,

            // Buffered parallel processing
            use_buffered_parallel: true,
            parallel_buffer_memory_limit: 104857600,  // 100MB
            parallel_buffer_match_limit: 10000,
            parallel_buffer_time_limit: 30,

        }
    }
}

impl Configurable for MatchingConfig {
    fn section_name() -> &'static str {
        "matching"
    }
    
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool> {
        match key {
            // Core Strategy Settings
            "process_strategy" => {
                let strategy = value.trim_matches('"').to_lowercase();
                match strategy.as_str() {
                    "sequential" => self.process_strategy = ProcessStrategy::Sequential,
                    "parallel" => self.process_strategy = ProcessStrategy::Parallel,
                    "grid" => self.process_strategy = ProcessStrategy::Grid,
                    _ => return Err(Error::Config(format!(
                        "Invalid process_strategy '{}'. Must be 'sequential', 'parallel', or 'grid'",
                        value
                    ))),
                }
                Ok(true)
            },
            "ngram_matching_strategy" => {
                let strategy = value.trim_matches('"').to_lowercase();
                match strategy.as_str() {
                    "sequential" => self.ngram_matching_strategy = NGramMatchingStrategy::Sequential,
                    "parallel" => self.ngram_matching_strategy = NGramMatchingStrategy::Parallel,
                    "parallelopt" => self.ngram_matching_strategy = NGramMatchingStrategy::ParallelOpt,
                    _ => return Err(Error::Config(format!(
                        "Invalid ngram_matching_strategy: {}", value
                    ))),
                }
                Ok(true)
            },
            "clustering_mode" => {
                let mode = value.trim_matches('"').to_lowercase();
                match mode.as_str() {
                    "sequential" => self.clustering_mode = ClusteringMode::Sequential,
                    "proximity" => self.clustering_mode = ClusteringMode::Proximity,
                    "both" => self.clustering_mode = ClusteringMode::Both,
                    _ => return Err(Error::Config(format!(
                        "Invalid clustering_mode: {}", value
                    ))),
                }
                Ok(true)
            },
            "proximity_distance" => {
                self.proximity_distance = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid proximity_distance: {}", value)))?;
                Ok(true)
            },
            "max_gap" => {
                self.max_gap = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid max_gap: {}", value)))?;
                Ok(true)
            },
            
            // Similarity Metrics & Thresholds
            "ngram_similarity_metric" => {
                self.ngram_similarity_metric = SimilarityMetric::from_str(value)
                    .ok_or_else(|| Error::Config(format!("Invalid similarity metric: {}", value)))?;
                Ok(true)
            },
            "text_similarity_metric" => {
                self.text_similarity_metric = SimilarityMetric::from_str(value)
                    .ok_or_else(|| Error::Config(format!("Invalid similarity metric: {}", value)))?;
                Ok(true)
            },
            "ngram_min_confidence" => {
                self.ngram_min_confidence = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid ngram_min_confidence: {}", value)))?;
                Ok(true)
            },
            "initial_text_min_confidence" => {
                self.initial_text_min_confidence = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid initial_text_min_confidence: {}", value)))?;
                Ok(true)
            },
            "merged_text_min_confidence" => {
                self.merged_text_min_confidence = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid merged_text_min_confidence: {}", value)))?;
                Ok(true)
            },
            "min_density" => {
                self.min_density = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid min_density: {}", value)))?;
                Ok(true)
            },
            "adjacent_merge_ratio" => {
                self.adjacent_merge_ratio = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid adjacent_merge_ratio: {}", value)))?;
                Ok(true)
            },
            
            // Filtering Options
            "use_precomputed_similarity_features" => {
                self.use_precomputed_similarity_features = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?;
                Ok(true)
            },
            "enable_prefix_filtering" => {
                self.enable_prefix_filtering = Some(value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?);
                Ok(true)
            },
            "enable_metadata_filtering" => {
                self.enable_metadata_filtering = Some(value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?);
                Ok(true)
            },
            "enable_quick_filtering" => {
                self.enable_quick_filtering = Some(value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?);
                Ok(true)
            },
            "enable_length_filtering" => {
                self.enable_length_filtering = Some(value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?);
                Ok(true)
            },
            
            // Chain/Length Requirements
            "min_word_length" => {
                self.min_word_length = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid min_word_length: {}", value)))?;
                Ok(true)
            },
            "min_chars_length" => {
                self.min_chars_length = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid min_chars_length: {}", value)))?;
                Ok(true)
            },
            
            // Banality and Isnad Settings
            "banality_detection_mode" => {
                self.banality_detection_mode = value.trim_matches('"').to_string();
                Ok(true)
            },
            "banality_auto_proportion" => {
                self.banality_auto_proportion = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid banality_auto_proportion: {}", value)))?;
                Ok(true)
            },
            "banality_auto_threshold" => {
                self.banality_auto_threshold = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid banality_auto_threshold: {}", value)))?;
                Ok(true)
            },
            "isnad_density_threshold" => {
                self.isnad_density_threshold = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid isnad_density_threshold: {}", value)))?;
                Ok(true)
            },
            "isnad_min_length" => {
                self.isnad_min_length = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid isnad_min_length: {}", value)))?;
                Ok(true)
            },
            "banality_density_threshold" => {
                self.banality_density_threshold = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid banality_density_threshold: {}", value)))?;
                Ok(true)
            },
            "isnad_phrases_file" => {
                self.isnad_phrases_file = value.trim_matches('"').to_string();
                Ok(true)
            },
            "isnad_words_file" => {
                self.isnad_words_file = value.trim_matches('"').to_string();
                Ok(true)
            },
            "banality_phrases_file" => {
                self.banality_phrases_file = value.trim_matches('"').to_string();
                Ok(true)
            },
            "banality_words_file" => {
                self.banality_words_file = value.trim_matches('"').to_string();
                Ok(true)
            },
            "auto_banal_batch_size" => {
                self.auto_banal_batch_size = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid auto_banal_batch_size: {}", value)))?;
                Ok(true)
            },

            // Post-processing
            "deduplication_strategy" => {
                self.deduplication_strategy = value.trim_matches('"').to_string();
                Ok(true)
            },
            "apply_text_weighting" => {
                self.apply_text_weighting = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?;
                Ok(true)
            },
            
            // Algorithm Config fields
            "levenshtein_max_distance" => {
                self.algorithm_config.levenshtein_max_distance = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid levenshtein_max_distance: {}", value)))?;
                Ok(true)
            },
            "vsa_min_term_weight" => {
                self.algorithm_config.vsa_min_term_weight = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid vsa_min_term_weight: {}", value)))?;
                Ok(true)
            },

            // Word and Banality Weighting
            "banality_scale_threshold" => {
                self.banality_scale_threshold = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid banality_scale_threshold: {}", value)))?;
                Ok(true)
            },
            "banality_scale_factor" => {
                self.banality_scale_factor = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid banality_scale_factor: {}", value)))?;
                Ok(true)
            },
            "lcs_weight" => {
                self.lcs_weight = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid lcs_weight: {}", value)))?;
                Ok(true)
            },
            "algorithm_weight" => {
                self.algorithm_weight = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid algorithm_weight: {}", value)))?;
                Ok(true)
            },
            
            // Caches and Chunks
            "cache_capacity" => {
                self.cache_capacity = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid cache_capacity: {}", value)))?;
                Ok(true)
            },
            "min_source_chunk" => {
                self.min_source_chunk = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid min_source_chunk: {}", value)))?;
                Ok(true)
            },
            "max_source_chunk" => {
                self.max_source_chunk = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid max_source_chunk: {}", value)))?;
                Ok(true)
            },
            // Grid Config fields
            "comparisons_per_cell" => {
                self.comparisons_per_cell = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid comparisons_per_cell: {}", value)))?;
                Ok(true)
            },
            "source_chunks_per_cpu" => {
                self.source_chunks_per_cpu = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid source_chunks_per_cpu: {}", value)))?;
                Ok(true)
            },
            "parallel_source_chunk_size" => {
                self.parallel_source_chunk_size = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid parallel_source_chunk_size: {}", value)))?;
                Ok(true)
            },
            "parallel_target_chunk_size" => {
                self.parallel_target_chunk_size = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid parallel_target_chunk_size: {}", value)))?;
                Ok(true)
            },
            "similarity_chunk_size" => {
                self.similarity_chunk_size = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid similarity_chunk_size: {}", value)))?;
                Ok(true)
            },

            // Buffered parallel processing
            "use_buffered_parallel" => {
                self.use_buffered_parallel = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid boolean: {}", value)))?;
                Ok(true)
            },
            "parallel_buffer_memory_limit" => {
                self.parallel_buffer_memory_limit = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid parallel_buffer_memory_limit: {}", value)))?;
                Ok(true)
            },
            "parallel_buffer_match_limit" => {
                self.parallel_buffer_match_limit = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid parallel_buffer_match_limit: {}", value)))?;
                Ok(true)
            },
            "parallel_buffer_time_limit" => {
                self.parallel_buffer_time_limit = value.parse().map_err(|_| 
                    Error::Config(format!("Invalid parallel_buffer_time_limit: {}", value)))?;
                Ok(true)
            },
            
            _ => Ok(false),
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Validate confidence thresholds (0-1 range)
        if !(0.0..=1.0).contains(&self.min_density) {
            return Err(Error::Config("min_density must be between 0 and 1".to_string()));
        }
        if !(0.0..=1.0).contains(&self.ngram_min_confidence) {
            return Err(Error::Config("ngram_min_confidence must be between 0 and 1".to_string()));
        }
        if !(0.0..=1.0).contains(&self.initial_text_min_confidence) {
            return Err(Error::Config("initial_text_min_confidence must be between 0 and 1".to_string()));
        }
        if !(0.0..=1.0).contains(&self.merged_text_min_confidence) {
            return Err(Error::Config("merged_text_min_confidence must be between 0 and 1".to_string()));
        }
        if !(0.0..=1.0).contains(&self.adjacent_merge_ratio) {
            return Err(Error::Config("adjacent_merge_ratio must be between 0 and 1".to_string()));
        }
                
        // Validate banality settings
        if !(0.0..=100.0).contains(&self.banality_auto_proportion) {
            return Err(Error::Config("banality_auto_proportion must be between 0 and 100".to_string()));
        }
        if !(0.0..=100.0).contains(&self.banality_auto_threshold) {
            return Err(Error::Config("banality_auto_threshold must be between 0 and 100".to_string()));
        }
        
        // Validate algorithm config
        if !(0.0..=1.0).contains(&self.algorithm_config.vsa_min_term_weight) {
            return Err(Error::Config("vsa_min_term_weight must be between 0 and 1".to_string()));
        }
                
        Ok(())
    }
}