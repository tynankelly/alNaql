// src/config/subsystems/matcher.rs

use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::config::FromIni;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    // Cosine matcher settings
    pub min_term_freq: usize,
    pub max_vocab_size: usize,
    
    // Levenshtein settings
    pub levenshtein_max_distance: usize,
    
    // VSA settings
    pub vsa_min_term_weight: f64,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            min_term_freq: 5,
            max_vocab_size: 10000,
            levenshtein_max_distance: 3,
            vsa_min_term_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatcherConfig {
    // Cluster merging settings
    pub max_gap: usize,
    pub min_density: f64,

    // N-gram level matching
    pub word_match_min_ratio: f64,
    pub ngram_similarity_metric: SimilarityMetric,
    pub ngram_min_confidence: f64,
    
    // Text level matching
    pub text_similarity_metric: SimilarityMetric,
    // Minimum confidence for initial text comparison
    pub initial_text_min_confidence: f64,
    // Minimum confidence for merged text comparison
    pub merged_text_min_confidence: f64,
    pub adjacent_merge_ratio: f64,
    // Algorithm-specific settings
    pub algorithm_config: AlgorithmConfig,
    // Debug missing phrase
    pub traced_phrase: String,
    // Chain length settings
    pub min_character_chain_length: usize,  // Minimum chain length for character n-grams
    pub min_word_length: usize,       // Minimum chain length for word n-grams
    pub min_chars_length: usize,             // Minimum characters in match text
    /// Whether to use proximity-based matching instead of strict sequential matching
    pub proximity_matching: bool,
    /// Maximum distance between ngrams in a proximity match
    pub proximity_distance: usize,
    /// Minimum density factor for proximity clusters (as a multiplier of min_density)
    pub proximity_density_factor: f64,
    
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            // From default.ini:
            max_gap: 10,
            min_density: 0.85,
            word_match_min_ratio: 0.66,
            ngram_similarity_metric: SimilarityMetric::Exact,
            ngram_min_confidence: 0.85,
            text_similarity_metric: SimilarityMetric::Jaccard,
            initial_text_min_confidence: 0.8,
            merged_text_min_confidence: 0.6,
            adjacent_merge_ratio: 0.3,
            algorithm_config: AlgorithmConfig::default(),
            traced_phrase: String::new(),
            min_character_chain_length: 5,
            min_word_length: 8,
            min_chars_length: 20,
            proximity_matching: false,
            proximity_distance: 30,
            proximity_density_factor: 0.7,
        }
    }
}

impl FromIni for MatcherConfig {
    fn from_ini_section(&mut self, section_name: &str, key: &str, value: &str) -> Option<Result<()>> {
        // Only handle matcher or matcher.algorithm sections
        if !section_name.starts_with("matcher") {
            return None;
        }

        match (section_name, key) {
            // Main matcher settings
            ("matcher", "ngram_similarity_metric") => {
                self.ngram_similarity_metric = match SimilarityMetric::from_str(value) {
                    Some(metric) => metric,
                    None => return Some(Err(Error::Config(
                        format!("Invalid similarity metric: {}", value)
                    ))),
                };
                Some(Ok(()))
            },
            ("matcher", "text_similarity_metric") => {
                self.text_similarity_metric = match SimilarityMetric::from_str(value) {
                    Some(metric) => metric,
                    None => return Some(Err(Error::Config(
                        format!("Invalid text similarity metric: {}", value)
                    ))),
                };
                Some(Ok(()))
            },
            ("matcher", "word_match_min_ratio") => {
                match value.parse::<f64>() {
                    Ok(ratio) if (0.0..=1.0).contains(&ratio) => {
                        self.word_match_min_ratio = ratio;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid word_match_min_ratio (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            ("matcher", "max_gap") => {
                match value.parse() {
                    Ok(gap) => {
                        self.max_gap = gap;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid max_gap: {}", value)
                    ))),
                }
            },
            ("matcher", "min_density") => {
                match value.parse::<f64>() {
                    Ok(threshold) if (0.0..=1.0).contains(&threshold) => {
                        self.min_density = threshold;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid min_density (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            ("matcher", "ngram_min_confidence") => {
                match value.parse::<f64>() {
                    Ok(confidence) if (0.0..=1.0).contains(&confidence) => {
                        self.ngram_min_confidence = confidence;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid ngram_min_confidence (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            ("matcher", "initial_text_min_confidence") => {
                match value.parse::<f64>() {
                    Ok(confidence) if (0.0..=1.0).contains(&confidence) => {
                        self.initial_text_min_confidence = confidence;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid initial_text_min_confidence (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            ("matcher", "merged_text_min_confidence") => {
                match value.parse::<f64>() {
                    Ok(confidence) if (0.0..=1.0).contains(&confidence) => {
                        self.merged_text_min_confidence = confidence;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid merged_text_min_confidence (must be between 0 and 1): {}", value)
                    ))),
                }
            },

            // Algorithm-specific settings
            ("matcher.algorithm", "min_term_freq") => {
                match value.parse() {
                    Ok(freq) => {
                        self.algorithm_config.min_term_freq = freq;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid min_term_freq: {}", value)
                    ))),
                }
            },
            ("matcher.algorithm", "max_vocab_size") => {
                match value.parse() {
                    Ok(size) => {
                        self.algorithm_config.max_vocab_size = size;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid max_vocab_size: {}", value)
                    ))),
                }
            },
            ("matcher.algorithm", "levenshtein_max_distance") => {
                match value.parse() {
                    Ok(dist) => {
                        self.algorithm_config.levenshtein_max_distance = dist;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid levenshtein_max_distance: {}", value)
                    ))),
                }
            },
            ("matcher.algorithm", "vsa_min_term_weight") => {
                match value.parse::<f64>() {
                    Ok(weight) if (0.0..=1.0).contains(&weight) => {
                        self.algorithm_config.vsa_min_term_weight = weight;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid vsa_min_term_weight (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            ("matcher", "adjacent_merge_ratio") => {
                match value.parse::<f64>() {
                    Ok(ratio) if (0.0..=10.0).contains(&ratio) => {
                        self.adjacent_merge_ratio = ratio;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid adjacent_merge_ratio (must be between 0 and 1): {}", value)
                    ))),
                }
            },
            // Add traced_phrase
            ("matcher", "traced_phrase") => {
                // Strip surrounding double quotes if present
                let cleaned_value = value.trim_matches('"').to_string();
                self.traced_phrase = cleaned_value;
                Some(Ok(()))
            },
            ("matcher", "min_character_chain_length") => {
                match value.parse() {
                    Ok(length) if length > 0 => {
                        self.min_character_chain_length = length;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid min_character_chain_length (must be > 0): {}", value)
                    ))),
                }
            },
            ("matcher", "min_word_length") => {
                match value.parse() {
                    Ok(length) if length > 0 => {
                        self.min_word_length = length;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid min_word_length (must be > 0): {}", value)
                    ))),
                }
            },
            ("matcher", "min_chars_length") => {
                match value.parse() {
                    Ok(length) if length > 0 => {
                        self.min_chars_length = length;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid min_chain_chars (must be > 0): {}", value)
                    ))),
                }
            },
            ("matcher", "proximity_matching") => {
                match value.parse::<bool>() {
                    Ok(val) => {
                        self.proximity_matching = val;
                        Some(Ok(()))
                    },
                    Err(_) => Some(Err(Error::Config(
                        format!("Invalid proximity_matching (must be true or false): {}", value)
                    ))),
                }
            },
            ("matcher", "proximity_distance") => {
                match value.parse::<usize>() {
                    Ok(dist) if dist > 0 => {
                        self.proximity_distance = dist;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid proximity_distance (must be > 0): {}", value)
                    ))),
                }
            },
            ("matcher", "proximity_density_factor") => {
                match value.parse::<f64>() {
                    Ok(factor) if (0.1..=1.0).contains(&factor) => {
                        self.proximity_density_factor = factor;
                        Some(Ok(()))
                    },
                    _ => Some(Err(Error::Config(
                        format!("Invalid proximity_density_factor (must be between 0.1 and 1.0): {}", value)
                    ))),
                }
            },

            // Unknown key
            _ => None,
        }
    }
}
impl MatcherConfig {
    pub fn validate(&self) -> Result<()> {
        // Validate confidence thresholds
        if !(0.0..=1.0).contains(&self.min_density) {
            return Err(Error::Config(
                "min_density must be between 0 and 1".to_string()
            ));
        }
        if self.algorithm_config.min_term_freq == 0 {
            return Err(Error::Config(
                "min_term_freq must be greater than 0".to_string()
            ));
        }
        if !(0.0..=1.0).contains(&self.algorithm_config.vsa_min_term_weight) {
            return Err(Error::Config(
                "vsa_min_term_weight must be between 0 and 1".to_string()
            ));
        }
        if self.proximity_distance == 0 {
            return Err(Error::Config(
                "proximity_distance must be greater than 0".to_string()
            ));
        }
        if !(0.1..=1.0).contains(&self.proximity_density_factor) {
            return Err(Error::Config(
                "proximity_density_factor must be between 0.1 and 1.0".to_string()
            ));
        }
        Ok(())
    }
}