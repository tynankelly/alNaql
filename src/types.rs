// types.rs
use serde::{Serialize, Deserialize};

// ===========================================================================
// NGRAM ENTRY FOR GENERATION (Temporary structure, not stored)
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NGramEntry {
    pub sequence_number: u64,
    pub file_path: String,
    pub text_position: TextPosition,
    pub text_content: NGramText,
    #[serde(default)]
    pub char_frequencies: CompactCharFrequencies,  // Direct reference, no wrapper
}

// Manual implementations for NGramEntry to handle the custom CompactCharFrequencies
impl PartialEq for NGramEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_number == other.sequence_number &&
        self.file_path == other.file_path &&
        self.text_position == other.text_position &&
        self.text_content == other.text_content &&
        self.char_frequencies == other.char_frequencies
    }
}

impl Eq for NGramEntry {}

impl std::hash::Hash for NGramEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.sequence_number.hash(state);
        self.file_path.hash(state);
        self.text_position.hash(state);
        self.text_content.hash(state);
        self.char_frequencies.hash(state);
    }
}

// ===========================================================================
// SIMD-OPTIMIZED CHARACTER FREQUENCY STRUCTURE
// ===========================================================================
// This structure is designed for optimal SIMD processing of Arabic text
// Total size: exactly 96 bytes with 32-byte alignment for AVX2 operations

#[repr(C, align(32))]  // Critical: ensures 32-byte alignment for SIMD operations
#[derive(Debug, Clone)]
pub struct CompactCharFrequencies {
    // Character frequency array (64 bytes)
    // Index mapping:
    // - Index 0: Space character (0x20)
    // - Index 1-21: First Arabic letter range (U+0621 to U+063A)
    // - Index 22-42: Second Arabic letter range (U+0641 to U+064A)
    // - Index 43-63: Padding (always zero, maintains alignment)
    pub counts: [u8; 64],
    
    // Metadata for fast filtering and similarity calculation (32 bytes total)
    pub total_chars: u32,        // Sum of all character counts
    pub unique_chars: u16,       // Number of distinct characters present
    pub magnitude_squared: f32,  // Pre-computed sum of squares for Cosine similarity
    pub char_bitmap: u64,        // Bitmap for quick filtering
    
    // Padding to ensure exactly 96 bytes total
    // This maintains alignment and allows for future metadata expansion
    pub padding: [u8; 14],
}

// Implement Default manually for arrays > 32
impl Default for CompactCharFrequencies {
    fn default() -> Self {
        Self {
            counts: [0; 64],
            total_chars: 0,
            unique_chars: 0,
            magnitude_squared: 0.0,
            char_bitmap: 0,
            padding: [0; 14],
        }
    }
}

// Custom Serialize implementation to handle large arrays
impl Serialize for CompactCharFrequencies {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("CompactCharFrequencies", 5)?;
        // Serialize the array as a slice (which Serde can handle)
        state.serialize_field("counts", &self.counts[..])?;
        state.serialize_field("total_chars", &self.total_chars)?;
        state.serialize_field("unique_chars", &self.unique_chars)?;
        state.serialize_field("magnitude_squared", &self.magnitude_squared)?;
        state.serialize_field("char_bitmap", &self.char_bitmap)?;
        state.serialize_field("padding", &self.padding[..])?;
        state.end()
    }
}

// Custom Deserialize implementation to handle large arrays
impl<'de> Deserialize<'de> for CompactCharFrequencies {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;
        
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { Counts, TotalChars, UniqueChars, MagnitudeSquared, CharBitmap, Padding }
        
        struct CompactCharFrequenciesVisitor;
        
        impl<'de> Visitor<'de> for CompactCharFrequenciesVisitor {
            type Value = CompactCharFrequencies;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CompactCharFrequencies")
            }
            
            fn visit_map<V>(self, mut map: V) -> Result<CompactCharFrequencies, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut counts: Option<Vec<u8>> = None;
                let mut total_chars = None;
                let mut unique_chars = None;
                let mut magnitude_squared = None;
                let mut char_bitmap =  None;
                let mut padding: Option<Vec<u8>> = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Counts => {
                            counts = Some(map.next_value()?);
                        }
                        Field::TotalChars => {
                            total_chars = Some(map.next_value()?);
                        }
                        Field::UniqueChars => {
                            unique_chars = Some(map.next_value()?);
                        }
                        Field::MagnitudeSquared => {
                            magnitude_squared = Some(map.next_value()?);
                        }
                        Field::CharBitmap => {
                            char_bitmap = Some(map.next_value()?);
                        }
                        Field::Padding => {
                            padding = Some(map.next_value()?);
                        }
                    }
                }
                
                let counts_vec = counts.ok_or_else(|| de::Error::missing_field("counts"))?;
                let padding_vec = padding.ok_or_else(|| de::Error::missing_field("padding"))?;
                
                // Convert Vec<u8> to [u8; 64] and [u8; 14]
                let mut counts_array = [0u8; 64];
                let mut padding_array = [0u8; 14];
                
                if counts_vec.len() != 64 {
                    return Err(de::Error::custom(format!("counts array must be 64 bytes, got {}", counts_vec.len())));
                }
                if padding_vec.len() != 14 {
                    return Err(de::Error::custom(format!("padding array must be 14 bytes, got {}", padding_vec.len())));
                }
                
                counts_array.copy_from_slice(&counts_vec);
                padding_array.copy_from_slice(&padding_vec);
                
                Ok(CompactCharFrequencies {
                    counts: counts_array,
                    total_chars: total_chars.ok_or_else(|| de::Error::missing_field("total_chars"))?,
                    unique_chars: unique_chars.ok_or_else(|| de::Error::missing_field("unique_chars"))?,
                    magnitude_squared: magnitude_squared.ok_or_else(|| de::Error::missing_field("magnitude_squared"))?,
                    char_bitmap: char_bitmap.ok_or_else(|| de::Error::missing_field("magnitude_squared"))?,
                    padding: padding_array,
                })
            }
        }
        
        const FIELDS: &'static [&'static str] = &["counts", "total_chars", "unique_chars", "magnitude_squared", "char_bitmap", "padding"];
        deserializer.deserialize_struct("CompactCharFrequencies", FIELDS, CompactCharFrequenciesVisitor)
    }
}

impl CompactCharFrequencies {
    /// Creates a new empty frequency structure
    pub fn new() -> Self {
        Self {
            counts: [0; 64],
            total_chars: 0,
            unique_chars: 0,
            magnitude_squared: 0.0,
            char_bitmap: 0,
            padding: [0; 14],
        }
    }
    
    /// Checks if the structure is empty (no characters counted)
    pub fn is_empty(&self) -> bool {
        self.total_chars == 0
    }
    
    /// Returns the count for a specific index
    pub fn count_at(&self, index: usize) -> u8 {
        if index < 64 {
            self.counts[index]
        } else {
            0
        }
    }
    
    /// Verifies the structure is properly aligned for SIMD operations
    #[cfg(debug_assertions)]
    pub fn verify_alignment(&self) -> bool {
        let ptr = self as *const Self as usize;
        ptr % 32 == 0
    }
}

// Manual implementations for comparison and hashing
impl PartialEq for CompactCharFrequencies {
    fn eq(&self, other: &Self) -> bool {
        self.counts == other.counts &&
        self.total_chars == other.total_chars &&
        self.unique_chars == other.unique_chars &&
        // Compare float bits for exact equality
        self.magnitude_squared.to_bits() == other.magnitude_squared.to_bits() &&
        self.char_bitmap == other.char_bitmap
        // Padding doesn't need to be compared
    }
}

impl Eq for CompactCharFrequencies {}

impl std::hash::Hash for CompactCharFrequencies {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.counts.hash(state);
        self.total_chars.hash(state);
        self.unique_chars.hash(state);
        // Hash the float bits for consistency
        self.magnitude_squared.to_bits().hash(state);
        self.char_bitmap.hash(state);
        // Don't hash padding as it's not meaningful data
    }
}

// Compile-time size verification
// The structure should be exactly 96 bytes (3 × 32 for optimal SIMD alignment)
#[cfg(test)]
mod size_tests {
    use super::*;
    
    #[test]
    fn test_compact_char_frequencies_size() {
        assert_eq!(std::mem::size_of::<CompactCharFrequencies>(), 96,
                   "CompactCharFrequencies must be exactly 96 bytes for SIMD optimization");
        assert_eq!(std::mem::align_of::<CompactCharFrequencies>(), 32,
                   "CompactCharFrequencies must be 32-byte aligned for AVX2");
    }
}

// ===========================================================================
// SUPPORTING STRUCTURES
// ===========================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextPosition {
    pub start_byte: usize,
    pub end_byte: usize,
    pub line_number: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NGramText {
    pub original_slice: String,
    pub cleaned_text: String,
}

// ===========================================================================
// TYPES FOR COLUMN STORAGE DURING MATCHING
// ===========================================================================
// These are stored separately in columnar databases, not as part of NGramEntry

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NGramMetadata {
    pub sequence_number: u64,
    pub text_length: u64,
    pub text_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionData {
    pub file_path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub line_number: u32,
}

// Note: CompactFeatures mentioned in the storage plan appears to be 
// for a separate optimization - storing frequently accessed metadata 
// in a compact format. If needed, it would be defined separately from
// NGramEntry since it's a storage optimization, not a generation structure.

// ===========================================================================
// MATCHING RESULTS
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceMatch {
    pub source_sequence: u64,
    pub target_sequence: u64,
    pub similarity_score: f64,
    pub metric_type: SimilarityMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Exact,
    Jaccard,
    Cosine,
    Levenshtein,
    VSA,
}

// Note: Error types are defined in error.rs
// This module uses crate::error::{Error, Result}