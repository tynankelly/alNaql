// ngram/arabic_char_mapping.rs
//! Character mapping infrastructure for SIMD-optimized Arabic text processing.
//! 
//! This module provides the canonical mapping between Arabic characters and their
//! positions in the 64-byte frequency array used for SIMD operations. All character
//! indexing throughout the system must use this module to ensure consistency.

use once_cell::sync::Lazy;

// ===========================================================================
// CHARACTER SET CONSTANTS
// ===========================================================================

/// Total number of valid characters in our Arabic character set
pub const ARABIC_CHAR_COUNT: usize = 43;

/// Space character - always at index 0
const SPACE_CHAR: char = ' ';
const SPACE_INDEX: u8 = 0;

/// First range of Arabic letters (ء to غ)
/// Unicode range: U+0621 to U+063A (26 characters)
const ARABIC_RANGE1_START: u32 = 0x0621;
const ARABIC_RANGE1_END: u32 = 0x063A;
const ARABIC_RANGE1_START_INDEX: u8 = 1;

/// Second range of Arabic letters (ف to ي)
/// Unicode range: U+0641 to U+064A (16 characters)  
const ARABIC_RANGE2_START: u32 = 0x0641;
const ARABIC_RANGE2_END: u32 = 0x064A;
const ARABIC_RANGE2_START_INDEX: u8 = 27; // After space + first range

/// Size of the lookup table (covers up to U+07FF, which includes all Arabic)
/// This is a power of 2 for efficient masking if needed
const LOOKUP_TABLE_SIZE: usize = 2048;

// ===========================================================================
// LOOKUP TABLE FOR O(1) CHARACTER TO INDEX CONVERSION
// ===========================================================================

/// Static lookup table for fast character-to-index conversion
/// Index: Unicode codepoint (limited to LOOKUP_TABLE_SIZE)
/// Value: Some(index) for valid Arabic characters, None otherwise
static CHAR_TO_INDEX: Lazy<[Option<u8>; LOOKUP_TABLE_SIZE]> = Lazy::new(|| {
    let mut table = [None; LOOKUP_TABLE_SIZE];
    
    // Map space character
    table[SPACE_CHAR as usize] = Some(SPACE_INDEX);
    
    // Map first range of Arabic letters (ء to غ)
    let mut index = ARABIC_RANGE1_START_INDEX;
    for codepoint in ARABIC_RANGE1_START..=ARABIC_RANGE1_END {
        if (codepoint as usize) < LOOKUP_TABLE_SIZE {
            table[codepoint as usize] = Some(index);
            index += 1;
        }
    }
    
    // Map second range of Arabic letters (ف to ي)
    index = ARABIC_RANGE2_START_INDEX;
    for codepoint in ARABIC_RANGE2_START..=ARABIC_RANGE2_END {
        if (codepoint as usize) < LOOKUP_TABLE_SIZE {
            table[codepoint as usize] = Some(index);
            index += 1;
        }
    }
    
    table
});

/// Static reverse lookup table for index-to-character conversion (mainly for debugging)
/// This is small (43 entries) so we can afford to store it separately
static INDEX_TO_CHAR: Lazy<[Option<char>; 64]> = Lazy::new(|| {
    let mut table = [None; 64];
    
    // Space
    table[SPACE_INDEX as usize] = Some(SPACE_CHAR);
    
    // First Arabic range
    let mut index = ARABIC_RANGE1_START_INDEX as usize;
    for codepoint in ARABIC_RANGE1_START..=ARABIC_RANGE1_END {
        if let Some(c) = char::from_u32(codepoint) {
            table[index] = Some(c);
            index += 1;
        }
    }
    
    // Second Arabic range  
    index = ARABIC_RANGE2_START_INDEX as usize;
    for codepoint in ARABIC_RANGE2_START..=ARABIC_RANGE2_END {
        if let Some(c) = char::from_u32(codepoint) {
            table[index] = Some(c);
            index += 1;
        }
    }
    
    table
});

// ===========================================================================
// PUBLIC API
// ===========================================================================

/// Convert an Arabic character to its index in the frequency array.
/// 
/// # Arguments
/// * `c` - Character to convert
/// 
/// # Returns
/// * `Some(index)` - Index in range [0, 42] for valid Arabic characters
/// * `None` - If the character is not in our Arabic character set
/// 
/// # Example
/// ```
/// use arabic_char_mapping::char_to_index;
/// 
/// assert_eq!(char_to_index(' '), Some(0));
/// assert_eq!(char_to_index('ا'), Some(7));  // Alef
/// assert_eq!(char_to_index('A'), None);      // Latin character
/// ```
#[inline]
pub fn char_to_index(c: char) -> Option<u8> {
    let codepoint = c as usize;
    if codepoint < LOOKUP_TABLE_SIZE {
        CHAR_TO_INDEX[codepoint]
    } else {
        None
    }
}

/// Convert an index back to its corresponding Arabic character.
/// 
/// # Arguments
/// * `index` - Index in the frequency array
/// 
/// # Returns
/// * `Some(char)` - The corresponding Arabic character
/// * `None` - If the index is invalid (>= 43)
/// 
/// # Example
/// ```
/// use arabic_char_mapping::index_to_char;
/// 
/// assert_eq!(index_to_char(0), Some(' '));
/// assert_eq!(index_to_char(7), Some('ا'));
/// assert_eq!(index_to_char(50), None);  // Invalid index
/// ```
#[inline]
pub fn index_to_char(index: u8) -> Option<char> {
    if (index as usize) < 64 {
        INDEX_TO_CHAR[index as usize]
    } else {
        None
    }
}

/// Check if a character is in our valid Arabic character set.
/// 
/// # Arguments
/// * `c` - Character to check
/// 
/// # Returns
/// * `true` if the character is Arabic or space
/// * `false` otherwise
#[inline]
pub fn is_valid_arabic_char(c: char) -> bool {
    char_to_index(c).is_some()
}

/// Get all valid Arabic characters in index order.
/// Useful for testing and debugging.
/// 
/// # Returns
/// A vector of all 43 valid characters in their index order
pub fn get_all_valid_chars() -> Vec<char> {
    let mut chars = Vec::with_capacity(ARABIC_CHAR_COUNT);
    for i in 0..ARABIC_CHAR_COUNT {
        if let Some(c) = index_to_char(i as u8) {
            chars.push(c);
        }
    }
    chars
}

/// Get a debug string showing the character mapping.
/// Useful for verification and testing.
pub fn debug_mapping() -> String {
    let mut output = String::new();
    output.push_str("Arabic Character Mapping (Index -> Character):\n");
    output.push_str("==============================================\n");
    
    for i in 0..ARABIC_CHAR_COUNT {
        if let Some(c) = index_to_char(i as u8) {
            let codepoint = c as u32;
            output.push_str(&format!(
                "Index {:2}: '{}' (U+{:04X})\n", 
                i, c, codepoint
            ));
        }
    }
    
    output
}

// ===========================================================================
// VALIDATION AND TESTING
// ===========================================================================

/// Validate that the mapping is consistent and bidirectional.
/// This should be called in tests to ensure the mapping is correct.
pub fn validate_mapping() -> Result<(), String> {
    // Check that we have exactly 43 mapped characters
    let mut mapped_count = 0;
    for i in 0..64 {
        if INDEX_TO_CHAR[i].is_some() {
            mapped_count += 1;
        }
    }
    
    if mapped_count != ARABIC_CHAR_COUNT {
        return Err(format!(
            "Expected {} mapped characters, found {}", 
            ARABIC_CHAR_COUNT, mapped_count
        ));
    }
    
    // Check bidirectional mapping
    for i in 0..ARABIC_CHAR_COUNT {
        let index = i as u8;
        if let Some(c) = index_to_char(index) {
            if let Some(reverse_index) = char_to_index(c) {
                if reverse_index != index {
                    return Err(format!(
                        "Bidirectional mapping failed for index {}: {} -> {} -> {}", 
                        index, index, c, reverse_index
                    ));
                }
            } else {
                return Err(format!(
                    "Character '{}' from index {} doesn't map back", 
                    c, index
                ));
            }
        }
    }
    
    // Verify space is at index 0
    if char_to_index(' ') != Some(0) {
        return Err("Space character must be at index 0".to_string());
    }
    
    // Verify Arabic ranges
    let alef = char::from_u32(0x0627).unwrap(); // ا
    if char_to_index(alef).is_none() {
        return Err("Basic Arabic letter 'alef' is not mapped".to_string());
    }
    
    Ok(())
}
