pub mod arabic;

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParserError {
   #[error("Configuration error: {0}")]
   ConfigError(String),
   
   #[error("IO error: {0}")]
   IoError(#[from] std::io::Error),
   
   #[error("Regex error: {0}")]
   RegexError(#[from] regex::Error),
   
   #[error("Invalid text: {0}")]
   InvalidText(String),
   
   #[error("Position tracking error: {0}")]
   PositionError(String),
   
   #[error("Tokenization error: {0}")]
   TokenizationError(String),
}

pub type Result<T> = std::result::Result<T, ParserError>;

#[derive(Debug, Clone)]
pub struct TextMapping {
   pub cleaned_text: String,
   pub char_map: Vec<(usize, usize)>,  // (start_byte, end_byte)
   pub line_map: Vec<usize>,           // line number for each character
}

// New struct for word tokenization
#[derive(Debug, Clone)]
pub struct TextToken {
   pub original_text: String,    // Original token text
   pub cleaned_text: String,     // Cleaned token text
   pub start_byte: usize,        // Start byte position in original text
   pub end_byte: usize,          // End byte position in original text
   pub line_number: usize,       // Line number where token appears
}

pub trait TextParser: Sync + Send {
    /// Clean text, removing non-Arabic characters
    fn clean_text(&self, text: &str) -> Result<String>;

    /// Clean text, returning mapping from cleaned text to original text
    fn clean_text_with_mapping(&self, text: &str) -> Result<TextMapping>;

    /// NEW METHOD: Clean text with absolute position information
    fn clean_text_with_mapping_absolute(&self, text: &str, absolute_position: usize) -> Result<TextMapping>;

    /// Clean text with context from surrounding text
    fn clean_text_with_context(&self, text: &str, full_text: &str, text_start: usize) -> Result<String>;

    /// Count valid characters in text
    fn count_valid_chars(&self, text: &str) -> usize;

    /// Determine if a character is countable (Arabic)
    fn is_countable_char(&self, c: char) -> bool;

    /// Tokenize text into words
    fn tokenize_text(&self, text: &str) -> Result<Vec<TextToken>>;

    /// Load stop words from a file
    fn load_stop_words<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;

    fn tokenize_text_with_mapping(&self, mapping: &TextMapping) -> Result<Vec<TextToken>>;
}

pub use self::arabic::ArabicParser;