// Enhanced ArabicParser implementation with stop words/phrases support

use lazy_static::lazy_static;
use log::{debug, info, warn};
use regex::Regex;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use crate::config::subsystems::ParserConfig;

use super::{TextParser, ParserError, Result, TextMapping, TextToken};

lazy_static! {
    static ref XML_WHITESPACE: Regex = Regex::new(r"\s*<|>\s*").unwrap();
    static ref XML_TAG: Regex = Regex::new(r"<[^>]+>").unwrap();
    static ref NUMBERS: Regex = Regex::new(r"[\p{N}٠-٩]").unwrap();
    static ref LATIN: Regex = Regex::new(r"[\p{Latin}]").unwrap();
    static ref PUNCTUATION: Regex = 
        Regex::new(r"['«»\(\)\[\]\{\}،,\.:;؛\?\!؟\-_\+\*&\^%\$#@\|/\\]").unwrap();
    static ref DIACRITICS: Regex = Regex::new(r"[\u{064B}-\u{065F}\u{0670}]").unwrap();
    static ref TATWEEL: Regex = Regex::new(r"\u{0640}").unwrap();
    static ref NON_ARABIC: Regex = Regex::new(r"[^\p{Arabic}\s]").unwrap();
    static ref WHITESPACE: Regex = Regex::new(r"\s+").unwrap();
    // Add a new regex for word boundary detection in Arabic text
    static ref WORD_BOUNDARY: Regex = Regex::new(r"[\s\p{P}]+").unwrap();
}

#[derive(Clone)]
pub struct ArabicParser {
    settings: ParserConfig,
    stop_phrases: HashSet<String>,
}

impl ArabicParser {
    pub fn new(settings: ParserConfig) -> Self {
        Self {
            settings,
            stop_phrases: HashSet::new(),
        }
    }
    
    pub fn new_with_defaults() -> Self {
        Self::new(ParserConfig::default())
    }
    
    // Add a convenience method to load stop words directly from the config
    pub fn load_stop_words_from_config(&mut self, config: &ParserConfig) -> Result<()> {
        if let Some(path) = &config.stop_words_file {
            debug!("Loading stop words from config path: {:?}", path);
            self.load_stop_words(path)
        } else {
            debug!("No stop words file specified in config");
            Ok(())
        }
    }
    
    fn normalize_char(&self, c: char) -> char {
        if !self.settings.normalize_arabic {
            return c;
        }
    
        let result = match c {
            'ئ' | 'ى' | 'ي' => {
                'ي'
            },
            'ة' => {
                'ه'
            },
            'أ' | 'إ' | 'آ' => {
                'ا'
            },
            'ؤ' => {
                'و'
            },
            _ => c
        };
        
        if c != result {
        }
        
        result
    }

    // Make this a private method in the impl block, not part of the trait
    fn is_ignorable_char(&self, c: char) -> bool {
        // Check if this is a diacritic or other ignorable character
        let c_str = c.to_string();
        DIACRITICS.is_match(&c_str) || 
        TATWEEL.is_match(&c_str) ||
        c == 'ـ'  // Explicit tatweel check
    }

    // New helper method to filter stop phrases
    fn filter_stop_phrases(&self, tokens: Vec<TextToken>) -> Vec<TextToken> {
        if self.stop_phrases.is_empty() {
            return tokens;
        }
        
        let mut filtered = Vec::with_capacity(tokens.len());
        let mut i = 0;
        
        while i < tokens.len() {
            let mut is_stop_phrase = false;
            let mut phrase_length = 0;
            
            // Try multi-word phrases first (up to 5 words)
            for len in (1..=std::cmp::min(5, tokens.len() - i)).rev() {
                let phrase = tokens[i..i+len].iter()
                    .map(|t| t.cleaned_text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                
                if self.stop_phrases.contains(&phrase) {
                    is_stop_phrase = true;
                    phrase_length = len;
                    break;
                }
            }
            
            if is_stop_phrase {
                // Skip this phrase
                i += phrase_length;
            } else {
                // Keep this token
                filtered.push(tokens[i].clone());
                i += 1;
            }
        }
        
        filtered
    }
}

impl TextParser for ArabicParser {
    fn clean_text_with_mapping_absolute(&self, text: &str, absolute_position: usize) -> Result<TextMapping> {
        let mut cleaned = String::new();
        let mut char_map = Vec::new();
        let mut line_map = Vec::new();
        let mut current_line = 1;
        
        // Add tag tracking state
        let mut tag_depth = 0;
    
        // Add this logging
        debug!("MAPPING_DEBUG: Cleaning text at absolute position {}, length {}", 
            absolute_position, text.len());
        debug!("MAPPING_DEBUG: Text sample: \"{}\"", 
            text.chars().take(50).collect::<String>().replace('\n', "\\n"));
        
        // Track state for handling non-adjacent Arabic text
        let mut last_arabic_end = None;  // Position after last Arabic character
        let mut last_non_ignorable = None;  // Position after last non-ignorable character
        
        // Collect all chars with their byte indices
        let chars: Vec<(usize, char)> = text.char_indices().collect();
                
        let mut i = 0;
        while i < chars.len() {
            let (pos, c) = chars[i];
            
            if c == '\n' {
                current_line += 1;
                i += 1;
                continue;
            }
            
            // Handle XML tag tracking with minimal validation
            if c == '<' {
                // Check if this is likely to be a tag
                let mut looks_like_tag = false;
                if i + 1 < chars.len() {
                    let next_char = chars[i + 1].1;
                    // Most XML tags start with a letter, digit, '/', '!', or '?'
                    looks_like_tag = next_char.is_alphanumeric() || 
                                     next_char == '/' || 
                                     next_char == '!' || 
                                     next_char == '?';
                }
                
                if looks_like_tag {
                    tag_depth += 1;
                    debug!("TAG DEPTH: Increased to {} at position {} (char '{}')", 
                           tag_depth, pos, c);
                }
            } else if c == '>' && tag_depth > 0 {
                tag_depth -= 1;
                debug!("TAG DEPTH: Decreased to {} at position {} (char '{}')", 
                       tag_depth, pos, c);
            }
            
            // Only process Arabic characters if we're not inside an XML tag
            if tag_depth == 0 && self.is_countable_char(c) {
                let normalized = self.normalize_char(c);
                
                // If we had a previous Arabic character and some non-ignorable, non-Arabic content between,
                // add a single space
                if let Some(last_end) = last_arabic_end {
                    if pos > last_end && last_non_ignorable.is_some() && last_non_ignorable.unwrap() > last_end {
                        // Add a space to represent the gap
                        cleaned.push(' ');
                        debug!("MAPPING_DEBUG: Added gap space, mapping ({}, {}) -> abs ({}, {})", 
                            cleaned.len()-1, cleaned.len(), 
                            last_end + absolute_position, pos + absolute_position);
                        
                        // Map this space to the entire gap between Arabic characters
                        char_map.push((last_end + absolute_position, pos + absolute_position));
                        line_map.push(current_line);
                    }
                }
                
                // Add the Arabic character
                cleaned.push(normalized);
                // Add this logging for character mapping
                if cleaned.len() <= 10 || cleaned.len() % 50 == 0 {  // Limit logging to avoid overwhelming
                    debug!("MAPPING_DEBUG: Char '{}' at pos {} -> '{}' at index {} maps to abs ({}, {})", 
                        c, pos, normalized, cleaned.len()-1,
                        pos + absolute_position, pos + c.len_utf8() + absolute_position);
                }
                
                char_map.push((pos + absolute_position, pos + c.len_utf8() + absolute_position));
                line_map.push(current_line);
                last_arabic_end = Some(pos + c.len_utf8());
            } else if self.is_countable_char(c) && tag_depth > 0 {
                debug!("SKIPPED ARABIC: '{}' at position {} due to being inside tag (depth: {})",
                       c, pos, tag_depth);
            } else if !self.is_ignorable_char(c) {
                // For non-Arabic, non-ignorable characters, update the last_non_ignorable position
                last_non_ignorable = Some(pos + c.len_utf8());
            }
            // For ignorable characters (diacritics, tatweel), we don't do anything
            
            i += 1;
        }
                
        debug!("POSITION MAPPING: Character mapping example:");
        for i in 0..std::cmp::min(10, char_map.len()) {
            let char_idx = i;
            let abs_pos = char_map[i];
            let char = if char_idx < cleaned.len() {
                cleaned.chars().nth(char_idx).unwrap_or('?')
            } else { '?' };
            debug!("POSITION MAPPING: Char '{}' at index {} maps to original position {:?}", 
                char, char_idx, abs_pos);
        }
    
        debug!("MAPPING_DEBUG: Finished mapping {} original bytes -> {} cleaned chars", 
        text.len(), cleaned.len());
        if !char_map.is_empty() {
            debug!("MAPPING_DEBUG: First mapped position: {:?}, Last: {:?}", 
                char_map.first().unwrap(), char_map.last().unwrap());
        }
    
        Ok(TextMapping {
            cleaned_text: cleaned,
            char_map,
            line_map,
        })
    }

    fn is_countable_char(&self, c: char) -> bool {
        // Basic Arabic alphabet range (ا through ي)
        matches!(c, '\u{0621}'..='\u{063A}' | '\u{0641}'..='\u{064A}')
    }

    fn clean_text(&self, text: &str) -> Result<String> {
        Ok(self.clean_text_with_mapping(text)?.cleaned_text)
    }
    fn clean_text_with_mapping(&self, text: &str) -> Result<TextMapping> {
        self.clean_text_with_mapping_absolute(text, 0)
    }

    fn clean_text_with_context(&self, text: &str, _full_text: &str, _text_start: usize) -> Result<String> {
        self.clean_text(text)
    }

    fn count_valid_chars(&self, text: &str) -> usize {
        text.chars()
            .filter(|&c| self.is_countable_char(c))
            .count()
    }

    // Enhanced load_stop_words method that normalizes stop words/phrases
    fn load_stop_words<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        info!("Attempting to load stop words from: {:?}", path_ref);
        
        // Try to find file with different path variations
        let file = match File::open(path_ref) {
            Ok(file) => {
                info!("Successfully opened stop words file: {:?}", path_ref);
                file
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Try with current directory explicitly
                let alt_path = std::env::current_dir()?.join(path_ref);
                info!("Original path not found, trying: {:?}", alt_path);
                
                match File::open(&alt_path) {
                    Ok(file) => {
                        info!("Successfully opened alternative path: {:?}", alt_path);
                        file
                    },
                    Err(e2) => {
                        warn!("Could not find stop words file at either path: {:?} or {:?} (errors: {}, {})",
                              path_ref, alt_path, e, e2);
                        return Ok(());
                    }
                }
            },
            Err(e) => return Err(ParserError::IoError(e)),
        };

        info!("Loading stop words from file");
        let reader = io::BufReader::new(file);
        
        let mut count = 0;
        for line in reader.lines() {
            let phrase = line?.trim().to_string();
            if !phrase.is_empty() && !phrase.starts_with('#') {
                // Apply same cleaning rules as for regular text
                let cleaned = self.clean_text(&phrase)?;
                if !cleaned.is_empty() {
                    self.stop_phrases.insert(cleaned);
                    count += 1;
                }
            }
        }
        
        info!("Successfully loaded {} stop words/phrases", count);
        Ok(())
    }
    
    // Modified tokenize_text method that includes stop phrase filtering
    fn tokenize_text(&self, text: &str) -> Result<Vec<TextToken>> {
        let mapping = self.clean_text_with_mapping(text)?;
        
        let mut tokens = Vec::new();
        
        // Get character indices for the cleaned text
        let chars: Vec<(usize, char)> = mapping.cleaned_text.char_indices().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find word boundaries - positions where a word starts after space/beginning
        let mut word_starts = Vec::new();
        let mut in_word = false;
        
        for i in 0..chars.len() {
            let c = chars[i].1;
            if !in_word && !c.is_whitespace() {
                // Start of a new word after whitespace or at beginning
                word_starts.push(i);
                in_word = true;
            } else if in_word && c.is_whitespace() {
                // End of word
                in_word = false;
            }
        }
               
        // For each word start, find the corresponding end
        for (idx, &start_pos) in word_starts.iter().enumerate() {
            // Find where this word ends (at whitespace or text end)
            let mut end_pos = if idx + 1 < word_starts.len() {
                // Find whitespace before next word start
                let next_word_start = word_starts[idx + 1];
                
                // Look for whitespace between current word start and next word start
                let mut whitespace_pos = start_pos;
                for i in start_pos..next_word_start {
                    if chars[i].1.is_whitespace() {
                        whitespace_pos = i;
                        break;
                    }
                }
                
                // If we found whitespace, use it, otherwise use next word start
                if whitespace_pos > start_pos {
                    whitespace_pos
                } else {
                    next_word_start
                }
            } else {
                // Last word - find next whitespace or end of string
                let mut end = chars.len();
                for i in start_pos..chars.len() {
                    if chars[i].1.is_whitespace() {
                        end = i;
                        break;
                    }
                }
                end
            };
            
            // Ensure the end position is within bounds
            if end_pos >= chars.len() {
                end_pos = chars.len();
            }
            
            // Ensure we don't exceed mapping bounds
            if start_pos >= mapping.char_map.len() || end_pos > mapping.char_map.len() {
                debug!("Word boundary exceeds mapping bounds: {}..{} (mapping length: {})",
                       start_pos, end_pos, mapping.char_map.len());
                continue;
            }
            
            // Get original text positions from the mapping
            let start_byte = mapping.char_map[start_pos].0;
            let end_byte = if end_pos < mapping.char_map.len() {
                mapping.char_map[end_pos - 1].1
            } else if !mapping.char_map.is_empty() {
                mapping.char_map.last().unwrap().1
            } else {
                start_byte // Fallback if mapping is empty
            };
            
            let line_number = mapping.line_map[start_pos];
            
            // Extract the word text from the characters
            let word_text = chars[start_pos..end_pos]
                .iter()
                .map(|(_, c)| *c)
                .collect::<String>();
                       
            tokens.push(TextToken {
                original_text: text[start_byte..end_byte].to_string(),
                cleaned_text: word_text,
                start_byte,
                end_byte,
                line_number,
            });
        }
                
        // Apply stop phrase filtering
        if !self.stop_phrases.is_empty() {
            tokens = self.filter_stop_phrases(tokens);
        }
        
        
        Ok(tokens)
    }
    fn tokenize_text_with_mapping(&self, mapping: &TextMapping) -> Result<Vec<TextToken>> {
        let text = &mapping.cleaned_text;
        let mut tokens = Vec::new();
        
        // Find word boundaries in the cleaned text
        let chars: Vec<char> = text.chars().collect();
        let mut word_starts = Vec::new();
        let mut in_word = false;
        
        for i in 0..chars.len() {
            let c = chars[i];
            if !in_word && !c.is_whitespace() {
                // Start of a new word
                word_starts.push(i);
                in_word = true;
            } else if in_word && c.is_whitespace() {
                // End of word
                in_word = false;
            }
        }
        
        // For each word start, create a token with absolute positions
        for (idx, &start_pos) in word_starts.iter().enumerate() {
            // Find word end (at whitespace or text end)
            let end_pos = if idx + 1 < word_starts.len() {
                // Find whitespace before next word
                let mut pos = start_pos;
                while pos < word_starts[idx + 1] {
                    if chars[pos].is_whitespace() {
                        break;
                    }
                    pos += 1;
                }
                pos
            } else {
                // Last word - find next whitespace or end
                let mut pos = start_pos;
                while pos < chars.len() {
                    if chars[pos].is_whitespace() {
                        break;
                    }
                    pos += 1;
                }
                pos
            };
            
            // Skip if out of bounds
            if start_pos >= mapping.char_map.len() || end_pos > mapping.char_map.len() {
                continue;
            }
            
            // Get ABSOLUTE positions from mapping
            let start_byte = mapping.char_map[start_pos].0;
            let end_byte = if end_pos > 0 && end_pos <= mapping.char_map.len() {
                mapping.char_map[end_pos - 1].1
            } else {
                mapping.char_map[mapping.char_map.len() - 1].1
            };
            
            let word_text = chars[start_pos..end_pos]
                .iter()
                .collect::<String>();
            
            let line_number = mapping.line_map[start_pos];
            
            tokens.push(TextToken {
                original_text: word_text.clone(),
                cleaned_text: word_text,
                start_byte,  // Absolute position
                end_byte,    // Absolute position
                line_number,
            });
        }
        
        // Apply stop phrase filtering
        if !self.stop_phrases.is_empty() {
            tokens = self.filter_stop_phrases(tokens);
        }
        
        Ok(tokens)
    }
}