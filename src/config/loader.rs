// config/loader.rs - Centralized INI parsing infrastructure

use std::path::PathBuf;
use crate::error::{Error, Result};

/// Trait for types that can be configured from INI files
pub trait Configurable: Default {
    /// Parse a key-value pair for this configuration
    /// Returns Ok(true) if the field was handled, Ok(false) if not recognized
    fn set_field(&mut self, key: &str, value: &str) -> Result<bool>;
    
    /// Validate the configuration after loading
    fn validate(&self) -> Result<()>;
    
    /// Get the INI section name for this config
    fn section_name() -> &'static str;
}

/// Helper functions for parsing common types
pub mod parse {
    use super::*;
    
    pub fn bool(value: &str) -> Result<bool> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid boolean value: {}", value))
        )
    }
    
    pub fn usize(value: &str) -> Result<usize> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid number: {}", value))
        )
    }
    
    pub fn u32(value: &str) -> Result<u32> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid u32 number: {}", value))
        )
    }
    
    pub fn u64(value: &str) -> Result<u64> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid u64 number: {}", value))
        )
    }
    
    pub fn i32(value: &str) -> Result<i32> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid i32 number: {}", value))
        )
    }
    
    pub fn f64(value: &str) -> Result<f64> {
        value.parse().map_err(|_| 
            Error::Config(format!("Invalid float: {}", value))
        )
    }
    
    pub fn path(value: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(value.trim_matches('"')))
    }
    
    pub fn string(value: &str) -> Result<String> {
        Ok(value.trim_matches('"').to_string())
    }
    
    pub fn optional_path(value: &str) -> Result<Option<PathBuf>> {
        if value.is_empty() || value == "none" || value == "null" {
            Ok(None)
        } else {
            Ok(Some(PathBuf::from(value.trim_matches('"'))))
        }
    }
    
    pub fn optional_u32(value: &str) -> Result<Option<u32>> {
        if value.is_empty() || value == "none" || value == "null" {
            Ok(None)
        } else {
            Ok(Some(value.parse().map_err(|_| 
                Error::Config(format!("Invalid optional u32: {}", value))
            )?))
        }
    }
    
    pub fn optional_usize(value: &str) -> Result<Option<usize>> {
        if value.is_empty() || value == "none" || value == "null" {
            Ok(None)
        } else {
            Ok(Some(value.parse().map_err(|_| 
                Error::Config(format!("Invalid optional usize: {}", value))
            )?))
        }
    }
    
    pub fn optional_bool(value: &str) -> Result<Option<bool>> {
        if value.is_empty() || value == "none" || value == "null" {
            Ok(None)
        } else {
            Ok(Some(value.parse().map_err(|_| 
                Error::Config(format!("Invalid optional bool: {}", value))
            )?))
        }
    }
}

/// Macro to reduce boilerplate in set_field implementations
#[macro_export]
macro_rules! config_fields {
    ($self:ident, $key:ident, $value:ident, {
        $($field:ident => $parser:ident $(| $validator:expr)?),* $(,)?
    }) => {
        match $key {
            $(
                stringify!($field) => {
                    let parsed = $crate::config::loader::parse::$parser($value)?;
                    $(
                        // Optional inline validation
                        let validator: fn(&_) -> bool = $validator;
                        if !validator(&parsed) {
                            return Err($crate::error::Error::Config(
                                format!("Invalid value for {}: {}", stringify!($field), $value)
                            ));
                        }
                    )?
                    $self.$field = parsed;
                    Ok(true)
                },
            )*
            _ => Ok(false)
        }
    };
}