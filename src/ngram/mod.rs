pub mod generator;
pub mod types;

pub use self::generator::NGramGenerator;
pub use self::types::NGramConfig;

pub use crate::error::{Error, Result};