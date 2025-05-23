// cluster/mod.rs
pub mod matcher;
pub mod chains;
pub mod validation;
pub mod parallel_chains;
pub mod spatial;



// Re-export the main struct to maintain the existing public API
pub use self::matcher::ClusteringMatcher;