pub mod algorithms;
pub mod similarity;
pub mod cluster;
pub mod types;
pub mod parallel;
pub mod ngram_pipeline;
pub mod banality;
pub mod concurrent;
// Re-export the main types
pub use self::cluster::ClusteringMatcher;
pub use self::similarity::SimilarityCalculator;
pub use self::parallel::ParallelMatcher;
pub use self::ngram_pipeline::ProcessingPipeline;
pub use self::ngram_pipeline::StorageAccessStrategy;
pub use self::types::{
    MatchResult,
    SimpleMatchResult,
    TextSpan,
    NGramMatch,
    MatchCandidate
};