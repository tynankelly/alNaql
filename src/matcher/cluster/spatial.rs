use rstar::{RTree, RTreeObject, AABB};
use crate::matcher::MatchResult;

/// Wrapper around match data to make it compatible with R-Tree spatial indexing
pub struct MatchRect {
    /// Original match index in the collection
    pub match_index: usize,
    /// Spatial boundaries in the source-target coordinate space (using i64 for R-tree compatibility)
    pub source_start: i64,
    pub source_end: i64,
    pub target_start: i64,
    pub target_end: i64,
}

impl RTreeObject for MatchRect {
    type Envelope = AABB<[i64; 2]>;  // Changed from usize to i64

    fn envelope(&self) -> Self::Envelope {
        // Create a 2D bounding box with corners at (source_start,target_start) and (source_end,target_end)
        AABB::from_corners(
            [self.source_start, self.target_start],
            [self.source_end, self.target_end]
        )
    }
}
/// Builds an R-tree spatial index from a collection of matches
/// 
/// This uses bulk loading for optimal performance and tree balance.
/// Builds an R-tree spatial index from a collection of matches
pub fn build_rtree(matches: &[MatchResult]) -> RTree<MatchRect> {
    // Convert each match to a MatchRect with its spatial coordinates
    let rects: Vec<MatchRect> = matches.iter().enumerate()
        .map(|(i, m)| MatchRect {
            match_index: i,
            source_start: m.source_range.start as i64,  // Convert to i64
            source_end: m.source_range.end as i64,      // Convert to i64
            target_start: m.target_range.start as i64,  // Convert to i64
            target_end: m.target_range.end as i64,      // Convert to i64
        })
        .collect();
    
    // Bulk load is much more efficient than individual insertions
    RTree::bulk_load(rects)
}
/// Finds all matches that overlap with the given match
pub fn find_overlapping_candidates(rtree: &RTree<MatchRect>, match_result: &MatchResult) -> Vec<usize> {
    // Create search envelope with the exact dimensions of the match
    // Converting usize positions to i64 for R-tree compatibility
    let search_area = AABB::from_corners(
        [match_result.source_range.start as i64, match_result.target_range.start as i64],
        [match_result.source_range.end as i64, match_result.target_range.end as i64]
    );
    
    // Find all matches that intersect with this envelope and return their indices
    rtree.locate_in_envelope_intersecting(&search_area)
        .map(|rect| rect.match_index)
        .collect()
}

/// Finds all matches that are adjacent to the given match (within the specified gap ratio)
pub fn find_adjacent_candidates(
    rtree: &RTree<MatchRect>, 
    match_result: &MatchResult,
    adjacent_merge_ratio: f64
) -> Vec<usize> {
    // Calculate maximum allowed gap based on match size
    let source_size = match_result.source_range.end - match_result.source_range.start;
    let target_size = match_result.target_range.end - match_result.target_range.start;
    
    let max_source_gap = (source_size as f64 * adjacent_merge_ratio).ceil() as usize;
    let max_target_gap = (target_size as f64 * adjacent_merge_ratio).ceil() as usize;
    
    // Create expanded envelope that includes the adjacent area
    // Converting all usize values to i64 for R-tree compatibility
    let search_area = AABB::from_corners(
        [
            match_result.source_range.start.saturating_sub(max_source_gap) as i64, 
            match_result.target_range.start.saturating_sub(max_target_gap) as i64
        ],
        [
            (match_result.source_range.end + max_source_gap) as i64,
            (match_result.target_range.end + max_target_gap) as i64
        ]
    );
    
    // Find all matches within the expanded area
    rtree.locate_in_envelope_intersecting(&search_area)
        .map(|rect| rect.match_index)
        .collect()
}