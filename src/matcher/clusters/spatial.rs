use rstar::{RTree, RTreeObject, AABB};
use crate::matcher::types::ReconstructedSegment;

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
pub fn build_rtree(matches: &[ReconstructedSegment]) -> RTree<MatchRect> {
    // Convert each match to a MatchRect with its spatial coordinates
    let rects: Vec<MatchRect> = matches.iter().enumerate()
        .map(|(i, m)| MatchRect {
            match_index: i,
            source_start: m.source_start_seq as i64,  // Direct field access, cast from u64
            source_end: m.source_end_seq as i64,      // Direct field access, cast from u64
            target_start: m.target_start_seq as i64,  // Direct field access, cast from u64
            target_end: m.target_end_seq as i64,      // Direct field access, cast from u64
        })
        .collect();
    
    // Bulk load is much more efficient than individual insertions
    RTree::bulk_load(rects)
}
/// Finds all matches that overlap with the given match
pub fn find_overlapping_candidates(
    rtree: &RTree<MatchRect>, 
    match_result: &ReconstructedSegment  // Changed from MatchResult
) -> Vec<usize> {
    // Create search envelope with the exact dimensions of the match
    // Converting u64 positions to i64 for R-tree compatibility
    let search_area = AABB::from_corners(
        [match_result.source_start_seq as i64, match_result.target_start_seq as i64],  // Bottom-left corner
        [match_result.source_end_seq as i64, match_result.target_end_seq as i64]      // Top-right corner
    );
    
    // Find all matches that intersect with this envelope and return their indices
    rtree.locate_in_envelope_intersecting(&search_area)
        .map(|rect| rect.match_index)
        .collect()
}

/// Finds all matches that are adjacent to the given match (within the specified gap ratio)
pub fn find_adjacent_candidates(
    rtree: &RTree<MatchRect>,
    match_result: &ReconstructedSegment,  // Changed from MatchResult
    adjacent_merge_ratio: f64
) -> Vec<usize> {
    // Calculate maximum allowed gap based on match size
    // This determines how far away we'll look for adjacent matches
    let source_size = match_result.source_end_seq - match_result.source_start_seq;
    let target_size = match_result.target_end_seq - match_result.target_start_seq;
    
    // The gap is proportional to the match size - larger matches can have larger gaps
    let max_source_gap = (source_size as f64 * adjacent_merge_ratio).ceil() as u64;
    let max_target_gap = (target_size as f64 * adjacent_merge_ratio).ceil() as u64;
    
    // Create expanded envelope that includes the adjacent area
    // We expand the match's rectangle in all directions by the calculated gap
    let search_area = AABB::from_corners(
        [
            // Bottom-left corner: subtract gap from start positions
            // saturating_sub ensures we don't go below 0
            match_result.source_start_seq.saturating_sub(max_source_gap) as i64,
            match_result.target_start_seq.saturating_sub(max_target_gap) as i64
        ],
        [
            // Top-right corner: add gap to end positions
            (match_result.source_end_seq + max_source_gap) as i64,
            (match_result.target_end_seq + max_target_gap) as i64
        ]
    );
    
    // Find all matches within the expanded area
    rtree.locate_in_envelope_intersecting(&search_area)
        .map(|rect| rect.match_index)
        .collect()
}

// ========================================================
// CLUSTER SPATIAL CODE
// =========================================================
// matcher/clusters/spatial.rs

use std::collections::HashMap;
use log::{debug, trace};
use crate::matcher::types::SequenceMatch;
use crate::matcher::is_traced_sequence;
use crate::storage::LMDBStorage;

// ============================================================================
// GRID CELL STRUCTURE
// ============================================================================

/// A 2D grid cell coordinate in (source_sequence, target_sequence) space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    source_cell: i64,  // Grid X coordinate
    target_cell: i64,  // Grid Y coordinate
}

impl GridCell {
    /// Map a match to its grid cell
    /// Example: match at (450, 780) with cell_size=100 → Cell(4, 7)
    #[inline]
    fn from_match(match_item: &SequenceMatch, cell_size: u64) -> Self {
        Self {
            source_cell: (match_item.source_sequence / cell_size) as i64,
            target_cell: (match_item.target_sequence / cell_size) as i64,
        }
    }
    
    /// Get all 9 neighboring cells (including self) for proximity search
    /// 
    /// Why 9? A match near a cell boundary might be close to matches in adjacent cells.
    /// 
    /// Visual layout:
    /// ```
    /// [NW] [N] [NE]    (-1,-1) (0,-1) (+1,-1)
    /// [W]  [C] [E]  →  (-1, 0) (0, 0) (+1, 0)
    /// [SW] [S] [SE]    (-1,+1) (0,+1) (+1,+1)
    /// ```
    fn neighbors(&self) -> [GridCell; 9] {
        [
            // Top row
            GridCell { source_cell: self.source_cell - 1, target_cell: self.target_cell - 1 },
            GridCell { source_cell: self.source_cell,     target_cell: self.target_cell - 1 },
            GridCell { source_cell: self.source_cell + 1, target_cell: self.target_cell - 1 },
            
            // Middle row
            GridCell { source_cell: self.source_cell - 1, target_cell: self.target_cell },
            GridCell { source_cell: self.source_cell,     target_cell: self.target_cell },  // Self
            GridCell { source_cell: self.source_cell + 1, target_cell: self.target_cell },
            
            // Bottom row
            GridCell { source_cell: self.source_cell - 1, target_cell: self.target_cell + 1 },
            GridCell { source_cell: self.source_cell,     target_cell: self.target_cell + 1 },
            GridCell { source_cell: self.source_cell + 1, target_cell: self.target_cell + 1 },
        ]
    }
}
// ============================================================================
// SPATIAL INDEX
// ============================================================================

/// Spatial hash grid for O(1) proximity queries
/// 
/// Instead of checking every match against every other match (O(n²)),
/// we organize matches into a 2D grid and only check nearby cells (O(1) per query).
pub struct SpatialIndex {
    /// Grid cells mapping to indices in the original matches array
    /// Example: grid[Cell(5,3)] = vec![42, 87, 103] means matches 42, 87, 103 are in cell (5,3)
    grid: HashMap<GridCell, Vec<usize>>,
    
    /// Size of each grid cell (typically same as proximity_distance)
    cell_size: u64,
    
    /// Total number of matches indexed (for statistics)
    match_count: usize,
}

impl SpatialIndex {
    /// Build spatial index from matches - O(n) complexity
    /// 
    /// This is the "building the library catalog" phase - we do it once,
    /// then use it many times for fast lookups.
    /// 
    /// # Arguments
    /// * `matches` - The match array to index
    /// * `proximity_distance` - Cell size (matches within this distance are candidates)
    pub fn new(matches: &[SequenceMatch], proximity_distance: usize) -> Self {
        let start = std::time::Instant::now();
        let cell_size = proximity_distance as u64;
        
        // Pre-allocate: empirically, ~8-12 matches per cell is typical
        // Better to slightly over-allocate than repeatedly resize
        let estimated_cells = (matches.len() / 10).max(16);
        let mut grid: HashMap<GridCell, Vec<usize>> = HashMap::with_capacity(estimated_cells);
        
        // Single pass through all matches: O(n)
        for (idx, match_item) in matches.iter().enumerate() {
            // Hash this match to its grid cell
            let cell = GridCell::from_match(match_item, cell_size);
            
            // Add this match's index to the cell's list
            grid.entry(cell)
                .or_insert_with(|| Vec::with_capacity(16))  // Start with room for 16 matches per cell
                .push(idx);
        }
        
        // Shrink vectors to reduce memory overhead
        // After we know the final size, we can trim excess capacity
        for cell_matches in grid.values_mut() {
            cell_matches.shrink_to_fit();
        }
        
        // Log useful statistics
        let cell_count = grid.len();
        let avg_per_cell = matches.len() as f64 / cell_count as f64;
        
        debug!("Built spatial index: {} matches → {} cells ({:.1} matches/cell avg) in {:?}",
            matches.len(), cell_count, avg_per_cell, start.elapsed());
        
        Self { 
            grid, 
            cell_size,
            match_count: matches.len(),
        }
    }
    
    /// Query for all matches that could be within proximity of given match
    /// Returns indices into original matches array - O(1) amortized
    /// 
    /// This is the "find books near this shelf" operation - super fast!
    #[inline]
    pub fn query_neighbors(&self, match_item: &SequenceMatch) -> Vec<usize> {
        // Find which cell this match is in
        let cell = GridCell::from_match(match_item, self.cell_size);
        
        // Get all 9 neighboring cells (including self)
        let neighbors = cell.neighbors();
        
        // Estimate capacity: ~10 matches per cell × 9 cells = 90
        let mut candidates = Vec::with_capacity(90);
        
        // Collect matches from all 9 neighboring cells
        for neighbor_cell in &neighbors {
            if let Some(indices) = self.grid.get(neighbor_cell) {
                candidates.extend_from_slice(indices);
            }
        }
        
        candidates
    }
    
    /// Get statistics about the spatial index (for debugging/logging)
    pub fn stats(&self) -> SpatialIndexStats {
        let cell_sizes: Vec<usize> = self.grid.values().map(|v| v.len()).collect();
        let max_cell_size = *cell_sizes.iter().max().unwrap_or(&0);
        let min_cell_size = *cell_sizes.iter().min().unwrap_or(&0);
        let avg_cell_size = if !cell_sizes.is_empty() {
            cell_sizes.iter().sum::<usize>() as f64 / cell_sizes.len() as f64
        } else {
            0.0
        };
        
        SpatialIndexStats {
            total_matches: self.match_count,
            cell_count: self.grid.len(),
            avg_matches_per_cell: avg_cell_size,
            max_matches_per_cell: max_cell_size,
            min_matches_per_cell: min_cell_size,
            memory_bytes: self.estimate_memory_usage(),
        }
    }
    
    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        // HashMap overhead + (key + Vec header + Vec data)
        let hashmap_overhead = std::mem::size_of::<HashMap<GridCell, Vec<usize>>>();
        let per_cell = std::mem::size_of::<GridCell>() + std::mem::size_of::<Vec<usize>>();
        let per_index = std::mem::size_of::<usize>();
        
        hashmap_overhead 
            + (self.grid.len() * per_cell)
            + (self.match_count * per_index)
    }
}

/// Statistics about a spatial index
#[derive(Debug)]
pub struct SpatialIndexStats {
    pub total_matches: usize,
    pub cell_count: usize,
    pub avg_matches_per_cell: f64,
    pub max_matches_per_cell: usize,
    pub min_matches_per_cell: usize,
    pub memory_bytes: usize,
}

// ============================================================================
// PROXIMITY CLUSTERING WITH SPATIAL INDEX
// ============================================================================

/// Proximity-based clustering using spatial indexing - O(n) instead of O(n²)
/// 
/// ALGORITHM OVERVIEW:
/// This replaces the original proximate_chains with a spatial index approach:
/// 
/// 1. Build 2D grid over (source_seq, target_seq) space - O(n)
/// 2. For each unprocessed match:
///    a. Query its 9 neighboring cells - O(1)
///    b. Check actual distances only for candidates in those cells - O(k), k ≈ 10-50
///    c. Use breadth-first search to grow the cluster
/// 3. Repeat until all matches are processed
/// 
/// # Performance Comparison
/// - Original: O(n²) - checks every match against every other match
///   Example: 100K matches = 10 billion comparisons
/// 
/// - Spatial:  O(n×k) where k = avg matches per cell (typically 5-20)
///   Example: 100K matches × 50 candidates = 5 million comparisons
///   Speedup: 2000x!
/// 
/// # Arguments
/// * `matches` - Array of sequence matches to cluster
/// * `proximity_distance` - Maximum distance for matches to be in same cluster
/// * `traced_phrase` - Debug phrase to trace through algorithm (empty string = no tracing)
/// * `source_storage` - Optional storage for tracing
pub fn proximate_chains_spatial(
    matches: &[SequenceMatch],
    proximity_distance: usize,
    traced_phrase: &str,
    source_storage: Option<&LMDBStorage>,
) -> Vec<Vec<usize>> {
    if matches.is_empty() {
        return Vec::new();
    }
    
    let total_start = std::time::Instant::now();
    
    // STEP 1: Build spatial index - O(n)
    // This is the "one-time setup cost"
    let spatial_index = SpatialIndex::new(matches, proximity_distance);
    
    // Log statistics if debug logging enabled
    if log::log_enabled!(log::Level::Debug) {
        let stats = spatial_index.stats();
        debug!("Spatial index stats: {} cells, {:.1} avg matches/cell, max {} matches/cell, {:.2} MB memory",
            stats.cell_count,
            stats.avg_matches_per_cell,
            stats.max_matches_per_cell,
            stats.memory_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    
    let cluster_start = std::time::Instant::now();
    
    // STEP 2: Build clusters using spatial queries
    let mut processed = vec![false; matches.len()];  // Track which matches we've assigned
    let mut clusters = Vec::new();
    let proximity_distance_u64 = proximity_distance as u64;
    
    // Track statistics for logging
    let mut total_candidates_checked = 0usize;
    let mut total_distance_checks = 0usize;
    
    // STEP 3: Process each match
    for i in 0..matches.len() {
        if processed[i] {
            continue;  // Already in a cluster
        }
        
        // Start new cluster with this match
        let mut cluster = vec![i];
        let mut to_explore = vec![i];  // Breadth-first search queue
        processed[i] = true;
        
        // Check if this cluster starts with traced n-gram (for debugging)
        let cluster_has_traced = !traced_phrase.is_empty() 
            && log::log_enabled!(log::Level::Debug)
            && source_storage.is_some()
            && is_traced_sequence(matches[i].source_sequence, source_storage, traced_phrase);
        
        if cluster_has_traced {
            debug!("TRACE [spatial]: Starting new cluster with traced n-gram at source={}, target={}",
                matches[i].source_sequence, matches[i].target_sequence);
        }
        
        // STEP 4: Breadth-first expansion using spatial queries
        // Keep growing the cluster until we can't find any more nearby matches
        while let Some(current_idx) = to_explore.pop() {
            let current_match = &matches[current_idx];
            
            // Query spatial index for nearby candidates - O(1) amortized
            // This replaces checking ALL matches (O(n)) with checking ~90 candidates!
            let candidates = spatial_index.query_neighbors(current_match);
            total_candidates_checked += candidates.len();
            
            // Check actual distances for unprocessed candidates
            for &candidate_idx in &candidates {
                if processed[candidate_idx] {
                    continue;  // Already in this or another cluster
                }
                
                let candidate = &matches[candidate_idx];
                total_distance_checks += 1;
                
                // Precise distance check using abs_diff (avoids overflow)
                let source_dist = current_match.source_sequence
                    .abs_diff(candidate.source_sequence);
                let target_dist = current_match.target_sequence
                    .abs_diff(candidate.target_sequence);
                
                // Is this candidate actually close enough?
                if source_dist <= proximity_distance_u64 && 
                   target_dist <= proximity_distance_u64 {
                    // Yes! Add to cluster and mark for exploration
                    cluster.push(candidate_idx);
                    to_explore.push(candidate_idx);
                    processed[candidate_idx] = true;
                    
                    // Trace if this match contains traced n-gram
                    if cluster_has_traced {
                        let match_has_traced = is_traced_sequence(
                            candidate.source_sequence, 
                            source_storage, 
                            traced_phrase
                        );
                        if match_has_traced {
                            trace!("TRACE [spatial]: Extended traced cluster with match at source={}, target={}",
                                candidate.source_sequence, candidate.target_sequence);
                        }
                    }
                }
            }
        }
        
        // STEP 5: Save cluster if it has 2+ matches (same as original behavior)
        if cluster.len() >= 2 {
            if cluster_has_traced {
                debug!("TRACE [spatial]: Completed traced cluster with {} matches", cluster.len());
            }
            clusters.push(cluster);
        } else if cluster_has_traced {
            debug!("TRACE [spatial]: DISCARDED traced cluster with only 1 match");
        }
    }
    
    let cluster_time = cluster_start.elapsed();
    let total_time = total_start.elapsed();
    
    // Log performance metrics
    debug!("Spatial clustering complete: {} matches → {} clusters in {:?} (index: {:?}, clustering: {:?})",
        matches.len(), clusters.len(), total_time, 
        total_time - cluster_time, cluster_time);
    
    // Log efficiency statistics (trace level)
    if log::log_enabled!(log::Level::Trace) {
        let efficiency = if total_candidates_checked > 0 {
            (total_distance_checks as f64 / total_candidates_checked as f64) * 100.0
        } else {
            0.0
        };
        trace!("  Candidates examined: {}", total_candidates_checked);
        trace!("  Distance checks: {}", total_distance_checks);
        trace!("  Efficiency: {:.1}% (lower = better spatial pruning)", efficiency);
        trace!("  Speedup vs O(n²): {:.1}x", 
            (matches.len() * matches.len()) as f64 / total_distance_checks as f64);
    }
    
    clusters
}