// src/matcher/ngrams_matcher/ngrams_grid/dimensions.rs

use log::info;

pub fn calculate_grid_dimensions(
    source_count: usize,
    target_count: usize,
    comparisons_per_cell: usize,
) -> (usize, usize) {
    let total_comparisons = source_count * target_count;
    
    // Calculate cells needed for target cell size
    let cells_needed = (total_comparisons + comparisons_per_cell - 1) / comparisons_per_cell;
    let cells_needed = cells_needed.max(1);  // At least 1 cell
    
    // Create square grid
    let grid_size = (cells_needed as f64).sqrt().ceil() as usize;
    
    // Adjust for aspect ratio if dataset is very non-square
    let aspect_ratio = source_count as f64 / target_count.max(1) as f64;
    let (source_parts, target_parts) = if aspect_ratio > 3.0 {
        // Source much larger
        let s = ((cells_needed as f64 * aspect_ratio.sqrt()).ceil()) as usize;
        let t = (cells_needed + s - 1) / s;
        (s.max(1), t.max(1))
    } else if aspect_ratio < 0.33 {
        // Target much larger  
        let t = ((cells_needed as f64 / aspect_ratio.sqrt()).ceil()) as usize;
        let s = (cells_needed + t - 1) / t;
        (s.max(1), t.max(1))
    } else {
        (grid_size, grid_size)
    };
    
    // Log configuration
    let actual_cells = source_parts * target_parts;
    let actual_comparisons_per_cell = total_comparisons / actual_cells;
    
    info!("Grid: {}×{} = {} cells for {:.1}B total comparisons",
        source_parts, target_parts, actual_cells,
        total_comparisons as f64 / 1_000_000_000.0);
    info!("Cell size: target={}M, actual={}M comparisons/cell",
        comparisons_per_cell / 1_000_000,
        actual_comparisons_per_cell / 1_000_000);
    
    (source_parts, target_parts)
}