// src/utils/job_scheduler.rs

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use log::{info, debug, trace};

use crate::utils::job_tracker::ComparisonJob;
use crate::config::AlNaqlConfig;

/// Scheduler for organizing jobs into non-conflicting batches
pub struct JobScheduler {
    /// All jobs to be scheduled
    jobs: Vec<ComparisonJob>,
    
    /// Conflict graph: job_id -> set of conflicting job_ids
    conflict_graph: HashMap<usize, HashSet<usize>>,
    
    /// Database usage map: database_path -> set of job_ids using it
    database_usage: HashMap<PathBuf, HashSet<usize>>,
    
    /// Scheduling strategy
    strategy: String,
}

impl JobScheduler {
    /// Create a new scheduler with the given jobs
    pub fn new(jobs: Vec<ComparisonJob>, config: &AlNaqlConfig) -> Self {
        let mut scheduler = Self {
            jobs: jobs.clone(),
            conflict_graph: HashMap::new(),
            database_usage: HashMap::new(),
            strategy: config.execution.parallel_strategy.clone(),
        };
        
        scheduler.build_conflict_graph();
        info!("JobScheduler initialized with {} jobs, strategy: {}", 
            jobs.len(), scheduler.strategy);
        
        scheduler
    }
    
    /// Build the conflict graph based on database usage
    fn build_conflict_graph(&mut self) {
        // First, build database usage map
        for job in &self.jobs {
            // Track which jobs use each database
            self.database_usage
                .entry(job.source_db_path.clone())
                .or_insert_with(HashSet::new)
                .insert(job.comparison_number);
            
            self.database_usage
                .entry(job.target_db_path.clone())
                .or_insert_with(HashSet::new)
                .insert(job.comparison_number);
        }
        
        // Build conflict graph based on strategy
        match self.strategy.as_str() {
            "smart" => self.build_smart_conflicts(),
            "pooled" => self.build_pooled_conflicts(),
            "unrestricted" => self.build_no_conflicts(),
            _ => self.build_smart_conflicts(), // Default to smart
        }
        
        // Log conflict statistics
        let total_conflicts: usize = self.conflict_graph.values()
            .map(|conflicts| conflicts.len())
            .sum();
        let avg_conflicts = if !self.conflict_graph.is_empty() {
            total_conflicts as f64 / self.conflict_graph.len() as f64
        } else {
            0.0
        };
        
        debug!("Conflict graph built: {} total conflicts, {:.1} avg per job", 
            total_conflicts, avg_conflicts);
    }
    
    /// Build conflicts for smart scheduling (no shared databases)
    fn build_smart_conflicts(&mut self) {
        // Build conflict pairs first
        let mut conflict_pairs = Vec::new();
        
        for i in 0..self.jobs.len() {
            for j in i + 1..self.jobs.len() {
                let job_i = &self.jobs[i];
                let job_j = &self.jobs[j];
                
                if self.has_database_conflict(job_i, job_j) {
                    conflict_pairs.push((job_i.comparison_number, job_j.comparison_number));
                }
            }
        }
        
        // Now add conflicts to the graph
        for (job_i_num, job_j_num) in conflict_pairs {
            self.conflict_graph
                .entry(job_i_num)
                .or_insert_with(HashSet::new)
                .insert(job_j_num);
            
            self.conflict_graph
                .entry(job_j_num)
                .or_insert_with(HashSet::new)
                .insert(job_i_num);
        }
    }
    
    /// Build conflicts for pooled scheduling (allow limited sharing)
    fn build_pooled_conflicts(&mut self) {
        // Count how many jobs use each database
        let db_usage_counts: HashMap<PathBuf, usize> = self.database_usage
            .iter()
            .map(|(path, jobs)| (path.clone(), jobs.len()))
            .collect();
        
        // Build conflict pairs first
        let mut conflict_pairs = Vec::new();
        
        for i in 0..self.jobs.len() {
            let job_i = &self.jobs[i];
            
            // Only create conflicts for heavily used databases (>2 concurrent users)
            let source_heavy = db_usage_counts.get(&job_i.source_db_path)
                .map(|&count| count > 2)
                .unwrap_or(false);
            let target_heavy = db_usage_counts.get(&job_i.target_db_path)
                .map(|&count| count > 2)
                .unwrap_or(false);
            
            if !source_heavy && !target_heavy {
                continue;
            }
            
            for j in i + 1..self.jobs.len() {
                let job_j = &self.jobs[j];
                
                // Only conflict if both jobs use the same heavily-used database
                if (source_heavy && (job_j.source_db_path == job_i.source_db_path || 
                                    job_j.target_db_path == job_i.source_db_path)) ||
                (target_heavy && (job_j.source_db_path == job_i.target_db_path || 
                                    job_j.target_db_path == job_i.target_db_path)) {
                    
                    conflict_pairs.push((job_i.comparison_number, job_j.comparison_number));
                }
            }
        }
        
        // Now add conflicts to the graph
        for (job_i_num, job_j_num) in conflict_pairs {
            self.conflict_graph
                .entry(job_i_num)
                .or_insert_with(HashSet::new)
                .insert(job_j_num);
            
            self.conflict_graph
                .entry(job_j_num)
                .or_insert_with(HashSet::new)
                .insert(job_i_num);
        }
    }
    
    /// Build no conflicts for unrestricted scheduling
    fn build_no_conflicts(&mut self) {
        // No conflicts - all jobs can run in parallel
        for job in &self.jobs {
            self.conflict_graph.insert(job.comparison_number, HashSet::new());
        }
    }
    
    /// Check if two jobs have a database conflict
    fn has_database_conflict(&self, job1: &ComparisonJob, job2: &ComparisonJob) -> bool {
        job1.source_db_path == job2.source_db_path ||
        job1.source_db_path == job2.target_db_path ||
        job1.target_db_path == job2.source_db_path ||
        job1.target_db_path == job2.target_db_path
    }
    
    /// Get parallel batches using graph coloring algorithm
    pub fn get_parallel_batches(&self, max_parallel: usize) -> Vec<Vec<ComparisonJob>> {
        info!("Computing parallel batches with max_parallel={}", max_parallel);
        
        // Use greedy graph coloring
        let coloring = self.greedy_graph_coloring(max_parallel);
        
        // Group jobs by color (batch)
        let mut batches: HashMap<usize, Vec<ComparisonJob>> = HashMap::new();
        
        for job in &self.jobs {
            if let Some(&color) = coloring.get(&job.comparison_number) {
                batches.entry(color)
                    .or_insert_with(Vec::new)
                    .push(job.clone());
            }
        }
        
        // Convert to vector and sort by batch number
        let mut result: Vec<Vec<ComparisonJob>> = batches.into_iter()
            .map(|(_, jobs)| jobs)
            .collect();
        
        // Sort batches by size (largest first) for better load balancing
        result.sort_by(|a, b| b.len().cmp(&a.len()));
        
        info!("Created {} batches from {} jobs", result.len(), self.jobs.len());
        for (i, batch) in result.iter().enumerate() {
            debug!("  Batch {}: {} jobs", i, batch.len());
        }
        
        result
    }
    
    /// Greedy graph coloring algorithm
    fn greedy_graph_coloring(&self, max_colors: usize) -> HashMap<usize, usize> {
        let mut coloring: HashMap<usize, usize> = HashMap::new();
        
        // Sort jobs by number of conflicts (highest first)
        // This heuristic tends to produce better colorings
        let mut sorted_jobs = self.jobs.clone();
        sorted_jobs.sort_by_key(|job| {
            self.conflict_graph
                .get(&job.comparison_number)
                .map(|conflicts| conflicts.len())
                .unwrap_or(0)
        });
        sorted_jobs.reverse();
        
        for job in sorted_jobs {
            let job_id = job.comparison_number;
            
            // Find colors used by conflicting jobs
            let mut used_colors = HashSet::new();
            if let Some(conflicts) = self.conflict_graph.get(&job_id) {
                for &conflict_id in conflicts {
                    if let Some(&color) = coloring.get(&conflict_id) {
                        used_colors.insert(color);
                    }
                }
            }
            
            // Find the lowest available color
            let mut assigned_color = None;
            for color in 0..max_colors {
                if !used_colors.contains(&color) {
                    assigned_color = Some(color);
                    break;
                }
            }
            
            // If no color available (shouldn't happen with max_colors), use overflow
            let color = assigned_color.unwrap_or(max_colors);
            coloring.insert(job_id, color);
            
            trace!("Job {} assigned to batch {}", job_id, color);
        }
        
        coloring
    }
    
    /// Get an optimal execution order within constraints
    pub fn get_execution_order(&self, max_parallel: usize) -> Vec<ExecutionBatch> {
        let mut batches = Vec::new();
        let mut assigned = HashSet::new();
        let mut batch_id = 0;
        
        // Use graph coloring to find independent sets
        let coloring = self.greedy_graph_coloring(max_parallel);
        
        // Group by color, but limit each batch to max_parallel jobs
        let mut color_groups: HashMap<usize, Vec<ComparisonJob>> = HashMap::new();
        for job in &self.jobs {
            if let Some(&color) = coloring.get(&job.comparison_number) {
                color_groups.entry(color).or_insert_with(Vec::new).push(job.clone());
            }
        }
        
        // Create batches from color groups, respecting max_parallel limit
        for (_color, mut jobs) in color_groups {
            // Split large color groups into smaller batches
            while !jobs.is_empty() {
                // Take at most max_parallel jobs for this batch
                let batch_size = std::cmp::min(jobs.len(), max_parallel);
                let batch_jobs: Vec<_> = jobs.drain(..batch_size).collect();
                
                for job in &batch_jobs {
                    assigned.insert(job.comparison_number);
                }
                
                batches.push(ExecutionBatch {
                    batch_id,
                    jobs: batch_jobs,
                    can_run_parallel: true,  // Jobs in same color can run in parallel
                });
                
                batch_id += 1;
            }
        }
        
        // Handle any unassigned jobs (shouldn't happen with proper coloring)
        let unassigned: Vec<_> = self.jobs.iter()
            .filter(|j| !assigned.contains(&j.comparison_number))
            .cloned()
            .collect();
        
        if !unassigned.is_empty() {
            // Split unassigned into batches of max_parallel
            let mut remaining = unassigned;
            while !remaining.is_empty() {
                let batch_size = std::cmp::min(remaining.len(), max_parallel);
                let batch_jobs: Vec<_> = remaining.drain(..batch_size).collect();
                
                batches.push(ExecutionBatch {
                    batch_id,
                    jobs: batch_jobs,
                    can_run_parallel: false,  // Unassigned jobs may have conflicts
                });
                batch_id += 1;
            }
        }
        
        info!("Created {} batches from {} jobs (max {} jobs per batch)", 
            batches.len(), self.jobs.len(), max_parallel);
        
        batches
    }
    
    /// Get jobs that have no conflicts and can run immediately
    pub fn get_independent_jobs(&self) -> Vec<ComparisonJob> {
        self.jobs.iter()
            .filter(|job| {
                self.conflict_graph
                    .get(&job.comparison_number)
                    .map(|conflicts| conflicts.is_empty())
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }
    
    /// Get statistics about the scheduling
    pub fn get_stats(&self) -> SchedulerStats {
        let total_jobs = self.jobs.len();
        let independent_jobs = self.get_independent_jobs().len();
        
        let max_conflicts = self.conflict_graph.values()
            .map(|conflicts| conflicts.len())
            .max()
            .unwrap_or(0);
        
        let total_conflicts: usize = self.conflict_graph.values()
            .map(|conflicts| conflicts.len())
            .sum();
        
        let databases_used = self.database_usage.len();
        
        SchedulerStats {
            total_jobs,
            independent_jobs,
            max_conflicts,
            total_conflicts,
            databases_used,
            strategy: self.strategy.clone(),
        }
    }

    /// Generate a visual representation of the conflict graph (for debugging)
    pub fn visualize_conflicts(&self) -> String {
        let mut result = String::new();
        result.push_str("Conflict Graph:\n");
        
        for job in &self.jobs {
            let conflicts = self.conflict_graph
                .get(&job.comparison_number)
                .map(|c| c.len())
                .unwrap_or(0);
            
            result.push_str(&format!(
                "  Job {} ({} -> {}): {} conflicts\n",
                job.comparison_number,
                job.source_db_path.file_name().unwrap_or_default().to_string_lossy(),
                job.target_db_path.file_name().unwrap_or_default().to_string_lossy(),
                conflicts
            ));
            
            if conflicts > 0 {
                if let Some(conflict_set) = self.conflict_graph.get(&job.comparison_number) {
                    let conflict_list: Vec<String> = conflict_set.iter()
                        .map(|id| id.to_string())
                        .collect();
                    result.push_str(&format!("    Conflicts with: {}\n", conflict_list.join(", ")));
                }
            }
        }
        
        result
    }
    
    /// Generate execution plan summary
    pub fn summarize_execution_plan(&self, max_parallel: usize) -> String {
        let batches = self.get_parallel_batches(max_parallel);
        let stats = self.get_stats();
        
        let mut result = String::new();
        result.push_str(&format!("{}\n", stats));
        result.push_str(&format!("\nExecution Plan ({} batches):\n", batches.len()));
        
        for (i, batch) in batches.iter().enumerate() {
            result.push_str(&format!(
                "  Batch {}: {} jobs in parallel\n",
                i + 1,
                batch.len()
            ));
            
            for job in batch {
                result.push_str(&format!(
                    "    - Job {}: {} -> {}\n",
                    job.comparison_number,
                    job.source_db_path.file_name().unwrap_or_default().to_string_lossy(),
                    job.target_db_path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
        }
        
        result
    }

    /// Check if a job conflicts with any jobs in the given set
    pub fn conflicts_with_any(&self, job_id: usize, other_jobs: &HashSet<usize>) -> bool {
        if let Some(conflicts) = self.conflict_graph.get(&job_id) {
            for other_id in other_jobs {
                if conflicts.contains(other_id) {
                    return true;
                }
            }
        }
        false
    }
    
    /// Get the set of job IDs that conflict with the given job
    pub fn get_conflicts(&self, job_id: usize) -> Option<&HashSet<usize>> {
        self.conflict_graph.get(&job_id)
    }
    
    /// Get the total number of conflicts for a job
    pub fn get_conflict_count(&self, job_id: usize) -> usize {
        self.conflict_graph
            .get(&job_id)
            .map(|conflicts| conflicts.len())
            .unwrap_or(0)
    }
    
    /// Check if a specific job conflicts with another specific job
    pub fn jobs_conflict(&self, job1_id: usize, job2_id: usize) -> bool {
        self.conflict_graph
            .get(&job1_id)
            .map(|conflicts| conflicts.contains(&job2_id))
            .unwrap_or(false)
    }
}

/// A batch of jobs that can run in parallel
#[derive(Debug, Clone)]
pub struct ExecutionBatch {
    pub batch_id: usize,
    pub jobs: Vec<ComparisonJob>,
    pub can_run_parallel: bool,
}

impl ExecutionBatch {
    /// Get the estimated time for this batch (assuming all run in parallel)
    pub fn estimated_time(&self) -> &'static str {
        // This is a placeholder - you could add actual time estimation logic
        match self.jobs.len() {
            0 => "0s",
            1..=5 => "~5min",
            6..=10 => "~10min",
            _ => ">10min",
        }
    }
}

/// Statistics about the scheduling
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_jobs: usize,
    pub independent_jobs: usize,
    pub max_conflicts: usize,
    pub total_conflicts: usize,
    pub databases_used: usize,
    pub strategy: String,
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scheduler Stats ({}): {} jobs, {} independent, {} max conflicts, {} total conflicts, {} databases",
            self.strategy,
            self.total_jobs,
            self.independent_jobs,
            self.max_conflicts,
            self.total_conflicts,
            self.databases_used
        )
    }
}

/// Create batches optimized for minimum total execution time
pub fn optimize_batches(
    jobs: Vec<ComparisonJob>,
    config: &AlNaqlConfig,
) -> Vec<ExecutionBatch> {
    let scheduler = JobScheduler::new(jobs, config);
    scheduler.get_execution_order(config.execution.max_parallel_comparisons)
}

/// Find jobs that can run next given currently running jobs
pub fn find_next_runnable(
    all_jobs: &[ComparisonJob],
    running_jobs: &HashSet<usize>,
    completed_jobs: &HashSet<usize>,
    config: &AlNaqlConfig,
) -> Vec<ComparisonJob> {
    let scheduler = JobScheduler::new(all_jobs.to_vec(), config);
    
    all_jobs.iter()
        .filter(|job| {
            // Skip if already running or completed
            if running_jobs.contains(&job.comparison_number) ||
               completed_jobs.contains(&job.comparison_number) {
                return false;
            }
            
            // Check if conflicts with any running job
            if let Some(conflicts) = scheduler.conflict_graph.get(&job.comparison_number) {
                for running_id in running_jobs {
                    if conflicts.contains(running_id) {
                        return false;
                    }
                }
            }
            
            true
        })
        .cloned()
        .collect()
}