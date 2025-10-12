use ahash::AHashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::fs::{self};
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use std::io::Write;
use tokio::time::Duration;
use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::RwLock;
use log::{info, debug, warn, error, LevelFilter};
use alnaql::config::storage::StorageConfig;
use alnaql::{
    AlNaqlConfig,
    Error,
    Result,
    storage::{
        create_storage_for_generation,
        create_metadata_storage,
        FileMetadata,
        LMDBStorage,
        open_storage, 
        create_storage_for_compaction,
        DatabaseSizeStats,
    },
    utils::{
        logger::Logger,
        boundary_handler::BoundaryHandler,
        mmap::MmapFileHandler,
        resource_coordinator::{
            SHARED_STATE,
            memory_manager::{MemoryManager},
            initialize_resource_management,
        },
        processing::ProcessingManager,
    },
    parser::{TextParser, ArabicParser},
    ngram::NGramGenerator,
    types::NGramEntry,
};


// Add a helper function to derive database path from file path
fn get_db_path_from_file_path(file_path: &str, output_dir: &Path) -> PathBuf {
    let file_name = Path::new(file_path).file_name().unwrap_or_default();
    output_dir.join(file_name).with_extension("db")
}

fn ensure_directories(config: &AlNaqlConfig) -> Result<()> {
    let dirs = [
        config.files.source_dir.to_str().unwrap_or("data/source"),
        config.files.target_dir.to_str().unwrap_or("data/target"),
        config.files.ngrams_source_dir.to_str().unwrap_or("ngrams/source"),
        config.files.ngrams_target_dir.to_str().unwrap_or("ngrams/target"),
        "logs",
        config.files.temp_dir.to_str().unwrap_or("temp"),
    ];

    for dir in dirs {
        fs::create_dir_all(dir)?;
    }

    Ok(())
}

async fn process_file(
    file_path: &Path,
    output_dir: &Path,
    logger: Arc<RwLock<Logger>>,
    config: &AlNaqlConfig,
    progress: &ProgressBar,
) -> Result<FileMetadata> {
    // Start timing and memory tracking
    let start_time = Instant::now();
    let memory_manager = MemoryManager::new();
    memory_manager.log_memory_status();
    
    
    // Log start of processing
    {
        let mut logger = logger.write();
        logger.log(&format!("Starting streaming process for: {:?}", file_path))?;
    }
    info!("Starting streaming process for: {:?}", file_path);

    // Get file size and prepare database path
    let file_size = fs::metadata(file_path)?.len();
    let db_path = output_dir.join(file_path.file_name().unwrap()).with_extension("db");

    if fs::metadata(&db_path).is_ok() {
        info!("Removing existing database at {:?}", db_path);
        fs::remove_dir_all(&db_path)?;
    }
    
    // Create the directory for LMDB
    fs::create_dir_all(&db_path)?;

    // Initialize database
    debug!("Initializing database at {:?} with file size {}", db_path, file_size);
    progress.set_message("Initializing database...");
    let mut db = create_storage_for_generation(
        &db_path, 
        config.storage.clone(), 
        file_size,
        config.generator.prefix_index_depth
    )?;
    // Prepare for bulk loading
    db.prepare_for_bulk_loading()?;

    // Initialize parser 
    let mut parser = ArabicParser::new(config.parser.clone());
    
    // Try to load stop words from the explicit path in config
    if let Some(ref stop_words_path) = config.parser.stop_words_file {
        match parser.load_stop_words(stop_words_path) {
            Ok(_) => {
                info!("Successfully loaded stop words from: {:?}", stop_words_path);
            },
            Err(e) => {
                warn!("Failed to load stop words from config path: {}", e);
                
                // Fallback to a default path if the configured path fails
                let default_path = PathBuf::from("filters/stop_words.txt");
                match parser.load_stop_words(&default_path) {
                    Ok(_) => {
                        info!("Successfully loaded stop words from default path");
                    },
                    Err(e) => warn!("Failed to load stop words from default path: {}", e),
                }
            }
        }
    } else {
        // If no path configured, try the default path
        let default_path = PathBuf::from("filters/stop_words.txt");
        match parser.load_stop_words(&default_path) {
            Ok(_) => {
                info!("Successfully loaded stop words from default path");
            },
            Err(e) => warn!("Failed to load stop words from default path: {}", e),
        }
    }
    
    // Create the generator with the configured parser
    let mut generator = NGramGenerator::new(parser, config.generator.clone());
    generator.set_file_path(file_path);
    generator.start_document_frequency_tracking();

        // Check if we are starting with memory constraints
    let initial_memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
    if initial_memory_pressure > 50 {
        warn!("Starting with constrained memory (pressure: {}%) - processing may be slower", initial_memory_pressure);
    }

    // Calculate optimal chunk size based on file size and memory pressure
    let chunk_size = 20 * 1024 * 1024;

    // Initialize memory-mapped file processing with optimal chunk size
    let mmap_handler = MmapFileHandler::new(file_path, chunk_size, 100)?;
    
    // Accumulate ALL prefix data across ALL chunks
    let mut all_prefix_data: AHashMap<Vec<u8>, Vec<u64>> = AHashMap::new();

    // Initialize boundary handler for consistent text streaming
    let mut boundary_handler = BoundaryHandler::new(config.generator.ngram_type);

    // Progress tracking variables
    let mut total_ngrams = 0;
    let mut total_chunks_processed = 0;
    let mut total_bytes_processed = 0;
    let estimated_chunks = (file_size as usize + chunk_size - 1) / chunk_size;
    
    // Update progress bar
    progress.set_length(estimated_chunks as u64);
    progress.set_message(format!("Processing {} chunks of {} bytes each", estimated_chunks, chunk_size));

    // Storage batch size - adjust based on memory availability
    // Batch size is the number of ngrams to accumulate before writing to database
    let base_batch_size = 5000;  // Standard batch size for normal conditions
    
    // Adjust based on memory pressure from SHARED_STATE
    let memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
    let batch_size = match memory_pressure {
        0..=25 => (base_batch_size * 2).min(10000),     // Low: up to 2x, max 5000 ngrams
        26..=50 => base_batch_size,                     // Normal: use base (2000 ngrams)
        51..=75 => (base_batch_size / 3).max(1000),     // High: divide by 3, min 500 ngrams
        _ => (base_batch_size / 4).max(500),            // Critical: quarter, min 250 ngrams
    };
    
    info!("Initial batch size: {} ngrams (memory pressure: {}%)", batch_size, memory_pressure);


    // Get initial max sequence from database
    let mut current_max_sequence = db.get_max_sequence()?
        .unwrap_or(0);  // If database is empty, start at 0
    info!("Starting with sequence number: {}", current_max_sequence + 1);

    // Track last cleanup time
    let mut last_cleanup_time = Instant::now();
    const MIN_CLEANUP_INTERVAL: Duration = Duration::from_secs(5);

    // Get the raw file data for boundary detection
    let file_data = mmap_handler.get_content();
    let file_size = mmap_handler.get_size();

    // Track actual position in file
    let mut file_position = 0;

    // Process the file in clean chunks
    while file_position < file_size {
        total_chunks_processed += 1;
        
        // Update progress
        progress.set_position(total_chunks_processed as u64);
        progress.set_message(format!("Processing chunk {}/{} ({:.1}% complete)", 
            total_chunks_processed, 
            estimated_chunks,
            (file_position as f64 / file_size as f64) * 100.0
        ));
        
        // Find safe boundary for this chunk
        let chunk_end = boundary_handler.find_chunk_boundary(
            file_data,
            file_position,
            chunk_size
        )?;
        
        // Extract chunk data (guaranteed valid UTF-8 with complete words)
        let chunk_bytes = &file_data[file_position..chunk_end];
        let chunk_text = std::str::from_utf8(chunk_bytes)?;
        
        debug!("Processing chunk {} at position {}, size {} bytes", 
            total_chunks_processed, file_position, chunk_bytes.len());
        
        // Set the next sequence number for the generator
        generator.set_next_sequence_number(current_max_sequence + 1);
        
        // Generate ngrams with CORRECT absolute position and NO overlaps
        let generation_result = generator.generate_ngrams_with_position(
            chunk_text, 
            file_position,  // Actual file position!
            0  // No overlap bytes!
        )?;
        
        let mut chunk_ngrams = generation_result.ngrams;
        
        info!("Generated {} ngrams from chunk {}", chunk_ngrams.len(), total_chunks_processed);
        
        if !chunk_ngrams.is_empty() {
            // Update the max sequence number
            current_max_sequence = chunk_ngrams.iter()
                .map(|ng| ng.sequence_number)
                .max()
                .unwrap_or(current_max_sequence);
            debug!("Updated max sequence to: {}", current_max_sequence);
            
            // Store ngrams and collect prefix data
            let chunk_prefix_data = store_ngram_batch(&mut chunk_ngrams, &mut db, batch_size).await?;
            total_ngrams += chunk_ngrams.len();
            
            // Merge prefix data
            for (prefix_key, sequences) in chunk_prefix_data {
                all_prefix_data
                    .entry(prefix_key)
                    .or_insert_with(Vec::new)
                    .extend(sequences);
            }
        }
        
        // Update position to the actual end of processed chunk
        total_bytes_processed += chunk_end - file_position;
        file_position = chunk_end;
        info!("Processed {} total bytes across {} chunks", 
            total_bytes_processed, total_chunks_processed);
        
        // Break if we've reached the end
        if file_position >= file_size {
            info!("Reached end of file");
            break;
        }
        
        // Check memory status and perform cleanup if necessary
        let current_memory_pressure = SHARED_STATE.memory_pressure.load(Ordering::Relaxed);
        let time_since_cleanup = last_cleanup_time.elapsed();
        
        // Check if we should force cleanup (high pressure and enough time passed)
        if current_memory_pressure > 85 && time_since_cleanup >= MIN_CLEANUP_INTERVAL {
            debug!("Forcing memory cleanup after chunk {} (pressure: {}%)", 
                    total_chunks_processed, current_memory_pressure);
            
            let memory_state = memory_manager.force_cleanup().await;
            last_cleanup_time = Instant::now();  // Update cleanup time
            
            // Log the memory state after cleanup
            info!("After cleanup: memory state is {}", memory_state);

        }
        
        // Check for memory degradation periodically
        if total_chunks_processed % 10 == 0 {
            memory_manager.check_memory_degradation();
            memory_manager.log_memory_status();
        }
    }

    // Extract frequency data for this document
    if let Some(frequencies) = generator.take_document_frequencies() {
        info!("Document frequency analysis: {} unique ngrams found", frequencies.len());
        
        // STEP 2: Calculate percentiles from frequencies
        let percentiles = calculate_percentiles(frequencies);
        info!("Calculated percentiles for {} unique ngrams", percentiles.len());
        
        // Log some statistics about the distribution
        if !percentiles.is_empty() {
            let high_percentile_count = percentiles.values()
                .filter(|&&p| p >= 95)
                .count();
            let mid_percentile_count = percentiles.values()
                .filter(|&&p| p >= 50 && p < 95)
                .count();
            
            debug!("Percentile distribution: {} ngrams (≥95th), {} ngrams (50-94th)", 
                high_percentile_count, mid_percentile_count);
        }
        
        // STEP 3: Store percentiles to LMDB
        db.store_percentiles(percentiles)?;
        info!("Successfully stored percentiles to database");
    }

    // NOW store ALL accumulated prefix data at once
    if !all_prefix_data.is_empty() {
        info!("Storing ALL accumulated prefix data");
        info!("Total unique prefix: {}", all_prefix_data.len());
        
        // Deduplicate sequences in each prefix group before storing
        for (_, sequences) in all_prefix_data.iter_mut() {
            sequences.sort_unstable();
            sequences.dedup();
        }
        
        // Store all prefix data in a single operation
        db.store_prefix_data(all_prefix_data)?;
        info!("Successfully stored all prefix data");
    }
    
    // Finalize database
    info!("Finalizing database with {} total ngrams", total_ngrams);
    db.verify_prefix_storage()?;
    
    // Final memory status report
    memory_manager.log_memory_status();
    
    // Process has completed, update progress and log results
    let elapsed = start_time.elapsed();
    let completion_msg = format!(
        "Processing complete: {} ngrams in {:.2} seconds ({:.2} ngrams/sec)", 
        total_ngrams, 
        elapsed.as_secs_f64(),
        total_ngrams as f64 / elapsed.as_secs_f64()
    );
    
    {
        let mut logger = logger.write();
        logger.log(&completion_msg)?;
    }
    info!("{}", completion_msg);
    
    progress.finish_with_message(format!("Completed: {} ngrams generated", total_ngrams));

    // Finalize and close the database
    info!("Finalizing database at {:?}", db_path);
    db.finish_bulk_loading()?;

    // ============================================================================
    // DATABASE COMPACTION (if enabled)
    // ============================================================================
    
    // Check if compaction is enabled in configuration
    if config.storage.enable_post_generation_compaction {
        info!("Checking if database compaction would be beneficial...");
        
        // Get comprehensive size statistics
        match db.get_size_statistics() {
            Ok(stats) => {
                let current_size = stats.page_stats.map_size;
                let optimal_size = stats.optimal_compact_size as u64;
                let space_saved = current_size.saturating_sub(optimal_size);
                let percent_saved = (space_saved as f64 / current_size as f64) * 100.0;
                
                info!("Current database size: {} MB", current_size / (1024 * 1024));
                info!("Optimal compact size: {} MB", optimal_size / (1024 * 1024));
                info!("Potential space savings: {} MB ({:.1}%)", 
                    space_saved / (1024 * 1024), percent_saved);
                
                // Check if compaction is worthwhile
                if config.storage.is_compaction_worthwhile(current_size, optimal_size) {
                    info!("Compaction will save significant space. Starting compaction process...");
                    
                    // Close the current database before compaction
                    info!("Closing current database for compaction...");
                    db.close()?;
                    
                    // Perform compaction (using prefix_index_depth from config)
                    match perform_database_compaction(
                        &db_path, 
                        &config.storage, 
                        stats,
                        config.generator.prefix_index_depth,  // Get it from config!
                    ) {
                        Ok(final_size) => {
                            let actual_saved = current_size - final_size;
                            info!("✓ Compaction successful!");
                            info!("  Original size: {} MB", current_size / (1024 * 1024));
                            info!("  Compacted size: {} MB", final_size / (1024 * 1024));
                            info!("  Space saved: {} MB ({:.1}%)", 
                                actual_saved / (1024 * 1024),
                                (actual_saved as f64 / current_size as f64) * 100.0);
                            
                            // Re-open the compacted database
                            db = open_storage(&db_path, config.storage.clone())?;
                        },
                        Err(e) => {
                            error!("Compaction failed: {}", e);
                            // Try to re-open the original database
                            db = open_storage(&db_path, config.storage.clone())?;
                        }
                    }
                } else {
                    info!("Compaction would not save significant space. Skipping.");
                }
            },
            Err(e) => {
                warn!("Failed to get database statistics: {}. Skipping compaction.", e);
            }
        }
    } else {
        debug!("Database compaction is disabled in configuration");
    }
    db.close()?;

    Ok(FileMetadata {
        file_path: file_path.to_string_lossy().to_string(),
        total_ngrams: total_ngrams as u64,
        file_size: file_size as u64,
        processed_at: SystemTime::now(),
    })
}

// Helper function to store a batch of ngrams and RETURN their prefix data (not store it)
async fn store_ngram_batch(
    ngrams: &mut Vec<NGramEntry>,
    db: &mut LMDBStorage,
    batch_size: usize,
) -> Result<AHashMap<Vec<u8>, Vec<u64>>> {
    if ngrams.is_empty() {
        return Ok(AHashMap::new());
    }
    
    // Accumulate all prefix data across sub-batches within this chunk
    let mut accumulated_prefix_data: AHashMap<Vec<u8>, Vec<u64>> = AHashMap::new();
    
    // Process in smaller sub-batches to reduce memory pressure
    for chunk in ngrams.chunks(batch_size.min(5000)) {
        // Store ngrams
        db.store_ngrams_batch(chunk)?;
        
        // Collect prefix data for this chunk
        let chunk_prefix_data = db.collect_prefix_data(chunk)?;
        
        // Merge this chunk's prefix data into the accumulator
        for (prefix_key, sequences) in chunk_prefix_data {
            accumulated_prefix_data
                .entry(prefix_key)
                .or_insert_with(Vec::new)
                .extend(sequences);
        }
        
        // Small delay to allow memory cleanup
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // RETURN the accumulated prefix data - DON'T STORE IT HERE
    Ok(accumulated_prefix_data)
}
/// Calculate percentile rankings for each ngram based on frequency
/// Returns a map of hash -> percentile (0-100)
fn calculate_percentiles(frequencies: AHashMap<u64, u32>) -> AHashMap<u64, u8> {
    if frequencies.is_empty() {
        return AHashMap::new();
    }
    
    // Create a vector of (hash, count) pairs and sort by count
    let mut freq_vec: Vec<(u64, u32)> = frequencies.into_iter().collect();
    freq_vec.sort_by_key(|&(_, count)| count);
    
    let total = freq_vec.len() as f64;
    let mut percentiles = AHashMap::new();
    
    for (index, (hash, count)) in freq_vec.into_iter().enumerate() {
        // Calculate percentile: what percentage of ngrams have lower or equal frequency
        let percentile = ((index + 1) as f64 / total * 100.0) as u8;
        percentiles.insert(hash, percentile);
        
        // Debug log for highly frequent ngrams
        if percentile >= 99 {
            debug!("Hash {} with count {} is in {}th percentile", 
                hash, count, percentile);
        }
    }
    
    percentiles
}

/// Perform database compaction to reduce storage size
fn perform_database_compaction(
    db_path: &Path,
    config: &StorageConfig,
    stats: DatabaseSizeStats,
    prefix_index_depth: usize,
) -> Result<u64> {
    use std::time::Instant;
    
    // Calculate optimal size with configured buffer
    let optimal_size = stats.optimal_compact_size;
    
    // Create path for compact database
    let compact_path = db_path.parent()
        .ok_or_else(|| Error::Config("Invalid database path".to_string()))?
        .join(format!("{}.compact", 
            db_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("database")));
    
    info!("Creating compact database at {:?} with size {} MB", 
        compact_path, optimal_size / (1024 * 1024));
    
    // Create progress bar for compaction
    let progress = ProgressBar::new(stats.total_entries);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} records ({percent}%) {msg}")
            .unwrap()
            .progress_chars("=>-")
    );
    progress.set_message("Compacting database...");
    
    // Open source database for reading
    let mut source_db = open_storage(db_path, config.clone())?;
    
    // Create compact database with optimal size  
    let mut compact_db = create_storage_for_compaction(
        &compact_path,
        config.clone(),
        optimal_size,
        prefix_index_depth,
    )?;
    
    let start_time = Instant::now();
    
    // Copy all data to compact database
    source_db.copy_all_data_to(&mut compact_db)?;
    
    // Update progress
    progress.finish_with_message(format!("Compaction complete in {:?}", start_time.elapsed()));
    
    // Get actual size of compact database
    let compact_metadata = fs::metadata(compact_path.join("data.mdb"))
        .map_err(|e| Error::storage(format!("Failed to get compact database metadata: {}", e)))?;
    let compact_size = compact_metadata.len();
    
    // Close both databases before replacement
    info!("Closing databases for atomic replacement...");
    source_db.close()?;
    compact_db.close()?;
    
    // Perform atomic replacement
    if config.keep_original_on_compaction {
        // Keep original with .original suffix
        let original_path = db_path.parent()
            .ok_or_else(|| Error::Config("Invalid database path".to_string()))?
            .join(format!("{}.original",
                db_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("database")));
        
        info!("Keeping original database at {:?}", original_path);
        fs::rename(db_path, original_path)
            .map_err(|e| Error::storage(format!("Failed to move original database: {}", e)))?;
    } else {
        // Delete original database
        info!("Removing original database...");
        fs::remove_dir_all(db_path)
            .map_err(|e| Error::storage(format!("Failed to remove original database: {}", e)))?;
    }
    
    // Rename compact to original name
    info!("Moving compact database to final location...");
    fs::rename(compact_path, db_path)
        .map_err(|e| Error::storage(format!("Failed to move compact database: {}", e)))?;
    
    Ok(compact_size)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration first
    let config = AlNaqlConfig::from_ini("default.ini")?;
    
    // Set up logging infrastructure
    std::fs::create_dir_all("logs")?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(format!("logs/ngram_generation_{}.log", timestamp))?;

    // Parse log level from debug config
    let log_level = match config.debug.ngram_log.to_lowercase().as_str() {
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        "none" => LevelFilter::Off,
        _ => {
            println!("Invalid log level '{}', defaulting to Info", config.debug.ngram_log);
            LevelFilter::Info
        }
    };

    // Initialize logging system with configuration
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log_level)  // Use the parsed log_level
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    info!("Starting ngram generation with log level: {:?}", log_level);

    // Create required directory structure
    ensure_directories(&config)?;

    // Initialize coordinated resource management system
    initialize_resource_management()?;
    
    // Initialize progress tracking
    let m = MultiProgress::new();
    let processing_manager = ProcessingManager::new(&config);
    
    // Process source directory files
    info!("Starting source directory processing...");
    let source_metadata = processing_manager.process_directory(
        &config.files.source_dir,
        &config.files.ngrams_source_dir,
        &config,
        &m,
        |entry, output_dir, config, m| {
            let output_dir = output_dir.to_path_buf();
            let config = config.clone();
            let m = m.clone();
            
            async move {
                let file_size = entry.metadata()?.len();
                let progress = m.add(ProgressBar::new(file_size));
                progress.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .unwrap());

                let logger = Arc::new(RwLock::new(Logger::with_file(
                    config.files.temp_dir.join(format!("processing_{}.log",
                        entry.path().file_name().unwrap().to_string_lossy())))?
                ));

                process_file(
                    &entry.path(),
                    &output_dir,
                    logger,
                    &config,
                    &progress
                ).await
            }
        }
    ).await?;
        
    // Create and store metadata in a dedicated source metadata database
    info!("Storing source metadata...");
    let source_db_path = config.files.ngrams_source_dir.join("metadata.db");
    // Remove existing database if it exists to avoid conflicts
    if fs::metadata(&source_db_path).is_ok() {
        info!("Removing existing metadata database at {:?}", source_db_path);
        fs::remove_dir_all(&source_db_path)?;
    }

    // Create the directory for the metadata database
    fs::create_dir_all(&source_db_path)?;
    info!("Creating new metadata database at {:?}", source_db_path);

    // Create a new metadata database - we don't need file_size for metadata
    let mut source_db = create_metadata_storage(&source_db_path, config.storage.clone())?;
    // Store the metadata
    source_db.store_metadata(&source_metadata, config.generator.prefix_index_depth)?;
    // Properly close the database
    info!("Closing source metadata database");
    source_db.close()?;

    // Process target directory files
    info!("Starting target directory processing...");
    let target_metadata = processing_manager.process_directory(
        &config.files.target_dir,
        &config.files.ngrams_target_dir,
        &config,
        &m,
        |entry, output_dir, config, m| {
            let output_dir = output_dir.to_path_buf();
            let config = config.clone();
            let m = m.clone();
            
            async move {
                let file_size = entry.metadata()?.len();
                let progress = m.add(ProgressBar::new(file_size));
                progress.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .unwrap());

                let logger = Arc::new(RwLock::new(Logger::with_file(
                    config.files.temp_dir.join(format!("processing_{}.log",
                        entry.path().file_name().unwrap().to_string_lossy())))?
                ));

                process_file(
                    &entry.path(),
                    &output_dir,
                    logger,
                    &config,
                    &progress
                ).await
            }
        }
    ).await?;

    // Create and store metadata in a dedicated target metadata database
    info!("Storing target metadata...");
    let target_db_path = config.files.ngrams_target_dir.join("metadata.db");

    // Remove existing database if it exists
    if fs::metadata(&target_db_path).is_ok() {
        info!("Removing existing metadata database at {:?}", target_db_path);
        fs::remove_dir_all(&target_db_path)?;
    }

    // Create the directory for the metadata database  
    fs::create_dir_all(&target_db_path)?;
    info!("Creating new metadata database at {:?}", target_db_path);
    // Create a new metadata database
    let mut target_db = create_metadata_storage(&target_db_path, config.storage.clone())?;
    // Store the metadata
    target_db.store_metadata(&target_metadata, config.generator.prefix_index_depth)?;
    // Properly close the database
    info!("Closing target metadata database");
    target_db.close()?;

    // Finalize database loading
    let mut all_db_paths = Vec::new();
    
    // Add metadata databases to the paths collection
    all_db_paths.push(source_db_path.clone());
    all_db_paths.push(target_db_path.clone());

    // Process source database paths
    for metadata in &source_metadata {
        let db_path = get_db_path_from_file_path(&metadata.file_path, &config.files.ngrams_source_dir);
        all_db_paths.push(db_path.clone());
    }

    // Process target database paths
    for metadata in &target_metadata {
        let db_path = get_db_path_from_file_path(&metadata.file_path, &config.files.ngrams_target_dir);
        all_db_paths.push(db_path.clone());
    }
    
    // Calculate and display final statistics
    let total_files = source_metadata.len() + target_metadata.len();
    let total_ngrams: u64 = source_metadata.iter().map(|m| m.total_ngrams).sum::<u64>() +
                           target_metadata.iter().map(|m| m.total_ngrams).sum::<u64>();
    let total_size: u64 = source_metadata.iter().map(|m| m.file_size).sum::<u64>() +
                         target_metadata.iter().map(|m| m.file_size).sum::<u64>();

    // Verify all environments are properly closed
    info!("Verifying all environments are closed properly...");
    let mut unclosed_count = 0;
    for path in &all_db_paths {
        if !alnaql::storage::verify_storage_closed(path) {
            warn!("Environment at {:?} may not be fully closed", path);
            unclosed_count += 1;
            
            // Try one more time to force closure
            if let Err(e) = alnaql::storage::LMDBStorage::remove_from_cache(path) {
                warn!("Error forcing closure of database at {:?}: {}", path, e);
            }
        }
    }

    if unclosed_count > 0 {
        warn!("{} environments may not have closed properly", unclosed_count);
    } else {
        info!("All environments closed successfully");
    }
    // Log summary information
    info!("\nProcessing Summary:");
    info!("Total files processed: {}", total_files);
    info!("Total ngrams generated: {}", total_ngrams);
    info!("Total data processed: {} bytes", total_size);

    info!("\nProcessing complete!");
    Ok(())
}