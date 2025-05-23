use std::path::{Path, PathBuf};
use std::fs::{self};
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use std::io::Write;
use tokio::time::Duration;
use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::RwLock;
use log::{info, debug, warn};
use kuttab::config::subsystems::generator::NGramType;
use kuttab::{
    KuttabConfig,
    Result,
    storage::{
        get_or_create_storage,
        FileMetadata,
        lmdb::LMDBStorage,
        lmdb::DB_CACHE,
    },
    utils::{
        logger::Logger,
        boundary_handler::BoundaryHandler,
        mmap::MmapFileHandler,
        memory_manager::{MemoryManager, MemoryState},
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

fn ensure_directories(config: &KuttabConfig) -> Result<()> {
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
    config: &KuttabConfig,
    progress: &ProgressBar,
) -> Result<FileMetadata> {
    // Start timing and memory tracking
    let start_time = Instant::now();
    let memory_manager = MemoryManager::new(1000, 10000);
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
    
    // Calculate optimal chunk size based on available memory and initial state
    let mut chunk_size = memory_manager.calculate_optimal_chunk_size();
    info!("Using chunk size of {} bytes", chunk_size);
    
    // Check if we are starting with memory constraints
    if memory_manager.is_memory_constrained() {
        warn!("Starting with constrained memory - processing may be slower");
    }
    if fs::metadata(&db_path).is_ok() {
        info!("Removing existing database at {:?}", db_path);
        fs::remove_dir_all(&db_path)?;
    }

    // Initialize database
    debug!("Initializing database at {:?} with file size {}", db_path, file_size);
    progress.set_message("Initializing database...");
    let mut db = get_or_create_storage(&db_path, config.storage.clone(), Some(file_size))?;
    // Prepare for bulk loading
    if let Some(db_ref) = DB_CACHE.get(&db_path) {
        db_ref.value().lock().prepare_bulk_load()?;
    }

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

    // Initialize memory-mapped file processing with optimal chunk size
    let mmap_handler = MmapFileHandler::new(file_path, chunk_size, 100)?;
    
    // Initialize boundary handler for consistent text streaming
    let overlap_size = match config.generator.ngram_type {
        NGramType::Character => config.generator.ngram_size * 8,
        NGramType::Word => config.generator.ngram_size * 50,
    };
    // Pass ngram_type to the boundary handler
    let mut boundary_handler = BoundaryHandler::new(overlap_size, config.generator.ngram_type);   
    // Progress tracking variables
    let mut total_ngrams = 0;
    let mut total_chunks_processed = 0;
    let mut total_bytes_processed = 0;
    let estimated_chunks = (file_size as usize + chunk_size - 1) / chunk_size;
    
    // Update progress bar
    progress.set_length(estimated_chunks as u64);
    progress.set_message(format!("Processing {} chunks of {} bytes each", estimated_chunks, chunk_size));

    // Storage batch size - adjust based on memory availability
    let mut batch_size = memory_manager.get_batch_size();
    info!("Initial batch size: {}", batch_size);

    // Get initial max sequence from database
    let mut current_max_sequence = db.get_max_sequence_number()?;
    info!("Starting with sequence number: {}", current_max_sequence + 1);

    // Initialize a set to track all processed positions across chunks
    let mut processed_positions = std::collections::HashSet::new();
    debug!("Initialized position tracking for overlap region handling");

    // Process the file in chunks with positions
    for (chunk, chunk_position) in mmap_handler.chunks_with_position(chunk_size) {
        total_chunks_processed += 1;
        total_bytes_processed += chunk.len();
        
        // Update progress
        progress.set_position(total_chunks_processed as u64);
        progress.set_message(format!("Processing chunk {}/{} ({:.1}% complete)", 
            total_chunks_processed, 
            estimated_chunks,
            (total_bytes_processed as f64 / file_size as f64) * 100.0
        ));
        
        // Set the next sequence number for the generator based on previous chunks
        generator.set_next_sequence_number(current_max_sequence + 1);
        
        // Set processed positions from previous chunks - use clone() to avoid ownership issues
        generator.set_processed_positions(processed_positions.clone());
        
        let chunk_info = format!("Processing chunk {} at position {}, length: {}", 
            total_chunks_processed, chunk_position, chunk.len());
        debug!("{}", chunk_info);

        // Process chunk with absolute position
        let chunk_result = boundary_handler.process_chunk(chunk, chunk_position);
        debug!("Chunk processing result: {}", if chunk_result.is_some() { "Some" } else { "None" });

        // Use updated pattern matching for the enhanced return type
        if let Some((text, text_absolute_position, overlap_bytes)) = chunk_result {
            // Log the absolute position information for debugging
            debug!("POSITION_DEBUG: Processing text at absolute position {} (length: {}, overlap: {})", 
                text_absolute_position, text.len(), overlap_bytes);
            debug!("POSITION_DEBUG: Text sample: \"{}\"", 
                text.chars().take(50).collect::<String>().replace('\n', "\\n"));
                
            // Pass the absolute position and overlap bytes to the generator
            let generation_result = generator.generate_ngrams_with_position(
                &text, 
                text_absolute_position, 
                overlap_bytes,
                &mut db
            )?;
            
            let mut chunk_ngrams = generation_result.ngrams;
            
            // Update processed positions for next chunk
            processed_positions = generation_result.positions;
            
            info!("Generated {} ngrams from chunk {}", chunk_ngrams.len(), total_chunks_processed);
            
            if !chunk_ngrams.is_empty() {
                // Update the max sequence number from the newly generated ngrams
                current_max_sequence = chunk_ngrams.iter()
                    .map(|ng| ng.sequence_number)
                    .max()
                    .unwrap_or(current_max_sequence);
                debug!("Updated max sequence to: {}", current_max_sequence);
                
                // Process in batch to avoid memory pressure
                store_ngram_batch(&mut chunk_ngrams, &mut db, batch_size, &logger).await?;
                total_ngrams += chunk_ngrams.len();
                
                // Re-evaluate batch size after processing
                if memory_manager.adjust_batch_size() {
                    let new_batch_size = memory_manager.get_batch_size();
                    info!("Adjusted batch size: {} → {}", batch_size, new_batch_size);
                    batch_size = new_batch_size;
                }
            }
        }
        
        // Check memory status and perform cleanup if necessary
        if memory_manager.should_force_cleanup() {
            debug!("Forcing memory cleanup after chunk {}", total_chunks_processed);
            let memory_state = memory_manager.force_cleanup().await;
            
            // Log the memory state after cleanup
            info!("After cleanup: memory state is {}", memory_state);
            
            // If memory state is still high, take more drastic measures
            if memory_state == MemoryState::High {
                warn!("Memory still high after cleanup, reducing chunk size");
                // Reduce chunk size for future iterations
                let new_chunk_size = (chunk_size * 2/3).max(1024 * 1024); // min 1MB
                if new_chunk_size < chunk_size {
                    info!("Reducing chunk size: {} → {} bytes", chunk_size, new_chunk_size);
                    chunk_size = new_chunk_size;
                }
            }
        }
        
        // Check for memory degradation periodically
        if total_chunks_processed % 10 == 0 {
            memory_manager.check_memory_degradation();
            memory_manager.log_memory_status();
        }
    }
    
    // Process any final text in the boundary handler
    let final_overlap = boundary_handler.get_overlap();
    if !final_overlap.is_empty() {
        // Get the absolute position of the overlap
        let overlap_position = boundary_handler.get_overlap_position();
        
        debug!("Processing final overlap text: {} bytes at absolute position {}", 
            final_overlap.len(), overlap_position);
        
        // Set the next sequence number for final processing
        generator.set_next_sequence_number(current_max_sequence + 1);
        
        // Set processed positions for final chunk - use clone() to avoid ownership issues
        generator.set_processed_positions(processed_positions.clone());
        
        // Add special handling for potentially small final overlaps
        if final_overlap.len() < config.generator.ngram_size * 2 {
            debug!("Warning: Final overlap is smaller than needed ({}), may cause issues", 
                final_overlap.len());
            
            // For small final overlaps, we can either:
            // 1. Skip processing it (safe option)
            if final_overlap.len() < config.generator.ngram_size {
                info!("Final overlap too small to process (need at least {} bytes), skipping", 
                    config.generator.ngram_size);
            } else {
                // 2. Try processing it anyway if it's at least large enough for one ngram
                debug!("Attempting to process small final overlap");
                match generator.generate_ngrams_with_position(final_overlap, overlap_position, 0, &mut db) {
                    Ok(result) => {
                        let final_ngrams = result.ngrams;
                        
                        if !final_ngrams.is_empty() {
                            info!("Successfully generated {} ngrams from final overlap", final_ngrams.len());
                            
                            // Update max sequence number one last time
                            current_max_sequence = final_ngrams.iter()
                                .map(|ng| ng.sequence_number)
                                .max()
                                .unwrap_or(current_max_sequence);
                                debug!("Final max sequence: {}", current_max_sequence);
                            
                            // Store these final ngrams
                            let mut final_batch = final_ngrams;
                            store_ngram_batch(&mut final_batch, &mut db, batch_size, &logger).await?;
                            total_ngrams += final_batch.len();
                        }
                    },
                    Err(e) => {
                        warn!("Failed to process final overlap: {}, skipping", e);
                    }
                }
            }
        } else {
            // Normal processing for sufficiently large final overlap
            match generator.generate_ngrams_with_position(final_overlap, overlap_position, 0, &mut db) {                
                Ok(result) => {
                    let final_ngrams = result.ngrams;
                    
                    if !final_ngrams.is_empty() {
                        info!("Generated {} final ngrams from overlap buffer", final_ngrams.len());
                        
                        // Update max sequence number one last time
                        current_max_sequence = final_ngrams.iter()
                            .map(|ng| ng.sequence_number)
                            .max()
                            .unwrap_or(current_max_sequence);
                            debug!("Final max sequence: {}", current_max_sequence);
                        
                        // Store these final ngrams
                        let mut final_batch = final_ngrams;
                        store_ngram_batch(&mut final_batch, &mut db, batch_size, &logger).await?;
                        total_ngrams += final_batch.len();
                    }
                },
                Err(e) => {
                    warn!("Error processing final overlap: {}. Continuing without it.", e);
                }
            }
        }
    }

    // Finalize database
    info!("Finalizing database with {} total ngrams", total_ngrams);
    db.validate_prefix_index(total_ngrams)?;
    
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

    // Close the database to ensure resources are released
    info!("Closing database at {:?}", db_path);
    db.close()?;
    
    // Force memory cleanup
    std::mem::drop(db);

    Ok(FileMetadata {
        file_path: file_path.to_string_lossy().to_string(),
        total_ngrams: total_ngrams as u64,
        file_size,
        processed_at: SystemTime::now(),
    })
}

// Helper function to store a batch of ngrams and their prefix data
async fn store_ngram_batch(
    ngrams: &mut Vec<NGramEntry>,
    db: &mut LMDBStorage,
    batch_size: usize,
    logger: &Arc<RwLock<Logger>>
) -> Result<()> {
    if ngrams.is_empty() {
        return Ok(());
    }
       
    // Process in smaller sub-batches to reduce memory pressure
    for chunk in ngrams.chunks(batch_size.min(5000)) {
        let msg = format!("Storing sub-batch of {} ngrams", chunk.len());
        {
            let mut logger_guard = logger.write();
            logger_guard.log(&msg)?;
        }
        debug!("{}", msg);
        
        // Store ngrams
        db.store_ngrams_batch(chunk)?;
        
        // Immediately process prefix data for this chunk
        let prefix_data = db.collect_prefix_data(chunk)?;
        if !prefix_data.is_empty() {
            db.store_prefix_data(prefix_data)?;
        }
        
        // Small delay to allow memory cleanup
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration first
    let config = KuttabConfig::from_ini("default.ini")?;
    
    // Set up logging infrastructure
    std::fs::create_dir_all("logs")?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(format!("logs/ngram_generation_{}.log", timestamp))?;

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
        .filter(None, config.generator.get_log_level())
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    info!("Starting ngram generation with log level: {:?}", config.generator.get_log_level());

    // Create required directory structure
    ensure_directories(&config)?;
    
    // Initialize progress tracking
    let m = MultiProgress::new();
    let processing_manager = ProcessingManager::new(&config.processor);
    
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
    // Create a new metadata database - we don't need file_size for metadata
    let mut source_db = get_or_create_storage(&source_db_path, config.storage.clone(), None)?;
    // Store the metadata
    source_db.store_metadata(&source_metadata)?;

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
    // Create a new metadata database
    let mut target_db = get_or_create_storage(&target_db_path, config.storage.clone(), None)?;
    // Store the metadata
    target_db.store_metadata(&target_metadata)?;

    // Finalize database loading
    let mut all_db_paths = Vec::new();
    
    // Add metadata databases to the paths collection
    all_db_paths.push(source_db_path.clone());
    all_db_paths.push(target_db_path.clone());

    // Process source database paths
    for metadata in &source_metadata {
        let db_path = get_db_path_from_file_path(&metadata.file_path, &config.files.ngrams_source_dir);
        all_db_paths.push(db_path.clone());
        if let Some(db) = DB_CACHE.get(&db_path) {
            let mut db_ref = db.value().lock();
            debug!("Finishing bulk load for source database: {:?}", db_path);
            db_ref.finish_bulk_load()?;
        }
    }

    // Process target database paths
    for metadata in &target_metadata {
        let db_path = get_db_path_from_file_path(&metadata.file_path, &config.files.ngrams_target_dir);
        all_db_paths.push(db_path.clone());
        if let Some(db) = DB_CACHE.get(&db_path) {
            let mut db_ref = db.value().lock();
            debug!("Finishing bulk load for target database: {:?}", db_path);
            db_ref.finish_bulk_load()?;
        }
    }
    
    // Ensure the metadata databases are properly finalized too
    for metadata_db_path in [&source_db_path, &target_db_path] {
        if let Some(db) = DB_CACHE.get(metadata_db_path) {
            let mut db_ref = db.value().lock();
            debug!("Finishing metadata database: {:?}", metadata_db_path);
            // No need to call finish_bulk_load since metadata databases aren't used in bulk mode
            // But we should ensure they're properly synced
            db_ref.cleanup()?;
        }
    }
    
    // Display complete system state
    info!("\nDisplaying complete system state:");
    // Access the first database path (we know we have at least one)
    if let Some(first_path) = all_db_paths.first() {
        if let Some(db_ref) = DB_CACHE.get(first_path) {
            let db = db_ref.value().lock();
            if let Err(e) = db.display_complete_system_state(&all_db_paths) {
                warn!("Error displaying system state: {}", e);
            }
        }
    }

    // Calculate and display final statistics
    let total_files = source_metadata.len() + target_metadata.len();
    let total_ngrams: u64 = source_metadata.iter().map(|m| m.total_ngrams).sum::<u64>() +
                           target_metadata.iter().map(|m| m.total_ngrams).sum::<u64>();
    let total_size: u64 = source_metadata.iter().map(|m| m.file_size).sum::<u64>() +
                         target_metadata.iter().map(|m| m.file_size).sum::<u64>();

    // Log summary information
    info!("\nProcessing Summary:");
    info!("Total files processed: {}", total_files);
    info!("Total ngrams generated: {}", total_ngrams);
    info!("Total data processed: {} bytes", total_size);

    info!("\nProcessing complete!");
    Ok(())
}
// Add this debugging function at the end of the file
#[allow(dead_code)]
fn debug_extract_text(file_path: &Path, start_byte: usize, end_byte: usize) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use std::fs::File;
    
    // Safety: Read a bit before and after to ensure we get complete UTF-8 characters
    let safe_start = if start_byte >= 4 { start_byte - 4 } else { 0 };
    let mut file = File::open(file_path)?;
    let file_size = file.metadata()?.len() as usize;
    let safe_end = std::cmp::min(end_byte + 4, file_size);
    
    // Read the extended range
    let buffer_size = safe_end - safe_start;
    let mut buffer = vec![0u8; buffer_size];
    
    file.seek(SeekFrom::Start(safe_start as u64))?;
    file.read_exact(&mut buffer)?;
    
    // Convert to string and handle UTF-8 errors safely
    let full_text = String::from_utf8_lossy(&buffer);
    
    // Find valid char boundaries closest to our target range
    let text_start = full_text.char_indices()
        .find(|(offset, _)| *offset >= (start_byte - safe_start))
        .map(|(offset, _)| offset)
        .unwrap_or(0);
        
    let text_end = full_text.char_indices()
        .filter(|(offset, _)| *offset <= (end_byte - safe_start))
        .last()
        .map(|(offset, c)| offset + c.len_utf8())
        .unwrap_or(full_text.len());
    
    // Extract the text at valid UTF-8 boundaries
    let extracted_text = &full_text[text_start..text_end];
    
    info!("Content from byte range {}..{} (adjusted to UTF-8 boundaries): \"{}\"", 
        safe_start + text_start, safe_start + text_end, 
        extracted_text.replace('\n', "\\n"));
    
    Ok(())
}