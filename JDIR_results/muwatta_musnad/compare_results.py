#!/usr/bin/env python3
import json
import re
import os
import time
from typing import List, Dict, Set
from collections import defaultdict

class ArabicCleaner:
    """Cleans and normalizes Arabic text."""
    
    def __init__(self, normalize_arabic: bool = True):
        self.normalize_arabic = normalize_arabic
        # Expanded pattern to include all Arabic characters and diacritics
        self.arabic_char_pattern = re.compile(r'[\u0600-\u06FF]')
    
    def normalize_char(self, c: str) -> str:
        """Normalize Arabic characters to standard forms."""
        if not self.normalize_arabic:
            return c
        
        # Map similar letters to a standard form
        if c in ['ئ', 'ى', 'ي', 'ی']:
            return 'ي'
        elif c == 'ة':
            return 'ه'
        elif c in ['أ', 'إ', 'آ', 'ٱ']:
            return 'ا'
        elif c == 'ؤ':
            return 'و'
        # Remove diacritics (tashkeel)
        elif c in ['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', 
                   '\u0650', '\u0651', '\u0652', '\u0653', '\u0654', '\u0655']:
            return ''
        else:
            return c
    
    def is_arabic_char(self, c: str) -> bool:
        """Check if a character is an Arabic character."""
        return bool(self.arabic_char_pattern.match(c))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Arabic text, removing non-Arabic content."""
        if not text:
            return ""
        
        # First remove alignment markers (dashes)
        text = re.sub(r'-+', '', text)
        
        # Pre-normalize the entire text to handle Arabic letter normalization
        normalized_text = ''
        for c in text:
            if self.is_arabic_char(c):
                normalized_text += self.normalize_char(c)
            else:
                normalized_text += c
        
        # Now extract and clean only the Arabic segments
        cleaned = []
        last_arabic_end = None
        
        for i, c in enumerate(normalized_text):
            if c == '\n':
                continue
            
            if self.is_arabic_char(c) and c != '':  # Skip empty chars from normalization
                if last_arabic_end is not None and i > last_arabic_end:
                    cleaned.append(' ')
                
                cleaned.append(c)
                last_arabic_end = i + 1
        
        return ''.join(cleaned)

class TextPair:
    """Represents a pair of related texts."""
    
    def __init__(self, text1, text2, source_file=""):
        self.text1 = text1.strip() if text1 else ""
        self.text2 = text2.strip() if text2 else ""
        self.source_file = source_file
        self.cleaned_text1 = ""
        self.cleaned_text2 = ""
    
    def clean(self, cleaner):
        """Clean both texts in the pair."""
        self.cleaned_text1 = cleaner.clean_text(self.text1)
        self.cleaned_text2 = cleaner.clean_text(self.text2)
    
    def to_dict(self):
        """Convert to dictionary for JSON output."""
        return {
            "filename": self.source_file,
            "text1": self.text1,
            "text2": self.text2
        }
    
    def __str__(self):
        """String representation for debugging."""
        return f"Text1: {self.text1[:30]}... | Text2: {self.text2[:30]}... | Source: {self.source_file}"
    
    def get_min_text_length(self) -> int:
        """Get the minimum length of the two texts."""
        return min(len(self.cleaned_text1 or ""), len(self.cleaned_text2 or ""))


def parse_csv_alignments(filepath):
    """Parse the tab-separated CSV file with Arabic text pairs."""
    pairs = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist")
        return pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()
            
            for line_num, line in enumerate(f, 2):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split by tabs
                    parts = line.split('\t')
                    
                    # Find Arabic text segments - they're usually longer parts
                    arabic_texts = []
                    for part in parts:
                        # Check if part has significant Arabic content
                        if re.search(r'[\u0600-\u06FF]', part) and len(part) > 20:
                            # Remove any ** markers and clean alignment dashes
                            clean_part = part.replace('**', '')
                            # Remove alignment dashes
                            clean_part = re.sub(r'-+', '', clean_part)
                            arabic_texts.append(clean_part.strip())
                    
                    if len(arabic_texts) >= 2:
                        pairs.append(TextPair(
                            text1=arabic_texts[0], 
                            text2=arabic_texts[1],
                            source_file=os.path.basename(filepath)
                        ))
                except Exception as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
    
    return pairs


def parse_jsonl_matches(filepath):
    """Parse JSONL files containing text match data."""
    pairs = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist")
        return pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract text pairs based on the format
                    text1 = ""
                    text2 = ""
                    
                    if "source_text" in data and "target_text" in data:
                        text1 = data["source_text"]
                        text2 = data["target_text"]
                    
                    elif "source_passage" in data and "target_passage" in data:
                        text1 = data["source_passage"]
                        text2 = data["target_passage"]
                    
                    if text1 and text2:  # Both texts must be non-empty
                        pairs.append(TextPair(
                            text1=text1,
                            text2=text2,
                            source_file=os.path.basename(filepath)
                        ))
                    else:
                        print(f"Skipping line {line_num} in {filepath}: Empty text field")
                except json.JSONDecodeError:
                    print(f"JSON decode error on line {line_num} in {filepath}")
                    continue
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
    
    return pairs


def exact_match_or_containment(pair1, pair2, min_length=0, ngram_threshold=0.5, word_overlap_threshold=0.6):
    """
    Check if pair2 texts are exact matches or fully contained within pair1 texts.
    
    MODIFIED: Removed minimum length requirements and lowered thresholds
    
    Args:
        pair1: First TextPair object
        pair2: Second TextPair object
        min_length: Minimum text length to consider for containment (SET TO 0)
        ngram_threshold: Threshold for n-gram matching (REDUCED to 0.5)
        word_overlap_threshold: Threshold for word overlap (REDUCED to 0.6)
    
    Returns:
        Boolean indicating whether the pairs match or one contains the other
    """
    # Get all combinations of texts to compare
    combinations = [
        (pair1.cleaned_text1, pair2.cleaned_text1),
        (pair1.cleaned_text1, pair2.cleaned_text2),
        (pair1.cleaned_text2, pair2.cleaned_text1),
        (pair1.cleaned_text2, pair2.cleaned_text2)
    ]
    
    for text1, text2 in combinations:
        if not text1 or not text2:
            continue
            
        # Check for exact match
        if text1 == text2:
            return True
            
        # Check if shorter text is fully contained in longer text
        shorter = text1 if len(text1) < len(text2) else text2
        longer = text2 if len(text1) < len(text2) else text1
        
        # Direct substring check
        if shorter in longer:
            return True
            
        # Content overlap check
        if len(shorter) > 0:  # Any length is acceptable now
            # Convert to words and check overlap
            words_shorter = shorter.split()
            words_longer = longer.split()
            
            # If words available, check for n-gram matches
            if len(words_shorter) >= 3:  # Need at least 3 words to form meaningful n-grams
                # Create n-grams from the shorter text
                n = 3  # REDUCED from 4 to 3 words for matching
                shorter_ngrams = [' '.join(words_shorter[i:i+n]) for i in range(len(words_shorter)-n+1)]
                longer_text = ' '.join(words_longer)
                
                # Count matching n-grams
                if shorter_ngrams:  # Check if there are any n-grams
                    matching_ngrams = sum(1 for ngram in shorter_ngrams if ngram in longer_text)
                    
                    # Check if proportion of matching n-grams exceeds threshold
                    if matching_ngrams >= len(shorter_ngrams) * ngram_threshold:
                        return True
            
            # Also check for high word overlap
            if words_shorter and words_longer:  # Ensure both have words
                intersection = len(set(words_shorter).intersection(set(words_longer)))
                if len(words_shorter) > 0:  # Avoid division by zero
                    containment = intersection / len(words_shorter)
                    
                    if containment >= word_overlap_threshold:
                        return True
    
    return False


def calculate_strict_similarity(pair1, pair2):
    """
    Calculate similarity with better support for extracts.
    MODIFIED: More lenient to work with any text length
    """
    # Check all text combinations
    combinations = [
        (pair1.cleaned_text1, pair2.cleaned_text1),
        (pair1.cleaned_text1, pair2.cleaned_text2),
        (pair1.cleaned_text2, pair2.cleaned_text1),
        (pair1.cleaned_text2, pair2.cleaned_text2)
    ]
    
    max_similarity = 0
    for text1, text2 in combinations:
        if not text1 or not text2:
            continue
        
        # Check for substring relationship first
        if text1 in text2 or text2 in text1:
            return 0.95  # High similarity score for substring relationships
        
        # Get words for similarity calculation
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Handle very short texts
        if not words1 or not words2:
            continue
            
        # Calculate word overlap
        intersection = len(words1.intersection(words2))
        
        # Calculate Jaccard similarity
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0
        
        # Check for significant shared phrases with lower thresholds
        has_significant_phrase = False
        if jaccard > 0.3:  # Reduced from 0.4 to 0.3
            # Convert to word lists to check for phrases
            words1_list = text1.split()
            words2_list = text2.split()
            
            # Look for sequences of 3+ identical words (reduced from 4 to 3)
            min_phrase_length = min(3, min(len(words1_list), len(words2_list)))
            if min_phrase_length >= 2:  # At least 2 words for a phrase
                for i in range(len(words1_list) - min_phrase_length + 1):
                    phrase = ' '.join(words1_list[i:i+min_phrase_length])
                    if phrase in ' '.join(words2_list):
                        has_significant_phrase = True
                        break
        
        # Enhanced similarity calculation
        if has_significant_phrase:
            similarity = max(jaccard * 1.3, 0.7)  # Boosted similarity
        else:
            similarity = jaccard
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def get_representative_samples(cluster, n=5):
    """
    Get representative samples from a cluster for comparison.
    Instead of just taking first/last elements, this tries to select diverse samples.
    """
    if len(cluster) <= n:
        return cluster
    
    # Sort by text length to get a range of text sizes
    sorted_by_length = sorted(cluster, key=lambda p: p.get_min_text_length())
    
    # Take evenly spaced samples across the sorted list
    step = max(1, len(sorted_by_length) // n)
    samples = [sorted_by_length[i] for i in range(0, len(sorted_by_length), step)]
    
    # If we didn't get enough samples, add some from the beginning/end
    while len(samples) < n and len(sorted_by_length) > len(samples):
        remaining = [p for p in sorted_by_length if p not in samples]
        if not remaining:
            break
        samples.append(remaining[0])  # Add the first remaining item
    
    return samples[:n]  # Limit to n samples


def improved_clustering_pipeline(all_pairs):
    """
    An improved clustering pipeline with no length filtering.
    
    Args:
        all_pairs: List of TextPair objects to cluster
        
    Returns:
        List of clusters (each cluster is a list of TextPair objects)
    """
    print("Starting improved clustering pipeline...")
    
    # Phase 1: Initial clustering (no length filtering)
    clusters = initial_clustering(all_pairs)
    print(f"Initial clustering complete: {len(clusters)} clusters")
    
    # Phase 2: Refine clusters iteratively
    refined_clusters = iterative_cluster_refinement(clusters)
    print(f"Refinement complete: {len(refined_clusters)} clusters")
    
    return refined_clusters


def initial_clustering(pairs):
    """
    Perform initial clustering based on exact matches and containment.
    MODIFIED: No length filtering
    """
    print("Performing initial clustering...")
    print(f"Total pairs to cluster: {len(pairs)}")
    
    # Group exact matches and containment
    clusters = []
    used_pairs = set()
    
    # Sort pairs by length (longest first) for efficient containment checks
    sorted_pairs = sorted(pairs, key=lambda p: max(len(p.cleaned_text1 or ""), len(p.cleaned_text2 or "")), reverse=True)
    
    # For each pair, try to find an existing cluster or create a new one
    for i, pair in enumerate(sorted_pairs):
        if i % 100 == 0 or i == len(sorted_pairs) - 1:
            print(f"  Processing pair {i+1}/{len(sorted_pairs)}")
        
        if pair in used_pairs:
            continue
            
        # First check if this pair can be added to an existing cluster
        added_to_cluster = False
        for cluster in clusters:
            # Check if pair is related to any pair in the cluster
            for cluster_pair in cluster:
                if exact_match_or_containment(cluster_pair, pair):
                    cluster.append(pair)
                    used_pairs.add(pair)
                    added_to_cluster = True
                    break
            if added_to_cluster:
                break
                
        # If not added to any existing cluster, create a new one
        if not added_to_cluster:
            # Create a new cluster with this pair
            new_cluster = [pair]
            used_pairs.add(pair)
            
            # Find all other pairs that are exact matches or contained in this pair
            for j, other_pair in enumerate(sorted_pairs):
                if other_pair in used_pairs:
                    continue
                    
                # Check if other_pair is an exact match or contained in any pair in the cluster
                if exact_match_or_containment(pair, other_pair):
                    new_cluster.append(other_pair)
                    used_pairs.add(other_pair)
            
            clusters.append(new_cluster)
    
    # Make sure all pairs are in clusters
    unused_pairs = [p for p in pairs if p not in used_pairs]
    if unused_pairs:
        print(f"Found {len(unused_pairs)} unassigned pairs. Creating single-item clusters.")
        for pair in unused_pairs:
            clusters.append([pair])
    
    return clusters


def iterative_cluster_refinement(initial_clusters, max_iterations=3):
    """
    Iteratively refine clusters through a controlled process of:
    1. Merging similar clusters
    2. Validating clusters (but never dropping items)
    
    MODIFIED: Ensure no pairs are dropped during refinement
    
    Args:
        initial_clusters: List of clusters to refine
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        List of refined clusters
    """
    print(f"Starting iterative refinement with {len(initial_clusters)} clusters...")
    
    # Count total pairs for verification
    total_pairs = sum(len(cluster) for cluster in initial_clusters)
    print(f"Total pairs before refinement: {total_pairs}")
    
    current_clusters = initial_clusters
    
    for iteration in range(max_iterations):
        print(f"Refinement iteration {iteration+1}/{max_iterations}")
        cluster_count_before = len(current_clusters)
        
        # Step 1: Perform hierarchical merging of similar clusters
        print("  Merging similar clusters...")
        merged_clusters = merge_similar_clusters(current_clusters)
        print(f"  After merging: {len(merged_clusters)} clusters")
        
        # Step 2: Identify and split problematic clusters
        print("  Validating and splitting problematic clusters...")
        all_refined_clusters = []
        problematic_items = []
        
        for cluster in merged_clusters:
            # Only validate clusters with more than 3 items
            if len(cluster) <= 3:
                all_refined_clusters.append(cluster)
                continue
                
            valid_items, problems = identify_cluster_problems(cluster)
            
            # Add valid items as a cluster if there are any
            if valid_items:
                all_refined_clusters.append(valid_items)
                
            # Collect problematic items for later processing
            problematic_items.extend(problems)
        
        # Step 3: Process all problematic items together
        if problematic_items:
            print(f"  Found {len(problematic_items)} problematic items across all clusters")
            
            # Try to form new clusters from problematic items
            problem_clusters = process_problematic_items(problematic_items)
            all_refined_clusters.extend(problem_clusters)
            
            print(f"  Created {len(problem_clusters)} new clusters from problematic items")
        
        current_clusters = all_refined_clusters
        print(f"  After refinement: {len(current_clusters)} clusters")
        
        # Verify no pairs were lost during refinement
        current_total = sum(len(cluster) for cluster in current_clusters)
        if current_total != total_pairs:
            print(f"WARNING: Pair count mismatch! Before: {total_pairs}, After: {current_total}")
            # Find and create clusters for any lost pairs
            all_pairs_set = set()
            for cluster in initial_clusters:
                for pair in cluster:
                    all_pairs_set.add(pair)
            
            current_pairs_set = set()
            for cluster in current_clusters:
                for pair in cluster:
                    current_pairs_set.add(pair)
            
            lost_pairs = all_pairs_set - current_pairs_set
            if lost_pairs:
                print(f"Recovering {len(lost_pairs)} lost pairs...")
                for pair in lost_pairs:
                    current_clusters.append([pair])  # Create single-item clusters
                print(f"After recovery: {len(current_clusters)} clusters")
        
        # Check if the number of clusters has stabilized
        if len(current_clusters) == cluster_count_before:
            print(f"Cluster count stabilized after {iteration+1} iterations")
            break
    
    # Final verification that no pairs were lost
    final_total = sum(len(cluster) for cluster in current_clusters)
    if final_total != total_pairs:
        print(f"FINAL WARNING: Pair count mismatch! Expected: {total_pairs}, Got: {final_total}")
    else:
        print(f"Verification successful: All {total_pairs} pairs preserved in {len(current_clusters)} clusters")
    
    return current_clusters


def merge_similar_clusters(clusters, similarity_threshold=0.75):
    """
    Merge clusters that have high similarity.
    REDUCED similarity threshold to merge more aggressively.
    """
    if len(clusters) <= 1:
        return clusters
        
    result_clusters = clusters.copy()
    
    # Continue merging until no more merges are possible
    merged_any = True
    merge_pass = 0
    
    while merged_any and merge_pass < 5:  # Limit number of passes
        merge_pass += 1
        merged_any = False
        
        # Find the most similar pair of clusters
        best_similarity = 0
        best_pair = (-1, -1)
        
        for i in range(len(result_clusters)):
            for j in range(i+1, len(result_clusters)):
                # Calculate maximum similarity between any texts in the clusters
                max_sim = 0
                
                # Use representative samples for cluster comparison
                sample_i = get_representative_samples(result_clusters[i], 5)
                sample_j = get_representative_samples(result_clusters[j], 5)
                
                for pair1 in sample_i:
                    for pair2 in sample_j:
                        sim = calculate_strict_similarity(pair1, pair2)
                        max_sim = max(max_sim, sim)
                
                # Update best match if this is better
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_pair = (i, j)
        
        # If best similarity exceeds threshold, merge those clusters
        if best_similarity >= similarity_threshold:
            i, j = best_pair
            print(f"    Merging clusters with similarity {best_similarity:.4f}")
            result_clusters[i].extend(result_clusters[j])
            result_clusters.pop(j)
            merged_any = True
    
    return result_clusters


def identify_cluster_problems(cluster, min_connections=0.3, min_similarity=0.5):
    """
    Identify items within a cluster that don't have sufficient connections
    to other items in the cluster.
    
    MODIFIED: Much more lenient thresholds to keep more items together
    
    Returns:
        tuple: (valid_items, problematic_items)
    """
    # For small clusters, assume all items are valid
    if len(cluster) <= 3:
        return cluster, []
        
    # Create a similarity matrix
    similarity_matrix = {}
    for idx1, pair1 in enumerate(cluster):
        similarity_matrix[idx1] = {}
        for idx2, pair2 in enumerate(cluster):
            if idx1 == idx2:
                similarity_matrix[idx1][idx2] = 1.0  # Self-similarity
            elif idx2 in similarity_matrix and idx1 in similarity_matrix[idx2]:
                # We already computed this pair
                similarity_matrix[idx1][idx2] = similarity_matrix[idx2][idx1]
            else:
                # Calculate pairwise similarity
                similarity_matrix[idx1][idx2] = calculate_strict_similarity(pair1, pair2)
    
    # Count connections for each item
    connected_counts = {}
    for idx in range(len(cluster)):
        # Count items with similarity above threshold
        connected_counts[idx] = sum(1 for other_idx in range(len(cluster)) 
                                  if similarity_matrix[idx][other_idx] >= min_similarity)
    
    # Find items with insufficient connections
    min_required = max(2, int(len(cluster) * min_connections))
    problem_indices = [idx for idx, count in connected_counts.items() if count < min_required]
    
    # If fewer than 20% are problematic, keep the cluster mostly intact
    if len(problem_indices) <= len(cluster) * 0.2:
        valid_items = [pair for idx, pair in enumerate(cluster) if idx not in problem_indices]
        problem_items = [pair for idx, pair in enumerate(cluster) if idx in problem_indices]
        return valid_items, problem_items
    
    # If more than 20% are problematic, try to identify the largest coherent subgroup
    # For simplicity, we'll use a greedy approach
    best_subgroup = []
    best_subgroup_size = 0
    
    for start_idx in range(len(cluster)):
        if start_idx in problem_indices:
            continue
            
        # Try to build a coherent subgroup starting with this item
        subgroup = [start_idx]
        for idx in range(len(cluster)):
            if idx == start_idx or idx in problem_indices:
                continue
                
            # Check if this item is similar to all items in current subgroup
            is_similar = all(similarity_matrix[idx][subgroup_idx] >= min_similarity 
                            for subgroup_idx in subgroup)
                            
            if is_similar:
                subgroup.append(idx)
        
        if len(subgroup) > best_subgroup_size:
            best_subgroup = subgroup
            best_subgroup_size = len(subgroup)
    
    # Extract the valid items and problematic items
    valid_items = [pair for idx, pair in enumerate(cluster) if idx in best_subgroup]
    problem_items = [pair for idx, pair in enumerate(cluster) if idx not in best_subgroup]
    
    return valid_items, problem_items


def process_problematic_items(problem_items):
    """
    Process problematic items to form new clusters.
    MODIFIED: Ensure all problematic items get assigned to clusters
    """
    # If very few items, just create individual clusters
    if len(problem_items) <= 2:
        return [[item] for item in problem_items]
        
    # Try a simplified clustering approach
    result_clusters = []
    remaining_items = set(problem_items)
    
    while remaining_items:
        # Take one item as the seed for a new cluster
        current_item = next(iter(remaining_items))
        current_cluster = [current_item]
        remaining_items.remove(current_item)
        
        # Find similar items
        similar_items = []
        for item in list(remaining_items):
            similarity = calculate_strict_similarity(current_item, item)
            # REDUCED similarity threshold to capture more related items
            if similarity >= 0.6:
                similar_items.append((item, similarity))
        
        # Sort by similarity (highest first) and add to cluster
        similar_items.sort(key=lambda x: x[1], reverse=True)
        for item, _ in similar_items:
            current_cluster.append(item)
            remaining_items.remove(item)
        
        result_clusters.append(current_cluster)
    
    # Verify all problematic items have been assigned
    all_assigned = set()
    for cluster in result_clusters:
        for item in cluster:
            all_assigned.add(item)
    
    missing = set(problem_items) - all_assigned
    if missing:
        print(f"  WARNING: {len(missing)} problematic items were not assigned. Creating single-item clusters.")
        for item in missing:
            result_clusters.append([item])
    
    return result_clusters


# UPDATED MAIN FUNCTION
def main():
    # Configuration files
    csv_file = "passim.csv"  # Tab-separated file with text pairs
    jsonl_file1 = "alnaql.jsonl"
    jsonl_file2 = "textpair.jsonl"
    output_file = "muwatta_musnad.json"
    
    # Initialize text cleaner
    cleaner = ArabicCleaner(normalize_arabic=True)
    
    # Parse all input files
    print(f"Parsing CSV file: {csv_file}")
    csv_pairs = parse_csv_alignments(csv_file)
    print(f"  Found {len(csv_pairs)} text pairs")
    
    print(f"Parsing JSONL file 1: {jsonl_file1}")
    jsonl1_pairs = parse_jsonl_matches(jsonl_file1)
    print(f"  Found {len(jsonl1_pairs)} text pairs")
    
    print(f"Parsing JSONL file 2: {jsonl_file2}")
    jsonl2_pairs = parse_jsonl_matches(jsonl_file2)
    print(f"  Found {len(jsonl2_pairs)} text pairs")
    
    # Combine all text pairs
    all_pairs = csv_pairs + jsonl1_pairs + jsonl2_pairs
    print(f"Total text pairs: {len(all_pairs)}")
    
    # Clean Arabic text
    print("Cleaning Arabic text...")
    for pair in all_pairs:
        pair.clean(cleaner)
    
    # Use the improved clustering pipeline
    print("Performing clustering...")
    clusters = improved_clustering_pipeline(all_pairs)
    
    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: len(c), reverse=True)
    
    # Verify all pairs made it into clusters
    all_clustered_pairs = set()
    for cluster in clusters:
        for pair in cluster:
            all_clustered_pairs.add(pair)
    
    if len(all_clustered_pairs) != len(all_pairs):
        print(f"WARNING: Not all pairs were clustered. Expected {len(all_pairs)}, got {len(all_clustered_pairs)}")
        # Find the missing pairs
        missing_pairs = set(all_pairs) - all_clustered_pairs
        print(f"Adding {len(missing_pairs)} missing pairs as individual clusters")
        for pair in missing_pairs:
            clusters.append([pair])
    else:
        print(f"All {len(all_pairs)} pairs successfully clustered into {len(clusters)} clusters")
    
    # Prepare output data
    output_data = []
    for i, cluster in enumerate(clusters):
        cluster_data = {
            "cluster_id": i,
            "size": len(cluster),
            "items": [pair.to_dict() for pair in cluster]
        }
        output_data.append(cluster_data)
    
    # Write output to JSON file
    print(f"Writing output to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Successfully processed all {len(all_pairs)} text pairs into {len(clusters)} clusters")
    
    # Print some statistics about cluster sizes
    cluster_sizes = [len(cluster) for cluster in clusters]
    single_item_clusters = sum(1 for size in cluster_sizes if size == 1)
    print(f"Statistics:")
    print(f"  - Single-item clusters: {single_item_clusters} ({single_item_clusters/len(clusters)*100:.1f}%)")
    print(f"  - Largest cluster size: {max(cluster_sizes)}")
    print(f"  - Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.2f}")


if __name__ == "__main__":
    main()