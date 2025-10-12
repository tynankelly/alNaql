#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def search_in_rust_files(search_text, root_dir=None):
    """
    Search for text in all .rs files in the given directory and subdirectories.
    
    Args:
        search_text (str): The text to search for
        root_dir (str): The root directory to search in (defaults to script location)
    
    Returns:
        list: List of tuples containing (file_path, line_number, line_content)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent
    else:
        root_dir = Path(root_dir)
    
    results = []
    
    # Find all .rs files recursively
    for rs_file in root_dir.rglob("*.rs"):
        try:
            with open(rs_file, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    if search_text in line:
                        results.append((rs_file, line_num, line.strip()))
        except Exception as e:
            print(f"Error reading {rs_file}: {e}")
    
    return results

def main():
    print("Rust File Text Search")
    print("=" * 30)
    
    # Get search text from user
    search_text = input("Enter the text to search for: ").strip()
    
    if not search_text:
        print("Error: No search text provided!")
        sys.exit(1)
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    print(f"Searching in directory: {script_dir.absolute()}")
    print(f"Search text: '{search_text}'")
    print("-" * 50)
    
    # Perform the search
    results = search_in_rust_files(search_text)
    
    if not results:
        print("No matches found in any .rs files.")
    else:
        print(f"Found {len(results)} matches in .rs files:")
        print()
        
        current_file = None
        for file_path, line_num, line_content in results:
            # Show file name only when it changes
            if file_path != current_file:
                current_file = file_path
                print(f"\n📁 {file_path.relative_to(script_dir)}")
            
            # Show line number and content
            print(f"   Line {line_num}: {line_content}")
    
    print(f"\nSearch complete. Processed all .rs files in {script_dir.absolute()}")

if __name__ == "__main__":
    main()