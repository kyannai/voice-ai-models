#!/usr/bin/env python3
"""
Patch audio paths in synthesized JSON files
Changes 'outputs/audio/...' to 'audio/...'
"""

import json
import sys
from pathlib import Path

def patch_audio_paths(input_file: str, output_file: str = None):
    """Patch audio paths in JSON file"""
    if output_file is None:
        output_file = input_file
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's the new format with 'results' key
    if 'results' in data:
        results = data['results']
    else:
        results = data
    
    # Patch audio paths
    patched_count = 0
    for result in results:
        if 'audio_path' in result:
            old_path = result['audio_path']
            # Replace 'outputs/audio/' with 'audio/'
            if old_path.startswith('outputs/audio/'):
                result['audio_path'] = old_path.replace('outputs/audio/', 'audio/', 1)
                patched_count += 1
            # Also handle absolute paths
            elif '/outputs/audio/' in old_path:
                # Get just the filename and prepend 'audio/'
                filename = Path(old_path).name
                result['audio_path'] = f'audio/{filename}'
                patched_count += 1
    
    print(f"Patched {patched_count} audio paths")
    
    # Save patched data
    print(f"Saving to {output_file}...")
    if 'results' in data:
        # Keep the metadata structure
        output_data = data
        output_data['results'] = results
    else:
        output_data = results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! Patched {patched_count} paths in {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_audio_paths.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    patch_audio_paths(input_file, output_file)

