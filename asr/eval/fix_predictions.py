#!/usr/bin/env python3
"""
Quick fix script to repair predictions files that have 'sentence' instead of 'reference'
"""

import json
import sys
from pathlib import Path

def fix_predictions_file(predictions_file: Path):
    """Fix a predictions file by mapping 'sentence' to 'reference'"""
    
    print(f"Loading: {predictions_file}")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if we need to fix anything
    predictions = data.get('predictions', [])
    if not predictions:
        print("  No predictions found in file")
        return
    
    fixed_count = 0
    for pred in predictions:
        # Map 'sentence' to 'reference' if needed
        if 'sentence' in pred and 'reference' not in pred:
            pred['reference'] = pred['sentence']
            fixed_count += 1
        
        # Also map 'path' to 'audio_path' if needed
        if 'path' in pred and 'audio_path' not in pred:
            pred['audio_path'] = pred['path']
    
    if fixed_count > 0:
        # Backup original file
        backup_file = predictions_file.with_suffix('.json.backup')
        print(f"  Creating backup: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save fixed file
        print(f"  Fixed {fixed_count} predictions")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved fixed file: {predictions_file}")
    else:
        print("  ✓ File already has correct field names")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_predictions.py <predictions.json>")
        print("\nExample:")
        print("  python fix_predictions.py results/predictions.json")
        sys.exit(1)
    
    predictions_file = Path(sys.argv[1])
    
    if not predictions_file.exists():
        print(f"Error: File not found: {predictions_file}")
        sys.exit(1)
    
    fix_predictions_file(predictions_file)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

