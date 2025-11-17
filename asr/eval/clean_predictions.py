#!/usr/bin/env python3
"""
Script to clean existing prediction files by removing model preambles
like "The audio says: '...'", "The audio transcription is: '...'", etc.

Usage:
    python clean_predictions.py <predictions.csv>
    python clean_predictions.py <predictions_dir>  # Will clean all predictions.csv in subdirs
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import logging

# Add transcribe directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent / "transcribe"))
from utils import clean_qwen_output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_csv_file(csv_path: Path, backup: bool = True) -> int:
    """
    Clean a predictions.csv file
    
    Args:
        csv_path: Path to predictions.csv
        backup: Whether to create a backup before cleaning
        
    Returns:
        Number of rows cleaned
    """
    logger.info(f"Cleaning {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    if 'hypothesis' not in df.columns:
        logger.warning(f"No 'hypothesis' column in {csv_path}, skipping")
        return 0
    
    # Create backup if requested
    if backup:
        backup_path = csv_path.with_suffix('.csv.bak')
        df.to_csv(backup_path, index=False, encoding='utf-8')
        logger.info(f"Created backup: {backup_path}")
    
    # Clean hypotheses
    cleaned_count = 0
    original_values = df['hypothesis'].copy()
    df['hypothesis'] = df['hypothesis'].apply(clean_qwen_output)
    
    # Count changes
    cleaned_count = (df['hypothesis'] != original_values).sum()
    
    if cleaned_count > 0:
        # Save cleaned file
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Cleaned {cleaned_count}/{len(df)} hypotheses in {csv_path}")
    else:
        logger.info(f"No changes needed for {csv_path}")
    
    return cleaned_count


def clean_json_file(json_path: Path, backup: bool = True) -> int:
    """
    Clean a predictions.json file
    
    Args:
        json_path: Path to predictions.json
        backup: Whether to create a backup before cleaning
        
    Returns:
        Number of predictions cleaned
    """
    logger.info(f"Cleaning {json_path}")
    
    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'predictions' not in data:
        logger.warning(f"No 'predictions' key in {json_path}, skipping")
        return 0
    
    # Create backup if requested
    if backup:
        backup_path = json_path.with_suffix('.json.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Created backup: {backup_path}")
    
    # Clean hypotheses
    cleaned_count = 0
    for pred in data['predictions']:
        if 'hypothesis' in pred:
            original = pred['hypothesis']
            cleaned = clean_qwen_output(original)
            if cleaned != original:
                pred['hypothesis'] = cleaned
                cleaned_count += 1
    
    if cleaned_count > 0:
        # Save cleaned file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Cleaned {cleaned_count}/{len(data['predictions'])} hypotheses in {json_path}")
    else:
        logger.info(f"No changes needed for {json_path}")
    
    return cleaned_count


def main():
    parser = argparse.ArgumentParser(
        description='Clean prediction files by removing model preambles'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to predictions.csv, predictions.json, or directory containing them'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='Recursively search for prediction files in subdirectories'
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    backup = not args.no_backup
    
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return 1
    
    total_cleaned = 0
    
    # Handle single file
    if path.is_file():
        if path.name == 'predictions.csv':
            total_cleaned += clean_csv_file(path, backup=backup)
        elif path.name == 'predictions.json':
            total_cleaned += clean_json_file(path, backup=backup)
        else:
            logger.error(f"File must be named 'predictions.csv' or 'predictions.json', got: {path.name}")
            return 1
    
    # Handle directory
    elif path.is_dir():
        # Find all prediction files
        csv_files = list(path.rglob('predictions.csv') if args.recursive else path.glob('predictions.csv'))
        json_files = list(path.rglob('predictions.json') if args.recursive else path.glob('predictions.json'))
        
        if not csv_files and not json_files:
            logger.error(f"No predictions.csv or predictions.json files found in {path}")
            return 1
        
        logger.info(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files")
        
        # Clean all CSV files
        for csv_file in csv_files:
            total_cleaned += clean_csv_file(csv_file, backup=backup)
        
        # Clean all JSON files
        for json_file in json_files:
            total_cleaned += clean_json_file(json_file, backup=backup)
    
    logger.info(f"\nTotal changes: {total_cleaned} hypotheses cleaned")
    return 0


if __name__ == '__main__':
    sys.exit(main())

