#!/usr/bin/env python3
"""
Shared utilities for FunASR-based transcribers
Contains common functionality for data loading, output cleaning, and result saving
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import pandas as pd

# Add parent directory to path to import dataset_loader
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_dataset_by_name(
    dataset_name: str,
    max_samples: Optional[int] = None,
    validate: bool = True
) -> List[Dict]:
    """
    Load a dataset by name from the dataset registry
    
    Args:
        dataset_name: Name of the dataset (from datasets_config.py)
        max_samples: Optional limit on number of samples
        validate: Whether to validate samples
        
    Returns:
        List of dictionaries with 'audio_path' and 'reference' keys
    """
    try:
        from dataset_loader import load_dataset
    except ImportError:
        raise ImportError(
            "Could not import dataset_loader. "
            "Make sure dataset_loader.py is in the parent directory."
        )
    
    return load_dataset(dataset_name, max_samples=max_samples, validate=validate)


def load_test_data(test_data_path: Union[str, Path], audio_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
    """
    Load test data from JSON, CSV, or TSV file
    
    Args:
        test_data_path: Path to test data file
        audio_dir: Optional base directory for audio files
        
    Returns:
        List of dictionaries with 'audio_path' and 'reference' keys
    """
    test_data_path = Path(test_data_path)
    logger.info(f"Loading data from: {test_data_path}")
    
    # Detect format
    if test_data_path.suffix == '.json':
        logger.info("Format: JSON")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif test_data_path.suffix == '.csv':
        logger.info("Format: CSV")
        df = pd.read_csv(test_data_path)
        data = df.to_dict('records')
    elif test_data_path.suffix == '.tsv':
        logger.info("Format: TSV")
        df = pd.read_csv(test_data_path, sep='\t')
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {test_data_path.suffix}")
    
    logger.info(f"Loaded {len(data)} samples from {test_data_path.suffix.upper()}")
    
    # Normalize field names (handle different naming conventions)
    for sample in data:
        # Map 'path' to 'audio_path' if needed
        if 'path' in sample and 'audio_path' not in sample:
            sample['audio_path'] = sample['path']
        
        # Map 'sentence' to 'reference' if needed
        if 'sentence' in sample and 'reference' not in sample:
            sample['reference'] = sample['sentence']
    
    # Resolve audio paths
    if audio_dir:
        audio_dir = Path(audio_dir)
        for sample in data:
            if 'audio_path' in sample:
                audio_path = Path(sample['audio_path'])
                if not audio_path.is_absolute():
                    sample['audio_path'] = str(audio_dir / audio_path)
    
    return data


def validate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Validate and filter samples to ensure they have required fields
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        List of valid samples
    """
    logger.info("Validating samples...")
    valid_samples = []
    
    for sample in samples:
        if 'audio_path' not in sample:
            logger.warning(f"Sample missing 'audio_path': {sample}")
            continue
        
        audio_path = Path(sample['audio_path'])
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue
        
        if 'reference' not in sample:
            logger.warning(f"Sample missing 'reference': {sample}")
            continue
        
        valid_samples.append(sample)
    
    logger.info(f"Valid samples: {len(valid_samples)}/{len(samples)}")
    return valid_samples


def clean_qwen_output(text: str) -> str:
    """
    Clean Qwen Audio model output by removing common preambles.
    
    Qwen Audio models (Qwen2-Audio, Qwen3-Omni) sometimes add preambles like:
    - "The audio says: '...'"
    - "The original content of this audio is: '...'"
    - "The transcription of the audio is: '...'"
    
    Strategy:
    1. First try to extract content within quotes (most reliable)
    2. If no quotes, try to match and remove known preamble patterns
    3. Otherwise return the original text
    
    Args:
        text: Raw output from Qwen Audio model
        
    Returns:
        Cleaned transcription text
    """
    if not text:
        return text
    
    cleaned = text.strip()
    
    # Strategy 1: Extract content within quotes (single or double)
    # This handles most cases like: "The original content of this audio is 'actual text'."
    # Try to find content between quotes, preferring the longest match
    quote_pattern = r"['\"](.+?)['\"]"
    quote_matches = re.findall(quote_pattern, cleaned, flags=re.DOTALL)
    
    if quote_matches:
        # If we found quoted content, use the longest one (usually the actual transcription)
        # This handles cases where there might be multiple quoted sections
        longest_match = max(quote_matches, key=len)
        if len(longest_match) > 10:  # Reasonable length for transcription
            return longest_match.strip()
    
    # Strategy 2: Try to match specific preamble patterns
    # This handles cases without quotes or with incomplete quotes
    patterns = [
        # Preambles with text after (no quotes)
        r"^The audio says:\s*(.+)$",
        r"^The audio transcription is:\s*(.+)$",
        r"^The original content of this audio is:\s*(.+)$",
        r"^The transcription of (?:the )?audio is:\s*(.+)$",
        r"^The audio content is:\s*(.+)$",
        r"^(?:The )?transcription:\s*(.+)$",
        r"^Audio transcription:\s*(.+)$",
    ]
    
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            # Extract the captured group (the actual transcription)
            cleaned = match.group(1).strip()
            # Remove any remaining quotes
            cleaned = re.sub(r"^['\"]|['\"]\.?$", "", cleaned).strip()
            return cleaned
    
    # Strategy 3: If text is entirely wrapped in quotes, remove them
    if (cleaned.startswith("'") and cleaned.endswith("'")) or \
       (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1].strip()
        # Also remove trailing period after closing quote if present
        cleaned = re.sub(r"^['\"]|['\"]\.?$", "", cleaned).strip()
    
    return cleaned


def save_predictions(predictions: List[Dict], output_dir: Union[str, Path], 
                     model_name: str, include_timing: bool = True):
    """
    Save predictions to JSON and CSV files in format compatible with calculate_metrics.py
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save outputs
        model_name: Name of the model (for logging)
        include_timing: Whether to include timing information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    df = pd.DataFrame(predictions)
    total_audio = df['audio_duration'].sum() if 'audio_duration' in df.columns else 0
    total_time = df['processing_time'].sum() if 'processing_time' in df.columns else 0
    avg_rtf = df['rtf'].mean() if 'rtf' in df.columns else 0
    
    # Create results dictionary (format expected by calculate_metrics.py)
    results = {
        "model": model_name,
        "num_samples": len(predictions),
        "timing": {
            "total_audio_duration": float(total_audio),
            "total_processing_time": float(total_time),
            "average_rtf": float(avg_rtf),
        },
        "predictions": predictions
    }
    
    # Save as JSON (complete with all details)
    json_file = output_dir / "predictions.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions (JSON) to: {json_file}")
    
    # Save as CSV (simplified for easy viewing)
    csv_file = output_dir / "predictions.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    logger.info(f"Saved predictions (CSV) to: {csv_file}")
    
    # Log summary statistics
    if include_timing and 'processing_time' in df.columns:
        logger.info(f"\n{'='*70}")
        logger.info(f"Transcription completed: {len(predictions)} samples")
        logger.info(f"Model: {model_name}")
        logger.info(f"Total audio duration: {total_audio:.2f}s")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average RTF: {avg_rtf:.3f}")
        logger.info(f"{'='*70}")

