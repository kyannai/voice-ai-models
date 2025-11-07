#!/usr/bin/env python3
"""
Shared utilities for FunASR-based transcribers
Contains common functionality for data loading, output cleaning, and result saving
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_test_data(test_data_path: Union[str, Path], audio_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
    """
    Load test data from JSON or CSV file
    
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
    else:
        raise ValueError(f"Unsupported file format: {test_data_path.suffix}")
    
    logger.info(f"Loaded {len(data)} samples from {test_data_path.suffix.upper()}")
    
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
    - "The original content of this audio is: '...'"
    - "The transcription of the audio is: '...'"
    - "The audio content is: '...'"
    
    This function strips these preambles and returns only the actual transcription.
    
    Args:
        text: Raw output from Qwen Audio model
        
    Returns:
        Cleaned transcription text
    """
    if not text:
        return text
    
    # Patterns to remove (in order of specificity)
    patterns = [
        # Full preambles with quotes
        r"^The original content of this audio is:\s*['\"](.+?)['\"]\.?\s*$",
        r"^The transcription of (?:the )?audio is:\s*['\"](.+?)['\"]\.?\s*$",
        r"^The audio content is:\s*['\"](.+?)['\"]\.?\s*$",
        r"^(?:The )?transcription:\s*['\"](.+?)['\"]\.?\s*$",
        r"^Audio transcription:\s*['\"](.+?)['\"]\.?\s*$",
        
        # Preambles without quotes
        r"^The original content of this audio is:\s*(.+)$",
        r"^The transcription of (?:the )?audio is:\s*(.+)$",
        r"^The audio content is:\s*(.+)$",
        r"^(?:The )?transcription:\s*(.+)$",
        r"^Audio transcription:\s*(.+)$",
        
        # Just quotes
        r"^['\"](.+?)['\"]\.?\s*$",
    ]
    
    cleaned = text.strip()
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            # Extract the captured group (the actual transcription)
            cleaned = match.group(1).strip()
            break
    
    # Remove any remaining leading/trailing quotes
    cleaned = re.sub(r"^['\"]|['\"]$", "", cleaned).strip()
    
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

