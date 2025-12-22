#!/usr/bin/env python3
"""
Dataset Loader Module

Handles loading of test datasets from various sources:
- Local JSON/CSV/TSV files
- HuggingFace datasets (auto-download and cache)

Usage:
    from dataset_loader import load_dataset
    
    # Load by name from registry
    samples = load_dataset("meso-malaya-test")
    
    # Or load with specific config
    samples = load_dataset("seacrowd-asr-malcsc", max_samples=100)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import shutil

import pandas as pd

from datasets_config import get_dataset_config, list_datasets

logger = logging.getLogger(__name__)


def load_local_dataset(config: Dict, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load a local dataset from JSON, CSV, or TSV file
    
    Args:
        config: Dataset configuration dictionary
        max_samples: Optional limit on number of samples to load
        
    Returns:
        List of dictionaries with 'audio_path' and 'reference' keys
    """
    test_data_path = Path(config["test_data"])
    audio_dir = Path(config["audio_dir"])
    
    logger.info(f"Loading local dataset from: {test_data_path}")
    
    # Load data based on format
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
    
    # Limit samples if requested
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Normalize field names (handle different naming conventions)
    for sample in data:
        # Map 'path' to 'audio_path' if needed
        if 'path' in sample and 'audio_path' not in sample:
            sample['audio_path'] = sample['path']
        
        # Map 'sentence' to 'reference' if needed
        if 'sentence' in sample and 'reference' not in sample:
            sample['reference'] = sample['sentence']
    
    # Resolve audio paths (make them absolute)
    for sample in data:
        if 'audio_path' in sample:
            audio_path = Path(sample['audio_path'])
            if not audio_path.is_absolute():
                # Resolve relative to audio_dir
                sample['audio_path'] = str(audio_dir / audio_path)
    
    return data


def load_huggingface_dataset(config: Dict, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load a HuggingFace dataset and convert to standard format
    
    Args:
        config: Dataset configuration dictionary
        max_samples: Optional limit on number of samples to load
        
    Returns:
        List of dictionaries with 'audio_path' and 'reference' keys
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HuggingFace datasets. "
            "Install it with: pip install datasets"
        )
    
    dataset_name = config["hf_dataset"]
    split = config["hf_split"]
    trust_remote_code = config.get("trust_remote_code", False)
    
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    logger.info(f"Split: {split}")
    logger.info(f"Trust remote code: {trust_remote_code}")
    
    # Load dataset from HuggingFace
    dataset = hf_load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=trust_remote_code
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Limit samples if requested
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Convert to standard format
    audio_field = config.get("audio_field", "audio")
    text_field = config.get("text_field", "text")
    
    logger.info(f"Converting dataset to standard format...")
    logger.info(f"  Audio field: {audio_field}")
    logger.info(f"  Text field: {text_field}")
    
    # Create temporary directory for audio files
    temp_dir = Path(tempfile.mkdtemp(prefix="asr_eval_hf_"))
    logger.info(f"Audio cache directory: {temp_dir}")
    
    samples = []
    for idx, item in enumerate(dataset):
        # Extract audio
        if audio_field not in item:
            logger.warning(f"Sample {idx} missing audio field '{audio_field}', skipping")
            continue
        
        audio_data = item[audio_field]
        
        # Handle different audio formats from HF datasets
        # Audio data can be a dict with 'path', 'array', 'sampling_rate'
        if isinstance(audio_data, dict):
            # Try to get the path first
            if 'path' in audio_data and audio_data['path']:
                audio_path = audio_data['path']
            elif 'array' in audio_data:
                # Need to save the audio array to a file
                import soundfile as sf
                audio_path = temp_dir / f"audio_{idx:06d}.wav"
                sr = audio_data.get('sampling_rate', 16000)
                sf.write(str(audio_path), audio_data['array'], sr)
            else:
                logger.warning(f"Sample {idx} has unexpected audio format, skipping")
                continue
        else:
            # Assume it's a path string
            audio_path = audio_data
        
        # Extract transcription
        if text_field not in item:
            logger.warning(f"Sample {idx} missing text field '{text_field}', skipping")
            continue
        
        reference = item[text_field]
        
        # Create sample entry
        sample = {
            'audio_path': str(audio_path),
            'reference': reference
        }
        
        samples.append(sample)
    
    logger.info(f"Converted {len(samples)} samples to standard format")
    
    return samples


def load_dataset(
    dataset_name: str,
    max_samples: Optional[int] = None,
    validate: bool = True
) -> List[Dict]:
    """
    Load a dataset by name from the registry
    
    Args:
        dataset_name: Name of the dataset (must be in registry)
        max_samples: Optional limit on number of samples to load
        validate: Whether to validate that audio files exist
        
    Returns:
        List of dictionaries with 'audio_path' and 'reference' keys
        
    Raises:
        ValueError: If dataset name is not found or invalid
        FileNotFoundError: If local dataset files don't exist
    """
    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"Loading Dataset: {dataset_name}")
    logger.info("="*70)
    logger.info(f"Description: {config['description']}")
    logger.info(f"Type: {config['type']}")
    
    # Load based on type
    if config["type"] == "local":
        samples = load_local_dataset(config, max_samples)
    elif config["type"] == "huggingface":
        samples = load_huggingface_dataset(config, max_samples)
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")
    
    # Validate samples
    if validate:
        samples = validate_samples(samples)
    
    logger.info(f"Final dataset size: {len(samples)} samples")
    logger.info("="*70)
    logger.info("")
    
    return samples


def validate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Validate samples to ensure they have required fields and audio files exist
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        List of valid samples
    """
    logger.info("Validating samples...")
    valid_samples = []
    missing_audio = 0
    missing_reference = 0
    
    for i, sample in enumerate(samples):
        # Check for required fields
        if 'audio_path' not in sample:
            missing_reference += 1
            logger.debug(f"Sample {i} missing 'audio_path'")
            continue
        
        if 'reference' not in sample:
            missing_reference += 1
            logger.debug(f"Sample {i} missing 'reference'")
            continue
        
        # Check if audio file exists
        audio_path = Path(sample['audio_path'])
        if not audio_path.exists():
            missing_audio += 1
            logger.debug(f"Audio file not found: {audio_path}")
            continue
        
        valid_samples.append(sample)
    
    # Log validation results
    invalid_count = len(samples) - len(valid_samples)
    if invalid_count > 0:
        logger.warning(f"Validation: {invalid_count} invalid samples removed")
        if missing_audio > 0:
            logger.warning(f"  - {missing_audio} samples with missing audio files")
        if missing_reference > 0:
            logger.warning(f"  - {missing_reference} samples with missing fields")
    
    logger.info(f"Valid samples: {len(valid_samples)}/{len(samples)}")
    
    return valid_samples


def list_available_datasets() -> List[str]:
    """
    Get list of all available dataset names
    
    Returns:
        List of dataset name strings
    """
    return list_datasets()


if __name__ == "__main__":
    # Simple CLI for testing dataset loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load dataset
    samples = load_dataset(
        args.dataset,
        max_samples=args.max_samples,
        validate=not args.no_validate
    )
    
    # Print first few samples
    print("\nFirst 3 samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Audio: {sample['audio_path']}")
        print(f"  Reference: {sample['reference'][:100]}...")

