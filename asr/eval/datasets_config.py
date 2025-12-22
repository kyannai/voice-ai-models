#!/usr/bin/env python3
"""
Dataset Registry Configuration

Defines all available test datasets for ASR evaluation.
Supports both local datasets (JSON/CSV/TSV) and HuggingFace datasets.

Usage:
    from datasets_config import DATASETS, get_dataset_config
    
    config = get_dataset_config("meso-malaya-test")
    print(config["type"])  # "local"
    print(config["test_data"])  # Path to JSON file
"""

from pathlib import Path
from typing import Dict, Optional

# Base directory for this file (asr/eval)
EVAL_DIR = Path(__file__).parent


# Dataset Registry
DATASETS = {
    "meso-malaya-test": {
        "type": "local",
        "description": "Malaya Malay test set from mesolitica",
        "test_data": EVAL_DIR / "test_data/malaya-test/malaya-malay-test-set.json",
        "audio_dir": EVAL_DIR / "test_data/malaya-test",
        "language": "ms",
        "num_samples": None,  # Will be determined on load
    },
    
    "ytl-malay-test": {
        "type": "local",
        "description": "YTL internal Malay test set",
        "test_data": EVAL_DIR / "test_data/ytl-malay-test/asr_ground_truths.json",
        "audio_dir": EVAL_DIR / "test_data/ytl-malay-test",
        "language": "ms",
        "num_samples": None,
    },
    
    "seacrowd-asr-malcsc": {
        "type": "local",
        "description": "SEACrowd ASR MALCSC dataset (Malay conversational speech)",
        "test_data": EVAL_DIR / "test_data/seacrowd-malcsc/seacrowd_malcsc.json",
        "audio_dir": EVAL_DIR / "test_data/seacrowd-malcsc",
        "language": "ms",
        "num_samples": 20,  # 20 conversational samples
    },
    
    "fleurs-test": {
        "type": "local",
        "description": "FLEURS Malay test set",
        "test_data": EVAL_DIR / "test_data/YTL_testsets/fleurs_test.tsv",
        "audio_dir": EVAL_DIR / "test_data/YTL_testsets",
        "language": "ms",
        "num_samples": None,
    },
    
    "malay-scripted": {
        "type": "local",
        "description": "YTL Malay scripted speech test set",
        "test_data": EVAL_DIR / "test_data/YTL_testsets/malay_scripted_meta.tsv",
        "audio_dir": EVAL_DIR / "test_data/YTL_testsets",
        "language": "ms",
        "num_samples": None,
    },
    
    "malay-conversational": {
        "type": "local",
        "description": "YTL Malay conversational speech test set",
        "test_data": EVAL_DIR / "test_data/YTL_testsets/malay_conversational_meta.tsv",
        "audio_dir": EVAL_DIR / "test_data/YTL_testsets",
        "language": "ms",
        "num_samples": None,
    },
}


def get_dataset_config(dataset_name: str) -> Dict:
    """
    Get configuration for a dataset by name
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
        
    Returns:
        Dataset configuration dictionary
        
    Raises:
        ValueError: If dataset name is not found
    """
    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    return DATASETS[dataset_name]


def list_datasets() -> list:
    """
    Get list of all available dataset names
    
    Returns:
        List of dataset name strings
    """
    return list(DATASETS.keys())


def validate_local_dataset(config: Dict) -> bool:
    """
    Validate that a local dataset's files exist
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    if config["type"] != "local":
        return True  # Only validate local datasets
    
    test_data = Path(config["test_data"])
    audio_dir = Path(config["audio_dir"])
    
    if not test_data.exists():
        return False
    
    if not audio_dir.exists():
        return False
    
    return True


def get_dataset_info(dataset_name: str) -> str:
    """
    Get human-readable information about a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Formatted string with dataset information
    """
    config = get_dataset_config(dataset_name)
    
    info = f"Dataset: {dataset_name}\n"
    info += f"  Description: {config['description']}\n"
    info += f"  Type: {config['type']}\n"
    info += f"  Language: {config['language']}\n"
    
    if config["type"] == "local":
        info += f"  Test data: {config['test_data']}\n"
        info += f"  Audio dir: {config['audio_dir']}\n"
        info += f"  Valid: {validate_local_dataset(config)}\n"
    elif config["type"] == "huggingface":
        info += f"  HF Dataset: {config['hf_dataset']}\n"
        info += f"  Split: {config['hf_split']}\n"
    
    return info

