#!/usr/bin/env python3
"""
Dataset Registry Configuration for TTS Evaluation

Reuses ASR test datasets for TTS evaluation.
The same text data used for ASR ground truth becomes input for TTS synthesis.

Target Languages:
- English (en)
- Malay (ms)
- Code-switching / Mixed (en-ms) - English and Malay mixed in one sentence

Usage:
    from datasets_config import list_datasets, get_dataset_config
    
    config = get_dataset_config("meso-malaya-test")
    print(config["type"])  # "local"
    print(config["test_data"])  # Path to JSON file
"""

from pathlib import Path
from typing import Dict, List, Optional

# Base directory for this file (tts/eval)
EVAL_DIR = Path(__file__).parent

# ASR eval directory (for shared test data)
ASR_EVAL_DIR = EVAL_DIR.parent.parent / "asr" / "eval"


# Supported languages for TTS
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ms": "Malay",
    "en-ms": "Code-switching (English-Malay mixed)",
}


# Dataset Registry - mirrors ASR datasets
# Text from these datasets will be used for TTS synthesis
# Language types:
#   - "ms": Pure Malay
#   - "en": Pure English
#   - "en-ms": Code-switching (mixed English-Malay in same sentence)
DATASETS = {
    "meso-malaya-test": {
        "type": "local",
        "description": "Malaya Malay test set from mesolitica (Malay text for TTS)",
        "test_data": ASR_EVAL_DIR / "test_data/malaya-test/malaya-malay-test-set.json",
        "audio_dir": ASR_EVAL_DIR / "test_data/malaya-test",  # Reference audio if needed
        "language": "ms",  # Pure Malay
        "num_samples": None,
    },
    
    "ytl-malay-test": {
        "type": "local",
        "description": "YTL internal Malay test set (may contain code-switching)",
        "test_data": ASR_EVAL_DIR / "test_data/ytl-malay-test/asr_ground_truths.json",
        "audio_dir": ASR_EVAL_DIR / "test_data/ytl-malay-test",
        "language": "en-ms",  # Likely contains code-switching
        "num_samples": None,
    },
    
    "seacrowd-asr-malcsc": {
        "type": "local",
        "description": "SEACrowd ASR MALCSC dataset (Malay conversational, may have code-switching)",
        "test_data": ASR_EVAL_DIR / "test_data/seacrowd-malcsc/seacrowd_malcsc.json",
        "audio_dir": ASR_EVAL_DIR / "test_data/seacrowd-malcsc",
        "language": "en-ms",  # Conversational often has code-switching
        "num_samples": 20,
    },
    
    "fleurs-test": {
        "type": "local",
        "description": "FLEURS Malay test set (formal Malay text)",
        "test_data": ASR_EVAL_DIR / "test_data/YTL_testsets/fleurs_test.tsv",
        "audio_dir": ASR_EVAL_DIR / "test_data/YTL_testsets",
        "language": "ms",  # Formal Malay
        "num_samples": None,
    },
    
    "malay-scripted": {
        "type": "local",
        "description": "YTL Malay scripted speech test set (formal Malay)",
        "test_data": ASR_EVAL_DIR / "test_data/YTL_testsets/malay_scripted_meta.tsv",
        "audio_dir": ASR_EVAL_DIR / "test_data/YTL_testsets",
        "language": "ms",  # Scripted = formal Malay
        "num_samples": None,
    },
    
    "malay-conversational": {
        "type": "local",
        "description": "YTL Malay conversational speech (likely contains code-switching)",
        "test_data": ASR_EVAL_DIR / "test_data/YTL_testsets/malay_conversational_meta.tsv",
        "audio_dir": ASR_EVAL_DIR / "test_data/YTL_testsets",
        "language": "en-ms",  # Conversational often has code-switching
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
    
    return DATASETS[dataset_name].copy()


def list_datasets() -> List[str]:
    """
    Get list of all available dataset names
    
    Returns:
        List of dataset name strings
    """
    return list(DATASETS.keys())


def validate_dataset(dataset_name: str) -> bool:
    """
    Validate that a dataset's files exist
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if valid, False otherwise
    """
    try:
        config = get_dataset_config(dataset_name)
    except ValueError:
        return False
    
    if config["type"] != "local":
        return True
    
    test_data = Path(config["test_data"])
    
    if not test_data.exists():
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
        info += f"  Valid: {validate_dataset(dataset_name)}\n"
    
    return info


def get_reference_audio_dir(dataset_name: str) -> Optional[Path]:
    """
    Get the reference audio directory for a dataset.
    This can be used for speaker cloning in TTS models.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to audio directory, or None if not available
    """
    config = get_dataset_config(dataset_name)
    audio_dir = config.get("audio_dir")
    
    if audio_dir and Path(audio_dir).exists():
        return Path(audio_dir)
    
    return None


def get_language_type(dataset_name: str) -> str:
    """
    Get the language type for a dataset.
    
    Returns:
        - "en": Pure English
        - "ms": Pure Malay
        - "en-ms": Code-switching (mixed English-Malay)
    """
    config = get_dataset_config(dataset_name)
    return config.get("language", "en-ms")


def is_code_switching(dataset_name: str) -> bool:
    """
    Check if dataset contains code-switching (mixed English-Malay).
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if dataset contains code-switching
    """
    lang = get_language_type(dataset_name)
    return lang == "en-ms"


def list_datasets_by_language(language: str) -> List[str]:
    """
    Get list of datasets for a specific language.
    
    Args:
        language: Language code ("en", "ms", or "en-ms")
        
    Returns:
        List of dataset names
    """
    return [
        name for name, config in DATASETS.items()
        if config.get("language") == language
    ]

