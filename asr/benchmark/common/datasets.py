"""
Dataset configuration and loading utilities for ASR benchmarks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    path: str        # Relative path to TSV file
    language: str    # Language code (e.g., "ms", "en", "zh")
    description: str = ""  # Brief description of the dataset


# Standard benchmark datasets
# Paths are relative to the script location (e.g., api/, parakeet/, whisper/)
DATASETS = {
    # Malay datasets
    "fleurs_test": DatasetConfig(
        path="../test_data/YTL_testsets/fleurs_test.tsv",
        language="ms",
        description="Google FLEURS Malay test set",
    ),
    "malay_conversational": DatasetConfig(
        path="../test_data/YTL_testsets/malay_conversational_meta.tsv",
        language="ms",
        description="Conversational Malay speech",
    ),
    "malay_scripted": DatasetConfig(
        path="../test_data/YTL_testsets/malay_scripted_meta.tsv",
        language="ms",
        description="Scripted/read Malay speech",
    ),
    # Chinese datasets
    "kespeech": DatasetConfig(
        path="../test_data/kespeech/kespeech_test.tsv",
        language="zh",
        description="KeSpeech Mandarin (HuggingFace: TwinkStart/KeSpeech)",
    ),
    "chinese_lips": DatasetConfig(
        path="../test_data/chinese_lips/chinese_lips_test.tsv",
        language="zh",
        description="Chinese-LiPS (HuggingFace: BAAI/Chinese-LiPS)",
    ),
    # Banking voice assistant dataset
    "supa": DatasetConfig(
        path="../test_data/supa/supa_test.tsv",
        language="ms",
        description="SUPA Banking Voice Assistant (Malay/English code-switching)",
    ),
}


def get_dataset_names() -> List[str]:
    """Get list of available dataset names."""
    return list(DATASETS.keys())


def get_datasets_by_language(language: str) -> List[str]:
    """
    Get list of dataset names filtered by language.
    
    Args:
        language: Language code (e.g., "ms", "zh")
    
    Returns:
        List of dataset names with matching language
    """
    return [name for name, config in DATASETS.items() if config.language == language]


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get the configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
    
    Returns:
        DatasetConfig object
    
    Raises:
        KeyError: If dataset_name is not in DATASETS
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {get_dataset_names()}")
    return DATASETS[dataset_name]


def get_dataset_path(dataset_name: str, base_dir: Path | None = None) -> Path:
    """
    Get the full path to a dataset TSV file.
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
        base_dir: Base directory for resolving relative paths.
                  If None, returns the relative path as-is.
    
    Returns:
        Path to the dataset TSV file
    
    Raises:
        KeyError: If dataset_name is not in DATASETS
    """
    config = get_dataset_config(dataset_name)
    if base_dir is not None:
        return base_dir / config.path
    return Path(config.path)


def get_dataset_language(dataset_name: str) -> str:
    """
    Get the language code for a dataset.
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
    
    Returns:
        Language code (e.g., "ms", "en", "zh")
    
    Raises:
        KeyError: If dataset_name is not in DATASETS
    """
    return get_dataset_config(dataset_name).language


def list_datasets() -> None:
    """Print a formatted list of all available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 70)
    print(f"{'Name':<25} {'Language':<10} {'Description'}")
    print("-" * 70)
    for name, config in DATASETS.items():
        print(f"{name:<25} {config.language:<10} {config.description}")
    print("=" * 70)
    print(f"\nTotal: {len(DATASETS)} datasets")
    
    # Group by language
    languages = sorted(set(c.language for c in DATASETS.values()))
    for lang in languages:
        datasets = get_datasets_by_language(lang)
        print(f"  {lang}: {', '.join(datasets)}")
    print()


def load_dataset(dataset_path: str | Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    """
    Load a dataset TSV file and return audio paths and reference transcripts.
    
    Args:
        dataset_path: Path to the TSV file
    
    Returns:
        Tuple of:
        - audio_dict: Dict mapping utterance_id to audio file path
        - ref_transcript: Dict mapping utterance_id to reference transcript
        - duration_dict: Dict mapping utterance_id to duration in seconds
    
    Raises:
        AssertionError: If utterance_ids are not unique
    """
    dataset_path = Path(dataset_path)
    tsv_dir = dataset_path.parent
    
    all_data = pd.read_csv(dataset_path, sep='\t')
    
    # Validate unique utterance IDs
    assert len(all_data['utterance_id'].tolist()) == all_data['utterance_id'].nunique(), \
        f"Duplicate utterance_ids found in {dataset_path}"
    
    audio_dict = {}
    ref_transcript = {}
    duration_dict = {}
    
    for idx, row in all_data.iterrows():
        audio_filepath = tsv_dir / row["path"]
        utt_id = str(Path(audio_filepath).stem)
        audio_dict[utt_id] = str(audio_filepath)
        ref_transcript[utt_id] = row["sentence"]
        duration_dict[utt_id] = float(row["duration"])
    
    return audio_dict, ref_transcript, duration_dict


if __name__ == "__main__":
    # Allow running this module directly to list datasets
    list_datasets()
