"""
Training dataset configuration and utilities for ASR training.
Similar to benchmark/common/datasets.py but for training data sources.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os


@dataclass
class TrainingDatasetConfig:
    """Configuration for a training dataset."""
    name: str                    # Short name/id
    path: str                    # Relative path from training_data/
    language: str                # Language code (e.g., "ms", "zh")
    description: str             # Brief description
    source: str                  # HuggingFace repo or data source
    prepare_script: str          # Relative path to prepare script
    estimated_hours: Optional[float] = None  # Estimated duration in hours
    requires_download: bool = True  # Whether data needs to be downloaded


# Training datasets registry
# Paths are relative to asr/train/training_data/
DATASETS: Dict[str, TrainingDatasetConfig] = {
    # Malaysian datasets
    "malaysian-stt": TrainingDatasetConfig(
        name="malaysian-stt",
        path="malaysian-stt",
        language="ms",
        description="Malaysian STT Whisper dataset (Stage 1)",
        source="mesolitica/Malaysian-STT-Whisper",
        prepare_script="malaysian-stt/src/prepare_data.py",
        estimated_hours=1000,  # Approximate
        requires_download=True,
    ),
    "malaysian-stt-stage2": TrainingDatasetConfig(
        name="malaysian-stt-stage2",
        path="malaysian-stt-stage2",
        language="ms",
        description="Malaysian STT Stage 2 (additional data)",
        source="mesolitica/Malaysian-STT-Whisper (stage2)",
        prepare_script="malaysian-stt-stage2/src/prepare_data.py",
        estimated_hours=500,  # Approximate
        requires_download=True,
    ),
    "synthetic-5k": TrainingDatasetConfig(
        name="synthetic-5k",
        path="5k_v3",
        language="ms",
        description="Synthetic Malay data (5k sentences, ElevenLabs TTS)",
        source="Local synthesis (ElevenLabs)",
        prepare_script="5k_v3/src/prepare_synthetic_manifests.py",
        estimated_hours=1.5,  # ~90 minutes of audio
        requires_download=False,  # Already synthesized locally
    ),
    # Chinese datasets
    "chinese-mandarin": TrainingDatasetConfig(
        name="chinese-mandarin",
        path="chinese-mandarin",
        language="zh",
        description="Chinese LiPS (~100h Mandarin)",
        source="BAAI/Chinese-LiPS",
        prepare_script="chinese-mandarin/src/prepare_data.py",
        estimated_hours=100,
        requires_download=True,
    ),
    "kespeech": TrainingDatasetConfig(
        name="kespeech",
        path="KeSpeech",
        language="zh",
        description="KeSpeech (~1500h Mandarin from OpenSLR)",
        source="OpenSLR/KeSpeech",
        prepare_script="KeSpeech/src/prepare_kespeech.py",
        estimated_hours=1500,
        requires_download=False,  # Already downloaded as split archives
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


def get_dataset_config(dataset_name: str) -> TrainingDatasetConfig:
    """
    Get the configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
    
    Returns:
        TrainingDatasetConfig object
    
    Raises:
        KeyError: If dataset_name is not in DATASETS
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {get_dataset_names()}")
    return DATASETS[dataset_name]


def get_dataset_path(dataset_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Get the full path to a dataset directory.
    
    Args:
        dataset_name: Name of the dataset (key in DATASETS)
        base_dir: Base directory for resolving relative paths.
                  If None, returns the relative path as-is.
    
    Returns:
        Path to the dataset directory
    """
    config = get_dataset_config(dataset_name)
    if base_dir is not None:
        return base_dir / config.path
    return Path(config.path)


def check_dataset_prepared(dataset_name: str, training_data_dir: Path) -> bool:
    """
    Check if a dataset has been prepared (has manifest files).
    
    Args:
        dataset_name: Name of the dataset
        training_data_dir: Path to training_data directory
    
    Returns:
        True if the dataset appears to be prepared
    """
    config = get_dataset_config(dataset_name)
    dataset_path = training_data_dir / config.path
    
    # Check for common manifest file patterns
    manifest_patterns = [
        "data/manifests/train_manifest.json",
        "data/processed/manifests/train_manifest.json",
        "manifests/train_manifest.json",
        "output/train_manifest.json",
    ]
    
    for pattern in manifest_patterns:
        if (dataset_path / pattern).exists():
            return True
    
    # Check for stamp file
    stamp_file = training_data_dir / f".{dataset_name}_prepared"
    return stamp_file.exists()


def check_dataset_downloaded(dataset_name: str, training_data_dir: Path) -> bool:
    """
    Check if a dataset has been downloaded (has raw data).
    
    Args:
        dataset_name: Name of the dataset
        training_data_dir: Path to training_data directory
    
    Returns:
        True if the dataset appears to be downloaded
    """
    config = get_dataset_config(dataset_name)
    
    if not config.requires_download:
        # For local/synthetic data, check if source files exist
        dataset_path = training_data_dir / config.path
        return dataset_path.exists() and any(dataset_path.iterdir())
    
    dataset_path = training_data_dir / config.path
    
    # Check for common raw data patterns
    raw_patterns = [
        "data/raw",
        "raw",
        "data",
    ]
    
    for pattern in raw_patterns:
        raw_dir = dataset_path / pattern
        if raw_dir.exists() and any(raw_dir.iterdir()):
            return True
    
    return False


def list_datasets(training_data_dir: Optional[Path] = None, show_status: bool = True) -> None:
    """
    Print a formatted list of all available training datasets.
    
    Args:
        training_data_dir: Path to training_data directory for status checks
        show_status: Whether to show downloaded/prepared status
    """
    print("\n" + "=" * 80)
    print("Available Training Datasets")
    print("=" * 80)
    
    # Header
    if show_status and training_data_dir:
        print(f"{'Name':<22} {'Lang':<5} {'Hours':<8} {'Status':<14} {'Description'}")
        print("-" * 80)
    else:
        print(f"{'Name':<22} {'Lang':<5} {'Hours':<8} {'Description'}")
        print("-" * 80)
    
    # List datasets
    for name, config in DATASETS.items():
        hours_str = f"~{config.estimated_hours:.0f}h" if config.estimated_hours else "?"
        
        if show_status and training_data_dir:
            downloaded = check_dataset_downloaded(name, training_data_dir)
            prepared = check_dataset_prepared(name, training_data_dir)
            
            if prepared:
                status = "✓ ready"
            elif downloaded:
                status = "↓ downloaded"
            else:
                status = "○ not ready"
            
            print(f"{name:<22} {config.language:<5} {hours_str:<8} {status:<14} {config.description}")
        else:
            print(f"{name:<22} {config.language:<5} {hours_str:<8} {config.description}")
    
    print("=" * 80)
    print(f"\nTotal: {len(DATASETS)} datasets")
    
    # Group by language
    languages = sorted(set(c.language for c in DATASETS.values()))
    for lang in languages:
        datasets = get_datasets_by_language(lang)
        total_hours = sum(
            DATASETS[d].estimated_hours or 0 
            for d in datasets
        )
        print(f"  {lang}: {', '.join(datasets)} (~{total_hours:.0f}h)")
    
    # Show sources
    print("\nData Sources:")
    for name, config in DATASETS.items():
        print(f"  {name}: {config.source}")
    print()


def print_dataset_details(dataset_name: str, training_data_dir: Optional[Path] = None) -> None:
    """
    Print detailed information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        training_data_dir: Path to training_data directory for status checks
    """
    config = get_dataset_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {config.name}")
    print(f"{'='*60}")
    print(f"  Language:     {config.language}")
    print(f"  Description:  {config.description}")
    print(f"  Source:       {config.source}")
    print(f"  Est. Hours:   {config.estimated_hours or 'Unknown'}")
    print(f"  Path:         training_data/{config.path}")
    print(f"  Prep Script:  {config.prepare_script}")
    
    if training_data_dir:
        downloaded = check_dataset_downloaded(dataset_name, training_data_dir)
        prepared = check_dataset_prepared(dataset_name, training_data_dir)
        print(f"\n  Status:")
        print(f"    Downloaded: {'✓' if downloaded else '✗'}")
        print(f"    Prepared:   {'✓' if prepared else '✗'}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List available training datasets")
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help="Path to training_data directory (for status checks)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Show details for a specific dataset"
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="Don't show download/prepare status"
    )
    
    args = parser.parse_args()
    
    # Auto-detect training_data_dir if not specified
    if args.training_data_dir is None:
        # Try to find relative to this script
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / "training_data",  # asr/train/training_data
            Path("training_data"),
            Path("../training_data"),
        ]
        for p in possible_paths:
            if p.exists():
                args.training_data_dir = str(p.resolve())
                break
    
    training_data_dir = Path(args.training_data_dir) if args.training_data_dir else None
    
    if args.dataset:
        print_dataset_details(args.dataset, training_data_dir)
    else:
        list_datasets(training_data_dir, show_status=not args.no_status)
