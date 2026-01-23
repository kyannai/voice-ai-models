"""
Common utilities for ASR benchmark scripts.

This module provides shared functionality across all benchmark scripts:
- Dataset loading and configuration
- Text normalization for Malay language
- WER evaluation utilities
- Common helper functions
"""

from .datasets import (
    DATASETS,
    DatasetConfig,
    get_dataset_config,
    get_dataset_language,
    get_dataset_names,
    get_dataset_path,
    get_datasets_by_language,
    list_datasets,
    load_dataset,
)
from .normalizer import (
    get_normalizer,
    postprocess_text_mal,
    normalize_superscripts,
)
from .evaluation import (
    compute_wer,
    print_results,
    run_evaluation,
)
from .utils import split_dict
from .api_config import resolve_api_config

__all__ = [
    # Datasets
    "DATASETS",
    "DatasetConfig",
    "get_dataset_config",
    "get_dataset_language",
    "get_dataset_names",
    "get_dataset_path",
    "get_datasets_by_language",
    "list_datasets",
    "load_dataset",
    # Normalizer
    "get_normalizer",
    "postprocess_text_mal",
    "normalize_superscripts",
    # Evaluation
    "compute_wer",
    "print_results",
    "run_evaluation",
    # Utils
    "split_dict",
    # API Config
    "resolve_api_config",
]
