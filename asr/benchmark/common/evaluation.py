"""
Evaluation utilities for ASR benchmarks.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from torchmetrics.text import WordErrorRate


def compute_wer(
    hypotheses: Dict[str, str],
    references: Dict[str, str],
    sample_interval: int = 50,
    verbose: bool = True,
) -> float:
    """
    Compute Word Error Rate between hypotheses and references.
    
    Args:
        hypotheses: Dict mapping utterance_id to hypothesis text
        references: Dict mapping utterance_id to reference text
        sample_interval: Print sample every N utterances (0 to disable)
        verbose: Whether to print sample comparisons
        
    Returns:
        WER as a float (0.0 to 1.0+)
    """
    wer_metric = WordErrorRate()
    count = 0
    
    for utt_id, hyp in hypotheses.items():
        if utt_id not in references:
            continue
            
        ref = references[utt_id]
        
        if verbose and sample_interval > 0 and count % sample_interval == 0:
            print(f"utt {utt_id}")
            print(f"  hyp: {hyp}")
            print(f"  ref: {ref}")
        
        wer_metric.update([hyp], [ref])
        count += 1
    
    return float(wer_metric.compute())


def load_json_results(results_dir: str | Path) -> Dict[str, str]:
    """
    Load transcription results from JSON files in a directory.
    
    Each JSON file should have format: {"text": "...", "text_norm": "..."}
    
    Args:
        results_dir: Directory containing JSON result files
        
    Returns:
        Dict mapping utterance_id (filename stem) to normalized text
    """
    results_dir = Path(results_dir)
    results = {}
    
    for json_path in results_dir.glob("*.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results[json_path.stem] = data.get("text_norm", "")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    return results


def run_evaluation(
    results_dir: str | Path,
    references: Dict[str, str],
    dataset_name: str,
    sample_interval: int = 50,
) -> float:
    """
    Run WER evaluation on saved JSON results.
    
    Args:
        results_dir: Directory containing JSON result files
        references: Dict mapping utterance_id to reference text
        dataset_name: Name of dataset (for logging)
        sample_interval: Print sample every N utterances
        
    Returns:
        WER as a float
    """
    hypotheses = load_json_results(results_dir)
    wer = compute_wer(hypotheses, references, sample_interval=sample_interval)
    print(f"Final WER for {dataset_name}: {wer:.3f}")
    return wer


def print_results(all_wers: Dict[str, float]) -> None:
    """
    Print formatted benchmark results.
    
    Args:
        all_wers: Dict mapping dataset name to WER value
    """
    print(f"\n{'=' * 60}")
    print("All Results:")
    print('=' * 60)
    for dataset_name, wer_value in all_wers.items():
        print(f"  {dataset_name}: {wer_value:.3f}")
    print()
