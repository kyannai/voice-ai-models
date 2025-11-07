#!/usr/bin/env python3
"""
Metrics calculation script for Whisper ASR evaluation
Calculates WER, CER, RTF, and Malaysian-specific metrics from predictions
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging
import re
import string

import numpy as np
import pandas as pd
import jiwer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison by:
    - Converting to lowercase
    - Converting hyphens to spaces (e.g., "siap-siap" → "siap siap")
    - Removing punctuation
    - Normalizing whitespace
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphens with spaces (to preserve word boundaries)
    # This handles cases like "siap-siap" → "siap siap"
    text = text.replace('-', ' ')
    
    # Remove punctuation (except hyphens which are already handled)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace (remove extra spaces, tabs, newlines)
    text = ' '.join(text.split())
    
    return text.strip()


class MetricsCalculator:
    """Calculator for ASR evaluation metrics"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        pass
    
    def calculate_wer(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict:
        """
        Calculate Word Error Rate with detailed breakdown
        
        Args:
            references: List of reference transcriptions
            hypotheses: List of hypothesis transcriptions
            
        Returns:
            Dictionary with WER and detailed metrics
        """
        # Calculate WER
        wer_score = jiwer.wer(references, hypotheses)
        
        # Get detailed measures using jiwer 4.x API
        try:
            # Try jiwer 4.x+ API
            from jiwer import process_words
            output = process_words(references, hypotheses)
            
            return {
                "wer": wer_score * 100,  # As percentage
                "substitutions": output.substitutions,
                "insertions": output.insertions,
                "deletions": output.deletions,
                "hits": output.hits,
                "total_words": output.substitutions + output.deletions + output.hits,
            }
        except (ImportError, AttributeError):
            # Fallback for jiwer 3.x
            measures = jiwer.compute_measures(references, hypotheses)
            
            return {
                "wer": wer_score * 100,  # As percentage
                "substitutions": measures["substitutions"],
                "insertions": measures["insertions"],
                "deletions": measures["deletions"],
                "hits": measures["hits"],
                "total_words": measures["substitutions"] + measures["deletions"] + measures["hits"],
            }
    
    def calculate_cer(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            references: List of reference transcriptions
            hypotheses: List of hypothesis transcriptions
            
        Returns:
            CER as percentage
        """
        cer = jiwer.cer(references, hypotheses)
        return cer * 100
    
    def calculate_per_sample_metrics(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict:
        """
        Calculate WER and CER for a single sample
        
        Args:
            reference: Reference transcription
            hypothesis: Hypothesis transcription
            
        Returns:
            Dictionary with per-sample metrics
        """
        # Handle empty strings
        if not reference.strip():
            return {
                "wer": None,
                "cer": None,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "hits": 0,
                "total_words": 0,
            }
        
        # Calculate single-sample WER
        try:
            wer_score = jiwer.wer([reference], [hypothesis]) * 100
        except Exception as e:
            logger.warning(f"Could not calculate WER: {e}")
            wer_score = None
        
        # Calculate single-sample CER
        try:
            cer_score = jiwer.cer([reference], [hypothesis]) * 100
        except Exception as e:
            logger.warning(f"Could not calculate CER: {e}")
            cer_score = None
        
        # Calculate word-level measures
        try:
            from jiwer import process_words
            output = process_words([reference], [hypothesis])
            
            return {
                "wer": round(wer_score, 2) if wer_score is not None else None,
                "cer": round(cer_score, 2) if cer_score is not None else None,
                "substitutions": output.substitutions,
                "insertions": output.insertions,
                "deletions": output.deletions,
                "hits": output.hits,
                "total_words": output.substitutions + output.deletions + output.hits,
            }
        except (ImportError, AttributeError):
            # Fallback for jiwer 3.x or if process_words not available
            return {
                "wer": round(wer_score, 2) if wer_score is not None else None,
                "cer": round(cer_score, 2) if cer_score is not None else None,
            }
    
    def calculate_metrics(
        self,
        predictions: List[Dict],
        model_name: str = None,
        normalize: bool = True
    ) -> Dict:
        """
        Calculate all evaluation metrics from predictions
        
        Args:
            predictions: List of prediction dicts with 'reference' and 'hypothesis' keys
            model_name: Optional model name for metadata
            normalize: Whether to normalize text (lowercase, remove punctuation) before metrics
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("Calculating metrics...")
        if normalize:
            logger.info("Text normalization: ENABLED (lowercase, no punctuation)")
        else:
            logger.info("Text normalization: DISABLED (case-sensitive, with punctuation)")
        logger.info(f"{'='*70}\n")
        
        # Extract references and hypotheses
        references = [p["reference"] for p in predictions]
        hypotheses = [p["hypothesis"] for p in predictions]
        
        # Normalize if requested
        if normalize:
            logger.info("Normalizing text for metrics calculation...")
            references_normalized = [normalize_text(ref) for ref in references]
            hypotheses_normalized = [normalize_text(hyp) for hyp in hypotheses]
        else:
            references_normalized = references
            hypotheses_normalized = hypotheses
        
        # Calculate overall metrics
        wer_results = self.calculate_wer(references_normalized, hypotheses_normalized)
        logger.info(f"✓ Overall WER calculated: {wer_results['wer']:.2f}%")
        
        cer = self.calculate_cer(references_normalized, hypotheses_normalized)
        logger.info(f"✓ Overall CER calculated: {cer:.2f}%")
        
        # Calculate per-sample metrics
        logger.info("Calculating per-sample metrics...")
        for prediction in tqdm(predictions, desc="Per-sample metrics", unit="sample"):
            ref = prediction["reference"]
            hyp = prediction["hypothesis"]
            
            # Normalize for per-sample metrics if requested
            if normalize:
                ref_norm = normalize_text(ref)
                hyp_norm = normalize_text(hyp)
            else:
                ref_norm = ref
                hyp_norm = hyp
            
            # Add per-sample metrics to each prediction
            sample_metrics = self.calculate_per_sample_metrics(ref_norm, hyp_norm)
            prediction.update(sample_metrics)
        
        logger.info(f"✓ Per-sample metrics calculated for {len(predictions)} samples")
        
        # Calculate timing statistics
        total_audio = sum(p.get("audio_duration", 0) for p in predictions)
        total_processing = sum(p.get("processing_time", 0) for p in predictions)
        avg_rtf = total_processing / total_audio if total_audio > 0 else 0
        
        # Compile results
        results = {
            "model": model_name,
            "num_samples": len(predictions),
            "normalized": normalize,  # Indicate if text was normalized
            "wer": wer_results,
            "cer": cer,
            "timing": {
                "total_audio_duration": total_audio,
                "total_processing_time": total_processing,
                "average_rtf": avg_rtf,
                "per_sample_rtf": [p.get("rtf", 0) for p in predictions],
            },
            "predictions": predictions
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print(f"EVALUATION SUMMARY: {results.get('model', 'Unknown')}")
        print("=" * 70)
        
        print(f"\nDataset: {results['num_samples']} samples")
        print(f"Text Normalization: {'ENABLED' if results.get('normalized', True) else 'DISABLED'}")
        
        print("\n--- Word Error Rate ---")
        wer = results["wer"]
        print(f"WER: {wer['wer']:.2f}%")
        print(f"  Substitutions: {wer['substitutions']}")
        print(f"  Insertions: {wer['insertions']}")
        print(f"  Deletions: {wer['deletions']}")
        print(f"  Hits: {wer['hits']}")
        print(f"  Total words: {wer['total_words']}")
        
        print(f"\n--- Character Error Rate ---")
        print(f"CER: {results['cer']:.2f}%")
        
        print("\n--- Performance ---")
        timing = results["timing"]
        print(f"Total audio duration: {timing['total_audio_duration']:.2f}s")
        print(f"Total processing time: {timing['total_processing_time']:.2f}s")
        print(f"Average RTF: {timing['average_rtf']:.3f}")
        print(f"  (RTF < 1.0 = faster than real-time)")
        
        print("=" * 70)


def load_predictions(predictions_file: Path) -> Dict:
    """
    Load predictions from JSON file
    
    Args:
        predictions_file: Path to predictions JSON file
        
    Returns:
        Dictionary with predictions and metadata
    """
    logger.info(f"Loading predictions from: {predictions_file}")
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data.get('predictions', []))} predictions")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics from Whisper predictions"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file (from transcribe_whisper.py)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (keep case and punctuation). Default: normalize text"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_data = load_predictions(Path(args.predictions))
    
    # Initialize calculator
    calculator = MetricsCalculator()
    
    # Calculate metrics
    results = calculator.calculate_metrics(
        predictions=predictions_data["predictions"],
        model_name=predictions_data.get("model"),
        normalize=not args.no_normalize  # Normalize by default unless --no-normalize is set
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved evaluation results to {results_file}")
    
    # Save summary as CSV (without full predictions)
    summary_file = output_dir / "evaluation_summary.csv"
    summary_data = {
        "model": [results.get("model")],
        "num_samples": [results["num_samples"]],
        "wer": [results["wer"]["wer"]],
        "cer": [results["cer"]],
        "substitutions": [results["wer"]["substitutions"]],
        "insertions": [results["wer"]["insertions"]],
        "deletions": [results["wer"]["deletions"]],
        "hits": [results["wer"]["hits"]],
        "total_words": [results["wer"]["total_words"]],
        "avg_rtf": [results["timing"]["average_rtf"]],
        "total_audio_duration": [results["timing"]["total_audio_duration"]],
        "total_processing_time": [results["timing"]["total_processing_time"]],
    }
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    logger.info(f"Saved evaluation summary to {summary_file}")
    
    # Print summary
    calculator.print_summary(results)


if __name__ == "__main__":
    main()

# python calculate_metrics.py   --predictions ./results_cuda/predictions.json   --output-dir ./results