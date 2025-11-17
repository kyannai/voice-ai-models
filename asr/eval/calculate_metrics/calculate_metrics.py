#!/usr/bin/env python3
"""
Metrics calculation script for ASR evaluation
Calculates WER, CER, MER, and RTF metrics from predictions
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging
import string

import pandas as pd
from jiwer import wer, cer, mer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation following standard ASR practices.
    - Lowercase
    - Replace hyphens with spaces (important for Malay reduplication: "laki-laki" → "laki laki")
    - Remove punctuation
    - Normalize whitespace
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Replace hyphens with spaces BEFORE removing punctuation
    # This is critical for Malay where hyphens are used for word reduplication
    # e.g., "laki-laki" (men) should match "laki laki" (not "lakilaki")
    text = text.replace('-', ' ')
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace (multiple spaces to single, strip)
    text = ' '.join(text.split())
    
    return text


class MetricsCalculator:
    """Calculator for ASR evaluation metrics"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        pass
    
    def calculate_metrics(
        self,
        predictions: List[Dict],
        model_name: str = None
    ) -> Dict:
        """
        Calculate all evaluation metrics from predictions
        
        Args:
            predictions: List of prediction dicts with 'reference' and 'hypothesis' keys
            model_name: Optional model name for metadata
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("Calculating metrics with text normalization...")
        logger.info(f"{'='*70}\n")
        
        # Create dataframe
        out_df = pd.DataFrame(predictions)
        
        # Extract references and hypotheses as lists
        refs = out_df["reference"].tolist()
        hyps = out_df["hypothesis"].tolist()
        
        # Apply normalization for overall metrics only
        logger.info("Normalizing text (lowercase, remove punctuation, normalize whitespace)...")
        refs_normalized = [normalize_text(ref) for ref in refs]
        hyps_normalized = [normalize_text(hyp) for hyp in hyps]
        
        # Calculate WER, CER, and MER using normalized text
        metrics = {}
        metrics["WER"] = round(wer(refs_normalized, hyps_normalized), 4)
        metrics["CER"] = round(cer(refs_normalized, hyps_normalized), 4)
        metrics["MER"] = round(mer(refs_normalized, hyps_normalized), 4)
        
        print(f"\n[METRICS] WER={metrics['WER']:.4f}, CER={metrics['CER']:.4f}, MER={metrics['MER']:.4f}")
        
        # Calculate per-sample metrics (WER, CER, MER)
        logger.info("\nCalculating per-sample metrics...")
        for prediction in tqdm(predictions, desc="Per-sample metrics", unit="sample"):
            ref = prediction["reference"]
            hyp = prediction["hypothesis"]
            
            # Calculate per-sample WER, CER, and MER
            try:
                prediction["wer"] = round(wer([ref], [hyp]) * 100, 2)  # As percentage
            except:
                prediction["wer"] = None
                
            try:
                prediction["cer"] = round(cer([ref], [hyp]) * 100, 2)  # As percentage
            except:
                prediction["cer"] = None
                
            try:
                prediction["mer"] = round(mer([ref], [hyp]) * 100, 2)  # As percentage
            except:
                prediction["mer"] = None
        
        # Calculate timing statistics
        total_audio = sum(p.get("audio_duration", 0) for p in predictions)
        total_processing = sum(p.get("processing_time", 0) for p in predictions)
        avg_rtf = total_processing / total_audio if total_audio > 0 else 0
        
        # Compile results
        results = {
            "model": model_name,
            "num_samples": len(predictions),
            "wer": metrics["WER"],
            "cer": metrics["CER"],
            "mer": metrics["MER"],
            "timing": {
                "total_audio_duration": total_audio,
                "total_processing_time": total_processing,
                "average_rtf": avg_rtf,
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
        
        print(f"\nWER: {results['wer']:.4f}")
        print(f"CER: {results['cer']:.4f}")
        print(f"MER: {results['mer']:.4f}")
        
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
        description="Calculate evaluation metrics from ASR predictions"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_data = load_predictions(Path(args.predictions))
    
    # Initialize calculator
    calculator = MetricsCalculator()
    
    # Calculate metrics (no normalization)
    results = calculator.calculate_metrics(
        predictions=predictions_data["predictions"],
        model_name=predictions_data.get("model")
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
        "wer": [results["wer"]],
        "cer": [results["cer"]],
        "mer": [results["mer"]],
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