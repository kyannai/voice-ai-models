#!/usr/bin/env python3
"""
Batch evaluation script for multiple ASR models
Evaluates multiple models on the same test dataset and generates comparison report
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

from datasets_config import list_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
MODELS = {
    "whisper-large-v3-turbo": {
        "model": "openai/whisper-large-v3-turbo",
        "family": "whisper",
        "description": "OpenAI Whisper Large V3 Turbo"
    },
    "malaysian-whisper-large-v3-turbo-v3": {
        "model": "mesolitica/Malaysian-whisper-large-v3-turbo-v3",
        "family": "whisper",
        "description": "Malaysian Whisper Large V3 Turbo"
    },
    "qwen2-audio-7b": {
        "model": "Qwen/Qwen2-Audio-7B-Instruct",
        "family": "qwen2audio",
        "description": "Qwen2-Audio 7B Instruct"
    },
    "qwen25-omni-7b": {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "family": "qwen25omni",
        "description": "Qwen2.5-Omni 7B"
    },
    "parakeet-tdt-0.6b": {
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "family": "parakeet",
        "description": "NVIDIA Parakeet TDT 0.6B V3"
    },
    "final_model.nemo": {
        "model": "../train/train_parakeet_tdt/outputs.bak/parakeet-tdt-malay-asr/final_model.nemo",
        "family": "parakeet",
        "description": "Fine-tuned Parakeet TDT 0.6B (Malay)"
    }
}


def find_model_output_dir(model_key: str, outputs_dir: Path) -> Path:
    """
    Find the most recent output directory for a model (case-insensitive search)
    
    Args:
        model_key: Model identifier key
        outputs_dir: Base outputs directory
        
    Returns:
        Most recent output directory Path, or None if not found
    """
    # Try exact match first - look for pattern _model_key_ or _model_key. to avoid partial matches
    # This prevents "whisper-large-v3-turbo" from matching "Malaysian-whisper-large-v3-turbo-v3"
    model_outputs = sorted(
        [d for d in outputs_dir.rglob("*") if d.is_dir() and 
         (f"_{model_key}_" in d.name or f"_{model_key}." in d.name or d.name.endswith(f"_{model_key}"))],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if model_outputs:
        return model_outputs[0]
    
    # Fallback: Try case-insensitive substring match but check it's not a partial match
    model_key_lower = model_key.lower()
    all_dirs = [d for d in outputs_dir.rglob("*") if d.is_dir()]
    matching_dirs = []
    
    for d in all_dirs:
        name_lower = d.name.lower()
        # Check if model_key is in the name with underscore boundaries or at the end
        if (f"_{model_key_lower}_" in name_lower or 
            f"_{model_key_lower}." in name_lower or 
            name_lower.endswith(f"_{model_key_lower}")):
            matching_dirs.append(d)
    
    if matching_dirs:
        # Return most recent
        return sorted(matching_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    return None


def run_evaluation(model_key: str, model_config: dict, test_dataset: str, 
                   device: str = "auto", language: str = "ms") -> dict:
    """
    Run evaluation for a single model
    
    Args:
        model_key: Short name for the model
        model_config: Model configuration dict
        test_dataset: Dataset name from registry
        device: Device to use (auto/cuda/cpu)
        language: Language code
        
    Returns:
        Dictionary with evaluation results path and status
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"Evaluating: {model_config['description']}")
    logger.info(f"Model: {model_config['model']}")
    logger.info("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "evaluate.py",
        "--model", model_config["model"],
        "--test-dataset", test_dataset,
        "--device", device,
        "--language", language
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # Find the output directory (most recent for this model)
        # Search recursively in outputs/ and outputs/outputs/
        outputs_dir = Path("outputs")
        output_dir = find_model_output_dir(model_key, outputs_dir)
        
        if output_dir:
            results_file = output_dir / "evaluation_results.json"
            
            if results_file.exists():
                logger.info(f"✓ Evaluation completed: {output_dir}")
                return {
                    "status": "success",
                    "output_dir": str(output_dir),
                    "results_file": str(results_file),
                    "model_key": model_key,
                    "model": model_config["model"],
                    "description": model_config["description"]
                }
        
        logger.warning(f"⚠ Results file not found for {model_key}")
        return {
            "status": "incomplete",
            "model_key": model_key,
            "error": "Results file not found"
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Evaluation failed for {model_key}: {e}")
        return {
            "status": "failed",
            "model_key": model_key,
            "error": str(e)
        }


def merge_results(evaluation_results: list, output_file: str):
    """
    Merge all evaluation results into a single comparison file
    
    Args:
        evaluation_results: List of evaluation result dicts
        output_file: Path to output merged JSON file
    """
    logger.info("")
    logger.info("="*80)
    logger.info("Merging results for side-by-side comparison")
    logger.info("="*80)
    
    # Load all results
    all_results = {}
    overall_metrics = {}
    
    for eval_result in evaluation_results:
        if eval_result["status"] != "success":
            continue
            
        model_key = eval_result["model_key"]
        results_file = eval_result["results_file"]
        
        logger.info(f"Loading results for {model_key}...")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_results[model_key] = data
        
        # Store overall metrics
        overall_metrics[model_key] = {
            "model": eval_result["model"],
            "description": eval_result["description"],
            "wer": data.get("wer"),
            "cer": data.get("cer"),
            "mer": data.get("mer"),
            "num_samples": data.get("num_samples"),
            "avg_rtf": data["timing"].get("average_rtf") if "timing" in data else None
        }
    
    # Create side-by-side comparison
    logger.info("Creating side-by-side comparison...")
    
    # Get reference model (first successful one)
    reference_model = None
    for model_key in all_results:
        if all_results[model_key].get("predictions"):
            reference_model = model_key
            break
    
    if not reference_model:
        logger.error("No successful evaluations found!")
        return
    
    # Build comparison structure
    comparison = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "num_models": len(all_results),
            "models": list(all_results.keys()),
            "num_samples": all_results[reference_model]["num_samples"]
        },
        "overall_metrics": overall_metrics,
        "samples": []
    }
    
    # Merge predictions sample by sample
    predictions = all_results[reference_model]["predictions"]
    
    for i, pred in enumerate(predictions):
        sample = {
            "sample_id": i,
            "audio_path": pred["audio_path"],
            "reference": pred["reference"],
            "audio_duration": pred.get("audio_duration"),
            "hypotheses": {},
            "metrics": {}
        }
        
        # Add hypothesis and metrics for each model
        for model_key, results in all_results.items():
            if i < len(results["predictions"]):
                model_pred = results["predictions"][i]
                
                sample["hypotheses"][model_key] = model_pred["hypothesis"]
                sample["metrics"][model_key] = {
                    "wer": model_pred.get("wer"),
                    "cer": model_pred.get("cer"),
                    "mer": model_pred.get("mer"),
                    "rtf": model_pred.get("rtf"),
                    "processing_time": model_pred.get("processing_time")
                }
        
        comparison["samples"].append(sample)
    
    # Save merged results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Merged results saved to: {output_path}")
    
    # Print summary
    print_comparison_summary(overall_metrics)


def print_comparison_summary(metrics: dict):
    """Print summary table of all models"""
    logger.info("")
    logger.info("="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info("")
    
    # Header
    print(f"{'Model':<30} {'WER':<10} {'CER':<10} {'MER':<10} {'RTF':<10}")
    print("-" * 70)
    
    # Sort by WER (best first)
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get("wer", float('inf')))
    
    for model_key, model_metrics in sorted_models:
        wer = model_metrics.get("wer", 0)
        cer = model_metrics.get("cer", 0)
        mer = model_metrics.get("mer", 0)
        rtf = model_metrics.get("avg_rtf", 0)
        
        print(f"{model_key:<30} {wer:<10.4f} {cer:<10.4f} {mer:<10.4f} {rtf:<10.3f}")
    
    logger.info("")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of multiple ASR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(list_datasets())}

Examples:
  # Evaluate all models on meso-malaya-test
  python batch_evaluate.py --test-dataset meso-malaya-test

  # Evaluate specific models on ytl-malay-test
  python batch_evaluate.py --test-dataset ytl-malay-test --models whisper-large-v3-turbo parakeet-tdt-0.6b
"""
    )
    
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        help="Test dataset name from registry"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation (default: auto)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="ms",
        help="Language code (default: ms for Malay)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model keys to evaluate (default: all models)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/batch_comparison.json",
        help="Output file for merged comparison (default: outputs/batch_comparison.json)"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only merge existing results"
    )
    
    args = parser.parse_args()
    
    # Determine which models to evaluate
    models_to_eval = args.models if args.models else list(MODELS.keys())
    
    logger.info("")
    logger.info("="*80)
    logger.info("BATCH ASR MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Test dataset: {args.test_dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Models to evaluate: {', '.join(models_to_eval)}")
    logger.info("="*80)
    
    evaluation_results = []
    
    # Run evaluations
    if not args.skip_eval:
        for model_key in models_to_eval:
            if model_key not in MODELS:
                logger.warning(f"Unknown model key: {model_key}, skipping...")
                continue
            
            result = run_evaluation(
                model_key=model_key,
                model_config=MODELS[model_key],
                test_dataset=args.test_dataset,
                device=args.device,
                language=args.language
            )
            
            evaluation_results.append(result)
    else:
        logger.info("Skipping evaluation, loading existing results...")
        
        # Find existing results (search recursively)
        outputs_dir = Path("outputs")
        for model_key in models_to_eval:
            if model_key not in MODELS:
                continue
                
            output_dir = find_model_output_dir(model_key, outputs_dir)
            
            if output_dir:
                results_file = output_dir / "evaluation_results.json"
                
                if results_file.exists():
                    evaluation_results.append({
                        "status": "success",
                        "output_dir": str(output_dir),
                        "results_file": str(results_file),
                        "model_key": model_key,
                        "model": MODELS[model_key]["model"],
                        "description": MODELS[model_key]["description"]
                    })
    
    # Merge results
    if evaluation_results:
        merge_results(evaluation_results, args.output)
    else:
        logger.error("No evaluation results to merge!")
        return 1
    
    logger.info("")
    logger.info("="*80)
    logger.info("✅ BATCH EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Merged comparison: {args.output}")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

