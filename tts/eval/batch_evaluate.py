#!/usr/bin/env python3
"""
Batch TTS Evaluation Script

Evaluate multiple TTS models on a dataset and generate comparison report.

Usage:
    python batch_evaluate.py \
        --models xtts-v2,kokoro,melotts,glm-tts \
        --test-dataset meso-malaya-test \
        --output-dir outputs/comparison

    python batch_evaluate.py \
        --config batch_config.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from datasets_config import list_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "xtts-v2": {
        "model": "xtts-v2",
        "display_name": "XTTS v2",
        "language": "en",
    },
    "kokoro": {
        "model": "kokoro",
        "display_name": "Kokoro TTS",
        "voice": "af_heart",
        "language": "en-us",
    },
    "melotts": {
        "model": "melotts",
        "display_name": "MeloTTS",
        "speaker": "EN-US",
        "language": "EN",
    },
    "glm-tts": {
        "model": "glm-tts",
        "display_name": "GLM-TTS",
        "use_fallback": True,  # Use edge-tts fallback by default
    },
}


class BatchTTSEvaluator:
    """Batch evaluator for comparing multiple TTS models"""
    
    def __init__(
        self,
        models: List[str],
        test_dataset: str,
        output_dir: Path,
        asr_model: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        max_samples: Optional[int] = None,
        model_configs: Optional[Dict] = None,
    ):
        """
        Initialize batch evaluator
        
        Args:
            models: List of model names to evaluate
            test_dataset: Test dataset name
            output_dir: Output directory for results
            asr_model: ASR model for back-transcription
            device: Device to run on
            max_samples: Optional limit on samples
            model_configs: Optional custom model configurations
        """
        self.models = models
        self.test_dataset = test_dataset
        self.output_dir = Path(output_dir)
        self.asr_model = asr_model
        self.device = device
        self.max_samples = max_samples
        
        # Merge custom configs with defaults
        self.model_configs = DEFAULT_MODEL_CONFIGS.copy()
        if model_configs:
            for name, config in model_configs.items():
                if name in self.model_configs:
                    self.model_configs[name].update(config)
                else:
                    self.model_configs[name] = config
        
        self.eval_root = Path(__file__).parent
        self.results = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to file and console"""
        log_file = self.output_dir / "batch_evaluation.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(f"{__name__}.batch")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        
        self.logger.info(f"Logging to: {log_file}")
    
    def evaluate_model(self, model_name: str) -> Optional[Dict]:
        """
        Run evaluation for a single model
        
        Args:
            model_name: Model name to evaluate
            
        Returns:
            Evaluation results dictionary or None if failed
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"EVALUATING: {model_name}")
        self.logger.info("=" * 80)
        
        # Get model config
        config = self.model_configs.get(model_name, {"model": model_name})
        
        # Build command
        cmd = [
            sys.executable,
            str(self.eval_root / "evaluate.py"),
            "--model", config.get("model", model_name),
            "--test-dataset", self.test_dataset,
            "--device", self.device,
            "--asr-model", self.asr_model,
            "--name", f"batch_{model_name}",
        ]
        
        # Add language
        if "language" in config:
            cmd.extend(["--language", config["language"]])
        
        # Add model-specific args
        if "voice" in config:
            cmd.extend(["--voice", config["voice"]])
        if "speaker" in config:
            cmd.extend(["--speaker", config["speaker"]])
        if "speaker_wav" in config:
            cmd.extend(["--speaker-wav", config["speaker_wav"]])
        if config.get("use_fallback"):
            cmd.append("--use-fallback")
        
        # Add max samples
        if self.max_samples:
            cmd.extend(["--max-samples", str(self.max_samples)])
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.eval_root,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                self.logger.error(f"Evaluation failed for {model_name}")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                return None
            
            self.logger.info(f"✓ {model_name} evaluation completed")
            
            # Find the output directory
            output_dirs = list((self.eval_root / "outputs").glob(f"batch_{model_name}_*"))
            if output_dirs:
                # Get the most recent one
                eval_dir = sorted(output_dirs, key=lambda x: x.stat().st_mtime)[-1]
                results_file = eval_dir / "evaluation_results.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Exception evaluating {model_name}: {e}")
            return None
    
    def run(self):
        """Run batch evaluation for all models"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BATCH TTS EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info("")
        self.logger.info(f"Models: {', '.join(self.models)}")
        self.logger.info(f"Dataset: {self.test_dataset}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"ASR Model: {self.asr_model}")
        if self.max_samples:
            self.logger.info(f"Max Samples: {self.max_samples}")
        self.logger.info("")
        
        # Evaluate each model
        for model_name in self.models:
            result = self.evaluate_model(model_name)
            if result:
                self.results[model_name] = result
            else:
                self.results[model_name] = {"error": "Evaluation failed"}
        
        # Generate comparison
        self.generate_comparison()
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("✓ BATCH EVALUATION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def generate_comparison(self):
        """Generate comparison report from all results"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("GENERATING COMPARISON REPORT")
        self.logger.info("=" * 80)
        
        # Compile comparison data
        comparison_data = []
        
        for model_name, result in self.results.items():
            if "error" in result:
                row = {
                    "model": model_name,
                    "status": "failed",
                }
            else:
                latency = result.get("latency", {})
                mos = result.get("mos", {})
                asr_wer = result.get("asr_wer", {})
                
                row = {
                    "model": model_name,
                    "status": "success",
                    "num_samples": result.get("num_samples", 0),
                    "mean_rtf": latency.get("mean_rtf"),
                    "realtime_capable": latency.get("realtime_capable"),
                    "mean_mos": mos.get("mean_mos"),
                    "mos_std": mos.get("std_mos"),
                    "mean_wer": asr_wer.get("mean_wer"),
                    "wer_std": asr_wer.get("std_wer"),
                }
            
            comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Sort by MOS (descending) if available
        if "mean_mos" in df.columns:
            df = df.sort_values("mean_mos", ascending=False, na_position="last")
        
        # Save comparison CSV
        csv_path = self.output_dir / "comparison.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved comparison CSV: {csv_path}")
        
        # Save detailed JSON
        json_path = self.output_dir / "comparison.json"
        comparison_json = {
            "timestamp": datetime.now().isoformat(),
            "test_dataset": self.test_dataset,
            "asr_model": self.asr_model,
            "models": self.results,
            "summary": comparison_data,
        }
        with open(json_path, "w") as f:
            json.dump(comparison_json, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved comparison JSON: {json_path}")
        
        # Print summary table
        self.print_comparison_table(df)
    
    def print_comparison_table(self, df: pd.DataFrame):
        """Print comparison table"""
        print("\n" + "=" * 80)
        print("TTS MODEL COMPARISON")
        print("=" * 80)
        print(f"\nDataset: {self.test_dataset}")
        print(f"ASR Model: {self.asr_model}")
        print("")
        
        # Format table
        headers = ["Model", "Samples", "RTF", "RT?", "MOS", "WER%"]
        rows = []
        
        for _, row in df.iterrows():
            if row.get("status") == "failed":
                rows.append([
                    row["model"],
                    "-",
                    "-",
                    "-",
                    "-",
                    "FAILED",
                ])
            else:
                rtf = f"{row.get('mean_rtf', 0):.3f}" if row.get('mean_rtf') else "-"
                rt = "✓" if row.get('realtime_capable') else "✗"
                mos = f"{row.get('mean_mos', 0):.2f}" if row.get('mean_mos') else "-"
                wer = f"{row.get('mean_wer', 0)*100:.1f}" if row.get('mean_wer') else "-"
                
                rows.append([
                    row["model"],
                    str(row.get("num_samples", 0)),
                    rtf,
                    rt,
                    mos,
                    wer,
                ])
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        header_line = " | ".join(
            str(h).ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        print(header_line)
        print("-" * len(header_line))
        
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            print(row_line)
        
        print("")
        print("Legend: RTF = Real-Time Factor (lower is faster)")
        print("        RT? = Real-time capable (RTF < 1.0)")
        print("        MOS = Mean Opinion Score (1-5, higher is better)")
        print("        WER% = Word Error Rate % (lower is better)")
        print("=" * 80)


def load_config(config_path: Path) -> Dict:
    """Load batch evaluation configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS Evaluation - Compare multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(list_datasets())}

Examples:
  # Evaluate all models
  python batch_evaluate.py \\
      --models xtts-v2,kokoro,melotts,glm-tts \\
      --test-dataset meso-malaya-test

  # Quick comparison with limited samples
  python batch_evaluate.py \\
      --models xtts-v2,kokoro \\
      --test-dataset ytl-malay-test \\
      --max-samples 10

  # Using configuration file
  python batch_evaluate.py --config batch_config.json
"""
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to evaluate"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        choices=list_datasets(),
        help="Test dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison",
        help="Output directory for results"
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="ASR model for back-transcription"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples per model"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(Path(args.config))
        models = config.get("models", [])
        test_dataset = config.get("test_dataset", args.test_dataset)
        output_dir = config.get("output_dir", args.output_dir)
        asr_model = config.get("asr_model", args.asr_model)
        device = config.get("device", args.device)
        max_samples = config.get("max_samples", args.max_samples)
        model_configs = config.get("model_configs", {})
    else:
        if not args.models or not args.test_dataset:
            parser.error("--models and --test-dataset are required (or use --config)")
        
        models = [m.strip() for m in args.models.split(",")]
        test_dataset = args.test_dataset
        output_dir = args.output_dir
        asr_model = args.asr_model
        device = args.device
        max_samples = args.max_samples
        model_configs = {}
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"batch_{test_dataset}_{timestamp}"
    
    # Run batch evaluation
    evaluator = BatchTTSEvaluator(
        models=models,
        test_dataset=test_dataset,
        output_dir=output_dir,
        asr_model=asr_model,
        device=device,
        max_samples=max_samples,
        model_configs=model_configs,
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()

