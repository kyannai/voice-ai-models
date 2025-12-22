#!/usr/bin/env python3
"""
Unified ASR Evaluation Script

Automatically detects model family and routes to the appropriate transcriber.
Supports: Whisper, Qwen2.5-Omni, Qwen3-Omni, Qwen2-Audio, Parakeet (NeMo), Paraformer, and more.

Usage:
    python evaluate.py --model nvidia/parakeet-tdt-0.6b-v3 --test-dataset meso-malaya-test --device auto
"""

import argparse
import logging
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from datasets_config import list_datasets, get_dataset_config


def detect_model_family(model_name: str) -> tuple:
    """
    Detect model family from model name
    
    Returns:
        (script_name, model_family_name)
    """
    model_lower = model_name.lower()
    model_path = Path(model_name)
    
    # Check for LoRA checkpoint (local path with adapter files)
    is_lora = (model_path.exists() and 
               (model_path / "adapter_config.json").exists() and
               (model_path / "adapter_model.safetensors").exists())
    
    if is_lora:
        # For LoRA checkpoints, check the path for model family hints
        # This allows us to route to the correct transcription script
        
        # Check for Qwen2.5-Omni in path
        if "qwen2.5-omni" in model_lower or "qwen2_5-omni" in model_lower or "qwen25omni" in model_lower:
            return ("transcribe_qwen25omni.py", "Qwen2.5-Omni-LoRA")
        
        # Check for Qwen3-Omni in path
        if "qwen3-omni" in model_lower or "qwen3omni" in model_lower:
            return ("transcribe_qwen3omni.py", "Qwen3-Omni-LoRA")
        
        # Check for Qwen2-Audio in path (or default for LoRA if no other match)
        if "qwen2-audio" in model_lower or "qwen2audio" in model_lower:
            return ("transcribe_qwen2audio.py", "Qwen2-Audio-LoRA")
        
        # If we can't determine from path, try reading adapter_config.json
        try:
            import json
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path", "").lower()
                
                if "qwen2.5-omni" in base_model or "qwen2_5-omni" in base_model or "qwen25omni" in base_model:
                    return ("transcribe_qwen25omni.py", "Qwen2.5-Omni-LoRA")
                elif "qwen3-omni" in base_model or "qwen3omni" in base_model:
                    return ("transcribe_qwen3omni.py", "Qwen3-Omni-LoRA")
                elif "qwen2-audio" in base_model or "qwen2audio" in base_model:
                    return ("transcribe_qwen2audio.py", "Qwen2-Audio-LoRA")
        except Exception:
            pass
        
        # Default to Qwen2-Audio for unknown LoRA checkpoints
        return ("transcribe_qwen2audio.py", "Qwen2-Audio-LoRA")
    
    # Check for Qwen2.5-Omni (newer, better ASR)
    if "qwen2.5-omni" in model_lower or "qwen2_5-omni" in model_lower or "qwen25omni" in model_lower:
        return ("transcribe_qwen25omni.py", "Qwen2.5-Omni")
    
    # Check for Qwen3-Omni
    if "qwen3-omni" in model_lower or "qwen3omni" in model_lower:
        return ("transcribe_qwen3omni.py", "Qwen3-Omni")
    
    # Check for Qwen2-Audio
    if "qwen2-audio" in model_lower or "qwen2audio" in model_lower:
        return ("transcribe_qwen2audio.py", "Qwen2-Audio")
    
    # Check for Whisper models
    if "whisper" in model_lower:
        return ("transcribe_whisper.py", "Whisper")
    
    # Check for NVIDIA Parakeet models (NeMo-based)
    # Support both HuggingFace models and local .nemo files
    if "parakeet" in model_lower:
        return ("transcribe_parakeet.py", "Parakeet")
    
    # Check for local .nemo files (fine-tuned Parakeet models)
    if model_path.exists():
        if model_path.is_file() and model_path.suffix == '.nemo':
            return ("transcribe_parakeet.py", "Parakeet-FineTuned")
        elif model_path.is_dir():
            # Check if directory contains a .nemo file
            nemo_files = list(model_path.glob("*.nemo"))
            if nemo_files:
                return ("transcribe_parakeet.py", "Parakeet-FineTuned")
    
    # Check for Paraformer and other FunASR models
    if "paraformer" in model_lower or model_lower in ["paraformer-zh", "paraformer-en", "paraformer-multilingual"]:
        return ("transcribe_paraformer.py", "Paraformer")
    
    # Default to Paraformer for unknown FunASR models
    return ("transcribe_paraformer.py", "FunASR")


class UnifiedEvaluator:
    """Main evaluator that orchestrates transcription + metrics calculation"""
    
    def __init__(self, args):
        self.args = args
        self.eval_root = Path(__file__).parent
        
        # Detect model family
        self.script_name, self.model_family = detect_model_family(args.model)
        
        self.output_dir = self.create_output_dir()
        self.setup_logging()
        
    def create_output_dir(self):
        """Create output directory with automatic naming"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract model name
        if self.args.hub == "local" or Path(self.args.model).exists():
            model_name = Path(self.args.model).name
        else:
            model_name = self.args.model.split("/")[-1]
        
        # Use dataset name directly from argument
        dataset_name = self.args.test_dataset
        
        # Build directory name
        name_parts = [
            self.model_family,
            model_name,
            dataset_name,
            self.args.device,
            timestamp
        ]
        
        # Add custom name if provided
        if self.args.name:
            name_parts.insert(0, self.args.name)
        
        dir_name = "_".join(name_parts)
        output_dir = self.eval_root / "outputs" / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def setup_logging(self):
        """Setup logging to both file and console"""
        log_file = self.output_dir / "evaluation.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to: {log_file}")
    
    def save_config(self):
        """Save evaluation configuration for reproducibility"""
        # Get dataset configuration
        dataset_config = get_dataset_config(self.args.test_dataset)
        
        # Convert Path objects to strings for JSON serialization
        dataset_config_serializable = {}
        for key, value in dataset_config.items():
            if isinstance(value, Path):
                dataset_config_serializable[key] = str(value)
            else:
                dataset_config_serializable[key] = value
        
        # Determine normalization configuration for config tracking
        if self.args.no_normalize:
            normalize_config = "disabled"
        elif self.args.normalize:
            normalize_config = self.args.normalize
        else:
            normalize_config = "all_steps"  # Default behavior
        
        config = {
            "model": self.args.model,
            "model_family": self.model_family,
            "hub": self.args.hub,
            "test_dataset": self.args.test_dataset,
            "dataset_config": dataset_config_serializable,
            "device": self.args.device,
            "max_samples": self.args.max_samples,
            "language": self.args.language,
            "asr_prompt": self.args.asr_prompt,
            "text_normalization": normalize_config,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
        }
        
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_file}")
        return config
    
    def run_transcription(self):
        """Run transcription using detected model family"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("STEP 1: Running Transcription")
        self.logger.info("="*80)
        self.logger.info(f"Model Family: {self.model_family}")
        self.logger.info(f"Script: {self.script_name}")
        self.logger.info("")
        
        script_path = self.eval_root / "transcribe" / self.script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Transcription script not found: {script_path}")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model", self.args.model,
            "--test-dataset", self.args.test_dataset,
            "--output-dir", str(self.output_dir),
            "--device", self.args.device,
        ]
        
        # Add hub/source (only for models that support it: Whisper, Paraformer)
        if self.model_family in ["Whisper", "Paraformer", "FunASR"]:
            cmd.extend(["--hub", self.args.hub])
        
        # Add language (for Whisper)
        if self.args.language and self.model_family == "Whisper":
            cmd.extend(["--language", self.args.language])
        
        # Add ASR prompt (for Qwen models: Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni)
        if self.args.asr_prompt and "Qwen" in self.model_family:
            cmd.extend(["--asr-prompt", self.args.asr_prompt])
        
        # Add max samples
        if self.args.max_samples:
            cmd.extend(["--max-samples", str(self.args.max_samples)])
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info("")
        
        result = subprocess.run(cmd, cwd=self.eval_root)
        
        if result.returncode != 0:
            raise RuntimeError(f"Transcription failed with code {result.returncode}")
        
        self.logger.info("")
        self.logger.info(f"✓ {self.model_family} transcription completed successfully")
        return True
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("STEP 2: Calculating Metrics")
        self.logger.info("="*80)
        
        predictions_file = self.output_dir / "predictions.json"
        
        if not predictions_file.exists():
            self.logger.error(f"Predictions file not found: {predictions_file}")
            self.logger.warning("Skipping metrics calculation")
            return False
        
        script_path = self.eval_root / "calculate_metrics" / "calculate_metrics.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--predictions", str(predictions_file),
            "--output-dir", str(self.output_dir)
        ]
        
        # Add normalization flags if specified
        if self.args.no_normalize:
            cmd.append("--no-normalize")
        elif self.args.normalize:
            cmd.extend(["--normalize", self.args.normalize])
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info("")
        
        result = subprocess.run(cmd, cwd=self.eval_root)
        
        if result.returncode != 0:
            self.logger.error(f"Metrics calculation failed with code {result.returncode}")
            return False
        
        self.logger.info("")
        self.logger.info("✓ Metrics calculation completed successfully")
        return True
    
    def print_summary(self):
        """Print evaluation summary"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*80)
        self.logger.info("")
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Model Family: {self.model_family}")
        
        # Try to load results
        results_file = self.output_dir / "evaluation_results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                
                self.logger.info(f"Samples: {results.get('num_samples', 'N/A')}")
                self.logger.info("")
                
                # Extract metrics (stored as simple floats)
                wer = results.get('wer', 0)
                cer = results.get('cer', 0)
                mer = results.get('mer', 0)
                
                # Print metrics
                self.logger.info(f"WER: {wer:.4f}")
                self.logger.info(f"CER: {cer:.4f}")
                self.logger.info(f"MER: {mer:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Could not load results: {e}")
        
        self.logger.info("")
        self.logger.info("Output files:")
        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                self.logger.info(f"  - {file.name}")
        
        self.logger.info("")
        self.logger.info(f"All results saved to: {self.output_dir}")
        self.logger.info("="*80)
    
    def run(self):
        """Run complete evaluation pipeline"""
        try:
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("🚀 STARTING ASR EVALUATION")
            self.logger.info("="*80)
            self.logger.info("")
            
            # Save configuration
            self.save_config()
            
            # Run transcription
            self.run_transcription()
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Print summary
            self.print_summary()
            
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("✓ EVALUATION COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified ASR Evaluation - Automatic model family detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(list_datasets())}

Examples:
  # Evaluate Whisper on meso-malaya-test
  python evaluate.py --model openai/whisper-large-v3-turbo --test-dataset meso-malaya-test

  # Evaluate Parakeet on ytl-malay-test
  python evaluate.py --model nvidia/parakeet-tdt-0.6b-v3 --test-dataset ytl-malay-test

  # Evaluate on SEACrowd dataset
  python evaluate.py --model openai/whisper-large-v3-turbo --test-dataset seacrowd-asr-malcsc
"""
    )
    
    # Required arguments
    parser.add_argument("--model", required=True, 
                       help="Model name or path (auto-detects family)")
    parser.add_argument("--test-dataset", required=True, 
                       choices=list_datasets(),
                       help="Test dataset name from registry")
    
    # Optional arguments
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device (default: auto)")
    parser.add_argument("--hub", default="hf", 
                       choices=["hf", "ms", "local"],
                       help="Model source: hf (HuggingFace), ms (ModelScope), local (default: hf)")
    parser.add_argument("--name", 
                       help="Custom run name (added to output directory)")
    parser.add_argument("--max-samples", type=int,
                       help="Limit number of samples for quick testing")
    
    # Model-specific arguments
    parser.add_argument("--language", default="ms",
                       help="Language for Whisper models (default: ms)")
    parser.add_argument("--asr-prompt",
                       help="ASR prompt for Qwen models")
    
    # Evaluation options
    parser.add_argument("--normalize", type=str,
                       help="Comma-separated list of normalization steps: lowercase, remove_hyphens, "
                            "remove_punctuation, normalize_whitespace. Use 'none' to disable. "
                            "If not specified, applies all steps.")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable text normalization (shortcut for --normalize none)")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = UnifiedEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
