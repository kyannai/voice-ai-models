#!/usr/bin/env python3
"""
Unified TTS Evaluation Script

Automatically detects TTS model family and routes to the appropriate synthesizer.
Supports: XTTS v2, Kokoro, MeloTTS, GLM-TTS

Target Languages:
- English (en): Fully supported by all models
- Malay (ms): Uses English mode as fallback (phonetically similar)
- Code-switching (en-ms): Mixed English-Malay sentences, uses English mode

The evaluation pipeline:
1. Load text data from dataset
2. Synthesize speech using the selected TTS model
3. Calculate metrics (MOS, ASR-back-WER, Latency)
4. Save results and summary

Usage:
    python evaluate.py --model xtts-v2 --test-dataset meso-malaya-test --device auto
    python evaluate.py --model kokoro --test-dataset ytl-malay-test --voice af_heart
    python evaluate.py --model melotts --test-dataset malay-conversational --language en-ms
"""

import argparse
import logging
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from datasets_config import list_datasets, get_dataset_config, get_language_type, SUPPORTED_LANGUAGES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model family to script mapping
MODEL_FAMILIES = {
    "xtts": {
        "script": "synthesize_xtts.py",
        "display_name": "XTTS-v2",
        "description": "Coqui XTTS v2 - multilingual with voice cloning",
    },
    "xtts-v2": {
        "script": "synthesize_xtts.py",
        "display_name": "XTTS-v2",
        "description": "Coqui XTTS v2 - multilingual with voice cloning",
    },
    "kokoro": {
        "script": "synthesize_kokoro.py",
        "display_name": "Kokoro",
        "description": "Kokoro TTS - lightweight and fast",
    },
    "melotts": {
        "script": "synthesize_melotts.py",
        "display_name": "MeloTTS",
        "description": "MeloTTS - multi-lingual from MyShell.ai",
    },
    "melo": {
        "script": "synthesize_melotts.py",
        "display_name": "MeloTTS",
        "description": "MeloTTS - multi-lingual from MyShell.ai",
    },
    "glm": {
        "script": "synthesize_glmtts.py",
        "display_name": "GLM-TTS",
        "description": "GLM-4-Voice - multimodal speech synthesis",
    },
    "glm-tts": {
        "script": "synthesize_glmtts.py",
        "display_name": "GLM-TTS",
        "description": "GLM-4-Voice - multimodal speech synthesis",
    },
}


def detect_model_family(model_name: str) -> tuple:
    """
    Detect model family from model name
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        (script_name, display_name) tuple
    """
    model_lower = model_name.lower()
    
    # Check for exact match first
    if model_lower in MODEL_FAMILIES:
        info = MODEL_FAMILIES[model_lower]
        return info["script"], info["display_name"]
    
    # Check for partial matches
    if "xtts" in model_lower:
        return MODEL_FAMILIES["xtts"]["script"], MODEL_FAMILIES["xtts"]["display_name"]
    
    if "kokoro" in model_lower:
        return MODEL_FAMILIES["kokoro"]["script"], MODEL_FAMILIES["kokoro"]["display_name"]
    
    if "melo" in model_lower:
        return MODEL_FAMILIES["melotts"]["script"], MODEL_FAMILIES["melotts"]["display_name"]
    
    if "glm" in model_lower:
        return MODEL_FAMILIES["glm"]["script"], MODEL_FAMILIES["glm"]["display_name"]
    
    # Default to XTTS
    logger.warning(f"Unknown model '{model_name}', defaulting to XTTS v2")
    return MODEL_FAMILIES["xtts"]["script"], MODEL_FAMILIES["xtts"]["display_name"]


class UnifiedTTSEvaluator:
    """Main TTS evaluator that orchestrates synthesis + metrics calculation"""
    
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
        
        # Build directory name
        name_parts = [
            self.model_family,
            self.args.test_dataset,
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
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to: {log_file}")
    
    def save_config(self):
        """Save evaluation configuration for reproducibility"""
        dataset_config = get_dataset_config(self.args.test_dataset)
        
        # Convert Path objects to strings
        dataset_config_serializable = {}
        for key, value in dataset_config.items():
            if isinstance(value, Path):
                dataset_config_serializable[key] = str(value)
            else:
                dataset_config_serializable[key] = value
        
        config = {
            "model": self.args.model,
            "model_family": self.model_family,
            "test_dataset": self.args.test_dataset,
            "dataset_config": dataset_config_serializable,
            "device": self.args.device,
            "max_samples": self.args.max_samples,
            "language": self.args.language,
            "asr_model": self.args.asr_model,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
        }
        
        # Add model-specific config
        if hasattr(self.args, 'speaker_wav') and self.args.speaker_wav:
            config["speaker_wav"] = self.args.speaker_wav
        if hasattr(self.args, 'voice') and self.args.voice:
            config["voice"] = self.args.voice
        if hasattr(self.args, 'speaker') and self.args.speaker:
            config["speaker"] = self.args.speaker
        
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_file}")
        return config
    
    def run_synthesis(self):
        """Run TTS synthesis using detected model family"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: Running TTS Synthesis")
        self.logger.info("=" * 80)
        self.logger.info(f"Model Family: {self.model_family}")
        self.logger.info(f"Script: {self.script_name}")
        self.logger.info("")
        
        script_path = self.eval_root / "synthesize" / self.script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Synthesis script not found: {script_path}")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--test-dataset", self.args.test_dataset,
            "--output-dir", str(self.output_dir),
            "--device", self.args.device,
        ]
        
        # Add model-specific arguments
        if self.model_family == "XTTS-v2":
            if self.args.speaker_wav:
                cmd.extend(["--speaker-wav", self.args.speaker_wav])
            cmd.extend(["--language", self.args.language])
            
        elif self.model_family == "Kokoro":
            if self.args.voice:
                cmd.extend(["--voice", self.args.voice])
            cmd.extend(["--language", self.args.language])
            
        elif self.model_family == "MeloTTS":
            if self.args.speaker:
                cmd.extend(["--speaker", self.args.speaker])
            # Map language for MeloTTS
            lang_map = {"ms": "EN", "en": "EN", "zh": "ZH", "ja": "JP", "ko": "KR"}
            melo_lang = lang_map.get(self.args.language.lower(), "EN")
            cmd.extend(["--language", melo_lang])
            
        elif self.model_family == "GLM-TTS":
            if self.args.use_fallback:
                cmd.append("--use-fallback")
        
        # Add max samples
        if self.args.max_samples:
            cmd.extend(["--max-samples", str(self.args.max_samples)])
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info("")
        
        result = subprocess.run(cmd, cwd=self.eval_root)
        
        if result.returncode != 0:
            raise RuntimeError(f"Synthesis failed with code {result.returncode}")
        
        self.logger.info("")
        self.logger.info(f"✓ {self.model_family} synthesis completed successfully")
        return True
    
    def calculate_metrics(self):
        """Calculate TTS evaluation metrics"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: Calculating Metrics")
        self.logger.info("=" * 80)
        
        synthesis_file = self.output_dir / "synthesis_results.json"
        
        if not synthesis_file.exists():
            self.logger.error(f"Synthesis results not found: {synthesis_file}")
            self.logger.warning("Skipping metrics calculation")
            return False
        
        script_path = self.eval_root / "calculate_metrics" / "calculate_metrics.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--synthesis-results", str(synthesis_file),
            "--output-dir", str(self.output_dir),
            "--asr-model", self.args.asr_model,
            "--language", self.args.language,
            "--device", self.args.device,
        ]
        
        # Add metric flags
        if self.args.no_mos:
            cmd.append("--no-mos")
        if self.args.no_asr_wer:
            cmd.append("--no-asr-wer")
        
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
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info("")
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Model Family: {self.model_family}")
        self.logger.info(f"Dataset: {self.args.test_dataset}")
        
        # Try to load results
        results_file = self.output_dir / "evaluation_results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                
                self.logger.info(f"Samples: {results.get('num_samples', 'N/A')}")
                
                # Print latency
                latency = results.get('latency', {})
                if latency:
                    self.logger.info("")
                    self.logger.info("--- Latency ---")
                    self.logger.info(f"Mean RTF: {latency.get('mean_rtf', 0):.4f}")
                    self.logger.info(f"Real-time capable: {latency.get('realtime_capable', False)}")
                
                # Print MOS
                mos = results.get('mos', {})
                if mos and 'mean_mos' in mos:
                    self.logger.info("")
                    self.logger.info("--- Neural MOS ---")
                    self.logger.info(f"Mean MOS: {mos['mean_mos']:.2f} / 5.0")
                
                # Print WER
                asr_wer = results.get('asr_wer', {})
                if asr_wer and 'mean_wer' in asr_wer:
                    self.logger.info("")
                    self.logger.info("--- ASR-back-WER ---")
                    self.logger.info(f"Mean WER: {asr_wer['mean_wer']*100:.2f}%")
                
            except Exception as e:
                self.logger.warning(f"Could not load results: {e}")
        
        self.logger.info("")
        self.logger.info("Output files:")
        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                self.logger.info(f"  - {file.name}")
        
        self.logger.info("")
        self.logger.info(f"All results saved to: {self.output_dir}")
        self.logger.info("=" * 80)
    
    def run(self):
        """Run complete TTS evaluation pipeline"""
        try:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("🚀 STARTING TTS EVALUATION")
            self.logger.info("=" * 80)
            self.logger.info("")
            
            # Save configuration
            self.save_config()
            
            # Run synthesis
            self.run_synthesis()
            
            # Calculate metrics
            if not self.args.synthesis_only:
                self.calculate_metrics()
            
            # Print summary
            self.print_summary()
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("✓ TTS EVALUATION COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    available_models = list(set(MODEL_FAMILIES.keys()))
    
    parser = argparse.ArgumentParser(
        description="Unified TTS Evaluation - Automatic model family detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(sorted(set(m['display_name'] for m in MODEL_FAMILIES.values())))}

Available datasets:
  {', '.join(list_datasets())}

Examples:
  # Evaluate XTTS v2 on meso-malaya-test
  python evaluate.py --model xtts-v2 --test-dataset meso-malaya-test

  # Evaluate Kokoro with specific voice
  python evaluate.py --model kokoro --test-dataset ytl-malay-test --voice af_heart

  # Evaluate MeloTTS with English speaker
  python evaluate.py --model melotts --test-dataset meso-malaya-test --speaker EN-US

  # Quick test with limited samples
  python evaluate.py --model xtts-v2 --test-dataset meso-malaya-test --max-samples 10
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="TTS model name or family (xtts-v2, kokoro, melotts, glm-tts)"
    )
    parser.add_argument(
        "--test-dataset",
        required=True,
        choices=list_datasets(),
        help="Test dataset name from registry"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--name",
        help="Custom run name (added to output directory)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples for quick testing"
    )
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "en", "ms", "en-ms"],
        help="Language: en (English), ms (Malay), en-ms (code-switching). "
             "Default 'auto' detects from dataset."
    )
    
    # ASR-back-WER options
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model for back-transcription"
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--speaker-wav",
        help="Speaker reference audio for XTTS v2 voice cloning"
    )
    parser.add_argument(
        "--voice",
        help="Voice name for Kokoro TTS"
    )
    parser.add_argument(
        "--speaker",
        help="Speaker ID for MeloTTS"
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use edge-tts fallback for GLM-TTS"
    )
    
    # Evaluation options
    parser.add_argument(
        "--synthesis-only",
        action="store_true",
        help="Only run synthesis, skip metrics calculation"
    )
    parser.add_argument(
        "--no-mos",
        action="store_true",
        help="Skip MOS calculation"
    )
    parser.add_argument(
        "--no-asr-wer",
        action="store_true",
        help="Skip ASR-back-WER calculation"
    )
    
    args = parser.parse_args()
    
    # Auto-detect language from dataset if not specified
    if args.language == "auto":
        args.language = get_language_type(args.test_dataset)
        logger.info(f"Auto-detected language: {args.language} ({SUPPORTED_LANGUAGES.get(args.language, 'Unknown')})")
    
    # Run evaluation
    evaluator = UnifiedTTSEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()

