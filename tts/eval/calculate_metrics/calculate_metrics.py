#!/usr/bin/env python3
"""
TTS Evaluation Metrics Calculator

Calculates three core metrics for TTS evaluation:
1. Neural MOS (Mean Opinion Score) - using UTMOS or SpeechMOS
2. ASR-back-WER (Word Error Rate) - synthesize then transcribe back
3. Latency/RTF (Real-Time Factor) - synthesis speed

Usage:
    python calculate_metrics.py \
        --synthesis-results outputs/xtts_eval/synthesis_results.json \
        --output-dir outputs/xtts_eval \
        --asr-model openai/whisper-large-v3-turbo
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import string

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralMOSCalculator:
    """
    Calculate Neural MOS (Mean Opinion Score) using pretrained models.
    
    Supports:
    - UTMOS (UTokyo MOS predictor)
    - SpeechMOS (speechmos library)
    """
    
    def __init__(
        self,
        model: str = "utmos",
        device: str = "auto",
    ):
        """
        Initialize Neural MOS calculator
        
        Args:
            model: MOS model to use ('utmos' or 'speechmos')
            device: Device to run on
        """
        self.model_name = model
        
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Neural MOS model: {model}")
        logger.info(f"Device: {self.device}")
        
        self.model = None
        self.predictor = None
        
        if model == "utmos":
            self._load_utmos()
        elif model == "speechmos":
            self._load_speechmos()
        else:
            raise ValueError(f"Unknown MOS model: {model}. Use 'utmos' or 'speechmos'")
    
    def _load_utmos(self):
        """Load UTMOS model"""
        try:
            import torch
            
            # UTMOS from SpeechMOS library
            self.predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            )
            self.predictor = self.predictor.to(self.device)
            self.predictor.eval()
            
            logger.info("UTMOS model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load UTMOS: {e}")
            logger.info("Falling back to speechmos library")
            self._load_speechmos()
    
    def _load_speechmos(self):
        """Load SpeechMOS model"""
        try:
            from speechmos import SpeechMOS
            
            self.predictor = SpeechMOS(device=self.device)
            self.model_name = "speechmos"
            
            logger.info("SpeechMOS loaded successfully")
            
        except ImportError:
            raise ImportError(
                "speechmos not installed. Install with: pip install speechmos"
            )
    
    def predict(
        self,
        audio_path: Union[str, Path],
    ) -> float:
        """
        Predict MOS for a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MOS score (1.0 - 5.0)
        """
        import torch
        import librosa
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.model_name == "utmos":
                # UTMOS expects (batch, samples) at 16kHz
                score = self.predictor(audio_tensor, sr)
                if isinstance(score, torch.Tensor):
                    score = score.item()
            else:
                # SpeechMOS
                score = self.predictor.predict(str(audio_path))
        
        # Clamp to valid range
        score = max(1.0, min(5.0, float(score)))
        
        return score
    
    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
    ) -> List[float]:
        """
        Predict MOS for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of MOS scores
        """
        scores = []
        
        for audio_path in tqdm(audio_paths, desc="Calculating MOS"):
            try:
                score = self.predict(audio_path)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to calculate MOS for {audio_path}: {e}")
                scores.append(None)
        
        return scores


class ASRBackWERCalculator:
    """
    Calculate ASR-back-WER by transcribing synthesized audio
    and comparing to original text.
    
    Supported languages:
    - en: English
    - ms: Malay
    - en-ms: Code-switching (treated as Malay for ASR)
    """
    
    # Map our language codes to Whisper language codes
    WHISPER_LANGUAGE_MAP = {
        "en": "en",
        "ms": "ms",
        "en-ms": "ms",  # Code-switching uses Malay mode (handles both)
    }
    
    def __init__(
        self,
        asr_model: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        language: str = "ms",
    ):
        """
        Initialize ASR-back-WER calculator
        
        Args:
            asr_model: ASR model to use for transcription
            device: Device to run on
            language: Language code (en, ms, or en-ms)
        """
        self.asr_model = asr_model
        # Map language code to Whisper language
        self.language = self.WHISPER_LANGUAGE_MAP.get(language, language)
        
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading ASR model: {asr_model}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Language: {language}")
        
        self._load_asr_model()
    
    def _load_asr_model(self):
        """Load ASR model for transcription"""
        import torch
        from transformers import pipeline
        
        # Use HuggingFace pipeline for flexibility
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            chunk_length_s=30,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        logger.info("ASR model loaded successfully")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> str:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        import librosa
        
        # Load audio at 16kHz
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        # Generate kwargs for language if applicable
        generate_kwargs = {}
        if "whisper" in self.asr_model.lower():
            generate_kwargs["language"] = self.language
        
        result = self.pipe(
            audio,
            generate_kwargs=generate_kwargs,
        )
        
        return result["text"].strip()
    
    def calculate_wer(
        self,
        reference: str,
        hypothesis: str,
        normalize: bool = True,
    ) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis
        
        Args:
            reference: Original text
            hypothesis: Transcribed text
            normalize: Whether to normalize text before comparison
            
        Returns:
            WER score (0.0 - 1.0+)
        """
        from jiwer import wer
        
        if normalize:
            reference = self._normalize_text(reference)
            hypothesis = self._normalize_text(hypothesis)
        
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        return wer(reference, hypothesis)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER calculation"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Replace hyphens with spaces (important for Malay)
        text = text.replace('-', ' ')
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def evaluate(
        self,
        audio_path: Union[str, Path],
        reference_text: str,
    ) -> Dict:
        """
        Evaluate single sample: transcribe and calculate WER
        
        Args:
            audio_path: Path to synthesized audio
            reference_text: Original text used for synthesis
            
        Returns:
            Dictionary with transcription and WER
        """
        try:
            hypothesis = self.transcribe(audio_path)
            wer_score = self.calculate_wer(reference_text, hypothesis)
            
            return {
                "reference": reference_text,
                "hypothesis": hypothesis,
                "wer": wer_score,
            }
        except Exception as e:
            logger.error(f"Failed to evaluate {audio_path}: {e}")
            return {
                "reference": reference_text,
                "hypothesis": "",
                "wer": 1.0,
                "error": str(e),
            }
    
    def evaluate_batch(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Evaluate batch of samples
        
        Args:
            samples: List of dicts with 'audio_path' and 'text' keys
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for sample in tqdm(samples, desc="ASR-back-WER"):
            audio_path = sample.get("audio_path")
            reference_text = sample.get("text", "")
            
            if not audio_path or not Path(audio_path).exists():
                results.append({
                    "reference": reference_text,
                    "hypothesis": "",
                    "wer": 1.0,
                    "error": "Audio file not found",
                })
                continue
            
            result = self.evaluate(audio_path, reference_text)
            results.append(result)
        
        return results


class TTSMetricsCalculator:
    """
    Main TTS metrics calculator combining all metrics:
    - Neural MOS
    - ASR-back-WER
    - Latency/RTF
    """
    
    def __init__(
        self,
        mos_model: str = "utmos",
        asr_model: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        language: str = "ms",
    ):
        """
        Initialize TTS metrics calculator
        
        Args:
            mos_model: Neural MOS model ('utmos' or 'speechmos')
            asr_model: ASR model for back-transcription
            device: Device to run on
            language: Language code
        """
        self.device = device
        self.language = language
        
        # Initialize sub-calculators lazily
        self._mos_calculator = None
        self._asr_calculator = None
        self.mos_model = mos_model
        self.asr_model = asr_model
    
    @property
    def mos_calculator(self):
        """Lazy-load MOS calculator"""
        if self._mos_calculator is None:
            self._mos_calculator = NeuralMOSCalculator(
                model=self.mos_model,
                device=self.device,
            )
        return self._mos_calculator
    
    @property
    def asr_calculator(self):
        """Lazy-load ASR calculator"""
        if self._asr_calculator is None:
            self._asr_calculator = ASRBackWERCalculator(
                asr_model=self.asr_model,
                device=self.device,
                language=self.language,
            )
        return self._asr_calculator
    
    def calculate_all_metrics(
        self,
        synthesis_results: List[Dict],
        calculate_mos: bool = True,
        calculate_asr_wer: bool = True,
    ) -> Dict:
        """
        Calculate all TTS evaluation metrics
        
        Args:
            synthesis_results: List of synthesis result dicts
            calculate_mos: Whether to calculate MOS
            calculate_asr_wer: Whether to calculate ASR-back-WER
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("CALCULATING TTS EVALUATION METRICS")
        logger.info(f"{'='*70}\n")
        
        # Filter successful samples
        successful = [r for r in synthesis_results if 'error' not in r]
        logger.info(f"Processing {len(successful)}/{len(synthesis_results)} successful samples")
        
        if not successful:
            logger.warning("No successful samples to evaluate")
            return {"error": "No successful samples"}
        
        # Calculate latency/RTF metrics
        logger.info("\n--- Latency/RTF Metrics ---")
        rtf_values = [r.get('rtf', 0) for r in successful]
        synthesis_times = [r.get('synthesis_time', 0) for r in successful]
        audio_durations = [r.get('audio_duration', 0) for r in successful]
        
        latency_metrics = {
            "mean_rtf": np.mean(rtf_values),
            "std_rtf": np.std(rtf_values),
            "min_rtf": np.min(rtf_values),
            "max_rtf": np.max(rtf_values),
            "total_synthesis_time": sum(synthesis_times),
            "total_audio_duration": sum(audio_durations),
            "realtime_capable": np.mean(rtf_values) < 1.0,
        }
        
        logger.info(f"Mean RTF: {latency_metrics['mean_rtf']:.4f}")
        logger.info(f"RTF Std: {latency_metrics['std_rtf']:.4f}")
        logger.info(f"Real-time capable: {latency_metrics['realtime_capable']}")
        
        # Calculate MOS
        mos_metrics = {}
        if calculate_mos:
            logger.info("\n--- Neural MOS Metrics ---")
            try:
                audio_paths = [r['audio_path'] for r in successful]
                mos_scores = self.mos_calculator.predict_batch(audio_paths)
                
                # Add MOS to each result
                for i, r in enumerate(successful):
                    r['mos'] = mos_scores[i]
                
                valid_scores = [s for s in mos_scores if s is not None]
                
                if valid_scores:
                    mos_metrics = {
                        "mean_mos": np.mean(valid_scores),
                        "std_mos": np.std(valid_scores),
                        "min_mos": np.min(valid_scores),
                        "max_mos": np.max(valid_scores),
                        "valid_samples": len(valid_scores),
                    }
                    
                    logger.info(f"Mean MOS: {mos_metrics['mean_mos']:.4f}")
                    logger.info(f"MOS Std: {mos_metrics['std_mos']:.4f}")
                    logger.info(f"MOS Range: [{mos_metrics['min_mos']:.2f}, {mos_metrics['max_mos']:.2f}]")
                
            except Exception as e:
                logger.error(f"Failed to calculate MOS: {e}")
                mos_metrics = {"error": str(e)}
        
        # Calculate ASR-back-WER
        asr_wer_metrics = {}
        if calculate_asr_wer:
            logger.info("\n--- ASR-back-WER Metrics ---")
            try:
                asr_results = self.asr_calculator.evaluate_batch(successful)
                
                # Add ASR results to each synthesis result
                for i, r in enumerate(successful):
                    r['asr_hypothesis'] = asr_results[i].get('hypothesis', '')
                    r['asr_wer'] = asr_results[i].get('wer', 1.0)
                
                wer_values = [r.get('wer', 1.0) for r in asr_results if 'error' not in r]
                
                if wer_values:
                    asr_wer_metrics = {
                        "mean_wer": np.mean(wer_values),
                        "std_wer": np.std(wer_values),
                        "min_wer": np.min(wer_values),
                        "max_wer": np.max(wer_values),
                        "valid_samples": len(wer_values),
                        "asr_model": self.asr_model,
                    }
                    
                    logger.info(f"Mean WER: {asr_wer_metrics['mean_wer']:.4f}")
                    logger.info(f"WER Std: {asr_wer_metrics['std_wer']:.4f}")
                    logger.info(f"WER Range: [{asr_wer_metrics['min_wer']:.4f}, {asr_wer_metrics['max_wer']:.4f}]")
                
            except Exception as e:
                logger.error(f"Failed to calculate ASR-back-WER: {e}")
                asr_wer_metrics = {"error": str(e)}
        
        # Compile all metrics
        results = {
            "num_samples": len(successful),
            "latency": latency_metrics,
            "mos": mos_metrics,
            "asr_wer": asr_wer_metrics,
            "samples": successful,
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("TTS EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nSamples evaluated: {results.get('num_samples', 0)}")
        
        # Latency
        if 'latency' in results:
            lat = results['latency']
            print(f"\n--- Latency ---")
            print(f"Mean RTF: {lat.get('mean_rtf', 0):.4f}")
            print(f"Real-time capable: {lat.get('realtime_capable', False)}")
        
        # MOS
        if 'mos' in results and 'mean_mos' in results['mos']:
            mos = results['mos']
            print(f"\n--- Neural MOS ---")
            print(f"Mean MOS: {mos['mean_mos']:.2f} / 5.0")
            print(f"MOS Std: {mos.get('std_mos', 0):.2f}")
        
        # ASR-WER
        if 'asr_wer' in results and 'mean_wer' in results['asr_wer']:
            asr = results['asr_wer']
            print(f"\n--- ASR-back-WER ---")
            print(f"Mean WER: {asr['mean_wer']*100:.2f}%")
            print(f"WER Std: {asr.get('std_wer', 0)*100:.2f}%")
            print(f"ASR Model: {asr.get('asr_model', 'N/A')}")
        
        print("\n" + "=" * 70)


def load_synthesis_results(results_path: Path) -> Dict:
    """Load synthesis results from JSON file"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data.get('results', []))} synthesis results")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Calculate TTS evaluation metrics (MOS, ASR-WER, Latency)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation
  python calculate_metrics.py \\
      --synthesis-results outputs/xtts_eval/synthesis_results.json \\
      --output-dir outputs/xtts_eval \\
      --asr-model openai/whisper-large-v3-turbo
  
  # MOS only (faster)
  python calculate_metrics.py \\
      --synthesis-results outputs/kokoro_eval/synthesis_results.json \\
      --output-dir outputs/kokoro_eval \\
      --no-asr-wer
  
  # ASR-WER only
  python calculate_metrics.py \\
      --synthesis-results outputs/melotts_eval/synthesis_results.json \\
      --output-dir outputs/melotts_eval \\
      --no-mos
"""
    )
    
    parser.add_argument(
        "--synthesis-results",
        type=str,
        required=True,
        help="Path to synthesis_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="ASR model for back-transcription (default: whisper-large-v3-turbo)"
    )
    parser.add_argument(
        "--mos-model",
        type=str,
        default="utmos",
        choices=["utmos", "speechmos"],
        help="Neural MOS model (default: utmos)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ms",
        help="Language code for ASR (default: ms)"
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
    
    # Load synthesis results
    synthesis_data = load_synthesis_results(Path(args.synthesis_results))
    synthesis_results = synthesis_data.get('results', [])
    
    if not synthesis_results:
        logger.error("No synthesis results found")
        return
    
    # Initialize calculator
    calculator = TTSMetricsCalculator(
        mos_model=args.mos_model,
        asr_model=args.asr_model,
        device=args.device,
        language=args.language,
    )
    
    # Calculate metrics
    results = calculator.calculate_all_metrics(
        synthesis_results,
        calculate_mos=not args.no_mos,
        calculate_asr_wer=not args.no_asr_wer,
    )
    
    # Add metadata
    results['model'] = synthesis_data.get('model', 'Unknown')
    results['asr_model'] = args.asr_model
    results['mos_model'] = args.mos_model
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        # Remove audio array from samples before saving
        save_results = results.copy()
        if 'samples' in save_results:
            for s in save_results['samples']:
                s.pop('audio', None)
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved evaluation results to {results_file}")
    
    # Save summary CSV
    summary_file = output_dir / "evaluation_summary.csv"
    summary_data = {
        "model": [results.get('model')],
        "num_samples": [results.get('num_samples')],
        "mean_rtf": [results.get('latency', {}).get('mean_rtf')],
        "realtime_capable": [results.get('latency', {}).get('realtime_capable')],
        "mean_mos": [results.get('mos', {}).get('mean_mos')],
        "mean_wer": [results.get('asr_wer', {}).get('mean_wer')],
        "asr_model": [args.asr_model],
    }
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    logger.info(f"Saved evaluation summary to {summary_file}")
    
    # Print summary
    calculator.print_summary(results)


if __name__ == "__main__":
    main()

