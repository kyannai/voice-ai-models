#!/usr/bin/env python3
"""
Synthesizer for Kokoro TTS

Kokoro is a lightweight, fast TTS model with good quality.
Uses StyleTTS2-inspired architecture with efficient inference.

Requirements:
    pip install kokoro>=0.3.0 misaki[en,ja,ko,zh]>=0.1.0

Usage:
    python synthesize_kokoro.py \
        --test-dataset meso-malaya-test \
        --output-dir outputs/kokoro_eval \
        --voice af_heart \
        --language en-us
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from threading import Lock

import torch
import numpy as np
from tqdm import tqdm

from utils import (
    load_dataset_by_name,
    save_audio,
    save_synthesis_results,
    create_output_dir,
    calculate_rtf,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KokoroSynthesizer:
    """Synthesizer for Kokoro TTS model
    
    Target languages for this project:
    - English (en): Fully supported with multiple voices
    - Malay (ms): Uses English mode (phonetically similar)
    - Code-switching (en-ms): Uses English mode for mixed sentences
    
    Note: Kokoro doesn't natively support Malay. English mode is used
    as a fallback which works reasonably for Malay/Malaysian English.
    """
    
    # Available voices in Kokoro
    AVAILABLE_VOICES = [
        # American English (recommended for Malaysian English accent)
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael",
        # British English
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis",
        # Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo",
        # Korean
        "kf_001", "kf_002", "kf_003", "kf_004",
        # Chinese
        "zf_001", "zf_002", "zf_003", "zf_004",
        "zm_001", "zm_002",
    ]
    
    # Language code to pipeline language mapping
    # For English, Malay, and code-switching: use English mode
    LANGUAGE_MAP = {
        "en": "en-us",
        "en-us": "en-us",
        "en-gb": "en-gb",
        "ms": "en-us",      # Malay -> English (phonetically similar)
        "en-ms": "en-us",   # Code-switching -> English
        "ja": "ja",
        "ko": "ko",
        "zh": "zh",
    }
    
    def __init__(
        self,
        voice: str = "af_heart",
        language: str = "en-us",
        device: str = "auto",
        speed: float = 1.0,
    ):
        """
        Initialize Kokoro TTS synthesizer
        
        Args:
            voice: Voice name (see AVAILABLE_VOICES)
            language: Language code (en-us, en-gb, ja, ko, zh)
            device: Device to run on ('cuda', 'cpu', or 'auto')
            speed: Speech speed multiplier (1.0 = normal)
        """
        self.voice = voice
        self.original_language = language
        self.speed = speed
        self.lock = Lock()
        
        # Map language
        if language in self.LANGUAGE_MAP:
            self.language = self.LANGUAGE_MAP[language]
        else:
            self.language = "en-us"
            logger.warning(f"Language '{language}' not supported, using 'en-us'")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Kokoro TTS")
        logger.info(f"Voice: {voice}")
        logger.info(f"Language: {self.language}")
        logger.info(f"Device: {self.device}")
        
        try:
            from kokoro import KPipeline
            
            # Initialize pipeline with language
            self.pipeline = KPipeline(lang_code=self.language, device=self.device)
            
            logger.info("Kokoro TTS loaded successfully")
            
        except ImportError:
            raise ImportError(
                "Kokoro not installed. Install with: pip install kokoro>=0.3.0 misaki[en,ja,ko,zh]"
            )
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Synthesize speech from text
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Dictionary with synthesis results
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning silence")
            return {
                "audio": np.zeros(1000, dtype=np.float32),
                "synthesis_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }
        
        start_time = time.time()
        
        with self.lock:
            try:
                # Kokoro returns generator of (graphemes, phonemes, audio) tuples
                # We need to concatenate all audio segments
                audio_segments = []
                
                for _, _, audio in self.pipeline(
                    text,
                    voice=self.voice,
                    speed=self.speed,
                ):
                    if audio is not None:
                        audio_segments.append(audio)
                
                if audio_segments:
                    # Concatenate all segments
                    audio = np.concatenate(audio_segments)
                else:
                    audio = np.zeros(1000, dtype=np.float32)
                
            except Exception as e:
                logger.error(f"Error synthesizing text: {e}")
                raise
        
        synthesis_time = time.time() - start_time
        
        # Kokoro uses 24kHz sample rate
        sample_rate = 24000
        audio_duration = len(audio) / sample_rate
        rtf = calculate_rtf(synthesis_time, audio_duration)
        
        result = {
            "audio": audio,
            "sample_rate": sample_rate,
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        
        # Save audio if output path provided
        if output_path:
            output_path = Path(output_path)
            save_audio(audio, output_path, sample_rate=sample_rate)
            result["audio_path"] = str(output_path)
        
        return result
    
    def synthesize_batch(
        self,
        texts: List[Dict],
        output_dir: Union[str, Path],
        filename_prefix: str = "tts",
    ) -> List[Dict]:
        """
        Synthesize batch of texts
        
        Args:
            texts: List of text dictionaries with 'id' and 'text' keys
            output_dir: Directory to save audio files
            filename_prefix: Prefix for audio filenames
            
        Returns:
            List of synthesis result dictionaries
        """
        output_dir = Path(output_dir)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for item in tqdm(texts, desc="Synthesizing"):
            text_id = item.get('id', len(results))
            text = item.get('text', '')
            
            # Generate output filename
            output_path = audio_dir / f"{filename_prefix}_{text_id:04d}.wav"
            
            try:
                result = self.synthesize(text, output_path)
                
                results.append({
                    "id": text_id,
                    "text": text,
                    "audio_path": str(output_path),
                    "synthesis_time": result["synthesis_time"],
                    "audio_duration": result["audio_duration"],
                    "rtf": result["rtf"],
                })
                
            except Exception as e:
                logger.error(f"Failed to synthesize sample {text_id}: {e}")
                results.append({
                    "id": text_id,
                    "text": text,
                    "error": str(e),
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using Kokoro TTS"
    )
    
    parser.add_argument(
        "--voice",
        default="af_heart",
        help=f"Voice name (options: {', '.join(KokoroSynthesizer.AVAILABLE_VOICES[:5])}...)"
    )
    parser.add_argument(
        "--test-dataset",
        required=True,
        help="Dataset name from registry (e.g., meso-malaya-test)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for synthesized audio"
    )
    parser.add_argument(
        "--language",
        default="en-us",
        help="Language code (en-us, en-gb, ja, ko, zh)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to synthesize"
    )
    
    args = parser.parse_args()
    
    # Load text data
    logger.info(f"Loading dataset: {args.test_dataset}")
    texts = load_dataset_by_name(args.test_dataset, max_samples=args.max_samples)
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Create output directory
    output_dir = create_output_dir(
        args.output_dir,
        f"Kokoro-{args.voice}",
        args.test_dataset,
    )
    
    # Initialize synthesizer
    synthesizer = KokoroSynthesizer(
        voice=args.voice,
        language=args.language,
        device=args.device,
        speed=args.speed,
    )
    
    # Synthesize all texts
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting synthesis on {len(texts)} samples")
    logger.info(f"Voice: {args.voice}")
    logger.info(f"Device: {synthesizer.device}")
    logger.info(f"{'='*70}\n")
    
    results = synthesizer.synthesize_batch(texts, output_dir)
    
    # Save results
    save_synthesis_results(results, output_dir, f"Kokoro-{args.voice}")
    
    # Print summary
    successful = [r for r in results if 'error' not in r]
    avg_rtf = sum(r['rtf'] for r in successful) / len(successful) if successful else 0
    total_audio = sum(r['audio_duration'] for r in successful)
    total_time = sum(r['synthesis_time'] for r in successful)
    
    logger.info(f"\n{'='*70}")
    logger.info("SYNTHESIS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    logger.info(f"Total audio duration: {total_audio:.2f}s")
    logger.info(f"Total synthesis time: {total_time:.2f}s")
    logger.info(f"Average RTF: {avg_rtf:.3f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()

