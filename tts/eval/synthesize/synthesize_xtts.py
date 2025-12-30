#!/usr/bin/env python3
"""
Synthesizer for XTTS v2 (Coqui TTS)

XTTS v2 is a multilingual text-to-speech model with voice cloning capabilities.
It supports 17 languages including Malay-adjacent languages.

Requirements:
    pip install TTS>=0.22.0

Usage:
    python synthesize_xtts.py \
        --test-dataset meso-malaya-test \
        --output-dir outputs/xtts_eval \
        --speaker-wav reference.wav \
        --language ms
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
    get_audio_duration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XTTSSynthesizer:
    """Synthesizer for XTTS v2 model
    
    Target languages for this project:
    - English (en): Fully supported
    - Malay (ms): Uses English as fallback (phonetically reasonable)
    - Code-switching (en-ms): Uses English mode for mixed sentences
    
    Note: XTTS v2 does not natively support Malay, but English mode works
    reasonably well for Malay text due to similar phoneme structures.
    For production, consider fine-tuning on Malaysian English/Malay data.
    """
    
    # XTTS v2 supported languages
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ]
    
    # Language mapping for unsupported languages
    # For English-Malay code-switching, we use English mode
    LANGUAGE_FALLBACK = {
        "ms": "en",      # Malay -> English (works reasonably for Malay)
        "en-ms": "en",   # Code-switching -> English (handles mixed sentences)
        "id": "en",      # Indonesian -> English
    }
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ):
        """
        Initialize XTTS v2 synthesizer
        
        Args:
            model_name: XTTS model name or path
            device: Device to run on ('cuda', 'cpu', or 'auto')
            speaker_wav: Path to speaker reference audio for voice cloning
            language: Target language code
        """
        self.model_name = model_name
        self.speaker_wav = speaker_wav
        self.original_language = language
        self.lock = Lock()
        
        # Handle language fallback for unsupported languages
        if language in self.LANGUAGE_FALLBACK:
            self.language = self.LANGUAGE_FALLBACK[language]
            logger.warning(
                f"Language '{language}' not directly supported by XTTS. "
                f"Using '{self.language}' as fallback."
            )
        else:
            self.language = language
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "cpu"  # XTTS has issues with MPS
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading XTTS v2 model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Language: {self.language}")
        
        # Load TTS model
        try:
            from TTS.api import TTS
            
            self.tts = TTS(model_name=model_name).to(self.device)
            
            # Get default speaker embedding if no speaker wav provided
            if speaker_wav:
                self.speaker_wav = str(speaker_wav)
                logger.info(f"Using speaker reference: {speaker_wav}")
            else:
                logger.info("No speaker reference provided, using default voice")
            
            logger.info("XTTS v2 model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "TTS library not installed. Install with: pip install TTS>=0.22.0"
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
            Dictionary with synthesis results:
            - audio: numpy array of audio
            - audio_path: path to saved audio (if output_path provided)
            - synthesis_time: time taken for synthesis
            - audio_duration: duration of synthesized audio
            - rtf: real-time factor
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
                if self.speaker_wav:
                    # Use voice cloning with speaker reference
                    wav = self.tts.tts(
                        text=text,
                        speaker_wav=self.speaker_wav,
                        language=self.language,
                    )
                else:
                    # Use default speaker
                    wav = self.tts.tts(
                        text=text,
                        language=self.language,
                    )
                
                # Convert to numpy array
                audio = np.array(wav, dtype=np.float32)
                
            except Exception as e:
                logger.error(f"Error synthesizing text: {e}")
                raise
        
        synthesis_time = time.time() - start_time
        
        # Calculate audio duration (XTTS uses 24kHz sample rate)
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
        description="Synthesize speech using XTTS v2"
    )
    
    parser.add_argument(
        "--model", 
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="XTTS model name or path"
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
        "--speaker-wav",
        help="Path to speaker reference audio for voice cloning"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Target language code (default: en)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
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
        "XTTS-v2",
        args.test_dataset,
    )
    
    # Initialize synthesizer
    synthesizer = XTTSSynthesizer(
        model_name=args.model,
        device=args.device,
        speaker_wav=args.speaker_wav,
        language=args.language,
    )
    
    # Synthesize all texts
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting synthesis on {len(texts)} samples")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {synthesizer.device}")
    logger.info(f"Language: {synthesizer.language}")
    logger.info(f"{'='*70}\n")
    
    results = synthesizer.synthesize_batch(texts, output_dir)
    
    # Save results
    save_synthesis_results(results, output_dir, "XTTS-v2")
    
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

