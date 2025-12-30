#!/usr/bin/env python3
"""
Synthesizer for MeloTTS

MeloTTS is a high-quality multi-lingual TTS model from MyShell.ai.
Supports English, Spanish, French, Chinese, Japanese, and Korean.

Requirements:
    pip install MeloTTS
    # or clone and install from: https://github.com/myshell-ai/MeloTTS

Usage:
    python synthesize_melotts.py \
        --test-dataset meso-malaya-test \
        --output-dir outputs/melotts_eval \
        --language EN \
        --speaker EN-US
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


class MeloTTSSynthesizer:
    """Synthesizer for MeloTTS model
    
    Target languages for this project:
    - English (en): Fully supported with multiple accents
    - Malay (ms): Uses English mode (EN-US or EN-INDIA for closer accent)
    - Code-switching (en-ms): Uses English mode for mixed sentences
    
    Note: MeloTTS doesn't support Malay. EN-INDIA accent may sound closer
    to Malaysian English due to similar prosodic patterns.
    """
    
    # Supported languages and their speaker IDs
    LANGUAGES = {
        "EN": ["EN-US", "EN-BR", "EN-INDIA", "EN-AU", "EN-Default"],
        "ES": ["ES"],
        "FR": ["FR"],
        "ZH": ["ZH"],
        "JP": ["JP"],
        "KR": ["KR"],
    }
    
    # Language fallbacks
    # For English, Malay, and code-switching: use English mode
    # EN-INDIA may sound closer to Malaysian English accent
    LANGUAGE_FALLBACK = {
        "ms": "EN",       # Malay -> English
        "en-ms": "EN",    # Code-switching -> English
        "id": "EN",       # Indonesian -> English
        "en": "EN",
        "en-us": "EN",
        "en-gb": "EN",
        "zh": "ZH",
        "ja": "JP",
        "ko": "KR",
        "es": "ES",
        "fr": "FR",
    }
    
    def __init__(
        self,
        language: str = "EN",
        speaker: str = "EN-US",
        device: str = "auto",
        speed: float = 1.0,
    ):
        """
        Initialize MeloTTS synthesizer
        
        Args:
            language: Language code (EN, ES, FR, ZH, JP, KR)
            speaker: Speaker ID (e.g., EN-US, EN-BR)
            device: Device to run on ('cuda', 'cpu', or 'auto')
            speed: Speech speed multiplier (1.0 = normal)
        """
        self.original_language = language
        self.speed = speed
        self.lock = Lock()
        
        # Map language if needed
        lang_upper = language.upper()
        if language.lower() in self.LANGUAGE_FALLBACK:
            self.language = self.LANGUAGE_FALLBACK[language.lower()]
        elif lang_upper in self.LANGUAGES:
            self.language = lang_upper
        else:
            self.language = "EN"
            logger.warning(f"Language '{language}' not supported, using 'EN'")
        
        # Validate speaker
        if speaker in self.LANGUAGES.get(self.language, []):
            self.speaker = speaker
        else:
            self.speaker = self.LANGUAGES[self.language][0]
            logger.warning(f"Speaker '{speaker}' not valid for {self.language}, using '{self.speaker}'")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "cpu"  # MeloTTS may have MPS issues
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading MeloTTS")
        logger.info(f"Language: {self.language}")
        logger.info(f"Speaker: {self.speaker}")
        logger.info(f"Device: {self.device}")
        
        try:
            from melo.api import TTS
            
            # Initialize TTS model
            self.tts = TTS(language=self.language, device=self.device)
            
            # Get speaker IDs
            self.speaker_ids = self.tts.hps.data.spk2id
            
            logger.info(f"Available speakers: {list(self.speaker_ids.keys())}")
            logger.info("MeloTTS loaded successfully")
            
        except ImportError:
            raise ImportError(
                "MeloTTS not installed. Install from: https://github.com/myshell-ai/MeloTTS"
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
                # Get speaker ID
                speaker_id = self.speaker_ids.get(
                    self.speaker, 
                    list(self.speaker_ids.values())[0]
                )
                
                if output_path:
                    # MeloTTS can save directly to file
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.tts.tts_to_file(
                        text,
                        speaker_id,
                        str(output_path),
                        speed=self.speed,
                    )
                    
                    # Load audio back for duration calculation
                    import soundfile as sf
                    audio, sample_rate = sf.read(str(output_path))
                    audio = np.array(audio, dtype=np.float32)
                else:
                    # Generate audio in memory
                    # MeloTTS returns audio as numpy array
                    audio = self.tts.tts_to_file(
                        text,
                        speaker_id,
                        None,  # Don't save to file
                        speed=self.speed,
                    )
                    sample_rate = self.tts.hps.data.sampling_rate
                    
                    if audio is None:
                        # Fallback: save to temp file and read back
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                            temp_path = f.name
                        
                        self.tts.tts_to_file(
                            text,
                            speaker_id,
                            temp_path,
                            speed=self.speed,
                        )
                        
                        import soundfile as sf
                        audio, sample_rate = sf.read(temp_path)
                        audio = np.array(audio, dtype=np.float32)
                        
                        Path(temp_path).unlink()
                
            except Exception as e:
                logger.error(f"Error synthesizing text: {e}")
                raise
        
        synthesis_time = time.time() - start_time
        
        # Calculate audio duration
        sample_rate = getattr(self.tts.hps.data, 'sampling_rate', 44100)
        audio_duration = len(audio) / sample_rate
        rtf = calculate_rtf(synthesis_time, audio_duration)
        
        result = {
            "audio": audio,
            "sample_rate": sample_rate,
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        
        if output_path:
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
        description="Synthesize speech using MeloTTS"
    )
    
    parser.add_argument(
        "--language",
        default="EN",
        help="Language code (EN, ES, FR, ZH, JP, KR)"
    )
    parser.add_argument(
        "--speaker",
        default="EN-US",
        help="Speaker ID (e.g., EN-US, EN-BR, ZH)"
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
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
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
        f"MeloTTS-{args.speaker}",
        args.test_dataset,
    )
    
    # Initialize synthesizer
    synthesizer = MeloTTSSynthesizer(
        language=args.language,
        speaker=args.speaker,
        device=args.device,
        speed=args.speed,
    )
    
    # Synthesize all texts
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting synthesis on {len(texts)} samples")
    logger.info(f"Language: {synthesizer.language}")
    logger.info(f"Speaker: {synthesizer.speaker}")
    logger.info(f"Device: {synthesizer.device}")
    logger.info(f"{'='*70}\n")
    
    results = synthesizer.synthesize_batch(texts, output_dir)
    
    # Save results
    save_synthesis_results(results, output_dir, f"MeloTTS-{args.speaker}")
    
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

