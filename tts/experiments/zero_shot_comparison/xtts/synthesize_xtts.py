#!/usr/bin/env python3
"""
Synthesizer for XTTS v2 (Coqui TTS) - Zero-shot voice cloning

XTTS v2 is a multilingual text-to-speech model with voice cloning capabilities.
It uses a short audio reference (~6s) to clone the voice.

Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
Note: Malay (ms) is NOT natively supported - falls back to English.

Requirements:
    pip install -r requirements.txt

Usage (Python):
    from synthesize_xtts import XTTSSynthesizer
    
    synth = XTTSSynthesizer(speaker_wav="reference.wav", language="ms")
    result = synth.synthesize("Hello, this is a test.", output_path="output.wav")

Usage (CLI):
    # From text file:
    python synthesize_xtts.py --input sentences.txt --speaker-wav reference.wav --output output.wav --language ms
    
    # Direct text:
    python synthesize_xtts.py --text "Hello world" --speaker-wav reference.wav --output output.wav --language ms
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from threading import Lock

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XTTSSynthesizer:
    """Synthesizer for XTTS v2 model with zero-shot voice cloning"""
    
    MODEL_NAME = "coqui/XTTS-v2"
    SAMPLE_RATE = 24000
    
    # Language mapping for unsupported languages
    LANGUAGE_FALLBACK = {
        "ms": "en",      # Malay -> English (works reasonably for Malay)
        "en-ms": "en",   # Code-switching -> English
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
            speaker_wav: Path to speaker reference audio for voice cloning (6-30s recommended)
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
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading XTTS v2 model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Language: {self.language}")
        
        self._load_model()
    
    def _load_model(self):
        """Load XTTS model"""
        try:
            # Fix for PyTorch 2.6+ weights_only issue
            # TTS library uses torch.load without weights_only=False, causing errors
            import torch
            _original_torch_load = torch.load
            
            def _patched_torch_load(*args, **kwargs):
                # Force weights_only=False for TTS model loading
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _patched_torch_load
            
            try:
                from TTS.api import TTS
                self.tts = TTS(model_name=self.model_name).to(self.device)
            finally:
                # Restore original torch.load
                torch.load = _original_torch_load
            
            if self.speaker_wav:
                self.speaker_wav = str(self.speaker_wav)
                logger.info(f"Using speaker reference: {self.speaker_wav}")
            else:
                logger.info("No speaker reference provided - will use default voice")
            
            logger.info("XTTS v2 model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"TTS library not installed. Install with:\n"
                f"  pip install -r requirements.txt\n\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS model: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_wav: Optional[str] = None,
    ) -> Dict:
        """
        Synthesize speech from text with zero-shot voice cloning
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            speaker_wav: Optional override for speaker reference audio
            
        Returns:
            Dictionary with synthesis results
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        ref_wav = speaker_wav or self.speaker_wav
        
        if not ref_wav:
            raise ValueError(
                "XTTS v2 requires a speaker reference audio for synthesis. "
                "Provide --speaker-wav argument with a 6-30 second WAV file."
            )
        
        if not Path(ref_wav).exists():
            raise FileNotFoundError(f"Speaker reference audio not found: {ref_wav}")
        
        start_time = time.time()
        
        with self.lock:
            audio = self._synthesize(text, ref_wav)
        
        synthesis_time = time.time() - start_time
        audio_duration = len(audio) / self.SAMPLE_RATE
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0.0
        
        result = {
            "audio": audio,
            "sample_rate": self.SAMPLE_RATE,
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        
        if output_path:
            output_path = Path(output_path)
            self._save_audio(audio, output_path)
            result["audio_path"] = str(output_path)
        
        return result
    
    def _synthesize(self, text: str, ref_wav: str) -> np.ndarray:
        """Synthesize using XTTS model with voice cloning"""
        try:
            # Convert to WAV if needed (XTTS works best with WAV)
            ref_path = self._prepare_reference_audio(ref_wav)
            
            wav = self.tts.tts(
                text=text,
                speaker_wav=ref_path,
                language=self.language,
            )
            return np.array(wav, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            raise
    
    def _prepare_reference_audio(self, audio_path: str) -> str:
        """Prepare reference audio - convert to WAV if needed"""
        import librosa
        import soundfile as sf
        import tempfile
        
        audio_path = Path(audio_path)
        
        # If already WAV, return as-is
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path)
        
        # Convert to WAV using librosa
        if not hasattr(self, '_temp_wav_path'):
            self._temp_wav_dir = tempfile.mkdtemp()
        
        temp_wav = Path(self._temp_wav_dir) / f"ref_{audio_path.stem}.wav"
        
        if not temp_wav.exists():
            logger.info(f"Converting {audio_path.suffix} to WAV...")
            audio, sr = librosa.load(str(audio_path), sr=22050)
            sf.write(str(temp_wav), audio, sr)
        
        return str(temp_wav)
    
    def _save_audio(self, audio: np.ndarray, output_path: Path):
        """Save audio to file"""
        import soundfile as sf
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.SAMPLE_RATE)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XTTS v2 zero-shot synthesis")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize (direct input)")
    text_group.add_argument("--input", help="Path to text file to synthesize")
    parser.add_argument("--speaker-wav", required=True, help="Reference audio for voice cloning (6-30s WAV)")
    parser.add_argument("--output", default="output_xtts.wav", help="Output audio path")
    parser.add_argument("--language", default="ms", help="Language code (default: ms for Malay)")
    
    args = parser.parse_args()
    
    # Get text from file or direct input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        text = input_path.read_text(encoding="utf-8").strip()
        logger.info(f"Read text from file: {args.input}")
    else:
        text = args.text
    
    if not text:
        raise ValueError("No text provided (empty file or empty --text)")
    
    synth = XTTSSynthesizer(speaker_wav=args.speaker_wav, language=args.language)
    result = synth.synthesize(text, output_path=args.output)
    
    print(f"Synthesized: {result['audio_duration']:.2f}s in {result['synthesis_time']:.2f}s (RTF: {result['rtf']:.3f})")
    print(f"Output saved to: {args.output}")
