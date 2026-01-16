#!/usr/bin/env python3
"""
Synthesizer for Chatterbox-Turbo (Resemble AI) - Zero-shot voice cloning

Chatterbox-Turbo is a 350M parameter TTS model optimized for low-latency voice agents.
It supports paralinguistic tags like [laugh], [chuckle], [cough] for added realism.

Requirements:
    pip install -r requirements.txt

Usage (Python):
    from synthesize_chatterbox import ChatterboxSynthesizer
    
    synth = ChatterboxSynthesizer(speaker_wav="reference.wav")
    result = synth.synthesize("Hello, this is a test.", output_path="output.wav")

Usage (CLI):
    # From text file:
    python synthesize_chatterbox.py --input sentences.txt --speaker-wav reference.wav --output output.wav
    
    # Direct text:
    python synthesize_chatterbox.py --text "Hello world" --speaker-wav reference.wav --output output.wav
    
    # With paralinguistic tags:
    python synthesize_chatterbox.py --text "That's hilarious [laugh]" --speaker-wav reference.wav --output output.wav
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


class ChatterboxSynthesizer:
    """Synthesizer for Chatterbox-Turbo model with zero-shot voice cloning"""
    
    SAMPLE_RATE = 24000  # Chatterbox outputs at 24kHz
    
    def __init__(
        self,
        device: str = "auto",
        speaker_wav: Optional[str] = None,
    ):
        """
        Initialize Chatterbox-Turbo synthesizer
        
        Args:
            device: Device to run on ('cuda', 'cpu', or 'auto')
            speaker_wav: Path to speaker reference audio for voice cloning (~10s recommended)
        """
        self.speaker_wav = speaker_wav
        self.lock = Lock()
        
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
        
        logger.info(f"Loading Chatterbox-Turbo model")
        logger.info(f"Device: {self.device}")
        
        self._load_model()
    
    def _patch_perth_watermarker(self):
        """
        Patch the perth watermarker if it's not properly loaded.
        This is a known issue where perth.PerthImplicitWatermarker can be None.
        """
        try:
            import perth
            if getattr(perth, "PerthImplicitWatermarker", None) is None:
                logger.warning("Perth watermarker not available, using dummy watermarker")
                
                class DummyWatermarker:
                    """No-op watermarker for when perth isn't properly installed"""
                    def __call__(self, audio, **kwargs):
                        return audio
                    
                    def embed(self, audio, **kwargs):
                        return audio
                    
                    def apply_watermark(self, audio, **kwargs):
                        return audio
                    
                    def get_watermark(self, audio, **kwargs):
                        return 0.0
                
                perth.PerthImplicitWatermarker = DummyWatermarker
        except ImportError:
            logger.warning("Perth package not installed, watermarking will be disabled")
    
    def _load_model(self):
        """Load Chatterbox-Turbo model"""
        try:
            # Fix for perth watermarker issue - patch if not properly loaded
            self._patch_perth_watermarker()
            
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self.SAMPLE_RATE = self.model.sr  # Use model's sample rate
            
            if self.speaker_wav:
                self.speaker_wav = str(self.speaker_wav)
                logger.info(f"Using speaker reference: {self.speaker_wav}")
            else:
                logger.info("No speaker reference provided - will require one at synthesis time")
            
            logger.info("Chatterbox-Turbo model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"chatterbox-tts library not installed. Install with:\n"
                f"  pip install chatterbox-tts\n\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox-Turbo model: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_wav: Optional[str] = None,
    ) -> Dict:
        """
        Synthesize speech from text with zero-shot voice cloning
        
        Args:
            text: Input text to synthesize (supports paralinguistic tags like [laugh], [chuckle])
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
                "Chatterbox-Turbo requires a speaker reference audio for synthesis. "
                "Provide --speaker-wav argument with a ~10 second WAV file."
            )
        
        if not Path(ref_wav).exists():
            raise FileNotFoundError(f"Speaker reference audio not found: {ref_wav}")
        
        start_time = time.time()
        
        with self.lock:
            audio = self._synthesize(text, ref_wav)
        
        synthesis_time = time.time() - start_time
        audio_duration = audio.shape[-1] / self.SAMPLE_RATE
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
    
    def _synthesize(self, text: str, ref_wav: str) -> torch.Tensor:
        """Synthesize using Chatterbox-Turbo model with voice cloning"""
        try:
            # Convert reference audio if needed
            ref_path = self._prepare_reference_audio(ref_wav)
            
            # Generate speech
            wav = self.model.generate(text, audio_prompt_path=ref_path)
            
            return wav
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
        if not hasattr(self, '_temp_wav_dir'):
            self._temp_wav_dir = tempfile.mkdtemp()
        
        temp_wav = Path(self._temp_wav_dir) / f"ref_{audio_path.stem}.wav"
        
        if not temp_wav.exists():
            logger.info(f"Converting {audio_path.suffix} to WAV...")
            audio, sr = librosa.load(str(audio_path), sr=22050)
            sf.write(str(temp_wav), audio, sr)
        
        return str(temp_wav)
    
    def _save_audio(self, audio: torch.Tensor, output_path: Path):
        """Save audio to file"""
        import torchaudio as ta
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), audio, self.SAMPLE_RATE)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatterbox-Turbo zero-shot synthesis")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize (direct input)")
    text_group.add_argument("--input", help="Path to text file to synthesize")
    parser.add_argument("--speaker-wav", required=True, help="Reference audio for voice cloning (~10s WAV)")
    parser.add_argument("--output", default="output_chatterbox.wav", help="Output audio path")
    
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
    
    synth = ChatterboxSynthesizer(speaker_wav=args.speaker_wav)
    result = synth.synthesize(text, output_path=args.output)
    
    print(f"Synthesized: {result['audio_duration']:.2f}s in {result['synthesis_time']:.2f}s (RTF: {result['rtf']:.3f})")
    print(f"Output saved to: {args.output}")
