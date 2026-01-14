#!/usr/bin/env python3
"""
Synthesizer for GLM-TTS - Zero-shot voice cloning with emotion control

GLM-TTS is a controllable, emotion-expressive zero-shot TTS system from Zhipu AI.
It uses LLM to generate speech tokens and Flow model to synthesize audio.

Repository: https://github.com/zai-org/GLM-TTS

Features:
- Zero-shot voice cloning with 3-10 seconds of prompt audio
- Emotion-expressive synthesis with RL-enhanced control
- Primarily supports Chinese with English mixed text

Installation:
    # Clone repo
    git clone https://github.com/zai-org/GLM-TTS.git
    cd GLM-TTS
    pip install -r requirements.txt
    
    # Download model
    mkdir -p ckpt
    huggingface-cli download zai-org/GLM-TTS --local-dir ckpt

Usage (Python):
    from synthesize_glmtts import GLMTTSSynthesizer
    
    synth = GLMTTSSynthesizer(speaker_wav="reference.wav")
    result = synth.synthesize("你好，这是一个测试。", output_path="output.wav")

Usage (CLI):
    python synthesize_glmtts.py --text "你好世界" --speaker-wav reference.wav --output output.wav
    
Note: For best results, use the GLM-TTS repo's inference script directly:
    cd GLM-TTS
    python glmtts_inference.py --data=example_zh --exp_name=test
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from threading import Lock

# Add GLM-TTS to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_GLMTTS_PATH = _SCRIPT_DIR / "GLM-TTS"
if _GLMTTS_PATH.exists() and str(_GLMTTS_PATH) not in sys.path:
    sys.path.insert(0, str(_GLMTTS_PATH))

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GLMTTSSynthesizer:
    """Synthesizer for GLM-TTS with zero-shot voice cloning
    
    Note: GLM-TTS has complex initialization. For production use,
    it's recommended to use the GLM-TTS repo's inference scripts directly.
    """
    
    MODEL_NAME = "GLM-TTS"
    SAMPLE_RATE = 22050
    
    def __init__(
        self,
        model_dir: str = "GLM-TTS/ckpt",
        device: str = "auto",
        speaker_wav: Optional[str] = None,
    ):
        """
        Initialize GLM-TTS synthesizer
        
        Args:
            model_dir: Path to model checkpoints (default: GLM-TTS/ckpt)
            device: Device to run on ('cuda', 'cpu', or 'auto')
            speaker_wav: Path to speaker reference audio (3-10s recommended)
        """
        self.model_dir = Path(model_dir)
        self.speaker_wav = speaker_wav
        self.lock = Lock()
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing GLM-TTS")
        logger.info(f"Model dir: {model_dir}")
        logger.info(f"Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load GLM-TTS model"""
        try:
            # Try to import GLM-TTS components
            from utils.tts_model_util import load_tts_model
            from cosyvoice.cli.frontend import CosyVoiceFrontEnd
            
            # Load model components
            self.model = load_tts_model(str(self.model_dir), self.device)
            logger.info("GLM-TTS model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"GLM-TTS not installed. Install from GitHub:\n"
                f"  git clone https://github.com/zai-org/GLM-TTS.git\n"
                f"  cd GLM-TTS\n"
                f"  pip install -r requirements.txt\n\n"
                f"Download model:\n"
                f"  mkdir -p ckpt\n"
                f"  huggingface-cli download zai-org/GLM-TTS --local-dir ckpt\n\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GLM-TTS model: {e}\n\n"
                f"For best results, use GLM-TTS inference script directly:\n"
                f"  cd GLM-TTS\n"
                f"  python glmtts_inference.py --data=example_zh --exp_name=test"
            )
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_wav: Optional[str] = None,
    ) -> Dict:
        """
        Synthesize speech from text with zero-shot voice cloning
        
        Args:
            text: Input text to synthesize (Chinese/English mixed supported)
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
                "Speaker reference audio is required. "
                "Provide speaker_wav parameter (3-10 seconds recommended)."
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
        """Synthesize using GLM-TTS"""
        try:
            # Use GLM-TTS inference
            audio = self.model.synthesize(
                text=text,
                prompt_wav=ref_wav,
            )
            return np.array(audio, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error in GLM-TTS synthesis: {e}")
            raise
    
    def _save_audio(self, audio: np.ndarray, output_path: Path):
        """Save audio to file"""
        import soundfile as sf
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.SAMPLE_RATE)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GLM-TTS zero-shot synthesis")
    parser.add_argument("--text", required=True, help="Text to synthesize (Chinese/English mixed)")
    parser.add_argument("--speaker-wav", required=True, help="Reference audio for voice cloning (3-10s)")
    parser.add_argument("--output", default="output_glmtts.wav", help="Output audio path")
    parser.add_argument("--model-dir", default="GLM-TTS/ckpt", help="Path to model checkpoints")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GLM-TTS Synthesizer")
    print("=" * 60)
    print()
    print("NOTE: For best results, use GLM-TTS inference script directly:")
    print("  cd GLM-TTS")
    print("  python glmtts_inference.py --data=example_zh --exp_name=test")
    print()
    print("=" * 60)
    
    synth = GLMTTSSynthesizer(
        model_dir=args.model_dir,
        speaker_wav=args.speaker_wav,
    )
    result = synth.synthesize(args.text, output_path=args.output)
    
    print(f"Synthesized: {result['audio_duration']:.2f}s in {result['synthesis_time']:.2f}s (RTF: {result['rtf']:.3f})")
    print(f"Output saved to: {args.output}")
