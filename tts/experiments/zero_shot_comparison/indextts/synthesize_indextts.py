#!/usr/bin/env python3
"""
Synthesizer for IndexTTS2 - Zero-shot voice cloning with emotion control

IndexTTS2 is an industrial-level zero-shot TTS system with emotion control.
It can clone a voice from a single audio sample.

Repository: https://github.com/index-tts/index-tts

Supported languages: Chinese, English (and mixed)
Note: Works well for Malay text using English phonetics.

Installation:
    pip install -r requirements.txt
    
    git clone https://github.com/index-tts/index-tts.git
    cd index-tts
    git lfs pull
    
    # Install uv package manager
    pip install -U uv
    
    # Install dependencies
    uv sync --all-extras
    
    # Download model from HuggingFace
    uv tool install "huggingface-hub[cli,hf_xet]"
    hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

Usage (Python):
    from synthesize_indextts import IndexTTSSynthesizer
    
    synth = IndexTTSSynthesizer(speaker_wav="reference.wav")
    result = synth.synthesize("Hello, this is a test.", output_path="output.wav")

Usage (CLI):
    # From text file:
    python synthesize_indextts.py --input sentences.txt --speaker-wav reference.wav --output output.wav
    
    # Direct text:
    python synthesize_indextts.py --text "Hello world" --speaker-wav reference.wav --output output.wav
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from threading import Lock

# Add local index-tts installation to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_INDEXTTS_PATH = _SCRIPT_DIR / "index-tts"
if _INDEXTTS_PATH.exists() and str(_INDEXTTS_PATH) not in sys.path:
    sys.path.insert(0, str(_INDEXTTS_PATH))

# Default paths relative to this script
_DEFAULT_MODEL_DIR = _SCRIPT_DIR / "checkpoints"

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexTTSSynthesizer:
    """Synthesizer for IndexTTS2 model with zero-shot voice cloning"""
    
    MODEL_NAME = "IndexTTS-2"
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        cfg_path: Optional[str] = None,
        device: str = "auto",
        speaker_wav: Optional[str] = None,
        use_fp16: bool = False,
    ):
        """
        Initialize IndexTTS2 synthesizer
        
        Args:
            model_dir: Path to model checkpoints directory (default: ./checkpoints)
            cfg_path: Path to config file (default: {model_dir}/config.yaml)
            device: Device to run on ('cuda', 'cpu', 'mps', or 'auto')
            speaker_wav: Path to speaker reference audio for voice cloning
            use_fp16: Use FP16 inference (faster, less VRAM)
        """
        # Use default model directory if not provided
        if model_dir is None:
            self.model_dir = str(_DEFAULT_MODEL_DIR)
        else:
            self.model_dir = model_dir
        self.cfg_path = cfg_path or f"{self.model_dir}/config.yaml"
        self.speaker_wav = speaker_wav
        self.use_fp16 = use_fp16
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
        
        logger.info(f"Loading IndexTTS2 model from {self.model_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"FP16: {use_fp16}")
        
        self._load_model()
    
    def _load_model(self):
        """Load IndexTTS2 model"""
        try:
            from indextts.infer_v2 import IndexTTS2
        except ImportError as e:
            raise ImportError(
                f"IndexTTS not installed. Install from GitHub:\n"
                f"  git clone https://github.com/index-tts/index-tts.git\n"
                f"  cd index-tts\n"
                f"  git lfs pull\n"
                f"  pip install -U uv\n"
                f"  uv sync --all-extras\n"
                f"\n"
                f"Then download the model:\n"
                f"  hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints\n"
                f"\n"
                f"Or install missing dependencies:\n"
                f"  pip install -r requirements.txt\n"
                f"\n"
                f"Original error: {e}"
            )
        
        try:
            self.model = IndexTTS2(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                use_fp16=self.use_fp16,
                use_cuda_kernel=False,
                use_deepspeed=False,
            )
            self._version = "v2"
            logger.info("IndexTTS2 model loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load IndexTTS2 model: {e}\n\n"
                f"Make sure all dependencies are installed:\n"
                f"  pip install descript-audiotools\n"
                f"  pip install -r requirements.txt"
            )
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_wav: Optional[str] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
    ) -> Dict:
        """
        Synthesize speech from text with zero-shot voice cloning
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            speaker_wav: Override for speaker reference audio
            emo_audio_prompt: Optional emotional reference audio
            emo_alpha: Emotion strength (0.0-1.0, default 1.0)
            
        Returns:
            Dictionary with synthesis results
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        ref_wav = speaker_wav or self.speaker_wav
        if not ref_wav:
            raise ValueError(
                "Speaker reference audio is required for IndexTTS zero-shot synthesis. "
                "Provide speaker_wav parameter."
            )
        
        # Validate reference audio exists
        ref_wav = self._prepare_reference_audio(ref_wav)
        
        start_time = time.time()
        
        with self.lock:
            audio = self._synthesize(text, ref_wav, output_path, emo_audio_prompt, emo_alpha)
        
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
            result["audio_path"] = str(output_path)
        
        return result
    
    def _prepare_reference_audio(self, audio_path: str) -> str:
        """Prepare reference audio - convert to WAV and resample to 24kHz if needed"""
        import librosa
        import soundfile as sf
        import tempfile
        
        TARGET_SR = 24000  # IndexTTS expects 24kHz
        
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Speaker reference audio not found: {audio_path}")
        
        # Check current sample rate
        info = sf.info(str(audio_path))
        current_sr = info.samplerate
        
        # If already 24kHz WAV, return as-is
        if audio_path.suffix.lower() == '.wav' and current_sr == TARGET_SR:
            return str(audio_path)
        
        # Need to resample - create temp directory
        if not hasattr(self, '_temp_wav_dir'):
            self._temp_wav_dir = tempfile.mkdtemp()
        
        temp_wav = Path(self._temp_wav_dir) / f"ref_{audio_path.stem}_{TARGET_SR}.wav"
        
        if not temp_wav.exists():
            if current_sr != TARGET_SR:
                logger.info(f"Resampling reference audio from {current_sr}Hz to {TARGET_SR}Hz...")
            else:
                logger.info(f"Converting {audio_path.suffix} to WAV...")
            
            # Load and resample to target sample rate
            audio, sr = librosa.load(str(audio_path), sr=TARGET_SR)
            sf.write(str(temp_wav), audio, TARGET_SR)
            logger.info(f"Saved resampled reference to: {temp_wav}")
        
        return str(temp_wav)
    
    def _synthesize(
        self, 
        text: str, 
        ref_wav: str,
        output_path: Optional[Union[str, Path]] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
    ) -> np.ndarray:
        """Synthesize using IndexTTS"""
        import soundfile as sf
        import tempfile
        
        try:
            # Prepare output path
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_file = str(output_path)
            else:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                out_file = temp_file.name
                temp_file.close()
            
            # Call IndexTTS
            if self._version == "v2":
                # IndexTTS2 API
                if emo_audio_prompt:
                    self.model.infer(
                        spk_audio_prompt=ref_wav,
                        text=text,
                        output_path=out_file,
                        emo_audio_prompt=emo_audio_prompt,
                        emo_alpha=emo_alpha,
                        verbose=False,
                    )
                else:
                    self.model.infer(
                        spk_audio_prompt=ref_wav,
                        text=text,
                        output_path=out_file,
                        verbose=False,
                    )
            else:
                # IndexTTS v1 API
                self.model.infer(ref_wav, text, out_file)
            
            # Load the generated audio
            audio, sr = sf.read(out_file)
            self.SAMPLE_RATE = sr
            
            # Clean up temp file if needed
            if not output_path:
                Path(out_file).unlink()
            
            return np.array(audio, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error in IndexTTS synthesis: {e}")
            raise
    
    def _save_audio(self, audio: np.ndarray, output_path: Path):
        """Save audio to file"""
        import soundfile as sf
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.SAMPLE_RATE)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS2 zero-shot synthesis")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize (direct input)")
    text_group.add_argument("--input", help="Path to text file to synthesize")
    parser.add_argument("--speaker-wav", required=True, help="Reference audio for voice cloning")
    parser.add_argument("--output", default="output_indextts.wav", help="Output audio path")
    parser.add_argument("--model-dir", default=None, help=f"Path to model checkpoints (default: {_DEFAULT_MODEL_DIR})")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference")
    
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
    
    synth = IndexTTSSynthesizer(
        model_dir=args.model_dir,  # None uses default path
        speaker_wav=args.speaker_wav,
        use_fp16=args.fp16,
    )
    result = synth.synthesize(text, output_path=args.output)
    
    print(f"Synthesized: {result['audio_duration']:.2f}s in {result['synthesis_time']:.2f}s (RTF: {result['rtf']:.3f})")
    print(f"Output saved to: {args.output}")
