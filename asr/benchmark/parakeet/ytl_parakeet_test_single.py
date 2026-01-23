#!/usr/bin/env python3
"""
Single audio file transcription using Parakeet ASR.

Usage:
    python ytl_parakeet_test_single.py -a /path/to/audio.wav
    python ytl_parakeet_test_single.py -a /path/to/audio.mp3 -m nvidia/parakeet-tdt-0.6b-v2
    python ytl_parakeet_test_single.py -a /path/to/audio.mp3 -m nvidia/parakeet-tdt-0.6b-v3  # Multilingual
    python ytl_parakeet_test_single.py -a /path/to/audio.wav -m /path/to/model.nemo
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


class ParakeetRecognizer:
    """NVIDIA Parakeet model-based ASR recognizer using NeMo."""
    
    def __init__(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v2"):
        import nemo.collections.asr as nemo_asr
        
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NeMo Parakeet model: {model_id}")
        print(f"Device: {self.device}")
        
        is_local_nemo = model_id.endswith('.nemo') and os.path.exists(model_id)
        
        if is_local_nemo:
            print(f"Loading from local .nemo file: {model_id}")
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
        else:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required for HuggingFace models. "
                    "Please add it to .env file or export it: export HF_TOKEN=your_token_here"
                )
            from huggingface_hub import login
            login(token=hf_token)
            print("Loading model from HuggingFace...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        
        print(f"Moving model to {self.device} and setting eval mode...")
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model.cfg, 'decoding'):
            if self.model.cfg.decoding.strategy != "beam":
                self.model.cfg.decoding.strategy = "greedy_batch"
                self.model.change_decoding_strategy(self.model.cfg.decoding)
        
        print(f"Model loaded successfully on {self.device}")
    
    def transcribe(self, wav_path: str) -> str:
        """Transcribe a single audio file."""
        import librosa
        
        audio, sr = librosa.load(wav_path, sr=16000)
        audio_tensor = torch.from_numpy(audio)
        
        with torch.inference_mode():
            with torch.no_grad():
                output = self.model.transcribe([audio_tensor])
        
        if isinstance(output, tuple):
            output = output[0]
        
        if isinstance(output, list) and len(output) > 0:
            first_result = output[0]
            if hasattr(first_result, 'text'):
                return first_result.text.strip()
            return str(first_result).strip()
        return str(output).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a single audio file using Parakeet ASR"
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        required=True,
        help="Path to audio file (wav, mp3, flac, etc.)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Path to .nemo model file or HuggingFace model ID"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    print(f"Audio file: {audio_path}")
    print(f"Model: {args.model}")
    print("-" * 60)

    recognizer = ParakeetRecognizer(model_id=args.model)

    print("\nTranscribing...")
    text = recognizer.transcribe(str(audio_path))

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(text)
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
