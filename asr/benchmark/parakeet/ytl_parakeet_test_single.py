#!/usr/bin/env python3
"""
Single audio file transcription using Parakeet ASR

Usage:
    python ytl_parakeet_test_single.py -a /path/to/audio.wav
    python ytl_parakeet_test_single.py -a /path/to/audio.mp3 -m nvidia/parakeet-tdt-0.6b
    python ytl_parakeet_test_single.py -a /path/to/audio.wav -m /path/to/model.nemo
"""

import os
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ParakeetRecognizer:
    """
    NVIDIA Parakeet model-based ASR recognizer using NeMo
    
    Features:
    - NeMo-based inference with bfloat16 precision
    - Batch processing support
    - Optimized for multilingual transcription
    """
    
    def __init__(self, model_id="nvidia/parakeet-tdt-0.6b", batch_size=16):
        """
        Initialize Parakeet recognizer
        
        Args:
            model_id: HuggingFace model ID or path to local .nemo file
                     (default: nvidia/parakeet-tdt-0.6b)
            batch_size: Batch size for processing (default: 16)
        """
        import nemo.collections.asr as nemo_asr
        
        self.model_id = model_id
        self.batch_size = batch_size
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NeMo Parakeet model: {model_id}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        
        # Check if model_id is a local .nemo file or HuggingFace model ID
        is_local_nemo = model_id.endswith('.nemo') and os.path.exists(model_id)
        
        if is_local_nemo:
            # Load from local .nemo file
            print(f"Loading from local .nemo file: {model_id}")
            print("Loading model...")
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
        else:
            # Load from HuggingFace
            hf_token = os.environ.get('HF_TOKEN')
            
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required for HuggingFace models. "
                    "Please add it to .env file or export it: export HF_TOKEN=your_token_here"
                )
            
            # Authenticate with HuggingFace
            from huggingface_hub import login
            login(token=hf_token)
            
            # Load NeMo ASR model from HuggingFace
            print("Loading model from HuggingFace (this may take a few minutes on first run)...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        
        # Move model to device and set eval mode
        print(f"Moving model to {self.device} and setting eval mode...")
        self.model.to(self.device)
        self.model.eval()
        
        # Configure decoding strategy for better performance
        if hasattr(self.model.cfg, 'decoding'):
            if self.model.cfg.decoding.strategy != "beam":
                self.model.cfg.decoding.strategy = "greedy_batch"
                self.model.change_decoding_strategy(self.model.cfg.decoding)
        
        print(f"Model loaded successfully on {self.device}")
        
        # Track statistics
        self.total_transcriptions = 0
    
    def _audio_to_tensor(self, wav_path):
        """
        Load audio file and convert to torch tensor
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            torch.Tensor: Audio tensor (float32, autocast will handle conversion)
        """
        import librosa
        
        # Load audio file - librosa returns float32 in range [-1, 1]
        audio, sr = librosa.load(wav_path, sr=16000)
        
        # Convert to torch tensor (keep as float32, autocast will handle dtype conversion)
        audio_tensor = torch.from_numpy(audio)
        
        return audio_tensor
    
    def transcribe(self, wav_path):
        """
        Transcribe a single audio file
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        audio_tensor = self._audio_to_tensor(wav_path)
        
        # Transcribe with inference mode
        with torch.inference_mode():
            with torch.no_grad():
                # NeMo's transcribe expects a list
                output = self.model.transcribe([audio_tensor])
        
        self.total_transcriptions += 1
        
        # Extract text from NeMo output
        # NeMo can return tuple of (hypotheses, None) or just hypotheses list
        if isinstance(output, tuple):
            output = output[0]
        
        # Now output should be a list
        text = ""
        if isinstance(output, list) and len(output) > 0:
            first_result = output[0]
            if hasattr(first_result, 'text'):
                text = first_result.text.strip()
            else:
                text = str(first_result).strip()
        else:
            text = str(output).strip()
        
        return text


def main():
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
        default="nvidia/parakeet-tdt-0.6b",
        help="Path to .nemo model file or HuggingFace model ID (default: nvidia/parakeet-tdt-0.6b)"
    )
    args = parser.parse_args()

    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    print(f"Audio file: {audio_path}")
    print(f"Model: {args.model}")
    print("-" * 60)

    # Initialize recognizer
    recognizer = ParakeetRecognizer(
        model_id=args.model,
        batch_size=1
    )

    # Transcribe
    print("\nTranscribing...")
    text = recognizer.transcribe(str(audio_path))

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(text)
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
