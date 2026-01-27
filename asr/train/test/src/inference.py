#!/usr/bin/env python3
"""
Run inference using Mesolitica Malaysian Whisper.

Model: mesolitica/Malaysian-whisper-large-v3-turbo-v3

Supports:
- Language selection (ms, en, zh, yue, ta, etc.)
- Multiple audio files
- Automatic audio preprocessing (mono conversion, resampling)

Usage:
    python inference.py --audio path/to/audio.wav
    python inference.py --audio path/to/audio.wav --language ms
    python inference.py --audio file1.wav file2.wav file3.wav
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# Default model
DEFAULT_MODEL = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

# Supported languages for this model
SUPPORTED_LANGUAGES = [
    "ms",   # Malay
    "en",   # English  
    "zh",   # Mandarin Chinese
    "yue",  # Cantonese
    "ta",   # Tamil
]


def preprocess_audio(audio_path: str, target_sr: int = 16000) -> tuple[str, bool]:
    """Preprocess audio: convert stereo to mono and resample if needed.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default 16kHz for ASR)
        
    Returns:
        Tuple of (path_to_use, is_temp_file)
        If preprocessing was needed, returns path to temp file and True.
        Otherwise returns original path and False.
    """
    # Load audio with librosa (handles many formats including mp3)
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    
    needs_processing = False
    
    # Check if stereo (shape will be (2, samples) for stereo, (samples,) for mono)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        # Convert stereo to mono by averaging channels
        waveform = waveform.mean(axis=0)
        needs_processing = True
    
    # Resample if needed
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        needs_processing = True
    
    if needs_processing:
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, waveform, target_sr)
        return temp_file.name, True
    
    return audio_path, False


class MesoliticaWhisperInference:
    """Mesolitica Malaysian Whisper ASR inference."""
    
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model: {model_id}")
        print(f"Device: {self.device}")
        
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        print("Model loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = None) -> str:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ms, en, zh, yue, ta, etc.)
                     If None, model will auto-detect
        
        Returns:
            Transcribed text
        """
        # Load audio with librosa (handles resampling and mono conversion)
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Prepare input features
        input_features = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        
        # Build generate kwargs - transcribe only, no translation
        generate_kwargs = {
            "do_sample": False,
            "num_beams": 1,
        }
        
        if language:
            generate_kwargs["language"] = language
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs
            )
        
        # Decode
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Mesolitica Malaysian Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic inference (auto-detect language)
  python inference.py --audio test.wav
  
  # Specify language
  python inference.py --audio test.wav --language ms
  python inference.py --audio test.wav --language en
  python inference.py --audio test.wav --language zh
  
  # Multiple files
  python inference.py --audio *.wav --language ms

Supported languages: {', '.join(SUPPORTED_LANGUAGES)}
        """
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        nargs='+',
        required=True,
        help="Audio file(s) to transcribe"
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help=f"Language code: {', '.join(SUPPORTED_LANGUAGES)} (default: auto-detect)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Validate language if provided
    if args.language and args.language not in SUPPORTED_LANGUAGES:
        print(f"Warning: Language '{args.language}' may not be supported.")
        print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    
    # Initialize model
    recognizer = MesoliticaWhisperInference(model_id=args.model)
    
    # Transcribe
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    if args.language:
        print(f"Language: {args.language}")
    else:
        print("Language: auto-detect")
    print("=" * 60)
    
    temp_files = []
    
    try:
        for audio_path in args.audio:
            if not os.path.exists(audio_path):
                print(f"\n⚠️  File not found: {audio_path}")
                continue
            
            # Preprocess audio if needed
            processed_path, is_temp = preprocess_audio(audio_path)
            if is_temp:
                temp_files.append(processed_path)
            
            # Transcribe
            transcription = recognizer.transcribe(processed_path, language=args.language)
            
            filename = os.path.basename(audio_path)
            print(f"\n📁 {filename}")
            print(f"   {transcription}")
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
