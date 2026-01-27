#!/usr/bin/env python3
"""
Run inference with Whisper models (original or fine-tuned).

Supports:
- OpenAI Whisper models from HuggingFace
- Fine-tuned Whisper checkpoints
- Language selection (ms, en, zh, etc.)
- Language restriction (suppress non-target languages for faster, more targeted inference)
- Multiple audio files
- Automatic audio preprocessing (mono conversion, resampling)

Usage:
    python inference.py --model openai/whisper-large-v3-turbo --audio test.wav
    python inference.py --model ./outputs/final --audio test.wav --language ms
    python inference.py --model ./outputs/final --audio test.wav --target-languages en ms zh ta
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Set

import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# Default model
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"

# Default target languages for Malaysian context
DEFAULT_TARGET_LANGUAGES = ["en", "ms", "zh", "ta"]

# All Whisper language codes (99 languages)
# From: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
ALL_WHISPER_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
]


def preprocess_audio(audio_path: str, target_sr: int = 16000) -> Tuple[str, bool]:
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


def get_language_token_ids(
    tokenizer,
    target_languages: List[str],
) -> Tuple[List[int], List[int]]:
    """Get token IDs for target and non-target languages.
    
    Args:
        tokenizer: WhisperTokenizer
        target_languages: List of language codes to keep (e.g., ["en", "ms", "zh", "ta"])
        
    Returns:
        Tuple of (target_token_ids, suppress_token_ids)
    """
    target_set = set(target_languages)
    target_token_ids = []
    suppress_token_ids = []
    
    for lang in ALL_WHISPER_LANGUAGES:
        # Whisper language tokens are in format: <|en|>, <|ms|>, etc.
        lang_token = f"<|{lang}|>"
        try:
            token_id = tokenizer.convert_tokens_to_ids(lang_token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                if lang in target_set:
                    target_token_ids.append(token_id)
                else:
                    suppress_token_ids.append(token_id)
        except Exception:
            pass  # Token not in vocabulary
    
    return target_token_ids, suppress_token_ids


class WhisperInference:
    """Whisper ASR inference class.
    
    Supports both HuggingFace model IDs and local checkpoints.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        target_languages: Optional[List[str]] = None,
    ):
        """Initialize Whisper model.
        
        Args:
            model_path: HuggingFace model ID or path to local checkpoint
            target_languages: List of language codes to allow (suppresses all others)
                            If None, all languages are allowed
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.target_languages = target_languages
        self.suppress_token_ids = None
        
        print(f"Loading model: {model_path}")
        print(f"Device: {self.device}")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Setup language restriction if specified
        if target_languages:
            target_ids, suppress_ids = get_language_token_ids(
                self.processor.tokenizer,
                target_languages
            )
            self.suppress_token_ids = suppress_ids
            print(f"Target languages: {target_languages}")
            print(f"  Enabled {len(target_ids)} language tokens")
            print(f"  Suppressing {len(suppress_ids)} non-target language tokens")
        
        print("Model loaded successfully!")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        num_beams: int = 1,
    ) -> str:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ms, en, zh, etc.) or None for auto-detect
            task: "transcribe" or "translate" (translate to English)
            num_beams: Number of beams for beam search (1 = greedy)
        
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
        
        # Build generate kwargs
        generate_kwargs = {
            "do_sample": False,
            "num_beams": num_beams,
            "task": task,
        }
        
        if language:
            generate_kwargs["language"] = language
        
        # Add language suppression if configured
        if self.suppress_token_ids:
            # Get existing suppress_tokens from model config and extend
            existing_suppress = list(self.model.generation_config.suppress_tokens or [])
            generate_kwargs["suppress_tokens"] = existing_suppress + self.suppress_token_ids
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        task: str = "transcribe",
        num_beams: int = 1,
    ) -> List[str]:
        """Transcribe multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            language: Language code or None for auto-detect
            task: "transcribe" or "translate"
            num_beams: Number of beams for beam search
        
        Returns:
            List of transcribed texts
        """
        results = []
        for audio_path in audio_paths:
            result = self.transcribe(
                audio_path,
                language=language,
                task=task,
                num_beams=num_beams,
            )
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Whisper models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic inference (auto-detect from all languages)
  python inference.py --model openai/whisper-large-v3-turbo --audio test.wav
  
  # Restrict to target languages only (faster, more targeted)
  python inference.py --model openai/whisper-large-v3-turbo --audio test.wav \\
      --target-languages en ms zh ta
  
  # Force specific language
  python inference.py --model openai/whisper-large-v3-turbo --audio test.wav --language ms
  
  # Use fine-tuned model with language restriction
  python inference.py --model ./outputs/final --audio test.wav \\
      --target-languages en ms zh ta
  
  # Beam search (more accurate, slower)
  python inference.py --model ./outputs/final --audio test.wav --num-beams 5

Default target languages: {', '.join(DEFAULT_TARGET_LANGUAGES)}
All {len(ALL_WHISPER_LANGUAGES)} Whisper languages: {', '.join(ALL_WHISPER_LANGUAGES[:15])}...
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or path to local checkpoint (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--audio", "-a",
        type=str,
        nargs='+',
        required=True,
        help="Audio file(s) to transcribe"
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Force specific language code (default: auto-detect)"
    )
    
    parser.add_argument(
        "--target-languages",
        type=str,
        nargs='+',
        default=None,
        help=f"Restrict to these languages only, suppressing all others. "
             f"Example: --target-languages en ms zh ta"
    )
    
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help=f"Use default target languages: {', '.join(DEFAULT_TARGET_LANGUAGES)}"
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe (default) or translate to English"
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1 = greedy, default)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for transcriptions (optional, prints to stdout by default)"
    )
    
    args = parser.parse_args()
    
    # Determine target languages
    target_languages = args.target_languages
    if args.use_defaults and not target_languages:
        target_languages = DEFAULT_TARGET_LANGUAGES
    
    # Validate language if provided
    if args.language and args.language not in ALL_WHISPER_LANGUAGES:
        print(f"Warning: Language '{args.language}' may not be supported.")
    
    # Validate target languages
    if target_languages:
        invalid_langs = [l for l in target_languages if l not in ALL_WHISPER_LANGUAGES]
        if invalid_langs:
            print(f"Warning: Unknown language codes: {invalid_langs}")
    
    # Initialize model
    recognizer = WhisperInference(
        model_path=args.model,
        target_languages=target_languages,
    )
    
    # Header
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    if args.language:
        print(f"Language: {args.language} (forced)")
    elif target_languages:
        print(f"Language: auto-detect from {target_languages}")
    else:
        print("Language: auto-detect (all languages)")
    print("=" * 60)
    
    temp_files = []
    results = []
    
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
            transcription = recognizer.transcribe(
                processed_path,
                language=args.language,
                task=args.task,
                num_beams=args.num_beams,
            )
            
            filename = os.path.basename(audio_path)
            print(f"\n📁 {filename}")
            print(f"   {transcription}")
            
            results.append({
                "file": filename,
                "transcription": transcription
            })
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    print("\n" + "=" * 60)
    
    # Save results to file if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
