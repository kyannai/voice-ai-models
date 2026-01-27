#!/usr/bin/env python3
"""
Run inference on audio files using a trained model.

Supports:
- .nemo model files
- .ckpt checkpoint files (requires base model for config)
- Soft bias for code-switching (default): prefers major language but allows mixing
- Hard suppression (--hard-suppression): completely blocks Chinese tokens
- Automatic language detection using Whisper tiny (--language-detection)

Usage:
    # With .nemo model
    python inference.py --model path/to/model.nemo --audio path/to/audio.wav
    
    # With .ckpt checkpoint (needs base model for architecture)
    python inference.py --checkpoint path/to/checkpoint.ckpt \\
                        --base-model path/to/base.nemo \\
                        --audio path/to/audio.wav
    
    # Soft bias with language detection (recommended for code-switching)
    python inference.py --model model.nemo --audio test.wav --language-detection
    
    # Custom bias strength (e.g., stronger suppression)
    python inference.py --model model.nemo --audio test.wav --language ms --bias-strength -5.0
    
    # Hard suppression (old behavior, completely blocks Chinese)
    python inference.py --model model.nemo --audio test.wav --language ms --hard-suppression
    
    # Boost Chinese but allow English
    python inference.py --model model.nemo --audio test.wav --language zh --bias-strength +5.0
"""
import argparse
import os
import sys
import tempfile

import torch
import soundfile as sf
import librosa
import nemo.collections.asr as nemo_asr


# Original vocabulary size (before Chinese expansion)
# Tokens >= this value are Chinese characters
ORIGINAL_VOCAB_SIZE = 8192

# Bias presets for different languages (soft bias for code-switching)
# Positive = boost Chinese tokens, Negative = suppress Chinese tokens
BIAS_PRESETS = {
    'zh': +3.0,    # Chinese major: boost Chinese, allow English/Malay
    'ms': -3.0,    # Malay major: prefer Latin, allow Chinese
    'en': -3.0,    # English major: prefer Latin, allow Chinese
}

# Hard suppression for backwards compatibility (completely blocks tokens)
HARD_SUPPRESSION_BIAS = -100.0

# Lazy-loaded Whisper model for language detection
_whisper_lid_model = None


def detect_language(audio_path: str) -> tuple[str, float]:
    """
    Detect language from audio using Whisper tiny model.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (language_code, confidence)
        language_code: 'zh' for Chinese, 'ms' for Malay, 'en' for English/other
        confidence: Float between 0 and 1
    """
    global _whisper_lid_model
    
    if _whisper_lid_model is None:
        import whisper
        print("[LID] Loading Whisper tiny model for language detection...")
        _whisper_lid_model = whisper.load_model("tiny")
        print("[LID] Whisper tiny model loaded")
    
    import whisper
    
    # Load and preprocess audio for Whisper
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(_whisper_lid_model.device)
    
    # Detect language
    _, probs = _whisper_lid_model.detect_language(mel)
    detected = max(probs, key=probs.get)
    confidence = probs[detected]
    
    # Map to our language codes
    if detected in ['zh', 'yue']:  # Chinese variants (Mandarin, Cantonese)
        return 'zh', confidence
    elif detected in ['ms', 'id']:  # Malay/Indonesian
        return 'ms', confidence
    else:
        return 'en', confidence  # Default to English for all others


def is_huggingface_model_id(path: str) -> bool:
    """Check if path looks like a HuggingFace model ID (e.g., 'nvidia/parakeet-tdt-0.6b')."""
    # HuggingFace model IDs have format: org/model-name
    # Local paths typically have extensions or are absolute/relative paths
    if os.path.exists(path):
        return False
    if path.endswith('.nemo') or path.endswith('.ckpt'):
        return False
    if '/' in path and not path.startswith('/') and not path.startswith('./'):
        # Looks like org/model format
        parts = path.split('/')
        if len(parts) == 2 and all(p and not p.startswith('.') for p in parts):
            return True
    return False


def load_base_model(model_path: str):
    """Load model from .nemo file or HuggingFace model ID.
    
    Args:
        model_path: Path to .nemo file OR HuggingFace model ID (e.g., 'nvidia/parakeet-tdt-0.6b')
    """
    if is_huggingface_model_id(model_path):
        print(f"Loading model from HuggingFace: {model_path}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_path)
    else:
        print(f"Loading model from file: {model_path}")
        model = nemo_asr.models.ASRModel.restore_from(model_path)
    return model


def load_model_from_nemo(nemo_path: str):
    """Load model from .nemo file or HuggingFace model ID."""
    return load_base_model(nemo_path)


def load_model_from_checkpoint(ckpt_path: str, base_model_path: str):
    """Load model from .ckpt checkpoint file.
    
    Args:
        ckpt_path: Path to the checkpoint file (.ckpt)
        base_model_path: Path to the base model (.nemo) or HuggingFace model ID for architecture/config
    """
    print(f"Loading base model architecture from: {base_model_path}")
    model = load_base_model(base_model_path)
    
    print(f"Loading checkpoint weights from: {ckpt_path}")
    # weights_only=False needed for NeMo checkpoints containing OmegaConf objects
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # NeMo checkpoints store state_dict under 'state_dict' key
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model


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
    # sr=None preserves original sample rate for checking
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


def apply_language_suppression(
    model, 
    language: str, 
    original_vocab_size: int = ORIGINAL_VOCAB_SIZE,
    bias_strength: float = None,
    hard_suppression: bool = False
):
    """
    Apply token bias based on language.
    
    Uses soft bias by default (preserves code-switching) or hard suppression
    for complete token blocking.
    
    Args:
        model: NeMo ASR model
        language: Major language code ('zh', 'ms', 'en')
        original_vocab_size: Token ID threshold (tokens >= this are Chinese)
        bias_strength: Custom bias (overrides preset). Positive=boost Chinese, Negative=suppress.
        hard_suppression: If True, use -100.0 (old behavior). Overrides bias_strength.
        
    Returns:
        original_bias: Original bias tensor to restore later (or None if no change)
    """
    # Determine bias to apply
    if hard_suppression:
        effective_bias = HARD_SUPPRESSION_BIAS if language != 'zh' else 0.0
    elif bias_strength is not None:
        effective_bias = bias_strength
    else:
        effective_bias = BIAS_PRESETS.get(language, 0.0)
    
    if effective_bias == 0.0:
        print(f"[Bias Mode] {language} - No bias applied (neutral)")
        return None
    
    # Apply bias to Chinese tokens
    try:
        output_layer = model.joint.joint_net[-1]  # Last linear layer in joint network
        original_bias = output_layer.bias.data.clone()
        
        vocab_size = output_layer.bias.shape[0]
        num_affected = vocab_size - original_vocab_size
        
        if num_affected > 0:
            direction = "Boosting" if effective_bias > 0 else "Suppressing"
            mode = "HARD" if hard_suppression else "soft"
            print(f"[Bias Mode] {language} - {direction} Chinese tokens ({mode})")
            print(f"  Bias strength: {effective_bias:+.1f}")
            print(f"  Affected tokens: {num_affected} (IDs {original_vocab_size}+)")
            output_layer.bias.data[original_vocab_size:] += effective_bias
        else:
            print(f"[Bias Mode] {language} - No Chinese tokens to affect (vocab size <= {original_vocab_size})")
        
        return original_bias
    except Exception as e:
        print(f"Warning: Could not apply language bias: {e}")
        return None


def restore_original_bias(model, original_bias):
    """Restore the original bias after inference."""
    if original_bias is not None:
        try:
            output_layer = model.joint.joint_net[-1]
            output_layer.bias.data = original_bias
            print("[Language Mode] Restored original vocabulary")
        except Exception:
            pass


def transcribe_audio(
    model, 
    audio_paths: list[str], 
    language: str = None, 
    original_vocab_size: int = ORIGINAL_VOCAB_SIZE,
    bias_strength: float = None,
    hard_suppression: bool = False
):
    """Transcribe audio files with optional language-based token bias.
    
    Args:
        model: NeMo ASR model
        audio_paths: List of audio file paths to transcribe
        language: Language code ('zh', 'ms', 'en') for bias selection
        original_vocab_size: Token ID threshold for Chinese tokens
        bias_strength: Custom bias strength (overrides language preset)
        hard_suppression: Use -100.0 instead of soft bias
    """
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    # Apply language bias if specified
    original_bias = None
    if language:
        original_bias = apply_language_suppression(
            model, language, original_vocab_size,
            bias_strength=bias_strength,
            hard_suppression=hard_suppression
        )
    
    results = []
    temp_files = []
    
    try:
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                results.append((audio_path, "[FILE NOT FOUND]"))
                continue
            
            try:
                # Preprocess audio (convert stereo to mono, resample if needed)
                processed_path, is_temp = preprocess_audio(audio_path)
                if is_temp:
                    temp_files.append(processed_path)
                
                # Transcribe
                transcription = model.transcribe([processed_path])
                
                # Handle different return types
                if hasattr(transcription[0], 'text'):
                    text = transcription[0].text
                else:
                    text = str(transcription[0])
                
                results.append((audio_path, text))
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append((audio_path, f"[ERROR: {e}]"))
    finally:
        # Restore original bias if we modified it
        restore_original_bias(model, original_bias)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ASR inference on audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With .nemo model
  python inference.py --model model.nemo --audio test.wav
  
  # With checkpoint
  python inference.py --checkpoint ckpt.ckpt --base-model base.nemo --audio test.wav
  
  # Soft bias with language detection (recommended for code-switching)
  python inference.py --model model.nemo --audio test.wav --language-detection
  
  # Custom bias strength (stronger suppression)
  python inference.py --model model.nemo --audio test.wav --language ms --bias-strength -5.0
  
  # Hard suppression (completely block Chinese, old behavior)
  python inference.py --model model.nemo --audio test.wav --language ms --hard-suppression
  
  # Boost Chinese but allow English words
  python inference.py --model model.nemo --audio test.wav --language zh --bias-strength +5.0

Bias Presets (soft, preserves code-switching):
  zh:  +3.0 (boost Chinese, allow English/Malay)
  ms:  -3.0 (prefer Latin, allow Chinese)
  en:  -3.0 (prefer Latin, allow Chinese)
        """
    )
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", 
        type=str,
        help="Path to .nemo model file"
    )
    model_group.add_argument(
        "--checkpoint", 
        type=str,
        help="Path to .ckpt checkpoint file (requires --base-model)"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base .nemo model (required when using --checkpoint)"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        nargs='+',
        required=True,
        help="Audio file(s) to transcribe"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=['en', 'ms', 'zh'],
        default=None,
        help="Language hint: 'zh' enables Chinese tokens, 'en'/'ms' suppresses them"
    )
    
    parser.add_argument(
        "--language-detection",
        action="store_true",
        default=False,
        help="Auto-detect language using Whisper tiny (~10ms). Suppresses Chinese tokens for non-Chinese audio."
    )
    
    parser.add_argument(
        "--original-vocab-size",
        type=int,
        default=ORIGINAL_VOCAB_SIZE,
        help=f"Original vocab size before Chinese expansion (default: {ORIGINAL_VOCAB_SIZE})"
    )
    
    parser.add_argument(
        "--bias-strength",
        type=float,
        default=None,
        help="Custom bias for Chinese tokens. Positive=boost, Negative=suppress. "
             "Overrides language presets (zh=+3.0, ms/en=-3.0)."
    )
    
    parser.add_argument(
        "--hard-suppression",
        action="store_true",
        default=False,
        help="Use hard suppression (-100.0) instead of soft bias. "
             "Completely blocks Chinese for non-zh languages (old behavior)."
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint and not args.base_model:
        parser.error("--base-model is required when using --checkpoint")
    
    # Check for conflicting arguments
    if args.language and args.language_detection:
        parser.error("Cannot use both --language and --language-detection. Choose one.")
    
    # Use the vocab size from args (allows override via --original-vocab-size)
    vocab_size_to_use = args.original_vocab_size
    if vocab_size_to_use != ORIGINAL_VOCAB_SIZE:
        print(f"Using custom original vocab size: {vocab_size_to_use}")
    
    # Load model
    if args.model:
        model = load_model_from_nemo(args.model)
    else:
        model = load_model_from_checkpoint(args.checkpoint, args.base_model)
    
    # Transcribe
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    print("=" * 60)
    
    # Determine language to use
    language_to_use = args.language
    
    # If language detection is enabled, detect language from first audio file
    if args.language_detection:
        # Detect language from first audio file (assuming all files are same language)
        first_audio = args.audio[0]
        if os.path.exists(first_audio):
            detected_lang, confidence = detect_language(first_audio)
            print(f"[LID] Detected language: {detected_lang} (confidence: {confidence:.1%})")
            language_to_use = detected_lang
        else:
            print(f"[LID] Warning: Cannot detect language, file not found: {first_audio}")
    
    results = transcribe_audio(
        model, 
        args.audio, 
        language_to_use, 
        vocab_size_to_use,
        bias_strength=args.bias_strength,
        hard_suppression=args.hard_suppression
    )
    
    for audio_path, transcription in results:
        filename = os.path.basename(audio_path)
        print(f"\n📁 {filename}")
        print(f"   {transcription}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
