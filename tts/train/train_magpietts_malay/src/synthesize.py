#!/usr/bin/env python3
"""
Synthesize speech from text using trained MagpieTTS model.

This script loads a trained/fine-tuned MagpieTTS model and generates
audio from input text. Supports phoneme-based synthesis for models
trained with phoneme input.

Usage:
    # Single text (with phonemes for Malay-trained model)
    python synthesize.py --model-path checkpoints/best_model.nemo --text "Selamat pagi" --use-phonemes
    
    # From file
    python synthesize.py --model-path checkpoints/best_model.nemo --input-file sentences.txt --use-phonemes
    
    # With speaker selection
    python synthesize.py --model-path checkpoints/best_model.nemo --text "Hello" --speaker 0
"""

import argparse
import logging
from pathlib import Path

import torch
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Phonemizer import (optional)
PHONEMIZER_AVAILABLE = False
try:
    from malay_phonemizer import MalayPhonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    pass

# Speaker mapping (same as in prepare_data.py)
SPEAKER_MAP = {
    "anwar_ibrahim": 0,
    "husein": 1,
    "kp_ms": 2,
    "kp_zh": 3,
    "shafiqah_idayu": 4,
}

# Reverse mapping for display
SPEAKER_NAMES = {v: k for k, v in SPEAKER_MAP.items()}

# Default sample rate for MagpieTTS
SAMPLE_RATE = 22050


def load_model(model_path: str, device: str = "cuda"):
    """
    Load a trained MagpieTTS model.
    
    Args:
        model_path: Path to .nemo checkpoint or HuggingFace model name
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model
    """
    try:
        # NeMo from GitHub main branch
        from nemo.collections.tts.models import MagpieTTSModel
    except ImportError:
        try:
            # Fallback: NeMo 2.x PyPI uses different class names
            from nemo.collections.tts.models import MagpieTTS_ModelInference as MagpieTTSModel
        except ImportError:
            logger.error("NeMo is not installed. Install with: pip install 'nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@main'")
            raise
    
    logger.info(f"Loading model from: {model_path}")
    
    if model_path.endswith('.nemo'):
        # Load from local checkpoint
        model = MagpieTTSModel.restore_from(model_path, map_location=device)
    else:
        # Load from HuggingFace or pretrained
        model = MagpieTTSModel.from_pretrained(model_path, map_location=device)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def synthesize_text(
    model,
    text: str,
    language: str = "ms",
    speaker_index: int = 0,
    apply_text_normalization: bool = True,
    phonemizer=None,
) -> tuple:
    """
    Synthesize audio from text.
    
    Args:
        model: Loaded MagpieTTS model
        text: Text to synthesize (or phonemes if phonemizer is None and model expects phonemes)
        language: Language code (default: "ms" for Malay)
        speaker_index: Speaker index (0-4 for Malaysian-TTS speakers)
        apply_text_normalization: Whether to apply text normalization
        phonemizer: Optional MalayPhonemizer for text-to-phoneme conversion
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Convert to phonemes if phonemizer is provided
    if phonemizer:
        original_text = text
        text = phonemizer.phonemize(text)
        logger.debug(f"Phonemized: '{original_text}' -> '{text}'")
    
    with torch.no_grad():
        audio, audio_len = model.do_tts(
            text,
            language=language,
            apply_TN=apply_text_normalization,
            speaker_index=speaker_index,
        )
    
    # Convert to numpy
    audio = audio.squeeze().cpu().numpy()
    
    return audio, SAMPLE_RATE


def synthesize_batch(
    model,
    texts: list[str],
    output_dir: Path,
    language: str = "ms",
    speaker_index: int = 0,
    apply_text_normalization: bool = True,
    prefix: str = "output",
    phonemizer=None,
):
    """
    Synthesize multiple texts and save to files.
    
    Args:
        model: Loaded MagpieTTS model
        texts: List of texts to synthesize
        output_dir: Directory to save audio files
        language: Language code
        speaker_index: Speaker index
        apply_text_normalization: Whether to apply text normalization
        prefix: Filename prefix
        phonemizer: Optional MalayPhonemizer for text-to-phoneme conversion
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(texts):
        logger.info(f"[{i+1}/{len(texts)}] Synthesizing: {text[:50]}...")
        
        try:
            audio, sr = synthesize_text(
                model,
                text,
                language=language,
                speaker_index=speaker_index,
                apply_text_normalization=apply_text_normalization,
                phonemizer=phonemizer,
            )
            
            output_path = output_dir / f"{prefix}_{i:04d}.wav"
            sf.write(str(output_path), audio, sr)
            logger.info(f"  Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue
    
    logger.info(f"Synthesis complete. Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech from text using trained MagpieTTS model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from single text
    python synthesize.py \\
        --model-path nvidia/magpie_tts_multilingual_357m \\
        --text "Selamat pagi, apa khabar?" \\
        --output-dir output/

    # Generate from text file (one sentence per line)
    python synthesize.py \\
        --model-path checkpoints/best_model.nemo \\
        --input-file sentences.txt \\
        --output-dir output/

    # Use specific speaker
    python synthesize.py \\
        --model-path checkpoints/best_model.nemo \\
        --text "Hello world" \\
        --speaker 1 \\
        --output-dir output/

Available speakers:
    0: anwar_ibrahim
    1: husein
    2: kp_ms
    3: kp_zh
    4: shafiqah_idayu
        """
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to .nemo checkpoint or HuggingFace model name"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (for single synthesis)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to text file with sentences (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output audio files (default: output/)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename for single text (default: auto-generated)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ms",
        help="Language code (default: ms for Malay)"
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        choices=list(SPEAKER_NAMES.keys()),
        help="Speaker index (default: 0)"
    )
    parser.add_argument(
        "--no-text-normalization",
        action="store_true",
        help="Disable text normalization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers and exit"
    )
    parser.add_argument(
        "--use-phonemes",
        action="store_true",
        help="Convert text to phonemes before synthesis (use for phoneme-trained models)"
    )
    
    args = parser.parse_args()
    
    # List speakers if requested
    if args.list_speakers:
        print("Available speakers:")
        for idx, name in sorted(SPEAKER_NAMES.items()):
            print(f"  {idx}: {name}")
        return
    
    # Validate input
    if not args.text and not args.input_file:
        parser.error("Either --text or --input-file is required")
    
    # Initialize phonemizer if requested
    phonemizer = None
    if args.use_phonemes:
        if not PHONEMIZER_AVAILABLE:
            logger.error("Phonemizer not available. Install with: pip install phonemizer")
            logger.error("Also install espeak-ng: apt install espeak-ng")
            return
        try:
            phonemizer = MalayPhonemizer()
            logger.info("Phonemizer enabled for Malay text")
        except Exception as e:
            logger.error(f"Failed to initialize phonemizer: {e}")
            return
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Prepare texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} sentences from {args.input_file}")
    
    # Synthesize
    output_dir = Path(args.output_dir)
    apply_tn = not args.no_text_normalization
    
    if len(texts) == 1 and args.output_file:
        # Single file output
        audio, sr = synthesize_text(
            model,
            texts[0],
            language=args.language,
            speaker_index=args.speaker,
            apply_text_normalization=apply_tn,
            phonemizer=phonemizer,
        )
        
        output_path = output_dir / args.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr)
        logger.info(f"Saved: {output_path}")
    else:
        # Batch output
        speaker_name = SPEAKER_NAMES.get(args.speaker, f"speaker{args.speaker}")
        synthesize_batch(
            model,
            texts,
            output_dir,
            language=args.language,
            speaker_index=args.speaker,
            apply_text_normalization=apply_tn,
            prefix=f"{args.language}_{speaker_name}",
            phonemizer=phonemizer,
        )


if __name__ == "__main__":
    main()
