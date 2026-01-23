#!/usr/bin/env python3
"""
Synthesize speech from text using trained MagpieTTS model.

Two-Phase Pipeline Support:
- Phase 1/2 models: Use raw Malay text directly (model has G2P built-in)
- Legacy models: Use --use-phonemes for external phoneme conversion

Usage:
    # With Phase 1/2 trained model (raw text - no phonemizer needed!)
    python synthesize.py --model-path models/malay_base.nemo --text "Selamat pagi" --language ms
    
    # With legacy voice-clone model (requires phonemizer)
    python synthesize.py --model-path legacy_model.pt --text "Selamat pagi" --use-phonemes
    
    # From file
    python synthesize.py --model-path models/malay.nemo --input-file sentences.txt
    
    # With speaker selection
    python synthesize.py --model-path models/malay.nemo --text "Hello" --speaker 2
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import soundfile as sf
from omegaconf import OmegaConf, open_dict

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

# Optional language ID for stronger code-switch detection
LANGID_AVAILABLE = False
try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    pass

# Speaker mapping (same as in prepare_data.py)
SPEAKER_MAP = {
    "anwar_ibrahim": 0,
    "husein": 1,
    "kp_ms": 2,
    "shafiqah_idayu": 3,
}

# Reverse mapping for display
SPEAKER_NAMES = {v: k for k, v in SPEAKER_MAP.items()}

# Default sample rate for MagpieTTS
SAMPLE_RATE = 22050

# Default G2P dictionary path
DEFAULT_G2P_DICT = "data/g2p/ipa_malay_dict.txt"


def load_g2p_dictionary(g2p_dict_path: str) -> dict:
    """Load G2P dictionary from file.
    
    Format: WORD IPA_phonemes (one per line)
    """
    g2p_dict = {}
    with open(g2p_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' ' not in line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                word, phonemes = parts
                g2p_dict[word.lower()] = phonemes
    return g2p_dict


def text_to_phonemes(text: str, g2p_dict: dict) -> str:
    """Convert text to IPA phonemes using G2P dictionary.
    
    Words not in dictionary are kept as-is (for punctuation, numbers, etc.)
    """
    import re
    
    # Tokenize into words and non-words
    tokens = re.findall(r"[\w']+|[^\w\s]+|\s+", text, re.UNICODE)
    
    result = []
    for token in tokens:
        if token.strip() and token.lower() in g2p_dict:
            # Word found in dictionary - use phonemes
            result.append(g2p_dict[token.lower()])
        else:
            # Keep as-is (punctuation, whitespace, unknown words)
            result.append(token)
    
    return ''.join(result)


def add_malay_tokenizer(model, g2p_dict_path: str = None):
    """
    Add Malay tokenizer to MagpieTTS model by replacing Spanish tokenizer entirely.
    
    We create a fresh IPATokenizer with:
    - Malay-specific IPA vocabulary (extracted from G2P dictionary)
    - Malay G2P dictionary for word-to-phoneme conversion
    
    This ensures all Malay phonemes (like ə, ŋ) are in the vocabulary.
    Use language='es' when calling do_tts to use the Malay tokenizer.
    """
    if g2p_dict_path is None:
        g2p_dict_path = DEFAULT_G2P_DICT
    
    g2p_dict_path = str(Path(g2p_dict_path).absolute())
    
    if not Path(g2p_dict_path).exists():
        logger.warning(f"G2P dictionary not found: {g2p_dict_path}")
        logger.warning("Malay tokenizer will not be added.")
        return False
    
    logger.info(f"Creating Malay tokenizer with fresh IPA vocab: {g2p_dict_path}")
    
    try:
        from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
        from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
        
        agg_tok = model.tokenizer
        
        if 'spanish_phoneme' not in agg_tok.tokenizers:
            logger.warning("  Spanish tokenizer not found, cannot replace")
            return False
        
        # Create Malay G2P
        malay_g2p = IpaG2p(
            phoneme_dict=g2p_dict_path,
            heteronyms=None,
            phoneme_probability=1.0,
            ignore_ambiguous_words=False,
            use_chars=False,
            use_stresses=True,
        )
        logger.info(f"  Created Malay IpaG2p with {len(malay_g2p.phoneme_dict)} entries")
        
        # Extract all unique IPA symbols from the Malay G2P dictionary
        all_phonemes = set()
        for word, pronunciations in malay_g2p.phoneme_dict.items():
            for pron in pronunciations:
                all_phonemes.update(pron)
        
        # Add common punctuation and special symbols
        punct_symbols = set(".,!?;:'-\"()[]{}…–—")
        all_phonemes.update(punct_symbols)
        
        # Sort for consistency
        malay_vocab = sorted(all_phonemes)
        logger.info(f"  Extracted {len(malay_vocab)} unique IPA symbols for Malay")
        logger.info(f"  Sample symbols: {malay_vocab[:20]}...")
        
        # Create fresh IPATokenizer with Malay vocabulary
        malay_tokenizer = IPATokenizer(
            g2p=malay_g2p,
            punct=True,
            apostrophe=True,
            pad_with_space=False,
        )
        
        # Override the tokenizer's vocabulary with our Malay IPA symbols
        # The IPATokenizer builds its vocab from the G2P, so this should be automatic
        # But let's verify the vocab includes our phonemes
        if hasattr(malay_tokenizer, 'tokens'):
            tok_vocab = set(malay_tokenizer.tokens)
            missing = all_phonemes - tok_vocab - {' '}  # space might be handled separately
            if missing:
                logger.warning(f"  Some phonemes not in tokenizer vocab: {list(missing)[:10]}")
        
        # Replace Spanish tokenizer entirely
        agg_tok.tokenizers['spanish_phoneme'] = malay_tokenizer
        logger.info("  Replaced Spanish tokenizer with fresh Malay tokenizer")
        logger.info("  Use language='es' to synthesize Malay text")
        
        # Ensure aggregate tokenizer maps language codes to the Spanish slot
        if hasattr(agg_tok, 'lang2tokenizer'):
            agg_tok.lang2tokenizer['es'] = 'spanish_phoneme'
            agg_tok.lang2tokenizer['ms'] = 'spanish_phoneme'
            logger.info("  Updated lang2tokenizer mapping: es/ms -> spanish_phoneme")
        
        logger.info("Malay tokenizer added successfully (using Spanish slot)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to add Malay tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_model(model_path: str, device: str = "cuda", g2p_dict: str = None, language: str = "en"):
    """
    Load a trained MagpieTTS model.
    
    Args:
        model_path: Path to .nemo/.pt checkpoint or HuggingFace model name
        device: Device to load model on ('cuda' or 'cpu')
        g2p_dict: Path to G2P dictionary (for Malay support with .pt files)
        language: Target language (if 'ms', will add Malay tokenizer for .pt files)
        
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
        # Load from local .nemo checkpoint
        model = MagpieTTSModel.restore_from(model_path, map_location=device)
    elif model_path.endswith('.pt'):
        # Load pretrained model and apply state dict from .pt file
        logger.info("Loading pretrained model and applying fine-tuned weights...")
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m", map_location=device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Fine-tuned weights loaded successfully")
    else:
        # Load from HuggingFace or pretrained
        model = MagpieTTSModel.from_pretrained(model_path, map_location=device)
    
    model = model.to(device)
    model.eval()
    
    # Add Malay tokenizer if language is 'ms'
    if language == 'ms':
        add_malay_tokenizer(model, g2p_dict)
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def detect_token_language(token: str, english_words: set, malay_abbrev: set) -> str:
    """Detect language for a token using langid if available, else heuristics."""
    token_lower = token.lower()
    if token_lower in english_words:
        return "en"
    if token_lower in malay_abbrev:
        return "ms"
    
    if LANGID_AVAILABLE and len(token) >= 3:
        try:
            label, _ = langid.classify(token)
            if label.startswith("en"):
                return "en"
            if label in ("ms", "id"):
                return "ms"
        except Exception:
            pass
    
    return "ms"


def split_text_by_language(text: str) -> list[tuple[str, str]]:
    """Split text into (segment, language) tuples using LID + heuristics."""
    import re
    
    try:
        from malay_phonemizer import MalayPhonemizer
        english_words = MalayPhonemizer.ENGLISH_WORDS
        malay_abbrev = set(MalayPhonemizer.ABBREVIATIONS.keys())
    except Exception:
        english_words = set()
        malay_abbrev = set()
    
    tokens = re.findall(r"[A-Za-z']+|[^\w\s]+|\s+", text, re.UNICODE)
    
    segments: list[tuple[str, str]] = []
    current = ""
    current_lang = None
    
    for tok in tokens:
        if tok.isspace():
            current += tok
            continue
        
        if re.match(r"[A-Za-z']+$", tok):
            lang = detect_token_language(tok, english_words, malay_abbrev)
        else:
            lang = current_lang or "ms"
        
        if current_lang is None:
            current_lang = lang
        
        if lang != current_lang and current.strip():
            segments.append((current.strip(), current_lang))
            current = tok
            current_lang = lang
        else:
            current += tok
            current_lang = lang
    
    if current.strip():
        segments.append((current.strip(), current_lang or "ms"))
    
    return segments


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
        text: Text to synthesize
        language: Language code (default: "ms" for Malay)
        speaker_index: Speaker index (0-4 for Malaysian-TTS speakers)
        apply_text_normalization: Whether to apply text normalization
        phonemizer: Optional MalayPhonemizer for text-to-phoneme conversion (legacy)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Legacy: Convert to phonemes if phonemizer is provided
    if phonemizer:
        original_text = text
        text = phonemizer.phonemize(text)
        logger.debug(f"Phonemized: '{original_text}' -> '{text}'")
    
    # For Malay: use Spanish slot (we replaced Spanish G2P with Malay)
    # Spanish has text normalization support, so we can keep it enabled
    actual_language = language
    if language == 'ms':
        actual_language = 'es'  # Use Spanish slot (contains Malay G2P)
    
    with torch.no_grad():
        audio, audio_len = model.do_tts(
            text,
            language=actual_language,
            apply_TN=apply_text_normalization,
            speaker_index=speaker_index,
        )
    
    # Convert to numpy
    audio = audio.squeeze().cpu().numpy()
    
    return audio, SAMPLE_RATE


def synthesize_text_with_codeswitch(
    model,
    text: str,
    speaker_index: int = 0,
    apply_text_normalization: bool = True,
    pause_seconds: float = 0.0,
) -> tuple:
    """Synthesize audio for code-switched text by splitting into language segments."""
    segments = split_text_by_language(text)
    if not segments:
        return synthesize_text(
            model,
            text,
            language="ms",
            speaker_index=speaker_index,
            apply_text_normalization=apply_text_normalization,
        )
    
    audio_parts = []
    pause = None
    if pause_seconds > 0:
        pause = torch.zeros(int(SAMPLE_RATE * pause_seconds))
    
    for segment_text, lang in segments:
        audio, _ = synthesize_text(
            model,
            segment_text,
            language=lang,
            speaker_index=speaker_index,
            apply_text_normalization=apply_text_normalization,
        )
        audio_parts.append(torch.tensor(audio))
        if pause is not None:
            audio_parts.append(pause)
    
    if not audio_parts:
        return synthesize_text(
            model,
            text,
            language="ms",
            speaker_index=speaker_index,
            apply_text_normalization=apply_text_normalization,
        )
    
    combined = torch.cat(audio_parts).cpu().numpy()
    return combined, SAMPLE_RATE


def synthesize_batch(
    model,
    texts: list[str],
    output_dir: Path,
    language: str = "ms",
    speaker_index: int = 0,
    apply_text_normalization: bool = True,
    prefix: str = "output",
    phonemizer=None,
    code_switch: bool = False,
    index_start: int = 0,
    separator: str = "_",
    pad_width: int = 4,
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
        phonemizer: Optional MalayPhonemizer for text-to-phoneme conversion (legacy)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(texts):
        display_index = i + 1
        logger.info(f"[{display_index}/{len(texts)}] Synthesizing: {text[:50]}...")
        
        try:
            if code_switch:
                audio, sr = synthesize_text_with_codeswitch(
                    model,
                    text,
                    speaker_index=speaker_index,
                    apply_text_normalization=apply_text_normalization,
                )
            else:
                audio, sr = synthesize_text(
                    model,
                    text,
                    language=language,
                    speaker_index=speaker_index,
                    apply_text_normalization=apply_text_normalization,
                    phonemizer=phonemizer,
                )
            
            file_index = i + index_start
            if pad_width > 0:
                index_str = f"{file_index:0{pad_width}d}"
            else:
                index_str = str(file_index)
            output_path = output_dir / f"{prefix}{separator}{index_str}.wav"
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
    # Phase 1/2 model: Direct text input (model has Malay G2P)
    python synthesize.py \\
        --model-path models/malay_base.nemo \\
        --text "Selamat pagi, apa khabar?" \\
        --language ms \\
        --output-dir output/

    # Legacy model: With external phonemizer
    python synthesize.py \\
        --model-path legacy_model.pt \\
        --text "Selamat pagi" \\
        --use-phonemes \\
        --output-dir output/

    # Generate from text file
    python synthesize.py \\
        --model-path models/malay.nemo \\
        --input-file sentences.txt \\
        --output-dir output/

    # Use specific speaker (kp_ms)
    python synthesize.py \\
        --model-path models/malay.nemo \\
        --text "Hello world" \\
        --speaker 2 \\
        --output-dir output/

Available speakers:
    0: anwar_ibrahim
    1: husein
    2: kp_ms
    3: shafiqah_idayu
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
        "--name",
        type=str,
        default=None,
        help="Output name prefix for batch synthesis (e.g. abc -> abc-1.wav, abc-2.wav)"
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
        "--code-switch",
        action="store_true",
        help="Split mixed Malay/English text and synthesize with per-segment languages"
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
        help="Convert text to phonemes using external phonemizer (legacy mode only, not needed for Phase 1/2 models)"
    )
    parser.add_argument(
        "--g2p-dict",
        type=str,
        default=DEFAULT_G2P_DICT,
        help=f"Path to G2P dictionary for Malay (default: {DEFAULT_G2P_DICT})"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to a NeMo manifest; if set, samples text from it"
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=3,
        help="Max number of texts to load from manifest (default: 3)"
    )
    
    args = parser.parse_args()
    
    # List speakers if requested
    if args.list_speakers:
        print("Available speakers:")
        for idx, name in sorted(SPEAKER_NAMES.items()):
            print(f"  {idx}: {name}")
        return
    
    # Validate input
    if not args.text and not args.input_file and not args.manifest:
        parser.error("Either --text, --input-file, or --manifest is required")
    
    # Initialize phonemizer if requested (legacy mode)
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
    
    # Load model (Malay tokenizer is added automatically if language='ms')
    model = load_model(
        args.model_path, 
        args.device, 
        g2p_dict=args.g2p_dict,
        language=args.language
    )
    
    # Prepare texts
    if args.manifest:
        texts = []
        with open(args.manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if len(texts) >= args.manifest_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "").strip()
                    if text:
                        texts.append(text)
                except Exception:
                    continue
        if not texts:
            logger.error(f"No texts found in manifest: {args.manifest}")
            return
        logger.info("Loaded texts from manifest:")
        for i, t in enumerate(texts, start=1):
            logger.info(f"  [{i}] {t}")
    elif args.text:
        raw_text = args.text.strip()
        if raw_text.startswith("[") and raw_text.endswith("]"):
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
                    texts = parsed
                else:
                    texts = [args.text]
            except Exception:
                logger.warning(
                    "Failed to parse --text as JSON list. "
                    "Use single quotes around the JSON, e.g. "
                    "TEXT='[\"a\",\"b\"]'."
                )
                texts = [args.text]
        else:
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
        if args.code_switch:
            audio, sr = synthesize_text_with_codeswitch(
                model,
                texts[0],
                speaker_index=args.speaker,
                apply_text_normalization=apply_tn,
            )
        else:
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
        if args.name:
            prefix = args.name
            index_start = 1
            separator = "-"
            pad_width = 0
        else:
            prefix = f"{args.language}_{speaker_name}"
            index_start = 0
            separator = "_"
            pad_width = 4
        synthesize_batch(
            model,
            texts,
            output_dir,
            language=args.language,
            speaker_index=args.speaker,
            apply_text_normalization=apply_tn,
            prefix=prefix,
            phonemizer=phonemizer,
            code_switch=args.code_switch,
            index_start=index_start,
            separator=separator,
            pad_width=pad_width,
        )


if __name__ == "__main__":
    main()
