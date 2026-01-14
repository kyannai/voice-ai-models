#!/usr/bin/env python3
"""
Prepare Malaysian-TTS dataset for MagpieTTS training.

This script converts the downloaded HuggingFace dataset to NeMo manifest format:
1. Loads metadata from saved HuggingFace dataset
2. Loads audio from extracted MP3 files
3. Resamples to target sample rate (22.05kHz for MagpieTTS)
4. Converts text to phonemes (optional, recommended)
5. Saves as WAV files
6. Creates NeMo manifest files with speaker IDs and language codes

Output manifest format (JSONL):
{"audio_filepath": "/path/to/audio.wav", "text": "phonemes or text", "duration": 2.5, "speaker": 0, "language": "ms"}

Usage:
    # With phonemes (recommended)
    python prepare_data.py --data-dir data/raw --audio-output-dir data/audio --manifest-output-dir data/manifests --use-phonemes
    
    # Without phonemes (character-level)
    python prepare_data.py --data-dir data/raw --audio-output-dir data/audio --manifest-output-dir data/manifests
"""

import argparse
import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm

# Suppress phonemizer warnings (they're expected for code-switching)
warnings.filterwarnings("ignore", message=".*language switch.*")
warnings.filterwarnings("ignore", message=".*words count mismatch.*")
warnings.filterwarnings("ignore", message=".*extra phones.*")

# Phonemizer import (optional)
PHONEMIZER_AVAILABLE = False
try:
    from malay_phonemizer import MalayPhonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    pass

# Number of parallel workers for audio processing
NUM_WORKERS = 8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Speaker name to ID mapping
SPEAKER_MAP = {
    "anwar_ibrahim": 0,
    "husein": 1,
    "kp_ms": 2,
    "kp_zh": 3,
    "shafiqah_idayu": 4,
}

# Target sample rate for MagpieTTS (22.05kHz)
DEFAULT_TARGET_SAMPLE_RATE = 22050


def load_and_resample_audio(
    audio_path: Path,
    target_sr: int
) -> tuple[np.ndarray, float] | None:
    """
    Load audio file (MP3/WAV) and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio array, duration in seconds) or None if failed
    """
    try:
        # Load audio with librosa (handles MP3, WAV, etc.)
        audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        
        # Calculate duration
        duration = len(audio) / target_sr
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio, duration
        
    except Exception as e:
        logger.debug(f"Error loading {audio_path}: {e}")
        return None


def save_audio(audio: np.ndarray, path: Path, sample_rate: int):
    """Save audio to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype='PCM_16')


def process_single_audio(args: tuple) -> dict | None:
    """
    Process a single audio file (for parallel processing).
    
    Args:
        args: Tuple of (idx, source_path, output_path, target_sr, text, speaker_id)
        
    Returns:
        Manifest entry dict or None if failed
    """
    idx, source_path, output_path, target_sr, text, speaker_id = args
    
    try:
        # Load and resample audio
        result = load_and_resample_audio(Path(source_path), target_sr)
        if result is None:
            return None
        
        audio, duration = result
        
        # Skip very short or very long audio
        if duration < 0.5 or duration > 30.0:
            return None
        
        # Save audio as WAV
        save_audio(audio, Path(output_path), target_sr)
        
        # Create manifest entry
        return {
            "audio_filepath": str(Path(output_path).absolute()),
            "text": text,
            "duration": round(duration, 3),
            "speaker": speaker_id,
            "language": "ms"
        }
    except Exception:
        return None


def batch_phonemize(texts: list[str], phonemizer) -> list[str]:
    """
    Phonemize texts in batch for better performance.
    
    Args:
        texts: List of texts to phonemize
        phonemizer: MalayPhonemizer instance
        
    Returns:
        List of phonemized texts
    """
    # Process in batches to avoid memory issues
    batch_size = 1000
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="    Phonemizing", unit="batch"):
        batch = texts[i:i + batch_size]
        batch_results = phonemizer.phonemize_batch(batch)
        results.extend(batch_results)
    
    return results


def prepare_speaker_data(
    speaker_name: str,
    metadata_dir: Path,
    source_audio_dir: Path,
    output_audio_dir: Path,
    target_sr: int,
    max_samples: int | None = None,
    phonemizer: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Process data for a single speaker.
    
    Args:
        speaker_name: Name of the speaker
        metadata_dir: Directory containing the speaker's metadata
        source_audio_dir: Directory containing source audio files (extracted from ZIPs)
        output_audio_dir: Directory to save processed audio files
        target_sr: Target sample rate
        max_samples: Maximum samples to process for this speaker
        phonemizer: Optional MalayPhonemizer instance for text-to-phoneme conversion
        
    Returns:
        List of manifest entries
    """
    logger.info(f"Processing speaker: {speaker_name}")
    if phonemizer:
        logger.info(f"  Using phoneme conversion (batch mode)")
    
    # Load dataset metadata
    dataset = load_from_disk(str(metadata_dir))
    total_samples = len(dataset)
    
    if max_samples and max_samples < total_samples:
        # Randomly sample indices
        indices = np.random.choice(total_samples, max_samples, replace=False)
        indices = sorted(indices)
        logger.info(f"  Limiting to {max_samples:,} samples (from {total_samples:,})")
    else:
        indices = list(range(total_samples))
        logger.info(f"  Processing all {total_samples:,} samples")
    
    speaker_id = SPEAKER_MAP.get(speaker_name, len(SPEAKER_MAP))
    speaker_output_dir = output_audio_dir / speaker_name
    speaker_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect all texts and audio paths
    logger.info(f"  Step 1/3: Collecting metadata...")
    texts = []
    valid_indices = []
    audio_paths = []
    audio_missing = 0
    
    for idx in tqdm(indices, desc="    Collecting", unit="samples"):
        sample = dataset[int(idx)]
        
        text = sample.get('normalized') or sample.get('original', '')
        if not text or not text.strip():
            continue
        
        audio_filename = sample.get('audio_filename', '')
        if not audio_filename:
            continue
        
        source_audio_path = source_audio_dir / audio_filename
        if not source_audio_path.exists():
            audio_missing += 1
            continue
        
        texts.append(text.strip())
        valid_indices.append(idx)
        audio_paths.append(str(source_audio_path))
    
    logger.info(f"    Found {len(texts):,} valid samples, {audio_missing:,} missing audio")
    
    # Step 2: Batch phonemize all texts
    if phonemizer and texts:
        logger.info(f"  Step 2/3: Batch phonemizing {len(texts):,} texts...")
        phonemized_texts = batch_phonemize(texts, phonemizer)
        texts = phonemized_texts
    else:
        logger.info(f"  Step 2/3: Skipping phonemization (not enabled)")
    
    # Step 3: Process audio in parallel
    logger.info(f"  Step 3/3: Processing audio (parallel, {NUM_WORKERS} workers)...")
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, (idx, text, source_path) in enumerate(zip(valid_indices, texts, audio_paths)):
        if not text or not text.strip():
            continue
        output_filename = f"{speaker_name}_{idx:08d}.wav"
        output_path = str(speaker_output_dir / output_filename)
        process_args.append((idx, source_path, output_path, target_sr, text, speaker_id))
    
    # Process in parallel
    manifest_entries = []
    errors = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_audio, args): args for args in process_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"    {speaker_name}", unit="samples"):
            result = future.result()
            if result:
                manifest_entries.append(result)
            else:
                errors += 1
    
    logger.info(f"  Processed: {len(manifest_entries):,} samples, Errors: {errors:,}")
    
    return manifest_entries


def split_train_val(
    entries: list[dict],
    train_ratio: float = 0.95,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """
    Split manifest entries into train and validation sets.
    
    Stratifies by speaker to ensure all speakers are represented in both sets.
    """
    np.random.seed(seed)
    
    # Group by speaker
    by_speaker: dict[int, list[dict]] = {}
    for entry in entries:
        speaker = entry['speaker']
        if speaker not in by_speaker:
            by_speaker[speaker] = []
        by_speaker[speaker].append(entry)
    
    train_entries = []
    val_entries = []
    
    for speaker, speaker_entries in by_speaker.items():
        # Shuffle
        indices = np.random.permutation(len(speaker_entries))
        split_idx = int(len(indices) * train_ratio)
        
        for i in indices[:split_idx]:
            train_entries.append(speaker_entries[i])
        for i in indices[split_idx:]:
            val_entries.append(speaker_entries[i])
    
    # Shuffle final lists
    np.random.shuffle(train_entries)
    np.random.shuffle(val_entries)
    
    return train_entries, val_entries


def save_manifest(entries: list[dict], path: Path):
    """Save manifest entries to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(entries):,} entries to {path}")


def print_statistics(entries: list[dict], name: str):
    """Print statistics for a set of manifest entries."""
    if not entries:
        logger.info(f"\n{name}: No entries")
        return
    
    durations = [e['duration'] for e in entries]
    text_lengths = [len(e['text']) for e in entries]
    
    # Count by speaker
    by_speaker: dict[int, int] = {}
    for e in entries:
        speaker = e['speaker']
        by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
    
    total_duration = sum(durations)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"  Total samples: {len(entries):,}")
    logger.info(f"  Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"  Avg duration: {np.mean(durations):.2f}s")
    logger.info(f"  Duration range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
    logger.info(f"  Avg text length: {np.mean(text_lengths):.1f} chars")
    
    logger.info(f"\n  Samples by speaker:")
    speaker_names = {v: k for k, v in SPEAKER_MAP.items()}
    for speaker_id, count in sorted(by_speaker.items()):
        speaker_name = speaker_names.get(speaker_id, f"speaker_{speaker_id}")
        pct = count / len(entries) * 100
        logger.info(f"    {speaker_name:20s}: {count:>8,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Malaysian-TTS dataset for MagpieTTS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare full dataset
    python prepare_data.py \\
        --data-dir data/raw \\
        --audio-output-dir data/audio \\
        --manifest-output-dir data/manifests
    
    # Prepare small subset for testing
    python prepare_data.py \\
        --data-dir data/raw \\
        --audio-output-dir data/audio \\
        --manifest-output-dir data/manifests \\
        --max-samples 10000
        """
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing downloaded data (with metadata/ and audio/ subdirs)"
    )
    parser.add_argument(
        "--audio-output-dir",
        type=str,
        required=True,
        help="Directory to save processed audio files (WAV)"
    )
    parser.add_argument(
        "--manifest-output-dir",
        type=str,
        required=True,
        help="Directory to save manifest files"
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_RATE,
        help=f"Target sample rate in Hz (default: {DEFAULT_TARGET_SAMPLE_RATE})"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Ratio of samples for training (default: 0.95)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to process (distributed across speakers)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--use-phonemes",
        action="store_true",
        help="Convert text to phonemes using espeak-ng (recommended for better quality)"
    )
    parser.add_argument(
        "--no-code-switching",
        action="store_true",
        help="Disable code-switching detection (use pure Malay phonemization)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    metadata_dir = data_dir / "metadata"
    source_audio_dir = data_dir / "audio"
    output_audio_dir = Path(args.audio_output_dir)
    manifest_output_dir = Path(args.manifest_output_dir)
    
    if not metadata_dir.exists():
        logger.error(f"Metadata directory does not exist: {metadata_dir}")
        logger.error("Run 'make download' or 'make download-small' first.")
        return
    
    if not source_audio_dir.exists():
        logger.error(f"Audio directory does not exist: {source_audio_dir}")
        logger.error("Run download without --skip-audio flag.")
        return
    
    # Find speaker directories
    speaker_dirs = [d for d in metadata_dir.iterdir() if d.is_dir()]
    if not speaker_dirs:
        logger.error(f"No speaker directories found in {metadata_dir}")
        return
    
    logger.info(f"{'='*70}")
    logger.info(f"MagpieTTS Malay Data Preparation")
    logger.info(f"{'='*70}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Source audio: {source_audio_dir}")
    logger.info(f"Output audio: {output_audio_dir}")
    logger.info(f"Manifest output: {manifest_output_dir}")
    logger.info(f"Target sample rate: {args.target_sample_rate} Hz")
    logger.info(f"Train/val split: {args.train_split:.0%}/{1-args.train_split:.0%}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples:,}")
    logger.info(f"Speakers found: {[d.name for d in speaker_dirs]}")
    
    np.random.seed(args.seed)
    
    # Initialize phonemizer if requested
    phonemizer = None
    if args.use_phonemes:
        if not PHONEMIZER_AVAILABLE:
            logger.error("Phonemizer not available. Install with: pip install phonemizer")
            logger.error("Also install espeak-ng: apt install espeak-ng (Ubuntu) or brew install espeak (macOS)")
            return
        
        try:
            phonemizer = MalayPhonemizer()
            handle_code_switching = not args.no_code_switching
            logger.info(f"Phonemizer enabled (code-switching: {handle_code_switching})")
        except Exception as e:
            logger.error(f"Failed to initialize phonemizer: {e}")
            return
    else:
        logger.info("Using character-level text (no phonemization)")
    
    # Calculate samples per speaker
    samples_per_speaker = None
    if args.max_samples:
        samples_per_speaker = args.max_samples // len(speaker_dirs)
    
    # Process each speaker
    all_entries = []
    
    for speaker_dir in sorted(speaker_dirs):
        speaker_name = speaker_dir.name
        entries = prepare_speaker_data(
            speaker_name=speaker_name,
            metadata_dir=speaker_dir,
            source_audio_dir=source_audio_dir,
            output_audio_dir=output_audio_dir,
            target_sr=args.target_sample_rate,
            max_samples=samples_per_speaker,
            phonemizer=phonemizer,
        )
        all_entries.extend(entries)
    
    if not all_entries:
        logger.error("No samples processed successfully!")
        return
    
    logger.info(f"\nTotal samples processed: {len(all_entries):,}")
    
    # Split train/val
    logger.info(f"\nSplitting train/val ({args.train_split:.0%}/{1-args.train_split:.0%})...")
    train_entries, val_entries = split_train_val(
        all_entries,
        train_ratio=args.train_split,
        seed=args.seed
    )
    
    # Print statistics
    print_statistics(train_entries, "Training Set")
    print_statistics(val_entries, "Validation Set")
    
    # Save manifests
    logger.info(f"\nSaving manifests...")
    train_manifest_path = manifest_output_dir / "train_manifest.json"
    val_manifest_path = manifest_output_dir / "val_manifest.json"
    
    save_manifest(train_entries, train_manifest_path)
    save_manifest(val_entries, val_manifest_path)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Data Preparation Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Train manifest: {train_manifest_path}")
    logger.info(f"  Samples: {len(train_entries):,}")
    logger.info(f"Val manifest: {val_manifest_path}")
    logger.info(f"  Samples: {len(val_entries):,}")
    logger.info(f"Audio files: {output_audio_dir}")
    logger.info(f"Text format: {'phonemes (IPA)' if args.use_phonemes else 'characters'}")
    
    # Show example manifest entry
    if train_entries:
        logger.info(f"\nExample manifest entry:")
        logger.info(json.dumps(train_entries[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
