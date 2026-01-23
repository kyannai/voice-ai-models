#!/usr/bin/env python3
"""
Prepare full KeSpeech dataset (1500h) from local files.

KeSpeech directory structure after extraction:
    KeSpeech/
    └── Tasks/
        └── ASR/
            ├── train_phase1/
            │   ├── text       (format: utt_id transcript)
            │   └── wav.scp    (format: utt_id wav_path)
            ├── train_phase2/
            ├── dev_phase1/
            ├── dev_phase2/
            └── test/

This script:
1. Extracts split tar.gz files if needed
2. Processes each subset (train_phase1, train_phase2, dev_phase1, dev_phase2, test)
3. Reads text and wav.scp files
4. Applies Chinese text normalization (removes <SPOKEN_NOISE>, converts numbers)
5. Creates NeMo manifests

Usage:
    python prepare_kespeech.py --data-dir /path/to/KeSpeech --output-dir ./data
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import soundfile as sf
from tqdm import tqdm
import re

# Add common directory to path for text_normalizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "common"))
try:
    from text_normalizer import normalize_text, preprocess_chinese_text
    TEXT_NORMALIZER_AVAILABLE = True
except ImportError:
    TEXT_NORMALIZER_AVAILABLE = False
    preprocess_chinese_text = None
    print("Warning: text_normalizer not available. Install num2words, cn2an, langid")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KeSpeech subsets
SUBSETS = ['train_phase1', 'train_phase2', 'dev_phase1', 'dev_phase2', 'test']
TRAIN_SUBSETS = ['train_phase1', 'train_phase2']
VAL_SUBSETS = ['dev_phase1', 'dev_phase2']
TEST_SUBSETS = ['test']


def extract_split_archives(splits_dir: Path, output_dir: Path) -> bool:
    """
    Extract split tar.gz archives or already-combined tar.gz.
    
    Args:
        splits_dir: Directory containing KeSpeech.tar.gz or KeSpeech.tar.gz.aa, .ab, etc.
        output_dir: Where to extract to
        
    Returns:
        True if extraction successful
    """
    # Check if already extracted
    tasks_dir = output_dir / "Tasks"
    if tasks_dir.exists() and (tasks_dir / "ASR").exists():
        logger.info("Data already extracted, skipping extraction")
        return True
    
    # Check for combined archive first
    combined_path = splits_dir / "KeSpeech.tar.gz"
    
    if combined_path.exists():
        logger.info(f"Found combined archive: {combined_path}")
    else:
        # Look for split files
        split_files = sorted(splits_dir.glob("KeSpeech.tar.gz.*"))
        
        if not split_files:
            logger.error(f"No archive files found in {splits_dir}")
            logger.error("Expected: KeSpeech.tar.gz or KeSpeech.tar.gz.aa, .ab, etc.")
            return False
        
        logger.info(f"Found {len(split_files)} split archive files")
        logger.info("Combining split files...")
        
        try:
            cmd = f"cat {splits_dir}/KeSpeech.tar.gz.* > {combined_path}"
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Combined archive: {combined_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to combine archives: {e}")
            return False
    
    # Extract
    logger.info(f"Extracting to {output_dir}...")
    logger.info("This may take a while (~135GB)...")
    try:
        cmd = f"tar -xzf {combined_path} -C {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info("Extraction complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract: {e}")
        return False


def normalize_chinese_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize Chinese text for ASR training.
    
    Uses comprehensive preprocessing:
    - Remove noise markers (<SPOKEN_NOISE>, [noise], [laughter], etc.)
    - Convert full-width to half-width characters
    - Remove punctuation (Chinese and English)
    - Clean up whitespace
    
    Note: Numbers are kept as-is (not converted to Chinese words).
    
    Args:
        text: Input Chinese text
        remove_punctuation: Whether to remove punctuation marks (default: True)
    """
    # Use the comprehensive Chinese preprocessor if available
    if TEXT_NORMALIZER_AVAILABLE and preprocess_chinese_text is not None:
        return preprocess_chinese_text(
            text,
            remove_punctuation=remove_punctuation,
            convert_fullwidth=True,
            remove_noise=True,
        )
    
    # Fallback: basic preprocessing if text_normalizer not available
    # Remove spoken noise markers
    text = re.sub(r'<SPOKEN_NOISE>', '', text)
    text = re.sub(r'\[SPOKEN_NOISE\]', '', text)
    text = re.sub(r'\[noise\]', '', text)
    text = re.sub(r'\[laughter\]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_text_file(text_path: Path) -> Dict[str, str]:
    """
    Load transcripts from text file.
    Format: utt_id transcript
    """
    transcripts = {}
    
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                transcripts[utt_id] = text
            elif len(parts) == 1:
                # Some entries might just have utt_id with empty text
                transcripts[parts[0]] = ""
    
    return transcripts


def load_wav_scp(wav_scp_path: Path) -> Dict[str, str]:
    """
    Load wav.scp file.
    Format: utt_id wav_path
    """
    wav_paths = {}
    
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, wav_path = parts
                wav_paths[utt_id] = wav_path
    
    return wav_paths


def get_audio_duration(wav_path: str) -> Optional[float]:
    """Get audio duration in seconds."""
    try:
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return None


# Module-level functions for multiprocessing (must be picklable)
def _check_batch(batch: List[str]) -> List[Tuple[str, bool]]:
    """Check a batch of files for existence."""
    return [(path, os.path.exists(path)) for path in batch]


def _get_duration_batch(batch: List[str]) -> List[Tuple[str, Optional[float]]]:
    """Get durations for a batch of files."""
    import soundfile as sf
    results = []
    for path in batch:
        try:
            info = sf.info(path)
            results.append((path, info.duration))
        except Exception:
            results.append((path, None))
    return results


def check_files_parallel(file_paths: List[str], num_workers: int = 32) -> Dict[str, bool]:
    """
    Check if files exist in parallel using multiprocessing.
    Returns dict mapping filepath -> exists.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Split into batches
    batch_size = max(1, len(file_paths) // num_workers // 4)
    batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]
    
    exists_dict = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_batch, batch) for batch in batches]
        
        with tqdm(total=len(file_paths), desc="Checking files") as pbar:
            for future in as_completed(futures):
                results = future.result()
                for path, exists in results:
                    exists_dict[path] = exists
                pbar.update(len(results))
    
    return exists_dict


def get_durations_parallel(file_paths: List[str], num_workers: int = 16) -> Dict[str, Optional[float]]:
    """
    Get audio durations in parallel using multiprocessing.
    Returns dict mapping filepath -> duration (or None if error).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Split into batches
    batch_size = max(1, len(file_paths) // num_workers // 4)
    batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]
    
    durations = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_get_duration_batch, batch) for batch in batches]
        
        with tqdm(total=len(file_paths), desc="Getting durations") as pbar:
            for future in as_completed(futures):
                results = future.result()
                for path, duration in results:
                    durations[path] = duration
                pbar.update(len(results))
    
    return durations


def process_subset(
    subset_dir: Path,
    base_dir: Path,
    normalize: bool = True,
    max_samples: Optional[int] = None,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
    num_workers: int = 16
) -> List[Dict]:
    """
    Process a single KeSpeech subset.
    
    Args:
        subset_dir: Path to subset directory (e.g., train_phase1/)
        base_dir: Base directory for resolving relative wav paths
        normalize: Whether to apply text normalization
        max_samples: Maximum samples to process
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        num_workers: Number of parallel workers
        
    Returns:
        List of manifest entries
    """
    subset_name = subset_dir.name
    
    # Load text and wav.scp
    text_path = subset_dir / "text"
    wav_scp_path = subset_dir / "wav.scp"
    
    if not text_path.exists():
        logger.warning(f"Text file not found: {text_path}")
        return []
    
    if not wav_scp_path.exists():
        logger.warning(f"wav.scp not found: {wav_scp_path}")
        return []
    
    logger.info(f"Processing {subset_name}...")
    
    transcripts = load_text_file(text_path)
    wav_paths_raw = load_wav_scp(wav_scp_path)
    
    logger.info(f"  Loaded {len(transcripts)} transcripts, {len(wav_paths_raw)} wav paths")
    
    # Match utterances
    utt_ids = list(set(transcripts.keys()) & set(wav_paths_raw.keys()))
    
    if max_samples:
        utt_ids = utt_ids[:max_samples]
    
    logger.info(f"  Matching utterances: {len(utt_ids)}")
    
    # Resolve all wav paths
    resolved_paths = {}
    for utt_id in utt_ids:
        wav_path = wav_paths_raw[utt_id]
        if not os.path.isabs(wav_path):
            wav_path = str(base_dir / wav_path)
        resolved_paths[utt_id] = wav_path
    
    # Check file existence in parallel
    logger.info(f"  Checking {len(resolved_paths)} audio files exist...")
    all_paths = list(resolved_paths.values())
    exists_dict = check_files_parallel(all_paths, num_workers=num_workers)
    
    # Filter to existing files
    existing_utt_ids = [utt_id for utt_id in utt_ids if exists_dict.get(resolved_paths[utt_id], False)]
    skipped_missing = len(utt_ids) - len(existing_utt_ids)
    
    if skipped_missing > 0:
        logger.warning(f"  Skipped {skipped_missing} missing audio files")
    
    logger.info(f"  Found {len(existing_utt_ids)} existing files")
    
    # Get durations in parallel for existing files
    existing_paths = [resolved_paths[utt_id] for utt_id in existing_utt_ids]
    durations = get_durations_parallel(existing_paths, num_workers=num_workers)
    
    # Build samples
    samples = []
    errors = 0
    skipped_duration = 0
    skipped_empty = 0
    
    for utt_id in tqdm(existing_utt_ids, desc=f"  Building {subset_name}"):
        try:
            text = transcripts[utt_id]
            wav_path = resolved_paths[utt_id]
            
            # Skip empty transcripts
            if not text.strip():
                skipped_empty += 1
                continue
            
            # Get duration
            duration = durations.get(wav_path)
            if duration is None:
                errors += 1
                continue
            
            # Filter by duration
            if duration < min_duration or duration > max_duration:
                skipped_duration += 1
                continue
            
            # Normalize text
            if normalize:
                text = normalize_chinese_text(text)
            
            # Skip if text became empty after normalization
            if not text.strip():
                skipped_empty += 1
                continue
            
            samples.append({
                'audio_filepath': wav_path,
                'text': text.strip(),
                'duration': round(duration, 4),
                'subset': subset_name
            })
            
        except Exception as e:
            errors += 1
    
    logger.info(f"  ✓ {len(samples)} samples")
    if skipped_duration > 0:
        logger.info(f"  Skipped {skipped_duration} samples (duration filter)")
    if skipped_empty > 0:
        logger.info(f"  Skipped {skipped_empty} samples (empty text)")
    if errors > 0:
        logger.warning(f"  {errors} errors")
    
    return samples


def save_manifest(samples: List[Dict], output_path: Path):
    """Save samples to NeMo manifest format (JSONL)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Remove subset field for manifest
            entry = {k: v for k, v in sample.items() if k != 'subset'}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(samples):,} samples to {output_path}")


def print_statistics(samples: List[Dict], name: str):
    """Print dataset statistics."""
    if not samples:
        logger.info(f"{name}: No samples")
        return
    
    total_duration = sum(s.get('duration', 0) for s in samples)
    avg_duration = total_duration / len(samples) if samples else 0
    avg_text_len = sum(len(s.get('text', '')) for s in samples) / len(samples) if samples else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"  Total samples: {len(samples):,}")
    logger.info(f"  Total duration: {total_duration / 3600:.1f} hours")
    logger.info(f"  Avg duration: {avg_duration:.2f}s")
    logger.info(f"  Avg text length: {avg_text_len:.1f} chars")
    
    # By subset
    subsets = {}
    for s in samples:
        subset = s.get('subset', 'unknown')
        if subset not in subsets:
            subsets[subset] = {'count': 0, 'duration': 0}
        subsets[subset]['count'] += 1
        subsets[subset]['duration'] += s.get('duration', 0)
    
    logger.info(f"\n  By subset:")
    for subset, stats in sorted(subsets.items()):
        hours = stats['duration'] / 3600
        logger.info(f"    {subset}: {stats['count']:,} samples ({hours:.1f}h)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare full KeSpeech dataset for NeMo training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing KeSpeech data (or KeSpeech.splits/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs='+',
        default=SUBSETS,
        help=f"Subsets to process (default: {SUBSETS})"
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Enable text normalization (convert numbers to words)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction (assume already extracted)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per subset (for testing)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    logger.info("=" * 60)
    logger.info("KeSpeech Full Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Subsets: {args.subsets}")
    logger.info(f"Text normalization: {'enabled' if args.normalize_text else 'disabled'}")
    logger.info("=" * 60)
    
    # Check for split archives (only if not already extracted)
    splits_dir = data_dir / "KeSpeech.splits"
    combined_archive = splits_dir / "KeSpeech.tar.gz"
    tasks_exists = (data_dir / "Tasks" / "ASR").exists() or (data_dir / "KeSpeech" / "Tasks" / "ASR").exists()
    
    if splits_dir.exists() and not args.no_extract and not tasks_exists:
        if combined_archive.exists():
            logger.info(f"\nFound archive in {splits_dir}")
            if not extract_split_archives(splits_dir, data_dir):
                logger.error("Failed to extract archives")
                sys.exit(1)
        else:
            logger.info(f"\nNo archive found in {splits_dir}, assuming already extracted")
    
    # Find Tasks/ASR directory and determine base_dir for wav paths
    asr_dir = None
    audio_base_dir = None  # Base directory for resolving wav.scp paths
    
    # Check possible locations
    if (data_dir / "KeSpeech" / "Tasks" / "ASR").exists():
        asr_dir = data_dir / "KeSpeech" / "Tasks" / "ASR"
        audio_base_dir = data_dir / "KeSpeech"  # Audio/ is under KeSpeech/
    elif (data_dir / "Tasks" / "ASR").exists():
        asr_dir = data_dir / "Tasks" / "ASR"
        audio_base_dir = data_dir  # Audio/ is at same level as Tasks/
    
    if asr_dir is None:
        logger.error(f"Could not find Tasks/ASR directory in {data_dir}")
        logger.error("Expected structure: {data_dir}/KeSpeech/Tasks/ASR/train_phase1/...")
        logger.error("Or: {data_dir}/Tasks/ASR/train_phase1/...")
        sys.exit(1)
    
    logger.info(f"\nFound ASR data at: {asr_dir}")
    logger.info(f"Audio base directory: {audio_base_dir}")
    
    # Process all subsets
    all_samples = []
    
    for subset in args.subsets:
        subset_dir = asr_dir / subset
        if not subset_dir.exists():
            logger.warning(f"Subset not found: {subset_dir}")
            continue
        
        samples = process_subset(
            subset_dir,
            base_dir=audio_base_dir,  # Use correct base for wav paths
            normalize=args.normalize_text,
            max_samples=args.max_samples,
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )
        all_samples.extend(samples)
    
    if not all_samples:
        logger.error("No samples processed!")
        sys.exit(1)
    
    # Split into train/val
    logger.info(f"\n✂️  Splitting samples...")
    
    train_samples = [s for s in all_samples if s.get('subset') in TRAIN_SUBSETS]
    val_samples = [s for s in all_samples if s.get('subset') in VAL_SUBSETS]
    test_samples = [s for s in all_samples if s.get('subset') in TEST_SUBSETS]
    
    # If no separate val split, use dev subsets or split train
    if not val_samples and train_samples:
        import random
        random.seed(42)
        random.shuffle(train_samples)
        split_idx = int(len(train_samples) * 0.95)
        val_samples = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]
    
    # Print statistics
    print_statistics(train_samples, "Training Set")
    print_statistics(val_samples, "Validation Set")
    if test_samples:
        print_statistics(test_samples, "Test Set")
    
    # Save manifests
    logger.info("\n💾 Saving manifests...")
    manifest_dir = output_dir / "manifests"
    
    save_manifest(train_samples, manifest_dir / "train_manifest.json")
    save_manifest(val_samples, manifest_dir / "val_manifest.json")
    if test_samples:
        save_manifest(test_samples, manifest_dir / "test_manifest.json")
    
    # Success
    logger.info("\n" + "=" * 60)
    logger.info("✅ KeSpeech Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"Train: {manifest_dir / 'train_manifest.json'}")
    logger.info(f"Val: {manifest_dir / 'val_manifest.json'}")
    if test_samples:
        logger.info(f"Test: {manifest_dir / 'test_manifest.json'}")
    
    total_hours = sum(s.get('duration', 0) for s in all_samples) / 3600
    logger.info(f"\nTotal: {len(all_samples):,} samples ({total_hours:.1f} hours)")
    
    # Example entry
    if train_samples:
        logger.info("\n📄 Example manifest entry:")
        example = {k: v for k, v in train_samples[0].items() if k != 'subset'}
        logger.info(json.dumps(example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
