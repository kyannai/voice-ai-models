#!/usr/bin/env python3
"""
Explore the downloaded Malaysian-TTS dataset.

This script loads the downloaded dataset and prints statistics about:
- Number of samples per speaker
- Text length distribution
- Audio file availability
- Sample entries for inspection

Usage:
    python explore_dataset.py --data-dir data/raw
"""

import argparse
import logging
from pathlib import Path

from datasets import load_from_disk
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def explore_dataset(data_dir: Path):
    """
    Explore the downloaded Malaysian-TTS dataset.
    
    Args:
        data_dir: Directory containing downloaded data (with metadata/ and audio/ subdirs)
    """
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "metadata"
    audio_dir = data_dir / "audio"
    
    if not metadata_dir.exists():
        logger.error(f"Metadata directory does not exist: {metadata_dir}")
        logger.error("Run 'make download' or 'make download-small' first.")
        return
    
    # Find all speaker directories
    speaker_dirs = [d for d in metadata_dir.iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        logger.error(f"No speaker directories found in {metadata_dir}")
        return
    
    logger.info(f"{'='*70}")
    logger.info(f"Malaysian-TTS Dataset Exploration")
    logger.info(f"{'='*70}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Metadata directory: {metadata_dir}")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Speakers found: {len(speaker_dirs)}")
    
    # Check audio directories
    if audio_dir.exists():
        audio_subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
        logger.info(f"Audio subdirectories: {len(audio_subdirs)}")
    else:
        logger.warning(f"Audio directory does not exist (run download without --skip-audio)")
        audio_subdirs = []
    
    total_samples = 0
    total_audio_found = 0
    total_audio_missing = 0
    all_text_lengths = []
    speaker_stats = []
    
    for speaker_dir in sorted(speaker_dirs):
        speaker_name = speaker_dir.name
        logger.info(f"\n{'-'*70}")
        logger.info(f"Speaker: {speaker_name}")
        logger.info(f"{'-'*70}")
        
        try:
            # Load dataset from disk
            dataset = load_from_disk(str(speaker_dir))
            num_samples = len(dataset)
            total_samples += num_samples
            
            logger.info(f"  Samples: {num_samples:,}")
            
            # Get column names
            columns = dataset.column_names
            logger.info(f"  Columns: {columns}")
            
            # Analyze text lengths
            if 'normalized' in columns:
                text_col = 'normalized'
            elif 'original' in columns:
                text_col = 'original'
            else:
                text_col = None
            
            if text_col:
                # Sample texts for length analysis
                sample_size = min(1000, num_samples)
                sample_indices = np.random.choice(num_samples, sample_size, replace=False)
                sample_texts = [dataset[int(i)][text_col] for i in sample_indices]
                text_lengths = [len(t) for t in sample_texts if t]
                
                if text_lengths:
                    all_text_lengths.extend(text_lengths)
                    avg_len = np.mean(text_lengths)
                    min_len = np.min(text_lengths)
                    max_len = np.max(text_lengths)
                    logger.info(f"  Text length (sampled {sample_size}): avg={avg_len:.1f}, min={min_len}, max={max_len}")
            
            # Check audio file availability (spot check)
            if audio_dir.exists():
                check_count = min(100, num_samples)
                found = 0
                missing = 0
                missing_examples = []
                
                for i in range(check_count):
                    audio_fn = dataset[i].get('audio_filename', '')
                    audio_path = audio_dir / audio_fn
                    if audio_path.exists():
                        found += 1
                    else:
                        missing += 1
                        if len(missing_examples) < 3:
                            missing_examples.append(audio_fn)
                
                logger.info(f"  Audio check ({check_count} samples): {found} found, {missing} missing")
                if missing_examples:
                    logger.info(f"    Missing examples: {missing_examples[:3]}")
                
                total_audio_found += found
                total_audio_missing += missing
            
            # Show sample entries
            logger.info(f"\n  Sample entries:")
            for i in range(min(3, num_samples)):
                sample = dataset[i]
                original = sample.get('original', 'N/A')
                if len(original) > 80:
                    original = original[:80] + '...'
                normalized = sample.get('normalized', 'N/A')
                if len(normalized) > 80:
                    normalized = normalized[:80] + '...'
                audio_fn = sample.get('audio_filename', 'N/A')
                
                logger.info(f"    [{i}] original: {original}")
                logger.info(f"        normalized: {normalized}")
                logger.info(f"        audio_filename: {audio_fn}")
            
            speaker_stats.append({
                'speaker': speaker_name,
                'samples': num_samples
            })
            
        except Exception as e:
            logger.error(f"  Error loading {speaker_name}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total speakers: {len(speaker_stats)}")
    logger.info(f"Total samples: {total_samples:,}")
    
    if total_audio_found + total_audio_missing > 0:
        audio_pct = total_audio_found / (total_audio_found + total_audio_missing) * 100
        logger.info(f"Audio availability (spot check): {audio_pct:.1f}%")
    
    if all_text_lengths:
        logger.info(f"\nText length statistics (sampled):")
        logger.info(f"  Mean: {np.mean(all_text_lengths):.1f} chars")
        logger.info(f"  Median: {np.median(all_text_lengths):.1f} chars")
        logger.info(f"  Min: {np.min(all_text_lengths)} chars")
        logger.info(f"  Max: {np.max(all_text_lengths)} chars")
        logger.info(f"  Std: {np.std(all_text_lengths):.1f} chars")
    
    logger.info(f"\nSamples per speaker:")
    for stat in speaker_stats:
        pct = stat['samples'] / total_samples * 100 if total_samples > 0 else 0
        logger.info(f"  {stat['speaker']:20s}: {stat['samples']:>8,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Explore the downloaded Malaysian-TTS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python explore_dataset.py --data-dir data/raw
        """
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing downloaded dataset"
    )
    
    args = parser.parse_args()
    explore_dataset(Path(args.data_dir))


if __name__ == "__main__":
    main()
