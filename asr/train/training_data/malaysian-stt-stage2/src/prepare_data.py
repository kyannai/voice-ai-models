#!/usr/bin/env python3
"""
Data preparation script for Malaysian STT Stage 2 dataset.
Converts parquet files to NeMo manifest format.

Stage2 parquet format:
- *_segments parquets: audio_filename, new_text (plain text)
- *_words parquets: audio_filename, new_text (Whisper timestamps for duration)

NeMo manifest format (JSONL):
{"audio_filepath": "/abs/path/audio.mp3", "text": "transcription", "duration": 2.78}
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_duration_from_whisper_timestamp(text: str) -> float:
    """
    Extract duration from Whisper-style timestamp string.
    Gets the last numeric timestamp.
    
    Example: "<|0.26|> Saya<|0.48|>...<|4.18|><|endoftext|>" -> 4.18
    """
    if not text:
        return 0.0
    times = re.findall(r'<\|(\d+\.\d+)\|>', text)
    return float(times[-1]) if times else 0.0


def get_dataset_pairs(data_dir: Path) -> Dict[str, Tuple[Path, Optional[Path]]]:
    """
    Find matching *_segments and *_words parquet file pairs.
    
    Returns dict mapping dataset_name -> (segments_path, words_path)
    """
    parquet_files = list(data_dir.glob("*.parquet"))
    
    # Group by base name
    segments_files = {}
    words_files = {}
    
    for pf in parquet_files:
        name = pf.stem.split('-00000')[0]  # Remove -00000-of-00001
        
        if name.endswith('_segments') or name.endswith('_segment_timestamp'):
            base_name = name.replace('_segments', '').replace('_segment_timestamp', '')
            segments_files[base_name] = pf
        elif name.endswith('_words') or name.endswith('_word_timestamp'):
            base_name = name.replace('_words', '').replace('_word_timestamp', '')
            words_files[base_name] = pf
        elif 'noise' in name or 'audioset' in name:
            # Skip noise/audioset (no transcription)
            continue
        else:
            # Single file without segments/words suffix - treat as segments
            segments_files[name] = pf
    
    # Match pairs
    pairs = {}
    for base_name, seg_path in segments_files.items():
        words_path = words_files.get(base_name)
        pairs[base_name] = (seg_path, words_path)
    
    return pairs


def load_and_process_dataset(
    segments_path: Path,
    words_path: Optional[Path],
    dataset_name: str
) -> pd.DataFrame:
    """
    Load a dataset from segments (text) and words (duration) parquets.
    
    If words_path is None, duration will be set to 0 (will need to compute from audio).
    """
    logger.info(f"Loading {dataset_name}...")
    
    # Load segments (text)
    seg_table = pq.read_table(segments_path)
    seg_df = seg_table.to_pandas()
    logger.info(f"  Segments rows: {len(seg_df):,}")
    
    # Rename new_text to text
    if 'new_text' in seg_df.columns:
        seg_df = seg_df.rename(columns={'new_text': 'text'})
    
    # Filter nulls/empty
    seg_df = seg_df.dropna(subset=['audio_filename', 'text'])
    seg_df = seg_df[seg_df['text'].str.len() > 0]
    
    # Clean text: remove Whisper markers <|...|> if present
    logger.info(f"  Cleaning text (removing Whisper markers)...")
    seg_df['text'] = seg_df['text'].str.replace(r'<\|[^|]*\|>', '', regex=True)
    seg_df['text'] = seg_df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Filter empty texts after cleaning
    seg_df = seg_df[seg_df['text'].str.len() > 0]
    
    # Load words for duration if available
    if words_path and words_path.exists():
        logger.info(f"  Loading words for duration extraction...")
        words_table = pq.read_table(words_path)
        words_df = words_table.to_pandas()
        
        # Extract duration from whisper timestamps
        logger.info(f"  Extracting durations...")
        words_df['duration'] = words_df['new_text'].apply(extract_duration_from_whisper_timestamp)
        
        # Keep only audio_filename and duration
        duration_df = words_df[['audio_filename', 'duration']].drop_duplicates(subset=['audio_filename'])
        
        # Merge with segments
        seg_df = seg_df.merge(duration_df, on='audio_filename', how='left')
        seg_df['duration'] = seg_df['duration'].fillna(0.0)
    else:
        logger.info(f"  No words file - duration will be 0 (compute from audio later)")
        seg_df['duration'] = 0.0
    
    # Add source
    seg_df['source'] = dataset_name
    
    # Select columns
    seg_df = seg_df[['audio_filename', 'text', 'duration', 'source']].copy()
    
    logger.info(f"  Final samples: {len(seg_df):,}")
    return seg_df


def load_all_datasets(
    data_dir: Path,
    datasets: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load all dataset pairs from the data directory.
    
    Args:
        data_dir: Directory containing parquet files
        datasets: Optional list of dataset names to include
    
    Returns concatenated DataFrame with all samples.
    """
    pairs = get_dataset_pairs(data_dir)
    
    if not pairs:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    logger.info(f"Found {len(pairs)} datasets")
    
    # Filter by dataset names if specified
    if datasets:
        filtered_pairs = {k: v for k, v in pairs.items() if k in datasets}
        if not filtered_pairs:
            available = list(pairs.keys())
            raise ValueError(f"No datasets match: {datasets}. Available: {available}")
        logger.info(f"Filtering to {len(filtered_pairs)} datasets: {list(filtered_pairs.keys())}")
        pairs = filtered_pairs
    
    dfs = []
    for dataset_name, (seg_path, words_path) in pairs.items():
        df = load_and_process_dataset(seg_path, words_path, dataset_name)
        dfs.append(df)
    
    # Concatenate all DataFrames
    logger.info("Concatenating all datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples: {len(combined_df):,}")
    
    return combined_df


def validate_audio_files(
    df: pd.DataFrame,
    audio_base_dir: Path,
    max_check: int = 1000
) -> Tuple[int, int]:
    """
    Validate that audio files exist (spot check).
    """
    check_df = df.sample(n=min(max_check, len(df)), random_state=42)
    
    found = 0
    missing = 0
    missing_examples = []
    
    for audio_filename in tqdm(check_df['audio_filename'], desc="Validating audio files"):
        audio_path = audio_base_dir / audio_filename
        if audio_path.exists():
            found += 1
        else:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(audio_path))
    
    if missing_examples:
        logger.warning(f"Missing audio file examples:")
        for ex in missing_examples:
            logger.warning(f"  {ex}")
    
    return found, missing


def filter_existing_audio(
    df: pd.DataFrame,
    audio_base_dir: Path
) -> pd.DataFrame:
    """
    Filter DataFrame to only include samples where audio file exists.
    This checks EVERY file (slower but ensures no missing files).
    """
    logger.info(f"Checking {len(df):,} audio files exist (this may take a while)...")
    
    # Vectorized check using apply
    def file_exists(audio_filename):
        return (audio_base_dir / audio_filename).exists()
    
    # Use tqdm for progress
    tqdm.pandas(desc="Checking audio files")
    exists_mask = df['audio_filename'].progress_apply(file_exists)
    
    original_count = len(df)
    df_filtered = df[exists_mask].copy()
    missing_count = original_count - len(df_filtered)
    
    logger.info(f"  Found: {len(df_filtered):,}, Missing: {missing_count:,}")
    if missing_count > 0:
        logger.warning(f"  Removed {missing_count:,} samples with missing audio files")
    
    return df_filtered


def split_train_val(
    df: pd.DataFrame,
    train_ratio: float = 0.95,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and validation sets, stratified by source.
    """
    train_dfs = []
    val_dfs = []
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source].sample(frac=1, random_state=seed)
        split_idx = int(len(source_df) * train_ratio)
        
        train_dfs.append(source_df.iloc[:split_idx])
        val_dfs.append(source_df.iloc[split_idx:])
        
        logger.info(f"  {source}: {split_idx:,} train, {len(source_df) - split_idx:,} val")
    
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=seed)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=seed)
    
    return train_df, val_df


def save_manifest_fast(
    df: pd.DataFrame,
    output_path: Path,
    audio_base_dir: Path
):
    """
    Save DataFrame to NeMo manifest format (JSONL) using fast batch operations.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create absolute paths column
    audio_base_str = str(audio_base_dir.absolute())
    df = df.copy()
    df['audio_filepath'] = audio_base_str + '/' + df['audio_filename']
    
    logger.info(f"Writing {len(df):,} samples to {output_path.name}...")
    
    # Select columns for manifest
    manifest_df = df[['audio_filepath', 'text', 'duration']]
    
    # Write as JSONL using pandas (much faster than row-by-row)
    manifest_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    logger.info(f"Saved {len(df):,} samples to {output_path}")


def print_statistics(df: pd.DataFrame, name: str):
    """Print dataset statistics."""
    total_duration = df['duration'].sum()
    avg_duration = df['duration'].mean()
    avg_text_len = df['text'].str.len().mean()
    
    # Count by source
    by_source = df['source'].value_counts()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"  Total samples: {len(df):,}")
    logger.info(f"  Total duration: {total_duration / 3600:.1f} hours")
    logger.info(f"  Avg duration: {avg_duration:.2f}s")
    logger.info(f"  Avg text length: {avg_text_len:.1f} chars")
    logger.info(f"\n  By source:")
    for source, count in by_source.items():
        logger.info(f"    {source}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Malaysian STT Stage 2 data for NeMo training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (from src/ directory)
  python prepare_data.py --data-dir ../data/data --audio-base-dir ../data --output-dir ../output
  
  # Test with small subset
  python prepare_data.py --data-dir ../data/data --audio-base-dir ../data --output-dir ./test_output \\
    --max-samples 10000 --validate-audio
  
  # Specific datasets only
  python prepare_data.py --data-dir ../data/data --audio-base-dir ../data --output-dir ../output \\
    --datasets malaysian_multiturn_chat_assistants text_chat_assistant
        """
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--audio-base-dir",
        type=str,
        required=True,
        help="Base directory for audio files (audio_filename paths are relative to this)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for manifest files"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Ratio of samples for training (default: 0.95)"
    )
    parser.add_argument(
        "--validate-audio",
        action="store_true",
        help="Validate that audio files exist (spot check 1000 files)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=None,
        help="Only include specific datasets (by base name). "
             "E.g., --datasets malaysian_multiturn_chat_assistants text_chat_assistant"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--filter-existing",
        action="store_true",
        help="Only include samples where audio file exists (checks ALL files, slower but safe)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir).resolve()
    audio_base_dir = Path(args.audio_base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # List datasets mode
    if args.list_datasets:
        pairs = get_dataset_pairs(data_dir)
        logger.info(f"Available datasets in {data_dir}:")
        for name, (seg_path, words_path) in sorted(pairs.items()):
            has_words = "✓" if words_path else "✗"
            logger.info(f"  {name} (words: {has_words})")
        return
    
    logger.info("="*60)
    logger.info("Malaysian STT Stage 2 Data Preparation")
    logger.info("="*60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Audio base directory: {audio_base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Train split: {args.train_split}")
    logger.info(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples:,}")
    if args.datasets:
        logger.info(f"Datasets filter: {args.datasets}")
    logger.info("="*60)
    
    # Load and process all datasets
    logger.info("\n📦 Loading and processing parquet files...")
    df = load_all_datasets(data_dir, datasets=args.datasets)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        logger.info(f"Limited to {len(df):,} samples")
    
    # Filter by duration (only if duration > 0)
    if df['duration'].sum() > 0:
        original_count = len(df)
        df = df[(df['duration'] >= args.min_duration) & (df['duration'] <= args.max_duration)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count:,} samples by duration")
    else:
        logger.warning("Duration is 0 for all samples - skipping duration filter")
    
    # Filter to only existing audio files (checks ALL files)
    if args.filter_existing:
        logger.info("\n🔍 Filtering to only existing audio files...")
        df = filter_existing_audio(df, audio_base_dir)
    # Validate audio files (spot check only)
    elif args.validate_audio:
        logger.info("\n🔍 Validating audio files (spot check)...")
        found, missing = validate_audio_files(df, audio_base_dir)
        logger.info(f"  Found: {found}, Missing: {missing}")
        if missing > 0:
            logger.warning(f"  {missing / (found + missing) * 100:.1f}% of checked files are missing")
    
    # Split train/val
    logger.info(f"\n✂️  Splitting train/val ({args.train_split:.0%}/{1-args.train_split:.0%})...")
    train_df, val_df = split_train_val(df, args.train_split, args.seed)
    
    # Print statistics
    print_statistics(train_df, "Training Set")
    print_statistics(val_df, "Validation Set")
    
    # Save manifests (using fast method)
    logger.info("\n💾 Saving manifests...")
    train_manifest_path = output_dir / "train_manifest.json"
    val_manifest_path = output_dir / "val_manifest.json"
    
    save_manifest_fast(train_df, train_manifest_path, audio_base_dir)
    save_manifest_fast(val_df, val_manifest_path, audio_base_dir)
    
    # Success summary
    logger.info("\n" + "="*60)
    logger.info("✅ Data preparation completed!")
    logger.info("="*60)
    logger.info(f"Train manifest: {train_manifest_path}")
    logger.info(f"  Samples: {len(train_df):,}")
    logger.info(f"Val manifest: {val_manifest_path}")
    logger.info(f"  Samples: {len(val_df):,}")
    logger.info("="*60)
    
    # Example manifest entry
    if len(train_df) > 0:
        logger.info("\n📄 Example manifest entry:")
        row = train_df.iloc[0]
        example = {
            'audio_filepath': str(audio_base_dir / row['audio_filename']),
            'text': row['text'],
            'duration': row['duration']
        }
        logger.info(json.dumps(example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
