#!/usr/bin/env python3
"""
Data preparation script for Malaysian STT dataset.
Converts parquet files with Whisper-style timestamps to NeMo manifest format.

Optimized for fast processing of millions of samples using vectorized operations.

Parquet format:
- audio_filename: relative path to audio file (e.g., "prepared-pseudolabel-chunks/0-0.mp3")
- segment_timestamp: Whisper format with text and timestamps
  e.g., "<|startoftranscript|><|ms|><|transcribe|><|0.00|> text here<|1.42|><|endoftext|>"

NeMo manifest format (JSONL):
{"audio_filepath": "/abs/path/audio.mp3", "text": "transcription", "duration": 2.78}

Text Normalization:
- Optionally converts numbers, currency, etc. to spoken words
- Supports English (num2words), Malay (custom), Chinese (cn2an)
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

import pandas as pd
import pyarrow.parquet as pq

# Add common directory to path for text_normalizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "common"))
try:
    from text_normalizer import (
        normalize_text, 
        normalize_text_batch, 
        detect_language,
        preprocess_text_for_asr,
        preprocess_text_batch_for_asr,
        is_valid_asr_text,
        get_invalid_characters,
        contains_non_latin_script,
        remove_emoji,
    )
    TEXT_NORMALIZER_AVAILABLE = True
except ImportError:
    TEXT_NORMALIZER_AVAILABLE = False
    is_valid_asr_text = None
    get_invalid_characters = None
    contains_non_latin_script = None
    remove_emoji = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _apply_func_to_batch(args):
    """Module-level function for multiprocessing. Applies func to a batch of items."""
    idx, batch, func = args
    return idx, [func(item) for item in batch]


def parallel_apply(series: pd.Series, func, num_workers: int = 32, desc: str = "Processing") -> pd.Series:
    """
    Apply a function to a pandas Series using multiprocessing.
    Much faster than .apply() for large datasets.
    
    Note: func must be a module-level function (picklable).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    data = series.tolist()
    n = len(data)
    
    if n < 10000:
        # For small datasets, just use regular apply
        return series.apply(func)
    
    batch_size = max(1000, n // (num_workers * 4))
    batches = [(i, data[i:i+batch_size], func) for i in range(0, n, batch_size)]
    
    results = [None] * len(batches)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_apply_func_to_batch, batch): batch[0] for batch in batches}
        
        with tqdm(
            total=n, 
            desc=desc, 
            disable=n < 50000,
            unit="samples",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for future in as_completed(futures):
                idx, batch_results = future.result()
                batch_idx = idx // batch_size
                results[batch_idx] = batch_results
                pbar.update(len(batch_results))
    
    # Flatten results
    flat_results = []
    for batch_result in results:
        flat_results.extend(batch_result)
    
    return pd.Series(flat_results, index=series.index)


def extract_language_vectorized(series: pd.Series) -> pd.Series:
    """
    Extract language code from Whisper-style timestamps.
    Format: <|startoftranscript|><|ms|><|transcribe|>...
    Returns: Series of language codes ('ms', 'en', 'zh', 'ta', etc.)
    """
    # Extract the language tag (2-letter code after startoftranscript)
    lang = series.str.extract(r'<\|([a-z]{2})\|>', expand=False)
    return lang.fillna('unknown')


def extract_text_vectorized(series: pd.Series) -> pd.Series:
    """
    Extract text from Whisper-style timestamps using vectorized operations.
    Removes all <|...|> markers.
    """
    # Remove all <|...|> markers
    text = series.str.replace(r'<\|[^|]*\|>', '', regex=True)
    # Normalize whitespace
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    return text


def extract_duration_vectorized(series: pd.Series) -> pd.Series:
    """
    Extract duration (last numeric timestamp) from Whisper-style timestamps.
    Uses vectorized regex extraction.
    """
    # Extract the last timestamp using regex - find all and take last
    # Pattern: look for the last occurrence of <|X.XX|> before <|endoftext|>
    # Optimized: extract all timestamps, then take the last one
    all_timestamps = series.str.extractall(r'<\|(\d+\.\d+)\|>')
    
    if len(all_timestamps) == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    
    # Get the last timestamp for each row
    last_timestamps = all_timestamps.groupby(level=0).last()[0]
    
    # Convert to float and fill missing with 0
    result = last_timestamps.astype(float).reindex(series.index, fill_value=0.0)
    
    return result


def load_and_process_parquet(pq_file: Path, source_name: str) -> pd.DataFrame:
    """
    Load a parquet file and process it using vectorized operations.
    
    Returns DataFrame with columns: audio_filename, text, duration, source, language
    """
    logger.info(f"Loading {pq_file.name}...")
    
    # Read parquet
    table = pq.read_table(pq_file)
    df = table.to_pandas()
    
    logger.info(f"  Rows: {len(df):,}")
    
    # Filter out rows with missing data
    df = df.dropna(subset=['audio_filename', 'segment_timestamp'])
    df = df[df['audio_filename'].str.len() > 0]
    df = df[df['segment_timestamp'].str.len() > 0]
    
    logger.info(f"  After filtering nulls: {len(df):,}")
    
    # Extract text, duration, and language using vectorized operations
    logger.info(f"  Extracting text...")
    df['text'] = extract_text_vectorized(df['segment_timestamp'])
    
    logger.info(f"  Extracting durations...")
    df['duration'] = extract_duration_vectorized(df['segment_timestamp'])
    
    logger.info(f"  Extracting language tags...")
    df['language'] = extract_language_vectorized(df['segment_timestamp'])
    
    # Filter empty texts
    df = df[df['text'].str.len() > 0]
    
    # Add source
    df['source'] = source_name
    
    # Select only needed columns
    df = df[['audio_filename', 'text', 'duration', 'source', 'language']].copy()
    
    logger.info(f"  Final samples: {len(df):,}")
    
    return df


def load_all_parquet_files(data_dir: Path, datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load all parquet files from the data directory.
    
    Args:
        data_dir: Directory containing parquet files
        datasets: Optional list of dataset names to include (e.g., ['malaysian_context_v2', 'extra'])
    
    Returns concatenated DataFrame with all samples.
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Filter by dataset names if specified
    if datasets:
        filtered_files = []
        for pq_file in parquet_files:
            source_name = pq_file.stem.split('-00000')[0]
            if source_name in datasets:
                filtered_files.append(pq_file)
        
        if not filtered_files:
            raise ValueError(f"No parquet files match datasets: {datasets}")
        
        logger.info(f"Filtering to {len(filtered_files)} datasets: {datasets}")
        parquet_files = filtered_files
    
    dfs = []
    for pq_file in parquet_files:
        source_name = pq_file.stem.split('-00000')[0]
        df = load_and_process_parquet(pq_file, source_name)
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


# Module-level function for multiprocessing (must be picklable)
def _check_audio_batch(args):
    """Check a batch of files. Returns list of (index, exists) tuples."""
    start_idx, batch, base_dir = args
    results = []
    for i, filename in enumerate(batch):
        path = os.path.join(base_dir, filename)
        results.append((start_idx + i, os.path.exists(path)))
    return results


def _process_asr_batch(batch_args):
    """Process a batch of texts for ASR (module-level for multiprocessing)."""
    batch, norm_lang, remove_punct, clean_asr = batch_args
    return preprocess_text_batch_for_asr(
        batch,
        language=norm_lang,
        lowercase=True,
        unicode_normalize=True,
        normalize_chars=True,
        expand_abbrevs=True,
        normalize_particles=True,
        normalize_numbers=True,
        remove_punctuation=remove_punct,
        clean_for_asr=clean_asr,
    )


def filter_existing_audio(
    df: pd.DataFrame,
    audio_base_dir: Path,
    num_workers: int = 32
) -> pd.DataFrame:
    """
    Filter DataFrame to only include samples where audio file exists.
    Uses multiprocessing for fast checking of millions of files.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    logger.info(f"Checking {len(df):,} audio files exist...")
    
    audio_base_str = str(audio_base_dir)
    filenames = df['audio_filename'].tolist()
    
    # Split into batches for parallel processing
    batch_size = 10000
    batches = []
    for i in range(0, len(filenames), batch_size):
        batches.append((i, filenames[i:i+batch_size], audio_base_str))
    
    logger.info(f"  Using {num_workers} workers to check {len(batches)} batches...")
    
    exists_mask = [False] * len(filenames)
    found_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_check_audio_batch, batch): batch[0] for batch in batches}
        
        with tqdm(total=len(filenames), desc="Checking audio files") as pbar:
            for future in as_completed(futures):
                results = future.result()
                for idx, exists in results:
                    exists_mask[idx] = exists
                    if exists:
                        found_count += 1
                pbar.update(len(results))
    
    # Apply mask to dataframe
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


def save_manifest(
    df: pd.DataFrame,
    output_path: Path,
    audio_base_dir: Path
):
    """
    Save DataFrame to NeMo manifest format (JSONL).
    Uses batch writing for efficiency.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create absolute paths
    audio_base_str = str(audio_base_dir.absolute())
    
    logger.info(f"Writing {len(df):,} samples to {output_path.name}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {output_path.name}"):
            audio_filepath = os.path.join(audio_base_str, row['audio_filename'])
            manifest_entry = {
                'audio_filepath': audio_filepath,
                'text': row['text'],
                'duration': row['duration']
            }
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(df):,} samples to {output_path}")


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
        description="Prepare Malaysian STT data for NeMo training (optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (from src/ directory)
  python prepare_data.py --data-dir ../data --audio-base-dir .. --output-dir ./output
  
  # Test with small subset
  python prepare_data.py --data-dir ../data --audio-base-dir .. --output-dir ./test_output \\
    --max-samples 10000 --validate-audio
  
  # Full run with custom split
  python prepare_data.py --data-dir ../data --audio-base-dir .. --output-dir ./output \\
    --train-split 0.9
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
        help="Only include specific datasets (by parquet name prefix). "
             "E.g., --datasets malaysian_context_v2 extra"
    )
    parser.add_argument(
        "--filter-existing",
        action="store_true",
        help="Only include samples where audio file exists (checks ALL files, slower but safe)"
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Normalize text (convert numbers, currency to words)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=['en', 'ms', 'zh', 'auto'],
        help="Language for text normalization (default: auto-detect)"
    )
    parser.add_argument(
        "--include-languages",
        type=str,
        nargs='+',
        default=None,
        help="Only include these language codes (e.g., --include-languages en ms)"
    )
    parser.add_argument(
        "--preprocess-asr",
        action="store_true",
        help="Apply full ASR preprocessing: lowercase, unicode normalization, "
             "abbreviation expansion, discourse particle normalization, and number normalization"
    )
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Remove punctuation marks (use with --preprocess-asr)"
    )
    parser.add_argument(
        "--clean-for-asr",
        action="store_true",
        help="Clean text to keep only alphanumeric, space, apostrophe ('), hyphen (-), and @ symbol"
    )
    parser.add_argument(
        "--filter-invalid-chars",
        action="store_true",
        help="Filter out samples with characters other than alphanumeric, space, ', -, @"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir).resolve()
    audio_base_dir = Path(args.audio_base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    logger.info("="*60)
    logger.info("Malaysian STT Data Preparation (Optimized)")
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
    if args.preprocess_asr:
        lang_str = args.language if args.language else "auto-detect from tags"
        logger.info(f"ASR preprocessing: ENABLED (language: {lang_str})")
        logger.info(f"  - Lowercase: yes")
        logger.info(f"  - Unicode NFC: yes")
        logger.info(f"  - Character normalization: yes (smart quotes, dashes)")
        logger.info(f"  - Abbreviation expansion: yes")
        logger.info(f"  - Discourse particle normalization: yes")
        logger.info(f"  - Number normalization: yes")
        logger.info(f"  - Remove punctuation: {'yes' if args.remove_punctuation else 'no'}")
        logger.info(f"  - Clean for ASR (keep only alphanumeric, ', -, @): {'yes' if args.clean_for_asr else 'no'}")
    elif args.normalize_text:
        lang_str = args.language if args.language else "auto-detect"
        logger.info(f"Text normalization: enabled (language: {lang_str})")
    if args.filter_invalid_chars:
        logger.info(f"Filter invalid chars: ENABLED (only keep alphanumeric, space, ', -, @)")
    if args.include_languages:
        logger.info(f"Including only languages: {args.include_languages}")
    logger.info("="*60)
    
    # Load and process all parquet files
    logger.info("\n📦 Loading and processing parquet files...")
    df = load_all_parquet_files(data_dir, datasets=args.datasets)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        logger.info(f"Limited to {len(df):,} samples")
    
    # Filter by duration
    original_count = len(df)
    df = df[(df['duration'] >= args.min_duration) & (df['duration'] <= args.max_duration)]
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count:,} samples by duration")
    
    # Filter by language (include only specified languages)
    if args.include_languages:
        logger.info(f"\n✓ Including only languages: {args.include_languages}")
        original_count = len(df)
        # Show counts for all languages
        for lang in df['language'].unique():
            lang_count = (df['language'] == lang).sum()
            status = "✓" if lang in args.include_languages else "✗"
            logger.info(f"  {status} {lang}: {lang_count:,} samples")
        # Filter to only included languages
        df = df[df['language'].isin(args.include_languages)]
        filtered_count = original_count - len(df)
        logger.info(f"  Removed {filtered_count:,} samples, keeping: {len(df):,}")
    
    # Filter out samples with Chinese/Tamil characters (before preprocessing)
    if args.preprocess_asr and contains_non_latin_script is not None:
        logger.info("\n🔤 Filtering samples with non-Latin scripts (Chinese, Tamil)...")
        original_count = len(df)
        
        # Check for Chinese/Tamil characters (parallel for speed)
        non_latin_mask = parallel_apply(
            df['text'], contains_non_latin_script, 
            desc="  Checking scripts"
        )
        non_latin_count = non_latin_mask.sum()
        
        if non_latin_count > 0:
            # Show some examples
            non_latin_samples = df[non_latin_mask].head(5)
            logger.info(f"  Examples of filtered samples:")
            for _, row in non_latin_samples.iterrows():
                text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                logger.info(f"    '{text_preview}'")
            
            # Filter out
            df = df[~non_latin_mask]
            logger.info(f"  Filtered {non_latin_count:,} samples with Chinese/Tamil characters")
            logger.info(f"  Remaining: {len(df):,} samples")
        else:
            logger.info(f"  No samples with Chinese/Tamil characters found")
    
    # Remove emoji from text (before other preprocessing)
    if args.preprocess_asr and remove_emoji is not None:
        logger.info("\n😀 Removing emoji from text...")
        original_texts = df['text'].copy()
        df['text'] = parallel_apply(df['text'], remove_emoji, desc="  Removing emoji")
        # Count how many were affected
        changed_count = (original_texts != df['text']).sum()
        logger.info(f"  Removed emoji from {changed_count:,} samples")
    
    # Full ASR preprocessing (lowercase, unicode, abbreviations, discourse particles, numbers)
    if args.preprocess_asr:
        if not TEXT_NORMALIZER_AVAILABLE:
            logger.error("Text normalizer not available. Install dependencies:")
            logger.error("  pip install num2words cn2an langid")
            sys.exit(1)
        
        logger.info("\n📝 Applying full ASR preprocessing (parallel)...")
        
        # Map Whisper language codes to normalizer codes
        lang_map = {'ms': 'ms', 'en': 'en', 'zh': 'zh', 'ta': 'en', 'id': 'ms'}
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        num_workers = min(32, os.cpu_count() or 4)
        batch_size = 10000
        
        # Process by language group for efficiency
        for lang_code in df['language'].unique():
            mask = df['language'] == lang_code
            norm_lang = lang_map.get(lang_code, 'en')
            
            texts = df.loc[mask, 'text'].tolist()
            
            # Split into batches
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batches.append((batch, norm_lang, args.remove_punctuation, args.clean_for_asr))
            
            # Process in parallel with progress bar
            processed = [None] * len(batches)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_asr_batch, batch_args): idx 
                          for idx, batch_args in enumerate(batches)}
                
                with tqdm(
                    total=len(texts), 
                    desc=f"  {lang_code} ({norm_lang})",
                    unit="samples",
                    unit_scale=True,
                    dynamic_ncols=True,
                    mininterval=0.5,
                ) as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        processed[idx] = future.result()
                        pbar.update(len(batches[idx][0]))
            
            # Flatten results
            all_processed = []
            for batch_result in processed:
                all_processed.extend(batch_result)
            
            df.loc[mask, 'text'] = all_processed
        
        logger.info(f"  ASR preprocessing complete: {len(df):,} samples")
    
    # Normalize text (convert numbers, currency to words) - legacy option
    elif args.normalize_text:
        if not TEXT_NORMALIZER_AVAILABLE:
            logger.error("Text normalizer not available. Install dependencies:")
            logger.error("  pip install num2words cn2an langid")
            sys.exit(1)
        
        logger.info("\n📝 Normalizing text (converting numbers to words)...")
        
        if args.language and args.language != 'auto':
            # Use specified language for all samples
            lang = args.language
            logger.info(f"  Using fixed language: {lang}")
            
            batch_size = 10000
            normalized_texts = []
            for i in tqdm(range(0, len(df), batch_size), desc="Normalizing"):
                batch = df['text'].iloc[i:i+batch_size].tolist()
                normalized_batch = normalize_text_batch(batch, language=lang)
                normalized_texts.extend(normalized_batch)
            df['text'] = normalized_texts
        else:
            # Use language from Whisper tags (fast, no langid needed)
            logger.info("  Using language from Whisper tags (fast mode)")
            
            # Map Whisper language codes to normalizer codes
            lang_map = {'ms': 'ms', 'en': 'en', 'zh': 'zh', 'ta': 'en', 'id': 'ms'}
            
            # Normalize by language group for efficiency
            for lang_code in df['language'].unique():
                mask = df['language'] == lang_code
                norm_lang = lang_map.get(lang_code, 'en')
                count = mask.sum()
                logger.info(f"    {lang_code} ({norm_lang}): {count:,} samples")
                
                texts = df.loc[mask, 'text'].tolist()
                normalized = normalize_text_batch(texts, language=norm_lang)
                df.loc[mask, 'text'] = normalized
        
        logger.info(f"  Normalized {len(df):,} samples")
    
    # Filter out samples with invalid characters (after preprocessing)
    if args.filter_invalid_chars:
        if is_valid_asr_text is None:
            logger.error("is_valid_asr_text not available. Text normalizer import failed.")
            sys.exit(1)
        
        logger.info("\n🔤 Filtering samples with invalid characters...")
        original_count = len(df)
        
        # Check each text for validity (parallel for speed)
        valid_mask = parallel_apply(
            df['text'], is_valid_asr_text, 
            desc="  Checking chars"
        )
        invalid_df = df[~valid_mask]
        
        # Log some examples of invalid samples
        if len(invalid_df) > 0:
            logger.info(f"  Examples of filtered samples:")
            for _, row in invalid_df.head(5).iterrows():
                invalid_chars = get_invalid_characters(row['text'])
                text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
                logger.info(f"    '{text_preview}' → invalid chars: {invalid_chars}")
        
        df = df[valid_mask]
        filtered_count = original_count - len(df)
        logger.info(f"  Filtered {filtered_count:,} samples with invalid characters")
        logger.info(f"  Remaining: {len(df):,} samples")
    
    # Validate audio files (spot check)
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
