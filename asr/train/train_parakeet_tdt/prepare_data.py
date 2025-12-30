#!/usr/bin/env python3
"""
Data preparation script for Parakeet TDT training
Converts training data to NeMo manifest format and validates samples

NeMo manifest format (JSONL):
Each line is a JSON object with:
{
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "transcription text",
    "duration": 2.5
}
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import librosa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_input_data(input_file: Path) -> List[Dict]:
    """Load input data from JSON or CSV"""
    logger.info(f"Loading data from {input_file}")
    
    if input_file.suffix == '.json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle synthesized.json format with 'results' wrapper
        if isinstance(data, dict) and 'results' in data:
            logger.info("Detected synthesized.json format with 'results' wrapper")
            if 'metadata' in data:
                logger.info(f"Metadata: {data['metadata']}")
            data = data['results']
        
    elif input_file.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(input_file)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def validate_and_convert_sample(
    sample: Dict,
    audio_base_dir: Path,
    require_duration: bool = True
) -> Dict:
    """
    Validate a single sample and convert to NeMo manifest format
    
    Input format (flexible):
    - audio_path or audio_filepath: path to audio file
    - text or transcription or reference: transcription text
    - duration (optional): audio duration in seconds
    
    Output format (NeMo manifest):
    - audio_filepath: absolute path to audio file
    - text: transcription text
    - duration: audio duration in seconds
    """
    # Get audio path
    audio_path = sample.get('audio_path') or sample.get('audio_filepath')
    if not audio_path:
        raise ValueError(f"Sample missing audio path: {sample}")
    
    # Resolve to absolute path
    audio_path = Path(audio_path)
    if not audio_path.is_absolute():
        audio_path = audio_base_dir / audio_path
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get transcription text
    text = sample.get('text') or sample.get('transcription') or sample.get('reference')
    if not text:
        raise ValueError(f"Sample missing transcription: {sample}")
    
    text = str(text).strip()
    if not text:
        raise ValueError(f"Empty transcription for {audio_path}")
    
    # Get or compute duration
    duration = sample.get('duration') or sample.get('audio_duration')
    if duration is None:
        # Load audio to get duration
        audio_array, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio_array) / sr
    elif require_duration and duration == 0:
        # Recompute duration if it's 0 and required
        audio_array, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio_array) / sr
    
    # Return NeMo manifest format
    return {
        'audio_filepath': str(audio_path.absolute()),
        'text': text,
        'duration': float(duration)
    }


def save_manifest(samples: List[Dict], output_file: Path):
    """Save samples to NeMo manifest format (JSONL)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(samples)} samples to {output_file}")


def print_statistics(samples: List[Dict], name: str):
    """Print dataset statistics"""
    durations = [s['duration'] for s in samples]
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    
    # Text length statistics
    text_lengths = [len(s['text']) for s in samples]
    avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Total Duration: {total_duration / 3600:.2f} hours")
    logger.info(f"  Avg Duration: {avg_duration:.2f}s")
    logger.info(f"  Min Duration: {min_duration:.2f}s")
    logger.info(f"  Max Duration: {max_duration:.2f}s")
    logger.info(f"  Avg Text Length: {avg_text_len:.1f} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Parakeet TDT training (convert to NeMo manifest)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSON/CSV"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data JSON/CSV"
    )
    parser.add_argument(
        "--audio-base-dir",
        type=str,
        help="Base directory for audio files (if paths are relative)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (filter longer samples)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds (filter shorter samples)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to prepare (default: all samples)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    train_data_path = Path(args.train_data)
    val_data_path = Path(args.val_data)
    audio_base_dir = Path(args.audio_base_dir) if args.audio_base_dir else Path.cwd()
    output_dir = Path(args.output_dir)
    
    logger.info("="*70)
    logger.info("Parakeet TDT Data Preparation")
    logger.info("="*70)
    logger.info(f"Train data: {train_data_path}")
    logger.info(f"Val data: {val_data_path}")
    logger.info(f"Audio base dir: {audio_base_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples:,}")
    else:
        logger.info(f"Max samples: All")
    logger.info("="*70)
    
    # Process training data
    logger.info("\n📦 Processing training data...")
    train_data = load_input_data(train_data_path)
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(train_data):
        logger.info(f"Limiting training data from {len(train_data):,} to {args.max_samples:,} samples")
        train_data = train_data[:args.max_samples]
    
    train_samples = []
    train_errors = []
    
    for idx, sample in enumerate(tqdm(train_data, desc="Validating train samples")):
        try:
            manifest_sample = validate_and_convert_sample(sample, audio_base_dir)
            
            # Filter by duration
            if args.min_duration <= manifest_sample['duration'] <= args.max_duration:
                train_samples.append(manifest_sample)
            else:
                logger.debug(f"Filtered out sample {idx} (duration: {manifest_sample['duration']:.2f}s)")
        
        except Exception as e:
            train_errors.append((idx, str(e)))
            logger.warning(f"Error processing train sample {idx}: {e}")
    
    # Process validation data
    logger.info("\n📦 Processing validation data...")
    val_data = load_input_data(val_data_path)
    
    # Limit validation samples proportionally (10% of training samples)
    if args.max_samples:
        max_val_samples = int(args.max_samples * 0.1)  # 10% of training
        if max_val_samples < len(val_data):
            logger.info(f"Limiting validation data from {len(val_data):,} to {max_val_samples:,} samples")
            val_data = val_data[:max_val_samples]
    
    val_samples = []
    val_errors = []
    
    for idx, sample in enumerate(tqdm(val_data, desc="Validating val samples")):
        try:
            manifest_sample = validate_and_convert_sample(sample, audio_base_dir)
            
            # Filter by duration
            if args.min_duration <= manifest_sample['duration'] <= args.max_duration:
                val_samples.append(manifest_sample)
            else:
                logger.debug(f"Filtered out sample {idx} (duration: {manifest_sample['duration']:.2f}s)")
        
        except Exception as e:
            val_errors.append((idx, str(e)))
            logger.warning(f"Error processing val sample {idx}: {e}")
    
    # Save manifests
    logger.info("\n💾 Saving manifests...")
    train_manifest_path = output_dir / 'train_manifest.json'
    val_manifest_path = output_dir / 'val_manifest.json'
    
    save_manifest(train_samples, train_manifest_path)
    save_manifest(val_samples, val_manifest_path)
    
    # Print statistics
    print_statistics(train_samples, "Training Set")
    print_statistics(val_samples, "Validation Set")
    
    # Print errors summary
    if train_errors or val_errors:
        logger.warning(f"\n⚠️  Errors Summary:")
        logger.warning(f"  Training errors: {len(train_errors)}/{len(train_data)}")
        logger.warning(f"  Validation errors: {len(val_errors)}/{len(val_data)}")
        
        # Save error log
        error_log_path = output_dir / 'preparation_errors.json'
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'train_errors': train_errors,
                'val_errors': val_errors
            }, f, indent=2, ensure_ascii=False)
        logger.warning(f"  Error details saved to: {error_log_path}")
    
    # Success summary
    logger.info("\n" + "="*70)
    logger.info("✅ Data preparation completed successfully!")
    logger.info("="*70)
    logger.info(f"Train manifest: {train_manifest_path}")
    logger.info(f"Val manifest: {val_manifest_path}")
    logger.info(f"\nValid samples:")
    logger.info(f"  Training: {len(train_samples)}/{len(train_data)}")
    logger.info(f"  Validation: {len(val_samples)}/{len(val_data)}")
    logger.info("="*70)
    
    # Example manifest entry
    if train_samples:
        logger.info("\n📄 Example manifest entry:")
        logger.info(json.dumps(train_samples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

