#!/usr/bin/env python3
"""
Prepare synthetic dataset for NeMo training
Loads synthesized.json, splits into train/val, and creates NeMo manifests
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import random
import librosa
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_synthesized_data(input_file: Path) -> List[Dict]:
    """Load synthesized.json with results wrapper"""
    logger.info(f"Loading data from {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle synthesized.json format with 'results' wrapper
    if isinstance(data, dict) and 'results' in data:
        logger.info("Detected synthesized.json format with 'results' wrapper")
        if 'metadata' in data:
            metadata = data['metadata']
            logger.info(f"Dataset metadata:")
            logger.info(f"  Total samples: {metadata.get('total_samples', 'N/A')}")
            logger.info(f"  Total duration: {metadata.get('total_duration_minutes', 0):.2f} minutes")
            logger.info(f"  Average duration: {metadata.get('average_duration_seconds', 0):.2f} seconds")
        data = data['results']
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def convert_to_nemo_manifest(
    sample: Dict,
    audio_base_dir: Path
) -> Dict:
    """
    Convert synthesized sample to NeMo manifest format
    
    Input (synthesized.json):
    {
        "text": "...",
        "audio_path": "audio/audio_000000.mp3",
        "duration": 14.811375,
        ...
    }
    
    Output (NeMo manifest):
    {
        "audio_filepath": "/absolute/path/to/audio.mp3",
        "text": "...",
        "duration": 14.811375
    }
    """
    # Get audio path
    audio_path = sample.get('audio_path') or sample.get('audio_filepath')
    if not audio_path:
        raise ValueError(f"Sample missing audio_path: {sample}")
    
    # Resolve to absolute path
    audio_path = Path(audio_path)
    if not audio_path.is_absolute():
        audio_path = audio_base_dir / audio_path
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get transcription text
    text = sample.get('text', '').strip()
    if not text:
        raise ValueError(f"Sample missing text: {sample}")
    
    # Get duration - compute from audio file if zero or missing
    duration = sample.get('duration', 0)
    if duration == 0 or duration is None:
        # Compute duration from audio file
        try:
            audio_array, sr = librosa.load(str(audio_path), sr=None)
            duration = len(audio_array) / sr
        except Exception as e:
            raise ValueError(f"Could not load audio file to compute duration: {audio_path}. Error: {e}")
    
    # Return NeMo manifest format
    return {
        'audio_filepath': str(audio_path.absolute()),
        'text': text,
        'duration': float(duration)
    }


def split_train_val(samples: List[Dict], train_split: float, seed: int = 42) -> tuple:
    """Split samples into train and validation sets"""
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_split)
    train_samples = shuffled[:split_idx]
    val_samples = shuffled[split_idx:]
    
    logger.info(f"Split into {len(train_samples)} train and {len(val_samples)} val samples")
    return train_samples, val_samples


def save_manifest(samples: List[Dict], output_path: Path):
    """Save manifest in NeMo JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def print_statistics(samples: List[Dict], dataset_name: str):
    """Print dataset statistics"""
    if not samples:
        return
    
    durations = [s['duration'] for s in samples]
    total_duration = sum(durations)
    avg_duration = total_duration / len(samples)
    min_duration = min(durations)
    max_duration = max(durations)
    
    logger.info(f"\n📊 {dataset_name} Statistics:")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Total duration: {total_duration/60:.2f} minutes ({total_duration:.1f} seconds)")
    logger.info(f"  Average duration: {avg_duration:.2f} seconds")
    logger.info(f"  Duration range: {min_duration:.2f}s - {max_duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic dataset for NeMo training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to synthesized.json file"
    )
    parser.add_argument(
        "--audio-base-dir",
        type=str,
        default=".",
        help="Base directory for audio files (default: same as input)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Training split ratio (default: 0.9 = 90%% train, 10%% val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
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
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    audio_base_dir = Path(args.audio_base_dir)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # If audio_base_dir is default, use input file's parent
    if args.audio_base_dir == ".":
        audio_base_dir = input_path.parent
    
    logger.info("="*70)
    logger.info("Synthetic Dataset Preparation for NeMo")
    logger.info("="*70)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Audio base dir: {audio_base_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Train split: {args.train_split:.1%}")
    logger.info(f"Duration filter: {args.min_duration}s - {args.max_duration}s")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📦 Loading synthesized data...")
    samples = load_synthesized_data(input_path)
    
    # Convert to NeMo format
    logger.info("\n🔄 Converting to NeMo manifest format...")
    
    # Check if we need to compute durations
    zero_durations = sum(1 for s in samples if s.get('duration', 0) == 0)
    if zero_durations > 0:
        logger.info(f"⏱️  {zero_durations}/{len(samples)} samples have zero duration - will compute from audio files...")
        logger.info("   (This may take a few minutes)")
    
    nemo_samples = []
    errors = []
    
    for idx, sample in enumerate(tqdm(samples, desc="Converting samples")):
        try:
            nemo_sample = convert_to_nemo_manifest(sample, audio_base_dir)
            
            # Filter by duration
            if args.min_duration <= nemo_sample['duration'] <= args.max_duration:
                nemo_samples.append(nemo_sample)
            else:
                logger.debug(f"Filtered out sample {idx} (duration: {nemo_sample['duration']:.2f}s)")
        
        except Exception as e:
            errors.append((idx, str(e)))
            logger.warning(f"Error processing sample {idx}: {e}")
    
    if errors:
        logger.warning(f"\n⚠️  Skipped {len(errors)}/{len(samples)} samples due to errors")
    
    logger.info(f"✓ Converted {len(nemo_samples)} valid samples")
    
    # Split into train/val
    logger.info("\n✂️  Splitting into train/val sets...")
    train_samples, val_samples = split_train_val(nemo_samples, args.train_split, args.seed)
    
    # Save manifests
    logger.info("\n💾 Saving manifests...")
    train_manifest_path = output_dir / 'synthetic_train_manifest.json'
    val_manifest_path = output_dir / 'synthetic_val_manifest.json'
    
    save_manifest(train_samples, train_manifest_path)
    save_manifest(val_samples, val_manifest_path)
    
    # Print statistics
    print_statistics(train_samples, "Training Set")
    print_statistics(val_samples, "Validation Set")
    
    # Success summary
    logger.info("\n" + "="*70)
    logger.info("✅ Dataset preparation completed successfully!")
    logger.info("="*70)
    logger.info(f"Train manifest: {train_manifest_path}")
    logger.info(f"Val manifest: {val_manifest_path}")
    logger.info(f"\nSamples:")
    logger.info(f"  Training: {len(train_samples)}")
    logger.info(f"  Validation: {len(val_samples)}")
    logger.info(f"  Total: {len(nemo_samples)}")
    logger.info("="*70)
    
    # Example manifest entry
    if train_samples:
        logger.info("\n📄 Example manifest entry:")
        logger.info(json.dumps(train_samples[0], indent=2, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    exit(main())

