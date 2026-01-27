#!/usr/bin/env python3
"""
Prepare Malaysian-STT data for Whisper fine-tuning.

This script wraps the existing prepare_data.py from training_data/malaysian-stt
and generates manifest files in the format expected by the Whisper training script.

The manifest format is NeMo-style JSONL:
{"audio_filepath": "/path/to/audio.mp3", "text": "transcription", "duration": 2.5}

Usage:
    python prepare_data.py --output-dir ./data
    python prepare_data.py --output-dir ./data --max-samples 10000
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Path to Malaysian-STT data preparation script
MALAYSIAN_STT_DIR = Path(__file__).parent.parent.parent / "training_data" / "malaysian-stt"
MALAYSIAN_STT_PREPARE_SCRIPT = MALAYSIAN_STT_DIR / "src" / "prepare_data.py"


def check_malaysian_stt_data(data_dir: Path) -> bool:
    """Check if Malaysian-STT data has been downloaded.
    
    Args:
        data_dir: Directory where Malaysian-STT data should be
        
    Returns:
        True if data exists, False otherwise
    """
    # Check for parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    return len(parquet_files) > 0


def print_download_instructions():
    """Print instructions for downloading Malaysian-STT data."""
    print("\n" + "=" * 70)
    print("Malaysian-STT Data Not Found")
    print("=" * 70)
    print("\nPlease download the dataset first:")
    print("")
    print("  cd ../training_data/malaysian-stt")
    print("  huggingface-cli download --repo-type dataset \\")
    print("    --local-dir './' \\")
    print("    --max-workers 20 \\")
    print("    mesolitica/Malaysian-STT-Whisper")
    print("")
    print("  # Then unzip the audio files:")
    print("  wget https://gist.githubusercontent.com/huseinzol05/...")
    print("  python3 unzip.py")
    print("")
    print("See: training_data/malaysian-stt/README.md for full instructions")
    print("=" * 70 + "\n")


def run_prepare_data(
    output_dir: Path,
    max_samples: Optional[int] = None,
    datasets: Optional[list] = None,
    train_split: float = 0.95,
):
    """Run the Malaysian-STT data preparation script.
    
    Args:
        output_dir: Output directory for manifest files
        max_samples: Maximum samples to process (optional)
        datasets: List of datasets to include (optional)
        train_split: Ratio for train/val split (default 0.95)
    """
    # Paths for Malaysian-STT data
    # The data directory structure after download:
    # training_data/malaysian-stt/
    #   ├── data/  <- parquet files here
    #   └── prepared-pseudolabel-chunks/  <- audio files here
    
    data_dir = MALAYSIAN_STT_DIR / "data" / "data"
    audio_base_dir = MALAYSIAN_STT_DIR / "data"
    
    # Check if data exists
    if not data_dir.exists():
        # Try alternate location
        data_dir = MALAYSIAN_STT_DIR / "data"
    
    if not check_malaysian_stt_data(data_dir):
        print_download_instructions()
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        str(MALAYSIAN_STT_PREPARE_SCRIPT),
        "--data-dir", str(data_dir),
        "--audio-base-dir", str(audio_base_dir),
        "--output-dir", str(output_dir),
        "--train-split", str(train_split),
    ]
    
    if max_samples is not None and max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    
    if datasets:
        cmd.extend(["--datasets"] + datasets)
    
    # Run the script
    print("Running Malaysian-STT data preparation...")
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\nError: Data preparation failed with code {result.returncode}")
        sys.exit(result.returncode)
    
    # Verify output files
    train_manifest = output_dir / "train_manifest.json"
    val_manifest = output_dir / "val_manifest.json"
    
    if not train_manifest.exists() or not val_manifest.exists():
        print("\nError: Manifest files not created")
        sys.exit(1)
    
    # Count samples
    train_count = sum(1 for _ in open(train_manifest, 'r'))
    val_count = sum(1 for _ in open(val_manifest, 'r'))
    
    print("\n" + "=" * 70)
    print("✅ Data Preparation Complete!")
    print("=" * 70)
    print(f"Train manifest: {train_manifest}")
    print(f"  Samples: {train_count:,}")
    print(f"Val manifest: {val_manifest}")
    print(f"  Samples: {val_count:,}")
    print("=" * 70 + "\n")


def create_sample_manifest(output_dir: Path, num_samples: int = 100):
    """Create a small sample manifest for testing.
    
    This is useful when the full dataset is not available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy manifests
    train_manifest = output_dir / "train_manifest.json"
    val_manifest = output_dir / "val_manifest.json"
    
    # Sample data (you can replace with real paths)
    sample_texts = [
        "Selamat pagi semua",
        "Apa khabar hari ini",
        "Terima kasih banyak",
        "Saya nak pergi ke bank",
        "Tolong ambil barang itu",
    ]
    
    print("Creating sample manifests for testing...")
    print("Note: These are placeholder files. Use real data for training.")
    
    with open(train_manifest, 'w') as f:
        for i in range(num_samples):
            entry = {
                "audio_filepath": f"/path/to/audio_{i}.wav",
                "text": sample_texts[i % len(sample_texts)],
                "duration": 2.5
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open(val_manifest, 'w') as f:
        for i in range(num_samples // 10):
            entry = {
                "audio_filepath": f"/path/to/val_audio_{i}.wav",
                "text": sample_texts[i % len(sample_texts)],
                "duration": 2.5
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created: {train_manifest} ({num_samples} samples)")
    print(f"Created: {val_manifest} ({num_samples // 10} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Malaysian-STT data for Whisper fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare full dataset
    python prepare_data.py --output-dir ./data
    
    # Prepare with limited samples (for testing)
    python prepare_data.py --output-dir ./data --max-samples 10000
    
    # Prepare specific datasets only
    python prepare_data.py --output-dir ./data --datasets malaysian_context_v2 extra
    
    # Create sample manifests for testing (no real data needed)
    python prepare_data.py --output-dir ./data --create-sample
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data",
        help="Output directory for manifest files (default: ./data)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (optional)"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=None,
        help="Only include specific datasets (e.g., malaysian_context_v2 extra)"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Ratio for train/val split (default: 0.95)"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample manifests for testing (no real data needed)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        create_sample_manifest(output_dir)
    else:
        run_prepare_data(
            output_dir=output_dir,
            max_samples=args.max_samples,
            datasets=args.datasets,
            train_split=args.train_split,
        )


if __name__ == "__main__":
    main()
