#!/usr/bin/env python3
"""
Download mesolitica/Malaysian-TTS dataset from HuggingFace.

This dataset has two components:
1. Parquet files with metadata (text, audio_filename)
2. ZIP files containing the actual MP3 audio files

The script downloads both and extracts the audio files.

Dataset: https://huggingface.co/datasets/mesolitica/Malaysian-TTS

Speakers:
- anwar_ibrahim: ~106k samples
- husein: ~127k samples  
- kp_ms: ~160k samples
- shafiqah_idayu: ~142k samples

Usage:
    python download_dataset.py --output-dir data/raw
    python download_dataset.py --output-dir data/raw --max-samples 10000
"""

import argparse
import logging
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_NAME = "mesolitica/Malaysian-TTS"

# All available speaker splits in the dataset
SPEAKER_SPLITS = [
    "anwar_ibrahim",
    "husein", 
    "kp_ms",
    "shafiqah_idayu"
]

# Mapping of audio directory prefixes to ZIP files
# These are the ZIP files that contain the audio for each speaker
AUDIO_ZIP_FILES = {
    # Anwar Ibrahim
    "anwar-berita-bisnes": "anwar-berita-bisnes.zip",
    "anwar-berita-politik": "anwar-berita-politik.zip",
    "anwar-berita-sukan": "anwar-berita-sukan.zip",
    "anwar-chatbot-normalized-v2": "anwar-chatbot-normalized-v2.zip",
    "anwar-chatbot-politics-normalized-v2": "anwar-chatbot-politics-normalized-v2.zip",
    "anwar-gaya-hidup": "anwar-gaya-hidup.zip",
    "anwar-news-politics-normalized-v2": "anwar-news-politics-normalized-v2.zip",
    # Husein
    "generate-husein-news-normalized-v2": "generate-husein-news-normalized-v2.zip",
    "generate-husein-wiki-normalized-v2": "generate-husein-wiki-normalized-v2.zip",
    "husein-chatbot-normalized-v2": "husein-chatbot-normalized-v2.zip",
    "husein-chatbot-politics-normalized-v2": "husein-chatbot-politics-normalized-v2.zip",
    "husein-news-politics-normalized-v2": "husein-news-politics-normalized-v2.zip",
    # KP MS (Malay)
    "kp-berita-bisnes": "kp-berita-bisnes.zip",
    "kp-berita-politik": "kp-berita-politik.zip",
    "kp-berita-sukan": "kp-berita-sukan.zip",
    "kp-chatbot": "kp-chatbot.zip",
    "kp-chatbot-politics": "kp-chatbot-politics.zip",
    "kp-gaya-hidup": "kp-gaya-hidup.zip",
    # Shafiqah Idayu
    "shafiqah-idayu-chatbot-normalized-v2": "shafiqah-idayu-chatbot-normalized-v2.zip",
    "shafiqah-idayu-chatbot-politics-normalized-v2": "shafiqah-idayu-chatbot-politics-normalized-v2.zip",
    "shafiqah-idayu-news-hiburan-normalized-v2": "shafiqah-idayu-news-hiburan-normalized-v2.zip",
    "shafiqah-idayu-news-politics-normalized-v2": "shafiqah-idayu-news-politics-normalized-v2.zip",
}


def get_required_zip_files(dataset, max_samples: int | None = None) -> set[str]:
    """
    Determine which ZIP files are needed based on audio_filename in dataset.
    
    Args:
        dataset: HuggingFace dataset
        max_samples: If set, only check first N samples
        
    Returns:
        Set of ZIP file names needed
    """
    needed_dirs = set()
    
    samples_to_check = dataset
    if max_samples and max_samples < len(dataset):
        samples_to_check = dataset.select(range(max_samples))
    
    for sample in samples_to_check:
        audio_fn = sample.get('audio_filename', '')
        if '/' in audio_fn:
            audio_dir = audio_fn.split('/')[0]
            needed_dirs.add(audio_dir)
    
    # Map directories to ZIP files
    zip_files = set()
    for audio_dir in needed_dirs:
        if audio_dir in AUDIO_ZIP_FILES:
            zip_files.add(AUDIO_ZIP_FILES[audio_dir])
        else:
            logger.warning(f"Unknown audio directory: {audio_dir}")
    
    return zip_files


def download_and_extract_zip(zip_file: str, output_dir: Path) -> bool:
    """
    Download a ZIP file from the dataset repo and extract it.
    
    Args:
        zip_file: Name of the ZIP file in the repo
        output_dir: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    extract_dir_name = zip_file.replace('.zip', '')
    if (audio_dir / extract_dir_name).exists():
        logger.info(f"  Already extracted: {extract_dir_name}")
        return True
    
    try:
        logger.info(f"  Downloading: {zip_file}")
        zip_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=zip_file,
            repo_type="dataset",
            local_dir=output_dir / "downloads"
        )
        
        logger.info(f"  Extracting: {zip_file}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(audio_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"  Error with {zip_file}: {e}")
        return False


def download_dataset(
    output_dir: Path,
    max_samples: int | None = None,
    speakers: list[str] | None = None,
    skip_audio: bool = False
):
    """
    Download the Malaysian-TTS dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum total samples to download (distributed across speakers)
        speakers: List of speaker splits to download (default: all)
        skip_audio: If True, only download metadata (for testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if speakers is None:
        speakers = SPEAKER_SPLITS
    
    logger.info(f"{'='*70}")
    logger.info(f"Malaysian-TTS Dataset Download")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Speakers: {speakers}")
    if max_samples:
        logger.info(f"Max samples: {max_samples:,}")
    if skip_audio:
        logger.info(f"Skipping audio download (metadata only)")
    
    # Calculate samples per speaker if max_samples is set
    samples_per_speaker = None
    if max_samples:
        samples_per_speaker = max_samples // len(speakers)
        logger.info(f"Samples per speaker: ~{samples_per_speaker:,}")
    
    all_needed_zips = set()
    total_downloaded = 0
    
    for speaker in speakers:
        logger.info(f"\n{'-'*70}")
        logger.info(f"Speaker: {speaker}")
        logger.info(f"{'-'*70}")
        
        try:
            # Load the dataset split
            logger.info(f"  Loading metadata...")
            dataset = load_dataset(
                DATASET_NAME,
                split=speaker,
                trust_remote_code=True
            )
            
            logger.info(f"  Total samples available: {len(dataset):,}")
            
            # Limit samples if specified
            if samples_per_speaker and len(dataset) > samples_per_speaker:
                dataset = dataset.select(range(samples_per_speaker))
                logger.info(f"  Limited to: {len(dataset):,} samples")
            
            # Save metadata to disk
            speaker_dir = output_dir / "metadata" / speaker
            speaker_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(speaker_dir))
            logger.info(f"  Saved metadata to: {speaker_dir}")
            
            # Collect required ZIP files
            if not skip_audio:
                needed_zips = get_required_zip_files(dataset, samples_per_speaker)
                logger.info(f"  Required audio ZIPs: {len(needed_zips)}")
                all_needed_zips.update(needed_zips)
            
            total_downloaded += len(dataset)
            
        except Exception as e:
            logger.error(f"  Error downloading {speaker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Download and extract audio ZIP files
    if not skip_audio and all_needed_zips:
        logger.info(f"\n{'-'*70}")
        logger.info(f"Downloading Audio Files")
        logger.info(f"{'-'*70}")
        logger.info(f"Total ZIP files to download: {len(all_needed_zips)}")
        
        for zip_file in sorted(all_needed_zips):
            download_and_extract_zip(zip_file, output_dir)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Download Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Total samples: {total_downloaded:,}")
    logger.info(f"Metadata: {output_dir / 'metadata'}")
    if not skip_audio:
        logger.info(f"Audio: {output_dir / 'audio'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download mesolitica/Malaysian-TTS dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download full dataset
    python download_dataset.py --output-dir data/raw
    
    # Download small subset for testing
    python download_dataset.py --output-dir data/raw --max-samples 10000
    
    # Download specific speakers only
    python download_dataset.py --output-dir data/raw --speakers husein shafiqah_idayu
    
    # Download metadata only (skip audio for testing)
    python download_dataset.py --output-dir data/raw --max-samples 1000 --skip-audio
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help=f"HuggingFace dataset name (default: {DATASET_NAME})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the downloaded dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to download (distributed across speakers)"
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=None,
        choices=SPEAKER_SPLITS,
        help=f"Specific speakers to download (default: all). Options: {SPEAKER_SPLITS}"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip downloading audio files (metadata only, for testing)"
    )
    
    args = parser.parse_args()
    
    download_dataset(
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        speakers=args.speakers,
        skip_audio=args.skip_audio
    )


if __name__ == "__main__":
    main()
