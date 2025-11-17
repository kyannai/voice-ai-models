#!/usr/bin/env python3
"""
Generic HuggingFace Dataset Downloader for ASR Evaluation

Download any HuggingFace dataset with audio and text fields, converting it
to a format compatible with our local dataset loader.

Usage:
    # Download specific split
    python download_hf_dataset.py SEACrowd/asr_malcsc --split test --output seacrowd-malcsc
    
    # Download with custom field names
    python download_hf_dataset.py facebook/voxpopuli --split test --lang en --audio-field audio --text-field normalized_text
    
    # List available datasets
    python download_hf_dataset.py --list-popular
"""

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_prepare_dataset(
    dataset_id: str,
    split: str = "test",
    output_name: str = None,
    audio_field: str = "audio",
    text_field: str = "text",
    id_field: str = "id",
    max_samples: int = None,
    subset: str = None,
    language: str = None
):
    """
    Download HuggingFace dataset and prepare it for local use
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "SEACrowd/asr_malcsc")
        split: Dataset split to download (train/test/validation)
        output_name: Output directory name (defaults to last part of dataset_id)
        audio_field: Field name containing audio data
        text_field: Field name containing transcription text
        id_field: Field name containing sample ID
        max_samples: Maximum number of samples to download
        subset: Dataset subset/config name (if applicable)
        language: Language code (if dataset has multiple languages)
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: uv pip install datasets")
        return False
    
    # Determine output directory name
    if output_name is None:
        output_name = dataset_id.split("/")[-1]
    
    logger.info("="*70)
    logger.info(f"Downloading HuggingFace Dataset: {dataset_id}")
    logger.info("="*70)
    logger.info(f"Split: {split}")
    logger.info(f"Output name: {output_name}")
    if subset:
        logger.info(f"Subset: {subset}")
    if language:
        logger.info(f"Language: {language}")
    logger.info("")
    
    try:
        # Build load_dataset arguments
        load_args = {"path": dataset_id, "split": split}
        if subset:
            load_args["name"] = subset
        if language:
            load_args["language"] = language
        
        # Try loading the dataset
        logger.info("Loading dataset from HuggingFace...")
        dataset = load_dataset(**load_args)
        
        logger.info(f"✓ Successfully loaded {len(dataset)} samples")
        
        # Limit samples if requested
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("\n" + "="*70)
        logger.info("TROUBLESHOOTING:")
        logger.info("="*70)
        logger.info("1. Dataset may use loading scripts (deprecated)")
        logger.info("2. Try: uv pip install 'datasets==2.14.0' (older version)")
        logger.info("3. Or download manually from:")
        logger.info(f"   https://huggingface.co/datasets/{dataset_id}")
        logger.info("="*70)
        return False
    
    # Create output directory
    output_dir = Path(f"test_data/{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nSaving dataset to: {output_dir}")
    logger.info(f"Audio field: {audio_field}")
    logger.info(f"Text field: {text_field}")
    logger.info("")
    
    # Convert dataset to our format
    samples = []
    skipped = 0
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # Extract audio and text
            audio_data = item.get(audio_field, {})
            text = item.get(text_field, "")
            
            if not text:
                logger.debug(f"Sample {idx}: No text found, skipping")
                skipped += 1
                continue
            
            # Save audio file
            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            
            # Handle different audio formats
            if isinstance(audio_data, dict):
                if "array" in audio_data and "sampling_rate" in audio_data:
                    sf.write(
                        str(audio_path),
                        audio_data["array"],
                        audio_data["sampling_rate"]
                    )
                elif "path" in audio_data:
                    # Audio is already a file, copy it
                    import shutil
                    shutil.copy(audio_data["path"], audio_path)
                else:
                    logger.debug(f"Sample {idx}: Unknown audio format, skipping")
                    skipped += 1
                    continue
            else:
                logger.debug(f"Sample {idx}: Audio is not a dict, skipping")
                skipped += 1
                continue
            
            # Create sample entry
            sample = {
                "audio_path": f"audio/{audio_filename}",
                "reference": text,
            }
            
            # Add ID if available
            if id_field and id_field in item:
                sample["id"] = item[id_field]
            else:
                sample["id"] = f"sample_{idx:06d}"
            
            samples.append(sample)
            
        except Exception as e:
            logger.debug(f"Error processing sample {idx}: {e}")
            skipped += 1
            continue
    
    if not samples:
        logger.error("No samples were successfully processed!")
        return False
    
    # Save JSON manifest
    json_file = output_dir / f"{split}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Saved {len(samples)} samples to: {json_file}")
    logger.info(f"Audio files saved to: {audio_dir}")
    if skipped > 0:
        logger.info(f"Skipped {skipped} samples (missing data or errors)")
    logger.info("")
    logger.info("="*70)
    logger.info("NEXT STEPS:")
    logger.info("="*70)
    logger.info("Add this configuration to datasets_config.py:")
    logger.info("")
    
    # Generate config
    config_name = output_name.replace("-", "_").replace(".", "_")
    logger.info(f'''    "{config_name}": {{
        "type": "local",
        "description": "{dataset_id} dataset (locally downloaded)",
        "test_data": EVAL_DIR / "test_data/{output_name}/{split}.json",
        "audio_dir": EVAL_DIR / "test_data/{output_name}",
        "language": "ms",  # Change as needed
        "num_samples": None,
    }},''')
    
    logger.info("")
    logger.info("Then use it:")
    logger.info(f'  python evaluate.py --model <model> --test-dataset {config_name}')
    logger.info("="*70)
    
    return True


def list_popular_datasets():
    """List some popular ASR datasets on HuggingFace"""
    datasets = [
        ("mozilla-foundation/common_voice_11_0", "Common Voice 11.0 - Multilingual"),
        ("google/fleurs", "FLEURS - Multilingual ASR"),
        ("facebook/voxpopuli", "VoxPopuli - European Parliament"),
        ("SEACrowd/asr_malcsc", "MALCSC - Malay Code-Switching"),
        ("openslr/librispeech_asr", "LibriSpeech - English"),
        ("facebook/multilingual_librispeech", "MLS - Multilingual"),
    ]
    
    print("\n" + "="*70)
    print("POPULAR ASR DATASETS ON HUGGINGFACE")
    print("="*70)
    for dataset_id, description in datasets:
        print(f"\n{dataset_id}")
        print(f"  {description}")
        print(f"  https://huggingface.co/datasets/{dataset_id}")
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for ASR evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download SEACrowd dataset
  python download_hf_dataset.py SEACrowd/asr_malcsc --split test
  
  # Download Common Voice with subset
  python download_hf_dataset.py mozilla-foundation/common_voice_11_0 --split test --subset ms --output common_voice_ms
  
  # Download with custom field names
  python download_hf_dataset.py facebook/voxpopuli --split test --audio-field audio --text-field normalized_text
  
  # Limit samples for testing
  python download_hf_dataset.py SEACrowd/asr_malcsc --split test --max-samples 100
"""
    )
    
    parser.add_argument(
        "dataset_id",
        nargs="?",
        help="HuggingFace dataset ID (e.g., SEACrowd/asr_malcsc)"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to download (default: test)"
    )
    parser.add_argument(
        "--output",
        help="Output directory name (default: inferred from dataset_id)"
    )
    parser.add_argument(
        "--audio-field",
        default="audio",
        help="Field name containing audio data (default: audio)"
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing transcription (default: text)"
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Field name containing sample ID (default: id)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to download"
    )
    parser.add_argument(
        "--subset",
        help="Dataset subset/config name (if applicable)"
    )
    parser.add_argument(
        "--language",
        help="Language code (if dataset has multiple languages)"
    )
    parser.add_argument(
        "--list-popular",
        action="store_true",
        help="List popular ASR datasets"
    )
    
    args = parser.parse_args()
    
    if args.list_popular:
        list_popular_datasets()
        return
    
    if not args.dataset_id:
        parser.error("dataset_id is required (or use --list-popular)")
    
    success = download_and_prepare_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        output_name=args.output,
        audio_field=args.audio_field,
        text_field=args.text_field,
        id_field=args.id_field,
        max_samples=args.max_samples,
        subset=args.subset,
        language=args.language
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

