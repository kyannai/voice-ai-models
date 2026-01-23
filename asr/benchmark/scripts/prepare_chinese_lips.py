#!/usr/bin/env python3
"""
Download and prepare Chinese-LiPS dataset for ASR benchmarking.

Dataset: https://huggingface.co/datasets/BAAI/Chinese-LiPS
Language: Mandarin Chinese (zh)
License: CC BY-NC-SA 4.0 (non-commercial use only)

This script downloads the test split and prepares it in the standard TSV format.
Note: This dataset also includes video modalities, but we only use audio for ASR.
"""

import argparse
import csv
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def download_audio_from_hf(repo_id: str, file_path: str, token: str = None) -> bytes | None:
    """Download audio file from HuggingFace repository."""
    try:
        from huggingface_hub import hf_hub_download
        
        # Download to cache and read
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=token,
        )
        with open(local_path, "rb") as f:
            return f.read()
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare Chinese-LiPS dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chinese_lips",
        help="Output directory (default: chinese_lips)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chinese-LiPS dataset (split: {args.split})...")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "BAAI/Chinese-LiPS",
            split=args.split,
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Features: {list(dataset.features.keys())}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(args.max_samples, total_samples)

    print(f"Processing {total_samples} samples...")

    tsv_path = output_dir / "chinese_lips_test.tsv"
    rows = []
    error_count = 0
    
    # Access the underlying arrow table
    arrow_table = dataset.data
    
    # Find the correct column names
    columns = list(dataset.features.keys())
    print(f"Available columns: {columns}")
    
    # Common column name variations for text
    text_col = None
    for col in ["TEXT", "text", "Text", "transcript", "transcription", "sentence"]:
        if col in columns:
            text_col = col
            break
    
    # Common column name variations for audio
    audio_col_name = None
    for col in ["WAV", "wav", "audio", "Audio", "speech"]:
        if col in columns:
            audio_col_name = col
            break
    
    if not text_col:
        print(f"Error: Could not find text column. Available columns: {columns}")
        return 1
    
    if not audio_col_name:
        print(f"Error: Could not find audio column. Available columns: {columns}")
        return 1
    
    print(f"Using text column: {text_col}, audio column: {audio_col_name}")
    
    # Check if audio column contains paths (strings) or bytes
    first_audio = arrow_table[audio_col_name][0].as_py()
    audio_is_path = isinstance(first_audio, str)
    if audio_is_path:
        print(f"Audio column contains file paths. Will download from HuggingFace...")
        print(f"Example path: {first_audio}")
    
    for idx in tqdm(range(total_samples), desc="Processing"):
        try:
            # Get text
            text = str(arrow_table[text_col][idx].as_py()) if arrow_table[text_col][idx].as_py() else None
            
            if not text or not text.strip():
                error_count += 1
                continue
            
            # Get audio from arrow table
            audio_data = arrow_table[audio_col_name][idx].as_py()
            
            if audio_data is None:
                error_count += 1
                if error_count <= 3:
                    tqdm.write(f"Sample {idx}: No audio data")
                continue
            
            # Handle different audio formats
            audio_bytes = None
            
            if isinstance(audio_data, str):
                # It's a file path - download from HuggingFace
                audio_bytes = download_audio_from_hf("BAAI/Chinese-LiPS", audio_data, hf_token)
                if not audio_bytes:
                    error_count += 1
                    if error_count <= 5:
                        tqdm.write(f"Sample {idx}: Could not download {audio_data}")
                    continue
            elif isinstance(audio_data, dict):
                audio_bytes = audio_data.get("bytes")
            elif isinstance(audio_data, bytes):
                audio_bytes = audio_data
            
            if not audio_bytes:
                error_count += 1
                if error_count <= 3:
                    tqdm.write(f"Sample {idx}: No audio bytes. Got: {type(audio_data)}")
                continue
            
            # Decode audio using soundfile
            try:
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                audio_array = audio_array.astype(np.float32)
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    tqdm.write(f"Sample {idx}: Could not decode audio: {e}")
                continue
            
            # Handle stereo -> mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            # Skip very short audio
            if duration < 0.1:
                error_count += 1
                continue
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                except ImportError:
                    pass
            
            # Generate unique ID and save
            utt_id = f"chinese_lips_{idx:06d}"
            audio_filename = f"{utt_id}.wav"
            audio_path_out = audio_dir / audio_filename
            sf.write(str(audio_path_out), audio_array, sample_rate)
            
            rows.append({
                "utterance_id": utt_id,
                "path": f"audio/{audio_filename}",
                "sentence": text.strip(),
                "duration": f"{duration:.3f}",
            })
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                tqdm.write(f"Error at sample {idx}: {e}")

    if error_count > 0:
        print(f"\nTotal errors: {error_count}")

    if not rows:
        print("No samples successfully processed!")
        return 1

    # Write TSV file
    print(f"\nWriting TSV to {tsv_path}...")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "path", "sentence", "duration"],
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    total_duration = sum(float(r["duration"]) for r in rows)
    success_rate = len(rows) / total_samples * 100
    
    print(f"\nDone!")
    print(f"  Samples processed: {len(rows)} / {total_samples} ({success_rate:.1f}%)")
    print(f"  Errors skipped: {error_count}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Output: {tsv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
