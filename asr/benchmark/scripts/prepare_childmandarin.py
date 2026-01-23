#!/usr/bin/env python3
"""
Download and prepare ChildMandarin dataset for ASR benchmarking.

Dataset: https://huggingface.co/datasets/BAAI/ChildMandarin
Language: Mandarin Chinese (zh) - Children aged 3-5
License: CC BY-NC-SA 4.0 (non-commercial use only)

This script downloads the test split and prepares it in the standard TSV format.
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare ChildMandarin dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="childmandarin",
        help="Output directory (default: childmandarin)"
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

    print(f"Loading ChildMandarin dataset (split: {args.split})...")
    print("Note: This dataset requires accepting terms on HuggingFace for access.")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "BAAI/ChildMandarin",
            split=args.split,
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Features: {list(dataset.features.keys())}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nIf you see an authentication error, make sure you:")
        print("  1. Have accepted the dataset terms at https://huggingface.co/datasets/BAAI/ChildMandarin")
        print("  2. Have set HF_TOKEN environment variable with your HuggingFace token")
        import traceback
        traceback.print_exc()
        return 1

    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(args.max_samples, total_samples)

    print(f"Processing {total_samples} samples...")

    tsv_path = output_dir / "childmandarin_test.tsv"
    rows = []
    error_count = 0
    
    # Access the underlying arrow table
    arrow_table = dataset.data
    
    # Find the correct column names
    columns = list(dataset.features.keys())
    print(f"Available columns: {columns}")
    
    # This dataset uses 'json' for metadata (including text) and 'wav' for audio
    has_json_format = "json" in columns and "wav" in columns
    
    if has_json_format:
        print("Detected webdataset format with 'json' and 'wav' columns")
        
        # Check the structure of first sample's json
        first_json = arrow_table["json"][0].as_py()
        if isinstance(first_json, bytes):
            import json as json_lib
            first_json = json_lib.loads(first_json.decode('utf-8'))
        print(f"JSON structure: {list(first_json.keys()) if isinstance(first_json, dict) else type(first_json)}")
    else:
        # Common column name variations
        text_col = None
        for col in ["text", "Text", "transcript", "transcription", "sentence"]:
            if col in columns:
                text_col = col
                break
        
        audio_col_name = None
        for col in ["audio", "Audio", "wav", "speech"]:
            if col in columns:
                audio_col_name = col
                break
        
        if not text_col or not audio_col_name:
            print(f"Error: Could not find text column ({text_col}) or audio column ({audio_col_name})")
            print(f"Available columns: {columns}")
            return 1
        
        print(f"Using text column: {text_col}, audio column: {audio_col_name}")
    
    for idx in tqdm(range(total_samples), desc="Processing"):
        try:
            if has_json_format:
                # Parse JSON metadata
                json_data = arrow_table["json"][idx].as_py()
                if isinstance(json_data, bytes):
                    import json as json_lib
                    json_data = json_lib.loads(json_data.decode('utf-8'))
                
                # Try different text field names
                text = None
                for key in ["text", "Text", "transcript", "transcription", "sentence", "label"]:
                    if key in json_data:
                        text = str(json_data[key])
                        break
                
                if not text or not text.strip():
                    error_count += 1
                    if error_count <= 3:
                        tqdm.write(f"Sample {idx}: No text in JSON. Keys: {list(json_data.keys())}")
                    continue
                
                # Get audio bytes
                audio_col = arrow_table["wav"][idx].as_py()
            else:
                # Get text
                text = str(arrow_table[text_col][idx].as_py()) if arrow_table[text_col][idx].as_py() else None
                
                if not text or not text.strip():
                    error_count += 1
                    continue
                
                # Get audio bytes from arrow table
                audio_col = arrow_table[audio_col_name][idx].as_py()
            
            if audio_col is None:
                error_count += 1
                if error_count <= 3:
                    tqdm.write(f"Sample {idx}: No audio data")
                continue
            
            # audio_col should be a dict with 'bytes' and 'path'
            audio_bytes = None
            if isinstance(audio_col, dict):
                audio_bytes = audio_col.get("bytes")
            elif isinstance(audio_col, bytes):
                audio_bytes = audio_col
            
            if not audio_bytes:
                error_count += 1
                if error_count <= 3:
                    tqdm.write(f"Sample {idx}: No audio bytes. Got: {type(audio_col)}")
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
            utt_id = f"childmandarin_{idx:06d}"
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
