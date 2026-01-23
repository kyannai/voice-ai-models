#!/usr/bin/env python3
"""
Download and prepare KeSpeech dataset for ASR benchmarking.

Dataset: https://huggingface.co/datasets/TwinkStart/KeSpeech
Language: Mandarin Chinese (zh)

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
    parser = argparse.ArgumentParser(description="Prepare KeSpeech dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/kespeech",
        help="Output directory (default: test_data/kespeech)"
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

    print(f"Loading KeSpeech dataset (split: {args.split})...")
    
    try:
        from datasets import load_dataset, Audio
        
        # Load dataset with audio feature cast to use soundfile decoder
        dataset = load_dataset(
            "TwinkStart/KeSpeech",
            split=args.split,
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Get the underlying arrow table to access raw audio bytes
        print("Accessing raw audio data from arrow table...")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(args.max_samples, total_samples)

    print(f"Processing {total_samples} samples...")

    tsv_path = output_dir / "kespeech_test.tsv"
    rows = []
    error_count = 0
    
    # Access the underlying data through the arrow format
    # This bypasses the automatic decoding
    arrow_table = dataset.data
    
    for idx in tqdm(range(total_samples), desc="Processing"):
        try:
            # Get text from arrow table
            text = str(arrow_table["Text"][idx].as_py()) if arrow_table["Text"][idx].as_py() else None
            
            if not text or not text.strip():
                error_count += 1
                continue
            
            # Get audio bytes from arrow table
            audio_col = arrow_table["audio"][idx].as_py()
            
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
            dataset_id = str(arrow_table["ID"][idx].as_py()) if arrow_table["ID"][idx].as_py() else f"{idx:06d}"
            utt_id = f"kespeech_{dataset_id}"
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
        
        # Debug: print first sample's audio structure
        print("\nDebug info for first sample:")
        try:
            audio_col = arrow_table["audio"][0].as_py()
            print(f"  Type: {type(audio_col)}")
            if isinstance(audio_col, dict):
                print(f"  Keys: {list(audio_col.keys())}")
                for k, v in audio_col.items():
                    if isinstance(v, bytes):
                        print(f"    {k}: {len(v)} bytes")
                    else:
                        print(f"    {k}: {type(v)} = {repr(v)[:100]}")
            else:
                print(f"  Value: {repr(audio_col)[:200]}")
        except Exception as e:
            print(f"  Error getting debug info: {e}")
        
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
