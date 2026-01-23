#!/usr/bin/env python3
"""
Download and prepare AISHELL-2 dataset for ASR benchmarking.

Dataset: https://www.aishelltech.com/aishell_2
Reference: https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell2
Language: Mandarin Chinese (zh)

Note: AISHELL-2 requires registration and manual download from the official website.
This script processes the downloaded data into the standard TSV format.

Alternative: Use the HuggingFace version if available.
"""

import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm
import soundfile as sf
import librosa


def process_aishell2_from_kaldi(data_dir: Path, output_dir: Path, split: str = "test"):
    """
    Process AISHELL-2 data from Kaldi-style directory structure.
    
    Expected structure:
    data_dir/
    ├── wav/
    │   └── <speaker_id>/
    │       └── *.wav
    └── transcript/
        └── <split>.txt  (or trans.txt)
    """
    wav_dir = data_dir / "wav"
    transcript_dir = data_dir / "transcript"
    
    # Try different transcript file names
    transcript_files = [
        transcript_dir / f"{split}.txt",
        transcript_dir / "trans.txt",
        data_dir / f"{split}.txt",
        data_dir / "trans.txt",
    ]
    
    transcript_file = None
    for tf in transcript_files:
        if tf.exists():
            transcript_file = tf
            break
    
    if not transcript_file:
        print(f"Error: No transcript file found. Tried: {transcript_files}")
        return []
    
    print(f"Using transcript: {transcript_file}")
    
    # Load transcripts
    transcripts = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                transcripts[utt_id] = text
    
    print(f"Loaded {len(transcripts)} transcripts")
    
    # Find and process audio files
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for wav_path in tqdm(list(wav_dir.rglob("*.wav")), desc="Processing audio"):
        utt_id = wav_path.stem
        
        if utt_id not in transcripts:
            continue
        
        try:
            # Load audio to get duration
            audio, sr = librosa.load(str(wav_path), sr=16000)
            duration = len(audio) / sr
            
            # Copy/convert audio
            new_audio_path = audio_dir / f"{utt_id}.wav"
            sf.write(str(new_audio_path), audio, 16000)
            
            rows.append({
                "utterance_id": utt_id,
                "path": f"audio/{utt_id}.wav",
                "sentence": transcripts[utt_id].strip(),
                "duration": f"{duration:.3f}",
            })
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    return rows


def try_huggingface_download(output_dir: Path, max_samples: int = None):
    """
    Try to download AISHELL-2 from HuggingFace if available.
    """
    try:
        from datasets import load_dataset
        
        print("Trying to load AISHELL-2 from HuggingFace...")
        # Try common HuggingFace dataset names
        dataset_names = [
            "speechcolab/aishell2",
            "aishell2",
        ]
        
        dataset = None
        for name in dataset_names:
            try:
                dataset = load_dataset(name, split="test")
                print(f"Found dataset: {name}")
                break
            except Exception:
                continue
        
        if dataset is None:
            return None
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
            try:
                audio = sample["audio"]
                text = sample.get("text", sample.get("sentence", sample.get("transcription", "")))
                
                utt_id = f"aishell2_{idx:06d}"
                audio_array = audio["array"]
                sample_rate = audio["sampling_rate"]
                duration = len(audio_array) / sample_rate
                
                audio_filename = f"{utt_id}.wav"
                audio_path = audio_dir / audio_filename
                sf.write(str(audio_path), audio_array, sample_rate)
                
                rows.append({
                    "utterance_id": utt_id,
                    "path": f"audio/{audio_filename}",
                    "sentence": text.strip(),
                    "duration": f"{duration:.3f}",
                })
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        return rows
        
    except ImportError:
        return None
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare AISHELL-2 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to downloaded AISHELL-2 data (Kaldi format). If not provided, tries HuggingFace."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/aishell2",
        help="Output directory (default: test_data/aishell2)"
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
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = None

    # If data directory provided, use Kaldi-style processing
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            return 1
        rows = process_aishell2_from_kaldi(data_dir, output_dir, args.split)
    else:
        # Try HuggingFace download
        rows = try_huggingface_download(output_dir, args.max_samples)
        
        if rows is None:
            print("\n" + "=" * 70)
            print("AISHELL-2 requires manual download.")
            print("=" * 70)
            print("\nOptions:")
            print("1. Download from official website: https://www.aishelltech.com/aishell_2")
            print("   Then run: python prepare_aishell2.py --data-dir /path/to/aishell2")
            print("\n2. Use the Kaldi recipe to download:")
            print("   https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell2")
            print("\n3. Check HuggingFace for community uploads")
            print("=" * 70)
            return 1

    if not rows:
        print("No data processed.")
        return 1

    # Apply max_samples limit if processing from local data
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[:args.max_samples]

    # Write TSV file
    tsv_path = output_dir / "aishell2_test.tsv"
    print(f"Writing TSV to {tsv_path}...")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "path", "sentence", "duration"],
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    total_duration = sum(float(r["duration"]) for r in rows)
    print(f"\nDone!")
    print(f"  Samples: {len(rows)}")
    print(f"  Duration: {total_duration / 3600:.2f} hours")
    print(f"  Output: {tsv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
