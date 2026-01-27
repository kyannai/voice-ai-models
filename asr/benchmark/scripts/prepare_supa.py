#!/usr/bin/env python3
"""
Prepare SUPA (Supadata) banking voice assistant testset.

Converts the JSONL metadata format to TSV format expected by the benchmark.

Dataset: BenchmarkPOC_ASR_20251223
Source: RytBank
Languages: Malay (ms), English (en) - mixed code-switching
Domain: Banking voice assistant (DuitNow, payments, FAQ, etc.)
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SUPA benchmark dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="supa",
        help="Output directory name (default: supa)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="supa/BenchmarkPOC_ASR_20251223",
        help="Source directory containing asr_metadata.jsonl and audio/",
    )
    args = parser.parse_args()

    # Script runs from test_data/ directory
    root = Path(".")
    source_dir = root / args.source_dir
    output_dir = root / args.output_dir
    
    metadata_file = source_dir / "asr_metadata.jsonl"
    audio_dir = source_dir / "audio"
    
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        print("\nExpected structure:")
        print("  test_data/supa/BenchmarkPOC_ASR_20251223/")
        print("    ├── asr_metadata.jsonl")
        print("    └── audio/")
        print("        ├── BANK_*.wav")
        print("        └── ...")
        return 1
    
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return 1

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse JSONL and convert to TSV format
    rows = []
    missing_audio = []
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Invalid JSON at line {line_num}: {e}")
                continue
            
            uid = record.get("uid", "")
            conversation = record.get("conversation", [])
            
            if len(conversation) < 2:
                print(f"WARNING: Incomplete conversation at line {line_num}, uid={uid}")
                continue
            
            # Extract audio metadata
            audio_meta = conversation[0].get("metadata", {})
            audio_filename = f"{uid}.wav"
            audio_path = audio_dir / audio_filename
            
            # Calculate duration from start/end times
            start_time = audio_meta.get("start_time", 0)
            end_time = audio_meta.get("end_time", 0)
            duration = end_time - start_time
            
            # Extract transcript
            transcript = conversation[1].get("text", "").strip()
            
            if not transcript:
                print(f"WARNING: Empty transcript at line {line_num}, uid={uid}")
                continue
            
            # Verify audio file exists
            if not audio_path.exists():
                missing_audio.append(str(audio_path))
                continue
            
            # Path relative to TSV file location (output_dir)
            relative_audio_path = f"../{args.source_dir}/audio/{audio_filename}"
            
            rows.append({
                "utterance_id": uid,
                "path": relative_audio_path,
                "sentence": transcript,
                "duration": f"{duration:.2f}",
            })
    
    if missing_audio:
        print(f"WARNING: {len(missing_audio)} audio files not found:")
        for path in missing_audio[:10]:
            print(f"  - {path}")
        if len(missing_audio) > 10:
            print(f"  ... and {len(missing_audio) - 10} more")
    
    if not rows:
        print("ERROR: No valid samples found")
        return 1
    
    # Write TSV file
    tsv_path = output_dir / "supa_test.tsv"
    fieldnames = ["utterance_id", "path", "sentence", "duration"]
    
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"SUPA dataset prepared successfully!")
    print(f"  Location: {output_dir}")
    print(f"  TSV file: {tsv_path}")
    print(f"  Samples: {len(rows)}")
    
    # Count by category based on filename prefix
    categories = {}
    for row in rows:
        uid = row["utterance_id"]
        # Extract category from uid like BANK_PROXY_UNIQUE_* -> PROXY
        parts = uid.split("_")
        if len(parts) >= 2:
            category = parts[1]  # PROXY, PAYMENT, FAQ, etc.
        else:
            category = "OTHER"
        categories[category] = categories.get(category, 0) + 1
    
    print(f"  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
