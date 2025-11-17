#!/usr/bin/env python3
"""
Convert SEACrowd MALCSC dataset to standard ASR evaluation format

The dataset has:
- WAV/ folder with audio files
- TXT/ folder with transcription files (format: [timestamp] speaker_id gender transcript)

We need to extract full transcriptions from each TXT file and create a JSON file.
"""

import json
from pathlib import Path
import re

def extract_transcription(txt_file: Path) -> str:
    """
    Extract full transcription from a TXT file.
    Each line has format: [start,end] speaker_id gender transcript
    We concatenate all transcripts, removing annotations like [UNK], [LAUGHTER], etc.
    """
    transcripts = []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse line: [timestamp] speaker_id gender,location transcript
            # Example: [0.730,2.599]	G0369	female,Malaysia	hai Syakirah
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            
            transcript = parts[3].strip()
            
            # Skip system sounds and non-speech
            if transcript in ['[SYSTEM]', '[*]', '[ENS]', '[SONANT]', '[MUSIC]', '[LAUGHTER]']:
                continue
            
            # Clean up annotations
            # Remove [UNK] markers
            transcript = transcript.replace('[UNK]', '')
            # Remove other annotations
            transcript = re.sub(r'\[.*?\]', '', transcript)
            
            # Clean up extra spaces
            transcript = ' '.join(transcript.split())
            
            if transcript:
                transcripts.append(transcript)
    
    # Join all transcripts with space
    full_transcript = ' '.join(transcripts)
    return full_transcript


def main():
    # Paths
    base_dir = Path(__file__).parent
    txt_dir = base_dir / "TXT"
    wav_dir = base_dir / "WAV"
    
    # Output JSON file
    output_json = base_dir / "seacrowd_malcsc.json"
    
    # Get all TXT files
    txt_files = sorted(txt_dir.glob("*.txt"))
    
    print(f"Found {len(txt_files)} TXT files")
    
    # Process each file
    samples = []
    for txt_file in txt_files:
        # Get corresponding WAV file
        wav_file = wav_dir / txt_file.with_suffix('.wav').name
        
        if not wav_file.exists():
            print(f"Warning: WAV file not found for {txt_file.name}")
            continue
        
        # Extract transcription
        transcript = extract_transcription(txt_file)
        
        if not transcript:
            print(f"Warning: Empty transcript for {txt_file.name}")
            continue
        
        # Create sample entry
        sample = {
            "audio_path": f"WAV/{wav_file.name}",
            "reference": transcript,
            "file_id": txt_file.stem
        }
        
        samples.append(sample)
        print(f"Processed: {txt_file.stem} -> {len(transcript)} chars")
    
    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated {output_json}")
    print(f"Total samples: {len(samples)}")
    
    # Print first sample as example
    if samples:
        print(f"\nExample (first sample):")
        print(f"  File: {samples[0]['file_id']}")
        print(f"  Audio: {samples[0]['audio_path']}")
        print(f"  Reference: {samples[0]['reference'][:100]}...")


if __name__ == "__main__":
    main()

