#!/usr/bin/env python3
"""
Convert Malaysian ASR dataset to LLaMA-Factory format for Qwen2.5-Omni
"""

import json
import os
from pathlib import Path
from typing import List, Dict

def convert_to_llamafactory_format(
    input_json: str,
    audio_base_dir: str,
    output_json: str
):
    """
    Convert ASR dataset to LLaMA-Factory format
    
    Input format:
    [{"audio_path": "path/to/audio.wav", "text": "transcription"}]
    
    Output format (LLaMA-Factory multimodal):
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<audio>\nTranscribe this Malay audio accurately."
                },
                {
                    "role": "assistant",
                    "content": "transcription"
                }
            ],
            "audios": ["absolute/path/to/audio.wav"]
        }
    ]
    
    Note: Audio paths are provided separately in the "audios" field,
    and the <audio> token is used as a placeholder in the content.
    """
    
    print(f"Loading data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Convert to LLaMA-Factory format
    converted_data = []
    audio_base = Path(audio_base_dir).resolve()  # Ensure absolute path
    
    for idx, item in enumerate(data):
        # Get audio path (make absolute)
        audio_path = item.get('audio_path', '')
        if not os.path.isabs(audio_path):
            audio_path = str(audio_base / audio_path)
        
        # Convert to absolute path (resolve any relative components)
        audio_path = str(Path(audio_path).resolve())
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        transcription = item.get('text', '').strip()
        if not transcription:
            print(f"Warning: Empty transcription at index {idx}")
            continue
        
        # Create conversation format (LLaMA-Factory expects audio paths separately)
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": "<audio>\nTranscribe this Malay audio to text:"
                },
                {
                    "role": "assistant",
                    "content": transcription
                }
            ],
            "audios": [audio_path]  # Audio paths provided separately
        }
        
        converted_data.append(conversation)
    
    # Save converted data
    print(f"Converted {len(converted_data)} samples")
    print(f"Saving to {output_json}...")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Data conversion complete!")
    return len(converted_data)


def main():
    # Paths (adjust these to your setup)
    # Use absolute paths to avoid issues when running from different directories
    script_dir = Path(__file__).parent.absolute()
    
    train_json = script_dir / "../training_data/malaysian-stt/malaysian_context_v2-00000-of-00001_train.json"
    val_json = script_dir / "../training_data/malaysian-stt/malaysian_context_v2-00000-of-00001_val.json"
    audio_base_dir = script_dir / "../training_data/malaysian-stt/"
    
    # Convert to absolute paths
    train_json = str(train_json.resolve())
    val_json = str(val_json.resolve())
    audio_base_dir = str(audio_base_dir.resolve())
    
    # Output directory
    output_dir = str((script_dir / "LLaMA-Factory/data").resolve())
    
    print("========================================")
    print("Converting Malaysian ASR Dataset")
    print("========================================\n")
    
    # Convert training data
    if os.path.exists(train_json):
        train_count = convert_to_llamafactory_format(
            train_json,
            audio_base_dir,
            f"{output_dir}/malaysian_asr_train.json"
        )
        print(f"✓ Training data: {train_count} samples\n")
    else:
        print(f"⚠ Training data not found: {train_json}\n")
        train_count = 0
    
    # Convert validation data
    if os.path.exists(val_json):
        val_count = convert_to_llamafactory_format(
            val_json,
            audio_base_dir,
            f"{output_dir}/malaysian_asr_val.json"
        )
        print(f"✓ Validation data: {val_count} samples\n")
    else:
        print(f"⚠ Validation data not found: {val_json}\n")
        val_count = 0
    
    # Create dataset info file
    dataset_info = {
        "malaysian_asr": {
            "file_name": "malaysian_asr_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "audios": "audios"  # Specify audios column for multimodal data
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }
    
    dataset_info_path = f"{output_dir}/dataset_info.json"
    print(f"Creating dataset info at {dataset_info_path}...")
    
    # Load existing dataset_info if it exists
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            existing_info = json.load(f)
        existing_info.update(dataset_info)
        dataset_info = existing_info
    
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print("\n========================================")
    print("✅ Dataset Preparation Complete!")
    print("========================================")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"\nNext step: Run training with:")
    print(f"  cd LLaMA-Factory")
    print(f"  llamafactory-cli train ../qwen25omni_asr_qlora.yaml")


if __name__ == "__main__":
    main()

