#!/usr/bin/env python3
"""
Malaysian pronunciation dictionary test

Generates audio pairs (without/with dictionary) for sentences containing:
- YTL, Petronas
- nasi lemak, rendang
- Kuala Lumpur, Putrajaya
- terima kasih

Usage:
    python test_malay_dictionary.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from script directory
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env")

from elevenlabs import ElevenLabs, PronunciationDictionaryVersionLocator

# Configuration
VOICE_ID = "UcqZLa941Kkt8ZhEEybf"
MODEL_ID = "eleven_turbo_v2"  # Must support phoneme tags

# Test sentences with Malaysian terms
TEST_SENTENCES = [
    {
        "id": "ytl",
        "text": "YTL Corporation is one of Malaysia's largest companies.",
    },
    {
        "id": "petronas",
        "text": "Petronas Towers in Kuala Lumpur is an iconic landmark.",
    },
    {
        "id": "food",
        "text": "I love eating nasi lemak with rendang for breakfast.",
    },
    {
        "id": "greeting",
        "text": "Terima kasih for visiting Putrajaya today.",
    },
]

# Output directory
OUTPUT_DIR = _script_dir / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dictionary file
DICTIONARY_FILE = _script_dir / "dictionary_malay.pls"


def save_audio(audio_generator, path: Path):
    """Save audio from generator to file."""
    with open(path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)


def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in .env")
    
    client = ElevenLabs(api_key=api_key)
    
    print("=" * 60)
    print("Malaysian Pronunciation Dictionary Test")
    print("=" * 60)
    print()
    
    # Create pronunciation dictionary from file
    print("[Step 1] Creating pronunciation dictionary from dictionary_malay.pls...")
    
    if not DICTIONARY_FILE.exists():
        raise FileNotFoundError(f"Dictionary file not found: {DICTIONARY_FILE}")
    
    with open(DICTIONARY_FILE, "rb") as f:
        dictionary = client.pronunciation_dictionaries.create_from_file(
            file=f.read(),
            name="malay_pronunciation_test",
        )
    
    print(f"   Dictionary ID: {dictionary.id}")
    print(f"   Version ID: {dictionary.version_id}")
    print()
    
    # Generate audio for each sentence
    print("[Step 2] Generating audio pairs (without/with dictionary)...")
    print()
    
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"   [{i}/{len(TEST_SENTENCES)}] {sentence['id']}")
        print(f"       Text: {sentence['text']}")
        
        # Without dictionary
        audio_without = client.text_to_speech.convert(
            text=sentence["text"],
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )
        path_without = OUTPUT_DIR / f"{sentence['id']}_without_dict.mp3"
        save_audio(audio_without, path_without)
        print(f"       Saved: {path_without.name}")
        
        # With dictionary
        audio_with = client.text_to_speech.convert(
            text=sentence["text"],
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
            pronunciation_dictionary_locators=[
                PronunciationDictionaryVersionLocator(
                    pronunciation_dictionary_id=dictionary.id,
                    version_id=dictionary.version_id,
                )
            ],
        )
        path_with = OUTPUT_DIR / f"{sentence['id']}_with_dict.mp3"
        save_audio(audio_with, path_with)
        print(f"       Saved: {path_with.name}")
        print()
    
    # Cleanup dictionary
    print("[Step 3] Cleaning up test dictionary...")
    try:
        client.pronunciation_dictionaries.delete(
            pronunciation_dictionary_id=dictionary.id,
        )
        print("   Dictionary deleted.")
    except Exception as e:
        print(f"   Could not delete dictionary: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("\nGenerated pairs:")
    for sentence in TEST_SENTENCES:
        print(f"  - {sentence['id']}_without_dict.mp3")
        print(f"  - {sentence['id']}_with_dict.mp3")
    print("\nListen to compare the pronunciation differences!")


if __name__ == "__main__":
    main()
