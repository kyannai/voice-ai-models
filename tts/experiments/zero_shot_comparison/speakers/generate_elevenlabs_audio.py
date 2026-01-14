#!/usr/bin/env python3
"""
Generate ~30 second Malay reference audio using ElevenLabs for zero-shot voice cloning.

Requirements:
    pip install elevenlabs python-dotenv pydub

Usage:
    # Create .env file in speakers/ folder with:
    #   ELEVENLABS_API_KEY=your_api_key
    
    python generate_elevenlabs_audio.py --voice-id <VOICE_ID> --output-dir ./output
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Load .env from the same directory as this script
from dotenv import load_dotenv
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Malay text for ~30 second reference audio
# Diverse in tone and phoneme coverage
REFERENCE_TEXT = """Selamat pagi! Apa khabar hari ini? Saya harap semuanya baik-baik sahaja. Cuaca hari ini sangat cantik, sesuai untuk kita berjalan-jalan di taman. Terima kasih banyak-banyak atas bantuan awak, saya sangat menghargainya."""


def get_elevenlabs_client(api_key: Optional[str] = None):
    """Initialize ElevenLabs client"""
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        raise ImportError(
            "ElevenLabs library not installed. Install with:\n"
            "  pip install elevenlabs"
        )
    
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ElevenLabs API key required. Either:\n"
            "  1. Set ELEVENLABS_API_KEY in .env file\n"
            "  2. Pass --api-key argument"
        )
    
    return ElevenLabs(api_key=api_key)


def synthesize_text(
    client,
    text: str,
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
) -> bytes:
    """Synthesize a single text using ElevenLabs API"""
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    
    # Collect audio bytes from generator
    audio_bytes = b""
    for chunk in audio:
        audio_bytes += chunk
    
    return audio_bytes


def generate_reference_audio(
    voice_id: str,
    output_dir: str = "./output",
    api_key: Optional[str] = None,
    model_id: str = "eleven_multilingual_v2",
) -> Dict:
    """
    Generate ~30 second reference audio for zero-shot voice cloning
    
    Args:
        voice_id: ElevenLabs voice ID to use
        output_dir: Directory to save output files
        api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
        model_id: ElevenLabs model ID
        
    Returns:
        Dictionary with generation results
    """
    from pydub import AudioSegment
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    client = get_elevenlabs_client(api_key)
    
    logger.info(f"Voice ID: {voice_id}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Text length: {len(REFERENCE_TEXT)} characters")
    
    final_output = output_path / f"elevenlabs_{voice_id}.wav"
    
    logger.info("Synthesizing audio...")
    
    try:
        audio_bytes = synthesize_text(
            client=client,
            text=REFERENCE_TEXT,
            voice_id=voice_id,
            model_id=model_id,
        )
        
        # Save as temporary MP3 then convert to WAV
        temp_mp3 = output_path / f"temp_{voice_id}.mp3"
        with open(temp_mp3, "wb") as f:
            f.write(audio_bytes)
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(str(temp_mp3))
        audio.export(str(final_output), format="wav")
        total_duration = len(audio) / 1000  # Duration in seconds
        
        # Clean up temp file
        temp_mp3.unlink()
        
        logger.info(f"Audio saved: {final_output}")
        logger.info(f"Duration: {total_duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to synthesize audio: {e}")
        raise
    
    # Save metadata
    metadata = {
        "voice_id": voice_id,
        "model_id": model_id,
        "total_characters": len(REFERENCE_TEXT),
        "output_file": str(final_output),
        "total_duration_seconds": total_duration,
        "text": REFERENCE_TEXT,
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Metadata saved: {metadata_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate ~30 second Malay reference audio using ElevenLabs"
    )
    parser.add_argument(
        "--voice-id",
        required=True,
        help="ElevenLabs voice ID to use for synthesis"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for generated audio (default: ./output)"
    )
    parser.add_argument(
        "--api-key",
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY in .env)"
    )
    parser.add_argument(
        "--model-id",
        default="eleven_multilingual_v2",
        help="ElevenLabs model ID (default: eleven_multilingual_v2)"
    )
    
    args = parser.parse_args()
    
    try:
        result = generate_reference_audio(
            voice_id=args.voice_id,
            output_dir=args.output_dir,
            api_key=args.api_key,
            model_id=args.model_id,
        )
        
        print("\n" + "="*50)
        print("Generation Complete!")
        print("="*50)
        print(f"Voice ID: {result['voice_id']}")
        print(f"Duration: {result['total_duration_seconds']:.1f}s")
        print(f"Output:   {result['output_file']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# python generate_elevenlabs_audio.py --voice-id YOUR_VOICE_ID --output-dir ./output
