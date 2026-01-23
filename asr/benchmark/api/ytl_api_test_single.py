#!/usr/bin/env python3
"""
Single audio file transcription using YTL API.

Usage:
    python ytl_api_test_single.py -a /path/to/audio.wav
    python ytl_api_test_single.py -a /path/to/audio.mp3 -m bukit-tinggi-v2
"""

import argparse
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import resolve_api_config

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a single audio file using YTL ASR API"
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        required=True,
        help="Path to audio file (wav, mp3, flac, etc.)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="bukit-tinggi-v2",
        help="Model name (default: bukit-tinggi-v2)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="production",
        choices=["staging", "production"],
        help="API environment (default: production)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL override"
    )
    args = parser.parse_args()

    # Validate audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    try:
        api_key, base_url = resolve_api_config(args.env, args.base_url)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Audio file: {audio_path}")
    print(f"Model: {args.model}")
    print(f"API URL: {base_url}")
    print("-" * 60)

    # Initialize client and transcribe
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    print("Transcribing...")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=args.model,
            file=f
        )

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(response.text)
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
