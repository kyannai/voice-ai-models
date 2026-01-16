#!/usr/bin/env python3
"""
Single audio file transcription using YTL API

Usage:
    python ytl_api_test_single.py -a /path/to/audio.wav
    python ytl_api_test_single.py -a /path/to/audio.mp3 -m bukit-tinggi-v2
"""

import argparse
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()


def resolve_api_config(env_name: str, base_url_override: str | None) -> tuple[str, str]:
    if base_url_override:
        api_key = os.getenv("ILMU_API_KEY")
        if not api_key:
            raise ValueError("ILMU_API_KEY environment variable not set")
        return api_key, base_url_override

    if env_name == "staging":
        api_key = os.getenv("ILMU_STAGING_API_KEY")
        base_url = os.getenv("ILMU_STAGING_URL")
    else:
        api_key = os.getenv("ILMU_PRODUCTION_API_KEY")
        base_url = os.getenv("ILMU_PRODUCTION_URL")

    if not api_key or not base_url:
        raise ValueError(
            "Missing API config. Set ILMU_STAGING_URL/ILMU_STAGING_API_KEY "
            "or ILMU_PRODUCTION_URL/ILMU_PRODUCTION_API_KEY in .env."
        )

    return api_key, base_url


def main():
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
        default="https://api.ytlailabs.tech/v1",
        help="API base URL (default: https://api.ytlailabs.tech/v1)"
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

    # Initialize client
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # Transcribe
    print("Transcribing...")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=args.model,
            file=f
        )

    text = response.text

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(text)
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
