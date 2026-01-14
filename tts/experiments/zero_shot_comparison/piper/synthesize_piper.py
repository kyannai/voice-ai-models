#!/usr/bin/env python3
"""
Piper TTS Synthesizer
=====================
Fast and local neural text-to-speech using Piper.

Note: This model uses PRE-TRAINED voices, not zero-shot voice cloning.
You need to download voice models first.

Available voices: https://huggingface.co/rhasspy/piper-voices

Usage:
    python synthesize_piper.py --text "Hello world" --output output.wav
    python synthesize_piper.py --text "Hello" --model en_US-lessac-medium
"""

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

# Common voice models (language_REGION-name-quality)
# Quality levels: x_low, low, medium, high
COMMON_VOICES = {
    # English
    "en_US-lessac-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-amy-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-ryan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    "en_GB-alba-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
    # Chinese
    "zh_CN-huayan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
    # Spanish
    "es_ES-davefx-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
    # French
    "fr_FR-upmc-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
    # German
    "de_DE-thorsten-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
}


class PiperSynthesizer:
    """Wrapper for Piper TTS."""

    def __init__(self, model_name: str = "en_US-lessac-medium", models_dir: str = "models"):
        """Initialize Piper TTS.
        
        Args:
            model_name: Voice model name (e.g., en_US-lessac-medium)
            models_dir: Directory to store downloaded models
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self) -> Path:
        """Get path to the model file, downloading if needed."""
        model_path = self.models_dir / f"{self.model_name}.onnx"
        config_path = self.models_dir / f"{self.model_name}.onnx.json"

        if not model_path.exists():
            print(f"Downloading model: {self.model_name}")
            self._download_model(model_path, config_path)

        return model_path

    def _download_model(self, model_path: Path, config_path: Path):
        """Download model and config from HuggingFace."""
        import urllib.request

        # Get URLs
        if self.model_name in COMMON_VOICES:
            model_url = COMMON_VOICES[self.model_name]
        else:
            # Try to construct URL from model name
            # Format: lang_REGION-name-quality
            parts = self.model_name.split("-")
            if len(parts) >= 3:
                lang_region = parts[0]
                lang = lang_region.split("_")[0]
                name = parts[1]
                quality = parts[2]
                model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}/{name}/{quality}/{self.model_name}.onnx"
            else:
                raise ValueError(f"Unknown model: {self.model_name}. Use one of: {list(COMMON_VOICES.keys())}")

        config_url = model_url + ".json"

        print(f"Downloading: {model_url}")
        urllib.request.urlretrieve(model_url, model_path)

        print(f"Downloading: {config_url}")
        urllib.request.urlretrieve(config_url, config_path)

        print(f"Model saved to: {model_path}")

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            
        Returns:
            Path to the output audio file
        """
        model_path = self._get_model_path()

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Synthesizing with model: {self.model_name}")
        print(f"Text: {text}")

        # Use piper CLI
        cmd = [
            "piper",
            "--model", str(model_path),
            "--output_file", str(output_path),
        ]

        # Run piper with text input via stdin
        result = subprocess.run(
            cmd,
            input=text,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(f"Piper failed: {result.stderr}")

        print(f"Audio saved to: {output_path}")
        return str(output_path)

    def synthesize_python_api(self, text: str, output_path: str) -> str:
        """Synthesize using Python API (alternative method).
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            
        Returns:
            Path to the output audio file
        """
        from piper import PiperVoice

        model_path = self._get_model_path()
        config_path = model_path.with_suffix(".onnx.json")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Synthesizing with model: {self.model_name}")
        print(f"Text: {text}")

        # Load voice
        voice = PiperVoice.load(str(model_path), str(config_path))

        # Synthesize to file
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize(text, wav_file)

        print(f"Audio saved to: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Piper TTS: Fast and local neural text-to-speech"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/piper.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_US-lessac-medium",
        help=f"Voice model name. Common: {list(COMMON_VOICES.keys())}",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store downloaded models",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List common voice models",
    )

    args = parser.parse_args()

    if args.list_voices:
        print("Common Piper voices:")
        for voice in COMMON_VOICES:
            print(f"  - {voice}")
        print("\nMore voices: https://huggingface.co/rhasspy/piper-voices")
        return

    synthesizer = PiperSynthesizer(
        model_name=args.model,
        models_dir=args.models_dir,
    )
    synthesizer.synthesize(text=args.text, output_path=args.output)


if __name__ == "__main__":
    main()
