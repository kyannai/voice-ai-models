#!/usr/bin/env python3
"""
MagpieTTS Synthesizer
=====================
NVIDIA MagpieTTS multilingual TTS using NeMo framework.

Note: This model uses FIXED speakers (Sofia, Aria, Jason, Leo, John),
not zero-shot voice cloning from reference audio.

Supports 7 languages: English, Spanish, German, French, Vietnamese, Italian, Mandarin

Usage:
    python synthesize_magpietts.py --text "Hello world" --output output.wav
    python synthesize_magpietts.py --text "Bonjour" --language fr --speaker Sofia
"""

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch


# Speaker mapping
SPEAKER_MAP = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}

# Language codes
SUPPORTED_LANGUAGES = ["en", "es", "de", "fr", "vi", "it", "zh"]


class MagpieTTSSynthesizer:
    """Wrapper for NVIDIA MagpieTTS model."""

    def __init__(self, model_name: str = "nvidia/magpie_tts_multilingual_357m"):
        """Initialize the MagpieTTS model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the MagpieTTS model."""
        if self.model is not None:
            return

        print(f"Loading MagpieTTS model: {self.model_name}")
        from nemo.collections.tts.models import MagpieTTSModel

        self.model = MagpieTTSModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        speaker: str = "Sofia",
        apply_text_normalization: bool = True,
    ) -> str:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output audio
            language: Language code (en, es, de, fr, vi, it, zh)
            speaker: Speaker name (John, Sofia, Aria, Jason, Leo)
            apply_text_normalization: Whether to apply text normalization
            
        Returns:
            Path to the output audio file
        """
        self.load_model()

        # Validate inputs
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {SUPPORTED_LANGUAGES}"
            )

        if speaker not in SPEAKER_MAP:
            raise ValueError(
                f"Unknown speaker: {speaker}. "
                f"Available: {list(SPEAKER_MAP.keys())}"
            )

        speaker_idx = SPEAKER_MAP[speaker]

        print(f"Synthesizing with speaker '{speaker}' (idx={speaker_idx}), language='{language}'")
        print(f"Text: {text}")

        # Generate audio
        with torch.no_grad():
            audio, audio_len = self.model.do_tts(
                text,
                language=language,
                apply_TN=apply_text_normalization,
                speaker_index=speaker_idx,
            )

        # Convert to numpy and handle shape
        audio_np = audio.cpu().numpy()
        
        # Squeeze extra dimensions and ensure 1D audio
        audio_np = audio_np.squeeze()
        
        # Trim to actual length if audio_len is provided
        if audio_len is not None:
            if isinstance(audio_len, torch.Tensor):
                audio_len = audio_len.item()
            audio_np = audio_np[:int(audio_len)]
        
        # MagpieTTS outputs at 22kHz
        sample_rate = 22050

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with explicit format
        sf.write(str(output_path), audio_np, sample_rate, format='WAV', subtype='PCM_16')
        print(f"Audio saved to: {output_path}")

        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="MagpieTTS: NVIDIA multilingual TTS"
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
        default="output/magpietts.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=SUPPORTED_LANGUAGES,
        help="Language code",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Sofia",
        choices=list(SPEAKER_MAP.keys()),
        help="Speaker name",
    )
    parser.add_argument(
        "--no-text-norm",
        action="store_true",
        help="Disable text normalization",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/magpie_tts_multilingual_357m",
        help="Model name on HuggingFace",
    )

    args = parser.parse_args()

    synthesizer = MagpieTTSSynthesizer(model_name=args.model)
    synthesizer.synthesize(
        text=args.text,
        output_path=args.output,
        language=args.language,
        speaker=args.speaker,
        apply_text_normalization=not args.no_text_norm,
    )


if __name__ == "__main__":
    main()
