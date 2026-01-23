#!/usr/bin/env python3
"""
Batch evaluation of Faster Whisper on benchmark datasets.

Usage:
    python ytl_faster_whisper_test.py
    python ytl_faster_whisper_test.py --model large-v3-turbo
"""

import argparse
import sys
from pathlib import Path

import tqdm
from faster_whisper import WhisperModel

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    get_dataset_names,
    get_dataset_path,
    load_dataset,
    postprocess_text_mal,
    print_results,
)
from common.evaluation import compute_wer


class FasterWhisperRecognizer:
    """Faster Whisper ASR recognizer."""
    
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.download_root = Path(__file__).parent / "faster_whisper"
        self.download_root.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Faster Whisper model: {model_name}")
        print(f"Device: {device}, Compute type: {compute_type}")
        
        self.model = WhisperModel(
            model_name,
            download_root=str(self.download_root),
            device=device,
            compute_type=compute_type,
        )
        print("Model loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "ms") -> str:
        """Transcribe a single audio file."""
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            without_timestamps=False,
            vad_filter=True,
            beam_size=5,
        )
        
        text = ""
        for segment in segments:
            text += segment.text + " "
        
        return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Faster Whisper benchmark on YTL test datasets"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=get_dataset_names(),
        help="Run on specific dataset only (default: run all)"
    )
    args = parser.parse_args()

    # Initialize recognizer
    recognizer = FasterWhisperRecognizer(
        model_name=args.model,
        device=args.device,
    )

    # Determine which datasets to run
    datasets_to_run = [args.dataset] if args.dataset else get_dataset_names()

    all_wers = {}
    for dataset_name in datasets_to_run:
        print(f"\n{'=' * 60}\nProcessing dataset: {dataset_name}\n{'=' * 60}")

        # Load dataset
        dataset_path = get_dataset_path(dataset_name, Path(__file__).parent)
        audio_dict, ref_transcript, duration_dict = load_dataset(dataset_path)
        duration_hours = sum(duration_dict.values()) / 3600
        print(f"Samples: {len(audio_dict)}, Duration: {duration_hours:.2f} hours")

        # Transcribe all audio files
        sys_transcription = {}
        for audio_utt in tqdm.tqdm(audio_dict, desc="Transcribing"):
            text = recognizer.transcribe(audio_dict[audio_utt])
            sys_transcription[audio_utt] = postprocess_text_mal([text])[0]

        # Compute WER
        wer = compute_wer(sys_transcription, ref_transcript, sample_interval=0)
        print(f"Final WER for {dataset_name}: {wer:.3f}")
        all_wers[dataset_name] = wer

    print_results(all_wers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
