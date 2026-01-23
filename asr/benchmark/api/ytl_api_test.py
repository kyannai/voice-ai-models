#!/usr/bin/env python3
"""
Batch evaluation of YTL ASR API on benchmark datasets.

Usage:
    python ytl_api_test.py --model bukit-tinggi-v2 --env production
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import openai
import tqdm
from dotenv import load_dotenv

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    DATASETS,
    get_dataset_names,
    get_dataset_path,
    load_dataset,
    postprocess_text_mal,
    print_results,
    resolve_api_config,
    run_evaluation,
    split_dict,
)

load_dotenv()


class YTLASRClient:
    """YTL ASR API client using OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def transcribe(self, wav_path: str) -> str:
        """Transcribe a single audio file."""
        with open(wav_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f
            )
        return response.text


def process_audio_batch(args):
    """Worker function for parallel audio processing."""
    audio_dict_slice, out_dir, api_key, base_url, model, sleep_s = args
    client = YTLASRClient(api_key=api_key, base_url=base_url, model=model)

    for uid, wav_path in tqdm.tqdm(audio_dict_slice.items()):
        out_path = Path(out_dir) / f"{uid}.json"
        if out_path.exists():
            continue

        try:
            text = client.transcribe(wav_path)
            result = {
                "text": text,
                "text_norm": postprocess_text_mal([text])[0]
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            time.sleep(sleep_s)  # Prevent rate limiting

        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch evaluation using YTL ASR API"
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
        "--dataset", "-d",
        type=str,
        choices=get_dataset_names(),
        help="Run on specific dataset only (default: run all)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)"
    )
    args = parser.parse_args()

    try:
        api_key, base_url = resolve_api_config(args.env)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Configure workers and rate limiting based on environment
    if args.env == "staging":
        workers = 1
        sleep_s = 0.5
    else:
        workers = 4
        sleep_s = 0.1

    # Determine which datasets to run
    datasets_to_run = [args.dataset] if args.dataset else get_dataset_names()

    all_wers = {}
    for dataset_name in datasets_to_run:
        print(f"\n{'=' * 60}\nProcessing dataset: {dataset_name}\n{'=' * 60}")

        # Load dataset
        dataset_path = get_dataset_path(dataset_name, Path(__file__).parent)
        audio_dict, ref_transcript, duration_dict = load_dataset(dataset_path)
        total_samples = len(audio_dict)
        
        # Apply sample limit if specified
        if args.max_samples and len(audio_dict) > args.max_samples:
            keys = list(audio_dict.keys())[:args.max_samples]
            audio_dict = {k: audio_dict[k] for k in keys}
            ref_transcript = {k: ref_transcript[k] for k in keys}
            duration_dict = {k: duration_dict[k] for k in keys}
            print(f"Limited to {args.max_samples} samples (from {total_samples} total)")
        
        duration_hours = sum(duration_dict.values()) / 3600
        print(f"Samples: {len(audio_dict)}, Duration: {duration_hours:.2f} hours")

        # Setup output directory
        save_dir = Path(__file__).parent / f"ytl_labs/{dataset_name}"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Split work and process in parallel
        audio_dict_slices = split_dict(audio_dict, workers)
        job_args = [
            (slice_data, save_dir, api_key, base_url, args.model, sleep_s)
            for slice_data in audio_dict_slices
        ]

        with mp.Pool(processes=workers) as pool:
            pool.map(process_audio_batch, job_args)

        # Evaluate results
        wer = run_evaluation(save_dir, ref_transcript, dataset_name)
        all_wers[dataset_name] = wer

    print_results(all_wers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
