#!/usr/bin/env python3
"""
Batch evaluation of NVIDIA Parakeet ASR on benchmark datasets.

Usage:
    python ytl_parakeet_test.py -m nvidia/parakeet-tdt-0.6b-v2
    python ytl_parakeet_test.py -m nvidia/parakeet-tdt-0.6b-v3  # Multilingual (25 languages)
    python ytl_parakeet_test.py -m /path/to/model.nemo -b 16 -w 1
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import torch
import tqdm
from dotenv import load_dotenv

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    get_dataset_names,
    get_dataset_path,
    load_dataset,
    postprocess_text_mal,
    print_results,
    run_evaluation,
    split_dict,
)

load_dotenv()


class ParakeetRecognizer:
    """NVIDIA Parakeet model-based ASR recognizer using NeMo."""
    
    def __init__(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v2", batch_size: int = 16):
        import nemo.collections.asr as nemo_asr
        
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NeMo Parakeet model: {model_id}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        
        is_local_nemo = model_id.endswith('.nemo') and os.path.exists(model_id)
        
        if is_local_nemo:
            print(f"Loading from local .nemo file: {model_id}")
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
        else:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required for HuggingFace models. "
                    "Please add it to .env file or export it: export HF_TOKEN=your_token_here"
                )
            from huggingface_hub import login
            login(token=hf_token)
            print("Loading model from HuggingFace...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        
        print(f"Moving model to {self.device} and setting eval mode...")
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model.cfg, 'decoding'):
            if self.model.cfg.decoding.strategy != "beam":
                self.model.cfg.decoding.strategy = "greedy_batch"
                self.model.change_decoding_strategy(self.model.cfg.decoding)
        
        print(f"Model loaded successfully on {self.device}")
        self.total_transcriptions = 0
    
    def _audio_to_tensor(self, wav_path: str) -> torch.Tensor:
        """Load audio file and convert to torch tensor."""
        import librosa
        audio, sr = librosa.load(wav_path, sr=16000)
        return torch.from_numpy(audio)
    
    def transcribe(self, wav_path: str) -> str:
        """Transcribe a single audio file."""
        audio_tensor = self._audio_to_tensor(wav_path)
        
        with torch.inference_mode():
            with torch.no_grad():
                output = self.model.transcribe([audio_tensor], verbose=False)
        
        self.total_transcriptions += 1
        return self._extract_text(output)
    
    def transcribe_batch(self, wav_paths: list) -> list:
        """Transcribe multiple audio files in batch."""
        audio_tensors = [self._audio_to_tensor(p) for p in wav_paths]
        
        with torch.inference_mode():
            with torch.no_grad():
                output = self.model.transcribe(audio_tensors, verbose=False)
        
        self.total_transcriptions += len(wav_paths)
        
        if isinstance(output, tuple):
            output = output[0]
        
        texts = []
        if isinstance(output, list):
            for result in output:
                if hasattr(result, 'text'):
                    texts.append(result.text.strip())
                else:
                    texts.append(str(result).strip())
        else:
            texts = [str(output).strip()]
        
        if len(texts) != len(wav_paths):
            raise ValueError(
                f"Expected {len(wav_paths)} transcriptions but got {len(texts)}"
            )
        
        return texts

    def _extract_text(self, output) -> str:
        """Extract text from NeMo output."""
        if isinstance(output, tuple):
            output = output[0]
        
        if isinstance(output, list) and len(output) > 0:
            first_result = output[0]
            if hasattr(first_result, 'text'):
                return first_result.text.strip()
            return str(first_result).strip()
        return str(output).strip()


def process_single_worker(audio_dict, out_dir, model_id, batch_size):
    """Run transcription with a single worker (recommended for GPU)."""
    recognizer = ParakeetRecognizer(model_id=model_id, batch_size=batch_size)

    # Collect audio files that need processing
    audio_items = [
        (uid, wav_path, Path(out_dir) / f"{uid}.json")
        for uid, wav_path in audio_dict.items()
        if not (Path(out_dir) / f"{uid}.json").exists()
    ]

    if not audio_items:
        print("All files already processed.")
        return

    print(f"Processing {len(audio_items)} files...")

    # Process in batches
    for i in tqdm.tqdm(range(0, len(audio_items), batch_size), desc="Batches"):
        batch = audio_items[i:i + batch_size]
        uids = [item[0] for item in batch]
        wav_paths = [item[1] for item in batch]
        out_paths = [item[2] for item in batch]

        try:
            texts = recognizer.transcribe_batch(wav_paths)

            for uid, out_path, text in zip(uids, out_paths, texts):
                result = {
                    "text": text,
                    "text_norm": postprocess_text_mal([text])[0]
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] Batch error: {e}")
            # Fall back to individual processing
            for uid, wav_path, out_path in batch:
                try:
                    text = recognizer.transcribe(wav_path)
                    result = {
                        "text": text,
                        "text_norm": postprocess_text_mal([text])[0]
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e2:
                    print(f"[ERROR] {wav_path}: {e2}")


def process_audio_batch(args):
    """Worker function for multi-worker processing."""
    audio_dict_slice, out_dir, model_id, batch_size = args
    process_single_worker(audio_dict_slice, out_dir, model_id, batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Parakeet ASR benchmark on YTL test datasets"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to .nemo model file or HuggingFace model ID"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of workers (default: 1, recommended for GPU)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=get_dataset_names(),
        help="Run on specific dataset only (default: run all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit the number of samples to process for each dataset"
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")

    # Determine which datasets to run
    datasets_to_run = [args.dataset] if args.dataset else get_dataset_names()

    all_wers = {}
    for dataset_name in datasets_to_run:
        print(f"\n{'=' * 60}\nProcessing dataset: {dataset_name}\n{'=' * 60}")

        # Load dataset
        dataset_path = get_dataset_path(dataset_name, Path(__file__).parent)
        audio_dict, ref_transcript, duration_dict = load_dataset(dataset_path)
        
        # Apply sample limiting if requested
        if args.max_samples:
            original_num_samples = len(audio_dict)
            selected_uids = random.sample(
                list(audio_dict.keys()), 
                min(args.max_samples, original_num_samples)
            )
            audio_dict = {uid: audio_dict[uid] for uid in selected_uids}
            ref_transcript = {uid: ref_transcript[uid] for uid in selected_uids}
            duration_dict = {uid: duration_dict[uid] for uid in selected_uids}
            
            duration_hours = sum(duration_dict.values()) / 3600
            print(f"Limited to {len(audio_dict)} samples (from {original_num_samples} total). Duration: {duration_hours:.2f} hours")
        else:
            duration_hours = sum(duration_dict.values()) / 3600
            print(f"Samples: {len(audio_dict)}, Duration: {duration_hours:.2f} hours")

        # Setup output directory
        save_dir = Path(__file__).parent / f"ytl_parakeet/{dataset_name}"
        save_dir.mkdir(exist_ok=True, parents=True)

        if args.workers == 1:
            process_single_worker(audio_dict, save_dir, args.model, args.batch_size)
        else:
            audio_dict_slices = split_dict(audio_dict, args.workers)
            job_args = [
                (slice_data, save_dir, args.model, args.batch_size)
                for slice_data in audio_dict_slices
            ]
            with mp.Pool(processes=args.workers) as pool:
                pool.map(process_audio_batch, job_args)

        # Evaluate results
        wer = run_evaluation(save_dir, ref_transcript, dataset_name)
        all_wers[dataset_name] = wer

    print_results(all_wers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
