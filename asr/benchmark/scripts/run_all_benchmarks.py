#!/usr/bin/env python3
"""
Comprehensive ASR benchmark runner.

Runs all models from inference_exp.py on all benchmark datasets and saves WER results.

Models:
- Faster Whisper: large-v3-turbo
- Whisper: openai/whisper-large-v3-turbo, mesolitica/Malaysian-whisper-large-v3-turbo-v3, whisper-malay-finetuned
- Parakeet: nvidia/parakeet-tdt-0.6b-v3, parakeet-tdt-5k-v3, parakeet-tdt-multilingual, parakeet-malay

Datasets:
- Malay (ms): fleurs_test, malay_conversational, malay_scripted, supa
- Chinese (zh): kespeech, childmandarin, chinese_lips

Usage:
    python run_all_benchmarks.py
    python run_all_benchmarks.py --output-dir results
    python run_all_benchmarks.py --models faster-whisper parakeet-v3
    python run_all_benchmarks.py --datasets fleurs_test supa
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import soundfile as sf
import torch
import tqdm

# Add parent directory for common imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    get_dataset_language,
    get_dataset_names,
    get_dataset_path,
    load_dataset,
    postprocess_text_mal,
)
from common.evaluation import compute_wer

# ============================================================================
# Model configuration
# ============================================================================

# Fine-tuned model paths (same as inference_exp.py)
WHISPER_FINETUNED_DIR = "/home/kyan/voice-ai/asr/train/train_whisper/outputs/whisper-malay"
PARAKEET_MULTILINGUAL_CHECKPOINT_DIR = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/outputs/parakeet-tdt-multilingual/parakeet-tdt-multilingual-v1/checkpoints"
PARAKEET_MULTILINGUAL_BASE = "/home/kyan/voice-ai/asr/train/models/parakeet-tdt-multilingual-init.nemo"
PARAKEET_MALAY_CHECKPOINT_DIR = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/outputs/parakeet-tdt-malay/parakeet-tdt-malay/checkpoints"
PARAKEET_MALAY_BASE = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/models/parakeet-tdt-5k-v3.nemo"
PARAKEET_5K_MODEL = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/models/parakeet-tdt-5k-v3.nemo"
PARAKEET_ORIGINAL_VOCAB_SIZE = 8192

# Target languages for restricted inference
TARGET_LANGUAGES = ["en", "ms", "zh"]


def get_latest_checkpoint(output_dir: str, pattern: str = "checkpoint-*") -> Optional[str]:
    """Find the latest checkpoint in the output directory."""
    import glob
    checkpoint_pattern = os.path.join(output_dir, pattern)
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    def get_checkpoint_num(path):
        match = re.search(r'[-=](\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    return max(checkpoints, key=get_checkpoint_num)


# ============================================================================
# Text normalization
# ============================================================================

# Compound words that should be normalized (space-separated → joined)
# Format: "spaced version" -> "normalized version"
COMPOUND_WORDS = {
    "duit now": "duitnow",
    "may bank": "maybank",
    "cimb clicks": "cimbclicks",
    "touch n go": "touchngo",
    "touch and go": "touchngo",
    "e wallet": "ewallet",
    "e-wallet": "ewallet",
    "m banking": "mbanking",
    "m-banking": "mbanking",
    "i banking": "ibanking",
    "i-banking": "ibanking",
    "pin number": "pinnumber",
    "pin kod": "pinkod",
    "atm card": "atmcard",
    "debit card": "debitcard",
    "credit card": "creditcard",
    "bank account": "bankaccount",
    "ic number": "icnumber",
    "i c": "ic",
    "n r i c": "nric",
    "nric number": "nricnumber",
}


def normalize_compound_words(text: str) -> str:
    """Normalize compound words to handle spacing variations."""
    text_lower = text.lower()
    for spaced, normalized in COMPOUND_WORDS.items():
        text_lower = text_lower.replace(spaced, normalized)
    return text_lower


def postprocess_text_zh(texts: List[str]) -> List[str]:
    """Normalize Chinese text for WER evaluation."""
    result = []
    for text in texts:
        # Remove punctuation and special characters
        text = re.sub(r'[，。！？、：；""''（）【】《》\s]+', '', text)
        # Convert to character-level (space-separated for WER)
        text = ' '.join(list(text))
        result.append(text.lower())
    return result


def postprocess_text_ms(texts: List[str]) -> List[str]:
    """Normalize Malay text with compound word handling."""
    # First apply standard Malay normalization
    normalized = postprocess_text_mal(texts)
    # Then normalize compound words
    return [normalize_compound_words(text) for text in normalized]


def postprocess_text(texts: List[str], language: str) -> List[str]:
    """Normalize text based on language."""
    if language == "zh":
        return postprocess_text_zh(texts)
    else:
        return postprocess_text_ms(texts)


# ============================================================================
# Audio preprocessing
# ============================================================================

def preprocess_audio(audio_path: str, target_sr: int = 16000) -> Tuple[str, bool]:
    """Preprocess audio: convert stereo to mono and resample if needed."""
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    
    needs_processing = False
    
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
        needs_processing = True
    
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        needs_processing = True
    
    if needs_processing:
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, waveform, target_sr)
        return temp_file.name, True
    
    return audio_path, False


# ============================================================================
# Model classes
# ============================================================================

class FasterWhisperModel:
    """Faster Whisper model."""
    
    name = "faster-whisper-large-v3-turbo"
    short_name = "faster-whisper"
    
    def __init__(self):
        from faster_whisper import WhisperModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading {self.name} on {device}...")
        self.model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
        print(f"  Loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "ms") -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language=language if language != "zh" else "zh",
            beam_size=5,
            task="transcribe",
        )
        return "".join(segment.text for segment in segments).strip()


class WhisperHFModel:
    """HuggingFace Whisper model."""
    
    def __init__(self, model_id: str, display_name: str):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        self.name = display_name
        self.short_name = model_id.split("/")[-1]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading {display_name} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"  Loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "ms") -> str:
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        input_features = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=language,
                task="transcribe",
            )
        
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class WhisperLoRAModel:
    """Whisper LoRA fine-tuned model."""
    
    def __init__(self, checkpoint_path: str, base_model: str = "openai/whisper-large-v3-turbo"):
        from peft import PeftModel
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        checkpoint_name = os.path.basename(checkpoint_path)
        self.name = f"whisper-malay-finetuned ({checkpoint_name})"
        self.short_name = "whisper-malay-finetuned"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading {self.name} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(checkpoint_path)
        
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, checkpoint_path)
        self.model = self.model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()
        print(f"  Loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "ms") -> str:
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        input_features = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=language,
                task="transcribe",
            )
        
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class ParakeetModel:
    """NVIDIA Parakeet model."""
    
    def __init__(
        self,
        model_path: str,
        display_name: str,
        model_type: str = "nemo",
        base_model: str = None,
    ):
        import nemo.collections.asr as nemo_asr
        
        self.name = display_name
        self.short_name = display_name.split()[0] if " " in display_name else display_name
        self.has_chinese = False
        
        print(f"Loading {display_name}...")
        
        if model_type == "nemo-hf":
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_path)
        elif model_type == "nemo-ckpt":
            self.model = nemo_asr.models.ASRModel.restore_from(base_model)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Check for Chinese tokens
        try:
            vocab_size = self.model.joint.joint_net[-1].bias.shape[0]
            self.has_chinese = vocab_size > PARAKEET_ORIGINAL_VOCAB_SIZE
            if self.has_chinese:
                print(f"  Chinese tokens: {vocab_size - PARAKEET_ORIGINAL_VOCAB_SIZE}")
        except Exception:
            pass
        
        print(f"  Loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "ms") -> str:
        processed_path, is_temp = preprocess_audio(audio_path)
        
        try:
            result = self.model.transcribe([processed_path])
            
            if hasattr(result[0], 'text'):
                text = result[0].text
            else:
                text = str(result[0])
            
            return text.strip()
        finally:
            if is_temp:
                try:
                    os.unlink(processed_path)
                except Exception:
                    pass


# ============================================================================
# Model registry
# ============================================================================

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get all available models with their configuration."""
    models = {}
    
    # Faster Whisper
    models["faster-whisper"] = {
        "class": FasterWhisperModel,
        "args": {},
        "display_name": "faster-whisper-large-v3-turbo",
    }
    
    # OpenAI Whisper
    models["whisper-openai"] = {
        "class": WhisperHFModel,
        "args": {
            "model_id": "openai/whisper-large-v3-turbo",
            "display_name": "openai/whisper-large-v3-turbo",
        },
        "display_name": "openai/whisper-large-v3-turbo",
    }
    
    # Mesolitica Whisper
    models["whisper-mesolitica"] = {
        "class": WhisperHFModel,
        "args": {
            "model_id": "mesolitica/Malaysian-whisper-large-v3-turbo-v3",
            "display_name": "mesolitica/Malaysian-whisper-large-v3-turbo-v3",
        },
        "display_name": "mesolitica/Malaysian-whisper-large-v3-turbo-v3",
    }
    
    # Whisper LoRA fine-tuned
    checkpoint = get_latest_checkpoint(WHISPER_FINETUNED_DIR)
    if checkpoint:
        models["whisper-finetuned"] = {
            "class": WhisperLoRAModel,
            "args": {"checkpoint_path": checkpoint},
            "display_name": f"whisper-malay-finetuned ({os.path.basename(checkpoint)})",
        }
    
    # Parakeet v3
    models["parakeet-v3"] = {
        "class": ParakeetModel,
        "args": {
            "model_path": "nvidia/parakeet-tdt-0.6b-v3",
            "display_name": "nvidia/parakeet-tdt-0.6b-v3",
            "model_type": "nemo-hf",
        },
        "display_name": "nvidia/parakeet-tdt-0.6b-v3",
    }
    
    # Parakeet 5k
    if os.path.exists(PARAKEET_5K_MODEL):
        models["parakeet-5k"] = {
            "class": ParakeetModel,
            "args": {
                "model_path": PARAKEET_5K_MODEL,
                "display_name": "parakeet-tdt-5k-v3",
                "model_type": "nemo",
            },
            "display_name": "parakeet-tdt-5k-v3",
        }
    
    # Parakeet multilingual
    ckpt = get_latest_checkpoint(PARAKEET_MULTILINGUAL_CHECKPOINT_DIR, "*.ckpt")
    if ckpt and os.path.exists(PARAKEET_MULTILINGUAL_BASE):
        step_match = re.search(r'step=(\d+)', os.path.basename(ckpt))
        step = step_match.group(1) if step_match else "?"
        models["parakeet-multilingual"] = {
            "class": ParakeetModel,
            "args": {
                "model_path": ckpt,
                "display_name": f"parakeet-multilingual (step={step})",
                "model_type": "nemo-ckpt",
                "base_model": PARAKEET_MULTILINGUAL_BASE,
            },
            "display_name": f"parakeet-multilingual (step={step})",
        }
    
    # Parakeet malay
    ckpt_malay = get_latest_checkpoint(PARAKEET_MALAY_CHECKPOINT_DIR, "*.ckpt")
    if ckpt_malay and os.path.exists(PARAKEET_MALAY_BASE):
        step_match = re.search(r'step=(\d+)', os.path.basename(ckpt_malay))
        step = step_match.group(1) if step_match else "?"
        models["parakeet-malay"] = {
            "class": ParakeetModel,
            "args": {
                "model_path": ckpt_malay,
                "display_name": f"parakeet-malay (step={step})",
                "model_type": "nemo-ckpt",
                "base_model": PARAKEET_MALAY_BASE,
            },
            "display_name": f"parakeet-malay (step={step})",
        }
    
    return models


# ============================================================================
# Benchmark runner
# ============================================================================

def compute_utterance_wer(hypothesis: str, reference: str) -> float:
    """Compute WER for a single utterance."""
    from torchmetrics.text import WordErrorRate
    wer_metric = WordErrorRate()
    wer_metric.update([hypothesis], [reference])
    return float(wer_metric.compute())


def run_benchmark(
    model,
    dataset_name: str,
    audio_dict: Dict[str, str],
    ref_transcript: Dict[str, str],
    language: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run benchmark on a single dataset with a model.
    
    Returns:
        Tuple of (overall_wer, list of per-utterance results)
        Each result dict has: utterance_id, groundtruth, predicted, wer
    """
    
    results = []
    
    for utt_id in tqdm.tqdm(audio_dict, desc=f"  {dataset_name}", leave=False):
        try:
            text = model.transcribe(audio_dict[utt_id], language=language)
            predicted = postprocess_text([text], language)[0]
        except Exception as e:
            print(f"    Error on {utt_id}: {e}")
            predicted = ""
        
        # Normalize reference
        groundtruth = postprocess_text([ref_transcript[utt_id]], language)[0]
        
        # Compute per-utterance WER
        utt_wer = compute_utterance_wer(predicted, groundtruth)
        
        results.append({
            "utterance_id": utt_id,
            "groundtruth": groundtruth,
            "predicted": predicted,
            "wer": utt_wer,
        })
    
    # Compute overall WER
    hypotheses = {r["utterance_id"]: r["predicted"] for r in results}
    references = {r["utterance_id"]: r["groundtruth"] for r in results}
    overall_wer = compute_wer(hypotheses, references, sample_interval=0, verbose=False)
    
    return overall_wer, results


def main():
    parser = argparse.ArgumentParser(
        description="Run all ASR models on all benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        help="Specific models to run (default: all available)",
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        nargs="+",
        help="Specific datasets to run (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples per dataset",
    )
    args = parser.parse_args()
    
    # Get datasets first (needed for folder naming)
    all_datasets = args.datasets if args.datasets else get_dataset_names()
    
    # Setup output directory with dataset name(s) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_prefix = "_".join(all_datasets) if len(all_datasets) <= 3 else f"{len(all_datasets)}_datasets"
    folder_name = f"{datasets_prefix}_{timestamp}"
    output_dir = Path(args.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("ASR BENCHMARK - ALL MODELS")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    
    # Get available models
    available_models = get_available_models()
    model_keys = args.models if args.models else list(available_models.keys())
    
    # Filter to available models
    model_keys = [k for k in model_keys if k in available_models]
    
    print(f"\nModels ({len(model_keys)}):")
    for key in model_keys:
        print(f"  - {available_models[key]['display_name']}")
    
    # Group by language
    ms_datasets = [d for d in all_datasets if get_dataset_language(d) == "ms"]
    zh_datasets = [d for d in all_datasets if get_dataset_language(d) == "zh"]
    
    print(f"\nDatasets:")
    print(f"  Malay ({len(ms_datasets)}): {', '.join(ms_datasets)}")
    print(f"  Chinese ({len(zh_datasets)}): {', '.join(zh_datasets)}")
    print(f"{'='*70}\n")
    
    # Load all datasets
    # Dataset paths are relative to subdirectories (e.g., parakeet/, whisper/),
    # so we use a subdirectory as base to make ../test_data/ resolve correctly
    base_dir = Path(__file__).parent.parent / "parakeet"
    datasets = {}
    for dataset_name in all_datasets:
        dataset_path = get_dataset_path(dataset_name, base_dir)
        audio_dict, ref_transcript, duration_dict = load_dataset(dataset_path)
        
        # Apply sample limit
        if args.max_samples and len(audio_dict) > args.max_samples:
            import random
            selected = random.sample(list(audio_dict.keys()), args.max_samples)
            audio_dict = {k: audio_dict[k] for k in selected}
            ref_transcript = {k: ref_transcript[k] for k in selected}
            duration_dict = {k: duration_dict[k] for k in selected}
        
        datasets[dataset_name] = {
            "audio_dict": audio_dict,
            "ref_transcript": ref_transcript,
            "duration_dict": duration_dict,
            "language": get_dataset_language(dataset_name),
        }
        
        duration_hours = sum(duration_dict.values()) / 3600
        print(f"Loaded {dataset_name}: {len(audio_dict)} samples, {duration_hours:.2f} hours")
    
    # Results storage
    all_results = {}  # model -> dataset -> WER
    
    # Run benchmarks
    for model_key in model_keys:
        model_config = available_models[model_key]
        print(f"\n{'='*70}")
        print(f"Model: {model_config['display_name']}")
        print(f"{'='*70}")
        
        try:
            model = model_config["class"](**model_config["args"])
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue
        
        all_results[model_key] = {}
        
        for dataset_name in all_datasets:
            data = datasets[dataset_name]
            
            wer, utterance_results = run_benchmark(
                model,
                dataset_name,
                data["audio_dict"],
                data["ref_transcript"],
                data["language"],
            )
            
            all_results[model_key][dataset_name] = wer
            print(f"  {dataset_name}: WER = {wer:.4f} ({wer*100:.2f}%)")
            
            # Save all results to a single JSON file
            results_dir = output_dir / model_key
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / f"{dataset_name}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_config["display_name"],
                    "dataset": dataset_name,
                    "overall_wer": wer,
                    "num_samples": len(utterance_results),
                    "results": utterance_results,
                }, f, ensure_ascii=False, indent=2)
        
        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save summary results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    # Print table header
    header = ["Model"] + all_datasets
    print(f"\n{header[0]:<40}", end="")
    for ds in all_datasets:
        print(f"{ds:>15}", end="")
    print()
    print("-" * (40 + 15 * len(all_datasets)))
    
    # Print results
    summary_rows = []
    for model_key in model_keys:
        if model_key not in all_results:
            continue
        
        display_name = available_models[model_key]["display_name"]
        short_name = display_name[:38] if len(display_name) > 38 else display_name
        print(f"{short_name:<40}", end="")
        
        row = {"model": display_name}
        for ds in all_datasets:
            wer = all_results[model_key].get(ds, float('nan'))
            row[ds] = wer
            print(f"{wer*100:>14.2f}%", end="")
        print()
        summary_rows.append(row)
    
    # Save CSV summary
    csv_path = output_dir / "wer_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + all_datasets)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    # Save JSON summary
    json_path = output_dir / "wer_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "models": {k: available_models[k]["display_name"] for k in model_keys if k in all_results},
            "datasets": {
                "ms": ms_datasets,
                "zh": zh_datasets,
            },
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {json_path.name}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
