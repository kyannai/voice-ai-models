#!/usr/bin/env python3
"""
Batch evaluation of Mesolitica Malaysian Whisper on benchmark datasets.

Usage:
    python ytl_mesolitica_whisper_test.py
    python ytl_mesolitica_whisper_test.py --model mesolitica/Malaysian-whisper-large-v3-turbo-v3
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torchaudio
import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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


def detect_and_remove_repetition(text: str, max_repetitions: int = 3) -> tuple:
    """
    Detect and remove repetitive patterns in transcription (hallucination detection).
    
    Args:
        text: Input transcription text
        max_repetitions: Maximum allowed consecutive repetitions of a pattern
        
    Returns:
        Tuple of (cleaned_text, was_hallucination)
    """
    if not text:
        return text, False
    
    # Check for repeated Unicode characters (e.g., Chinese "嗯嗯嗯嗯..." hallucination)
    if len(text) > 20:
        i = 0
        while i < len(text):
            char = text[i]
            count = 1
            j = i + 1
            while j < len(text) and text[j] == char:
                count += 1
                j += 1
            
            if count > 10 and (ord(char) > 127 or char in ['a', 'e', 'i', 'o', 'u', '.']):
                cleaned = text[:i].strip()
                print(f"[WARNING] Detected character repetition: '{char}' repeated {count} times")
                return cleaned if cleaned else char, True
            
            i = j if j > i else i + 1
    
    # Check for repeated 2-4 character patterns
    for pattern_len in range(1, 5):
        if len(text) >= pattern_len * 10:
            for start_pos in range(min(len(text) - pattern_len * 5, 100)):
                pattern = text[start_pos:start_pos + pattern_len]
                count = 0
                pos = start_pos
                while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == pattern:
                    count += 1
                    pos += pattern_len
                
                if count > 10:
                    cleaned = text[:start_pos].strip()
                    print(f"[WARNING] Detected pattern hallucination: '{pattern}' repeated {count} times")
                    return cleaned if cleaned else pattern, True
    
    # Check for comma-separated repetitions
    comma_pattern = re.findall(r'(\b\w{1,5}\b)(?:,\s*\1){5,}', text)
    if comma_pattern:
        for repeated_word in comma_pattern:
            repetition_start = text.find(repeated_word + ',')
            if repetition_start != -1:
                temp = text[repetition_start:]
                count = temp.count(repeated_word + ',')
                if count > max_repetitions:
                    cleaned = text[:repetition_start].strip()
                    if not cleaned:
                        cleaned = repeated_word
                    print(f"[WARNING] Detected comma-separated hallucination: '{repeated_word}' repeated {count} times")
                    return cleaned, True
    
    # Check for repeated words/phrases
    words = text.split()
    if len(words) > 3:
        idx = 0
        while idx < len(words):
            for pattern_len in range(1, min(10, (len(words) - idx) // (max_repetitions + 1) + 1)):
                if idx + pattern_len > len(words):
                    break
                
                pattern = words[idx:idx + pattern_len]
                repetitions = 0
                check_idx = idx
                while check_idx + pattern_len <= len(words):
                    if words[check_idx:check_idx + pattern_len] == pattern:
                        repetitions += 1
                        check_idx += pattern_len
                    else:
                        break
                
                if repetitions > max_repetitions:
                    pattern_str = " ".join(pattern)
                    cleaned = " ".join(words[:idx + pattern_len])
                    print(f"[WARNING] Detected word repetition at position {idx}: '{pattern_str}' repeated {repetitions} times")
                    return cleaned, True
            
            idx += 1
    
    # Check for simple repeated words
    cleaned_words = [re.sub(r'[^\w\s]', '', w.lower()) for w in words if w]
    if len(cleaned_words) > 5:
        i = 0
        while i < len(cleaned_words):
            word = cleaned_words[i]
            if not word:
                i += 1
                continue
            
            consecutive_count = 1
            j = i + 1
            while j < len(cleaned_words) and cleaned_words[j] == word:
                consecutive_count += 1
                j += 1
            
            if consecutive_count > max_repetitions * 2:
                cleaned = " ".join(words[:i + 1])
                print(f"[WARNING] Detected simple word hallucination: '{word}' repeated {consecutive_count} times")
                return cleaned, True
            
            i = j if j > i + 1 else i + 1
    
    # Check for repeated punctuation
    if text.count('.') > 10:
        dot_match = re.search(r'\.{10,}', text)
        if dot_match:
            cleaned = text[:dot_match.start()].strip()
            print(f"[WARNING] Detected punctuation hallucination: {text.count('.')} dots")
            return cleaned, True
    
    return text, False


class MesoliticaWhisperRecognizer:
    """Mesolitica Malaysian Whisper ASR recognizer."""
    
    def __init__(self, model_id: str = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model {model_id} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Statistics
        self.hallucination_count = 0
        self.total_transcriptions = 0
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Prepare input features
        input_features = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="ms",
                task="transcribe",
                return_timestamps=True,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
            )
        
        # Decode
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Apply hallucination detection
        self.total_transcriptions += 1
        cleaned_text, was_hallucination = detect_and_remove_repetition(transcription.strip())
        if was_hallucination:
            self.hallucination_count += 1
            print(f"[HALLUCINATION] Detected in: {audio_path}")
            print(f"  Original length: {len(transcription)} chars, Cleaned: {len(cleaned_text)} chars")
        
        return cleaned_text
    
    def print_stats(self):
        """Print hallucination statistics."""
        print("\n" + "-" * 50)
        print("Hallucination Statistics:")
        print("-" * 50)
        print(f"  Total transcriptions: {self.total_transcriptions}")
        print(f"  Hallucinations detected: {self.hallucination_count}")
        if self.total_transcriptions > 0:
            rate = self.hallucination_count / self.total_transcriptions * 100
            print(f"  Hallucination rate: {rate:.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Mesolitica Malaysian Whisper benchmark on YTL test datasets"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mesolitica/Malaysian-whisper-large-v3-turbo-v3",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=get_dataset_names(),
        help="Run on specific dataset only (default: run all)"
    )
    args = parser.parse_args()

    # Initialize recognizer
    recognizer = MesoliticaWhisperRecognizer(model_id=args.model)

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
    recognizer.print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
