#!/usr/bin/env python3
"""
Transcription script for Whisper models (OpenAI, mesolitica/Malaysia-AI)
Supports HuggingFace, ModelScope, and local model paths
"""

import argparse
import os
import time
import re
from pathlib import Path
from typing import Dict, Optional, Union
import logging
from threading import Lock

import torch
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

from utils import (
    load_test_data,
    load_dataset_by_name,
    validate_samples,
    save_predictions
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_and_remove_repetition(text: str, max_repetitions: int = 3) -> tuple[str, bool]:
    """
    Detect and remove repetitive patterns in transcription (hallucination detection)
    
    Args:
        text: Input transcription text
        max_repetitions: Maximum allowed consecutive repetitions of a pattern
        
    Returns:
        Tuple of (cleaned_text, was_hallucination)
    """
    if not text:
        return text, False
    
    original_text = text
    
    # Check for repeated Unicode characters (e.g., Chinese "嗯嗯嗯嗯..." hallucination)
    # This must be checked FIRST before word splitting since CJK chars don't use spaces
    # Look for the same character repeated many times consecutively
    if len(text) > 20:
        # Check for repeated single characters (especially non-ASCII like CJK)
        char_counts = {}
        i = 0
        while i < len(text):
            char = text[i]
            # Count consecutive occurrences of this character
            count = 1
            j = i + 1
            while j < len(text) and text[j] == char:
                count += 1
                j += 1
            
            # If a single character repeats more than 10 times consecutively, likely hallucination
            # Especially for non-ASCII characters (Chinese, etc.)
            if count > 10 and (ord(char) > 127 or char in ['a', 'e', 'i', 'o', 'u', '.']):
                # Find where the excessive repetition starts
                # Keep text before the repetition
                cleaned = text[:i].strip()
                logger.warning(f"Detected character repetition hallucination: '{char}' repeated {count} times")
                return cleaned if cleaned else char, True
            
            i = j if j > i else i + 1
    
    # Check for repeated 2-4 character patterns (e.g., "嗯嗯" or "um um" patterns)
    # This catches patterns like "嗯嗯嗯嗯" or "uh uh uh uh"
    for pattern_len in range(1, 5):
        if len(text) >= pattern_len * 10:  # Need at least 10 repetitions to check
            # Try to find patterns that repeat
            for start_pos in range(min(len(text) - pattern_len * 5, 100)):  # Check first 100 chars
                pattern = text[start_pos:start_pos + pattern_len]
                
                # Count how many times this pattern repeats consecutively
                count = 0
                pos = start_pos
                while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == pattern:
                    count += 1
                    pos += pattern_len
                
                # If pattern repeats more than 10 times, likely hallucination
                if count > 10:
                    cleaned = text[:start_pos].strip()
                    logger.warning(f"Detected character pattern hallucination: '{pattern}' repeated {count} times")
                    return cleaned if cleaned else pattern, True
    
    # Check for comma-separated repetitions (e.g., "eh, eh, eh, eh, eh,")
    # This is a very common hallucination pattern - must check FIRST
    comma_pattern = re.findall(r'(\b\w{1,5}\b)(?:,\s*\1){5,}', text)
    if comma_pattern:
        # Found repeated words separated by commas
        for repeated_word in comma_pattern:
            # Find where the repetition starts
            repetition_start = text.find(repeated_word + ',')
            if repetition_start != -1:
                # Count how many times it repeats
                temp = text[repetition_start:]
                count = temp.count(repeated_word + ',')
                if count > max_repetitions:
                    # Truncate at the start of excessive repetition
                    cleaned = text[:repetition_start].strip()
                    if not cleaned:  # If nothing before repetition, keep one instance
                        cleaned = repeated_word
                    logger.warning(f"Detected comma-separated hallucination: '{repeated_word}' repeated {count} times")
                    return cleaned, True
    
    # Check for repeated words/phrases anywhere in the text (e.g., "Kategori. Kategori. Kategori...")
    words = text.split()
    if len(words) > 3:
        # Look for patterns where the same word/phrase repeats many times
        # Use a sliding window approach to detect repetitions efficiently
        idx = 0
        while idx < len(words):
            # Try different pattern lengths starting from this position
            for pattern_len in range(1, min(10, (len(words) - idx) // (max_repetitions + 1) + 1)):
                if idx + pattern_len > len(words):
                    break
                
                pattern = words[idx:idx + pattern_len]
                
                # Count consecutive repetitions from this position
                repetitions = 0
                check_idx = idx
                while check_idx + pattern_len <= len(words):
                    if words[check_idx:check_idx + pattern_len] == pattern:
                        repetitions += 1
                        check_idx += pattern_len
                    else:
                        break
                
                # If we find excessive repetition, truncate
                if repetitions > max_repetitions:
                    pattern_str = " ".join(pattern)
                    # Keep everything before the repetition + one instance of the pattern
                    cleaned = " ".join(words[:idx + pattern_len])
                    logger.warning(f"Detected word repetition hallucination at position {idx}: pattern '{pattern_str}' repeated {repetitions} times")
                    return cleaned, True
            
            idx += 1
    
    # Check for simple repeated words (case-insensitive, ignoring punctuation)
    # This catches patterns like "kata kata kata kata"
    cleaned_words = [re.sub(r'[^\w\s]', '', w.lower()) for w in words if w]
    if len(cleaned_words) > 5:
        # Count consecutive same words
        i = 0
        while i < len(cleaned_words):
            word = cleaned_words[i]
            if not word:
                i += 1
                continue
            
            # Count how many times this word repeats consecutively
            consecutive_count = 1
            j = i + 1
            while j < len(cleaned_words) and cleaned_words[j] == word:
                consecutive_count += 1
                j += 1
            
            # If same word repeats > threshold times
            if consecutive_count > max_repetitions * 2:  # More lenient for simple words
                # Truncate before the excessive repetition
                cleaned = " ".join(words[:i + 1])  # Keep 1 instance only
                logger.warning(f"Detected simple word hallucination: '{word}' repeated {consecutive_count} times")
                return cleaned, True
            
            i = j if j > i + 1 else i + 1
    
    # Check for repeated punctuation patterns (e.g., "...")
    # If more than 10 consecutive dots, it's likely hallucination
    if text.count('.') > 10:
        dot_match = re.search(r'\.{10,}', text)
        if dot_match:
            # Find where the excessive dots start and truncate
            cleaned = text[:dot_match.start()].strip()
            logger.warning(f"Detected punctuation hallucination: {text.count('.')} dots found")
            return cleaned, True
    
    return text, False


class WhisperTranscriber:
    """Transcriber for Whisper ASR models (supports HuggingFace, ModelScope, local)"""
    
    def __init__(
        self,
        model_name: str = "mesolitica/whisper-small-malaysian-v2",
        device: str = "auto",
        language: str = "ms",
        task: str = "transcribe",
        hub: str = "hf",
        hf_token: Optional[str] = None,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 1.5,
        logprob_threshold: float = -1.0,
        condition_on_previous_text: bool = False,
        detect_hallucination: bool = True
    ):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Model name or local path
            device: Device to run on ('cuda', 'mps', 'cpu', or 'auto')
            language: Language for transcription ('ms', 'en', or None for auto-detect)
            task: Task type ('transcribe' or 'translate')
            hub: Model source ('hf' for HuggingFace, 'ms' for ModelScope, 'local' for local path)
            hf_token: Hugging Face API token (optional, will try to load from env)
            no_speech_threshold: Threshold for detecting silence (higher = more aggressive)
            compression_ratio_threshold: Threshold for detecting repetition (lower = more aggressive)
            logprob_threshold: Filter low-confidence predictions
            condition_on_previous_text: Not used (kept for API compatibility, pipeline handles this internally)
            detect_hallucination: Enable post-processing to detect and remove repetition patterns
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.hub = hub
        self.lock = Lock()  # Thread lock for model access
        
        # Anti-hallucination parameters
        self.no_speech_threshold = no_speech_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.detect_hallucination = detect_hallucination
        
        # Get HF token from argument or environment
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if self.hf_token:
            logger.info("Using Hugging Face authentication token")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Whisper model: {model_name}")
        logger.info(f"Hub/Source: {hub}")
        logger.info(f"Device: {self.device}")
        
        # Determine model path based on hub
        if hub == "local" or Path(model_name).exists():
            model_path = model_name
            logger.info(f"Loading from local path: {model_path}")
        elif hub == "ms":
            # For ModelScope, we'd need to use modelscope library
            # For now, fall back to HF if not local
            logger.warning("ModelScope hub not fully supported for Whisper, using HuggingFace")
            model_path = model_name
        else:  # hf
            model_path = model_name
        
        # Load model and processor
        load_kwargs = {}
        if self.hf_token and hub == "hf":
            load_kwargs["token"] = self.hf_token
        
        # Use pipeline for automatic long-form transcription with chunking
        # Pipeline device: use device index for CUDA/MPS, -1 for CPU
        if self.device == "cuda":
            pipe_device = 0  # Use first CUDA device
        elif self.device == "mps":
            pipe_device = 0  # MPS also uses 0
        else:
            pipe_device = -1  # CPU
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            chunk_length_s=30,  # Process in 30-second chunks
            device=pipe_device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **load_kwargs
        )
        
        # Also keep processor for compatibility
        self.processor = WhisperProcessor.from_pretrained(model_path, **load_kwargs)
        
        logger.info("Whisper model loaded successfully (using pipeline for long-form transcription)")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        return_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            return_timestamps: Whether to return timestamps
            
        Returns:
            Dictionary with transcription and timing info
        """
        # Load audio
        audio_array, sr = librosa.load(str(audio_path), sr=16000)
        audio_duration = len(audio_array) / sr
        
        # Validate audio
        if len(audio_array) == 0:
            logger.warning(f"Empty audio file: {audio_path}")
            return {
                "text": "",
                "audio_duration": 0.0,
                "processing_time": 0.0,
                "rtf": 0.0,
            }
        
        # Start timing
        start_time = time.time()
        
        with self.lock:  # Thread-safe model access
            try:
                # Use pipeline for long-form transcription
                # Pipeline automatically handles chunking for long audio
                generate_kwargs = {
                    "task": self.task,
                    "do_sample": False,
                    "temperature": 0.0,
                    "num_beams": 1,
                }
                
                # Only add language if specified (None can cause issues)
                if self.language:
                    generate_kwargs["language"] = self.language
                
                # Add anti-hallucination parameters if they differ from None
                if self.no_speech_threshold is not None:
                    generate_kwargs["no_speech_threshold"] = self.no_speech_threshold
                if self.compression_ratio_threshold is not None:
                    generate_kwargs["compression_ratio_threshold"] = self.compression_ratio_threshold
                if self.logprob_threshold is not None:
                    generate_kwargs["logprob_threshold"] = self.logprob_threshold
                
                # Note: condition_on_previous_text is not supported by the pipeline interface
                # The pipeline handles this internally based on chunk_length_s
                
                try:
                    result = self.pipe(
                        audio_array,
                        generate_kwargs=generate_kwargs,
                        return_timestamps=return_timestamps,
                    )
                except (TypeError, AttributeError) as inner_e:
                    # If we get comparison errors with thresholds, retry with minimal params
                    if "NoneType" in str(inner_e) or "not supported between" in str(inner_e):
                        logger.warning(f"Threshold parameters caused error for {audio_path}, retrying with minimal params")
                        minimal_kwargs = {"task": self.task}
                        if self.language:
                            minimal_kwargs["language"] = self.language
                        result = self.pipe(
                            audio_array,
                            generate_kwargs=minimal_kwargs,
                            return_timestamps=return_timestamps,
                        )
                    else:
                        raise
                
                transcription = result["text"]
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                logger.debug(f"generate_kwargs: {generate_kwargs}")
                logger.debug(f"audio_duration: {audio_duration}, array shape: {audio_array.shape}")
                raise
        
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        # Post-process to detect and remove hallucination
        hallucination_detected = False
        if self.detect_hallucination:
            transcription, hallucination_detected = detect_and_remove_repetition(transcription.strip())
        else:
            transcription = transcription.strip()
        
        result_dict = {
            "text": transcription,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "rtf": rtf,
        }
        
        if hallucination_detected:
            result_dict["hallucination_detected"] = True
        
        return result_dict


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper models")
    parser.add_argument("--model", required=True, help="Whisper model name or path")
    parser.add_argument("--test-dataset", help="Dataset name from registry (e.g., meso-malaya-test)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--language", default="ms", choices=["ms", "en", "auto"],
                       help="Language for transcription (default: ms)")
    parser.add_argument("--hub", default="hf", choices=["hf", "ms", "local"],
                       help="Model source: hf (HuggingFace), ms (ModelScope), local (local path)")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    
    # Anti-hallucination parameters
    parser.add_argument("--no-speech-threshold", type=float, default=0.4,
                       help="Threshold for detecting silence (lower = more aggressive, default: 0.4)")
    parser.add_argument("--compression-ratio-threshold", type=float, default=1.35,
                       help="Threshold for detecting repetition (lower = more aggressive, default: 1.35)")
    parser.add_argument("--logprob-threshold", type=float, default=-0.5,
                       help="Filter low-confidence predictions (default: -0.5)")
    parser.add_argument("--condition-on-previous-text", action="store_true",
                       help="Not used (kept for compatibility, pipeline handles this internally)")
    parser.add_argument("--no-detect-hallucination", action="store_true",
                       help="Disable post-processing hallucination detection")
    
    args = parser.parse_args()
    
    # Validate that test dataset is provided
    if not args.test_dataset:
        parser.error("--test-dataset is required")
    
    # Convert language
    language = args.language if args.language != "auto" else None
    
    # Load test data using dataset name
    logger.info(f"Loading dataset: {args.test_dataset}")
    test_data = load_dataset_by_name(
        args.test_dataset,
        max_samples=args.max_samples,
        validate=True
    )
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_name=args.model,
        device=args.device,
        hub=args.hub,
        language=language,
        hf_token=args.hf_token,
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        logprob_threshold=args.logprob_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        detect_hallucination=not args.no_detect_hallucination,
    )
    
    # Transcribe all samples
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting transcription on {len(test_data)} samples")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {transcriber.device}")
    logger.info(f"Anti-hallucination settings:")
    logger.info(f"  - No speech threshold: {args.no_speech_threshold}")
    logger.info(f"  - Compression ratio threshold: {args.compression_ratio_threshold}")
    logger.info(f"  - Logprob threshold: {args.logprob_threshold}")
    logger.info(f"  - Post-processing detection: {'enabled' if not args.no_detect_hallucination else 'disabled'}")
    logger.info(f"{'='*70}\n")
    
    predictions = []
    hallucination_count = 0
    for idx, sample in enumerate(tqdm(test_data, desc="Transcribing")):
        try:
            result = transcriber.transcribe(sample['audio_path'])
            
            prediction = {
                "audio_path": sample['audio_path'],
                "reference": sample['reference'],
                "hypothesis": result['text'],
                "audio_duration": result['audio_duration'],
                "processing_time": result['processing_time'],
                "rtf": result['rtf'],
                "index": idx
            }
            
            # Track hallucination
            if result.get('hallucination_detected', False):
                prediction['hallucination_detected'] = True
                hallucination_count += 1
            
            predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Failed to transcribe sample {idx}: {e}")
            continue
    
    # Save results
    save_predictions(predictions, args.output_dir, args.model)
    logger.info(f"\n✓ Transcription completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    # Report hallucination statistics
    if not args.no_detect_hallucination and hallucination_count > 0:
        hallucination_rate = (hallucination_count / len(predictions)) * 100 if predictions else 0
        logger.info(f"\n⚠️  Hallucination Detection Summary:")
        logger.info(f"  - Samples with detected hallucinations: {hallucination_count}/{len(predictions)} ({hallucination_rate:.1f}%)")
        logger.info(f"  - These samples were automatically cleaned to remove repetition")


if __name__ == "__main__":
    main()
