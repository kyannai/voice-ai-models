#!/usr/bin/env python3
"""
Transcription script for Whisper models (OpenAI, mesolitica/Malaysia-AI)
Supports HuggingFace, ModelScope, and local model paths
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union
import logging
from threading import Lock

import torch
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils import (
    load_test_data,
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


class WhisperTranscriber:
    """Transcriber for Whisper ASR models (supports HuggingFace, ModelScope, local)"""
    
    def __init__(
        self,
        model_name: str = "mesolitica/whisper-small-malaysian-v2",
        device: str = "auto",
        language: str = "ms",
        task: str = "transcribe",
        hub: str = "hf",
        hf_token: Optional[str] = None
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
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.hub = hub
        self.lock = Lock()  # Thread lock for model access
        
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
        
        self.processor = WhisperProcessor.from_pretrained(model_path, **load_kwargs)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Whisper model loaded successfully")
    
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
        
        # Start timing
        start_time = time.time()
        
        with self.lock:  # Thread-safe model access
            try:
                # Process audio
                inputs = self.processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate transcription
                forced_decoder_ids = None
                if self.language:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.language,
                        task=self.task
                    )
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        return_timestamps=return_timestamps
                    )
        
        # Decode transcription
        transcription = self.processor.batch_decode(
                    generated_ids,
            skip_special_tokens=True
        )[0]
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                raise
        
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        return {
            "text": transcription.strip(),
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "rtf": rtf,
        }


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper models")
    parser.add_argument("--model", required=True, help="Whisper model name or path")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON/CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--audio-dir", help="Base directory for audio files")
    parser.add_argument("--language", default="ms", choices=["ms", "en", "auto"],
                       help="Language for transcription (default: ms)")
    parser.add_argument("--hub", default="hf", choices=["hf", "ms", "local"],
                       help="Model source: hf (HuggingFace), ms (ModelScope), local (local path)")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    
    args = parser.parse_args()
    
    # Convert language
    language = args.language if args.language != "auto" else None
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data, args.audio_dir)
    test_data = validate_samples(test_data)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(test_data):
        logger.info(f"Limiting to first {args.max_samples} samples (--max-samples)")
        test_data = test_data[:args.max_samples]
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_name=args.model,
        device=args.device,
        hub=args.hub,
        language=language,
        hf_token=args.hf_token,
    )
    
    # Transcribe all samples
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting transcription on {len(test_data)} samples")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {transcriber.device}")
    logger.info(f"{'='*70}\n")
    
    predictions = []
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
            predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Failed to transcribe sample {idx}: {e}")
            continue
    
    # Save results
    save_predictions(predictions, args.output_dir, args.model)
    logger.info(f"\n✓ Transcription completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
