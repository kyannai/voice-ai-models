#!/usr/bin/env python3
"""
Transcription script for Qwen3-Omni models
Handles Qwen3-Omni-30B-A3B-Instruct and other Qwen3-Omni variants
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Union
from threading import Lock

import librosa
import torch
from tqdm import tqdm
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor
)

from utils import (
    load_test_data,
    validate_samples,
    clean_qwen_output,
    save_predictions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Qwen3OmniTranscriber:
    """Transcriber for Qwen3-Omni models"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: str = "auto",
        asr_prompt: str = "Transcribe the audio into text.",
    ):
        """
        Initialize Qwen3-Omni transcriber
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or 'auto')
            asr_prompt: Prompt for ASR task
        """
        self.model_name = model_name
        self.asr_prompt = asr_prompt
        self.lock = Lock()  # Thread lock for model access
        
        logger.info(f"Loading Qwen3-Omni model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"ASR Prompt: '{asr_prompt}'")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Store dtype for later use in inference
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        logger.info("Loading Qwen3-Omni model...")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device if self.device == "cuda" else "cpu",
            torch_dtype=self.dtype,
        )
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Load processor
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        
        # Enable CUDA optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled")
        
        logger.info("Qwen3-Omni model loaded and optimized for inference!")
    
    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
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
                # Prepare conversation format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": str(audio_path)},
                            {"type": "text", "text": self.asr_prompt},
                        ],
                    }
                ]
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                # Process inputs (Qwen3-Omni expects lists) - following official sample
                inputs = self.processor(
                    text=text,
                    audio=[audio_array],  # List format as per official docs
                    images=None,
                    videos=None,
                    return_tensors="pt",
                    padding=True
                )
                
                # Store input_ids length BEFORE moving to device (official approach)
                input_ids_length = inputs["input_ids"].shape[1]
                
                # Move to device and convert to model dtype (official approach)
                inputs = inputs.to(self.device).to(self.dtype)
                
                # Generate (following official sample code)
                with torch.no_grad():
                    # Official code: text_ids, audio = model.generate(...)
                    # Returns tuple of (text_ids_dict, audio_tensor)
                    text_ids, audio = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        min_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                
                # Decode (following official sample: text_ids.sequences[:, inputs["input_ids"].shape[1]:])
                # text_ids is a GenerateDecoderOnlyOutput with .sequences attribute
                result = self.processor.batch_decode(
                    text_ids.sequences[:, input_ids_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Clean output
                transcription = clean_qwen_output(result)
                
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
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen3-Omni")
    parser.add_argument("--model", required=True, help="Qwen3-Omni model name or path")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON/CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--audio-dir", help="Base directory for audio files")
    parser.add_argument("--asr-prompt", default="Transcribe the audio into text.", 
                       help="ASR prompt")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    
    args = parser.parse_args()
    
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
    transcriber = Qwen3OmniTranscriber(
        model_name=args.model,
        device=args.device,
        asr_prompt=args.asr_prompt
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

