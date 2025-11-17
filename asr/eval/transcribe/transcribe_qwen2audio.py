#!/usr/bin/env python3
"""
Transcription script for Qwen2-Audio models
Handles Qwen2-Audio-7B-Instruct and LoRA fine-tuned checkpoints
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union
from threading import Lock

import librosa
import torch
from tqdm import tqdm
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor
)

from utils import (
    load_test_data,
    load_dataset_by_name,
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


class Qwen2AudioTranscriber:
    """Transcriber for Qwen2-Audio models (including LoRA checkpoints)"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        asr_prompt: str = "Transcribe this audio accurately.",
        chunk_length_s: int = 30,
    ):
        """
        Initialize Qwen2-Audio transcriber
        
        Args:
            model_name: HuggingFace model name or path to LoRA checkpoint
            device: Device to run on ('cuda', 'cpu', or 'auto')
            asr_prompt: Prompt for ASR task
            chunk_length_s: Length of audio chunks in seconds (default: 30)
        """
        self.model_name = model_name
        self.asr_prompt = asr_prompt
        self.chunk_length_s = chunk_length_s
        self.lock = Lock()  # Thread lock for model access
        
        # Detect if this is a LoRA checkpoint
        model_path = Path(model_name)
        self.is_lora = (
            model_path.exists() and
            (model_path / "adapter_config.json").exists() and
            (model_path / "adapter_model.safetensors").exists()
        )
        
        if self.is_lora:
            logger.info(f"Detected LoRA checkpoint: {model_name}")
            # Load base model name from adapter config
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
                self.base_model_name = adapter_config.get("base_model_name_or_path")
                logger.info(f"Base model: {self.base_model_name}")
        else:
            logger.info(f"Loading Qwen2-Audio model: {model_name}")
        
        logger.info(f"Device: {device}")
        logger.info(f"ASR Prompt: '{asr_prompt}'")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model with float16 (Qwen2-Audio preference)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if self.is_lora:
            # Load base model + LoRA adapter
            logger.info(f"Loading base model: {self.base_model_name}")
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.base_model_name,
                device_map=self.device if self.device == "cuda" else "cpu",
                torch_dtype=dtype,
            )
            
            logger.info(f"Loading LoRA adapter from: {model_name}")
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(base_model, model_name)
            
            # Merge LoRA weights for faster inference
            logger.info("Merging LoRA weights into base model for faster inference...")
            self.model = peft_model.merge_and_unload()
            
            # Load processor from base model
            self.processor = AutoProcessor.from_pretrained(self.base_model_name)
            
        else:
            # Load full model
            logger.info("Loading Qwen2-Audio model...")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.device if self.device == "cuda" else "cpu",
                torch_dtype=dtype,
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Force greedy decoding (override generation config)
        self.model.generation_config.do_sample = False
        self.model.generation_config.num_beams = 1
        self.model.generation_config.temperature = 0.0
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        
        # Enable CUDA optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled")
        
        model_type = "LoRA checkpoint" if self.is_lora else "Full model"
        logger.info(f"Qwen2-Audio {model_type} loaded and optimized for inference!")
        logger.info(f"  Chunking: {self.chunk_length_s}s chunks for long audio")
    
    def _transcribe_chunk(self, audio_array, audio_path: str = "audio.wav") -> str:
        """
        Transcribe a single audio chunk
        
        Args:
            audio_array: Audio array (numpy array from librosa)
            audio_path: Path to audio file (for conversation format)
            
        Returns:
            Transcription text
        """
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
                
                # Process inputs
                inputs = self.processor(
                    text=text,
                    audio=audio_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                
                # Store input_ids length BEFORE moving to device
                input_ids_length = inputs["input_ids"].shape[1]
                
                # Move to device
                inputs = inputs.to(self.device)
                
                # Estimate max_new_tokens based on audio length (roughly 3 tokens per second)
                audio_duration = len(audio_array) / 16000
                max_new_tokens = max(128, int(audio_duration * 3))
                
                # Generate
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=1,
                        do_sample=False,
                        temperature=0.0,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                
                # Remove input tokens
                generate_ids = generate_ids[:, input_ids_length:]
                
                # Decode
                result = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Clean output
                transcription = clean_qwen_output(result)
                return transcription.strip()
                
            except Exception as e:
                logger.error(f"Error transcribing chunk: {e}")
                return ""
    
    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file (with chunking for long audio)
        
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
        
        # Check if we need to chunk
        if audio_duration > self.chunk_length_s:
            # Chunk the audio
            logger.info(f"Audio is {audio_duration:.1f}s, chunking into {self.chunk_length_s}s segments...")
            
            chunk_samples = self.chunk_length_s * sr
            transcriptions = []
            
            for i in range(0, len(audio_array), chunk_samples):
                chunk = audio_array[i:i + chunk_samples]
                chunk_transcription = self._transcribe_chunk(chunk, str(audio_path))
                if chunk_transcription:
                    transcriptions.append(chunk_transcription)
            
            # Concatenate all chunks
            transcription = " ".join(transcriptions)
        else:
            # Transcribe directly
            transcription = self._transcribe_chunk(audio_array, str(audio_path))
        
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        return {
            "text": transcription.strip(),
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "rtf": rtf,
        }


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen2-Audio")
    parser.add_argument("--model", required=True, 
                       help="Qwen2-Audio model name or path to LoRA checkpoint")
    parser.add_argument("--test-dataset", required=True, help="Dataset name from registry (e.g., meso-malaya-test, seacrowd-asr-malcsc)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--asr-prompt", 
                       default="Transcribe this audio accurately.",
                       help="ASR prompt")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--chunk-length", type=int, default=30,
                       help="Chunk length in seconds for long audio (default: 30)")
    
    args = parser.parse_args()
    
    # Load test data using dataset name
    logger.info(f"Loading dataset: {args.test_dataset}")
    test_data = load_dataset_by_name(
        args.test_dataset,
        max_samples=args.max_samples,
        validate=True
    )
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize transcriber
    transcriber = Qwen2AudioTranscriber(
        model_name=args.model,
        device=args.device,
        asr_prompt=args.asr_prompt,
        chunk_length_s=args.chunk_length
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

