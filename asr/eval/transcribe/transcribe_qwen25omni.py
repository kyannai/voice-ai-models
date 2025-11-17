#!/usr/bin/env python3
"""
Transcription script for Qwen2.5-Omni models
Handles Qwen2.5-Omni-7B and other Qwen2.5-Omni variants

Key differences from Qwen3-Omni:
- Uses Qwen2_5OmniForConditionalGeneration (not Qwen3OmniMoeForConditionalGeneration)
- Uses Qwen2_5OmniProcessor (not Qwen3OmniMoeProcessor)
- Smaller model (7B vs 30B) - faster and more memory efficient
- Better ASR performance according to benchmarks
"""

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Union
from threading import Lock

import librosa
import torch
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)
from peft import PeftModel

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

# Suppress the "System prompt modified, audio output may not work" warning
# We don't need audio output for ASR, so this warning is not relevant
warnings.filterwarnings(
    "ignore",
    message=".*System prompt modified.*audio output may not work.*"
)

# Also suppress at the transformers logging level (more aggressive)
import logging as transformers_logging
transformers_logging.getLogger("transformers").setLevel(transformers_logging.ERROR)

# Add a custom filter to catch this specific warning
class SystemPromptWarningFilter(logging.Filter):
    def filter(self, record):
        return "System prompt modified" not in record.getMessage()

# Apply filter to all loggers that might show this warning
for logger_name in ["transformers", "transformers.models.qwen2_audio", ""]:
    transformers_logging.getLogger(logger_name).addFilter(SystemPromptWarningFilter())


class Qwen25OmniTranscriber:
    """Transcriber for Qwen2.5-Omni models"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "auto",
        asr_prompt: str = "Transcribe the audio into text.",
        base_model: str = None,
        chunk_length_s: int = 30,
    ):
        """
        Initialize Qwen2.5-Omni transcriber
        
        Args:
            model_name: HuggingFace model name or path to checkpoint
            device: Device to run on ('cuda', 'cpu', or 'auto')
            asr_prompt: Prompt for ASR task
            base_model: Base model name (required if model_name is a LoRA checkpoint)
            chunk_length_s: Length of audio chunks in seconds (default: 30)
        """
        self.model_name = model_name
        self.asr_prompt = asr_prompt
        self.chunk_length_s = chunk_length_s
        self.lock = Lock()  # Thread lock for model access
        
        logger.info(f"Loading Qwen2.5-Omni model: {model_name}")
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
        
        # Check if this is a LoRA checkpoint
        model_path = Path(model_name)
        is_lora_checkpoint = False
        if model_path.exists():
            # Check for adapter files
            adapter_files = list(model_path.glob("adapter_*.bin")) + list(model_path.glob("adapter_*.safetensors"))
            is_lora_checkpoint = len(adapter_files) > 0 or (model_path / "adapter_config.json").exists()
        
        if is_lora_checkpoint:
            logger.info("📦 Detected LoRA checkpoint - loading with PEFT")
            if not base_model:
                # Try to infer base model from adapter config
                import json
                adapter_config_path = model_path / "adapter_config.json"
                if adapter_config_path.exists():
                    with open(adapter_config_path) as f:
                        adapter_config = json.load(f)
                        base_model = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Omni-7B")
                        logger.info(f"🔍 Inferred base model from adapter config: {base_model}")
                        
                        # Validate base model name and fix common issues
                        # Handle case where base_model_name_or_path has incorrect format
                        if "Qwen2.5-Omni-7B/thinker" in base_model or "Qwen2_5-Omni-7B/thinker" in base_model:
                            logger.warning(f"⚠️ Invalid base model name detected: {base_model}")
                            base_model = "Qwen/Qwen2.5-Omni-7B"
                            logger.info(f"✓ Corrected to valid base model: {base_model}")
                        elif "Qwen2.5-Omni" in base_model and not base_model.startswith("Qwen/"):
                            # Fix missing Qwen/ prefix
                            logger.warning(f"⚠️ Base model missing 'Qwen/' prefix: {base_model}")
                            base_model = "Qwen/Qwen2.5-Omni-7B"
                            logger.info(f"✓ Corrected to valid base model: {base_model}")
                else:
                    base_model = "Qwen/Qwen2.5-Omni-7B"
                    logger.warning(f"⚠️ Could not find adapter config, using default base model: {base_model}")
        
        logger.info("Loading Qwen2.5-Omni model...")
        
        # Try to use Flash-Attention 2 if available (optional optimization)
        try:
            import flash_attn
            use_flash_attn = True
            logger.info("Flash-Attention 2 detected - will enable for faster inference")
        except ImportError:
            use_flash_attn = False
            logger.info("Flash-Attention 2 not found - using default attention (slower but works)")
        
        # Load model (following official code pattern)
        if is_lora_checkpoint:
            # Load base model first
            logger.info(f"Loading base model: {base_model}")
            if use_flash_attn:
                base_model_obj = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    base_model,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
            else:
                base_model_obj = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    base_model,
                    torch_dtype="auto",
                    device_map="auto",
                )
            
            # Load LoRA adapters
            logger.info(f"Loading LoRA adapters from: {model_name}")
            self.model = PeftModel.from_pretrained(base_model_obj, model_name)
            
            # Merge and unload LoRA for faster inference (optional)
            logger.info("Merging LoRA weights for faster inference...")
            self.model = self.model.merge_and_unload()
            logger.info("✓ LoRA weights merged")
            
            # Load processor from base model
            self.processor = Qwen2_5OmniProcessor.from_pretrained(base_model)
        else:
            # Load regular model or merged checkpoint
            if use_flash_attn:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",  # Official code uses "auto"
                    device_map="auto",    # Official code uses "auto"
                    attn_implementation="flash_attention_2",
                )
            else:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",  # Official code uses "auto"
                    device_map="auto",    # Official code uses "auto"
                )
            
            # Load processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
        # Disable talker (audio output module) to save ~2GB GPU memory
        # We only need text output for ASR
        logger.info("Disabling talker module (audio output not needed for ASR)")
        self.model.disable_talker()
        
        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable CUDA optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled")
        
        logger.info("✓ Qwen2.5-Omni model loaded and optimized for inference!")
        logger.info(f"  Memory optimization: Talker disabled (~2GB saved)")
        logger.info(f"  Chunking: {self.chunk_length_s}s chunks for long audio")
        if is_lora_checkpoint:
            logger.info(f"  LoRA checkpoint: Merged for faster inference")
    
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
                
                # Process inputs (following official code exactly)
                inputs = self.processor(
                    text=text,
                    audio=[audio_array],  # List format as per official docs
                    images=None,
                    videos=None,
                    return_tensors="pt",
                    padding=True
                )
                
                # Store input_ids length BEFORE moving to device
                input_ids_length = inputs["input_ids"].shape[1]
                
                # Move to device
                inputs = inputs.to(self.model.device)
                
                # Estimate max_new_tokens based on audio length (roughly 3 tokens per second)
                audio_duration = len(audio_array) / 16000  # sr=16000
                max_new_tokens = max(128, int(audio_duration * 3))
                
                # Generate
                with torch.no_grad():
                    output = self.model.generate(
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
                    
                    # Handle both cases: with talker (tuple) or without talker (single value)
                    if isinstance(output, tuple):
                        text_ids, audio_output = output
                    else:
                        text_ids = output
                
                # Decode
                if hasattr(text_ids, 'sequences'):
                    result = self.processor.batch_decode(
                        text_ids.sequences[:, input_ids_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    result = self.processor.batch_decode(
                        text_ids[:, input_ids_length:],
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
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen2.5-Omni")
    parser.add_argument("--model", required=True, help="Qwen2.5-Omni model name or path to checkpoint")
    parser.add_argument("--test-dataset", required=True, help="Dataset name from registry (e.g., meso-malaya-test, seacrowd-asr-malcsc)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--asr-prompt", default="Transcribe the audio into text.", 
                       help="ASR prompt")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--base-model", help="Base model name (only needed for LoRA checkpoints if auto-detection fails)")
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
    transcriber = Qwen25OmniTranscriber(
        model_name=args.model,
        device=args.device,
        asr_prompt=args.asr_prompt,
        base_model=args.base_model,
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
