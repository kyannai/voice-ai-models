#!/usr/bin/env python3
"""
Fine-tuning script for Qwen2.5-Omni on Malay ASR
Uses LoRA for parameter-efficient fine-tuning

Key differences from Qwen2-Audio:
- Uses Qwen2_5OmniForConditionalGeneration
- Uses Qwen2_5OmniProcessor
- Disables talker module to save ~2GB GPU memory
- Flash-Attention 2 support (optional)
"""

import os
import json
import yaml
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Try to import unsloth FIRST (before transformers/peft for optimal performance)
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

import torch
import librosa
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*System prompt modified.*audio output may not work.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress "System prompt modified" warnings (irrelevant for ASR since talker is disabled)
class SystemPromptWarningFilter(logging.Filter):
    def filter(self, record):
        return "System prompt modified" not in record.getMessage()

# Apply filter to root logger (where Qwen warnings come from)
logging.getLogger().addFilter(SystemPromptWarningFilter())


@dataclass
class Qwen25OmniDataCollator:
    """Custom data collator for Qwen2.5-Omni that handles audio inputs properly"""
    processor: Any
    sampling_rate: int = 16000
    _call_count: int = 0  # Track number of calls
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}
        
        # Extract data
        texts = [f["text"] for f in features]
        audio_paths = [f["audio_path"] for f in features]
        transcriptions = [f["transcription"] for f in features]
        
        # Load audio arrays (Qwen2.5-Omni processor expects audio arrays, not paths)
        audio_arrays = []
        for path in audio_paths:
            audio_array, sr = librosa.load(path, sr=self.sampling_rate)
            audio_arrays.append(audio_array)
        
        # Process with audio arrays - Qwen2.5-Omni style
        inputs = self.processor(
            text=texts,
            audio=audio_arrays,  # Audio arrays as lists
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Debug logging - only log every 1000 batches to reduce noise
        self._call_count += 1
        if self._call_count <= 5 or self._call_count % 1000 == 0:
            logger.debug(f"Batch {self._call_count}: size={len(audio_paths)}, input_shape={inputs['input_ids'].shape}")
        
        # Create labels from input_ids
        labels = inputs["input_ids"].clone()
        
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        
        return inputs


class Qwen25OmniDataset(torch.utils.data.Dataset):
    """Custom dataset for Qwen2.5-Omni fine-tuning"""
    
    def __init__(
        self,
        data: List[Dict],
        processor,
        asr_instruction: str = "Transcribe:",
        max_audio_length: float = 30.0,
        sampling_rate: int = 16000
    ):
        self.data = data
        self.processor = processor
        self.asr_instruction = asr_instruction
        self.max_audio_length = max_audio_length
        self.sampling_rate = sampling_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Qwen2.5-Omni uses conversation format
        # For training, we format with the instruction and expected response
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": sample['full_audio_path']},
                    {"type": "text", "text": self.asr_instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample['text']},
                ],
            }
        ]
        
        # Apply chat template
        training_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,  # Don't add prompt for training
            tokenize=False
        )
        
        return {
            "text": training_text,
            "audio_path": sample['full_audio_path'],
            "transcription": sample['text'],
        }


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: Dict) -> tuple:
    """Load training and validation data"""
    # Check if separate train/val files are specified in config
    if 'val_json' in config['data'] and config['data'].get('val_json'):
        # Load from separate files
        train_file = Path(config['data']['train_json'])
        val_file = Path(config['data']['val_json'])
        
        logger.info(f"Loading train data from {train_file}")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        logger.info(f"Loading val data from {val_file}")
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        # Fix paths if needed (make absolute)
        audio_base_dir = config['data'].get('audio_base_dir')
        if audio_base_dir:
            base_dir = Path(audio_base_dir)
        else:
            base_dir = train_file.parent
        
        for sample in train_data + val_data:
            if 'audio_path' in sample:
                audio_path = Path(sample['audio_path'])
                if not audio_path.is_absolute():
                    sample['full_audio_path'] = str(base_dir / audio_path)
                else:
                    sample['full_audio_path'] = str(audio_path)
            elif 'full_audio_path' not in sample:
                audio_dir = config['data'].get('train_audio_dir', base_dir)
                sample['full_audio_path'] = str(Path(audio_dir) / sample['audio_file'])
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data
    else:
        # Load single file and split
        data_file = Path(config['data']['data_json'])
        logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            all_data = json.load(f)
        
        # Split into train/val
        split_ratio = config['data'].get('val_split', 0.1)
        split_idx = int(len(all_data) * (1 - split_ratio))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data


def prepare_model_and_processor(config: Dict):
    """Load model and processor"""
    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)
    
    logger.info(f"Loading Qwen2.5-Omni model: {model_name}")
    
    if USE_UNSLOTH:
        logger.info("Using Unsloth for optimized training")
        return prepare_model_with_unsloth(config)
    else:
        logger.info("Unsloth not available, using standard PEFT (may have compatibility issues)")
        return prepare_model_with_peft(config)


def prepare_model_with_unsloth(config: Dict):
    """Load model using Unsloth (optimized, better compatibility)"""
    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)
    lora_config = config.get('lora', {})
    quant_config = config.get('quantization', {})
    
    # Load processor first
    logger.info("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # Prepare Unsloth model parameters
    max_seq_length = 2048  # Reasonable for audio + text
    dtype = None  # Auto-detect
    load_in_4bit = quant_config.get('load_in_4bit', True)
    
    logger.info(f"Loading model with Unsloth (4-bit: {load_in_4bit})...")
    
    # Load model with Unsloth
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
    )
    
    # Disable talker module
    logger.info("Disabling talker module to save ~2GB GPU memory")
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    elif hasattr(model, 'model') and hasattr(model.model, 'disable_talker'):
        model.model.disable_talker()
    
    # Apply LoRA with Unsloth
    if lora_config.get('enabled', True):
        logger.info("Applying LoRA with Unsloth")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get('bias', "none"),
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
            random_state=42,
        )
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    return model, processor


def prepare_model_with_peft(config: Dict):
    """Load model using standard PEFT (fallback, may have issues with new models)"""
    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)
    
    # Check for Flash-Attention 2
    try:
        import flash_attn
        use_flash_attn = True
        logger.info("Flash-Attention 2 detected - will enable for faster training")
    except ImportError:
        use_flash_attn = False
        logger.info("Flash-Attention 2 not found - using default attention")
    
    # Load processor
    logger.info("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # Load model with quantization if enabled
    quant_config = config.get('quantization', {})
    model_kwargs = {
        'cache_dir': cache_dir,
        'torch_dtype': "auto",
        'device_map': "auto",
    }
    
    if use_flash_attn:
        model_kwargs['attn_implementation'] = "flash_attention_2"
    
    if quant_config.get('enabled', False):
        if quant_config.get('load_in_4bit', False):
            logger.info("Loading in 4-bit mode")
            model_kwargs['load_in_4bit'] = True
            model_kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16
            model_kwargs['bnb_4bit_use_double_quant'] = True
            model_kwargs['bnb_4bit_quant_type'] = "nf4"
        elif quant_config.get('load_in_8bit', False):
            logger.info("Loading in 8-bit mode")
            model_kwargs['load_in_8bit'] = True
    
    logger.info("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Disable talker module to save ~2GB GPU memory (ASR doesn't need audio output)
    logger.info("Disabling talker module to save ~2GB GPU memory")
    model.disable_talker()
    
    # Enable gradient checkpointing if specified
    if config['training'].get('gradient_checkpointing', False):
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training if quantization is enabled
    if quant_config.get('enabled', False):
        logger.info("Preparing model for k-bit training")
        try:
            model = prepare_model_for_kbit_training(model)
        except NotImplementedError:
            logger.info("Model doesn't support standard k-bit preparation, using manual setup")
            for param in model.parameters():
                param.requires_grad = False
    
    # Apply LoRA if enabled
    lora_config = config.get('lora', {})
    if lora_config.get('enabled', True):
        logger.info("Applying LoRA configuration")
        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', "none"),
            task_type=lora_config.get('task_type', "CAUSAL_LM")
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, processor


def print_config_summary(config: Dict, train_data: List, val_data: List):
    """Print a summary of the training configuration"""
    logger.info("=" * 70)
    logger.info("📋 TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("🗂️ Dataset:")
    logger.info(f"  Train samples: {len(train_data):,}")
    logger.info(f"  Val samples: {len(val_data):,}")
    logger.info(f"  Audio max length: {config['data']['max_audio_length']}s")
    logger.info("")
    logger.info("🤖 Model:")
    logger.info(f"  Base model: {config['model']['name']}")
    logger.info(f"  LoRA enabled: {config['lora'].get('enabled', True)}")
    logger.info(f"  LoRA rank: {config['lora'].get('r', 16)}")
    logger.info(f"  LoRA alpha: {config['lora'].get('lora_alpha', 32)}")
    logger.info(f"  Quantization: {'4-bit' if config['quantization'].get('load_in_4bit') else '8-bit' if config['quantization'].get('load_in_8bit') else 'None'}")
    logger.info(f"  Talker module: Disabled (~2GB saved)")
    logger.info("")
    logger.info("🎯 Training Hyperparameters:")
    logger.info(f"  Batch size per device: {config['training']['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"  Warmup steps: {config['training'].get('warmup_steps', 0)}")
    logger.info(f"  Optimizer: {config['training'].get('optim', 'adamw')}")
    logger.info(f"  Precision: {'BF16' if config['training'].get('bf16') else 'FP16' if config['training'].get('fp16') else 'FP32'}")
    logger.info(f"  Gradient checkpointing: {config['training'].get('gradient_checkpointing', False)}")
    logger.info("")
    logger.info("💾 Checkpointing:")
    logger.info(f"  Output dir: {config['training']['output_dir']}")
    logger.info(f"  Save every: {config['training'].get('save_steps', 500)} steps")
    logger.info(f"  Eval every: {config['training'].get('eval_steps', 500)} steps")
    logger.info(f"  Keep best: {config['training'].get('save_total_limit', 3)} checkpoints")
    logger.info("")
    logger.info("⚡ DataLoader:")
    logger.info(f"  Num workers: {config['training'].get('dataloader_num_workers', 4)}")
    logger.info(f"  Pin memory: {config['training'].get('dataloader_pin_memory', True)}")
    logger.info("")
    
    # Calculate expected training steps
    batch_size = config['training']['per_device_train_batch_size']
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch = len(train_data) // effective_batch_size
    total_steps = steps_per_epoch * config['training']['num_train_epochs']
    
    logger.info("📊 Expected Training:")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")


def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Load data
    train_data, val_data = load_data(config)
    
    # Prepare model and processor
    model, processor = prepare_model_and_processor(config)
    
    # Print configuration summary
    print_config_summary(config, train_data, val_data)
    
    # Create datasets
    asr_instruction = config['prompt'].get('asr_instruction', "Transcribe this Malay audio to text:")
    
    train_dataset = Qwen25OmniDataset(
        data=train_data,
        processor=processor,
        asr_instruction=asr_instruction,
        max_audio_length=config['data']['max_audio_length'],
        sampling_rate=config['data']['sampling_rate']
    )
    
    val_dataset = Qwen25OmniDataset(
        data=val_data,
        processor=processor,
        asr_instruction=asr_instruction,
        max_audio_length=config['data']['max_audio_length'],
        sampling_rate=config['data']['sampling_rate']
    )
    
    # Create data collator
    data_collator = Qwen25OmniDataCollator(
        processor=processor,
        sampling_rate=config['data']['sampling_rate']
    )
    
    # Training arguments
    training_config = config['training']
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', training_config['per_device_train_batch_size']),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        learning_rate=training_config['learning_rate'],
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', "paged_adamw_32bit"),
        warmup_steps=training_config.get('warmup_steps', 100),
        logging_dir=f"{training_config['output_dir']}/logs",
        logging_steps=training_config.get('logging_steps', 10),
        eval_strategy="steps",
        eval_steps=training_config.get('eval_steps', 500),
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        weight_decay=training_config.get('weight_decay', 0.01),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
        report_to=["wandb"] if config.get('wandb', {}).get('enabled', False) else ["tensorboard"],
        run_name=config.get('wandb', {}).get('run_name', 'qwen25omni-finetune'),
        remove_unused_columns=False,  # Keep all columns for custom data collator
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    logger.info("")
    logger.info("🚀 Starting training...")
    logger.info("")
    
    # Train
    trainer.train()
    
    # Save final model
    final_model_dir = Path(training_config['output_dir']) / "final_model"
    logger.info(f"💾 Saving final model to {final_model_dir}")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    
    logger.info("")
    logger.info("✅ Training completed successfully!")
    logger.info(f"📁 Model saved to: {final_model_dir}")
    logger.info("")


if __name__ == "__main__":
    main()

