#!/usr/bin/env python3
"""
Fine-tune OpenAI Whisper on Malaysian STT data.

Uses HuggingFace Transformers Seq2SeqTrainer for training.

Features:
- WhisperProcessor for feature extraction and tokenization
- WhisperForConditionalGeneration as the base model
- WER (Word Error Rate) evaluation metric
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for memory efficiency

Usage:
    python train.py --config configs/whisper_malay.yaml
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# PEFT for LoRA training
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA training disabled. Install with: pip install peft")

import librosa


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration file for required fields."""
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Check data paths
    if not Path(config['data']['train_manifest']).exists():
        raise FileNotFoundError(f"Train manifest not found: {config['data']['train_manifest']}")
    
    if not Path(config['data']['val_manifest']).exists():
        raise FileNotFoundError(f"Validation manifest not found: {config['data']['val_manifest']}")
    
    print("✓ Configuration validated successfully")


class WhisperASRDataset(Dataset):
    """Dataset for Whisper ASR fine-tuning.
    
    Loads audio files and transcriptions from NeMo-style manifest files.
    Each line in the manifest is a JSON object with:
    - audio_filepath: path to audio file
    - text: transcription
    - duration: audio duration in seconds
    """
    
    def __init__(
        self,
        manifest_path: str,
        processor: WhisperProcessor,
        max_samples: Optional[int] = None,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.1,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
    ):
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.sampling_rate = sampling_rate
        self.language = language
        self.task = task
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    # Filter by duration
                    duration = sample.get('duration', 0)
                    if min_audio_length <= duration <= max_audio_length:
                        self.samples.append(sample)
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples):,} samples from {manifest_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        audio_path = sample['audio_filepath']
        text = sample['text']
        
        # Load audio
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return a dummy sample
            waveform = torch.zeros(self.sampling_rate)  # 1 second of silence
            text = ""
        
        # Extract features
        input_features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features[0]
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,  # Whisper max tokens
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper ASR.
    
    Pads input features and labels to the same length within a batch.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        # Labels should be padded with -100 (ignored in loss computation)
        max_label_length = max(len(l) for l in labels)
        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            padded_label = torch.cat([
                label,
                torch.full((padding_length,), -100, dtype=label.dtype)
            ])
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.stack(padded_labels)
        
        return batch


def compute_metrics(pred, tokenizer, metric):
    """Compute WER metric for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Malaysian STT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py --config configs/whisper_malay.yaml
    python train.py --config configs/whisper_malay.yaml --resume
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load and validate config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Print training info
    print("\n" + "=" * 70)
    print("🚀 Whisper Fine-tuning Configuration")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Output: {config['training']['output_dir']}")
    print(f"Train Manifest: {config['data']['train_manifest']}")
    print(f"Val Manifest: {config['data']['val_manifest']}")
    print(f"\n📊 Training Parameters:")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
    effective_batch = (
        config['training']['per_device_train_batch_size'] * 
        config['training']['gradient_accumulation_steps']
    )
    print(f"  Effective Batch: {effective_batch}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"\n🔧 Hardware:")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("=" * 70 + "\n")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / 'training_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved training config to {config_save_path}")
    
    # Load model and processor
    print(f"\nLoading model: {config['model']['name']}")
    model_name = config['model']['name']
    
    processor = WhisperProcessor.from_pretrained(model_name)
    # Load model in float32 - let the Trainer handle mixed precision via bf16=True
    # This avoids dtype mismatch during generation/evaluation
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )
    
    # Disable cache for training (required for gradient checkpointing compatibility)
    # Gradient checkpointing will be enabled by the Trainer via TrainingArguments
    model.config.use_cache = False
    
    if config['training'].get('gradient_checkpointing', False):
        print("Gradient checkpointing will be enabled by Trainer...")
    
    # Set language and task for generation
    language = config['model'].get('language')
    task = config['model'].get('task', 'transcribe')
    
    if language:
        # Set forced decoder IDs for language
        model.generation_config.language = language
        model.generation_config.task = task
        print(f"Set generation language: {language}, task: {task}")
    
    # Apply LoRA if configured
    lora_config = config.get('lora', {})
    if lora_config.get('enabled', False):
        if not PEFT_AVAILABLE:
            raise RuntimeError("LoRA is enabled but PEFT is not installed. Run: pip install peft")
        
        print("\n🔧 Applying LoRA configuration...")
        
        # Freeze encoder if specified
        if lora_config.get('freeze_encoder', True):
            print("  Freezing encoder (acoustic features)...")
            for param in model.model.encoder.parameters():
                param.requires_grad = False
        
        # Configure LoRA
        # Note: Don't use TaskType.SEQ_2_SEQ_LM for Whisper - it expects input_ids
        # Whisper uses input_features (audio), so we omit task_type
        peft_config = LoraConfig(
            r=lora_config.get('r', 32),
            lora_alpha=lora_config.get('lora_alpha', 64),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            modules_to_save=lora_config.get('modules_to_save'),
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        print(f"  LoRA rank (r): {lora_config.get('r', 32)}")
        print(f"  LoRA alpha: {lora_config.get('lora_alpha', 64)}")
        print(f"  Target modules: {lora_config.get('target_modules')}")
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")
        print("  ✓ LoRA applied successfully\n")
    
    # Load datasets
    print("\nLoading datasets...")
    max_samples = config['data'].get('max_samples')
    if max_samples == -1:
        max_samples = None
    
    max_val_samples = config['data'].get('max_val_samples')
    if max_val_samples == -1:
        max_val_samples = None
    
    train_dataset = WhisperASRDataset(
        manifest_path=config['data']['train_manifest'],
        processor=processor,
        max_samples=max_samples,
        max_audio_length=config['data'].get('max_audio_length', 30.0),
        min_audio_length=config['data'].get('min_audio_length', 0.1),
        sampling_rate=config['data'].get('sampling_rate', 16000),
        language=language,
        task=task,
    )
    
    val_dataset = WhisperASRDataset(
        manifest_path=config['data']['val_manifest'],
        processor=processor,
        max_samples=max_val_samples,
        max_audio_length=config['data'].get('max_audio_length', 30.0),
        min_audio_length=config['data'].get('min_audio_length', 0.1),
        sampling_rate=config['data'].get('sampling_rate', 16000),
        language=language,
        task=task,
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Create compute_metrics function with closure
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor.tokenizer, wer_metric)
    
    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        run_name=config['training'].get('run_name', 'whisper-finetuning'),
        
        # Training parameters
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=config['training'].get('max_steps', -1),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        
        # Optimizer
        optim=config['training'].get('optimizer', 'adamw_torch'),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        
        # Scheduler
        lr_scheduler_type=config['training'].get('lr_scheduler_type', 'linear'),
        warmup_steps=config['training'].get('warmup_steps', 500),
        warmup_ratio=config['training'].get('warmup_ratio', 0.0),
        
        # Precision
        fp16=config['training'].get('fp16', False),
        bf16=config['training'].get('bf16', True),
        bf16_full_eval=config['training'].get('bf16', True),  # Keep bf16 during eval to avoid dtype mismatch
        
        # Logging
        logging_steps=config['training'].get('logging_steps', 50),
        logging_dir=str(output_dir / 'logs'),
        report_to=["tensorboard"],
        
        # Evaluation
        eval_strategy=config['training'].get('evaluation_strategy', 'steps'),
        eval_steps=config['training'].get('eval_steps', 1000),
        
        # Saving
        save_strategy=config['training'].get('save_strategy', 'steps'),
        save_steps=config['training'].get('save_steps', 1000),
        save_total_limit=config['training'].get('save_total_limit', 3),
        load_best_model_at_end=config['training'].get('load_best_model_at_end', True),
        metric_for_best_model=config['training'].get('metric_for_best_model', 'wer'),
        greater_is_better=config['training'].get('greater_is_better', False),
        
        # DataLoader
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 4),
        dataloader_pin_memory=config['training'].get('dataloader_pin_memory', True),
        remove_unused_columns=config['training'].get('remove_unused_columns', False),
        
        # Generation settings for eval
        predict_with_generate=True,
        generation_max_length=config.get('generation', {}).get('max_new_tokens', 448),
        generation_num_beams=config.get('generation', {}).get('num_beams', 1),
        
        # Gradient checkpointing - use_reentrant=False is required for Whisper
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['training'].get('gradient_checkpointing', False) else None,
    )
    
    # Handle resume from checkpoint
    resume_from_checkpoint = config['training'].get('resume_from_checkpoint')
    if args.resume or resume_from_checkpoint == "latest":
        # Find latest checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            resume_from_checkpoint = str(latest_checkpoint)
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            resume_from_checkpoint = None
            print("No checkpoints found, starting from scratch")
    elif resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor,
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print("Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {output_dir / 'logs'}")
    print("=" * 70 + "\n")
    
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        print("\nSaving final model...")
        final_model_path = output_dir / "final"
        trainer.save_model(str(final_model_path))
        processor.save_pretrained(str(final_model_path))
        
        print("\n" + "=" * 70)
        print("✅ Training Completed Successfully!")
        print("=" * 70)
        print(f"📁 Outputs saved to: {output_dir}")
        print(f"📄 Final model: {final_model_path}")
        print(f"📊 TensorBoard logs: tensorboard --logdir {output_dir / 'logs'}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Training Failed")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
