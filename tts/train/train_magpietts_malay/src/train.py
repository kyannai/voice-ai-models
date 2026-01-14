#!/usr/bin/env python3
"""
Fine-tune MagpieTTS on Malaysian-TTS dataset.

This script fine-tunes the pretrained MagpieTTS model on the prepared
Malaysian TTS dataset using NeMo framework with GPU acceleration.

Usage:
    python train.py --config configs/magpietts_malay.yaml
    python train.py --config configs/magpietts_malay.yaml --gpus 2
"""

import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

# Use NeMo's trainer for compatibility with NeMo models
try:
    from nemo.lightning import Trainer as NeMoTrainer
    USE_NEMO_TRAINER = True
except ImportError:
    NeMoTrainer = None
    USE_NEMO_TRAINER = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_training(config_path: str, gpus: int = 1, resume_from: str | None = None):
    """
    Setup and run training.
    
    Args:
        config_path: Path to training config YAML
        gpus: Number of GPUs to use
        resume_from: Path to checkpoint to resume from
    """
    try:
        from nemo.collections.tts.models import MagpieTTSModel
        from nemo.utils.exp_manager import exp_manager
    except ImportError:
        logger.error("NeMo is not installed. Install with: pip install nemo_toolkit[tts]")
        raise
    
    # Load config
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA device count: {num_gpus}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Handle -1 (all GPUs) or specific number
        if gpus == -1:
            gpus = num_gpus
            logger.info(f"Using all {gpus} available GPU(s)")
        else:
            logger.info(f"Using {gpus} GPU(s)")
    else:
        logger.warning("CUDA not available! Training will be slow on CPU.")
        gpus = 0
    
    # Setup trainer
    # Use NeMo's Trainer for compatibility with NeMo models
    trainer_config = config.get('trainer', {})
    
    trainer_kwargs = dict(
        devices=gpus if gpus > 0 else 1,
        accelerator='gpu' if gpus > 0 else 'cpu',
        max_epochs=trainer_config.get('max_epochs', 100),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        precision=trainer_config.get('precision', '16-mixed'),
        strategy=trainer_config.get('strategy', 'auto'),
        enable_checkpointing=False,  # exp_manager handles checkpointing
        logger=False,  # exp_manager handles logging
    )
    
    if USE_NEMO_TRAINER and NeMoTrainer is not None:
        logger.info("Using NeMo Trainer for compatibility")
        trainer = NeMoTrainer(**trainer_kwargs)
    else:
        logger.info("Using PyTorch Lightning Trainer")
        trainer = pl.Trainer(**trainer_kwargs)
    
    # Setup experiment manager
    exp_config = config.get('exp_manager', {})
    exp_manager(trainer, exp_config)
    
    # Load pretrained model or resume
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = MagpieTTSModel.load_from_checkpoint(resume_from)
    else:
        pretrained_model = config.get('pretrained_model', 'nvidia/magpie_tts_multilingual_357m')
        logger.info(f"Loading pretrained model: {pretrained_model}")
        model = MagpieTTSModel.from_pretrained(pretrained_model)
    
    # Get model config from our YAML
    model_config = config.get('model', {})
    train_manifest = model_config.get('train_manifest')
    val_manifest = model_config.get('val_manifest')
    batch_size = model_config.get('batch_size', 8)
    sample_rate = model_config.get('sample_rate', 22050)
    
    logger.info(f"Train manifest: {train_manifest}")
    logger.info(f"Val manifest: {val_manifest}")
    logger.info(f"Batch size: {batch_size}")
    
    # MagpieTTS has complex internal dataset setup that doesn't work with simple configs
    # We'll create our own dataloaders and attach them directly to the model
    # (OmegaConf is already imported at top of file)
    
    from nemo.collections.tts.data.dataset import TTSDataset
    from torch.utils.data import DataLoader
    import json
    
    # Update learning rate in model config
    lr = model_config.get('learning_rate', 2e-4)
    if hasattr(model.cfg, 'optim'):
        with open_dict(model.cfg):
            model.cfg.optim.lr = lr
        logger.info(f"Learning rate: {lr}")
    
    # Create a simple TTS dataset from our manifests
    logger.info("Creating training dataloader...")
    
    # Load manifests
    with open(train_manifest, 'r') as f:
        train_samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(train_samples)} training samples")
    
    with open(val_manifest, 'r') as f:
        val_samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(val_samples)} validation samples")
    
    # Create dataset class for MagpieTTS
    from torch.utils.data import Dataset
    import librosa
    import numpy as np
    
    # Get the model's tokenizer for text processing
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
    if tokenizer:
        logger.info(f"Using model's tokenizer: {type(tokenizer)}")
    else:
        logger.warning("Model tokenizer not found - will pass raw text")
    
    # MagpieTTS max sequence length (from model's causal mask)
    MAX_TEXT_TOKENS = 2000  # Leave some margin below 2048
    # Shorter audio = shorter text = less likely to exceed token limit
    # 10 seconds is typically ~100-200 tokens for phonemes
    MAX_DURATION_FOR_TRAINING = 15.0  # Reduced from 20s to avoid long sequences
    
    class MagpieTTSDataset(Dataset):
        """Dataset for MagpieTTS that produces correctly formatted batches."""
        
        def __init__(self, samples, sample_rate=22050, max_duration=MAX_DURATION_FOR_TRAINING, tokenizer=None, max_text_tokens=MAX_TEXT_TOKENS, max_text_chars=1500):
            # Filter by duration and text length
            filtered = []
            skipped_duration = 0
            skipped_text = 0
            for s in samples:
                if s.get('duration', 0) > max_duration:
                    skipped_duration += 1
                    continue
                # Also filter by raw text character length as a proxy for token count
                # Phonemes are roughly 1:1 with characters
                if len(s.get('text', '')) > max_text_chars:
                    skipped_text += 1
                    continue
                filtered.append(s)
            
            self.samples = filtered
            self.sample_rate = sample_rate
            self.tokenizer = tokenizer
            self.max_text_tokens = max_text_tokens
            
            if skipped_duration > 0 or skipped_text > 0:
                logger.info(f"  Filtered: {skipped_duration} too long (>{max_duration}s), {skipped_text} text too long (>{max_text_chars} chars)")
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load audio
            audio, sr = librosa.load(sample['audio_filepath'], sr=self.sample_rate)
            audio = torch.FloatTensor(audio)
            
            text = sample['text']
            language = sample.get('language', 'ms')
            
            # Tokenize text if tokenizer available
            if self.tokenizer:
                try:
                    # MagpieTTS tokenizer typically expects (text, language) for phoneme tokenizer
                    if hasattr(self.tokenizer, 'encode'):
                        text_tokens = self.tokenizer.encode(text)
                    else:
                        text_tokens = list(text.encode('utf-8'))  # Fallback to byte encoding
                    text_tokens = torch.LongTensor(text_tokens)
                except Exception as e:
                    # Fallback: convert characters to indices
                    text_tokens = torch.LongTensor([ord(c) % 256 for c in text])
            else:
                # Simple character encoding fallback
                text_tokens = torch.LongTensor([ord(c) % 256 for c in text])
            
            # Safety truncation (should rarely happen after pre-filtering)
            if len(text_tokens) > self.max_text_tokens:
                text_tokens = text_tokens[:self.max_text_tokens]
            
            return {
                'audio': audio,
                'audio_len': len(audio),
                'text': text_tokens,
                'text_len': len(text_tokens),
                'raw_text': text,
                'speaker': sample.get('speaker', 0),
                'language': language,
            }
    
    def collate_fn(batch):
        """Collate function for MagpieTTS batch format."""
        # Find max lengths
        max_audio_len = max(item['audio'].shape[0] for item in batch)
        max_text_len = max(item['text'].shape[0] for item in batch)
        
        # Pad and collect
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        speakers = []
        languages = []
        
        for item in batch:
            # Pad audio
            audio = item['audio']
            audio_pad = max_audio_len - len(audio)
            if audio_pad > 0:
                audio = torch.nn.functional.pad(audio, (0, audio_pad))
            audios.append(audio)
            audio_lens.append(item['audio_len'])
            
            # Pad text
            text = item['text']
            text_pad = max_text_len - len(text)
            if text_pad > 0:
                text = torch.nn.functional.pad(text, (0, text_pad), value=0)
            texts.append(text)
            text_lens.append(item['text_len'])
            
            speakers.append(item['speaker'])
            languages.append(item['language'])
        
        return {
            'audio': torch.stack(audios),
            'audio_lens': torch.LongTensor(audio_lens),
            'text': torch.stack(texts),
            'text_lens': torch.LongTensor(text_lens),
            'speaker': torch.LongTensor(speakers),
            'language': languages,
        }
    
    train_dataset = MagpieTTSDataset(train_samples, sample_rate=sample_rate, tokenizer=tokenizer)
    val_dataset = MagpieTTSDataset(val_samples, sample_rate=sample_rate, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # Attach dataloaders to model
    model._train_dl = train_loader
    model._validation_dl = val_loader
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model)
    
    logger.info("Training complete!")
    
    # Save final model
    output_dir = exp_config.get('exp_dir', 'experiments')
    final_model_path = Path(output_dir) / 'final_model.nemo'
    model.save_to(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MagpieTTS on Malaysian-TTS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python train.py --config configs/magpietts_malay.yaml

    # Train with multiple GPUs
    python train.py --config configs/magpietts_malay.yaml --gpus 4

    # Resume from checkpoint
    python train.py --config configs/magpietts_malay.yaml --resume checkpoints/last.ckpt
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    setup_training(
        config_path=args.config,
        gpus=args.gpus,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
