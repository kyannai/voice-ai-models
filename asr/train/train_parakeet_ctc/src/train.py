#!/usr/bin/env python3
"""
Training script for Parakeet CTC multilingual model.

This script fine-tunes the modified Parakeet CTC model with expanded vocabulary
on multilingual (Malay + Chinese) data.

Usage:
    python src/train.py --config configs/parakeet_multilingual.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from omegaconf import OmegaConf, DictConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[NeMo I %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ['model', 'data', 'training']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    logger.info("✓ Configuration validated successfully")
    return config


def setup_model(config: Dict[str, Any], trainer) -> Any:
    """Load and configure the CTC model."""
    import nemo.collections.asr as nemo_asr
    
    model_path = config['model']['name']
    
    if model_path.endswith('.nemo'):
        logger.info(f"Loading model from .nemo file: {model_path}")
        
        # Check if this is an adapted model with multilingual tokenizer
        model_dir = Path(model_path).parent
        mapping_path = model_dir / 'token_mapping.json'
        
        if mapping_path.exists():
            logger.info(f"Found token mapping: {mapping_path}")
            import json
            with open(mapping_path) as f:
                mapping = json.load(f)
            logger.info(f"  Old vocab: {mapping['old_vocab_size']}")
            logger.info(f"  New vocab: {mapping['new_vocab_size']}")
            logger.info(f"  Overlap: {mapping['overlap_percentage']}%")
        
        # Load as BPE model - the tokenizer files should be inside the .nemo
        # Use map_location to avoid GPU memory issues during loading
        try:
            model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                model_path,
                trainer=trainer,
                map_location='cpu',
                strict=False,
            )
            logger.info("✓ Loaded model (strict=False)")
        except Exception as e:
            logger.warning(f"Failed with BPE model, trying generic: {e}")
            model = nemo_asr.models.EncDecCTCModel.restore_from(
                model_path,
                trainer=trainer,
                map_location='cpu',
                strict=False,
            )
            logger.info("✓ Loaded as generic CTC model")
        
        # Move model to GPU after loading
        if torch.cuda.is_available():
            model = model.cuda()
            
    elif model_path.startswith('nvidia/'):
        logger.info(f"Loading pretrained model: {model_path}")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_path,
            trainer=trainer
        )
    else:
        logger.info(f"Loading model from path: {model_path}")
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            model_path,
            trainer=trainer
        )
    
    # Log model info
    if hasattr(model, 'tokenizer'):
        logger.info(f"Tokenizer vocab size: {model.tokenizer.vocab_size}")
    if hasattr(model, 'decoder'):
        decoder_classes = getattr(model.decoder, 'num_classes_with_blank', None)
        if decoder_classes:
            logger.info(f"Decoder classes: {decoder_classes}")
    
    # Enable gradient checkpointing if requested
    if config['model'].get('gradient_checkpointing', False):
        logger.info("Enabling gradient checkpointing...")
        if hasattr(model.encoder, 'set_gradient_checkpointing'):
            model.encoder.set_gradient_checkpointing(True)
        elif hasattr(model.encoder, 'gradient_checkpointing'):
            model.encoder.gradient_checkpointing = True
        else:
            # Try to enable via layers
            for layer in model.encoder.layers if hasattr(model.encoder, 'layers') else []:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
        logger.info("✓ Gradient checkpointing enabled")
    
    return model


def setup_data_loaders(model, config: Dict[str, Any]) -> None:
    """Configure training and validation data loaders."""
    
    train_config = {
        'manifest_filepath': config['data']['train_manifest'],
        'sample_rate': config['data'].get('sampling_rate', 16000),
        'batch_size': config['training']['per_device_train_batch_size'],
        'shuffle': True,
        'num_workers': config['training'].get('dataloader_num_workers', 4),
        'pin_memory': config['training'].get('dataloader_pin_memory', True),
        'max_duration': config['data'].get('max_audio_length', 20.0),
        'min_duration': config['data'].get('min_audio_length', 0.1),
        'trim': True,
    }
    
    # Add max samples limit if specified
    if config['data'].get('max_samples'):
        train_config['max_utts'] = config['data']['max_samples']
    
    val_config = {
        'manifest_filepath': config['data']['val_manifest'],
        'sample_rate': config['data'].get('sampling_rate', 16000),
        'batch_size': config['training']['per_device_eval_batch_size'],
        'shuffle': False,
        'num_workers': config['training'].get('dataloader_num_workers', 4),
        'pin_memory': config['training'].get('dataloader_pin_memory', True),
        'max_duration': config['data'].get('max_audio_length', 20.0),
        'min_duration': config['data'].get('min_audio_length', 0.1),
    }
    
    if config['data'].get('max_val_samples'):
        val_config['max_utts'] = config['data']['max_val_samples']
    
    logger.info("Setting up training data loader...")
    model.setup_training_data(train_data_config=OmegaConf.create(train_config))
    
    logger.info("Setting up validation data loader...")
    model.setup_validation_data(val_data_config=OmegaConf.create(val_config))


def setup_optimizer(model, config: Dict[str, Any]) -> None:
    """Configure optimizer and learning rate scheduler."""
    
    optim_config = {
        'name': config['training'].get('optimizer', 'adamw'),
        'lr': config['training']['learning_rate'],
        'betas': [0.9, 0.999],
        'weight_decay': config['training'].get('weight_decay', 0.0),
        'sched': {
            'name': config['training'].get('scheduler', 'CosineAnnealing'),
            'warmup_steps': config['training'].get('warmup_steps', 1000),
            'warmup_ratio': config['training'].get('warmup_ratio', None),
            'min_lr': config['training'].get('min_learning_rate', 1e-7),
        }
    }
    
    model.setup_optimization(optim_config=DictConfig(optim_config))


def create_trainer(config: Dict[str, Any]) -> Any:
    """Create PyTorch Lightning trainer."""
    # NeMo requires lightning.pytorch, not pytorch_lightning
    try:
        from lightning.pytorch import Trainer
    except ImportError:
        from pytorch_lightning import Trainer
    import lightning.pytorch as pl
    
    trainer_config = {
        'devices': config['training'].get('num_gpus', 1),
        'num_nodes': config['training'].get('num_nodes', 1),
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'max_epochs': int(config['training']['num_train_epochs']),
        'max_steps': int(config['training'].get('max_steps', -1)),
        'val_check_interval': int(config['training'].get('eval_steps', 1000)),
        'log_every_n_steps': int(config['training'].get('logging_steps', 50)),
        'accumulate_grad_batches': int(config['training'].get('gradient_accumulation_steps', 1)),
        'gradient_clip_val': config['training'].get('max_grad_norm', 1.0),
        'enable_checkpointing': not config['training'].get('disable_checkpointing', False),
        'logger': False,  # We'll use exp_manager
        'enable_progress_bar': True,
    }
    
    # Set precision
    if config['training'].get('bf16', False):
        trainer_config['precision'] = 'bf16-mixed'
    elif config['training'].get('fp16', False):
        trainer_config['precision'] = '16-mixed'
    else:
        trainer_config['precision'] = '32-true'
    
    trainer = Trainer(**trainer_config)
    return trainer


def setup_experiment_manager(trainer, config: Dict[str, Any]) -> None:
    """Set up NeMo experiment manager for logging and checkpointing."""
    from nemo.utils.exp_manager import exp_manager
    
    exp_config = {
        'exp_dir': config['training']['output_dir'],
        'name': config['training']['run_name'],
        'create_tensorboard_logger': True,
        'create_checkpoint_callback': not config['training'].get('disable_checkpointing', False),
        'checkpoint_callback_params': {
            'monitor': 'val_wer',
            'mode': 'min',
            'save_top_k': config['training'].get('save_total_limit', 3),
            'every_n_train_steps': config['training'].get('save_steps', 1000),
            'filename': 'parakeet-ctc-{epoch:02d}-{step}-{val_wer:.4f}',
            'save_last': False,
        },
        'resume_if_exists': config['training'].get('resume_from_checkpoint', False),
        'resume_ignore_no_checkpoint': True,
    }
    
    exp_manager(trainer, OmegaConf.create(exp_config))


def print_model_info(model, config: Dict[str, Any]) -> None:
    """Print model and training configuration."""
    print("\n" + "=" * 70)
    print("🚀 NVIDIA Parakeet CTC Training Configuration")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Output: {config['training']['output_dir']}")
    print(f"Train Manifest: {config['data']['train_manifest']}")
    print(f"Val Manifest: {config['data']['val_manifest']}")
    
    if config['data'].get('max_samples'):
        print(f"Max Train Samples: {config['data']['max_samples']}")
    if config['data'].get('max_val_samples'):
        print(f"Max Val Samples: {config['data']['max_val_samples']}")
    
    print(f"\n📊 Training Parameters:")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    eff_batch = config['training']['per_device_train_batch_size'] * config['training'].get('gradient_accumulation_steps', 1)
    print(f"  Effective Batch: {eff_batch}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Max Audio Length: {config['data'].get('max_audio_length', 20.0)}s")
    
    if torch.cuda.is_available():
        print(f"\n🔧 Hardware:")
        print(f"  Device: CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Parakeet CTC model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    config_save_path = output_dir / 'training_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved training config to {config_save_path}")
    
    # Create trainer
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Creating PyTorch Lightning Trainer")
    logger.info("=" * 70)
    trainer = create_trainer(config)
    
    # Setup experiment manager
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Setting up Experiment Manager")
    logger.info("=" * 70)
    setup_experiment_manager(trainer, config)
    
    # Load model
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Loading Base Model")
    logger.info("=" * 70)
    model = setup_model(config, trainer)
    
    # Print model info
    print_model_info(model, config)
    
    # Freeze encoder if requested
    if config['training'].get('freeze_encoder', False):
        logger.info("Freezing encoder layers...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("✓ Encoder frozen")
    
    # Disable spec augmentation if in overfit mode
    if config['training'].get('disable_spec_augment', False):
        if hasattr(model.cfg, 'spec_augment') and model.cfg.spec_augment is not None:
            model.cfg.spec_augment.freq_masks = 0
            model.cfg.spec_augment.time_masks = 0
        logger.info("🚫 Disabled spec augmentation (overfit mode)")
    
    # Setup data loaders
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Setting up Dataloaders")
    logger.info("=" * 70)
    setup_data_loaders(model, config)
    
    # Setup optimizer
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Setting up Optimizer")
    logger.info("=" * 70)
    setup_optimizer(model, config)
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Starting Training")
    logger.info("=" * 70)
    logger.info("Training will now begin. Monitor progress in TensorBoard:")
    logger.info(f"  tensorboard --logdir {config['training']['output_dir']}")
    logger.info("=" * 70 + "\n")
    
    trainer.fit(model)
    
    # Save final model
    if not config['training'].get('disable_checkpointing', False):
        final_path = output_dir / 'final_model.nemo'
        model.save_to(str(final_path))
        logger.info(f"✓ Final model saved to {final_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
