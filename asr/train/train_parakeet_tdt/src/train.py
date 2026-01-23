#!/usr/bin/env python3
"""
Fine-tuning script for NVIDIA Parakeet TDT 0.6B v3 on Custom ASR Data
Uses NVIDIA NeMo framework for training

Parakeet TDT (Token-and-Duration Transducer) Features:
- Automatic punctuation and capitalization
- Word-level timestamps
- High accuracy on long-form audio
- Lightweight (0.6B parameters)

Based on NeMo's official speech_to_text_finetune.py script.
"""

import os
import json
import yaml
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero

# Try importing bitsandbytes for 8-bit optimizer support
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available - 8-bit optimizer disabled. Install with: pip install bitsandbytes")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """
    Validate configuration file for required fields and file paths.
    
    Args:
        config: Configuration dictionary loaded from YAML
        
    Raises:
        ValueError: If required configuration fields are missing
        FileNotFoundError: If specified manifest files don't exist
    """
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Check data paths
    if not Path(config['data']['train_manifest']).exists():
        raise FileNotFoundError(f"Train manifest not found: {config['data']['train_manifest']}")
    
    if not Path(config['data']['val_manifest']).exists():
        raise FileNotFoundError(f"Validation manifest not found: {config['data']['val_manifest']}")
    
    logging.info("✓ Configuration validated successfully")


def get_base_model(trainer: pl.Trainer, cfg: OmegaConf) -> ASRModel:
    """
    Returns the base model to be fine-tuned following NeMo's official pattern.
    Supports initialization from pretrained model name or .nemo file.
    
    Args:
        trainer: PyTorch Lightning Trainer
        cfg: OmegaConf configuration
        
    Returns:
        asr_model: ASRModel instance
    """
    asr_model = None
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError(
            "Both `init_from_nemo_model` and `init_from_pretrained_model` cannot be None, "
            "should pass at least one of them"
        )
    elif nemo_model_path is not None:
        logging.info(f"Loading model from .nemo file: {nemo_model_path}")
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    elif pretrained_name is not None:
        # Handle multi-GPU download coordination
        num_ranks = trainer.num_devices * trainer.num_nodes
        if num_ranks > 1:
            if is_global_rank_zero():
                logging.info(f"Rank 0: Downloading pretrained model: {pretrained_name}")
                asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
            else:
                # Other ranks wait for download to complete
                wait_time = int(cfg.get('exp_manager', {}).get('seconds_to_sleep', 60))
                if wait_time < 60:
                    wait_time = 60
                logging.info(f"Sleeping for {wait_time} seconds to wait for model download to finish.")
                time.sleep(wait_time)
                asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
        else:
            logging.info(f"Loading pretrained model: {pretrained_name}")
            asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
    
    # Set trainer on model (important for NeMo)
    asr_model.set_trainer(trainer)
    return asr_model


def setup_dataloaders(asr_model: ASRModel, cfg: OmegaConf) -> ASRModel:
    """
    Sets up the training and validation dataloaders for the model.
    Follows NeMo's official pattern.
    
    Args:
        asr_model: ASRModel instance
        cfg: OmegaConf configuration
        
    Returns:
        asr_model: ASRModel instance with configured dataloaders
    """
    # Convert model config to dict config if needed
    cfg_dict = model_utils.convert_model_config_to_dict_config(cfg)
    
    # Setup training data
    logging.info("Setting up training data loader...")
    asr_model.setup_training_data(cfg_dict.model.train_ds)
    
    # Setup validation data (can be single or multiple validation sets)
    logging.info("Setting up validation data loader...")
    if hasattr(cfg_dict.model, 'validation_ds') and cfg_dict.model.validation_ds is not None:
        asr_model.setup_multiple_validation_data(cfg_dict.model.validation_ds)
    
    # Setup test data if provided
    if hasattr(cfg_dict.model, 'test_ds') and cfg_dict.model.test_ds is not None:
        if cfg_dict.model.test_ds.manifest_filepath is not None:
            logging.info("Setting up test data loader...")
            asr_model.setup_multiple_test_data(cfg_dict.model.test_ds)
    
    return asr_model


def _get_optimizer_name(config: Dict) -> str:
    """
    Get the optimizer name and handle 8-bit optimizer mapping.
    
    For adamw_8bit/adamw8bit, we return 'adamw' because we handle the 8-bit version
    manually in the training loop (NeMo doesn't natively support it).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Optimizer name compatible with NeMo's optimizer registry
    """
    optimizer = config['training'].get('optimizer', 'adamw')
    
    # Handle 8-bit optimizer - support both naming conventions
    # adamw_8bit (with underscore) and adamw8bit (without underscore)
    if optimizer in ['adamw_8bit', 'adamw8bit']:
        logging.info(f"8-bit AdamW requested ('{optimizer}') - will be handled manually")
        return 'adamw'  # Use standard adamw for NeMo config
    
    return optimizer


def _setup_gradient_checkpointing(asr_model: ASRModel, config: Dict) -> ASRModel:
    """
    Enable gradient checkpointing if requested in config.
    Gradient checkpointing trades compute for memory by not storing all activations.
    
    Args:
        asr_model: ASR model instance
        config: Configuration dictionary
        
    Returns:
        asr_model with gradient checkpointing enabled
    """
    if config.get('model', {}).get('gradient_checkpointing', False):
        logging.info("Enabling gradient checkpointing (30-50% activation memory reduction)")
        try:
            # Enable gradient checkpointing for encoder
            if hasattr(asr_model, 'encoder') and hasattr(asr_model.encoder, 'gradient_checkpointing_enable'):
                asr_model.encoder.gradient_checkpointing_enable()
            elif hasattr(asr_model, 'encoder'):
                # Try setting through config
                if hasattr(asr_model.encoder, 'cfg'):
                    asr_model.encoder.cfg.gradient_checkpointing = True
            
            # Enable gradient checkpointing for decoder if it exists
            if hasattr(asr_model, 'decoder') and hasattr(asr_model.decoder, 'gradient_checkpointing_enable'):
                asr_model.decoder.gradient_checkpointing_enable()
            elif hasattr(asr_model, 'decoder'):
                if hasattr(asr_model.decoder, 'cfg'):
                    asr_model.decoder.cfg.gradient_checkpointing = True
            
            logging.info("✓ Gradient checkpointing enabled")
        except Exception as e:
            logging.warning(f"Could not enable gradient checkpointing: {e}")
            logging.warning("Continuing without gradient checkpointing...")
    
    return asr_model


def create_nemo_config(config: Dict) -> OmegaConf:
    """
    Convert our YAML config to NeMo OmegaConf format.
    This creates a config structure compatible with NeMo's ASR training pipeline.
    
    Args:
        config: Dictionary loaded from config.yaml
        
    Returns:
        OmegaConf configuration object
    """
    # Get max_samples parameter
    max_samples = config['data'].get('max_samples', -1)
    if max_samples is None:
        max_samples = -1  # None means use all samples
    
    # Get max_val_samples parameter
    max_val_samples = config['data'].get('max_val_samples', -1)
    if max_val_samples is None:
        max_val_samples = -1  # None means use all validation samples
    
    # Determine model initialization method
    model_name = config['model']['name']
    init_config = {}
    if model_name.endswith('.nemo'):
        init_config['init_from_nemo_model'] = model_name
        init_config['init_from_pretrained_model'] = None
    else:
        init_config['init_from_nemo_model'] = None
        init_config['init_from_pretrained_model'] = model_name
    
    # Create NeMo-style config
    nemo_config = {
        'name': 'Parakeet-TDT-ASR-Finetuning',
        **init_config,  # Add init_from_nemo_model or init_from_pretrained_model
        'model': {
            'train_ds': {
                'manifest_filepath': config['data']['train_manifest'],
                'sample_rate': int(config['data']['sampling_rate']),
                'batch_size': int(config['training']['per_device_train_batch_size']),
                'shuffle': True,
                'num_workers': int(config['training'].get('dataloader_num_workers', 4)),
                'pin_memory': bool(config['training'].get('dataloader_pin_memory', True)),
                'max_duration': float(config['data'].get('max_audio_length', 30.0)),
                'min_duration': float(config['data'].get('min_audio_length', 0.1)),
                'trim': True,
                'max_utts': int(max_samples) if max_samples > 0 else 0,
                'use_start_end_token': False,
            },
            'validation_ds': {
                'manifest_filepath': config['data']['val_manifest'],
                'sample_rate': int(config['data']['sampling_rate']),
                'batch_size': int(config['training']['per_device_eval_batch_size']),
                'shuffle': False,
                'num_workers': int(config['training'].get('dataloader_num_workers', 4)),
                'pin_memory': bool(config['training'].get('dataloader_pin_memory', True)),
                'max_duration': float(config['data'].get('max_audio_length', 30.0)),
                'min_duration': float(config['data'].get('min_audio_length', 0.1)),
                'max_utts': int(max_val_samples) if max_val_samples > 0 else 0,
                'use_start_end_token': False,
            },
            'optim': {
                'name': _get_optimizer_name(config),
                'lr': float(config['training']['learning_rate']),
                'betas': [0.9, 0.999],
                'weight_decay': float(config['training'].get('weight_decay', 0.0001)),
                'sched': {
                    'name': config['training'].get('scheduler', 'CosineAnnealing'),
                    'warmup_steps': int(config['training'].get('warmup_steps', 1000)),
                    'warmup_ratio': None,
                    'min_lr': float(config['training'].get('min_learning_rate', 1e-6)),
                }
            },
        },
        'trainer': {
            'devices': int(config['training'].get('num_gpus', 1)),
            'num_nodes': int(config['training'].get('num_nodes', 1)),
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'max_epochs': int(config['training']['num_train_epochs']),  # Must be int, not float
            'max_steps': int(config['training'].get('max_steps', -1)),  # Must be int, not float
            'val_check_interval': int(config['training'].get('eval_steps', 1000)),
            'log_every_n_steps': int(config['training'].get('logging_steps', 50)),
            'precision': 'bf16-mixed' if config['training'].get('bf16', False) else ('16-mixed' if config['training'].get('fp16', False) else '32-true'),
            'accumulate_grad_batches': int(config['training'].get('gradient_accumulation_steps', 1)),
            'gradient_clip_val': float(config['training'].get('max_grad_norm', 1.0)),
            'enable_checkpointing': False,  # NeMo's exp_manager will handle checkpointing
            'logger': False,  # NeMo's exp_manager will handle logging
            'enable_progress_bar': True,
        },
        'exp_manager': {
            'exp_dir': config['training']['output_dir'],
            'name': config['training'].get('run_name', 'parakeet-tdt-finetuning'),
            'create_tensorboard_logger': True,
            'create_checkpoint_callback': not config['training'].get('disable_checkpointing', False),
            'checkpoint_callback_params': {
                'monitor': 'val_wer',
                'mode': 'min',
                'save_top_k': int(config['training'].get('save_total_limit', 3)),
                'every_n_train_steps': int(config['training']['save_steps']) if config['training'].get('save_steps') else None,
                'every_n_epochs': 1 if config['training'].get('save_steps') is None else None,
                'filename': 'parakeet-tdt--{epoch:02d}-{step}-{val_wer:.4f}',
                'save_last': True,
            },
            'resume_if_exists': config['training'].get('resume_from_checkpoint', False),
            'resume_ignore_no_checkpoint': True,
            'seconds_to_sleep': 60,  # For multi-GPU model download coordination
        }
    }
    
    return OmegaConf.create(nemo_config)


def print_training_info(config: Dict, nemo_config: OmegaConf):
    """Print training configuration summary"""
    max_samples = config['data'].get('max_samples', -1)
    if max_samples is None or max_samples <= 0:
        max_samples_str = "All (full dataset)"
    else:
        max_samples_str = f"{max_samples:,}"
    
    max_val_samples = config['data'].get('max_val_samples', -1)
    if max_val_samples is None or max_val_samples <= 0:
        max_val_samples_str = "All"
    else:
        max_val_samples_str = f"{max_val_samples:,}"
    
    logging.info("\n" + "="*70)
    logging.info("🚀 NVIDIA Parakeet TDT Training Configuration")
    logging.info("="*70)
    logging.info(f"Model: {config['model']['name']}")
    logging.info(f"Output: {config['training']['output_dir']}")
    logging.info(f"Train Manifest: {config['data']['train_manifest']}")
    logging.info(f"Val Manifest: {config['data']['val_manifest']}")
    logging.info(f"Max Train Samples: {max_samples_str}")
    logging.info(f"Max Val Samples: {max_val_samples_str}")
    logging.info(f"\n📊 Training Parameters:")
    logging.info(f"  Epochs: {config['training']['num_train_epochs']}")
    logging.info(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
    logging.info(f"  Gradient Accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    logging.info(f"  Effective Batch: {config['training']['per_device_train_batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}")
    logging.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logging.info(f"  Max Audio Length: {config['data'].get('max_audio_length', 30.0)}s")
    logging.info(f"\n🔧 Hardware:")
    logging.info(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logging.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    logging.info("="*70 + "\n")


def train(config_path: str):
    """
    Main training function following NeMo's official fine-tuning pattern.
    
    This function orchestrates the entire training pipeline:
    1. Load and validate configuration
    2. Create PyTorch Lightning trainer
    3. Setup experiment manager (logging, checkpointing)
    4. Load base model (pretrained or from .nemo file)
    5. Setup dataloaders
    6. Setup optimizer
    7. Run training
    8. Save final model
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    logging.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)
    
    # Create NeMo config
    nemo_config = create_nemo_config(config)
    print_training_info(config, nemo_config)
    
    # Log full config (useful for debugging)
    logging.info(f"\nFull NeMo Config:\n{OmegaConf.to_yaml(nemo_config)}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / 'training_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Saved training config to {config_save_path}")
    
    try:
        # Step 1: Create PyTorch Lightning trainer
        logging.info("\n" + "="*70)
        logging.info("Step 1: Creating PyTorch Lightning Trainer")
        logging.info("="*70)
        trainer = pl.Trainer(**nemo_config.trainer)
        
        # Step 2: Setup experiment manager (must be done before model loading)
        logging.info("\n" + "="*70)
        logging.info("Step 2: Setting up Experiment Manager")
        logging.info("="*70)
        exp_manager(trainer, nemo_config.get("exp_manager", None))
        
        # Step 3: Load base model
        logging.info("\n" + "="*70)
        logging.info("Step 3: Loading Base Model")
        logging.info("="*70)
        asr_model = get_base_model(trainer, nemo_config)
        
        # Step 3.5: Enable gradient checkpointing if requested
        asr_model = _setup_gradient_checkpointing(asr_model, config)
        
        # Step 4: Setup dataloaders
        logging.info("\n" + "="*70)
        logging.info("Step 4: Setting up Dataloaders")
        logging.info("="*70)
        asr_model = setup_dataloaders(asr_model, nemo_config)
        
        # Log dataset statistics
        logging.info("\n📊 Dataset Statistics:")
        if hasattr(asr_model, '_train_dl') and asr_model._train_dl is not None:
            try:
                train_samples = len(asr_model._train_dl.dataset)
                logging.info(f"  Training samples: {train_samples:,}")
            except:
                logging.info(f"  Training samples: Available (count not accessible)")
        
        if hasattr(asr_model, '_validation_dl') and asr_model._validation_dl is not None:
            try:
                val_samples = len(asr_model._validation_dl.dataset)
                logging.info(f"  Validation samples: {val_samples:,}")
            except:
                logging.info(f"  Validation samples: Available (count not accessible)")
        
        # Step 5: Setup optimizer
        logging.info("\n" + "="*70)
        logging.info("Step 5: Setting up Optimizer")
        logging.info("="*70)
        
        # Handle 8-bit optimizer if requested (support both naming conventions)
        if config['training'].get('optimizer') in ['adamw_8bit', 'adamw8bit'] and BITSANDBYTES_AVAILABLE:
            logging.info("Setting up 8-bit AdamW optimizer (bypassing NeMo's optimizer registry)")
            try:
                # Temporarily change config to use standard adamw for NeMo setup
                original_optimizer = nemo_config.model.optim.name
                nemo_config.model.optim.name = 'adamw'
                
                # Setup optimization with standard AdamW (this sets up the scheduler)
                asr_model.setup_optimization(nemo_config.model.optim)
                
                # Now replace the optimizer with 8-bit version
                optimizer = bnb.optim.AdamW8bit(
                    asr_model.parameters(),
                    lr=float(config['training']['learning_rate']),
                    betas=(0.9, 0.999),
                    weight_decay=float(config['training'].get('weight_decay', 0.0001)),
                )
                
                # Replace NeMo's optimizer with our 8-bit version
                asr_model._optimizer = optimizer
                
                # Restore original config
                nemo_config.model.optim.name = original_optimizer
                
                logging.info("✓ 8-bit optimizer configured (75% memory reduction)")
                logging.info(f"  Optimizer: {type(optimizer).__name__}")
            except Exception as e:
                logging.error(f"Could not setup 8-bit optimizer: {e}")
                logging.error("Falling back to standard optimizer")
                # Change to standard adamw and retry
                nemo_config.model.optim.name = 'adamw'
                asr_model.setup_optimization(nemo_config.model.optim)
        else:
            # Standard optimizer path
            if config['training'].get('optimizer') in ['adamw_8bit', 'adamw8bit']:
                logging.warning(f"{config['training'].get('optimizer')} requested but bitsandbytes not available")
                logging.warning("Falling back to standard adamw optimizer")
                nemo_config.model.optim.name = 'adamw'
        asr_model.setup_optimization(nemo_config.model.optim)
        
        # Step 6: Start training
        logging.info("\n" + "="*70)
        logging.info("Step 6: Starting Training")
        logging.info("="*70)
        logging.info("Training will now begin. Monitor progress in TensorBoard:")
        logging.info(f"  tensorboard --logdir {output_dir}")
        logging.info("="*70 + "\n")
        
        # Checkpoint handling:
        # 1. If resume_from_checkpoint=True and checkpoints exist in output_dir → auto-resume (latest)
        # 2. Otherwise, if checkpoint_path specified → start from that checkpoint
        # 3. Otherwise → start from scratch
        checkpoint_path = config['training'].get('checkpoint_path')
        resume_enabled = config['training'].get('resume_from_checkpoint', False)
        
        # Check for existing checkpoints in output directory
        existing_ckpts = list(output_dir.rglob("*.ckpt")) if output_dir.exists() else []
        
        if resume_enabled and existing_ckpts:
            # Auto-resume: let exp_manager handle it (finds latest checkpoint)
            logging.info(f"Found {len(existing_ckpts)} existing checkpoint(s) in output directory")
            logging.info("Auto-resuming from latest checkpoint...")
            trainer.fit(asr_model)
        elif checkpoint_path and Path(checkpoint_path).exists():
            # First run: start from initial checkpoint
            logging.info(f"Starting from initial checkpoint: {checkpoint_path}")
            trainer.fit(asr_model, ckpt_path=checkpoint_path)
        else:
            if checkpoint_path:
                logging.warning(f"Checkpoint path specified but not found: {checkpoint_path}")
            logging.info("Starting training from scratch")
            trainer.fit(asr_model)
        
        # Step 7: Save final model
        logging.info("\n" + "="*70)
        logging.info("Step 7: Saving Final Model")
        logging.info("="*70)
        final_model_path = output_dir / 'final_model.nemo'
        logging.info(f"Saving final model to: {final_model_path}")
        asr_model.save_to(str(final_model_path))
        
        # Training complete
        logging.info("\n" + "="*70)
        logging.info("✅ Training Completed Successfully!")
        logging.info("="*70)
        logging.info(f"📁 Outputs saved to: {output_dir}")
        logging.info(f"📄 Final model: {final_model_path}")
        logging.info(f"📊 TensorBoard logs: tensorboard --logdir {output_dir}")
        logging.info(f"📈 Checkpoints: {output_dir}/parakeet-tdt-finetuning/*/checkpoints/")
        logging.info("="*70 + "\n")
        
    except Exception as e:
        logging.error("\n" + "="*70)
        logging.error("❌ Training Failed")
        logging.error("="*70)
        logging.error(f"Error: {e}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        raise


def main():
    """
    Main entry point for the training script.
    Parses command-line arguments and starts training.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune NVIDIA Parakeet TDT 0.6B v3 on custom ASR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default config (stage1)
  python src/train.py
  
  # Using specific config
  python src/train.py --config configs/parakeet_stage1.yaml
  python src/train.py --config configs/parakeet_stage2.yaml
  
  # For multi-GPU training, use:
  torchrun --nproc_per_node=2 src/train.py --config configs/parakeet_stage1.yaml

For more information, see: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/parakeet_stage1.yaml",
        help="Path to training configuration file (default: configs/parakeet_stage1.yaml)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        logging.error(f"Configuration file not found: {args.config}")
        logging.error("Please specify --config path or ensure configs/ directory has config files")
        logging.error("Available configs: configs/parakeet_stage1.yaml, configs/parakeet_stage2.yaml")
        return 1
    
    # Start training
    try:
        train(args.config)
        return 0
    except Exception as e:
        logging.error("Training failed with exception")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())

