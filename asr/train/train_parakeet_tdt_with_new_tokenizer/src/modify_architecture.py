#!/usr/bin/env python3
"""
Modify Parakeet TDT model architecture for multilingual tokenizer.

This script:
1. Loads the base parakeet-tdt-0.6b-v3 model from HuggingFace
2. Replaces the tokenizer with a new multilingual SentencePiece model
3. Resizes the decoder embedding and joint output layers to match new vocab size
4. Saves the modified model as a new .nemo file

The encoder weights are preserved, while decoder/joint layers are re-initialized
for the new vocabulary size.

Usage:
    python modify_architecture.py \
        --base-model nvidia/parakeet-tdt-0.6b-v3 \
        --tokenizer-path ../common/tokenizers/tokenizer_multilingual.model \
        --output-path ./models/parakeet-tdt-0.6b-multilingual-init.nemo

Dependencies:
    pip install nemo_toolkit[asr] sentencepiece torch
"""

import os
import sys
import shutil
import argparse
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_base_model(model_name_or_path: str):
    """Load the base Parakeet model."""
    import nemo.collections.asr as nemo_asr
    
    if model_name_or_path.endswith('.nemo'):
        logger.info(f"Loading model from .nemo file: {model_name_or_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_name_or_path)
    else:
        logger.info(f"Loading pretrained model: {model_name_or_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name_or_path)
    
    return model


def get_vocab_size_from_tokenizer(tokenizer_path: str) -> int:
    """Get vocabulary size from SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp.get_piece_size()


def resize_embedding(old_embedding: nn.Module, new_vocab_size: int) -> nn.Module:
    """
    Create a new embedding layer for new vocabulary size.
    
    IMPORTANT: We do NOT copy old embeddings because the vocabulary has changed.
    Token ID 500 in the old vocab != Token ID 500 in the new vocab.
    Fresh random initialization is required for proper training.
    
    Args:
        old_embedding: Original nn.Embedding layer (used only for dimensions)
        new_vocab_size: Target vocabulary size
        
    Returns:
        New nn.Embedding layer with random initialization
    """
    old_vocab_size, embedding_dim = old_embedding.weight.shape
    
    logger.info(f"  Creating new embedding: {old_vocab_size} -> {new_vocab_size} (dim={embedding_dim})")
    
    # Create new embedding layer
    new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
    
    # Use Xavier uniform initialization for better training dynamics
    # This gives a good balance for both small and large vocabs
    nn.init.xavier_uniform_(new_embedding.weight)
    
    logger.info(f"  Initialized {new_vocab_size} embeddings with Xavier uniform (no copying - vocab changed)")
    
    return new_embedding


def resize_linear(old_linear: nn.Module, new_out_features: int) -> nn.Module:
    """
    Create a new linear layer for new output dimension.
    
    IMPORTANT: We do NOT copy old weights because the vocabulary has changed.
    Output index 500 in the old layer != Output index 500 in the new layer.
    Fresh random initialization is required for proper training.
    
    Args:
        old_linear: Original nn.Linear layer (used only for dimensions)
        new_out_features: Target output features (vocabulary size + blank + durations)
        
    Returns:
        New nn.Linear layer with random initialization
    """
    old_out_features = old_linear.out_features
    in_features = old_linear.in_features
    has_bias = old_linear.bias is not None
    
    logger.info(f"  Creating new linear: {in_features}x{old_out_features} -> {in_features}x{new_out_features}")
    
    # Create new linear layer
    new_linear = nn.Linear(in_features, new_out_features, bias=has_bias)
    
    # Use Xavier uniform initialization for better training dynamics
    nn.init.xavier_uniform_(new_linear.weight)
    if has_bias:
        nn.init.zeros_(new_linear.bias)
    
    logger.info(f"  Initialized {new_out_features} outputs with Xavier uniform (no copying - vocab changed)")
    
    return new_linear


def update_model_for_new_tokenizer(model, new_vocab_size: int):
    """
    Update model's decoder and joint network for new vocabulary size.
    
    For RNNT models like Parakeet TDT:
    - decoder.embedding: Maps token IDs to embeddings (vocab_size x embed_dim)
    - joint.fc2 (or joint.joint_net[-1]): Maps hidden to logits (hidden x vocab_size+1)
    
    The +1 is for the blank token in RNNT.
    """
    # Update decoder embedding
    # In RNNT/TDT models, embedding is at decoder.prediction.embed (size = vocab + 1)
    logger.info("\n📝 Updating decoder embedding...")
    decoder_embed_size = new_vocab_size + 1  # vocab + blank
    
    if hasattr(model.decoder, 'prediction') and hasattr(model.decoder.prediction, 'embed'):
        old_embedding = model.decoder.prediction.embed
        new_embedding = resize_embedding(old_embedding, decoder_embed_size)
        model.decoder.prediction.embed = new_embedding
        logger.info(f"  Resized decoder.prediction.embed to {decoder_embed_size}")
    elif hasattr(model.decoder, 'embedding'):
        old_embedding = model.decoder.embedding
        new_embedding = resize_embedding(old_embedding, decoder_embed_size)
        model.decoder.embedding = new_embedding
        logger.info(f"  Resized decoder.embedding to {decoder_embed_size}")
    elif hasattr(model.decoder, 'embed'):
        old_embedding = model.decoder.embed
        new_embedding = resize_embedding(old_embedding, decoder_embed_size)
        model.decoder.embed = new_embedding
        logger.info(f"  Resized decoder.embed to {decoder_embed_size}")
    else:
        logger.warning("Could not find decoder embedding layer")
        logger.warning(f"  decoder attributes: {[a for a in dir(model.decoder) if not a.startswith('_')]}")
    
    # Update joint network output layer
    # For TDT models: vocab_size + num_durations + 1 (blank)
    # Default TDT has 5 durations [0,1,2,3,4], so output = vocab + 6
    num_durations = 5  # TDT default
    new_output_size = new_vocab_size + num_durations + 1
    logger.info(f"  TDT output: {new_vocab_size} vocab + {num_durations} durations + 1 blank = {new_output_size}")
    
    logger.info("\n📝 Updating joint network output layer...")
    
    # NeMo's RNNTJoint uses joint_net which is a Sequential containing the output linear
    if hasattr(model.joint, 'joint_net'):
        # Find the last linear layer in joint_net
        joint_net = model.joint.joint_net
        if isinstance(joint_net, nn.Sequential):
            # Find last Linear layer
            for i in range(len(joint_net) - 1, -1, -1):
                if isinstance(joint_net[i], nn.Linear):
                    old_linear = joint_net[i]
                    new_linear = resize_linear(old_linear, new_output_size)
                    joint_net[i] = new_linear
                    break
        else:
            logger.warning(f"Unexpected joint_net type: {type(joint_net)}")
    elif hasattr(model.joint, 'fc2'):
        # Some models use fc2 directly
        old_linear = model.joint.fc2
        new_linear = resize_linear(old_linear, new_output_size)
        model.joint.fc2 = new_linear
    elif hasattr(model.joint, 'pred'):
        # Or pred layer
        old_linear = model.joint.pred
        new_linear = resize_linear(old_linear, new_output_size)
        model.joint.pred = new_linear
    else:
        logger.warning("Could not find joint network output layer")
        logger.warning(f"Available joint attributes: {[a for a in dir(model.joint) if not a.startswith('_')]}")
    
    # Update model config
    logger.info("\n📝 Updating model configuration...")
    if hasattr(model, 'cfg'):
        from omegaconf import open_dict
        with open_dict(model.cfg):
            if hasattr(model.cfg, 'decoder'):
                model.cfg.decoder.vocab_size = new_vocab_size
            if hasattr(model.cfg, 'joint'):
                # TDT uses num_classes = vocab + durations + blank
                model.cfg.joint.num_classes = new_output_size
    
    return model


def replace_tokenizer(model, tokenizer_path: str):
    """
    Replace the model's tokenizer with a new SentencePiece model.
    
    Args:
        model: NeMo ASR model
        tokenizer_path: Path to new SentencePiece .model file
        
    Returns:
        Updated model
    """
    logger.info(f"\n📝 Replacing tokenizer with: {tokenizer_path}")
    
    # Get new vocab size
    new_vocab_size = get_vocab_size_from_tokenizer(tokenizer_path)
    logger.info(f"  New vocabulary size: {new_vocab_size}")
    
    # Get old vocab size for comparison
    if hasattr(model, 'tokenizer'):
        old_vocab_size = model.tokenizer.vocab_size if hasattr(model.tokenizer, 'vocab_size') else 'unknown'
        logger.info(f"  Old vocabulary size: {old_vocab_size}")
    
    # Update tokenizer in model config
    # The actual tokenizer replacement happens when we save/reload or via NeMo's change_vocabulary
    try:
        # Use NeMo's built-in method if available
        if hasattr(model, 'change_vocabulary'):
            logger.info("  Using model.change_vocabulary() method")
            model.change_vocabulary(
                new_tokenizer_dir=str(Path(tokenizer_path).parent),
                new_tokenizer_type='bpe',
            )
            # Ensure TDT joint output size is updated for durations + blank.
            model = update_model_for_new_tokenizer(model, new_vocab_size)
            # Update tokenizer config to point to new model path.
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'tokenizer'):
                model.cfg.tokenizer.dir = str(Path(tokenizer_path).parent)
                model.cfg.tokenizer.type = 'bpe'
            return model
    except Exception as e:
        logger.warning(f"  change_vocabulary failed: {e}")
        logger.info("  Falling back to manual replacement")
    
    # Manual tokenizer replacement
    # First, resize the layers
    model = update_model_for_new_tokenizer(model, new_vocab_size)
    
    # Now update the tokenizer configuration
    # This will be picked up when the model is saved and reloaded
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'tokenizer'):
        model.cfg.tokenizer.dir = str(Path(tokenizer_path).parent)
        model.cfg.tokenizer.type = 'bpe'
    
    return model


def save_modified_model(model, output_path: str, tokenizer_path: str):
    """
    Save the modified model as a .nemo file.
    
    NeMo .nemo files are tar archives containing:
    - model_config.yaml
    - model_weights.ckpt  
    - tokenizer files
    
    We need to include the new tokenizer in the archive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving modified model to: {output_path}")
    
    # Copy tokenizer to model's expected location
    tokenizer_path = Path(tokenizer_path)
    if hasattr(model, '_cfg') and hasattr(model._cfg, 'tokenizer'):
        # Use OmegaConf.open_dict to allow modifications to frozen config
        from omegaconf import OmegaConf, open_dict
        new_vocab_size = get_vocab_size_from_tokenizer(str(tokenizer_path))
        with open_dict(model._cfg):
            # Update tokenizer config - use the correct key names
            if 'model_path' in model._cfg.tokenizer:
                model._cfg.tokenizer.model_path = tokenizer_path.name
            if 'dir' in model._cfg.tokenizer:
                model._cfg.tokenizer.dir = str(tokenizer_path.parent)
            # Also update vocab_size in the config
            model._cfg.tokenizer.vocab_size = new_vocab_size
    
    # Save the model
    try:
        # First save normally
        model.save_to(str(output_path))
        logger.info(f"  Model saved successfully")
        
        # Now we need to add the tokenizer to the .nemo archive
        import tarfile
        
        # Add tokenizer to the archive
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try different archive formats (NeMo might use tar or tar.gz)
            tar_mode = None
            for mode in ['r:gz', 'r:', 'r:*']:
                try:
                    with tarfile.open(output_path, mode) as tar:
                        tar.extractall(tmpdir)
                    tar_mode = mode.replace('r', 'w')
                    if tar_mode == 'w:*':
                        tar_mode = 'w:'  # Default to uncompressed if auto-detected
                    break
                except Exception:
                    continue
            
            if tar_mode is None:
                logger.warning("  Could not open .nemo archive, tokenizer not added")
                logger.warning("  You may need to manually ensure tokenizer is available during training")
                return output_path
            
            # Copy tokenizer files
            tokenizer_dest = Path(tmpdir) / tokenizer_path.name
            shutil.copy2(tokenizer_path, tokenizer_dest)
            logger.info(f"  Copied tokenizer: {tokenizer_path.name}")
            
            # Also copy vocab file if it exists
            vocab_path = tokenizer_path.with_suffix('.vocab')
            if vocab_path.exists():
                shutil.copy2(vocab_path, Path(tmpdir) / vocab_path.name)
                logger.info(f"  Copied vocab: {vocab_path.name}")
            
            # Update model_config.yaml to reflect new vocab size
            config_path = Path(tmpdir) / "model_config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Get new vocab size from tokenizer
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(str(tokenizer_path))
                new_vocab_size = sp.vocab_size()
                
                # TDT models have 5 durations + 1 blank, so output = vocab + 6
                num_durations = 5
                tdt_output_size = new_vocab_size + num_durations + 1
                
                # Update vocab sizes in config
                if 'decoder' in config:
                    # Decoder embedding size is vocab + 1 (for blank)
                    config['decoder']['vocab_size'] = new_vocab_size
                    logger.info(f"  Updated decoder.vocab_size: {new_vocab_size}")
                if 'joint' in config:
                    # Joint output is vocab + durations + blank for TDT
                    config['joint']['num_classes'] = tdt_output_size
                    logger.info(f"  Updated joint.num_classes: {tdt_output_size}")
                    # Remove vocab_size if it was incorrectly added
                    if 'vocab_size' in config['joint']:
                        del config['joint']['vocab_size']
                if 'tokenizer' in config:
                    config['tokenizer']['vocab_size'] = new_vocab_size
                    # Update tokenizer path to point to the new tokenizer
                    config['tokenizer']['model_path'] = tokenizer_path.name
                    logger.info(f"  Updated tokenizer config")
                
                # Write updated config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info(f"  Updated model_config.yaml")
            
            # Re-create archive with tokenizer (use same format)
            with tarfile.open(output_path, tar_mode) as tar:
                for item in Path(tmpdir).iterdir():
                    tar.add(item, arcname=item.name)
        
        logger.info(f"  Tokenizer added to .nemo archive")
        
    except Exception as e:
        logger.error(f"Error during save/tokenizer injection: {e}")
        logger.warning("  Model may have been saved but tokenizer injection failed")
        logger.warning("  Continuing anyway - ensure tokenizer is available during training")
    
    return output_path


def print_model_info(model, title: str):
    """Print model architecture info."""
    logger.info(f"\n{'='*60}")
    logger.info(title)
    logger.info('='*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Decoder info
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        if hasattr(decoder, 'embedding'):
            logger.info(f"Decoder embedding: {decoder.embedding.weight.shape}")
        elif hasattr(decoder, 'embed'):
            logger.info(f"Decoder embedding: {decoder.embed.weight.shape}")
    
    # Joint info
    if hasattr(model, 'joint'):
        joint = model.joint
        if hasattr(joint, 'joint_net') and isinstance(joint.joint_net, nn.Sequential):
            for i, layer in enumerate(joint.joint_net):
                if isinstance(layer, nn.Linear):
                    logger.info(f"Joint layer {i}: {layer.in_features} -> {layer.out_features}")


def main():
    parser = argparse.ArgumentParser(
        description="Modify Parakeet TDT model for multilingual tokenizer"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="Base model name or path to .nemo file"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to new SentencePiece .model file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./models/parakeet-tdt-0.6b-multilingual-init.nemo",
        help="Output path for modified model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print info, don't save model"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.tokenizer_path).exists():
        logger.error(f"Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Parakeet TDT Model Architecture Modification")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"New tokenizer: {args.tokenizer_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 60)
    
    # Get new vocab size
    new_vocab_size = get_vocab_size_from_tokenizer(args.tokenizer_path)
    logger.info(f"\nNew tokenizer vocabulary size: {new_vocab_size}")
    
    # Load base model
    logger.info("\n📦 Loading base model...")
    model = load_base_model(args.base_model)
    
    # Print original model info
    print_model_info(model, "Original Model Architecture")
    
    # Replace tokenizer and resize layers
    logger.info("\n🔧 Modifying model architecture...")
    model = replace_tokenizer(model, args.tokenizer_path)
    
    # Print modified model info
    print_model_info(model, "Modified Model Architecture")
    
    if not args.dry_run:
        # Save modified model
        output_path = save_modified_model(model, args.output_path, args.tokenizer_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Model Modification Complete!")
        logger.info("=" * 60)
        logger.info(f"Modified model saved to: {output_path}")
        logger.info(f"New vocabulary size: {new_vocab_size}")
        logger.info("\nNext steps:")
        logger.info("  1. Fine-tune the model with multilingual data:")
        logger.info("     make train CONFIG=configs/parakeet_multilingual.yaml")
        logger.info("=" * 60)
    else:
        logger.info("\n[DRY RUN] Model not saved")


if __name__ == "__main__":
    main()
