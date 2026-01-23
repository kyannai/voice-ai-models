#!/usr/bin/env python3
"""
Transfer vocabulary embeddings from old tokenizer to new tokenizer.

This script finds tokens that exist in BOTH vocabularies and copies their embeddings,
rather than randomly initializing everything. This dramatically improves training
convergence when changing tokenizers.

Usage:
    python src/transfer_vocabulary.py \
        --old-model ./models/original-parakeet.nemo \
        --new-tokenizer ../common/tokenizers/tokenizer_multilingual.model \
        --output ./models/parakeet-tdt-0.6b-multilingual-init.nemo
"""

import argparse
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict

import sentencepiece as spm
import torch
import torch.nn as nn
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_tokenizer(path: str) -> spm.SentencePieceProcessor:
    """Load SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def get_token_to_id_map(sp: spm.SentencePieceProcessor) -> dict:
    """Get mapping from token string to ID."""
    return {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}


def find_vocabulary_overlap(old_sp: spm.SentencePieceProcessor, 
                            new_sp: spm.SentencePieceProcessor) -> dict:
    """
    Find tokens that exist in both vocabularies.
    
    Returns:
        Dict mapping new_token_id -> old_token_id for overlapping tokens
    """
    old_map = get_token_to_id_map(old_sp)
    new_map = get_token_to_id_map(new_sp)
    
    overlap = {}
    for token, new_id in new_map.items():
        if token in old_map:
            overlap[new_id] = old_map[token]
    
    return overlap


def transfer_embeddings(old_embedding: nn.Embedding, 
                        new_vocab_size: int,
                        overlap_map: dict) -> nn.Embedding:
    """
    Create new embedding layer, copying weights for overlapping tokens.
    
    Args:
        old_embedding: Original embedding layer
        new_vocab_size: Size of new vocabulary
        overlap_map: Dict mapping new_id -> old_id for overlapping tokens
        
    Returns:
        New embedding layer with transferred weights
    """
    old_vocab_size, embed_dim = old_embedding.weight.shape
    
    # Create new embedding with Xavier init for new tokens
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    nn.init.xavier_uniform_(new_embedding.weight)
    
    # Copy weights for overlapping tokens
    copied = 0
    with torch.no_grad():
        for new_id, old_id in overlap_map.items():
            if new_id < new_vocab_size and old_id < old_vocab_size:
                new_embedding.weight[new_id] = old_embedding.weight[old_id].clone()
                copied += 1
    
    logger.info(f"  Transferred {copied}/{new_vocab_size} embeddings ({100*copied/new_vocab_size:.1f}%)")
    logger.info(f"  Randomly initialized {new_vocab_size - copied} new token embeddings")
    
    return new_embedding


def transfer_linear_output(old_linear: nn.Linear,
                          new_out_features: int,
                          overlap_map: dict,
                          num_durations: int = 5) -> nn.Linear:
    """
    Create new output linear layer, copying weights for overlapping tokens.
    
    For TDT models, the output is: [vocab_logits..., duration_logits..., blank_logit]
    We transfer vocab logits and keep duration/blank random or averaged.
    
    Args:
        old_linear: Original output linear layer
        new_out_features: New output size (vocab + durations + blank)
        overlap_map: Dict mapping new_id -> old_id for overlapping tokens
        num_durations: Number of duration tokens in TDT
        
    Returns:
        New linear layer with transferred weights
    """
    in_features = old_linear.in_features
    old_out = old_linear.out_features
    has_bias = old_linear.bias is not None
    
    # Create new layer with Xavier init
    new_linear = nn.Linear(in_features, new_out_features, bias=has_bias)
    nn.init.xavier_uniform_(new_linear.weight)
    if has_bias:
        nn.init.zeros_(new_linear.bias)
    
    # For TDT: output layout is [vocab..., dur0, dur1, dur2, dur3, dur4, blank]
    # where dur/blank indices are at the end
    old_vocab_size = old_out - num_durations - 1
    new_vocab_size = new_out_features - num_durations - 1
    
    # Copy weights for overlapping vocab tokens
    copied = 0
    with torch.no_grad():
        for new_id, old_id in overlap_map.items():
            if new_id < new_vocab_size and old_id < old_vocab_size:
                new_linear.weight[new_id] = old_linear.weight[old_id].clone()
                if has_bias:
                    new_linear.bias[new_id] = old_linear.bias[old_id].clone()
                copied += 1
        
        # Copy duration weights (they're at positions vocab_size to vocab_size+num_durations)
        for i in range(num_durations):
            old_dur_idx = old_vocab_size + i
            new_dur_idx = new_vocab_size + i
            if old_dur_idx < old_out and new_dur_idx < new_out_features:
                new_linear.weight[new_dur_idx] = old_linear.weight[old_dur_idx].clone()
                if has_bias:
                    new_linear.bias[new_dur_idx] = old_linear.bias[old_dur_idx].clone()
        
        # Copy blank token weight (last position)
        old_blank_idx = old_out - 1
        new_blank_idx = new_out_features - 1
        new_linear.weight[new_blank_idx] = old_linear.weight[old_blank_idx].clone()
        if has_bias:
            new_linear.bias[new_blank_idx] = old_linear.bias[old_blank_idx].clone()
    
    logger.info(f"  Transferred {copied}/{new_vocab_size} vocab weights ({100*copied/new_vocab_size:.1f}%)")
    logger.info(f"  Transferred duration and blank weights")
    
    return new_linear


def modify_model_with_transfer(model, new_tokenizer_path: str, old_tokenizer_path: str):
    """
    Modify model for new tokenizer, transferring weights where possible.
    
    Args:
        model: NeMo ASR model
        new_tokenizer_path: Path to new SentencePiece .model file  
        old_tokenizer_path: Path to old SentencePiece .model file
        
    Returns:
        Modified model with transferred weights
    """
    # Load tokenizers
    logger.info("Loading tokenizers...")
    old_sp = load_tokenizer(old_tokenizer_path)
    new_sp = load_tokenizer(new_tokenizer_path)
    
    old_vocab = old_sp.get_piece_size()
    new_vocab = new_sp.get_piece_size()
    
    logger.info(f"  Old vocabulary: {old_vocab} tokens")
    logger.info(f"  New vocabulary: {new_vocab} tokens")
    
    # Find overlapping tokens
    logger.info("\nFinding vocabulary overlap...")
    overlap_map = find_vocabulary_overlap(old_sp, new_sp)
    logger.info(f"  Found {len(overlap_map)} overlapping tokens ({100*len(overlap_map)/new_vocab:.1f}% of new vocab)")
    
    # Sample overlapping tokens for verification
    sample_tokens = list(overlap_map.keys())[:10]
    logger.info("  Sample overlapping tokens:")
    for new_id in sample_tokens:
        old_id = overlap_map[new_id]
        token = new_sp.id_to_piece(new_id)
        logger.info(f"    '{token}': old_id={old_id} -> new_id={new_id}")
    
    # Update decoder embedding with transfer
    logger.info("\n📝 Updating decoder embedding with vocabulary transfer...")
    decoder_embed_size = new_vocab + 1  # vocab + blank
    
    if hasattr(model.decoder, 'prediction') and hasattr(model.decoder.prediction, 'embed'):
        old_embedding = model.decoder.prediction.embed
        new_embedding = transfer_embeddings(old_embedding, decoder_embed_size, overlap_map)
        model.decoder.prediction.embed = new_embedding
        logger.info(f"  Updated decoder.prediction.embed: {old_embedding.weight.shape[0]} -> {decoder_embed_size}")
    else:
        logger.warning("Could not find decoder embedding")
    
    # Update joint network output with transfer
    logger.info("\n📝 Updating joint network output with vocabulary transfer...")
    num_durations = 5  # TDT default
    new_output_size = new_vocab + num_durations + 1
    
    if hasattr(model.joint, 'joint_net'):
        joint_net = model.joint.joint_net
        if isinstance(joint_net, nn.Sequential):
            for i in range(len(joint_net) - 1, -1, -1):
                if isinstance(joint_net[i], nn.Linear):
                    old_linear = joint_net[i]
                    new_linear = transfer_linear_output(old_linear, new_output_size, overlap_map, num_durations)
                    joint_net[i] = new_linear
                    logger.info(f"  Updated joint_net[{i}]: {old_linear.out_features} -> {new_output_size}")
                    break
    else:
        logger.warning("Could not find joint network")
    
    # Update model config
    logger.info("\n📝 Updating model configuration...")
    if hasattr(model, 'cfg'):
        from omegaconf import open_dict
        with open_dict(model.cfg):
            if hasattr(model.cfg, 'decoder'):
                model.cfg.decoder.vocab_size = new_vocab
            if hasattr(model.cfg, 'joint'):
                model.cfg.joint.num_classes = new_output_size
    
    return model


def extract_tokenizer_from_nemo(nemo_path: str, output_dir: str) -> str:
    """Extract tokenizer from a .nemo file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Find tokenizer model file
        for f in Path(tmpdir).iterdir():
            if f.suffix == '.model' and 'tokenizer' in f.name.lower():
                output_path = Path(output_dir) / f.name
                shutil.copy2(f, output_path)
                return str(output_path)
            # Also check for .model files without 'tokenizer' in name
            if f.suffix == '.model':
                output_path = Path(output_dir) / f.name
                shutil.copy2(f, output_path)
                return str(output_path)
    
    raise ValueError(f"Could not find tokenizer in {nemo_path}")


def load_model(model_path_or_name: str):
    """Load model from .nemo file or download from pretrained."""
    import nemo.collections.asr as nemo_asr
    
    if model_path_or_name.endswith('.nemo') and Path(model_path_or_name).exists():
        logger.info(f"Loading from .nemo file: {model_path_or_name}")
        return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path_or_name)
    else:
        logger.info(f"Downloading pretrained model: {model_path_or_name}")
        return nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_path_or_name)


def main():
    parser = argparse.ArgumentParser(description="Transfer vocabulary with embedding copying")
    parser.add_argument("--old-model", required=True, 
                        help="Path to original .nemo model OR pretrained model name (e.g. nvidia/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--new-tokenizer", required=True, help="Path to new tokenizer .model")
    parser.add_argument("--output", required=True, help="Output path for modified .nemo")
    parser.add_argument("--old-tokenizer", default=None, 
                        help="Path to old tokenizer (extracted from old-model if not provided)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Vocabulary Transfer with Embedding Copying")
    print("=" * 70)
    
    # Load model (from file or download)
    model = load_model(args.old_model)
    
    # Extract old tokenizer if not provided  
    old_tokenizer_path = args.old_tokenizer
    tmpdir_to_keep = None
    
    if old_tokenizer_path is None:
        logger.info("\nExtracting tokenizer from model...")
        # Create a temp directory that we'll keep until we're done
        tmpdir_to_keep = tempfile.mkdtemp()
        
        # Save model temporarily to extract tokenizer
        tmp_model_path = Path(tmpdir_to_keep) / "temp.nemo"
        model.save_to(str(tmp_model_path))
        old_tokenizer_path = extract_tokenizer_from_nemo(str(tmp_model_path), tmpdir_to_keep)
        logger.info(f"  Extracted tokenizer: {old_tokenizer_path}")
    
    # Modify with vocabulary transfer
    model = modify_model_with_transfer(model, args.new_tokenizer, old_tokenizer_path)
    
    # Clean up temp directory if we created one
    if tmpdir_to_keep:
        shutil.rmtree(tmpdir_to_keep, ignore_errors=True)
    
    # Save modified model
    logger.info(f"\n💾 Saving modified model to: {args.output}")
    
    # Update tokenizer config
    new_tokenizer_path = Path(args.new_tokenizer)
    if hasattr(model, '_cfg') and hasattr(model._cfg, 'tokenizer'):
        from omegaconf import open_dict
        with open_dict(model._cfg):
            model._cfg.tokenizer.model_path = new_tokenizer_path.name
            model._cfg.tokenizer.dir = str(new_tokenizer_path.parent)
    
    # Save model
    model.save_to(args.output)
    
    # Add new tokenizer to archive
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(args.output, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Copy tokenizer files
        shutil.copy2(new_tokenizer_path, Path(tmpdir) / new_tokenizer_path.name)
        vocab_path = new_tokenizer_path.with_suffix('.vocab')
        if vocab_path.exists():
            shutil.copy2(vocab_path, Path(tmpdir) / vocab_path.name)
        
        # Update config
        config_path = Path(tmpdir) / "model_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            new_sp = load_tokenizer(str(new_tokenizer_path))
            new_vocab = new_sp.get_piece_size()
            
            if 'tokenizer' in config:
                config['tokenizer']['model_path'] = new_tokenizer_path.name
            if 'decoder' in config:
                config['decoder']['vocab_size'] = new_vocab
            if 'joint' in config:
                config['joint']['num_classes'] = new_vocab + 6  # vocab + 5 durations + blank
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Repack
        with tarfile.open(args.output, 'w:') as tar:
            for f in Path(tmpdir).iterdir():
                tar.add(f, arcname=f.name)
    
    print("\n" + "=" * 70)
    print("✅ Done! Model saved with transferred embeddings.")
    print("=" * 70)


if __name__ == "__main__":
    main()
