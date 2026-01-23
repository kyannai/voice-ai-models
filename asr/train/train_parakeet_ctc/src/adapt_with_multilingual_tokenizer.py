#!/usr/bin/env python3
"""
Adapt Parakeet CTC model to use the multilingual tokenizer.

This script:
1. Loads the original Parakeet CTC model
2. Extracts the multilingual tokenizer from the TDT model
3. Maps overlapping tokens and transfers their embeddings
4. Creates a new .nemo file with the updated model

Usage:
    python src/adapt_with_multilingual_tokenizer.py \
        --base-model nvidia/parakeet-ctc-1.1b \
        --tokenizer-source ../train_parakeet_tdt_with_new_tokenizer/models/parakeet-tdt-0.6b-multilingual-transferred.nemo \
        --output ./models/parakeet-ctc-1.1b-multilingual.nemo
"""

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_sentencepiece_vocab(vocab_path: Path) -> dict:
    """Load SentencePiece vocab file into a dict."""
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts:
                token = parts[0]
                vocab[token] = idx
    return vocab


def create_token_mapping(old_vocab: dict, new_vocab: dict) -> dict:
    """Create mapping from old token IDs to new token IDs for overlapping tokens."""
    mapping = {}  # old_id -> new_id
    for token, old_id in old_vocab.items():
        if token in new_vocab:
            mapping[old_id] = new_vocab[token]
    return mapping


def transfer_decoder_weights(old_ckpt: dict, new_vocab_size: int, token_mapping: dict) -> dict:
    """Transfer decoder weights based on token mapping."""
    
    # Find decoder layer keys
    decoder_weight_key = None
    decoder_bias_key = None
    
    for key in old_ckpt.keys():
        if 'decoder.decoder_layers' in key and key.endswith('.weight'):
            decoder_weight_key = key
        if 'decoder.decoder_layers' in key and key.endswith('.bias'):
            decoder_bias_key = key
    
    if decoder_weight_key is None:
        raise ValueError("Could not find decoder weight key in checkpoint")
    
    old_weight = old_ckpt[decoder_weight_key]
    old_out_channels = old_weight.shape[0]
    in_channels = old_weight.shape[1]
    kernel_size = old_weight.shape[2] if len(old_weight.shape) > 2 else 1
    old_vocab_size = old_out_channels - 1  # -1 for blank
    
    new_out_channels = new_vocab_size + 1  # +1 for blank
    
    logger.info(f"Decoder weight: {decoder_weight_key}")
    logger.info(f"  Old: {old_out_channels} (vocab {old_vocab_size} + 1 blank)")
    logger.info(f"  New: {new_out_channels} (vocab {new_vocab_size} + 1 blank)")
    
    # Create new weight tensor with proper initialization
    if len(old_weight.shape) == 3:
        new_weight = torch.zeros(new_out_channels, in_channels, kernel_size, dtype=old_weight.dtype)
    else:
        new_weight = torch.zeros(new_out_channels, in_channels, dtype=old_weight.dtype)
    
    # Initialize with mean of existing embeddings for stability
    mean_embedding = old_weight[:old_vocab_size].mean(dim=0)
    std_embedding = old_weight[:old_vocab_size].std()
    
    # Initialize new tokens with small random noise around mean
    for i in range(new_vocab_size):
        if len(old_weight.shape) == 3:
            new_weight[i] = mean_embedding + torch.randn_like(mean_embedding) * std_embedding * 0.01
        else:
            new_weight[i] = mean_embedding + torch.randn_like(mean_embedding) * std_embedding * 0.01
    
    # Transfer weights for overlapping tokens
    transferred = 0
    for old_id, new_id in token_mapping.items():
        if old_id < old_vocab_size and new_id < new_vocab_size:
            new_weight[new_id] = old_weight[old_id]
            transferred += 1
    
    logger.info(f"  Transferred {transferred}/{old_vocab_size} token embeddings ({100*transferred/old_vocab_size:.1f}%)")
    
    # Initialize blank token (last position)
    new_weight[new_vocab_size] = old_weight[old_vocab_size]
    
    old_ckpt[decoder_weight_key] = new_weight
    
    # Handle bias
    if decoder_bias_key and decoder_bias_key in old_ckpt:
        old_bias = old_ckpt[decoder_bias_key]
        new_bias = torch.zeros(new_out_channels, dtype=old_bias.dtype)
        
        mean_bias = old_bias[:old_vocab_size].mean()
        new_bias[:new_vocab_size] = mean_bias
        
        for old_id, new_id in token_mapping.items():
            if old_id < old_vocab_size and new_id < new_vocab_size:
                new_bias[new_id] = old_bias[old_id]
        
        new_bias[new_vocab_size] = old_bias[old_vocab_size]
        old_ckpt[decoder_bias_key] = new_bias
    
    return old_ckpt


def main():
    parser = argparse.ArgumentParser(description="Adapt CTC model with multilingual tokenizer")
    parser.add_argument("--base-model", default="nvidia/parakeet-ctc-1.1b",
                        help="Base CTC model (HF model ID or .nemo path)")
    parser.add_argument("--tokenizer-source", required=True,
                        help="Path to .nemo file containing multilingual tokenizer")
    parser.add_argument("--output", required=True,
                        help="Output path for adapted model")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Parakeet CTC - Multilingual Tokenizer Adaptation")
    print("=" * 70)
    
    # Step 1: Get base model path
    logger.info("\n[1/7] Locating base model...")
    if args.base_model.startswith("nvidia/"):
        from huggingface_hub import hf_hub_download
        model_name = args.base_model.split('/')[-1]
        base_nemo_path = hf_hub_download(
            repo_id=args.base_model,
            filename=f"{model_name}.nemo"
        )
    else:
        base_nemo_path = args.base_model
    
    logger.info(f"Base model: {base_nemo_path}")
    
    # Step 2: Extract multilingual tokenizer
    logger.info("\n[2/7] Extracting multilingual tokenizer...")
    tokenizer_source = Path(args.tokenizer_source)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract tokenizer source
        tokdir = tmpdir / "tokenizer_src"
        tokdir.mkdir()
        
        # Detect archive type
        import subprocess
        file_type = subprocess.check_output(['file', str(tokenizer_source)]).decode()
        
        if 'gzip' in file_type:
            with tarfile.open(tokenizer_source, 'r:gz') as tar:
                tar.extractall(tokdir)
        else:
            with tarfile.open(tokenizer_source, 'r:') as tar:
                tar.extractall(tokdir)
        
        # Find multilingual tokenizer files
        ml_model = tokdir / "tokenizer_multilingual.model"
        ml_vocab = tokdir / "tokenizer_multilingual.vocab"
        
        if not ml_model.exists() or not ml_vocab.exists():
            raise FileNotFoundError(f"Multilingual tokenizer files not found in {tokenizer_source}")
        
        # Load new vocabulary
        new_vocab = load_sentencepiece_vocab(ml_vocab)
        new_vocab_size = len(new_vocab)
        logger.info(f"New tokenizer: {new_vocab_size} tokens")
        
        # Extract base model
        logger.info("\n[3/7] Extracting base model...")
        basedir = tmpdir / "base"
        basedir.mkdir()
        
        with tarfile.open(base_nemo_path, 'r:*') as tar:
            tar.extractall(basedir)
        
        # Find old tokenizer files
        old_vocab_file = None
        old_model_file = None
        for f in basedir.iterdir():
            if f.name.endswith('_tokenizer.vocab'):
                old_vocab_file = f
            if f.name.endswith('_tokenizer.model'):
                old_model_file = f
        
        if old_vocab_file is None:
            raise FileNotFoundError("Original tokenizer vocab not found")
        
        old_vocab = load_sentencepiece_vocab(old_vocab_file)
        old_vocab_size = len(old_vocab)
        logger.info(f"Old tokenizer: {old_vocab_size} tokens")
        
        # Step 4: Create token mapping
        logger.info("\n[4/7] Creating token mapping...")
        token_mapping = create_token_mapping(old_vocab, new_vocab)
        logger.info(f"Overlapping tokens: {len(token_mapping)}/{old_vocab_size}")
        
        # Step 5: Transfer decoder weights
        logger.info("\n[5/7] Transferring decoder weights...")
        ckpt_path = basedir / "model_weights.ckpt"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = transfer_decoder_weights(ckpt, new_vocab_size, token_mapping)
        
        # Step 6: Update config
        logger.info("\n[6/7] Updating configuration...")
        config_path = basedir / "model_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get vocabulary list from new tokenizer
        vocab_list = sorted(new_vocab.keys(), key=lambda x: new_vocab[x])
        
        # Update decoder config
        config['decoder']['num_classes'] = new_vocab_size
        config['decoder']['vocabulary'] = vocab_list
        
        # Update tokenizer paths to point to new files
        config['tokenizer']['model_path'] = 'nemo:tokenizer_multilingual.model'
        config['tokenizer']['spe_tokenizer_vocab'] = 'nemo:tokenizer_multilingual.vocab'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"  decoder.num_classes: {old_vocab_size + 1} -> {new_vocab_size + 1}")
        logger.info(f"  decoder.vocabulary: {old_vocab_size} -> {new_vocab_size} tokens")
        
        # Step 7: Replace tokenizer files and repack
        logger.info("\n[7/7] Repacking model archive...")
        
        # Remove old tokenizer files
        for f in basedir.iterdir():
            if '_tokenizer.' in f.name:
                f.unlink()
                logger.info(f"  Removed: {f.name}")
        
        # Copy new tokenizer files
        shutil.copy(ml_model, basedir / "tokenizer_multilingual.model")
        shutil.copy(ml_vocab, basedir / "tokenizer_multilingual.vocab")
        logger.info(f"  Added: tokenizer_multilingual.model")
        logger.info(f"  Added: tokenizer_multilingual.vocab")
        
        # Save updated checkpoint
        torch.save(ckpt, ckpt_path)
        
        # Create output archive
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in basedir.iterdir():
                tar.add(item, arcname=item.name)
    
    # Save token mapping for reference
    mapping_path = output_path.parent / "token_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "overlapping_tokens": len(token_mapping),
            "overlap_percentage": round(100 * len(token_mapping) / old_vocab_size, 2)
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Model adaptation complete!")
    print("=" * 70)
    print(f"  Old vocab: {old_vocab_size} tokens")
    print(f"  New vocab: {new_vocab_size} tokens")
    print(f"  Transferred embeddings: {len(token_mapping)} ({100*len(token_mapping)/old_vocab_size:.1f}%)")
    print(f"  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
