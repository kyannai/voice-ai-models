#!/usr/bin/env python3
"""
Surgical vocabulary expansion for Parakeet CTC model.

This script:
1. Keeps ALL existing tokens from the original model (preserves embeddings)
2. Adds Chinese characters from training data
3. Resizes the decoder output layer (Conv1d)
4. Creates a new tokenizer vocab file
5. Updates model config and saves as new .nemo

Usage:
    python src/expand_vocabulary.py \
        --base-model nvidia/parakeet-ctc-1.1b \
        --manifest path/to/train_manifest.json \
        --output ./models/parakeet-ctc-1.1b-multilingual.nemo
"""

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def extract_chinese_chars_from_manifest(manifest_path: str, limit: int = None) -> list:
    """Extract unique Chinese characters from a manifest file."""
    char_freq = Counter()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                for c in text:
                    if '\u4e00' <= c <= '\u9fff':
                        char_freq[c] += 1
            except json.JSONDecodeError:
                continue
    
    sorted_chars = [char for char, _ in char_freq.most_common()]
    return sorted_chars


def resize_ctc_decoder(model, new_vocab_size: int) -> None:
    """Resize the CTC decoder output layer."""
    
    old_decoder = model.decoder.decoder_layers[-1]
    old_out_channels = old_decoder.out_channels
    old_in_channels = old_decoder.in_channels
    
    new_out_channels = new_vocab_size + 1  # +1 for blank
    
    logger.info(f"Resizing decoder: {old_out_channels} -> {new_out_channels}")
    
    new_decoder = nn.Conv1d(
        in_channels=old_in_channels,
        out_channels=new_out_channels,
        kernel_size=old_decoder.kernel_size,
        stride=old_decoder.stride,
        padding=old_decoder.padding,
        dilation=old_decoder.dilation,
        groups=old_decoder.groups,
        bias=old_decoder.bias is not None
    )
    
    nn.init.xavier_uniform_(new_decoder.weight)
    if new_decoder.bias is not None:
        nn.init.zeros_(new_decoder.bias)
    
    with torch.no_grad():
        old_vocab_size = old_out_channels - 1
        new_decoder.weight[:old_vocab_size] = old_decoder.weight[:old_vocab_size]
        if new_decoder.bias is not None and old_decoder.bias is not None:
            new_decoder.bias[:old_vocab_size] = old_decoder.bias[:old_vocab_size]
        
        # Copy blank token
        new_decoder.weight[new_vocab_size] = old_decoder.weight[old_vocab_size]
        if new_decoder.bias is not None and old_decoder.bias is not None:
            new_decoder.bias[new_vocab_size] = old_decoder.bias[old_vocab_size]
    
    model.decoder.decoder_layers[-1] = new_decoder
    model.decoder._num_classes = new_out_channels
    
    logger.info(f"✓ Decoder resized: {old_out_channels} -> {new_out_channels}")
    logger.info(f"  - Preserved {old_vocab_size} existing token weights")
    logger.info(f"  - Added {new_vocab_size - old_vocab_size} new token slots")
    logger.info(f"  - Blank token moved from {old_vocab_size} to {new_vocab_size}")


def save_expanded_model(model, new_chars, output_path: Path, old_vocab_size: int):
    """Save model with expanded vocabulary by modifying the .nemo archive."""
    
    # First save model normally
    temp_nemo = output_path.parent / "temp_model.nemo"
    model.save_to(str(temp_nemo))
    
    # Get original vocab
    tokenizer = model.tokenizer.tokenizer
    old_vocab = [tokenizer.id_to_piece(i) for i in range(old_vocab_size)]
    new_vocab = old_vocab + new_chars
    new_vocab_size = len(new_vocab)
    
    # Extract, modify, and repack
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract
        with tarfile.open(temp_nemo, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Modify config
        config_path = tmpdir / 'model_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update decoder config
        config['decoder']['num_classes'] = new_vocab_size + 1
        config['decoder']['vocabulary'] = new_vocab
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Create new vocab file for tokenizer
        vocab_txt_path = tmpdir / 'vocab.txt'
        with open(vocab_txt_path, 'w', encoding='utf-8') as f:
            for token in new_vocab:
                f.write(f"{token}\n")
        
        # Update tokenizer config to use our vocab
        # For character-level tokens, we'll create a simple mapping
        
        # Repack
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in tmpdir.iterdir():
                tar.add(item, arcname=item.name)
    
    # Cleanup
    temp_nemo.unlink()
    
    logger.info(f"✓ Model saved with expanded vocabulary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Expand Parakeet CTC vocabulary")
    parser.add_argument("--base-model", default="nvidia/parakeet-ctc-1.1b")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-chinese-chars", type=int, default=4000)
    parser.add_argument("--manifest-limit", type=int, default=None)
    args = parser.parse_args()
    
    import nemo.collections.asr as nemo_asr
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Parakeet CTC Vocabulary Expansion")
    print("=" * 70)
    
    # Step 1: Extract Chinese characters
    logger.info("\n[1/5] Extracting Chinese characters from training data...")
    chinese_chars = extract_chinese_chars_from_manifest(
        args.manifest, 
        limit=args.manifest_limit
    )
    logger.info(f"Found {len(chinese_chars)} unique Chinese characters")
    
    if len(chinese_chars) > args.max_chinese_chars:
        chinese_chars = chinese_chars[:args.max_chinese_chars]
        logger.info(f"Limited to top {args.max_chinese_chars} most frequent")
    
    # Step 2: Load base model
    logger.info("\n[2/5] Loading base model...")
    if args.base_model.startswith("nvidia/"):
        model = nemo_asr.models.ASRModel.from_pretrained(args.base_model)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.base_model)
    
    old_vocab_size = model.tokenizer.vocab_size
    logger.info(f"Original vocab size: {old_vocab_size}")
    
    # Step 3: Filter new chars
    logger.info("\n[3/5] Creating expanded vocabulary...")
    existing_tokens = set()
    tokenizer = model.tokenizer.tokenizer
    for i in range(tokenizer.get_piece_size()):
        existing_tokens.add(tokenizer.id_to_piece(i))
    
    new_chars = [c for c in chinese_chars if c not in existing_tokens]
    logger.info(f"Characters to add (not in existing vocab): {len(new_chars)}")
    
    new_vocab_size = old_vocab_size + len(new_chars)
    logger.info(f"New vocab size: {new_vocab_size}")
    
    # Step 4: Resize decoder
    logger.info("\n[4/5] Resizing decoder layer...")
    resize_ctc_decoder(model, new_vocab_size)
    
    # Step 5: Update config and save
    logger.info("\n[5/5] Updating config and saving model...")
    
    # Update model config
    from omegaconf import open_dict
    
    old_vocab = [tokenizer.id_to_piece(i) for i in range(old_vocab_size)]
    full_vocab = old_vocab + new_chars
    
    with open_dict(model.cfg):
        model.cfg.decoder.num_classes = new_vocab_size + 1
        model.cfg.decoder.vocabulary = full_vocab
    
    # Save model
    model.save_to(str(output_path))
    
    # Also save character mapping for the custom tokenizer
    mapping_path = output_path.parent / "chinese_char_mapping.json"
    char_mapping = {char: old_vocab_size + i for i, char in enumerate(new_chars)}
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "new_chars": new_chars,
            "char_to_id": char_mapping
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Vocabulary expansion complete!")
    print("=" * 70)
    print(f"  Original vocab: {old_vocab_size}")
    print(f"  Added Chinese chars: {len(new_chars)}")
    print(f"  New vocab size: {new_vocab_size}")
    print(f"  Model saved to: {output_path}")
    print(f"  Char mapping saved to: {mapping_path}")
    print()
    print("NOTE: This model requires a custom tokenizer wrapper for Chinese.")
    print("      The original BPE tokenizer handles English/Malay tokens.")
    print("      Chinese characters are mapped to IDs >= " + str(old_vocab_size))
    print("=" * 70)


if __name__ == "__main__":
    main()
