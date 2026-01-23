#!/usr/bin/env python3
"""
Vocabulary expansion for Parakeet CTC - Direct Archive Modification.

This approach:
1. Downloads the base model
2. Modifies the decoder weights in memory
3. Updates the config YAML
4. Repacks everything into a new .nemo file
5. The model is then loaded with strict=False during training

Usage:
    python src/expand_vocabulary_v2.py \
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
    
    return [char for char, _ in char_freq.most_common()]


def expand_decoder_weights(ckpt: dict, old_vocab_size: int, new_vocab_size: int) -> dict:
    """Expand decoder layer weights to accommodate new vocabulary."""
    
    # Find decoder layer keys
    decoder_weight_key = None
    decoder_bias_key = None
    
    for key in ckpt.keys():
        if 'decoder.decoder_layers' in key and key.endswith('.weight'):
            decoder_weight_key = key
        if 'decoder.decoder_layers' in key and key.endswith('.bias'):
            decoder_bias_key = key
    
    if decoder_weight_key is None:
        raise ValueError("Could not find decoder weight key in checkpoint")
    
    logger.info(f"Found decoder weight: {decoder_weight_key}")
    
    old_weight = ckpt[decoder_weight_key]
    old_out_channels = old_weight.shape[0]
    in_channels = old_weight.shape[1]
    kernel_size = old_weight.shape[2] if len(old_weight.shape) > 2 else 1
    
    new_out_channels = new_vocab_size + 1  # +1 for blank
    
    logger.info(f"Expanding decoder: {old_out_channels} -> {new_out_channels}")
    
    # Create new weight tensor
    if len(old_weight.shape) == 3:
        new_weight = torch.zeros(new_out_channels, in_channels, kernel_size, dtype=old_weight.dtype)
    else:
        new_weight = torch.zeros(new_out_channels, in_channels, dtype=old_weight.dtype)
    
    # Xavier init for new tokens
    torch.nn.init.xavier_uniform_(new_weight.view(new_out_channels, -1))
    new_weight = new_weight.view_as(new_weight)
    
    # Copy existing token weights (vocab tokens, not blank)
    new_weight[:old_vocab_size] = old_weight[:old_vocab_size]
    
    # Copy blank token weight to new position
    new_weight[new_vocab_size] = old_weight[old_vocab_size]
    
    ckpt[decoder_weight_key] = new_weight
    
    # Handle bias if present
    if decoder_bias_key and decoder_bias_key in ckpt:
        old_bias = ckpt[decoder_bias_key]
        new_bias = torch.zeros(new_out_channels, dtype=old_bias.dtype)
        new_bias[:old_vocab_size] = old_bias[:old_vocab_size]
        new_bias[new_vocab_size] = old_bias[old_vocab_size]
        ckpt[decoder_bias_key] = new_bias
        logger.info(f"Expanded bias: {len(old_bias)} -> {len(new_bias)}")
    
    logger.info(f"✓ Weights expanded successfully")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Expand Parakeet CTC vocabulary")
    parser.add_argument("--base-model", default="nvidia/parakeet-ctc-1.1b")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-chinese-chars", type=int, default=4000)
    args = parser.parse_args()
    
    import nemo.collections.asr as nemo_asr
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Parakeet CTC Vocabulary Expansion v2")
    print("=" * 70)
    
    # Step 1: Extract Chinese characters
    logger.info("\n[1/6] Extracting Chinese characters...")
    chinese_chars = extract_chinese_chars_from_manifest(args.manifest)
    logger.info(f"Found {len(chinese_chars)} unique Chinese characters")
    
    if len(chinese_chars) > args.max_chinese_chars:
        chinese_chars = chinese_chars[:args.max_chinese_chars]
        logger.info(f"Limited to top {args.max_chinese_chars}")
    
    # Step 2: Get the base model path
    logger.info("\n[2/6] Locating base model...")
    if args.base_model.startswith("nvidia/"):
        # Download and get path
        model = nemo_asr.models.ASRModel.from_pretrained(args.base_model)
        # Find the cached .nemo file
        from huggingface_hub import hf_hub_download
        model_name = args.base_model.split('/')[-1]
        base_nemo_path = hf_hub_download(
            repo_id=args.base_model,
            filename=f"{model_name}.nemo"
        )
        old_vocab_size = model.tokenizer.vocab_size
        tokenizer = model.tokenizer.tokenizer
        del model
        torch.cuda.empty_cache()
    else:
        base_nemo_path = args.base_model
        # Need to load model to get vocab size
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.base_model)
        old_vocab_size = model.tokenizer.vocab_size
        tokenizer = model.tokenizer.tokenizer
        del model
        torch.cuda.empty_cache()
    
    logger.info(f"Base model: {base_nemo_path}")
    logger.info(f"Original vocab size: {old_vocab_size}")
    
    # Step 3: Get existing vocabulary
    logger.info("\n[3/6] Building expanded vocabulary...")
    existing_tokens = set()
    for i in range(tokenizer.get_piece_size()):
        existing_tokens.add(tokenizer.id_to_piece(i))
    
    new_chars = [c for c in chinese_chars if c not in existing_tokens]
    logger.info(f"New characters to add: {len(new_chars)}")
    
    old_vocab = [tokenizer.id_to_piece(i) for i in range(old_vocab_size)]
    new_vocab = old_vocab + new_chars
    new_vocab_size = len(new_vocab)
    logger.info(f"New vocab size: {new_vocab_size}")
    
    # Step 4: Extract and modify the .nemo archive
    logger.info("\n[4/6] Modifying model archive...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract
        with tarfile.open(base_nemo_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # List contents
        contents = list(tmpdir.iterdir())
        logger.info(f"Archive contents: {[c.name for c in contents]}")
        
        # Load and modify checkpoint
        ckpt_path = tmpdir / 'model_weights.ckpt'
        logger.info(f"\n[5/6] Expanding decoder weights...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = expand_decoder_weights(ckpt, old_vocab_size, new_vocab_size)
        torch.save(ckpt, ckpt_path)
        
        # Modify config
        logger.info("\n[6/6] Updating configuration...")
        config_path = tmpdir / 'model_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # NeMo CTC expects num_classes = len(vocabulary) + 1 (blank is added internally)
        # The vocabulary list should NOT include blank
        config['decoder']['num_classes'] = new_vocab_size
        config['decoder']['vocabulary'] = new_vocab
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"  Updated decoder.num_classes: {old_vocab_size + 1} -> {new_vocab_size + 1}")
        logger.info(f"  Updated decoder.vocabulary: {old_vocab_size} -> {new_vocab_size}")
        
        # Repack
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in tmpdir.iterdir():
                tar.add(item, arcname=item.name)
    
    # Save character mapping
    mapping_path = output_path.parent / "chinese_char_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "new_chars": new_chars,
            "char_to_id": {char: old_vocab_size + i for i, char in enumerate(new_chars)}
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Vocabulary expansion complete!")
    print("=" * 70)
    print(f"  Original vocab: {old_vocab_size}")
    print(f"  Added Chinese chars: {len(new_chars)}")
    print(f"  New vocab size: {new_vocab_size}")
    print(f"  Model saved to: {output_path}")
    print(f"  Char mapping: {mapping_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
