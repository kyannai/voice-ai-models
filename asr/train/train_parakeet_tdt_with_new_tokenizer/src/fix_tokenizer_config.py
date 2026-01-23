#!/usr/bin/env python3
"""
Fix tokenizer configuration in the TDT model.

The transfer_vocabulary.py script didn't properly update all tokenizer references.
This script fixes the .nemo archive to point to the correct tokenizer files.

Usage:
    python src/fix_tokenizer_config.py \
        --model ./models/parakeet-tdt-0.6b-multilingual-transferred.nemo \
        --output ./models/parakeet-tdt-0.6b-multilingual-fixed.nemo
"""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Fix tokenizer config in TDT model")
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Fixing TDT Tokenizer Configuration")
    print("=" * 70)
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract archive
        print(f"\n[1/4] Extracting {model_path.name}...")
        with tarfile.open(model_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        contents = list(tmpdir.iterdir())
        print(f"  Contents: {[c.name for c in contents]}")
        
        # Find tokenizer files
        print("\n[2/4] Identifying tokenizer files...")
        tokenizer_model = None
        tokenizer_vocab = None
        old_vocab_files = []
        
        for f in contents:
            if f.name == 'tokenizer_multilingual.model':
                tokenizer_model = f.name
            elif f.name == 'tokenizer_multilingual.vocab':
                tokenizer_vocab = f.name
            elif '_tokenizer.vocab' in f.name or '_vocab.txt' in f.name:
                old_vocab_files.append(f.name)
        
        print(f"  New tokenizer model: {tokenizer_model}")
        print(f"  New tokenizer vocab: {tokenizer_vocab}")
        print(f"  Old vocab files to remove: {old_vocab_files}")
        
        if not tokenizer_model or not tokenizer_vocab:
            raise ValueError("Multilingual tokenizer files not found!")
        
        # Generate vocab.txt from tokenizer.vocab for TDT compatibility
        print("\n[2.5/4] Generating vocab.txt for TDT compatibility...")
        vocab_txt_name = "tokenizer_multilingual_vocab.txt"
        vocab_txt_path = tmpdir / vocab_txt_name
        
        # Read SentencePiece vocab and create vocab.txt
        # Skip special tokens (first ~260 in multilingual tokenizer)
        with open(tmpdir / tokenizer_vocab, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        vocab_tokens = []
        for line in lines:
            parts = line.strip().split('\t')
            if parts:
                token = parts[0]
                # Skip special tokens like <pad>, <unk>, <0xXX>
                if token.startswith('<') and token.endswith('>'):
                    continue
                vocab_tokens.append(token)
        
        with open(vocab_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_tokens))
        
        print(f"  Created: {vocab_txt_name} ({len(vocab_tokens)} tokens)")
        
        # Update config
        print("\n[3/4] Updating model_config.yaml...")
        config_path = tmpdir / "model_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        old_tokenizer_cfg = config.get('tokenizer', {})
        print(f"  Old config:")
        print(f"    model_path: {old_tokenizer_cfg.get('model_path')}")
        print(f"    spe_tokenizer_vocab: {old_tokenizer_cfg.get('spe_tokenizer_vocab')}")
        print(f"    vocab_path: {old_tokenizer_cfg.get('vocab_path')}")
        
        # Update tokenizer section
        # NeMo TDT requires 'dir' field - use '.' for relative paths within archive
        config['tokenizer']['dir'] = '.'
        config['tokenizer']['model_path'] = f'nemo:{tokenizer_model}'
        config['tokenizer']['spe_tokenizer_vocab'] = f'nemo:{tokenizer_vocab}'
        config['tokenizer']['vocab_path'] = f'nemo:{vocab_txt_name}'
        
        print(f"  New config:")
        print(f"    model_path: {config['tokenizer'].get('model_path')}")
        print(f"    spe_tokenizer_vocab: {config['tokenizer'].get('spe_tokenizer_vocab')}")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Remove old vocab files
        print("\n[4/4] Cleaning up old vocab files...")
        for old_file in old_vocab_files:
            old_path = tmpdir / old_file
            if old_path.exists():
                old_path.unlink()
                print(f"  Removed: {old_file}")
        
        # Repack
        print(f"\nRepacking to {output_path}...")
        with tarfile.open(output_path, 'w:') as tar:
            for item in tmpdir.iterdir():
                tar.add(item, arcname=item.name)
    
    print("\n" + "=" * 70)
    print("✓ Tokenizer configuration fixed!")
    print("=" * 70)
    print(f"  Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
