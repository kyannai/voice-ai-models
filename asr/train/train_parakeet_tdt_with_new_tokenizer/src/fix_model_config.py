#!/usr/bin/env python3
"""
Fix the config mismatch in the modified .nemo model.

The modify_architecture.py script resized the layers correctly but the config
inside the .nemo file still has the old num_classes value.

Usage:
    python src/fix_model_config.py --model ./models/parakeet-tdt-0.6b-multilingual-init.nemo
"""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml


def fix_nemo_config(model_path: str):
    """Fix the config inside a .nemo file."""
    model_path = Path(model_path)
    backup_path = model_path.with_suffix('.nemo.backup')
    
    print(f"Fixing config in: {model_path}")
    
    # Create backup
    shutil.copy2(model_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract .nemo archive
        print("Extracting .nemo archive...")
        with tarfile.open(model_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Read and fix config
        config_path = tmpdir / "model_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.yaml not found in {model_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get vocab size from tokenizer config
        vocab_size = config.get('tokenizer', {}).get('vocab_size', None)
        if vocab_size is None:
            # Try to load tokenizer to get vocab size
            import sentencepiece as spm
            for f in tmpdir.iterdir():
                if f.suffix == '.model':
                    sp = spm.SentencePieceProcessor()
                    sp.load(str(f))
                    vocab_size = sp.vocab_size()
                    print(f"Got vocab size from tokenizer: {vocab_size}")
                    break
        
        if vocab_size is None:
            raise ValueError("Could not determine vocab size")
        
        # Calculate correct values
        # TDT: 5 durations [0,1,2,3,4]
        num_durations = 5
        decoder_vocab = vocab_size + 1  # vocab + blank for embedding
        joint_num_classes = vocab_size + num_durations + 1  # vocab + durations + blank
        
        print(f"\nVocab size: {vocab_size}")
        print(f"Expected decoder.vocab_size: {vocab_size}")  # NeMo uses vocab_size without blank
        print(f"Expected joint.num_classes: {joint_num_classes}")
        
        # Show current values
        print(f"\nCurrent config values:")
        print(f"  decoder.vocab_size: {config.get('decoder', {}).get('vocab_size', 'NOT SET')}")
        print(f"  joint.num_classes: {config.get('joint', {}).get('num_classes', 'NOT SET')}")
        
        # Fix the config
        if 'decoder' in config:
            config['decoder']['vocab_size'] = vocab_size
        
        if 'joint' in config:
            config['joint']['num_classes'] = joint_num_classes
            # Remove vocab_size if it exists (it shouldn't be here for TDT)
            if 'vocab_size' in config['joint']:
                del config['joint']['vocab_size']
        
        print(f"\nFixed config values:")
        print(f"  decoder.vocab_size: {config['decoder']['vocab_size']}")
        print(f"  joint.num_classes: {config['joint']['num_classes']}")
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Re-create the .nemo archive
        print(f"\nRe-creating .nemo archive...")
        
        # Determine archive format (NeMo typically uses uncompressed tar)
        with tarfile.open(model_path, 'w:') as tar:
            for item in tmpdir.iterdir():
                tar.add(item, arcname=item.name)
        
        print(f"\n✓ Fixed model saved to: {model_path}")
        print(f"  Backup at: {backup_path}")


def verify_fix(model_path: str):
    """Verify the fix worked."""
    print("\n" + "=" * 60)
    print("Verifying fix...")
    print("=" * 60)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import nemo.collections.asr as nemo_asr
    
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        model_path, 
        map_location='cpu'
    )
    
    vocab = model.tokenizer.vocab_size
    expected_joint = vocab + 5 + 1
    actual_joint_config = model.cfg.joint.num_classes
    
    print(f"Tokenizer vocab: {vocab}")
    print(f"cfg.joint.num_classes: {actual_joint_config}")
    print(f"Expected: {expected_joint}")
    
    if actual_joint_config == expected_joint:
        print("\n✓ Config is now correct!")
        return True
    else:
        print("\n✗ Config still has mismatch!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix config mismatch in .nemo model")
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo model")
    parser.add_argument("--verify", action="store_true", help="Verify after fixing")
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return 1
    
    fix_nemo_config(args.model)
    
    if args.verify:
        success = verify_fix(args.model)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    exit(main())
