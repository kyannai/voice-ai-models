#!/usr/bin/env python3
"""
Diagnostic script to check model architecture for tokenizer/embedding mismatches.

Usage:
    python src/check_model_architecture.py --model ./models/parakeet-tdt-0.6b-multilingual-init.nemo
"""

import argparse
import sys
from pathlib import Path

import torch


def check_model(model_path: str):
    """Check model architecture for consistency."""
    import nemo.collections.asr as nemo_asr
    
    print("=" * 70)
    print("Model Architecture Diagnostic")
    print("=" * 70)
    print(f"Model: {model_path}")
    print()
    
    # Load model
    print("Loading model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    
    # 1. Check tokenizer
    print("\n" + "=" * 70)
    print("1. TOKENIZER")
    print("=" * 70)
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        print(f"  Vocab size: {vocab_size}")
        
        # Test encoding
        test_texts = [
            "hello world",
            "你好世界",  # Chinese
            "selamat pagi",  # Malay
        ]
        print(f"\n  Test encodings:")
        for text in test_texts:
            try:
                tokens = tokenizer.text_to_ids(text)
                decoded = tokenizer.ids_to_text(tokens)
                print(f"    '{text}' -> {tokens[:10]}{'...' if len(tokens) > 10 else ''} -> '{decoded}'")
            except Exception as e:
                print(f"    '{text}' -> ERROR: {e}")
    else:
        print("  WARNING: No tokenizer found!")
        vocab_size = None
    
    # 2. Check decoder embedding
    print("\n" + "=" * 70)
    print("2. DECODER EMBEDDING")
    print("=" * 70)
    decoder_embed_size = None
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        if hasattr(decoder, 'prediction') and hasattr(decoder.prediction, 'embed'):
            embed = decoder.prediction.embed
            decoder_embed_size = embed.weight.shape[0]
            embed_dim = embed.weight.shape[1]
            print(f"  decoder.prediction.embed: {decoder_embed_size} x {embed_dim}")
        elif hasattr(decoder, 'embedding'):
            embed = decoder.embedding
            decoder_embed_size = embed.weight.shape[0]
            embed_dim = embed.weight.shape[1]
            print(f"  decoder.embedding: {decoder_embed_size} x {embed_dim}")
        elif hasattr(decoder, 'embed'):
            embed = decoder.embed
            decoder_embed_size = embed.weight.shape[0]
            embed_dim = embed.weight.shape[1]
            print(f"  decoder.embed: {decoder_embed_size} x {embed_dim}")
        else:
            print("  WARNING: Could not find decoder embedding!")
            print(f"  Available attributes: {[a for a in dir(decoder) if not a.startswith('_')]}")
    
    # 3. Check joint network output
    print("\n" + "=" * 70)
    print("3. JOINT NETWORK OUTPUT")
    print("=" * 70)
    joint_output_size = None
    if hasattr(model, 'joint'):
        joint = model.joint
        if hasattr(joint, 'joint_net'):
            joint_net = joint.joint_net
            if hasattr(joint_net, '__iter__'):
                for i, layer in enumerate(joint_net):
                    if hasattr(layer, 'out_features'):
                        print(f"  joint.joint_net[{i}]: Linear({layer.in_features} -> {layer.out_features})")
                        joint_output_size = layer.out_features
            else:
                print(f"  joint.joint_net type: {type(joint_net)}")
        if hasattr(joint, 'num_classes'):
            print(f"  joint.num_classes (config): {joint.num_classes}")
    
    # 4. Check model config
    print("\n" + "=" * 70)
    print("4. MODEL CONFIG")
    print("=" * 70)
    if hasattr(model, 'cfg'):
        cfg = model.cfg
        if hasattr(cfg, 'decoder'):
            print(f"  decoder.vocab_size (config): {cfg.decoder.get('vocab_size', 'NOT SET')}")
        if hasattr(cfg, 'joint'):
            print(f"  joint.num_classes (config): {cfg.joint.get('num_classes', 'NOT SET')}")
        if hasattr(cfg, 'tokenizer'):
            print(f"  tokenizer.vocab_size (config): {cfg.tokenizer.get('vocab_size', 'NOT SET')}")
    
    # 5. Consistency check
    print("\n" + "=" * 70)
    print("5. CONSISTENCY CHECK")
    print("=" * 70)
    
    # For TDT models:
    # - decoder_embed_size should be vocab_size + 1 (blank)
    # - joint_output_size should be vocab_size + num_durations + 1
    # Default TDT has 5 durations [0,1,2,3,4]
    num_durations = 5
    
    expected_decoder_embed = vocab_size + 1 if vocab_size else None
    expected_joint_output = vocab_size + num_durations + 1 if vocab_size else None
    
    all_ok = True
    
    if vocab_size and decoder_embed_size:
        if decoder_embed_size == expected_decoder_embed:
            print(f"  ✓ Decoder embedding size ({decoder_embed_size}) matches vocab+1 ({expected_decoder_embed})")
        else:
            print(f"  ✗ MISMATCH: Decoder embedding ({decoder_embed_size}) != vocab+1 ({expected_decoder_embed})")
            all_ok = False
    
    if vocab_size and joint_output_size:
        if joint_output_size == expected_joint_output:
            print(f"  ✓ Joint output size ({joint_output_size}) matches vocab+durations+1 ({expected_joint_output})")
        else:
            print(f"  ✗ MISMATCH: Joint output ({joint_output_size}) != vocab+durations+1 ({expected_joint_output})")
            all_ok = False
    
    # 6. Check if blank token is properly configured
    print("\n" + "=" * 70)
    print("6. BLANK TOKEN CHECK")
    print("=" * 70)
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'blank_idx'):
        print(f"  decoder.blank_idx: {model.decoder.blank_idx}")
    if hasattr(model, 'joint') and hasattr(model.joint, 'blank_idx'):
        print(f"  joint.blank_idx: {model.joint.blank_idx}")
    if hasattr(model, 'loss') and hasattr(model.loss, 'blank'):
        print(f"  loss.blank: {model.loss.blank}")
    
    # 7. Check decoding config
    print("\n" + "=" * 70)
    print("7. DECODING CONFIG")
    print("=" * 70)
    if hasattr(model, 'decoding') and hasattr(model.decoding, 'cfg'):
        dec_cfg = model.decoding.cfg
        print(f"  strategy: {dec_cfg.get('strategy', 'NOT SET')}")
        if hasattr(dec_cfg, 'tdt_decoder'):
            tdt = dec_cfg.tdt_decoder
            print(f"  durations: {tdt.get('durations', 'NOT SET')}")
    
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All checks passed - architecture looks consistent")
    else:
        print("✗ ARCHITECTURE MISMATCH DETECTED - see above for details")
    print("=" * 70)
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check model architecture for consistency")
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo model file")
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    success = check_model(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
