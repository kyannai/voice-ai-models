#!/usr/bin/env python3
"""
Expand nvidia/parakeet-tdt-0.6b-v3 vocabulary with new characters from manifests.

This script uses protobuf manipulation to expand the SentencePiece tokenizer
while preserving original model weights for English tokens.

TDT (Transducer) models have a different architecture than CTC:
- Decoder (prediction network) with embedding layer
- Joint network with output layer

Both need to be expanded to match the new vocabulary size.

Process:
1. Load the original parakeet-tdt-0.6b-v3 model
2. Extract unique characters from manifest files (ordered by frequency)
3. Expand the SentencePiece tokenizer by adding chars to the protobuf
4. Expand decoder embeddings and joint network output layer
5. Restore original weights for existing tokens
6. Save the expanded model

Usage:
    # Single manifest with max tokens
    python expand_tdt_with_chinese.py \\
        --manifests /path/to/chinese.json:5000 \\
        --output ./models/parakeet-tdt-0.6b-v3-zh.nemo

    # Multiple manifests with different limits
    python expand_tdt_with_chinese.py \\
        --manifests /path/to/chinese.json:5000 /path/to/malay.json:3000 \\
        --output ./models/parakeet-tdt-0.6b-v3-multilingual.nemo

The expanded model will produce IDENTICAL results on English audio.
Fine-tune on multilingual data to train the new token embeddings.

Dependencies:
    pip install nemo_toolkit[asr] sentencepiece torch tqdm protobuf
"""

import argparse
import json
import logging
import tarfile
import tempfile
from pathlib import Path

import torch

# Import shared tokenizer expansion utilities
from tokenizer_expansion import (
    extract_chars_from_manifests,
    expand_sentencepiece_model,
    parse_manifest_args,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Expand nvidia/parakeet-tdt-0.6b-v3 vocabulary with characters from manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single Chinese manifest
  python expand_tdt_with_chinese.py --manifests chinese.json:5000 --output model-zh.nemo

  # Multiple manifests (Chinese + Malay)
  python expand_tdt_with_chinese.py --manifests zh.json:5000 malay.json:3000 --output model-multi.nemo

  # No max limit (use all unique characters)
  python expand_tdt_with_chinese.py --manifests chinese.json --output model-zh.nemo
        """
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="Base model name or path to .nemo file (default: nvidia/parakeet-tdt-0.6b-v3)"
    )
    parser.add_argument(
        "--manifests",
        type=str,
        nargs='+',
        required=True,
        help="Manifest files in format 'path:max_tokens' or just 'path'. "
             "Example: --manifests chinese.json:5000 malay.json:3000"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for expanded .nemo model"
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=None,
        help="Maximum total characters to add across all manifests (optional)"
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=None,
        help="Limit manifest lines to process per file (for testing)"
    )
    parser.add_argument(
        "--test-audio",
        type=str,
        default=None,
        help="Audio file to transcribe before/after for comparison"
    )
    
    args = parser.parse_args()
    
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse manifest specifications
    manifest_specs = parse_manifest_args(args.manifests)
    
    print("=" * 70)
    print("Parakeet TDT Vocabulary Expansion")
    print("=" * 70)
    print(f"Base model:  {args.base_model}")
    print(f"Output:      {args.output}")
    print(f"Manifests:")
    for path, max_tokens in manifest_specs:
        limit_str = f" (max {max_tokens})" if max_tokens else " (no limit)"
        print(f"  - {path}{limit_str}")
    print("=" * 70)
    
    # Step 1: Extract characters from all manifests
    logger.info("[1/6] Extracting characters from manifests...")
    new_chars = extract_chars_from_manifests(
        manifest_specs=manifest_specs,
        line_limit=args.manifest_limit,
    )
    
    if args.max_total_chars and len(new_chars) > args.max_total_chars:
        new_chars = new_chars[:args.max_total_chars]
        logger.info(f"Limited to top {args.max_total_chars} characters total")
    
    # Step 2: Load the base model
    logger.info("[2/6] Loading base model...")
    if args.base_model.endswith('.nemo'):
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.base_model)
        nemo_path = args.base_model
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(args.base_model)
        from huggingface_hub import hf_hub_download
        model_name = args.base_model.split('/')[-1]
        nemo_path = hf_hub_download(repo_id=args.base_model, filename=f"{model_name}.nemo")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr_model = asr_model.to(device)
    
    old_vocab_size = asr_model.tokenizer.vocab_size
    logger.info(f"Original vocabulary size: {old_vocab_size}")
    
    # Get original vocab list
    vocab_list = [asr_model.tokenizer.tokenizer.id_to_piece(i) for i in range(old_vocab_size)]
    
    # Optional: transcribe with original model
    transcription_before = None
    if args.test_audio:
        logger.info(f"Transcribing test audio with ORIGINAL model...")
        try:
            asr_model.eval()
            result = asr_model.transcribe([args.test_audio])
            transcription_before = result[0].text if hasattr(result[0], 'text') else str(result[0])
            logger.info(f"  Result: {transcription_before}")
        except Exception as e:
            transcription_before = f"[Error: {e}]"
    
    # Step 3: Save original weights for decoder and joint network
    # change_vocabulary() reinitializes ALL decoder and joint weights, so we must save everything
    logger.info("[3/6] Saving original decoder and joint network weights...")
    
    # ===== DECODER PREDICTION NETWORK =====
    # Embedding: [vocab_size + 1, embed_dim] where +1 is for blank token
    ori_embed_weight = asr_model.decoder.prediction.embed.weight.clone()
    logger.info(f"  Decoder embed shape: {ori_embed_weight.shape}")
    logger.info(f"    - Vocab tokens: 0 to {old_vocab_size - 1}")
    logger.info(f"    - Blank token: {old_vocab_size}")
    
    # LSTM layers (must be preserved exactly)
    ori_decoder_state = {}
    for name, param in asr_model.decoder.prediction.named_parameters():
        if 'embed' not in name:  # Skip embedding, we handle it separately
            ori_decoder_state[name] = param.clone()
            logger.info(f"  Saved decoder param: {name} {param.shape}")
    
    # ===== JOINT NETWORK =====
    # Joint network output layer (last layer in joint_net)
    # Shape: [vocab_size + num_durations + 1, hidden_dim]
    # For TDT with durations [0,1,2,3,4], that's 5 durations + 1 blank = 6 extra tokens
    joint_layers = list(asr_model.joint.joint_net.children())
    last_layer_idx = len(joint_layers) - 1
    ori_joint_weight = asr_model.joint.joint_net[last_layer_idx].weight.clone()
    ori_joint_bias = asr_model.joint.joint_net[last_layer_idx].bias.clone()
    num_special_tokens = ori_joint_weight.shape[0] - old_vocab_size  # durations + blank
    logger.info(f"  Joint output shape: {ori_joint_weight.shape}")
    logger.info(f"    - Vocab tokens: 0 to {old_vocab_size - 1}")
    logger.info(f"    - Special tokens (durations+blank): {old_vocab_size} to {ori_joint_weight.shape[0] - 1} ({num_special_tokens} tokens)")
    
    # Other joint network layers (must be preserved exactly)
    ori_joint_state = {}
    for name, param in asr_model.joint.named_parameters():
        if 'joint_net.2' not in name:  # Skip output layer, we handle it separately
            ori_joint_state[name] = param.clone()
            logger.info(f"  Saved joint param: {name} {param.shape}")
    
    # Step 4: Expand tokenizer
    logger.info("[4/6] Expanding SentencePiece tokenizer...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract original .nemo archive
        with tarfile.open(nemo_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Find the tokenizer.model file
        tokenizer_model_files = list(tmpdir.glob("*tokenizer.model"))
        if not tokenizer_model_files:
            raise FileNotFoundError("Could not find tokenizer.model in .nemo archive")
        
        original_tokenizer_path = tokenizer_model_files[0]
        
        # Expand the tokenizer (creates tokenizer.model, tokenizer.vocab, vocab.txt)
        new_vocab_size = expand_sentencepiece_model(
            str(original_tokenizer_path),
            new_chars,
            tmpdir
        )
        
        # Step 5: Update model with expanded tokenizer
        logger.info("[5/6] Updating model with expanded vocabulary...")
        
        # Create expanded vocabulary list
        new_chars_added = [c for c in new_chars if c not in set(vocab_list)]
        new_chars_added = new_chars_added[:new_vocab_size - old_vocab_size]
        expanded_vocab = vocab_list + new_chars_added
        
        # Update config
        with open_dict(asr_model.cfg):
            asr_model.cfg.decoder.vocab_size = new_vocab_size
            asr_model.cfg.joint.vocabulary = expanded_vocab
            asr_model.cfg.joint.num_classes = new_vocab_size
        
        # Apply new tokenizer (this also resets decoder/joint weights)
        logger.info("  Applying expanded tokenizer...")
        asr_model.change_vocabulary(
            new_tokenizer_dir=str(tmpdir),
            new_tokenizer_type="bpe"
        )
        
        # Restore original weights
        logger.info("  Restoring original weights...")
        
        with torch.no_grad():
            # ===== RESTORE DECODER EMBEDDING =====
            # Original layout: [vocab (0 to old_vocab_size-1), blank (old_vocab_size)]
            # New layout: [vocab (0 to old_vocab_size-1), new_chars, blank (new_vocab_size)]
            new_embed = asr_model.decoder.prediction.embed
            new_num_embeddings = new_embed.num_embeddings
            
            # Copy original vocab embeddings (positions 0 to old_vocab_size-1)
            new_embed.weight[:old_vocab_size].copy_(ori_embed_weight[:old_vocab_size])
            
            # Copy blank token from old position (old_vocab_size) to new position (new_vocab_size)
            new_embed.weight[new_vocab_size].copy_(ori_embed_weight[old_vocab_size])
            
            logger.info(f"  ✓ Restored {old_vocab_size} decoder embeddings")
            logger.info(f"  ✓ Moved blank token: {old_vocab_size} → {new_vocab_size}")
            logger.info(f"  ✓ Decoder embed expanded: {ori_embed_weight.shape[0]} → {new_num_embeddings}")
            
            # ===== RESTORE DECODER LSTM WEIGHTS =====
            decoder_params = dict(asr_model.decoder.prediction.named_parameters())
            for name, ori_param in ori_decoder_state.items():
                if name in decoder_params:
                    decoder_params[name].copy_(ori_param)
            logger.info(f"  ✓ Restored {len(ori_decoder_state)} decoder LSTM parameters")
            
            # ===== RESTORE JOINT OUTPUT LAYER =====
            # Original layout: [vocab (0 to old_vocab_size-1), durations+blank (old_vocab_size to old_vocab_size+5)]
            # New layout: [vocab (0 to old_vocab_size-1), new_chars, durations+blank (new_vocab_size to new_vocab_size+5)]
            new_joint = asr_model.joint.joint_net[last_layer_idx]
            new_out_features = new_joint.out_features
            
            # Copy original vocab weights (positions 0 to old_vocab_size-1)
            new_joint.weight[:old_vocab_size].copy_(ori_joint_weight[:old_vocab_size])
            new_joint.bias[:old_vocab_size].copy_(ori_joint_bias[:old_vocab_size])
            
            # Copy duration and blank tokens
            # Original positions: [old_vocab_size : old_vocab_size + num_special_tokens]
            # New positions: [new_vocab_size : new_vocab_size + num_special_tokens]
            new_joint.weight[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
                ori_joint_weight[old_vocab_size:old_vocab_size + num_special_tokens]
            )
            new_joint.bias[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
                ori_joint_bias[old_vocab_size:old_vocab_size + num_special_tokens]
            )
            
            logger.info(f"  ✓ Restored {old_vocab_size} joint output weights")
            logger.info(f"  ✓ Moved special tokens: {old_vocab_size}-{old_vocab_size + num_special_tokens - 1} → {new_vocab_size}-{new_vocab_size + num_special_tokens - 1}")
            logger.info(f"  ✓ Joint expanded: {ori_joint_weight.shape[0]} → {new_out_features}")
            
            # ===== RESTORE OTHER JOINT NETWORK WEIGHTS =====
            joint_params = dict(asr_model.joint.named_parameters())
            for name, ori_param in ori_joint_state.items():
                if name in joint_params:
                    joint_params[name].copy_(ori_param)
            logger.info(f"  ✓ Restored {len(ori_joint_state)} other joint parameters")
            
            # ===== CRITICAL: INITIALIZE NEW TOKEN WEIGHTS TO VERY NEGATIVE OUTPUT =====
            # New tokens (8192 to new_vocab_size-1) have random weights that produce
            # higher logits than original vocab, causing them to be selected during decoding.
            # Set their weights to zero and bias to very negative so they never get selected.
            new_joint.weight[old_vocab_size:new_vocab_size].zero_()
            new_joint.bias[old_vocab_size:new_vocab_size].fill_(-1000.0)
            
            # Similarly for embedding, set new embeddings to zero (they need fine-tuning anyway)
            new_embed.weight[old_vocab_size:new_vocab_size].zero_()
            
            logger.info(f"  ✓ Initialized {new_vocab_size - old_vocab_size} new tokens with zero weights/neg bias")
        
        # Step 6: Save model
        logger.info("[6/6] Saving expanded model...")
        asr_model.save_to(str(output_path))
    
    # Save metadata
    mapping_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "base_model": args.base_model,
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "num_chars_added": len(new_chars_added),
            "manifests": [{"path": p, "max_tokens": m} for p, m in manifest_specs],
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ TDT Vocabulary Expansion Complete!")
    print("=" * 70)
    print(f"  Original vocab:  {old_vocab_size:,}")
    print(f"  New vocab size:  {new_vocab_size:,}")
    print(f"  Chars added:     {len(new_chars_added):,}")
    print(f"  Model saved to:  {output_path}")
    print(f"  Metadata:        {mapping_path}")
    print("=" * 70)
    print("\nNote: Fine-tune on target language data to train the new token embeddings.")
    print("      English performance is preserved (identical to original).")
    
    if args.test_audio:
        print("\n" + "=" * 70)
        print("🔊 TRANSCRIPTION COMPARISON")
        print("=" * 70)
        print(f"BEFORE: {transcription_before}")
        # Note: After transcription would require reloading the model
        print("(After transcription requires reloading the saved model)")
        print("=" * 70)


if __name__ == "__main__":
    main()
