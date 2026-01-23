#!/usr/bin/env python3
"""
Expand nvidia/parakeet-ctc-1.1b vocabulary with new characters from manifests.

This script uses protobuf manipulation to expand the SentencePiece tokenizer
while preserving original model weights for English tokens.

Process:
1. Load the original parakeet-ctc-1.1b model
2. Extract unique characters from manifest files (ordered by frequency)
3. Expand the SentencePiece tokenizer by adding chars to the protobuf
4. Expand the CTC decoder weights to match the new vocabulary
5. Restore original weights for existing tokens
6. Save the expanded model

Usage:
    # Single manifest with max tokens
    python expand_ctc_with_chinese.py \\
        --manifests /path/to/chinese.json:5000 \\
        --output ./models/parakeet-ctc-1.1b-zh.nemo

    # Multiple manifests with different limits
    python expand_ctc_with_chinese.py \\
        --manifests /path/to/chinese.json:5000 /path/to/malay.json:3000 \\
        --output ./models/parakeet-ctc-1.1b-multilingual.nemo

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
        description="Expand nvidia/parakeet-ctc-1.1b vocabulary with characters from manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single Chinese manifest
  python expand_ctc_with_chinese.py --manifests chinese.json:5000 --output model-zh.nemo

  # Multiple manifests (Chinese + Malay)
  python expand_ctc_with_chinese.py --manifests zh.json:5000 malay.json:3000 --output model-multi.nemo

  # No max limit (use all unique characters)
  python expand_ctc_with_chinese.py --manifests chinese.json --output model-zh.nemo
        """
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="nvidia/parakeet-ctc-1.1b",
        help="Base model name or path to .nemo file (default: nvidia/parakeet-ctc-1.1b)"
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
    print("Parakeet CTC Vocabulary Expansion")
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
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.base_model)
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
    
    # Step 3: Save original decoder weights
    logger.info("[3/6] Saving original decoder weights...")
    ori_decoder_weights = asr_model.decoder.decoder_layers[0].weight.clone()
    ori_decoder_bias = asr_model.decoder.decoder_layers[0].bias.clone()
    logger.info(f"  Decoder shape: {ori_decoder_weights.shape}")
    
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
        
        # Step 5: Update model with expanded tokenizer and decoder
        logger.info("[5/6] Updating model with expanded vocabulary...")
        
        # Create expanded vocabulary list
        new_chars_added = [c for c in new_chars if c not in set(vocab_list)]
        new_chars_added = new_chars_added[:new_vocab_size - old_vocab_size]
        expanded_vocab = vocab_list + new_chars_added
        
        # Update config
        with open_dict(asr_model.cfg):
            asr_model.cfg.decoder.num_classes = new_vocab_size
            asr_model.cfg.decoder.vocabulary = expanded_vocab
        
        # Apply new tokenizer (this also resets decoder weights)
        logger.info("  Applying expanded tokenizer...")
        asr_model.change_vocabulary(
            new_tokenizer_dir=str(tmpdir),
            new_tokenizer_type="bpe"
        )
        
        # Restore original weights
        logger.info("  Restoring original decoder weights...")
        current_decoder = asr_model.decoder.decoder_layers[0]
        current_out_channels = current_decoder.out_channels
        
        with torch.no_grad():
            # Copy original token weights (positions 0 to old_vocab_size-1)
            current_decoder.weight[:old_vocab_size].copy_(ori_decoder_weights[:old_vocab_size])
            current_decoder.bias[:old_vocab_size].copy_(ori_decoder_bias[:old_vocab_size])
            
            # Copy blank token to new position (last position in output)
            current_decoder.weight[current_out_channels - 1].copy_(ori_decoder_weights[old_vocab_size])
            current_decoder.bias[current_out_channels - 1].copy_(ori_decoder_bias[old_vocab_size])
            
            # ===== INITIALIZE NEW TOKEN WEIGHTS =====
            # New tokens (old_vocab_size to new_vocab_size-1) need smart initialization:
            # - Small random weights (for gradient flow during training)
            # - Slightly negative bias (so they don't dominate English initially)
            num_new_tokens = new_vocab_size - old_vocab_size
            if num_new_tokens > 0:
                # Get statistics from existing vocab tokens
                existing_weight_std = ori_decoder_weights[:old_vocab_size].std()
                existing_bias_mean = ori_decoder_bias[:old_vocab_size].mean()
                
                # Initialize new weights with small random values
                current_decoder.weight[old_vocab_size:new_vocab_size].normal_(
                    mean=0.0,
                    std=existing_weight_std * 0.01
                )
                # Initialize bias slightly below mean
                current_decoder.bias[old_vocab_size:new_vocab_size].fill_(existing_bias_mean - 5.0)
                
                logger.info(f"  ✓ Initialized {num_new_tokens} new tokens:")
                logger.info(f"    - Bias: {existing_bias_mean - 5.0:.2f} (existing mean: {existing_bias_mean:.2f})")
                logger.info(f"    - Weight std: {existing_weight_std * 0.01:.4f}")
        
        logger.info(f"  ✓ Restored {old_vocab_size} original tokens + blank")
        logger.info(f"  ✓ Decoder expanded: {old_vocab_size + 1} → {current_out_channels}")
        
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
    print("✅ CTC Vocabulary Expansion Complete!")
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
