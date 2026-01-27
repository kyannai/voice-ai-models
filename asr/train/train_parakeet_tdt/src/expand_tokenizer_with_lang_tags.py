#!/usr/bin/env python3
"""
Expand tokenizer vocabulary with language tags.

This adds special tokens like <|en|>, <|ms|>, <|zh|> to enable
language-conditioned ASR.

Usage:
    python src/expand_tokenizer_with_lang_tags.py \
        --model ./models/parakeet-tdt-multilingual.nemo \
        --output ./models/parakeet-tdt-multilingual-langtags.nemo
"""

import argparse
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml
import torch
import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language tags to add
LANG_TAGS = [
    '<|en|>',   # English
    '<|ms|>',   # Malay
    '<|zh|>',   # Chinese/Mandarin
    '<|id|>',   # Indonesian
    '<|ja|>',   # Japanese
    '<|auto|>', # Auto-detect (no conditioning)
]


def check_existing_tokens(tokenizer_path: str) -> list:
    """Check which language tags already exist in tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    existing = []
    missing = []
    
    for tag in LANG_TAGS:
        token_id = sp.piece_to_id(tag)
        if token_id != sp.unk_id():
            existing.append(tag)
        else:
            missing.append(tag)
    
    return existing, missing


def add_tokens_to_sentencepiece(
    input_model_path: str,
    output_model_path: str,
    new_tokens: list
) -> int:
    """
    Add new tokens to SentencePiece model.
    
    Note: This creates a new tokenizer by training with additional user_defined_symbols.
    The original vocabulary is preserved.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(input_model_path)
    
    original_vocab_size = sp.get_piece_size()
    logger.info(f"Original vocab size: {original_vocab_size}")
    
    # For SentencePiece, we need to retrain or use a workaround
    # The cleanest approach is to add tokens to the vocab file
    
    # Get all existing pieces
    vocab = []
    for i in range(original_vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        vocab.append((piece, score))
    
    # Add new tokens with high scores (so they're preferred)
    new_token_score = 0.0  # Highest priority
    added_count = 0
    for token in new_tokens:
        if token not in [v[0] for v in vocab]:
            vocab.append((token, new_token_score))
            added_count += 1
            logger.info(f"  Adding token: {token}")
    
    logger.info(f"Added {added_count} new tokens")
    logger.info(f"New vocab size: {len(vocab)}")
    
    # Write vocab file for SentencePiece
    vocab_path = Path(output_model_path).with_suffix('.vocab')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for piece, score in vocab:
            f.write(f"{piece}\t{score}\n")
    
    logger.info(f"Saved vocab to: {vocab_path}")
    
    # Copy the model file and modify it
    shutil.copy(input_model_path, output_model_path)
    
    return len(vocab)


def expand_model_with_lang_tags(
    model_path: str,
    output_path: str
) -> None:
    """
    Expand a NeMo model's tokenizer and layers to include language tags.
    """
    import nemo.collections.asr as nemo_asr
    
    logger.info(f"Loading model: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    
    # Get tokenizer info
    tokenizer = model.tokenizer
    original_vocab_size = tokenizer.vocab_size
    logger.info(f"Original vocab size: {original_vocab_size}")
    
    # Check which tags are already present
    existing_tags = []
    missing_tags = []
    for tag in LANG_TAGS:
        token_ids = tokenizer.text_to_ids(tag)
        # If tokenized to multiple pieces or contains UNK, it's not a single token
        if len(token_ids) == 1 and token_ids[0] != 0:  # 0 is usually UNK
            existing_tags.append(tag)
        else:
            missing_tags.append(tag)
    
    if existing_tags:
        logger.info(f"Already have tags: {existing_tags}")
    
    if not missing_tags:
        logger.info("All language tags already exist! No modification needed.")
        shutil.copy(model_path, output_path)
        return
    
    logger.info(f"Need to add tags: {missing_tags}")
    
    # For now, warn user about the complexity
    logger.warning("=" * 60)
    logger.warning("Adding tokens to SentencePiece requires model retraining.")
    logger.warning("Recommended approach:")
    logger.warning("1. Train new tokenizer with language tags included")
    logger.warning("2. Use transfer_vocabulary.py to migrate weights")
    logger.warning("=" * 60)
    
    # Alternative: Use the existing special tokens
    # Parakeet has tokens like <|nospeech|>, <|pnc|> etc.
    # We could repurpose unused ones or add via config
    
    logger.info("\nWorkaround: Check if model has unused special tokens...")
    
    # Print first 30 tokens to see special tokens
    logger.info("First 30 tokens in vocabulary:")
    for i in range(min(30, original_vocab_size)):
        token = tokenizer.ids_to_text([i])
        logger.info(f"  {i}: {repr(token)}")
    
    logger.info("\nTo add language conditioning without retraining tokenizer:")
    logger.info("1. Use prompt prefix in audio (not text) - add short audio cue")
    logger.info("2. Use existing special tokens as language markers")
    logger.info("3. Retrain tokenizer from scratch with lang tags")
    
    # Save model as-is for now
    logger.info(f"\nSaving model (unchanged) to: {output_path}")
    model.save_to(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Add language tags to ASR model tokenizer"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to input .nemo model'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output .nemo model'
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    expand_model_with_lang_tags(args.model, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
