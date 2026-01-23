#!/usr/bin/env python3
"""
Shared utilities for expanding NeMo ASR model vocabularies.

This module provides common functions used by both CTC and TDT expansion scripts:
- Extract characters from manifest files (any language)
- Expand SentencePiece tokenizers via protobuf manipulation

Usage:
    from tokenizer_expansion import (
        extract_chars_from_manifests,
        expand_sentencepiece_model,
        parse_manifest_args,
    )
"""

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Set, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Common Unicode ranges for different scripts
UNICODE_RANGES = {
    'cjk': [
        ('\u4e00', '\u9fff'),   # CJK Unified Ideographs
        ('\u3400', '\u4dbf'),   # CJK Extension A
        ('\u20000', '\u2a6df'), # CJK Extension B
        ('\uf900', '\ufaff'),   # CJK Compatibility Ideographs
    ],
    'hangul': [
        ('\uac00', '\ud7af'),   # Hangul Syllables
        ('\u1100', '\u11ff'),   # Hangul Jamo
    ],
    'hiragana': [('\u3040', '\u309f')],
    'katakana': [('\u30a0', '\u30ff')],
    'thai': [('\u0e00', '\u0e7f')],
    'arabic': [('\u0600', '\u06ff')],
    'devanagari': [('\u0900', '\u097f')],  # Hindi
    'tamil': [('\u0b80', '\u0bff')],
    'latin_extended': [
        ('\u0100', '\u017f'),   # Latin Extended-A
        ('\u0180', '\u024f'),   # Latin Extended-B
        ('\u1e00', '\u1eff'),   # Latin Extended Additional
    ],
}


def is_in_ranges(char: str, ranges: List[Tuple[str, str]]) -> bool:
    """Check if a character is within any of the specified Unicode ranges."""
    for start, end in ranges:
        if start <= char <= end:
            return True
    return False


def parse_manifest_args(manifest_specs: List[str]) -> List[Tuple[str, int]]:
    """
    Parse manifest specifications in format "path:max_tokens" or just "path".
    
    Args:
        manifest_specs: List of strings like ["zh.json:5000", "malay.json:3000"]
    
    Returns:
        List of (path, max_tokens) tuples. max_tokens is None if not specified.
    
    Examples:
        ["train.json:5000"]           -> [("train.json", 5000)]
        ["train.json"]                -> [("train.json", None)]
        ["zh.json:5000", "my.json:3000"] -> [("zh.json", 5000), ("my.json", 3000)]
    """
    result = []
    for spec in manifest_specs:
        if ':' in spec:
            # Check if it's a Windows path (e.g., C:\path) or a port spec
            parts = spec.rsplit(':', 1)
            if len(parts) == 2 and parts[1].isdigit():
                path, max_tokens = parts[0], int(parts[1])
            else:
                path, max_tokens = spec, None
        else:
            path, max_tokens = spec, None
        result.append((path, max_tokens))
    return result


def extract_chars_from_manifest(
    manifest_path: str,
    existing_tokens: Optional[Set[str]] = None,
    max_chars: Optional[int] = None,
    line_limit: Optional[int] = None,
    scripts: Optional[List[str]] = None,
) -> Tuple[List[str], Counter]:
    """
    Extract unique characters from a manifest file, ordered by frequency.
    
    Args:
        manifest_path: Path to JSONL manifest file with 'text' field
        existing_tokens: Optional set of tokens already in vocabulary (to skip)
        max_chars: Maximum number of characters to return
        line_limit: Optional limit on number of lines to process (for testing)
        scripts: Optional list of script names to filter (e.g., ['cjk', 'latin_extended'])
                 If None, extracts all non-ASCII printable characters
    
    Returns:
        Tuple of (list of characters ordered by frequency, frequency Counter)
    """
    char_freq = Counter()
    
    # Build character filter if scripts specified
    filter_ranges = []
    if scripts:
        for script in scripts:
            if script in UNICODE_RANGES:
                filter_ranges.extend(UNICODE_RANGES[script])
    
    logger.info(f"Extracting characters from: {manifest_path}")
    if scripts:
        logger.info(f"  Filtering for scripts: {scripts}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {Path(manifest_path).name}")):
            if line_limit and i >= line_limit:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                for c in text:
                    # Skip if already in existing tokens
                    if existing_tokens and c in existing_tokens:
                        continue
                    
                    # If scripts specified, check if char is in those ranges
                    if filter_ranges:
                        if is_in_ranges(c, filter_ranges):
                            char_freq[c] += 1
                    else:
                        # Default: include any non-ASCII printable character
                        # Skip ASCII (already in English tokenizer)
                        # Skip control characters
                        if ord(c) > 127 and unicodedata.category(c)[0] != 'C':
                            char_freq[c] += 1
                            
            except json.JSONDecodeError:
                continue
    
    chars = [char for char, _ in char_freq.most_common()]
    
    if max_chars and len(chars) > max_chars:
        chars = chars[:max_chars]
        logger.info(f"  Limited to top {max_chars} characters by frequency")
    
    logger.info(f"  Found {len(chars):,} unique characters")
    return chars, char_freq


def extract_chars_from_manifests(
    manifest_specs: List[Tuple[str, Optional[int]]],
    existing_tokens: Optional[Set[str]] = None,
    line_limit: Optional[int] = None,
    scripts: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract characters from multiple manifest files.
    
    Args:
        manifest_specs: List of (path, max_tokens) tuples
        existing_tokens: Optional set of tokens already in vocabulary
        line_limit: Optional limit on lines per manifest (for testing)
        scripts: Optional list of script names to filter
    
    Returns:
        Combined list of characters, deduplicated, ordered by combined frequency
    """
    combined_freq = Counter()
    all_chars = []
    
    for manifest_path, max_chars in manifest_specs:
        chars, freq = extract_chars_from_manifest(
            manifest_path=manifest_path,
            existing_tokens=existing_tokens,
            max_chars=max_chars,
            line_limit=line_limit,
            scripts=scripts,
        )
        
        # Add to combined frequency (for ordering)
        combined_freq.update(freq)
        
        # Track chars from this manifest (respecting its max limit)
        for c in chars:
            if c not in existing_tokens if existing_tokens else True:
                if c not in all_chars:
                    all_chars.append(c)
    
    # Re-sort by combined frequency
    all_chars.sort(key=lambda c: combined_freq.get(c, 0), reverse=True)
    
    logger.info(f"Total unique characters from all manifests: {len(all_chars):,}")
    return all_chars


# Keep old function name for backwards compatibility
def extract_chinese_chars_from_manifest(manifest_path: str, limit: int = None) -> List[str]:
    """
    Extract unique Chinese characters from a manifest file, ordered by frequency.
    
    DEPRECATED: Use extract_chars_from_manifest() with scripts=['cjk'] instead.
    
    Args:
        manifest_path: Path to JSONL manifest file with 'text' field
        limit: Optional limit on number of lines to process (for testing)
    
    Returns:
        List of Chinese characters, ordered by frequency (most common first)
    """
    chars, _ = extract_chars_from_manifest(
        manifest_path=manifest_path,
        line_limit=limit,
        scripts=['cjk'],
    )
    return chars


def expand_sentencepiece_model(
    original_model_path: str,
    new_chars: List[str],
    output_dir: Path
) -> int:
    """
    Expand a SentencePiece model by adding new characters using protobuf manipulation.
    
    This function:
    1. Loads the original SentencePiece model
    2. Adds new characters as USER_DEFINED tokens with low scores
    3. Saves tokenizer.model, tokenizer.vocab, and vocab.txt
    
    Args:
        original_model_path: Path to original tokenizer.model file
        new_chars: List of new characters to add (duplicates auto-filtered)
        output_dir: Directory to save expanded tokenizer files
    
    Returns:
        New vocabulary size
    
    New tokens are added with:
        - Type: USER_DEFINED (always kept during encoding)
        - Score: -1000.0 - index * 0.001 (low priority, preserves frequency order)
    """
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    
    logger.info(f"Loading original tokenizer: {original_model_path}")
    
    # Load and parse the original model
    sp = spm.SentencePieceProcessor(model_file=original_model_path)
    model_proto = sp_pb2_model.ModelProto()
    model_proto.ParseFromString(sp.serialized_model_proto())
    
    original_vocab_size = len(model_proto.pieces)
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    
    # Get existing pieces for deduplication
    existing_pieces = {piece.piece for piece in model_proto.pieces}
    
    # Filter out characters that already exist
    new_chars_to_add = [c for c in new_chars if c not in existing_pieces]
    logger.info(f"Characters already in vocab: {len(new_chars) - len(new_chars_to_add)}")
    logger.info(f"New characters to add: {len(new_chars_to_add)}")
    
    # Add new Chinese characters as USER_DEFINED pieces with low scores
    base_score = -1000.0
    for i, char in enumerate(new_chars_to_add):
        new_piece = model_proto.pieces.add()
        new_piece.piece = char
        new_piece.score = base_score - i * 0.001  # Slightly decreasing scores
        new_piece.type = sp_pb2_model.ModelProto.SentencePiece.Type.USER_DEFINED
    
    new_vocab_size = len(model_proto.pieces)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    
    # Update trainer spec with new vocab size
    model_proto.trainer_spec.vocab_size = new_vocab_size
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer.model
    model_path = output_dir / "tokenizer.model"
    with open(model_path, 'wb') as f:
        f.write(model_proto.SerializeToString())
    logger.info(f"Saved: {model_path}")
    
    # Save tokenizer.vocab (with scores)
    vocab_path = output_dir / "tokenizer.vocab"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for piece in model_proto.pieces:
            f.write(f"{piece.piece}\t{piece.score}\n")
    logger.info(f"Saved: {vocab_path}")
    
    # Save vocab.txt (tokens only, NeMo artifact format)
    vocab_txt_path = output_dir / "vocab.txt"
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for piece in model_proto.pieces:
            f.write(f"{piece.piece}\n")
    logger.info(f"Saved: {vocab_txt_path}")
    
    return new_vocab_size
