#!/usr/bin/env python3
"""
Generate Malay G2P (Grapheme-to-Phoneme) dictionary from training corpus.

This script extracts all unique words from the training manifests and
generates IPA phonemes using espeak-ng. The resulting dictionary is used
by NeMo's G2P system during training and inference.

This is run ONCE during Phase 1 (Language Training) to create the dictionary
that gets packaged with the model.

Output format (NeMo IPA dictionary):
    WORD IPA_phonemes
    SELAMAT səlamat
    PAGI paɡi

Usage:
    python generate_g2p_dict.py --manifest data/manifests/train_manifest.json \
                                --output data/g2p/ipa_malay_dict.txt
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from phonemizer import phonemize
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Malay language code for espeak-ng
MALAY_LANG = "ms"

# Characters to strip from words
PUNCTUATION = r'[.,!?;:"\'\-\(\)\[\]{}…""''«»—–]'


def extract_words_from_manifest(manifest_path: str) -> Counter:
    """Extract all unique words from a manifest file.
    
    Args:
        manifest_path: Path to JSONL manifest file
        
    Returns:
        Counter of word frequencies
    """
    word_counts = Counter()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get('text', '')
            
            # Normalize and split into words
            text = text.lower().strip()
            # Remove punctuation
            text = re.sub(PUNCTUATION, ' ', text)
            # Split on whitespace
            words = text.split()
            
            for word in words:
                word = word.strip()
                if word and len(word) > 0:
                    word_counts[word] += 1
    
    return word_counts


def phonemize_word(word: str, language: str = MALAY_LANG) -> str:
    """Convert a single word to IPA phonemes.
    
    Args:
        word: Input word
        language: Language code for espeak-ng
        
    Returns:
        IPA phoneme string
    """
    try:
        result = phonemize(
            word,
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
        )
        return result.strip()
    except Exception as e:
        logger.warning(f"Failed to phonemize '{word}': {e}")
        return ""


def phonemize_batch(words: list[str], language: str = MALAY_LANG) -> dict[str, str]:
    """Phonemize a batch of words efficiently.
    
    Args:
        words: List of words to phonemize
        language: Language code for espeak-ng
        
    Returns:
        Dictionary mapping words to IPA phonemes
    """
    try:
        # Batch phonemize for efficiency
        results = phonemize(
            words,
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
        )
        
        # Handle single result vs list
        if isinstance(results, str):
            results = [results]
        
        # Map words to phonemes
        word_to_phoneme = {}
        for word, phoneme in zip(words, results):
            phoneme = phoneme.strip()
            if phoneme:
                word_to_phoneme[word] = phoneme
        
        return word_to_phoneme
        
    except Exception as e:
        logger.warning(f"Batch phonemization failed: {e}")
        # Fallback to individual processing
        word_to_phoneme = {}
        for word in words:
            phoneme = phonemize_word(word, language)
            if phoneme:
                word_to_phoneme[word] = phoneme
        return word_to_phoneme


def generate_g2p_dictionary(
    manifest_paths: list[str],
    output_path: str,
    min_word_freq: int = 1,
    batch_size: int = 1000,
) -> dict:
    """Generate G2P dictionary from manifest files.
    
    Args:
        manifest_paths: List of paths to JSONL manifest files
        output_path: Path to output dictionary file
        min_word_freq: Minimum word frequency to include
        batch_size: Batch size for phonemization
        
    Returns:
        Statistics dictionary
    """
    logger.info("Extracting words from manifests...")
    
    # Collect words from all manifests
    all_words = Counter()
    for manifest_path in manifest_paths:
        logger.info(f"  Processing: {manifest_path}")
        word_counts = extract_words_from_manifest(manifest_path)
        all_words.update(word_counts)
    
    # Filter by frequency
    words_to_process = [
        word for word, count in all_words.items()
        if count >= min_word_freq
    ]
    
    logger.info(f"Total unique words: {len(all_words)}")
    logger.info(f"Words with freq >= {min_word_freq}: {len(words_to_process)}")
    
    # Phonemize in batches
    logger.info("Generating phonemes...")
    dictionary = {}
    
    for i in tqdm(range(0, len(words_to_process), batch_size)):
        batch = words_to_process[i:i + batch_size]
        batch_results = phonemize_batch(batch)
        dictionary.update(batch_results)
    
    # Save dictionary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by word for consistency
    sorted_words = sorted(dictionary.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            phoneme = dictionary[word]
            # NeMo format: WORD phonemes (uppercase word)
            f.write(f"{word.upper()} {phoneme}\n")
    
    logger.info(f"Dictionary saved to: {output_path}")
    logger.info(f"Total entries: {len(dictionary)}")
    
    # Calculate statistics
    failed_count = len(words_to_process) - len(dictionary)
    
    stats = {
        'total_unique_words': len(all_words),
        'words_processed': len(words_to_process),
        'dictionary_entries': len(dictionary),
        'failed_words': failed_count,
        'output_path': str(output_path),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate Malay G2P dictionary from training corpus"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to manifest file(s) (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/g2p/ipa_malay_dict.txt",
        help="Output dictionary path",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum word frequency to include (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for phonemization (default: 1000)",
    )
    
    args = parser.parse_args()
    
    stats = generate_g2p_dictionary(
        manifest_paths=args.manifest,
        output_path=args.output,
        min_word_freq=args.min_freq,
        batch_size=args.batch_size,
    )
    
    logger.info("Generation complete!")
    logger.info(f"  Total words in corpus: {stats['total_unique_words']}")
    logger.info(f"  Dictionary entries: {stats['dictionary_entries']}")
    logger.info(f"  Failed words: {stats['failed_words']}")


if __name__ == "__main__":
    main()
