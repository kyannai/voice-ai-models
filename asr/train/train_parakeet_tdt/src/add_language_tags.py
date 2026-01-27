#!/usr/bin/env python3
"""
Add language tags to training manifests for multilingual ASR.

This solves the language confusion problem by:
1. Adding language prefix tokens to each transcript
2. Teaching the model to condition on expected language

Usage:
    python src/add_language_tags.py \
        --manifest train_manifest.json \
        --output train_manifest_with_lang.json \
        --lang ms  # or zh, en, or auto

For auto-detection, install langdetect:
    pip install langdetect

After training with language tags, at inference time:
    1. Prepend expected language tag: "<|ms|>" for Malay
    2. Or auto-detect language first, then prepend tag
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

# Try to import langdetect for auto mode
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# Language tag mapping
LANG_TAGS = {
    'en': '<|en|>',
    'ms': '<|ms|>',
    'zh': '<|zh|>',
    'id': '<|id|>',  # Indonesian (similar to Malay)
    'ja': '<|ja|>',
}

# Characters that indicate specific languages
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
MALAY_WORDS = {'saya', 'nak', 'pergi', 'ke', 'tak', 'ada', 'dengan', 'untuk', 
               'yang', 'ini', 'itu', 'tidak', 'dia', 'kita', 'kami', 'mereka',
               'sudah', 'boleh', 'juga', 'lagi', 'bila', 'mana', 'apa', 'siapa',
               'macam', 'mau', 'mahu', 'buat', 'tengah', 'dekat', 'dalam'}


def detect_language_heuristic(text: str) -> str:
    """
    Detect language using heuristics (faster than langdetect).
    
    Returns: 'zh' for Chinese, 'ms' for Malay, 'en' for English
    """
    text_lower = text.lower()
    
    # Check for Chinese characters
    if CHINESE_PATTERN.search(text):
        # If mostly Chinese, return zh
        chinese_chars = len(CHINESE_PATTERN.findall(text))
        total_chars = len(text.replace(' ', ''))
        if chinese_chars / max(total_chars, 1) > 0.3:
            return 'zh'
    
    # Check for Malay-specific words
    words = set(text_lower.split())
    malay_matches = words.intersection(MALAY_WORDS)
    if len(malay_matches) >= 2:
        return 'ms'
    
    # Default to English
    return 'en'


def detect_language_auto(text: str) -> str:
    """
    Detect language using langdetect library (more accurate but slower).
    Falls back to heuristic if langdetect fails.
    """
    if not LANGDETECT_AVAILABLE:
        return detect_language_heuristic(text)
    
    try:
        lang = detect(text)
        # Map to our supported languages
        if lang in ['zh-cn', 'zh-tw', 'zh']:
            return 'zh'
        elif lang in ['ms', 'id']:  # Indonesian is close to Malay
            return 'ms'
        elif lang in ['en']:
            return 'en'
        else:
            # For unsupported languages, use heuristic
            return detect_language_heuristic(text)
    except LangDetectException:
        return detect_language_heuristic(text)


def add_language_tag(text: str, lang: Optional[str] = None) -> tuple[str, str]:
    """
    Add language tag to text.
    
    Args:
        text: Original transcript
        lang: Language code ('en', 'ms', 'zh') or None for auto-detection
        
    Returns:
        Tuple of (tagged_text, detected_language)
    """
    # Auto-detect if not specified
    if lang is None or lang == 'auto':
        lang = detect_language_auto(text)
    
    # Get tag
    tag = LANG_TAGS.get(lang, LANG_TAGS['en'])
    
    # Prepend tag to text
    tagged_text = f"{tag} {text}"
    
    return tagged_text, lang


def process_manifest(
    input_path: str, 
    output_path: str, 
    lang: Optional[str] = None,
    add_lang_field: bool = True
) -> dict:
    """
    Process a NeMo manifest file and add language tags.
    
    Args:
        input_path: Path to input manifest
        output_path: Path to output manifest
        lang: Fixed language code, or 'auto' for auto-detection
        add_lang_field: Whether to add a 'lang' field to each entry
        
    Returns:
        Statistics dict
    """
    stats = {'total': 0, 'en': 0, 'ms': 0, 'zh': 0, 'other': 0}
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            entry = json.loads(line)
            original_text = entry.get('text', '')
            
            # Add language tag
            tagged_text, detected_lang = add_language_tag(original_text, lang)
            entry['text'] = tagged_text
            
            # Optionally add lang field
            if add_lang_field:
                entry['lang'] = detected_lang
            
            # Write output
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Update stats
            stats['total'] += 1
            stats[detected_lang] = stats.get(detected_lang, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add language tags to ASR training manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect languages in manifest
    python src/add_language_tags.py \\
        --manifest train.json \\
        --output train_tagged.json \\
        --lang auto
    
    # Force all entries to be Malay
    python src/add_language_tags.py \\
        --manifest malay_only.json \\
        --output malay_tagged.json \\
        --lang ms
    
    # Process multiple manifests
    python src/add_language_tags.py \\
        --manifest data/ms/*.json \\
        --output-dir data/ms_tagged/ \\
        --lang ms
        """
    )
    parser.add_argument(
        '--manifest', '-m',
        type=str,
        required=True,
        help='Path to input manifest file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output manifest file'
    )
    parser.add_argument(
        '--lang', '-l',
        type=str,
        default='auto',
        choices=['auto', 'en', 'ms', 'zh', 'id', 'ja'],
        help='Language code or "auto" for auto-detection (default: auto)'
    )
    parser.add_argument(
        '--no-lang-field',
        action='store_true',
        help='Do not add "lang" field to manifest entries'
    )
    
    args = parser.parse_args()
    
    print(f"Processing: {args.manifest}")
    print(f"Output: {args.output}")
    print(f"Language: {args.lang}")
    print()
    
    if args.lang == 'auto' and not LANGDETECT_AVAILABLE:
        print("Warning: langdetect not installed, using heuristic detection")
        print("Install with: pip install langdetect")
        print()
    
    stats = process_manifest(
        args.manifest,
        args.output,
        args.lang if args.lang != 'auto' else None,
        add_lang_field=not args.no_lang_field
    )
    
    print(f"Processed {stats['total']} entries:")
    for lang, count in sorted(stats.items()):
        if lang != 'total' and count > 0:
            pct = count / stats['total'] * 100
            print(f"  {lang}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput saved to: {args.output}")
    print("\nNext steps:")
    print("1. Update your training config to use the tagged manifest")
    print("2. Retrain the model")
    print("3. At inference, prepend language tag: '<|ms|>' for Malay, '<|zh|>' for Chinese")


if __name__ == '__main__':
    main()
