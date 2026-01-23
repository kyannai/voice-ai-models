#!/usr/bin/env python3
"""
Test tokenizer language support for a given .nemo model.

This script loads a .nemo model's tokenizer and tests whether it can properly
tokenize Chinese (cn), Malay (ms), and English (en) text.

Supports both:
- HuggingFace model names (e.g., nvidia/parakeet-tdt-0.6b-v3)
- Local .nemo file paths (e.g., /path/to/model.nemo)

A language is considered "supported" if:
1. The tokenizer can encode the text without errors
2. The text can be decoded back (may differ due to tokenization)
3. Most characters are represented as known tokens (not <unk>)

Usage:
    python test_tokenizer_languages.py nvidia/parakeet-tdt-0.6b-v3
    python test_tokenizer_languages.py /path/to/model.nemo
    python test_tokenizer_languages.py nvidia/parakeet-tdt-0.6b-v3 --verbose
    python test_tokenizer_languages.py /path/to/model.nemo --json

Dependencies:
    pip install sentencepiece huggingface_hub
"""

import argparse
import json
import logging
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test sentences for each language
TEST_SENTENCES = {
    "en": [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming technology.",
        "Please enter your password to continue.",
        "Thank you for your patience and understanding.",
    ],
    "ms": [
        "Selamat pagi, apa khabar?",
        "Terima kasih kerana sudi datang ke majlis ini.",
        "Saya ingin memesan nasi lemak dan teh tarik.",
        "Kuala Lumpur adalah ibu negara Malaysia.",
        "Sila masukkan kata laluan anda untuk teruskan.",
    ],
    "cn": [
        "你好，今天怎么样？",
        "机器学习正在改变世界。",
        "请输入您的密码以继续。",
        "北京是中国的首都。",
        "谢谢您的耐心和理解。",
    ],
}

# Display names for languages
LANGUAGE_NAMES = {
    "en": "English",
    "ms": "Malay",
    "cn": "Chinese",
}


def is_huggingface_model(model_path: str) -> bool:
    """Check if the path looks like a HuggingFace model name."""
    # HuggingFace models have format: org/model-name
    # Local paths typically have / with more segments, or end with .nemo
    if model_path.endswith('.nemo'):
        return False
    if '/' in model_path:
        parts = model_path.split('/')
        # HuggingFace: exactly 2 parts like "nvidia/parakeet-tdt-0.6b-v3"
        if len(parts) == 2 and not Path(model_path).exists():
            return True
    return False


def download_from_huggingface(model_name: str) -> str:
    """Download .nemo file from HuggingFace and return local path."""
    from huggingface_hub import hf_hub_download
    
    # Model name like "nvidia/parakeet-tdt-0.6b-v3"
    # The .nemo file is typically named after the model
    model_basename = model_name.split('/')[-1]
    nemo_filename = f"{model_basename}.nemo"
    
    logger.info(f"Downloading {nemo_filename} from HuggingFace: {model_name}")
    
    try:
        local_path = hf_hub_download(
            repo_id=model_name,
            filename=nemo_filename
        )
        logger.info(f"Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {model_name}: {e}")


def load_tokenizer_from_nemo(nemo_path: str):
    """Load SentencePiece tokenizer from a .nemo file or HuggingFace model."""
    import sentencepiece as spm
    
    # Handle HuggingFace model names
    if is_huggingface_model(nemo_path):
        nemo_path = download_from_huggingface(nemo_path)
    
    nemo_path = Path(nemo_path)
    if not nemo_path.exists():
        raise FileNotFoundError(f"Model not found: {nemo_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract the .nemo archive
        with tarfile.open(nemo_path, 'r:*') as tar:
            tar.extractall(tmpdir)
        
        # Find the tokenizer.model file
        tokenizer_files = list(tmpdir.glob("**/tokenizer.model")) + list(tmpdir.glob("**/*.model"))
        if not tokenizer_files:
            raise FileNotFoundError("Could not find tokenizer.model in .nemo archive")
        
        tokenizer_path = tokenizer_files[0]
        logger.debug(f"Found tokenizer: {tokenizer_path}")
        
        # Load the tokenizer
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        
        return sp


def analyze_tokenization(sp, text: str) -> Dict:
    """Analyze how well a text is tokenized."""
    try:
        # Encode the text
        token_ids = sp.encode(text, out_type=int)
        tokens = sp.encode(text, out_type=str)
        
        # Decode back
        decoded = sp.decode(token_ids)
        
        # Count unknown tokens
        unk_id = sp.unk_id()
        unk_count = sum(1 for tid in token_ids if tid == unk_id)
        
        # Check coverage
        total_tokens = len(token_ids)
        known_tokens = total_tokens - unk_count
        coverage = known_tokens / total_tokens if total_tokens > 0 else 0
        
        return {
            "success": True,
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "decoded": decoded,
            "total_tokens": total_tokens,
            "unk_count": unk_count,
            "coverage": coverage,
        }
    except Exception as e:
        return {
            "success": False,
            "text": text,
            "error": str(e),
            "coverage": 0,
        }


def test_language(sp, lang_code: str, verbose: bool = False) -> Tuple[bool, Dict]:
    """Test tokenizer support for a specific language."""
    sentences = TEST_SENTENCES.get(lang_code, [])
    if not sentences:
        return False, {"error": f"No test sentences for language: {lang_code}"}
    
    results = []
    total_coverage = 0
    total_unk = 0
    total_tokens = 0
    
    for sentence in sentences:
        analysis = analyze_tokenization(sp, sentence)
        results.append(analysis)
        
        if analysis["success"]:
            total_coverage += analysis["coverage"]
            total_unk += analysis["unk_count"]
            total_tokens += analysis["total_tokens"]
    
    avg_coverage = total_coverage / len(results) if results else 0
    unk_ratio = total_unk / total_tokens if total_tokens > 0 else 1.0
    
    # A language is "supported" if:
    # - Average coverage is >= 80% (most chars are known tokens)
    # - Less than 20% unknown tokens overall
    is_supported = avg_coverage >= 0.8 and unk_ratio < 0.2
    
    summary = {
        "supported": is_supported,
        "avg_coverage": avg_coverage,
        "total_tokens": total_tokens,
        "total_unk": total_unk,
        "unk_ratio": unk_ratio,
        "sentences_tested": len(sentences),
    }
    
    if verbose:
        summary["details"] = results
    
    return is_supported, summary


def main():
    parser = argparse.ArgumentParser(
        description="Test tokenizer language support for a .nemo model"
    )
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name (e.g., nvidia/parakeet-tdt-0.6b-v3) or path to .nemo file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed tokenization results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,ms,cn",
        help="Comma-separated list of languages to test (default: en,ms,cn)"
    )
    
    args = parser.parse_args()
    
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    # Validate languages
    for lang in languages:
        if lang not in TEST_SENTENCES:
            print(f"Error: Unknown language code '{lang}'. Available: {', '.join(TEST_SENTENCES.keys())}")
            sys.exit(1)
    
    # Load tokenizer
    if not args.json:
        print("=" * 70)
        print("Tokenizer Language Support Test")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Testing: {', '.join(LANGUAGE_NAMES.get(l, l) for l in languages)}")
        print("=" * 70)
        print()
    
    try:
        sp = load_tokenizer_from_nemo(args.model)
        vocab_size = sp.get_piece_size()
        
        if not args.json:
            print(f"Tokenizer loaded successfully")
            print(f"Vocabulary size: {vocab_size:,}")
            print()
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Test each language
    results = {
        "model": args.model,
        "vocab_size": vocab_size,
        "languages": {},
    }
    
    all_supported = True
    
    for lang in languages:
        is_supported, summary = test_language(sp, lang, verbose=args.verbose)
        results["languages"][lang] = summary
        
        if not is_supported:
            all_supported = False
        
        if not args.json:
            status = "✅ SUPPORTED" if is_supported else "❌ NOT SUPPORTED"
            print(f"{LANGUAGE_NAMES.get(lang, lang)} ({lang}): {status}")
            print(f"  Coverage: {summary['avg_coverage']*100:.1f}%")
            print(f"  Unknown tokens: {summary['total_unk']}/{summary['total_tokens']} ({summary['unk_ratio']*100:.1f}%)")
            
            if args.verbose and "details" in summary:
                print(f"  Details:")
                for detail in summary["details"]:
                    if detail["success"]:
                        print(f"    Text: {detail['text'][:50]}...")
                        print(f"    Tokens: {detail['tokens'][:10]}...")
                        print(f"    Coverage: {detail['coverage']*100:.1f}%")
                    else:
                        print(f"    Text: {detail['text'][:50]}...")
                        print(f"    Error: {detail['error']}")
            print()
    
    results["all_supported"] = all_supported
    
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        if all_supported:
            print("✅ All tested languages are supported!")
        else:
            unsupported = [l for l in languages if not results["languages"][l]["supported"]]
            print(f"⚠️  Some languages not fully supported: {', '.join(unsupported)}")
        print("=" * 70)
    
    # Exit with error code if not all supported
    sys.exit(0 if all_supported else 1)


if __name__ == "__main__":
    main()
