"""
Text normalization utilities for Malay language ASR evaluation.

Uses Malaya library for language-specific normalization.
"""

import re
from typing import List


# Lazy initialization for Malaya components (heavy import)
_normalizer_mal = None
_pattern_normalise = None


def _init_malaya():
    """Lazily initialize Malaya components."""
    global _normalizer_mal, _pattern_normalise
    
    if _normalizer_mal is None:
        import malaya
        
        lm = malaya.language_model.kenlm(model='bahasa-wiki-news')
        corrector = malaya.spelling_correction.probability.load(language_model=lm)
        _normalizer_mal = malaya.normalizer.rules.load(corrector, None)
        
        chars_to_ignore_regex = r"""[\/:\\;"−*`‑―''""„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
        _pattern_normalise = re.compile(chars_to_ignore_regex, flags=re.UNICODE)


def get_normalizer():
    """
    Get the Malaya normalizer instance.
    
    Returns:
        Malaya normalizer object
    """
    _init_malaya()
    return _normalizer_mal


def normalize_superscripts(text: str) -> str:
    """
    Convert superscript numbers to Malay words.
    
    Args:
        text: Input text
        
    Returns:
        Text with superscripts replaced
    """
    superscripts = {
        '¹': 'satu',
        '²': 'dua',
        '³': 'tiga',
    }
    for k, v in superscripts.items():
        text = text.replace(k, v)
    return text


def postprocess_text_mal(texts: List[str]) -> List[str]:
    """
    Normalize and postprocess Malay text for WER evaluation.
    
    This function:
    - Converts to lowercase
    - Removes special characters
    - Normalizes URLs, emails
    - Cleans up whitespace
    
    Args:
        texts: List of text strings to normalize
        
    Returns:
        List of normalized text strings
    """
    _init_malaya()
    
    result_text = []
    for sentence in texts:
        sentence = normalize_superscripts(sentence).lower()
        sentence = _pattern_normalise.sub(' ', sentence)
        sentence = _normalizer_mal.normalize(
            sentence,
            normalize_url=True,
            normalize_email=True,
            normalize_time=False,
            normalize_emoji=False
        )['normalize']
        
        # Remove standalone hyphens (not part of words)
        sentence = re.sub(r'(?<!\w)-(?!\w)', ' ', sentence)
        # Remove remaining non-word characters except hyphens
        sentence = re.sub(r"[^\w\s\-]", ' ', sentence)
        # Clean up whitespace
        sentence = re.sub(r'(\s{2,})', ' ', re.sub(r'(\s+$)|(\A\s+)', '', sentence))
        result_text.append(sentence)
    
    return result_text
