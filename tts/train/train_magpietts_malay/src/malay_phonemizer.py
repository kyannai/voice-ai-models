#!/usr/bin/env python3
"""
Malay Phonemizer using espeak-ng backend.

Converts Malay text to IPA phonemes for TTS training.
This provides more consistent pronunciation representation than character-level.

Usage:
    from malay_phonemizer import MalayPhonemizer
    
    phonemizer = MalayPhonemizer()
    phonemes = phonemizer.phonemize("Selamat pagi")
    # Output: "səlamat paɡi"

Requirements:
    - phonemizer: pip install phonemizer
    - espeak-ng: apt install espeak-ng (Ubuntu) or brew install espeak (macOS)
"""

import logging
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import phonemizer
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    logger.warning("phonemizer not installed. Install with: pip install phonemizer")


class MalayPhonemizer:
    """
    Malay text to phoneme converter using espeak-ng.
    
    Features:
    - Converts Malay text to IPA phonemes
    - Handles code-switching (Malay-English)
    - Normalizes common variations
    - Preserves word boundaries
    """
    
    # Malay language code for espeak
    MALAY_LANG = "ms"
    ENGLISH_LANG = "en-us"
    
    # Common Malay abbreviations and their expansions
    ABBREVIATIONS = {
        "yg": "yang",
        "utk": "untuk",
        "dgn": "dengan",
        "kpd": "kepada",
        "dlm": "dalam",
        "sbb": "sebab",
        "mcm": "macam",
        "org": "orang",
        "nk": "nak",
        "tk": "tak",
        "x": "tidak",
        "dr": "dari",
        "bg": "bagi",
        "sgt": "sangat",
        "lg": "lagi",
        "je": "sahaja",
        "ja": "sahaja",
        "apa2": "apa-apa",
        "bila2": "bila-bila",
        "mana2": "mana-mana",
    }
    
    # English words commonly used in Malaysian speech (code-switching)
    ENGLISH_WORDS = {
        "meeting", "report", "email", "deadline", "project", "office",
        "okay", "ok", "sorry", "please", "thanks", "thank", "welcome",
        "phone", "laptop", "computer", "software", "hardware", "online",
        "boss", "manager", "staff", "team", "client", "customer",
        "lunch", "breakfast", "dinner", "coffee", "tea",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }
    
    def __init__(
        self,
        preserve_punctuation: bool = True,
        with_stress: bool = False,
        word_separator: str = " ",
        phone_separator: str = "",
    ):
        """
        Initialize the Malay phonemizer.
        
        Args:
            preserve_punctuation: Keep punctuation in output
            with_stress: Include stress markers in phonemes
            word_separator: Separator between words (default: space)
            phone_separator: Separator between phonemes (default: none)
        """
        if not PHONEMIZER_AVAILABLE:
            raise ImportError(
                "phonemizer is required. Install with: pip install phonemizer\n"
                "Also ensure espeak-ng is installed: apt install espeak-ng"
            )
        
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        
        # Separator configuration
        self.separator = Separator(
            word=word_separator,
            phone=phone_separator,
            syllable=""
        )
        
        # Verify espeak backend is available
        try:
            self._verify_backend()
        except Exception as e:
            logger.error(f"espeak-ng backend error: {e}")
            raise
        
        logger.info("MalayPhonemizer initialized successfully")
    
    def _verify_backend(self):
        """Verify espeak-ng backend is working."""
        test_text = "ujian"
        result = phonemize(
            test_text,
            language=self.MALAY_LANG,
            backend="espeak",
            strip=True,
        )
        if not result:
            raise RuntimeError("espeak-ng backend not producing output")
        logger.debug(f"Backend test: '{test_text}' -> '{result}'")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Malay text before phonemization.
        
        - Expands abbreviations
        - Normalizes whitespace
        - Handles common variations
        """
        # Lowercase for matching (but preserve original case for output)
        text_lower = text.lower()
        
        # Expand abbreviations
        words = text.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower()
            # Remove punctuation for matching
            word_clean = re.sub(r'[^\w]', '', word_lower)
            
            if word_clean in self.ABBREVIATIONS:
                # Preserve punctuation
                punct_match = re.search(r'[^\w]+$', word)
                punct = punct_match.group() if punct_match else ""
                normalized_words.append(self.ABBREVIATIONS[word_clean] + punct)
            else:
                normalized_words.append(word)
        
        text = " ".join(normalized_words)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_english_words(self, text: str) -> list[tuple[str, bool]]:
        """
        Detect English words in text for code-switching handling.
        
        Returns list of (word, is_english) tuples.
        """
        words = text.split()
        result = []
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            is_english = word_clean in self.ENGLISH_WORDS
            result.append((word, is_english))
        
        return result
    
    def phonemize(
        self,
        text: str,
        normalize: bool = True,
        handle_code_switching: bool = True,
    ) -> str:
        """
        Convert Malay text to phonemes.
        
        Args:
            text: Input text in Malay
            normalize: Apply text normalization
            handle_code_switching: Use English phonemes for English words
            
        Returns:
            Phoneme string (IPA)
        """
        if not text or not text.strip():
            return ""
        
        # Normalize text
        if normalize:
            text = self.normalize_text(text)
        
        if handle_code_switching:
            # Process with code-switching awareness
            return self._phonemize_with_code_switching(text)
        else:
            # Simple Malay phonemization
            return self._phonemize_simple(text, self.MALAY_LANG)
    
    def _phonemize_simple(self, text: str, language: str) -> str:
        """Simple phonemization without code-switching."""
        try:
            result = phonemize(
                text,
                language=language,
                backend="espeak",
                separator=self.separator,
                strip=True,
                preserve_punctuation=self.preserve_punctuation,
                with_stress=self.with_stress,
            )
            return result.strip()
        except Exception as e:
            logger.warning(f"Phonemization error for '{text}': {e}")
            return text  # Return original on error
    
    def _phonemize_with_code_switching(self, text: str) -> str:
        """
        Phonemize with awareness of English words in Malay text.
        
        Uses English phonemes for English words, Malay for the rest.
        """
        word_analysis = self.detect_english_words(text)
        
        # Group consecutive words by language
        groups = []
        current_group = []
        current_is_english = None
        
        for word, is_english in word_analysis:
            if current_is_english is None:
                current_is_english = is_english
            
            if is_english == current_is_english:
                current_group.append(word)
            else:
                if current_group:
                    groups.append((current_group, current_is_english))
                current_group = [word]
                current_is_english = is_english
        
        if current_group:
            groups.append((current_group, current_is_english))
        
        # Phonemize each group with appropriate language
        phoneme_parts = []
        for words, is_english in groups:
            text_part = " ".join(words)
            lang = self.ENGLISH_LANG if is_english else self.MALAY_LANG
            phonemes = self._phonemize_simple(text_part, lang)
            phoneme_parts.append(phonemes)
        
        return " ".join(phoneme_parts)
    
    def phonemize_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        handle_code_switching: bool = False,  # Disabled for batch mode (faster)
    ) -> list[str]:
        """
        Phonemize a batch of texts efficiently using native batch processing.
        
        Args:
            texts: List of input texts
            normalize: Apply text normalization
            handle_code_switching: Handle English words (disabled by default for speed)
            
        Returns:
            List of phoneme strings
        """
        if not texts:
            return []
        
        # Normalize texts first
        if normalize:
            texts = [self.normalize_text(t) for t in texts]
        
        if handle_code_switching:
            # Slower path: process individually with code-switching
            results = []
            for text in texts:
                phonemes = self._phonemize_with_code_switching(text)
                results.append(phonemes)
            return results
        else:
            # Fast path: batch phonemize all texts at once with Malay
            try:
                results = phonemize(
                    texts,
                    language=self.MALAY_LANG,
                    backend="espeak",
                    separator=self.separator,
                    strip=True,
                    preserve_punctuation=self.preserve_punctuation,
                    with_stress=self.with_stress,
                )
                # phonemize returns a list when given a list
                if isinstance(results, str):
                    results = [results]
                return results
            except Exception as e:
                logger.warning(f"Batch phonemization error: {e}")
                # Fallback to individual processing
                return [self._phonemize_simple(t, self.MALAY_LANG) for t in texts]


def test_phonemizer():
    """Test the Malay phonemizer with sample sentences."""
    print("Testing Malay Phonemizer")
    print("=" * 60)
    
    phonemizer = MalayPhonemizer()
    
    test_sentences = [
        "Selamat pagi, apa khabar?",
        "Saya pergi ke pasar semalam.",
        "Terima kasih banyak-banyak.",
        "Okay, so basically kita kena siapkan report ni sebelum Jumaat.",
        "Meeting kita postpone ke next week.",
        "Tolong email saya dokumen tu.",
        "Dia org yg sgt baik.",
    ]
    
    for sentence in test_sentences:
        phonemes = phonemizer.phonemize(sentence)
        print(f"\nOriginal:  {sentence}")
        print(f"Phonemes:  {phonemes}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_phonemizer()
