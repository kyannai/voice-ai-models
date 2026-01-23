#!/usr/bin/env python3
"""
Unified text normalizer for multilingual ASR training.
Converts numbers, currency, dates, etc. to spoken words for EN/MS/ZH.

Dependencies:
    pip install num2words cn2an langid
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_num2words = None
_cn2an = None
_langid = None


def _get_num2words():
    """Lazy import num2words."""
    global _num2words
    if _num2words is None:
        from num2words import num2words
        _num2words = num2words
    return _num2words


def _get_cn2an():
    """Lazy import cn2an."""
    global _cn2an
    if _cn2an is None:
        import cn2an
        _cn2an = cn2an
    return _cn2an


def _get_langid():
    """Lazy import langid."""
    global _langid
    if _langid is None:
        import langid
        _langid = langid
    return _langid


# =============================================================================
# Malay Number Conversion (from number_to_malay.py)
# =============================================================================

def number_to_malay_words(num: int) -> str:
    """
    Convert integer to Malay words.
    Examples:
        1 → "satu"
        15 → "lima belas"
        100 → "seratus"
        1234 → "seribu dua ratus tiga puluh empat"
    """
    if num == 0:
        return "kosong"
    
    ones = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
    
    if num < 0:
        return "negatif " + number_to_malay_words(-num)
    
    if num >= 1000000000:
        billions = num // 1000000000
        remainder = num % 1000000000
        result = "satu bilion" if billions == 1 else number_to_malay_words(billions) + " bilion"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    if num >= 1000000:
        millions = num // 1000000
        remainder = num % 1000000
        result = "sejuta" if millions == 1 else number_to_malay_words(millions) + " juta"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    if num >= 1000:
        thousands = num // 1000
        remainder = num % 1000
        result = "seribu" if thousands == 1 else number_to_malay_words(thousands) + " ribu"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    if num >= 100:
        hundreds = num // 100
        remainder = num % 100
        result = "seratus" if hundreds == 1 else ones[hundreds] + " ratus"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    if 11 <= num <= 19:
        return "sebelas" if num == 11 else ones[num % 10] + " belas"
    
    if num >= 20:
        tens = num // 10
        remainder = num % 10
        result = ones[tens] + " puluh"
        if remainder > 0:
            result += " " + ones[remainder]
        return result
    
    if num == 10:
        return "sepuluh"
    
    return ones[num]


def currency_to_malay(amount: float) -> str:
    """Convert currency amount to Malay words."""
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    
    result = number_to_malay_words(whole) + " ringgit"
    if cents > 0:
        result += " " + number_to_malay_words(cents) + " sen"
    return result


# =============================================================================
# English Number Conversion (using num2words)
# =============================================================================

def number_to_english_words(num: int) -> str:
    """
    Convert integer to English words.
    Examples:
        1 → "one"
        15 → "fifteen"
        100 → "one hundred"
        1234 → "one thousand two hundred thirty-four"
    """
    num2words = _get_num2words()
    try:
        return num2words(num, lang='en')
    except Exception:
        return str(num)


def currency_to_english(amount: float) -> str:
    """Convert currency amount to English words (ringgit)."""
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    
    result = number_to_english_words(whole) + " ringgit"
    if cents > 0:
        result += " " + number_to_english_words(cents) + " sen"
    return result


# =============================================================================
# Chinese Number Conversion (using cn2an)
# =============================================================================

def number_to_chinese_words(num: int) -> str:
    """
    Convert integer to Chinese words.
    Examples:
        1 → "一"
        15 → "十五"
        100 → "一百"
        1234 → "一千二百三十四"
    """
    cn2an = _get_cn2an()
    try:
        return cn2an.an2cn(str(num))
    except Exception:
        return str(num)


def currency_to_chinese(amount: float) -> str:
    """Convert currency amount to Chinese words (yuan)."""
    cn2an = _get_cn2an()
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    
    try:
        result = cn2an.an2cn(str(whole)) + "元"
        if cents > 0:
            result += cn2an.an2cn(str(cents)) + "分"
        return result
    except Exception:
        return str(amount) + "元"


# =============================================================================
# Language Detection
# =============================================================================

def detect_language(text: str) -> str:
    """
    Detect language of text.
    Returns: 'en', 'ms', 'zh', or 'unknown'
    """
    # Check for Chinese characters first (most reliable)
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    
    # Use langid for en/ms detection
    try:
        langid = _get_langid()
        lang, confidence = langid.classify(text)
        
        # Map langid codes
        if lang in ['ms', 'id']:  # Indonesian is similar to Malay
            return 'ms'
        elif lang == 'en':
            return 'en'
        elif lang == 'zh':
            return 'zh'
        else:
            # Default to English for unknown Latin scripts
            return 'en'
    except Exception:
        return 'unknown'


# =============================================================================
# Unified Text Normalization
# =============================================================================

# Regex patterns for number/currency detection
PATTERNS = {
    # RM currency: RM 15, RM15, RM 1,234.50
    'rm_currency': re.compile(r'RM\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'),
    # Yuan currency: 15元, 1234元
    'yuan_currency': re.compile(r'(\d+(?:\.\d{2})?)元'),
    # Standalone numbers: 123, 1,234, 12.5
    'number': re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b'),
    # Percentage: 15%, 12.5%
    'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*%'),
}


def normalize_number(num_str: str, language: str) -> str:
    """Convert a number string to words in the specified language."""
    # Remove commas
    num_str = num_str.replace(',', '')
    
    try:
        # Handle decimals
        if '.' in num_str:
            parts = num_str.split('.')
            whole = int(parts[0])
            decimal = parts[1]
            
            if language == 'ms':
                result = number_to_malay_words(whole) + " perpuluhan"
                for digit in decimal:
                    result += " " + number_to_malay_words(int(digit))
            elif language == 'zh':
                cn2an = _get_cn2an()
                result = cn2an.an2cn(num_str)
            else:  # en
                num2words = _get_num2words()
                result = num2words(float(num_str), lang='en')
            return result
        else:
            num = int(num_str)
            if language == 'ms':
                return number_to_malay_words(num)
            elif language == 'zh':
                return number_to_chinese_words(num)
            else:  # en
                return number_to_english_words(num)
    except Exception as e:
        logger.warning(f"Failed to normalize number '{num_str}': {e}")
        return num_str


def normalize_currency_rm(amount_str: str, language: str) -> str:
    """Convert RM currency to words."""
    amount_str = amount_str.replace(',', '')
    try:
        amount = float(amount_str)
        if language == 'ms':
            return currency_to_malay(amount)
        else:  # en
            return currency_to_english(amount)
    except Exception:
        return "RM " + amount_str


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """
    Normalize text by converting numbers, currency, etc. to words.
    
    Args:
        text: Input text
        language: Language code ('en', 'ms', 'zh') or None for auto-detect
        
    Returns:
        Normalized text with numbers converted to words
    """
    if not text or not text.strip():
        return text
    
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(text)
    
    # Normalize RM currency (for EN/MS)
    if language in ['en', 'ms']:
        def replace_rm(match):
            return normalize_currency_rm(match.group(1), language)
        text = PATTERNS['rm_currency'].sub(replace_rm, text)
    
    # Normalize Yuan currency (for ZH)
    if language == 'zh':
        def replace_yuan(match):
            return currency_to_chinese(float(match.group(1)))
        text = PATTERNS['yuan_currency'].sub(replace_yuan, text)
    
    # Normalize percentages
    def replace_percentage(match):
        num = float(match.group(1))
        if language == 'ms':
            if num == int(num):
                return number_to_malay_words(int(num)) + " peratus"
            else:
                return normalize_number(match.group(1), language) + " peratus"
        elif language == 'zh':
            return "百分之" + number_to_chinese_words(int(num)) if num == int(num) else match.group(0)
        else:  # en
            if num == int(num):
                return number_to_english_words(int(num)) + " percent"
            else:
                return normalize_number(match.group(1), language) + " percent"
    text = PATTERNS['percentage'].sub(replace_percentage, text)
    
    # Normalize standalone numbers
    def replace_number(match):
        return normalize_number(match.group(1), language)
    text = PATTERNS['number'].sub(replace_number, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_text_batch(texts: list, language: Optional[str] = None) -> list:
    """Normalize a batch of texts."""
    return [normalize_text(text, language) for text in texts]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Text Normalizer Test")
    print("=" * 60)
    
    # Test Malay
    print("\n--- Malay ---")
    test_ms = [
        "Harga RM 15 sahaja",
        "Beliau berumur 25 tahun",
        "Jumlah RM 1,234.50",
        "Kenaikan 15% tahun ini",
    ]
    for text in test_ms:
        print(f"  {text}")
        print(f"  → {normalize_text(text, 'ms')}")
    
    # Test English
    print("\n--- English ---")
    test_en = [
        "The price is RM 15 only",
        "She is 25 years old",
        "Total amount RM 1,234.50",
        "Growth of 15% this year",
    ]
    for text in test_en:
        print(f"  {text}")
        print(f"  → {normalize_text(text, 'en')}")
    
    # Test Chinese
    print("\n--- Chinese ---")
    test_zh = [
        "价格是15元",
        "他今年25岁",
        "总共1234元",
        "增长了15%",
    ]
    for text in test_zh:
        print(f"  {text}")
        print(f"  → {normalize_text(text, 'zh')}")
    
    # Test auto-detection
    print("\n--- Auto-Detection ---")
    test_auto = [
        "Hello world 123",
        "Selamat pagi 456",
        "你好世界789",
    ]
    for text in test_auto:
        lang = detect_language(text)
        print(f"  {text} [detected: {lang}]")
        print(f"  → {normalize_text(text)}")
