#!/usr/bin/env python3
"""
Unified text normalizer for multilingual ASR training.
Converts numbers, currency, dates, etc. to spoken words for EN/MS/ZH.

Features:
- Number/currency/percentage to words conversion
- Unicode normalization (NFC)
- Lowercase normalization
- Abbreviation expansion (EN/MS)
- Discourse particle normalization (Malaysian: lah, leh, loh, ah variants)

Dependencies:
    pip install num2words cn2an langid
"""

import re
import logging
import unicodedata
from typing import Optional, List

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
# Chinese Text Preprocessing
# =============================================================================

# Full-width to half-width character mapping
# Full-width characters are common in Chinese text (e.g., ２０２４, Ａ, ！)
_FULLWIDTH_OFFSET = 0xFEE0  # Difference between full-width and ASCII

def fullwidth_to_halfwidth(text: str) -> str:
    """
    Convert full-width characters to half-width (ASCII).
    
    Examples:
        "２０２４年" → "2024年"
        "Ａ　Ｂ　Ｃ" → "A B C"
        "！？" → "!?"
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width ASCII variants (！to ～, range 0xFF01-0xFF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - _FULLWIDTH_OFFSET))
        # Full-width space
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


# Chinese punctuation marks to remove
CHINESE_PUNCTUATION = (
    '。，、；：？！'  # Period, comma, enumeration comma, semicolon, colon, question, exclamation
    '""''「」『』'    # Quotation marks
    '（）【】〔〕'    # Brackets
    '《》〈〉'        # Angle brackets (book titles)
    '——……'           # Dash and ellipsis
    '·'               # Middle dot
)

# Pattern for Chinese punctuation
_CHINESE_PUNCT_PATTERN = re.compile(f'[{re.escape(CHINESE_PUNCTUATION)}]')

# Pattern for noise markers in Chinese speech corpora
_CHINESE_NOISE_PATTERN = re.compile(
    r'<SPOKEN_NOISE>|'
    r'\[SPOKEN_NOISE\]|'
    r'\[noise\]|'
    r'\[laughter\]|'
    r'\[cough\]|'
    r'\[breath\]|'
    r'<UNK>|'
    r'\[UNK\]',
    re.IGNORECASE
)


def remove_chinese_punctuation(text: str) -> str:
    """Remove Chinese punctuation marks."""
    return _CHINESE_PUNCT_PATTERN.sub('', text)


def remove_noise_markers(text: str) -> str:
    """Remove spoken noise markers common in speech corpora."""
    return _CHINESE_NOISE_PATTERN.sub('', text)


def preprocess_chinese_text(
    text: str,
    remove_punctuation: bool = True,
    convert_fullwidth: bool = True,
    remove_noise: bool = True,
) -> str:
    """
    Comprehensive Chinese text preprocessing for ASR training.
    
    Steps:
    1. Remove noise markers (<SPOKEN_NOISE>, [noise], etc.)
    2. Convert full-width to half-width characters
    3. Remove punctuation (Chinese and English)
    4. Normalize whitespace
    
    Note: Numbers are kept as-is (not converted to Chinese words).
    
    Args:
        text: Input Chinese text
        remove_punctuation: Remove punctuation marks
        convert_fullwidth: Convert full-width to half-width
        remove_noise: Remove noise markers
        
    Returns:
        Preprocessed text ready for ASR training
    """
    if not text or not text.strip():
        return text
    
    # 1. Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove noise markers
    if remove_noise:
        text = remove_noise_markers(text)
    
    # 3. Convert full-width to half-width
    if convert_fullwidth:
        text = fullwidth_to_halfwidth(text)
    
    # 4. Remove punctuation
    if remove_punctuation:
        # Remove Chinese punctuation
        text = remove_chinese_punctuation(text)
        # Remove English punctuation (but keep apostrophes and numbers)
        text = re.sub(r"[^\w\s\u4e00-\u9fff']", '', text)
    
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# =============================================================================
# Abbreviation Expansion
# =============================================================================

# Common abbreviations for English
ABBREVIATIONS_EN = {
    # Titles
    r'\bdr\.?\b': 'doctor',
    r'\bmr\.?\b': 'mister',
    r'\bmrs\.?\b': 'missus',
    r'\bms\.?\b': 'miss',
    r'\bprof\.?\b': 'professor',
    # Units
    r'\bkm\b': 'kilometers',
    r'\bkg\b': 'kilograms',
    r'\bmm\b': 'millimeters',
    r'\bcm\b': 'centimeters',
    r'\bml\b': 'milliliters',
    r'\bhr\b': 'hour',
    r'\bhrs\b': 'hours',
    r'\bmin\b': 'minutes',
    r'\bsec\b': 'seconds',
    # Common
    r'\betc\.?\b': 'et cetera',
    r'\be\.g\.?\b': 'for example',
    r'\bi\.e\.?\b': 'that is',
    r'\bvs\.?\b': 'versus',
    r'\bno\.?\b': 'number',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\brd\b': 'road',
    r'\bapt\b': 'apartment',
}

# Common abbreviations for Malay
ABBREVIATIONS_MS = {
    # Titles
    r'\bdr\.?\b': 'doktor',
    r'\ben\.?\b': 'encik',
    r'\bpn\.?\b': 'puan',
    r'\bcik\.?\b': 'cik',
    r'\bprof\.?\b': 'profesor',
    r'\bdato\'?\.?\b': 'dato',
    r'\bdatuk\.?\b': 'datuk',
    r'\btan sri\.?\b': 'tan sri',
    r'\btun\.?\b': 'tun',
    # Units  
    r'\bkm\b': 'kilometer',
    r'\bkg\b': 'kilogram',
    r'\bmm\b': 'milimeter',
    r'\bcm\b': 'sentimeter',
    r'\bml\b': 'mililiter',
    r'\bjam\b': 'jam',
    r'\bmin\b': 'minit',
    r'\bsaat\b': 'saat',
    # Common
    r'\bdll\.?\b': 'dan lain-lain',
    r'\bdsb\.?\b': 'dan sebagainya',
    r'\byg\b': 'yang',
    r'\bdgn\b': 'dengan',
    r'\butk\b': 'untuk',
    r'\bkpd\b': 'kepada',
    r'\bdrpd\b': 'daripada',
    r'\bsdh\b': 'sudah',
    r'\bblm\b': 'belum',
    r'\btdk\b': 'tidak',
    r'\bbrp\b': 'berapa',
    r'\bjln\b': 'jalan',
    r'\btmn\b': 'taman',
    r'\bkaw\b': 'kawasan',
}

# Compile abbreviation patterns for efficiency
_ABBREV_PATTERNS_EN = [(re.compile(pattern, re.IGNORECASE), replacement) 
                        for pattern, replacement in ABBREVIATIONS_EN.items()]
_ABBREV_PATTERNS_MS = [(re.compile(pattern, re.IGNORECASE), replacement) 
                        for pattern, replacement in ABBREVIATIONS_MS.items()]


def expand_abbreviations(text: str, language: str = 'en') -> str:
    """
    Expand common abbreviations to full words.
    
    Args:
        text: Input text
        language: 'en' for English, 'ms' for Malay
        
    Returns:
        Text with abbreviations expanded
    """
    patterns = _ABBREV_PATTERNS_MS if language == 'ms' else _ABBREV_PATTERNS_EN
    
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    
    return text


# =============================================================================
# Discourse Particle Normalization (Malaysian)
# =============================================================================

# Malaysian discourse particles with their variants
# These are common in Malaysian English and Malay colloquial speech
DISCOURSE_PARTICLES = {
    # "lah" variants - emphasis/assertion
    'lah': [
        r'\blaa+h?\b',      # laah, laaah
        r'\bla+\b',         # la, laa, laaa (but not "la" as in music)
        r'\blor\b',         # lor (variant)
    ],
    # "leh" variants - suggestion/possibility  
    'leh': [
        r'\ble+h\b',        # leh, leeh
        r'\ble+\b',         # le, lee
    ],
    # "loh" / "lor" variants - obviousness
    'loh': [
        r'\blo+h\b',        # loh, looh
        r'\blo+r\b',        # lor, loor
        r'\blo+\b',         # lo, loo
    ],
    # "ah" variants - question/softening
    'ah': [
        r'\ba+h\b',         # ah, aah, aaah
        r'\bar+h?\b',       # arh, ar (but careful with "are")
    ],
    # "meh" variants - doubt/dismissal
    'meh': [
        r'\bme+h\b',        # meh, meeh
    ],
    # "hor" variants - seeking agreement
    'hor': [
        r'\bho+r\b',        # hor, hoor
    ],
    # "kan" - seeking confirmation (Malay)
    'kan': [
        r'\bka+n\b',        # kan, kaan
    ],
    # "ya" / "yah" - affirmation
    'ya': [
        r'\bya+h?\b',       # ya, yah, yaah
    ],
}

# Compile discourse particle patterns
_DISCOURSE_PATTERNS = []
for normalized, variants in DISCOURSE_PARTICLES.items():
    for pattern in variants:
        _DISCOURSE_PATTERNS.append((re.compile(pattern, re.IGNORECASE), normalized))


def normalize_discourse_particles(text: str) -> str:
    """
    Normalize Malaysian discourse particles to standard forms.
    
    Converts variants like "lahhh", "lorr", "mehh" to standard "lah", "loh", "meh".
    
    Examples:
        "Can lahhh" → "Can lah"
        "Like that lorr" → "Like that loh"
        "Really mehh" → "Really meh"
        
    Args:
        text: Input text
        
    Returns:
        Text with normalized discourse particles
    """
    for pattern, normalized in _DISCOURSE_PATTERNS:
        text = pattern.sub(normalized, text)
    
    return text


# =============================================================================
# Character Normalization and Validation (for EN/MS ASR)
# =============================================================================

# Character normalization mappings (single char → single char only)
# Using explicit Unicode escapes to avoid encoding issues
CHARACTER_NORMALIZATIONS_SINGLE = {
    # Smart/curly double quotes → straight double quote
    '\u201c': '"',  # " left double quotation mark
    '\u201d': '"',  # " right double quotation mark
    '\u201e': '"',  # „ double low-9 quotation mark
    '\u201f': '"',  # ‟ double high-reversed-9 quotation mark
    # Smart/curly single quotes → straight apostrophe
    '\u2018': "'",  # ' left single quotation mark
    '\u2019': "'",  # ' right single quotation mark
    '\u201a': "'",  # ‚ single low-9 quotation mark
    '\u201b': "'",  # ‛ single high-reversed-9 quotation mark
    '\u0060': "'",  # ` grave accent (backtick)
    '\u00b4': "'",  # ´ acute accent
    # Special dashes → hyphen-minus
    '\u2014': '-',  # — em dash
    '\u2013': '-',  # – en dash
    '\u2212': '-',  # − minus sign
    '\u2010': '-',  # ‐ hyphen
    '\u2011': '-',  # ‑ non-breaking hyphen
    '\u2012': '-',  # ‒ figure dash
    '\u2015': '-',  # ― horizontal bar
    # Spaces → regular space
    '\u00a0': ' ',  # non-breaking space
    '\u2002': ' ',  # en space
    '\u2003': ' ',  # em space
    '\u2009': ' ',  # thin space
}

# Characters to remove (zero-width, BOM, etc.)
CHARACTERS_TO_REMOVE = '\u200b\u200c\u200d\ufeff\u200e\u200f'

# Multi-char replacements (handled separately with str.replace)
CHARACTER_NORMALIZATIONS_MULTI = {
    '\u2026': '...',  # … ellipsis → three dots
}

# Compile translation table for fast single-char replacement
_CHAR_NORM_TABLE = str.maketrans(CHARACTER_NORMALIZATIONS_SINGLE)
_CHAR_REMOVE_TABLE = str.maketrans('', '', CHARACTERS_TO_REMOVE)


def normalize_characters(text: str) -> str:
    """
    Normalize special characters to their ASCII equivalents.
    
    Converts:
    - Smart quotes → straight quotes (" " ' ' → " ')
    - Special dashes → hyphen (— – → -)
    - Non-breaking spaces → regular spaces
    - Ellipsis → three dots
    - Removes zero-width characters and BOM
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized characters
    """
    # Single-char replacements
    text = text.translate(_CHAR_NORM_TABLE)
    # Remove zero-width chars
    text = text.translate(_CHAR_REMOVE_TABLE)
    # Multi-char replacements
    for old, new in CHARACTER_NORMALIZATIONS_MULTI.items():
        text = text.replace(old, new)
    return text


# =============================================================================
# Script Detection (Non-Latin scripts)
# =============================================================================

# Chinese character ranges (CJK Unified Ideographs)
# Note: Only using BMP ranges (U+0000-U+FFFF) to avoid regex escape issues
# CJK Unified Ideographs: U+4E00-U+9FFF
# CJK Extension A: U+3400-U+4DBF  
# (CJK Extension B+ are in supplementary planes and rarely used in common text)
_CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# Tamil character range
_TAMIL_PATTERN = re.compile(r'[\u0B80-\u0BFF]')

# Arabic character range
_ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')

# Thai character range
_THAI_PATTERN = re.compile(r'[\u0E00-\u0E7F]')

# Cyrillic character range (Russian, etc.)
_CYRILLIC_PATTERN = re.compile(r'[\u0400-\u04FF]')

# Japanese Hiragana and Katakana
_JAPANESE_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')

# Korean Hangul
_KOREAN_PATTERN = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]')

# Devanagari (Hindi, Sanskrit, etc.)
_DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

# Bengali
_BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')

# Telugu
_TELUGU_PATTERN = re.compile(r'[\u0C00-\u0C7F]')

# Hebrew
_HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')

# Greek (excluding common mathematical symbols used in English)
_GREEK_PATTERN = re.compile(r'[\u0370-\u03FF]')

# Emoji pattern (comprehensive)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002B50-\U00002B55"  # Stars
    "\U0000231A-\U0000231B"  # Watch, Hourglass
    "\U0000FE0F"             # Variation selector
    "]+",
    flags=re.UNICODE
)


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(_CHINESE_PATTERN.search(text))


def contains_tamil(text: str) -> bool:
    """Check if text contains Tamil characters."""
    return bool(_TAMIL_PATTERN.search(text))


def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(_ARABIC_PATTERN.search(text))


def contains_thai(text: str) -> bool:
    """Check if text contains Thai characters."""
    return bool(_THAI_PATTERN.search(text))


def contains_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(_CYRILLIC_PATTERN.search(text))


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese Hiragana/Katakana."""
    return bool(_JAPANESE_PATTERN.search(text))


def contains_korean(text: str) -> bool:
    """Check if text contains Korean Hangul."""
    return bool(_KOREAN_PATTERN.search(text))


def contains_non_latin_script(text: str) -> bool:
    """
    Check if text contains non-Latin scripts that should be excluded from EN/MS ASR training.
    
    Excludes: Chinese, Tamil, Arabic, Thai, Cyrillic, Japanese, Korean, 
              Devanagari, Bengali, Telugu, Hebrew, Greek
    
    Note: Accented Latin characters (é, ñ, ü, ç, etc.) are ALLOWED
          as they appear in English loanwords like café, naïve, résumé.
    
    Returns:
        True if text contains any non-Latin script characters
    """
    return (
        contains_chinese(text) or 
        contains_tamil(text) or
        contains_arabic(text) or
        contains_thai(text) or
        contains_cyrillic(text) or
        contains_japanese(text) or
        contains_korean(text) or
        bool(_DEVANAGARI_PATTERN.search(text)) or
        bool(_BENGALI_PATTERN.search(text)) or
        bool(_TELUGU_PATTERN.search(text)) or
        bool(_HEBREW_PATTERN.search(text)) or
        bool(_GREEK_PATTERN.search(text))
    )


def remove_emoji(text: str) -> str:
    """
    Remove emoji characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with emoji removed
    """
    return _EMOJI_PATTERN.sub('', text)


# =============================================================================
# Character Validation for ASR
# =============================================================================

# Valid characters for EN/MS ASR:
# - Basic Latin: a-z, A-Z, 0-9
# - Accented Latin (for loanwords like café, naïve, résumé):
#   - Latin-1 Supplement: À-ÖØ-öø-ÿ (U+00C0-U+00FF minus ×÷)
#   - Latin Extended-A: U+0100-U+017F (Ā, ă, Ą, etc.)
# - Space, apostrophe, hyphen, @
_VALID_CHARS_PATTERN = re.compile(
    r"^[a-zA-Z0-9\s'\-@"
    r"\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF"  # Latin-1 Supplement (accented)
    r"\u0100-\u017F"  # Latin Extended-A
    r"]+$"
)
_INVALID_CHARS_PATTERN = re.compile(
    r"[^a-zA-Z0-9\s'\-@"
    r"\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF"  # Latin-1 Supplement (accented)
    r"\u0100-\u017F"  # Latin Extended-A
    r"]"
)


def is_valid_asr_text(text: str) -> bool:
    """
    Check if text contains only valid characters for EN/MS ASR training.
    
    Valid characters: a-z, A-Z, 0-9, space, apostrophe ('), hyphen (-), at-sign (@)
    
    Args:
        text: Input text to validate
        
    Returns:
        True if text contains only valid characters, False otherwise
    """
    if not text or not text.strip():
        return False
    return bool(_VALID_CHARS_PATTERN.match(text))


def get_invalid_characters(text: str) -> set:
    """
    Get set of invalid characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Set of characters that are not valid for ASR
    """
    return set(_INVALID_CHARS_PATTERN.findall(text))


def clean_text_for_asr(text: str) -> str:
    """
    Clean text by removing emoji and all punctuation except apostrophe, hyphen, and @.
    
    This removes: emoji, . , ! ? : ; " ( ) [ ] { } / \\ * & ^ % $ # + = < > | ~
    Keeps: letters (including accented like é, ñ, ü), numbers, space, ' - @
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text with only valid ASR characters
    """
    # Remove emoji first
    text = remove_emoji(text)
    # Remove all characters except alphanumeric (including accented), space, apostrophe, hyphen, @
    text = _INVALID_CHARS_PATTERN.sub('', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
# Comprehensive ASR Text Preprocessing
# =============================================================================

def preprocess_text_for_asr(
    text: str,
    language: Optional[str] = None,
    lowercase: bool = True,
    unicode_normalize: bool = True,
    normalize_chars: bool = True,
    expand_abbrevs: bool = True,
    normalize_particles: bool = True,
    normalize_numbers: bool = True,
    remove_punctuation: bool = False,
    clean_for_asr: bool = False,
) -> str:
    """
    Comprehensive text preprocessing for ASR training.
    
    For EN/MS:
    1. Unicode normalization (NFC)
    2. Character normalization (smart quotes → straight, dashes → hyphen)
    3. Abbreviation expansion
    4. Number/currency normalization
    5. Discourse particle normalization (Malaysian)
    6. Lowercase conversion
    7. Punctuation removal (optional) OR clean for ASR (keep only ', -, @)
    8. Whitespace normalization
    
    For ZH (Chinese):
    1. Unicode normalization (NFC)
    2. Remove noise markers
    3. Full-width to half-width conversion
    4. Punctuation removal (Chinese + English)
    5. Whitespace normalization
    
    Args:
        text: Input text
        language: Language code ('en', 'ms', 'zh') or None for auto-detect
        lowercase: Convert to lowercase (default: True, ignored for Chinese)
        unicode_normalize: Apply NFC normalization (default: True)
        normalize_chars: Normalize smart quotes, dashes, etc. (default: True)
        expand_abbrevs: Expand abbreviations (default: True)
        normalize_particles: Normalize discourse particles (default: True)
        normalize_numbers: Convert numbers to words (default: True)
        remove_punctuation: Remove all punctuation except apostrophe (default: False)
        clean_for_asr: Remove all chars except alphanumeric, space, ', -, @ (default: False)
        
    Returns:
        Preprocessed text ready for ASR training
    """
    if not text or not text.strip():
        return text
    
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(text)
    
    # Use specialized Chinese preprocessing for ZH
    if language == 'zh':
        return preprocess_chinese_text(
            text,
            remove_punctuation=remove_punctuation or clean_for_asr,
            convert_fullwidth=True,
            remove_noise=True,
        )
    
    # EN/MS preprocessing below
    
    # 1. Unicode normalization (NFC - composed form)
    if unicode_normalize:
        text = unicodedata.normalize('NFC', text)
    
    # 2. Character normalization (smart quotes → straight, dashes → hyphen)
    if normalize_chars:
        text = normalize_characters(text)
    
    # 3. Expand abbreviations (before number normalization)
    if expand_abbrevs and language in ['en', 'ms']:
        text = expand_abbreviations(text, language)
    
    # 4. Normalize numbers, currency, percentages
    if normalize_numbers:
        text = normalize_text(text, language)
    
    # 5. Normalize discourse particles (Malaysian EN/MS)
    if normalize_particles and language in ['en', 'ms']:
        text = normalize_discourse_particles(text)
    
    # 6. Lowercase
    if lowercase:
        text = text.lower()
    
    # 7. Clean for ASR (keep only alphanumeric, space, ', -, @)
    if clean_for_asr:
        text = clean_text_for_asr(text)
    # Or just remove punctuation (keep apostrophe only)
    elif remove_punctuation:
        text = re.sub(r"[^\w\s']", ' ', text)
    
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text_batch_for_asr(
    texts: List[str],
    language: Optional[str] = None,
    lowercase: bool = True,
    unicode_normalize: bool = True,
    normalize_chars: bool = True,
    expand_abbrevs: bool = True,
    normalize_particles: bool = True,
    normalize_numbers: bool = True,
    remove_punctuation: bool = False,
    clean_for_asr: bool = False,
) -> List[str]:
    """
    Batch preprocessing for ASR training.
    
    Args:
        texts: List of input texts
        language: Language code or None for auto-detect
        ... (other args same as preprocess_text_for_asr)
        
    Returns:
        List of preprocessed texts
    """
    return [
        preprocess_text_for_asr(
            text,
            language=language,
            lowercase=lowercase,
            unicode_normalize=unicode_normalize,
            normalize_chars=normalize_chars,
            expand_abbrevs=expand_abbrevs,
            normalize_particles=normalize_particles,
            normalize_numbers=normalize_numbers,
            remove_punctuation=remove_punctuation,
            clean_for_asr=clean_for_asr,
        )
        for text in texts
    ]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Text Normalizer Test")
    print("=" * 60)
    
    # Test Malay
    print("\n--- Malay Number Normalization ---")
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
    print("\n--- English Number Normalization ---")
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
    print("\n--- Chinese Number Normalization ---")
    test_zh = [
        "价格是15元",
        "他今年25岁",
        "总共1234元",
        "增长了15%",
    ]
    for text in test_zh:
        print(f"  {text}")
        print(f"  → {normalize_text(text, 'zh')}")
    
    # Test Abbreviation Expansion
    print("\n--- Abbreviation Expansion (English) ---")
    test_abbrev_en = [
        "Dr. Smith lives on 5th St.",
        "The distance is 10 km etc.",
        "Mr. Lee vs Mrs. Tan",
    ]
    for text in test_abbrev_en:
        print(f"  {text}")
        print(f"  → {expand_abbreviations(text, 'en')}")
    
    print("\n--- Abbreviation Expansion (Malay) ---")
    test_abbrev_ms = [
        "En. Ahmad tinggal di Jln. Merdeka",
        "Pn. Siti dgn Dr. Razak",
        "Utk maklumat lanjut, sila hubungi kpd pihak yg berkenaan",
    ]
    for text in test_abbrev_ms:
        print(f"  {text}")
        print(f"  → {expand_abbreviations(text, 'ms')}")
    
    # Test Discourse Particle Normalization
    print("\n--- Discourse Particle Normalization ---")
    test_particles = [
        "Can lahhh, no problem",
        "Like that lorr",
        "Really mehh?",
        "Good lahh this one",
        "You know hor, this thing ahhh",
        "Oklah, I go now",
        "Betul kan?",
    ]
    for text in test_particles:
        print(f"  {text}")
        print(f"  → {normalize_discourse_particles(text)}")
    
    # Test Full ASR Preprocessing Pipeline
    print("\n--- Full ASR Preprocessing (Malay) ---")
    test_full_ms = [
        "En. Ahmad beli 5 kg beras RM 25",
        "Dr. Siti kata OK lahhh!",
        "Utk 15% diskaun, hubungi kpd pihak yg berkenaan.",
    ]
    for text in test_full_ms:
        print(f"  Original: {text}")
        print(f"  → ASR:    {preprocess_text_for_asr(text, 'ms')}")
    
    print("\n--- Full ASR Preprocessing (English) ---")
    test_full_en = [
        "Dr. Smith said it's 50% off!",
        "Can lahhh, the price is RM 100 only.",
        "Mr. Lee lives on 5th St. etc.",
    ]
    for text in test_full_en:
        print(f"  Original: {text}")
        print(f"  → ASR:    {preprocess_text_for_asr(text, 'en')}")
    
    # Test Chinese Preprocessing
    print("\n--- Chinese Text Preprocessing ---")
    print("  (Numbers kept as-is, full-width→half-width, punctuation removed)")
    test_zh_preprocess = [
        ("我今年２５岁。", "我今年25岁"),                    # Full-width numbers
        ("价格是１００元！", "价格是100元"),                  # Full-width + punctuation
        ("<SPOKEN_NOISE>你好，世界！", "你好世界"),          # Noise marker + punctuation
        ("第1个问题是什么？", "第1个问题是什么"),            # Numbers kept as-is
        ("Ａ　Ｂ　Ｃ，这是全角字符。", "A B C这是全角字符"),  # Full-width letters
        ("[noise]今天天气很好。[laughter]", "今天天气很好"), # Noise markers
    ]
    for text, expected in test_zh_preprocess:
        result = preprocess_chinese_text(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {text}")
        print(f"    → {result}")
    
    print("\n--- Full ASR Preprocessing (Chinese) ---")
    test_full_zh = [
        "我在２０２４年买了１００本书。",
        "<SPOKEN_NOISE>价格是50元，很便宜！",
        "第1名获得了100%的支持。",
    ]
    for text in test_full_zh:
        print(f"  Original: {text}")
        print(f"  → ASR:    {preprocess_text_for_asr(text, 'zh')}")
    
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
