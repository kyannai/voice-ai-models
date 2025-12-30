#!/usr/bin/env python3
"""
Convert numbers to Malay words for ASR training
Ensures transcripts match spoken audio exactly
"""

def number_to_malay_words(num: int) -> str:
    """
    Convert integer to Malay words
    Examples:
        1 → "satu"
        15 → "lima belas"
        100 → "seratus"
        500 → "lima ratus"
        1234 → "seribu dua ratus tiga puluh empat"
    """
    if num == 0:
        return "kosong"
    
    # Basic numbers
    ones = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
    
    # Handle negative numbers
    if num < 0:
        return "negatif " + number_to_malay_words(-num)
    
    # Billions
    if num >= 1000000000:
        billions = num // 1000000000
        remainder = num % 1000000000
        if billions == 1:
            result = "satu bilion"
        else:
            result = number_to_malay_words(billions) + " bilion"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    # Millions
    if num >= 1000000:
        millions = num // 1000000
        remainder = num % 1000000
        if millions == 1:
            result = "sejuta"
        else:
            result = number_to_malay_words(millions) + " juta"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    # Thousands
    if num >= 1000:
        thousands = num // 1000
        remainder = num % 1000
        if thousands == 1:
            result = "seribu"
        else:
            result = number_to_malay_words(thousands) + " ribu"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    # Hundreds
    if num >= 100:
        hundreds = num // 100
        remainder = num % 100
        if hundreds == 1:
            result = "seratus"
        else:
            result = ones[hundreds] + " ratus"
        if remainder > 0:
            result += " " + number_to_malay_words(remainder)
        return result
    
    # Teens (11-19)
    if 11 <= num <= 19:
        if num == 11:
            return "sebelas"
        else:
            return ones[num % 10] + " belas"
    
    # Tens (20-99)
    if num >= 20:
        tens = num // 10
        remainder = num % 10
        result = ones[tens] + " puluh"
        if remainder > 0:
            result += " " + ones[remainder]
        return result
    
    # Ten
    if num == 10:
        return "sepuluh"
    
    # Ones (1-9)
    return ones[num]


def currency_to_malay(amount: float, use_rm: bool = False) -> str:
    """
    Convert currency to Malay words
    Examples:
        500, False → "lima ratus ringgit"
        70, False → "tujuh puluh ringgit"
        1234.50, False → "seribu dua ratus tiga puluh empat ringgit lima puluh sen"
    """
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    
    if use_rm:
        result = "RM " + number_to_malay_words(whole)
        if cents > 0:
            result += " ringgit " + number_to_malay_words(cents) + " sen"
    else:
        result = number_to_malay_words(whole) + " ringgit"
        if cents > 0:
            result += " " + number_to_malay_words(cents) + " sen"
    
    return result


def date_to_malay(day: int, month_name: str, year: int) -> str:
    """
    Convert date to Malay words
    Example: (15, "Januari", 2024) → "lima belas Januari dua ribu dua puluh empat"
    """
    day_words = number_to_malay_words(day)
    year_words = year_to_malay_words(year)
    
    return f"{day_words} {month_name} {year_words}"


def year_to_malay_words(year: int) -> str:
    """
    Convert year to Malay words
    Examples:
        2024 → "dua ribu dua puluh empat"
        2000 → "dua ribu"
        1999 → "seribu sembilan ratus sembilan puluh sembilan"
    """
    if year >= 2000:
        thousands = year // 1000
        remainder = year % 1000
        
        if remainder == 0:
            return number_to_malay_words(thousands) + " ribu"
        else:
            return number_to_malay_words(thousands) + " ribu " + number_to_malay_words(remainder)
    else:
        # For years before 2000, read normally
        return number_to_malay_words(year)


def phone_to_malay(phone: str) -> str:
    """
    Convert phone number to Malay words
    Example: "012-345-6789" → "kosong satu dua tiga empat lima enam tujuh lapan sembilan"
    """
    # Remove all non-digit characters
    digits = ''.join(c for c in phone if c.isdigit())
    
    # Convert each digit
    digit_words = []
    for digit in digits:
        digit_words.append(number_to_malay_words(int(digit)))
    
    return " ".join(digit_words)


def percentage_to_malay(percent: float) -> str:
    """
    Convert percentage to Malay words
    Example: 15.5 → "lima belas perpuluhan lima peratus"
    """
    if percent == int(percent):
        # Whole number
        return number_to_malay_words(int(percent)) + " peratus"
    else:
        # Decimal
        whole = int(percent)
        decimal = int((percent - whole) * 10)
        return f"{number_to_malay_words(whole)} perpuluhan {number_to_malay_words(decimal)} peratus"


def time_to_malay(hour: int, minute: int = None) -> str:
    """
    Convert time to Malay words
    Examples:
        (3,) → "pukul tiga"
        (3, 30) → "pukul tiga tiga puluh"
        (12, 45) → "pukul dua belas empat puluh lima"
    """
    result = "pukul " + number_to_malay_words(hour)
    
    if minute is not None and minute > 0:
        result += " " + number_to_malay_words(minute)
    
    return result


def ic_number_to_malay(ic: str) -> str:
    """
    Convert IC number to Malay words (read digit by digit)
    Example: "901231-08-1234" → "sembilan kosong satu dua tiga satu kosong lapan satu dua tiga empat"
    """
    # Remove all non-digit characters
    digits = ''.join(c for c in ic if c.isdigit())
    
    # Convert each digit
    digit_words = []
    for digit in digits:
        digit_words.append(number_to_malay_words(int(digit)))
    
    return " ".join(digit_words)


def ordinal_to_malay(num: int) -> str:
    """
    Convert number to ordinal in Malay
    Examples:
        1 → "pertama"
        2 → "kedua"
        3 → "ketiga"
        5 → "kelima"
        10 → "kesepuluh"
    """
    ordinals = {
        1: "pertama",
        2: "kedua",
        3: "ketiga",
        4: "keempat",
        5: "kelima",
        6: "keenam",
        7: "ketujuh",
        8: "kelapan",
        9: "kesembilan",
        10: "kesepuluh"
    }
    
    if num in ordinals:
        return ordinals[num]
    else:
        # For larger numbers, use "ke-" prefix with number words
        return "ke " + number_to_malay_words(num)


def format_amount_for_tts(amount_str: str) -> str:
    """
    Format currency amount string for TTS (remove commas)
    Example: "1,234,567" → "1234567"
    """
    return amount_str.replace(',', '')


# Test the functions
if __name__ == "__main__":
    print("Testing number_to_malay_words:")
    test_numbers = [0, 1, 10, 11, 15, 20, 25, 70, 100, 500, 1234, 10000, 100000, 1000000, 1500000]
    for num in test_numbers:
        print(f"  {num} → {number_to_malay_words(num)}")
    
    print("\nTesting currency_to_malay:")
    print(f"  500 → {currency_to_malay(500, False)}")
    print(f"  70 → {currency_to_malay(70, False)}")
    print(f"  1234.50 → {currency_to_malay(1234.50, False)}")
    
    print("\nTesting date_to_malay:")
    print(f"  15 Januari 2024 → {date_to_malay(15, 'Januari', 2024)}")
    
    print("\nTesting phone_to_malay:")
    print(f"  012-345-6789 → {phone_to_malay('012-345-6789')}")
    
    print("\nTesting percentage_to_malay:")
    print(f"  15.5 → {percentage_to_malay(15.5)}")
    print(f"  70 → {percentage_to_malay(70)}")
    
    print("\nTesting time_to_malay:")
    print(f"  3:30 → {time_to_malay(3, 30)}")
    print(f"  12:45 → {time_to_malay(12, 45)}")

