#!/usr/bin/env python3
"""
Test script to verify hyphen normalization fix for Malay ASR
"""

import sys
sys.path.insert(0, 'calculate_metrics')
from calculate_metrics import normalize_text

print("="*70)
print("Testing Hyphen Normalization Fix for Malay")
print("="*70)
print()

# Test cases from the actual data
test_cases = [
    ("laki laki", "Laki-laki", "Should match after normalization"),
    ("ditengah tengah", "Di tengah-tengah", "Reduplication with prefix"),
    ("tiba tiba", "tiba-tiba", "Simple reduplication"),
    ("anak anak", "anak-anak", "Another reduplication"),
    ("tangan aku disentuh lembut", "Tangan aku disentuh lembut.", "No hyphen, just punctuation"),
]

print("Before fix: Hyphens removed → words merged → mismatch")
print("After fix:  Hyphens → spaces → words stay separate → match!")
print()

all_pass = True
for ref, hyp, description in test_cases:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    match = ref_norm == hyp_norm
    
    status = "✓ PASS" if match else "✗ FAIL"
    if not match:
        all_pass = False
    
    print(f"{status}: {description}")
    print(f"  Ref: '{ref}' → '{ref_norm}'")
    print(f"  Hyp: '{hyp}' → '{hyp_norm}'")
    print()

print("="*70)
if all_pass:
    print("✅ All tests passed! Hyphen normalization is working correctly.")
else:
    print("❌ Some tests failed. Check the normalization logic.")
print("="*70)

