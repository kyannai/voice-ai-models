#!/usr/bin/env python3
"""
Test script to verify the repetition detection fix works
"""
import sys
sys.path.insert(0, '/Users/kyan/data/swprojects/ytl/voice-ai/asr/eval/transcribe')

from transcribe_whisper import detect_and_remove_repetition

def test_case_1():
    """Test the specific case from the user"""
    hypothesis = "masalah sosial makin bertambah lah macam yelah ada orang yang macam tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga tak jaga"
    reference = "masalah sosial yang makin bertambahlah macam iyalah ada orang yang macam tak jaga"
    
    cleaned, is_hallucination = detect_and_remove_repetition(hypothesis)
    
    print("="*80)
    print("TEST CASE 1: 'tak jaga' repetition")
    print("="*80)
    print(f"Original length: {len(hypothesis)} chars, {len(hypothesis.split())} words")
    print(f"Cleaned length:  {len(cleaned)} chars, {len(cleaned.split())} words")
    print(f"Hallucination detected: {is_hallucination}")
    print(f"\nCleaned text: {cleaned}")
    print(f"\nExpected to contain: 'tak jaga' (once)")
    print(f"Actually contains 'tak jaga' count: {cleaned.count('tak jaga')}")
    print(f"\n✓ PASS" if is_hallucination and cleaned.count('tak jaga') <= 2 else "✗ FAIL")
    print()
    return is_hallucination and cleaned.count('tak jaga') <= 2

def test_case_2():
    """Test comma-separated repetition"""
    text = "saya suka makan eh, eh, eh, eh, eh, eh, eh, eh"
    cleaned, is_hallucination = detect_and_remove_repetition(text)
    
    print("="*80)
    print("TEST CASE 2: Comma-separated repetition")
    print("="*80)
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Hallucination detected: {is_hallucination}")
    print(f"\n✓ PASS" if is_hallucination and cleaned == "saya suka makan" else "✗ FAIL")
    print()
    return is_hallucination and cleaned == "saya suka makan"

def test_case_3():
    """Test normal text (no hallucination)"""
    text = "ini adalah teks normal tanpa pengulangan"
    cleaned, is_hallucination = detect_and_remove_repetition(text)
    
    print("="*80)
    print("TEST CASE 3: Normal text (no hallucination)")
    print("="*80)
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Hallucination detected: {is_hallucination}")
    print(f"\n✓ PASS" if not is_hallucination and cleaned == text else "✗ FAIL")
    print()
    return not is_hallucination and cleaned == text

def test_case_4():
    """Test repetition at the start"""
    text = "kata kata kata kata saya makan nasi"
    cleaned, is_hallucination = detect_and_remove_repetition(text, max_repetitions=2)
    
    print("="*80)
    print("TEST CASE 4: Repetition at the start")
    print("="*80)
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Hallucination detected: {is_hallucination}")
    expected = "kata kata"
    print(f"\n✓ PASS" if is_hallucination and cleaned == expected else "✗ FAIL")
    print()
    return is_hallucination and cleaned == expected

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING REPETITION DETECTION")
    print("="*80 + "\n")
    
    results = []
    results.append(("User's specific case", test_case_1()))
    results.append(("Comma-separated", test_case_2()))
    results.append(("Normal text", test_case_3()))
    results.append(("Start repetition", test_case_4()))
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

