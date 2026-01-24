#!/usr/bin/env python3
"""
Check SentencePiece tokenizer vocabulary and language support.

Usage:
    python check_sentencepiece_tokenizer.py --model path/to/tokenizer.model
    python check_sentencepiece_tokenizer.py --model ../../common/tokenizers/tokenizer_multilingual.model
"""
import argparse
import sentencepiece as spm


# Test sentences for each language
TEST_SENTENCES = {
    "English": [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I would like to order some food please.",
        "The price is one hundred and twenty three dollars.",
        "Thank you very much for your help.",
    ],
    "Malay": [
        "Selamat pagi, apa khabar?",
        "Cuaca hari ini sangat indah.",
        "Saya ingin memesan makanan.",
        "Harganya seratus dua puluh tiga ringgit.",
        "Terima kasih banyak atas bantuan anda.",
    ],
    "Chinese": [
        "你好，最近怎么样？",
        "今天天气很好。",
        "我想点一些食物。",
        "价格是一百二十三元。",
        "非常感谢你的帮助。",
        "我们公司的发展非常快。",
        "这个市场的潜力很大。",
    ],
}


def check_roundtrip(sp, text):
    """Check if text survives encode-decode roundtrip."""
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)
    
    # Normalize for comparison (remove extra spaces)
    text_normalized = ' '.join(text.split())
    decoded_normalized = ' '.join(decoded.split())
    
    return ids, decoded, text_normalized == decoded_normalized


def count_unk_tokens(sp, text):
    """Count UNK tokens in encoded text."""
    ids = sp.encode(text, out_type=int)
    unk_id = sp.unk_id()
    return sum(1 for i in ids if i == unk_id)


def main():
    parser = argparse.ArgumentParser(description="Check SentencePiece tokenizer")
    parser.add_argument("--model", required=True, help="Path to .model file")
    parser.add_argument("--output", default=None, help="Save results to file")
    parser.add_argument("--show-vocab", type=int, default=20, 
                        help="Show first/last N vocab entries")
    parser.add_argument("--custom-texts", nargs="+", default=None,
                        help="Custom texts to test")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.model)
    
    results = []
    results.append("=" * 70)
    results.append(f"Tokenizer: {args.model}")
    results.append(f"Vocab size: {sp.vocab_size()}")
    results.append(f"UNK ID: {sp.unk_id()}")
    results.append(f"BOS ID: {sp.bos_id()}")
    results.append(f"EOS ID: {sp.eos_id()}")
    results.append(f"PAD ID: {sp.pad_id()}")
    results.append("=" * 70)
    
    # Show vocab sample
    if args.show_vocab > 0:
        results.append(f"\n📚 Vocabulary Sample (first {args.show_vocab}):")
        for i in range(min(args.show_vocab, sp.vocab_size())):
            piece = sp.id_to_piece(i)
            results.append(f"  {i:5d}: {piece!r}")
        
        results.append(f"\n   ... (last {args.show_vocab}):")
        start = max(0, sp.vocab_size() - args.show_vocab)
        for i in range(start, sp.vocab_size()):
            piece = sp.id_to_piece(i)
            results.append(f"  {i:5d}: {piece!r}")
    
    # Language support tests
    results.append("\n" + "=" * 70)
    results.append("🌍 Language Support Test")
    results.append("=" * 70)
    
    language_stats = {}
    
    for lang, sentences in TEST_SENTENCES.items():
        results.append(f"\n--- {lang} ---")
        passed = 0
        total_unk = 0
        
        for text in sentences:
            ids, decoded, is_match = check_roundtrip(sp, text)
            unk_count = count_unk_tokens(sp, text)
            total_unk += unk_count
            
            # Determine status
            if unk_count > 0:
                status = f"⚠️  UNK tokens ({unk_count})"
            elif is_match:
                status = "✅ Perfect"
                passed += 1
            else:
                status = "⚠️  Decode mismatch"
            
            tokens_str = sp.encode(text, out_type=str)
            results.append(f"\n  Input:   \"{text}\"")
            results.append(f"  Tokens:  {tokens_str[:10]}{'...' if len(tokens_str) > 10 else ''}")
            results.append(f"  IDs:     {ids[:10]}{'...' if len(ids) > 10 else ''}")
            results.append(f"  Decoded: \"{decoded}\"")
            results.append(f"  Status:  {status}")
        
        language_stats[lang] = {
            "passed": passed,
            "total": len(sentences),
            "unk_tokens": total_unk
        }
    
    # Custom texts
    if args.custom_texts:
        results.append("\n--- Custom Texts ---")
        for text in args.custom_texts:
            ids, decoded, is_match = check_roundtrip(sp, text)
            unk_count = count_unk_tokens(sp, text)
            tokens_str = sp.encode(text, out_type=str)
            
            if unk_count > 0:
                status = f"⚠️  UNK tokens ({unk_count})"
            elif is_match:
                status = "✅ Perfect"
            else:
                status = "⚠️  Decode mismatch"
            
            results.append(f"\n  Input:   \"{text}\"")
            results.append(f"  Tokens:  {tokens_str}")
            results.append(f"  IDs:     {ids}")
            results.append(f"  Decoded: \"{decoded}\"")
            results.append(f"  Status:  {status}")
    
    # Summary
    results.append("\n" + "=" * 70)
    results.append("📊 Summary")
    results.append("=" * 70)
    
    all_passed = True
    for lang, stats in language_stats.items():
        pct = (stats["passed"] / stats["total"]) * 100
        unk_status = f", {stats['unk_tokens']} UNK tokens" if stats["unk_tokens"] > 0 else ""
        status_icon = "✅" if stats["passed"] == stats["total"] and stats["unk_tokens"] == 0 else "⚠️"
        results.append(f"  {status_icon} {lang}: {stats['passed']}/{stats['total']} passed ({pct:.0f}%){unk_status}")
        if stats["passed"] < stats["total"] or stats["unk_tokens"] > 0:
            all_passed = False
    
    if all_passed:
        results.append("\n🎉 All languages fully supported!")
    else:
        results.append("\n⚠️  Some issues detected - see details above")
    
    results.append("=" * 70)
    
    # Print results
    output_text = '\n'.join(results)
    print(output_text)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
