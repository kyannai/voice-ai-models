#!/usr/bin/env python3
"""Check tokenizer vocabulary and language support."""
import argparse
import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser(description="Check tokenizer language support")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3",
                        help="Model name or path to .nemo file")
    parser.add_argument("--test-texts", nargs="+", 
                        default=["hello world", "你好世界", "今天天气很好", "saya makan nasi"],
                        help="Test texts to encode/decode")
    parser.add_argument("--show-vocab", type=int, default=0,
                        help="Show first N vocab entries")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results (optional)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    
    if args.model.endswith('.nemo'):
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
    else:
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(args.model)
    
    tokenizer = model.tokenizer
    
    results = []
    results.append("=" * 60)
    results.append(f"Model: {args.model}")
    results.append(f"Tokenizer type: {type(tokenizer).__name__}")
    results.append(f"Vocab size: {tokenizer.vocab_size}")
    results.append("=" * 60)
    
    if args.show_vocab > 0:
        results.append(f"\nFirst {args.show_vocab} vocab entries:")
        for i in range(min(args.show_vocab, tokenizer.vocab_size)):
            token = tokenizer.ids_to_text([i])
            results.append(f"  {i:5d}: {token!r}")
        results.append("")
    
    results.append("\nLanguage Support Test:")
    results.append("-" * 60)
    
    for text in args.test_texts:
        tokens = tokenizer.text_to_ids(text)
        decoded = tokenizer.ids_to_text(tokens)
        
        # Check if decoded matches input (accounting for spacing)
        is_supported = decoded.replace(" ", "").lower() == text.replace(" ", "").lower()
        status = "✓ Supported" if is_supported else "✗ NOT supported"
        
        # Check for UNK tokens (usually ID 1 or contains ⁇)
        has_unk = "⁇" in decoded or (len(set(tokens)) == 1 and len(tokens) > 1)
        if has_unk:
            status = "✗ NOT supported (UNK tokens)"
        
        results.append(f"\nInput:   '{text}'")
        results.append(f"Tokens:  {tokens}")
        results.append(f"Decoded: '{decoded}'")
        results.append(f"Status:  {status}")
    
    results.append("\n" + "=" * 60)
    
    # Print results
    for line in results:
        print(line)
    
    # Save results if output dir specified
    if args.output_dir:
        import os
        from datetime import datetime
        os.makedirs(args.output_dir, exist_ok=True)
        
        model_name = args.model.replace("/", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"tokenizer_check_{model_name}_{timestamp}.txt")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
