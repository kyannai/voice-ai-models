#!/usr/bin/env python3
"""
Debug script to verify WER calculation
"""
import json
import sys
from pathlib import Path
from jiwer import wer, cer, mer

def check_result_file(result_file):
    """Check a single result file"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    model = data.get('model', 'Unknown')
    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print(f"{'='*70}")
    
    print(f"\nReported metrics:")
    print(f"  WER: {data['wer']:.4f} ({data['wer']*100:.2f}%)")
    print(f"  CER: {data['cer']:.4f} ({data['cer']*100:.2f}%)")
    print(f"  MER: {data['mer']:.4f} ({data['mer']*100:.2f}%)")
    print(f"  Total samples: {len(data['predictions'])}")
    
    # Manually recalculate
    refs = [p['reference'] for p in data['predictions']]
    hyps = [p['hypothesis'] for p in data['predictions']]
    
    manual_wer = wer(refs, hyps)
    manual_cer = cer(refs, hyps)
    manual_mer = mer(refs, hyps)
    
    print(f"\nManual calculation:")
    print(f"  WER: {manual_wer:.4f} ({manual_wer*100:.2f}%)")
    print(f"  CER: {manual_cer:.4f} ({manual_cer*100:.2f}%)")
    print(f"  MER: {manual_mer:.4f} ({manual_mer*100:.2f}%)")
    
    # Check if they match
    if abs(manual_wer - data['wer']) < 0.0001:
        print("\n✓ Metrics match!")
    else:
        print(f"\n⚠️  MISMATCH! Difference: {abs(manual_wer - data['wer']):.6f}")
    
    # Show first 5 samples
    print(f"\n{'='*70}")
    print("First 5 sample predictions:")
    print(f"{'='*70}")
    for i in range(min(5, len(data['predictions']))):
        p = data['predictions'][i]
        sample_wer = wer([p['reference']], [p['hypothesis']]) * 100
        print(f"\nSample {i+1}:")
        print(f"  Ref: '{p['reference']}'")
        print(f"  Hyp: '{p['hypothesis']}'")
        print(f"  WER: {sample_wer:.2f}%")

if __name__ == "__main__":
    outputs_dir = Path("outputs")
    
    # Find all evaluation_results.json files
    result_files = list(outputs_dir.glob("*/evaluation_results.json"))
    
    if not result_files:
        print("No evaluation_results.json files found!")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files\n")
    
    for result_file in sorted(result_files):
        check_result_file(result_file)
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print("WER values should be:")
    print("  0.00-0.10 (0-10%): Excellent")
    print("  0.10-0.30 (10-30%): Good")
    print("  0.30-0.50 (30-50%): Acceptable")
    print("  0.50-1.00 (50-100%): Poor")
    print("  >1.00 (>100%): Very poor or calculation error")

