#!/usr/bin/env python3
"""
Debug script to check what batch_evaluate is finding
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from batch_evaluate import MODELS, find_model_output_dir

outputs_dir = Path("outputs")

# Test the model keys being used
model_keys_to_eval = ['whisper-large-v3-turbo', 'malaysian-whisper-large-v3-turbo-v3', 'parakeet-tdt-0.6b', 'final_model.nemo']

print("="*70)
print("Testing what batch_evaluate will find:")
print("="*70)

evaluation_results = []

for model_key in model_keys_to_eval:
    print(f"\n{model_key}:")
    
    if model_key not in MODELS:
        print(f"  ✗ Not in MODELS dict!")
        continue
    
    output_dir = find_model_output_dir(model_key, outputs_dir)
    
    if output_dir:
        results_file = output_dir / "evaluation_results.json"
        
        if results_file.exists():
            print(f"  ✓ Found directory: {output_dir.name}")
            print(f"  ✓ Has evaluation_results.json")
            
            # Load and check
            import json
            with open(results_file) as f:
                data = json.load(f)
            
            print(f"  Model in file: {data['model']}")
            print(f"  WER in file: {data['wer']:.4f}")
            print(f"  First hypothesis: '{data['predictions'][0]['hypothesis']}'")
            
            evaluation_results.append({
                "status": "success",
                "output_dir": str(output_dir),
                "results_file": str(results_file),
                "model_key": model_key,
                "model": MODELS[model_key]["model"],
                "description": MODELS[model_key]["description"]
            })
        else:
            print(f"  ✗ No evaluation_results.json")
    else:
        print(f"  ✗ Directory not found")

print(f"\n{'='*70}")
print(f"Total evaluation results found: {len(evaluation_results)}")
print(f"{'='*70}")

for i, er in enumerate(evaluation_results, 1):
    print(f"\n{i}. {er['model_key']}")
    print(f"   Model: {er['model']}")
    print(f"   File: {er['results_file']}")

