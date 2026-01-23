#!/usr/bin/env python3
"""Test inference on sample audio files."""
import argparse
import json
import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("--manifest", required=True, help="Path to manifest file")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    args = parser.parse_args()
    
    print("Loading model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
    model.eval()
    model.cuda()
    
    print(f"Loading samples from {args.manifest}...")
    with open(args.manifest) as f:
        samples = [json.loads(line) for line in f][:args.num_samples]
    
    print("=" * 60)
    for i, s in enumerate(samples, 1):
        pred = model.transcribe([s['audio_filepath']])[0]
        print(f"[{i}/{len(samples)}]")
        print(f"  Reference: {s['text']}")
        print(f"  Predicted: {pred}")
        print("-" * 40)
    print("=" * 60)


if __name__ == "__main__":
    main()
