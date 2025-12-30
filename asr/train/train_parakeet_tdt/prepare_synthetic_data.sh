#!/bin/bash
# Prepare synthetic names & numbers dataset for Parakeet training
# This script converts synthesized.json to NeMo manifest format

set -e  # Exit on error

echo "=================================================="
echo "Preparing Synthetic Names & Numbers Dataset"
echo "=================================================="

# Define paths
SYNTHETIC_DATA_DIR="../training_data/synthetic_names_numbers"
SYNTHESIZED_JSON="$SYNTHETIC_DATA_DIR/synthesized.json"
OUTPUT_DIR="./data"

# Check if synthetic data exists
if [ ! -f "$SYNTHESIZED_JSON" ]; then
    echo "❌ Error: synthesized.json not found at $SYNTHESIZED_JSON"
    echo ""
    echo "Expected location: ../training_data/synthetic_names_numbers/synthesized.json"
    echo "Please copy your synthetic dataset there first."
    exit 1
fi

# Check if audio directory exists
if [ ! -d "$SYNTHETIC_DATA_DIR/audio" ]; then
    echo "❌ Error: audio directory not found at $SYNTHETIC_DATA_DIR/audio"
    echo ""
    echo "Expected structure:"
    echo "  ../training_data/synthetic_names_numbers/"
    echo "    ├── audio/"
    echo "    │   ├── audio_000000.mp3"
    echo "    │   ├── audio_000001.mp3"
    echo "    │   └── ..."
    echo "    └── synthesized.json"
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(ls -1 "$SYNTHETIC_DATA_DIR/audio/"*.mp3 2>/dev/null | wc -l)
echo "✓ Found $AUDIO_COUNT audio files"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Converting synthesized.json to NeMo manifest format..."
echo "  Input: $SYNTHESIZED_JSON"
echo "  Audio base: $SYNTHETIC_DATA_DIR"
echo "  Output: $OUTPUT_DIR/synthetic_*_manifest.json"
echo ""

# Run prepare_synthetic_manifests.py to convert format
python prepare_synthetic_manifests.py \
  --input "$SYNTHESIZED_JSON" \
  --audio-base-dir "$SYNTHETIC_DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-split 0.9 \
  --min-duration 0.1 \
  --max-duration 30.0 \
  --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Dataset preparation completed!"
    echo "=================================================="
    echo ""
    echo "Generated manifests:"
    echo "  - Train: $OUTPUT_DIR/synthetic_train_manifest.json"
    echo "  - Val:   $OUTPUT_DIR/synthetic_val_manifest.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review the manifests to ensure correctness"
    echo "  2. Run training with:"
    echo "     python train_parakeet_tdt.py --config config_synthetic_names_numbers.yaml"
    echo ""
else
    echo ""
    echo "❌ Error: Dataset preparation failed"
    echo "Check the error messages above"
    exit 1
fi

