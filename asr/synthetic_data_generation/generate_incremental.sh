#!/bin/bash
# Incremental generation script for synthetic data
# First generate 10 hours, verify, then generate 90 more hours without duplicates

set -e

echo "========================================"
echo "Incremental Synthetic Data Generation"
echo "========================================"

# Configuration
CONFIG="config.yaml"
BATCH1_DIR="outputs/batch1_10h"
BATCH2_DIR="outputs/batch2_90h"
FINAL_DIR="outputs/final_100h"

# Step 1: Generate first 10 hours (16 samples per name)
echo ""
echo "========================================"
echo "STEP 1: Generate 10 hours (Batch 1)"
echo "========================================"
echo ""

# Update config for 10 hours
echo "Configuring for 10 hours (16 samples per name)..."
python -c "
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['generation']['target_hours'] = 10
config['generation']['mixed']['samples_per_name'] = 16
with open('$CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print('✓ Config updated for 10 hours')
"

# Generate sentences for batch 1
echo ""
echo "Generating sentences for batch 1..."
python scripts/generate_sentences.py \
    --config $CONFIG \
    --output $BATCH1_DIR/sentences.json \
    --seed 42

echo ""
echo "✅ Batch 1 (10h) sentences generated: $BATCH1_DIR/sentences.json"
echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo "1. Review the generated sentences in $BATCH1_DIR/sentences.json"
echo "2. Synthesize a few samples to verify quality:"
echo "   python scripts/synthesize_with_elevenlabs.py \\"
echo "     --config $CONFIG \\"
echo "     --input $BATCH1_DIR/sentences.json \\"
echo "     --output $BATCH1_DIR/synthesized.json \\"
echo "     --max-samples 10"
echo ""
echo "3. Once verified, run this script with --step2 to generate remaining 90 hours:"
echo "   bash generate_incremental.sh --step2"
echo "========================================"

# If --step2 is provided, generate remaining 90 hours
if [ "$1" == "--step2" ]; then
    echo ""
    echo "========================================"
    echo "STEP 2: Generate 90 hours (Batch 2)"
    echo "========================================"
    echo ""
    
    # Update config for 90 hours (144 samples per name total, minus 16 already generated = 128 more)
    echo "Configuring for 90 additional hours (128 samples per name)..."
    python -c "
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['generation']['target_hours'] = 90
config['generation']['mixed']['samples_per_name'] = 128  # 144 total - 16 from batch1
with open('$CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print('✓ Config updated for 90 additional hours')
"
    
    # Generate sentences for batch 2, avoiding duplicates from batch 1
    echo ""
    echo "Generating sentences for batch 2 (avoiding duplicates)..."
    python scripts/generate_sentences.py \
        --config $CONFIG \
        --output $BATCH2_DIR/sentences.json \
        --existing $BATCH1_DIR/sentences.json \
        --seed 123  # Different seed for variety
    
    echo ""
    echo "✅ Batch 2 (90h) sentences generated: $BATCH2_DIR/sentences.json"
    
    # Merge both batches
    echo ""
    echo "Merging both batches into final dataset..."
    python -c "
import json
from pathlib import Path

# Load both batches
with open('$BATCH1_DIR/sentences.json', 'r') as f:
    batch1 = json.load(f)
with open('$BATCH2_DIR/sentences.json', 'r') as f:
    batch2 = json.load(f)

# Combine
combined = batch1 + batch2

# Create final output directory
Path('$FINAL_DIR').mkdir(parents=True, exist_ok=True)

# Save combined
with open('$FINAL_DIR/sentences.json', 'w') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f'✓ Merged {len(batch1):,} + {len(batch2):,} = {len(combined):,} sentences')
print(f'✓ Saved to $FINAL_DIR/sentences.json')
"
    
    echo ""
    echo "========================================"
    echo "✅ Incremental generation complete!"
    echo "========================================"
    echo ""
    echo "Summary:"
    echo "  Batch 1 (10h): $BATCH1_DIR/"
    echo "  Batch 2 (90h): $BATCH2_DIR/"
    echo "  Combined (100h): $FINAL_DIR/"
    echo ""
    echo "Next: Synthesize the final dataset:"
    echo "  python scripts/synthesize_with_elevenlabs.py \\"
    echo "    --config $CONFIG \\"
    echo "    --input $FINAL_DIR/sentences.json \\"
    echo "    --output $FINAL_DIR/synthesized.json"
    echo "========================================"
fi

