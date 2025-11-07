#!/bin/bash
# Setup script for full dataset training with memory optimizations
# This installs the required dependencies for 8-bit optimizer support

set -e

echo "========================================="
echo "Full Dataset Training Setup"
echo "========================================="
echo ""

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating one with: source ../.venv/bin/activate"
    echo ""
fi

# Install bitsandbytes for 8-bit optimizer
echo "📦 Installing bitsandbytes (for 8-bit AdamW optimizer)..."
pip install bitsandbytes

echo ""
echo "✓ Installation complete!"
echo ""
echo "========================================="
echo "Quick Start for Full Dataset Training"
echo "========================================="
echo ""
echo "1. Use the full dataset config:"
echo "   cp config_full_dataset.yaml config.yaml"
echo ""
echo "2. Verify your training data is prepared:"
echo "   ls -lh data/train_manifest.json"
echo "   ls -lh data/val_manifest.json"
echo ""
echo "3. Start training:"
echo "   bash run_training.sh"
echo ""
echo "Memory optimizations enabled:"
echo "  ✓ 8-bit AdamW optimizer (75% memory reduction)"
echo "  ✓ Gradient checkpointing (30-50% activation memory reduction)"
echo "  ✓ CUDA memory fragmentation prevention"
echo ""
echo "Expected memory usage: ~50-55GB on A100-80GB"
echo "Expected training time: ~24 hours for 5.2M samples (1 epoch)"
echo ""
echo "For more information, see FULL_DATASET_TRAINING.md"
echo ""

