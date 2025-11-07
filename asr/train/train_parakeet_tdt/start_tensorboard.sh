#!/bin/bash
# Quick script to start TensorBoard for monitoring training
# Run this in a separate terminal while training is running

set -e

echo "========================================"
echo "Starting TensorBoard"
echo "========================================"
echo ""

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "❌ TensorBoard not found!"
    echo ""
    echo "Install with:"
    echo "  pip install tensorboard"
    echo ""
    exit 1
fi

# Check if outputs directory exists
if [ ! -d "./outputs" ]; then
    echo "⚠️  No outputs directory found yet"
    echo "   Training hasn't started or no checkpoints saved yet"
    echo "   TensorBoard will wait for data..."
    echo ""
fi

echo "✓ Starting TensorBoard..."
echo ""
echo "📊 Open in your browser:"
echo "   http://localhost:6006"
echo ""
echo "📈 What to watch:"
echo "   - val_wer: Should DECREASE (lower is better)"
echo "   - train_loss: Should DECREASE smoothly"
echo "   - lr: Learning rate schedule"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""
echo "========================================"
echo ""

tensorboard --logdir ./outputs --reload_interval 5

