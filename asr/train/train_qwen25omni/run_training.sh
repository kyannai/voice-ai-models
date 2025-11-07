#!/bin/bash
# Training launcher for Qwen2.5-Omni fine-tuning
# Usage: bash run_training.sh

set -e  # Exit on error

echo "======================================================================"
echo "🚀 Qwen2.5-Omni Fine-tuning Launcher"
echo "======================================================================"
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found!"
    echo "Please create config.yaml before training."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected!"
    echo "It's recommended to use a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Training might be slow on CPU."
else
    echo "🖥️  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Check for Flash-Attention 2
python -c "import flash_attn" 2>/dev/null && echo "✅ Flash-Attention 2 detected" || echo "ℹ️  Flash-Attention 2 not found (optional but recommended)"
echo ""

# Show training config summary
echo "📋 Configuration:"
echo "  - Model: $(grep 'name:' config.yaml | head -1 | awk '{print $2}' | tr -d '"')"
echo "  - Output: $(grep 'output_dir:' config.yaml | awk '{print $2}' | tr -d '"')"
echo "  - Epochs: $(grep 'num_train_epochs:' config.yaml | awk '{print $2}')"
echo "  - Batch size: $(grep 'per_device_train_batch_size:' config.yaml | awk '{print $2}')"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "======================================================================"
echo "Starting training..."
echo "======================================================================"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run training
python train_qwen25omni.py 2>&1 | tee training.log

echo ""
echo "======================================================================"
echo "✅ Training completed!"
echo "======================================================================"
echo ""
echo "📁 Check outputs in: $(grep 'output_dir:' config.yaml | awk '{print $2}' | tr -d '"')"
echo "📊 View logs: tensorboard --logdir $(grep 'output_dir:' config.yaml | awk '{print $2}' | tr -d '"')/logs"
echo ""

