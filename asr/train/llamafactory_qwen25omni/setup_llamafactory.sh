#!/bin/bash
# Setup script for LLaMA-Factory with Qwen2.5-Omni

set -e

echo "=========================================="
echo "LLaMA-Factory Setup for Qwen2.5-Omni"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: No virtual environment detected!"
    echo ""
    echo "Please activate the virtual environment first:"
    echo "  source ../.venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✓ Using uv for package installation"
    PIP_CMD="uv pip"
else
    echo "⚠ uv not found, using standard pip"
    PIP_CMD="python -m pip"
fi
echo ""

# Clone LLaMA-Factory
if [ ! -d "LLaMA-Factory" ]; then
    echo "📦 Cloning LLaMA-Factory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
else
    echo "✓ LLaMA-Factory already exists"
    cd LLaMA-Factory
    git pull
fi

echo ""
echo "📦 Installing LLaMA-Factory..."
$PIP_CMD install -e ".[torch,metrics]"

echo ""
echo "📦 Installing additional dependencies for Qwen2.5-Omni..."
$PIP_CMD install deepspeed
# Flash-attention optional (requires CUDA)
# $PIP_CMD install flash-attn --no-build-isolation

echo ""
echo "✅ LLaMA-Factory setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Prepare your dataset: cd .. && python prepare_data.py"
echo "   2. Start training: cd LLaMA-Factory && llamafactory-cli train ../qwen25omni_asr_qlora.yaml"
echo ""
