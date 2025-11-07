#!/bin/bash
# Unified setup script for ASR evaluation environment

set -e  # Exit on error

echo "=========================================="
echo "ASR Training Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✓ uv is available"
    USE_UV=true
else
    echo "⚠ uv not found, will use pip instead"
    echo "  Install uv for faster package installation: curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

echo ""
echo "Installing all dependencies from requirements.txt..."
echo ""

# Install all requirements (consolidated)
echo "📦 Installing requirements..."
if [ "$USE_UV" = true ]; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi
echo "✓ All requirements installed"
echo ""

# Install Unsloth separately (extras cause dependency conflicts)
echo "📦 Installing Unsloth for Qwen2.5-Omni training..."
echo "   (Installing with minimal dependencies to avoid conflicts)"

if [ "$USE_UV" = true ]; then
    # Install unsloth_zoo first (required dependency)
    uv pip install unsloth_zoo
    # Then install base Unsloth without extras (they conflict with xformers)
    uv pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"
else
    pip install unsloth_zoo
    pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"
fi

# Verify installation
echo "   Verifying Unsloth installation..."
if python3 -c "from unsloth import FastLanguageModel" 2>/dev/null; then
    echo "✓ Unsloth installed successfully"
else
    echo "⚠ Unsloth import failed - training may not work for Qwen2.5-Omni"
    echo "  Try manual install:"
    echo "    uv pip install unsloth_zoo"
    echo "    uv pip install --no-deps 'unsloth @ git+https://github.com/unslothai/unsloth.git'"
fi
echo ""

echo "✅ Setup complete!"
echo ""
echo "📚 Next Steps:"
echo "   1. Activate environment: source .venv/bin/activate"
echo "   2. Configure training: cd train_qwen25omni && nano config.yaml"
echo "   3. Start training: bash run_training.sh"
echo ""
echo "💡 Optional Optimization:"
echo "   Install Flash-Attention 2 for 2-3x speedup:"
echo "   pip install flash-attn --no-build-isolation"
echo ""
echo "📖 See README.md for detailed instructions"
echo ""


