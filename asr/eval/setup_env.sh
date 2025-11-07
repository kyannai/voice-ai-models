#!/bin/bash
# Unified setup script for ASR evaluation environment

set -e  # Exit on error

echo "=========================================="
echo "ASR Evaluation Environment Setup"
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

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "What's installed:"
echo "  ✓ Core dependencies (torch, librosa, pandas, jiwer, etc.)"
echo "  ✓ Whisper framework (transformers)"
echo "  ✓ FunASR/Qwen2-Audio framework"
echo "  ✓ Metrics calculation tools"
echo ""
echo "Next steps:"
echo ""
echo "For Whisper evaluation:"
echo "  cd transcribe/whisper"
echo "  python transcribe_whisper.py --help"
echo ""
echo "For FunASR/Qwen2-Audio evaluation:"
echo "  cd transcribe/funasr"
echo "  python transcribe_funasr.py --help"
echo ""
echo "For metrics calculation:"
echo "  cd calculate_metrics"
echo "  python calculate_metrics.py --help"
echo ""
echo "For more details, see README.md in each folder"
echo ""

