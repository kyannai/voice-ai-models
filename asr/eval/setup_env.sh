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
echo "Installing dependencies..."
echo ""

# Install PyTorch with CUDA first (if not already installed)
echo "📦 Checking PyTorch installation..."
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✓ PyTorch already installed (version: $TORCH_VERSION)"
else
    echo "PyTorch not found. Installing with CUDA 11.8 support..."
    echo "This may take a few minutes..."
    if [ "$USE_UV" = true ]; then
        uv pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 \
          --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 \
          --index-url https://download.pytorch.org/whl/cu118
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ PyTorch installed successfully"
    else
        echo "❌ PyTorch installation failed!"
        exit 1
    fi
fi

# Install package in editable mode (uses pyproject.toml)
echo ""
echo "📦 Installing asr-eval package..."
if [ "$USE_UV" = true ]; then
    uv pip install -e .
else
    pip install -e .
fi

if [ $? -eq 0 ]; then
    echo "✓ Base dependencies installed"
else
    echo "❌ Installation failed!"
    echo "Check the error messages above."
    exit 1
fi
echo ""

# Ask about optional dependencies
echo ""
echo "=========================================="
echo "Optional Dependencies"
echo "=========================================="
echo ""
echo "Do you want to install optional model support?"
echo ""
echo "Options:"
echo "  1) Whisper only (current - already installed)"
echo "  2) + Qwen models (Qwen2-Audio, Qwen2.5-Omni)"
echo "  3) + Parakeet models (NVIDIA NeMo)"
echo "  4) All models (Qwen + Parakeet)"
echo "  5) Skip optional dependencies"
echo ""
read -p "Enter choice [1-5] (default: 5): " choice
choice=${choice:-5}

case $choice in
    2)
        echo ""
        echo "Installing Qwen model support..."
        if [ "$USE_UV" = true ]; then
            uv pip install -e ".[qwen]"
        else
            pip install -e ".[qwen]"
        fi
        echo "✓ Qwen models support installed"
        ;;
    3)
        echo ""
        echo "Installing Parakeet model support..."
        if [ "$USE_UV" = true ]; then
            uv pip install -e ".[parakeet]"
        else
            pip install -e ".[parakeet]"
        fi
        
        echo ""
        echo "Installing NeMo toolkit (this may take several minutes)..."
        if [ "$USE_UV" = true ]; then
            uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
        else
            pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
        fi
        
        if [ $? -eq 0 ]; then
            echo "✓ Parakeet models support installed (including NeMo)"
        else
            echo "⚠️  Parakeet dependencies installed, but NeMo failed"
            echo "This is often due to transformers version conflicts"
            echo "You may need to manually resolve the conflict"
        fi
        ;;
    4)
        echo ""
        echo "Installing all model support (Qwen + Parakeet)..."
        echo "Note: This may take several minutes..."
        if [ "$USE_UV" = true ]; then
            uv pip install -e ".[all]"
        else
            pip install -e ".[all]"
        fi
        
        echo ""
        echo "Installing NeMo toolkit (this may take several minutes)..."
        if [ "$USE_UV" = true ]; then
            uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
        else
            pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
        fi
        
        if [ $? -eq 0 ]; then
            echo "✓ All models support installed (including NeMo)"
        else
            echo "⚠️  All dependencies installed, but NeMo failed"
            echo "This is often due to transformers version conflicts"
            echo "You may need to manually resolve the conflict"
        fi
        ;;
    1|5)
        echo ""
        echo "Skipping optional dependencies (Whisper-only mode)"
        echo ""
        echo "To install later:"
        echo "  For Qwen:     uv pip install -e '.[qwen]'"
        echo "  For Parakeet: uv pip install -e '.[parakeet]'"
        echo "                uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
        echo "  For all:      uv pip install -e '.[all]'"
        echo "                uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
        ;;
    *)
        echo "Invalid choice. Skipping optional dependencies."
        ;;
esac

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "What's installed:"
echo "  ✓ PyTorch with CUDA support"
echo "  ✓ Core dependencies (transformers, librosa, pandas, jiwer)"
echo "  ✓ Whisper evaluation support"
echo "  ✓ Metrics calculation (WER, CER, MER)"
echo ""
echo "Optional installations:"
echo "  For Qwen models:    uv pip install -e '.[qwen]'"
echo "  For Parakeet:       uv pip install -e '.[parakeet]'"
echo "                      uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
echo "  For all models:     uv pip install -e '.[all]'"
echo "                      uv pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
echo "  For development:    uv pip install -e '.[dev]'"
echo ""
echo "Note: NeMo toolkit (for Parakeet) must be installed separately due to"
echo "      dependency conflicts. It requires transformers>=4.53.0."
echo ""
echo "Quick start:"
echo ""
echo "  # Single model evaluation"
echo "  python evaluate.py --model openai/whisper-large-v3-turbo \\"
echo "    --test-data test_data/malaya-test/malaya-malay-test-set.json \\"
echo "    --audio-dir test_data/malaya-test"
echo ""
echo "  # Batch evaluation (multiple models)"
echo "  python batch_evaluate.py \\"
echo "    --test-data test_data/malaya-test/malaya-malay-test-set.json \\"
echo "    --audio-dir test_data/malaya-test"
echo ""
echo "For more details:"
echo "  - See README.md for overview"
echo "  - See INSTALL.md for installation options"
echo "  - Run ./run_batch_eval.sh for 3-model comparison"
echo ""

