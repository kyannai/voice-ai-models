#!/bin/bash

# Clean environment setup for ASR training with NeMo 1.23.0
# This script removes the old venv and creates a fresh one with correct versions

set -e  # Exit on error

cd "$(dirname "$0")"

echo "🧹 Removing old virtual environment..."
rm -rf .venv

echo "📦 Creating new virtual environment with uv..."
uv venv

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing build dependencies..."
uv pip install Cython

echo "📦 Installing base training dependencies (transformers, torch, etc.)..."
uv pip install -r requirements.txt

echo "📦 Installing NeMo from GitHub main branch (recommended for Parakeet)..."
echo "   (This may take 5-10 minutes - compiling from source)"
# Install NeMo from GitHub - this avoids all version conflicts!
# Reference: https://github.com/deepanshu-yadav/Hindi_GramVani_Finetune
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

echo ""
echo "✅ Clean environment setup complete!"
echo ""
echo "Verifying critical packages..."
python -c "import nemo; print(f'✓ NeMo: {nemo.__version__}')"
python -c "import nemo.collections.asr; print('✓ NeMo ASR: imports successfully')"
python -c "import pytorch_lightning; print(f'✓ PyTorch Lightning: {pytorch_lightning.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"

echo ""
echo "🚀 Environment is ready! You can now:"
echo "   cd train_parakeet_tdt"
echo "   ./run_training.sh"
echo ""
echo "Note: NeMo was installed from GitHub main branch for best compatibility"
echo "Reference: https://github.com/deepanshu-yadav/Hindi_GramVani_Finetune"

