#!/bin/bash
# Complete pipeline for synthetic data generation and training
# Usage: bash run_pipeline.sh

set -e  # Exit on error

echo "========================================"
echo "Synthetic Data Generation Pipeline"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for API key
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: ELEVENLABS_API_KEY not set!"
    echo "Please set your ElevenLabs API key:"
    echo "export ELEVENLABS_API_KEY='your_api_key_here'"
    echo ""
    read -p "Press Enter to continue if you've set it in .env file..."
fi

# Step 1: Generate sentences
echo ""
echo "========================================"
echo "Step 1: Generating sentences..."
echo "========================================"
python scripts/generate_sentences.py \
    --config config.yaml \
    --output outputs/sentences.json

# Step 2: Synthesize audio with ElevenLabs
echo ""
echo "========================================"
echo "Step 2: Synthesizing audio..."
echo "========================================"
python scripts/synthesize_with_elevenlabs.py \
    --config config.yaml \
    --input outputs/sentences.json \
    --output outputs/synthesized.json \
    --resume  # Skip existing files

# Step 3: Prepare NeMo manifests
echo ""
echo "========================================"
echo "Step 3: Preparing NeMo manifests..."
echo "========================================"
python scripts/prepare_nemo_manifest.py \
    --config config.yaml \
    --input outputs/synthesized.json

echo ""
echo "========================================"
echo "✅ Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review the generated manifests in outputs/manifests/"
echo "2. Train Parakeet-TDT with:"
echo "   cd ../train/train_parakeet_tdt"
echo "   python train_parakeet_tdt.py --config ../../synthetic_data_generation/config_parakeet_training.yaml"
echo ""
echo "Dataset statistics are available in the logs above."
echo "========================================"

