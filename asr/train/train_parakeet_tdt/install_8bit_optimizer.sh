#!/bin/bash
# Quick install script for 8-bit optimizer support

set -e

echo "========================================"
echo "Installing 8-bit Optimizer Support"
echo "========================================"
echo ""

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating one first:"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing bitsandbytes..."
pip install bitsandbytes

echo ""
echo "✓ Installation complete!"
echo ""
echo "The 8-bit AdamW optimizer is now available."
echo "It will reduce optimizer memory by ~75% with minimal speed impact."
echo ""
echo "Your config is already set to use it:"
echo "  optimizer: \"adamw_8bit\""
echo ""
echo "Now you can run training:"
echo "  bash run_training.sh"
echo ""

