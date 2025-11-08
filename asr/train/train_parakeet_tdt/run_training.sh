#!/bin/bash
# Training launcher for NVIDIA Parakeet TDT 0.6B v3
# This script handles environment setup and training execution

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NVIDIA Parakeet TDT Training Launcher${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if virtual environment exists
if [ -d "../.venv" ]; then
    echo -e "${GREEN}✓${NC} Found virtual environment"
    source ../.venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Found virtual environment"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠${NC}  No virtual environment found"
    echo -e "   Consider creating one with: python3 -m venv .venv"
fi

# Setup CUDA library path (to prevent libcudart.so.11.0 errors)
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-11.8/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda-12.1/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
fi

# Setup CUDA memory management (prevents fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}✗${NC} config.yaml not found!"
    echo -e "   Please create a config.yaml file first"
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration file found"

# Check if NeMo is installed
if ! python -c "import nemo" 2>/dev/null; then
    echo -e "${RED}✗${NC} NeMo toolkit not installed!"
    echo -e "   Install with: ${YELLOW}cd ../.. && pip install -r requirements.txt${NC}"
    echo -e "   Or just NeMo: ${YELLOW}pip install nemo_toolkit[asr]${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} NeMo toolkit installed"

# Check for CUDA
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓${NC} CUDA available: ${GPU_NAME}"
else
    echo -e "${YELLOW}⚠${NC}  CUDA not available, will train on CPU (very slow!)"
fi

# Skip confirmation - auto-start training
# echo -e "\n${BLUE}Ready to start training. Continue? [y/N]${NC}"
# read -r response
# if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
#     echo -e "${YELLOW}Training cancelled${NC}"
#     exit 0
# fi

# Start training
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Training${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}💡 TIP: Monitor training in real-time!${NC}"
echo -e "   Open a separate terminal and run:"
echo -e "   ${GREEN}tensorboard --logdir ./outputs${NC}"
echo -e "   Then open: ${GREEN}http://localhost:6006${NC}"
echo -e "   Look for the '${YELLOW}val_wer${NC}' graph (should decrease over time)"
echo -e ""

python train_parakeet_tdt.py --config config.yaml

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\nNext steps:"
    echo -e "  1. View logs: ${YELLOW}tensorboard --logdir ./outputs${NC}"
    echo -e "  2. Test model: ${YELLOW}cd ../../eval/transcribe${NC}"
    echo -e "  3. Run inference with the trained model"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Training failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\nCheck the logs above for error details"
    exit 1
fi

