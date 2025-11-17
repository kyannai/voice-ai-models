# Fix for Qwen2.5-Omni Import Error

## Problem
```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'Qwen2_5OmniForConditionalGeneration'
```

This is caused by incompatible versions of PyTorch and torchvision.

## Solution

Run these commands in your virtual environment:

```bash
# Activate your environment first
cd ~/voice-ai/asr/eval
source .venv/bin/activate

# Uninstall existing torch packages
pip uninstall -y torch torchvision torchaudio

# Reinstall with compatible versions (CUDA 11.8)
uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1:
# uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
#   --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; import torchvision; print(f'Torch: {torch.__version__}, Torchvision: {torchvision.__version__}')"

# Test Qwen import
python -c "from transformers import Qwen2_5OmniForConditionalGeneration; print('✓ Qwen2.5-Omni imports successfully!')"
```

## Alternative: Use transformers-compatible torch

If the above doesn't work, try installing the latest compatible versions:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## After Fixing

Run your evaluation again:
```bash
python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```
