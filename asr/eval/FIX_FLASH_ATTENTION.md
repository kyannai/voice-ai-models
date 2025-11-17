# Fix Flash-Attention Compatibility Issue

## Problem
```
ImportError: flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv
```

Flash-attention was compiled with PyTorch 2.9.0 but you now have PyTorch 2.1.2.

## Solution 1: Uninstall Flash-Attention (Recommended - Faster)

Flash-attention is optional. Qwen2.5-Omni will fall back to standard attention:

```bash
cd ~/voice-ai/asr/eval
source .venv/bin/activate

# Simply uninstall flash-attention
pip uninstall -y flash-attn

# Run evaluation
python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```

**Note:** Without flash-attention, inference will be slightly slower but will work fine.

## Solution 2: Reinstall Flash-Attention (Slower - requires compilation)

If you want optimal speed, reinstall flash-attention for PyTorch 2.1.2:

```bash
cd ~/voice-ai/asr/eval
source .venv/bin/activate

# Uninstall old version
pip uninstall -y flash-attn

# Install compatible version (takes 5-10 minutes to compile)
pip install flash-attn==2.3.6 --no-build-isolation

# This will compile flash-attention for PyTorch 2.1.2
# Wait for compilation to complete...

# Run evaluation
python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```

## Recommended Approach

For quick testing: Use **Solution 1** (uninstall flash-attn)  
For production: Use **Solution 2** (reinstall compatible version)

The code already handles flash-attention gracefully - it will use it if available, otherwise fall back to standard attention.
