# Fix PyTorch Version Requirement for Qwen2.5-Omni

## Problem
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, 
we now require users to upgrade torch to at least v2.6
```

## Root Cause
- Transformers library requires PyTorch 2.6+ for security reasons
- We're using PyTorch 2.1.2 for compatibility with other packages
- The model is trying to load speaker embeddings with `torch.load()`

## Solution 1: Downgrade Transformers (Recommended - Quick)

Use an older transformers version that doesn't have this strict check:

```bash
cd ~/voice-ai/asr/eval
source .venv/bin/activate

# Downgrade transformers to version before security check
pip install transformers==4.44.2

# Run evaluation
python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```

## Solution 2: Upgrade PyTorch (May Cause Other Issues)

Upgrade to PyTorch 2.6+ (warning: may break other dependencies):

```bash
cd ~/voice-ai/asr/eval
source .venv/bin/activate

# Uninstall current torch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch 2.6+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Note: You'll need to reinstall flash-attn if you want it
# pip uninstall -y flash-attn
# pip install flash-attn --no-build-isolation

# Run evaluation
python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```

## Solution 3: Use Transformers Main Branch (Advanced)

Install transformers from main branch which may have better compatibility:

```bash
cd ~/voice-ai/asr/eval
source .venv/bin/activate

pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git

python evaluate.py --model Qwen/Qwen2.5-Omni-7B --test-dataset seacrowd-asr-malcsc
```

## Recommended Approach

**For immediate testing:** Use Solution 1 (downgrade transformers to 4.44.2)

This is the safest option that:
- Keeps PyTorch 2.1.2 (compatible with your setup)
- Works with Qwen2.5-Omni
- Avoids the security check (acceptable for research/testing)

## Security Note

The vulnerability (CVE-2025-32434) is related to `torch.load()`. For production use, 
consider upgrading to PyTorch 2.6+ when possible. For research/evaluation, 
the older version is acceptable if you trust your model sources.
