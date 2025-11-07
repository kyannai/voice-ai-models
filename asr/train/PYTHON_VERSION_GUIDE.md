# Python Version Guide for NeMo Training

**Issue:** You have Python 3.12, but PyTorch 2.1.2 only supports Python 3.8-3.11

## Quick Solution

Run the updated setup script - it now handles this automatically:

```bash
cd /home/kyan/voice-ai/asr/train
./setup_clean_venv.sh
```

The script will detect Python 3.12 and give you two options:
- **Option A:** Use Python 3.11 with PyTorch 2.1.2 (RECOMMENDED)
- **Option B:** Use Python 3.12 with PyTorch 2.2.2 (also works)

---

## Option A: Python 3.11 + PyTorch 2.1.2 (RECOMMENDED)

**Why:** Most stable, best tested with NeMo

### Install Python 3.11

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**RHEL/CentOS:**
```bash
sudo dnf install python3.11 python3.11-devel
```

**macOS:**
```bash
brew install python@3.11
```

### Then Run Setup

```bash
./setup_clean_venv.sh
```

When prompted, choose **A** for Python 3.11.

---

## Option B: Python 3.12 + PyTorch 2.2.2 (Alternative)

**Why:** If you can't install Python 3.11

### Compatibility

| Component | Status with PyTorch 2.2.2 |
|-----------|---------------------------|
| NeMo | ✅ Works |
| Numba CUDA | ✅ Works (with CUDA 11.8/12.1) |
| Training | ✅ Works |
| All features | ✅ Supported |

### Setup

```bash
./setup_clean_venv.sh
```

When prompted, choose **B** for Python 3.12.

---

## Python & PyTorch Compatibility Matrix

| Python Version | PyTorch 2.1.2 | PyTorch 2.2.2 | PyTorch 2.3.0 |
|----------------|---------------|---------------|---------------|
| 3.8 | ✅ | ✅ | ✅ |
| 3.9 | ✅ | ✅ | ✅ |
| 3.10 | ✅ | ✅ | ✅ |
| 3.11 | ✅ | ✅ | ✅ |
| 3.12 | ❌ | ✅ | ✅ |

### NeMo Compatibility

| PyTorch Version | NeMo Compatibility | Notes |
|-----------------|-------------------|-------|
| 2.1.2 | ✅ Excellent | Most tested, recommended |
| 2.2.2 | ✅ Good | Fully supported |
| 2.3.0 | ✅ Good | Works well |
| 2.4.0+ | ⚠️ Experimental | May have issues |

---

## Manual Setup (If Script Doesn't Work)

### With Python 3.11

```bash
cd /home/kyan/voice-ai/asr/train

# Remove old venv
rm -rf .venv

# Create venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install packages
pip install --upgrade pip uv

# Install PyTorch 2.1.2
uv pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
uv pip install -r requirements.txt

# Install NeMo
uv pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### With Python 3.12

```bash
cd /home/kyan/voice-ai/asr/train

# Remove old venv
rm -rf .venv

# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install packages
pip install --upgrade pip uv

# Install PyTorch 2.2.2
uv pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
uv pip install -r requirements.txt

# Install NeMo
uv pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

---

## Verification

After setup, verify your installation:

```bash
source .venv/bin/activate

python -c "
import sys
import torch
import nemo

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'NeMo: {nemo.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Expected Output (Python 3.11)

```
Python: 3.11.x ...
PyTorch: 2.1.2+cu118
NeMo: 2.0.0 (or similar)
CUDA available: True
```

### Expected Output (Python 3.12)

```
Python: 3.12.x ...
PyTorch: 2.2.2+cu118
NeMo: 2.0.0 (or similar)
CUDA available: True
```

---

## Recommendation

For **production training**, use:
- **Python 3.11**
- **PyTorch 2.1.2**
- **CUDA 11.8**

This is the **most tested and stable** combination for NeMo.

---

## Why Not Python 3.12 with PyTorch 2.1.2?

PyTorch wheels are built for specific Python versions:
- PyTorch 2.1.2 was released before Python 3.12
- No wheels were built for `cp312` (CPython 3.12) ABI tag
- You'd have to build PyTorch from source (not recommended)

---

## Questions?

- 📁 Updated setup script: `setup_clean_venv.sh`
- 📄 Requirements: `requirements.txt`
- 📖 Fresh install guide: `FRESH_INSTALL_GUIDE.md`

---

**Last Updated:** 2025-11-06  
**Status:** ✅ Setup script updated to handle Python 3.12

