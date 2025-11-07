# Requirements Update Summary

**Updated:** 2025-11-06  
**Status:** Both training and evaluation requirements updated to latest stable versions

## Overview

Both `train/requirements.txt` and `eval/requirements.txt` have been updated to use the latest stable versions of all dependencies as of November 2024/early 2025.

## Key Version Updates

### Core ML Frameworks

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `torch` | ≥2.0.0 | ≥2.5.0 | Latest stable PyTorch |
| `torchaudio` | ≥2.0.0 | ≥2.5.0 | Matches PyTorch version |
| `transformers` | ≥4.30.0/4.37.0 | ≥4.45.0 | Latest HuggingFace |
| `accelerate` | ≥0.20.0 | ≥0.34.0 | Better training acceleration |

### LoRA & Quantization

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `peft` | ≥0.7.0 | ≥0.13.0 | Improved LoRA support |
| `bitsandbytes` | ≥0.41.0 | ≥0.44.0 | Better quantization |

### NeMo Framework

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `omegaconf` | ≥2.3.0 | ≥2.3.0 | Stable (no change) |
| `hydra-core` | ≥1.3.0 | ≥1.3.2 | Patch update |
| `lightning` | (implicit) | ≥2.0.0 | **NEW:** Explicit dependency |

### Audio Processing

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `librosa` | ≥0.10.0 | ≥0.10.2 | Patch updates |
| `soundfile` | ≥0.12.0 | ≥0.12.1 | Latest stable |
| `audioread` | ≥3.0.0 | ≥3.0.0 | No change |

### Data Processing

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `numpy` | ≥1.24.0 | ≥1.26.0,<2.0 | **IMPORTANT:** Capped at <2.0 for compatibility |
| `pandas` | ≥2.0.0 | ≥2.2.0 | Latest stable |
| `datasets` | ≥2.14.0 | ≥2.20.0 | (Training only) |

### Evaluation Metrics

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `jiwer` | ≥3.0.0 | ≥3.0.3 | Patch updates |
| `scikit-learn` | ≥1.3.0 | ≥1.5.0 | Latest stable |

### Multimodal Dependencies

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `pillow` | ≥9.0.0 | ≥10.4.0 | Major update |
| `torchvision` | ≥0.15.0 | ≥0.20.0 | Matches torch 2.5 |

### Training Utilities

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `tensorboard` | ≥2.14.0 | ≥2.17.0 | Latest stable |
| `tqdm` | ≥4.65.0 | ≥4.66.0 | Minor update |
| `wandb` | (commented) | ≥0.18.0 | **NEW:** Recommended for tracking |

### FunASR/ModelScope

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `funasr` | ≥1.0.0 | ≥1.1.0 | Latest stable |
| `modelscope` | ≥1.11.0 | ≥1.18.0 | Significant update |
| `onnxruntime` | ≥1.14.0 | ≥1.19.0 | Major update |

### Configuration

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `pyyaml` | ≥6.0 | ≥6.0.2 | Patch update |
| `python-dotenv` | ≥1.0.0 | ≥1.0.1 | Patch update |

## New Additions

### Training Requirements
- `lightning>=2.0.0` - Explicit PyTorch Lightning dependency for NeMo
- `wandb>=0.18.0` - Recommended experiment tracking (uncommented)

### Evaluation Requirements
- `lightning>=2.0.0` - Explicit PyTorch Lightning dependency for NeMo

### Optional Dependencies (Commented)
Both files now include commented optional dependencies:
- `deepspeed` - For distributed training optimization
- `flash-attn` - Flash Attention for faster training
- `faster-whisper` - Faster Whisper inference
- `mlflow` - Alternative experiment tracking

## Important Notes

### NumPy Version Cap
```python
numpy>=1.26.0,<2.0
```
NumPy 2.x introduces breaking changes. Many ML libraries are still catching up, so we cap at <2.0 for stability.

### PyTorch Lightning Explicit
```python
lightning>=2.0.0
```
Previously implicit through NeMo. Now explicit to avoid version conflicts.

### Wandb Uncommented
```python
wandb>=0.18.0  # Optional but recommended for experiment tracking
```
Previously commented out. Now recommended for tracking experiments.

## Compatibility Matrix

### Tested Configurations

| Framework | PyTorch | Python | Notes |
|-----------|---------|--------|-------|
| **Parakeet TDT (NeMo)** | 2.5.0 | 3.10+ | Requires PyTorch Lightning 2.0+ |
| **Qwen2.5-Omni** | 2.5.0 | 3.10+ | Requires transformers 4.45+ |
| **Qwen3-Omni** | 2.5.0 | 3.10+ | Requires pillow 10.4+ |
| **Whisper** | 2.5.0 | 3.8+ | Most compatible |
| **Paraformer** | 2.5.0 | 3.8+ | Via FunASR |

### CUDA Compatibility
- PyTorch 2.5.0 supports CUDA 11.8, 12.1, 12.4
- For optimal performance with NeMo: Use CUDA 11.8+
- Most packages auto-detect CUDA

## Installation

### Fresh Installation

**Training Environment:**
```bash
cd asr/train
pip install -r requirements.txt

# Install NeMo separately
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

**Evaluation Environment:**
```bash
cd asr/eval
pip install -r requirements.txt

# Install NeMo separately if needed
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Upgrading Existing Environment

```bash
# Upgrade all packages to latest versions
pip install --upgrade -r requirements.txt

# Check for conflicts
pip check
```

### Optional Dependencies

```bash
# Add experiment tracking
pip install wandb>=0.18.0

# Add distributed training optimization
pip install deepspeed>=0.14.0

# Add Flash Attention (requires CUDA)
pip install flash-attn>=2.6.0

# Add faster Whisper inference
pip install faster-whisper>=1.0.0
```

## Breaking Changes

### NumPy 2.x
If you're using NumPy 2.x, you may encounter issues. We explicitly cap at `<2.0`.

### PyTorch 2.5.0
- Some older models may need updates
- Check model compatibility with PyTorch 2.5
- Most HuggingFace models are compatible

### Transformers 4.45.0
- Includes many new model architectures
- Some APIs have deprecation warnings
- Qwen models fully supported

## Verification

After installation, verify key packages:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
python -c "import nemo; print(f'NeMo: {nemo.__version__}')"
```

## Migration Guide

### From Old Requirements

If you're upgrading from the old requirements:

1. **Backup your environment**
   ```bash
   pip freeze > old_requirements.txt
   ```

2. **Create fresh environment** (recommended)
   ```bash
   python -m venv venv_new
   source venv_new/bin/activate
   pip install -r requirements.txt
   ```

3. **Or upgrade in place** (advanced)
   ```bash
   pip install --upgrade -r requirements.txt
   pip check  # Verify no conflicts
   ```

4. **Test your models**
   - Run inference on test data
   - Check training still works
   - Verify metrics calculation

## Troubleshooting

### Common Issues

**Issue:** `pip install` fails with dependency conflicts  
**Solution:** Use a fresh virtual environment or conda environment

**Issue:** NumPy 2.x installed  
**Solution:** `pip install "numpy>=1.26.0,<2.0" --force-reinstall`

**Issue:** NeMo installation fails  
**Solution:** Install from source:
```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install -e ".[asr]"
```

**Issue:** CUDA out of memory with PyTorch 2.5  
**Solution:** PyTorch 2.5 has better memory management, but may allocate differently. Adjust batch sizes.

## Future Updates

These requirements will be reviewed and updated:
- Quarterly for minor version bumps
- Immediately for security patches
- When new major model releases require updates

## References

- [PyTorch Releases](https://pytorch.org/get-started/previous-versions/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers/releases)
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
- [FunASR Releases](https://github.com/alibaba-damo-academy/FunASR/releases)

---

**Last Updated:** 2025-11-06  
**Maintainer:** Voice AI Team  
**Status:** ✅ Production Ready

