# Parakeet TDT ASR Training

Fine-tune NVIDIA Parakeet TDT (Token-and-Duration Transducer) 0.6B v3 for Malay language automatic speech recognition (ASR).

## Overview

**Why Parakeet TDT?**
- **Lightning-Fast**: 60 minutes of audio in ~1 second
- **High Accuracy**: 98% on long audio files (up to 24 minutes)
- **Auto-Punctuation**: Built-in punctuation and capitalization
- **Word Timestamps**: Precise word-level timing
- **Lightweight**: Only 0.6B parameters (~4-6GB VRAM)
- **Easy to Fine-tune**: No quantization needed

**Model**: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)  
**Dataset**: [mesolitica/Malaysian-STT-Whisper](https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper) (~5.2M samples)  
**Framework**: NVIDIA NeMo

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (A100 recommended)
- ~100GB disk space for dataset + training outputs

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
make install
make install-8bit  # Recommended for large datasets

# 3. Download and prepare data
make download      # Download from HuggingFace
make unzip         # Extract zip files
make prepare       # Create NeMo manifests

# 4. Start training
make train

# 5. Monitor training
make tensorboard
```

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
make install
```

For large dataset training (recommended):
```bash
make install-8bit
```

### Step 2: Download Dataset

```bash
# Download Malaysian-STT dataset from HuggingFace (~100GB)
make download

# Extract zip files
make unzip
```

### Step 3: Prepare Training Data

```bash
# Full dataset (~5.2M samples)
make prepare

# OR small subset for testing (10k samples)
make prepare-small
```

This creates NeMo manifest files:
- `data/manifests/train_manifest.json` - Training manifest
- `data/manifests/val_manifest.json` - Validation manifest

**NeMo Manifest Format** (JSONL):
```jsonl
{"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 2.5}
```

### Step 4: Configure Training

Edit the appropriate config file:

```bash
# Stage 1: Initial fine-tuning
vim configs/parakeet_stage1.yaml

# Stage 2: Continued training
vim configs/parakeet_stage2.yaml
```

Key settings:
```yaml
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  gradient_checkpointing: true

data:
  train_manifest: "./data/manifests/train_manifest.json"
  val_manifest: "./data/manifests/val_manifest.json"

training:
  num_train_epochs: 1
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4
```

### Step 5: Start Training

```bash
# Train with stage1 config (default)
make train

# Train with stage2 config
make train-stage2

# Train with custom config
make train-custom CONFIG=path/to/config.yaml
```

### Step 6: Monitor Training

```bash
make tensorboard
```

Open http://localhost:6006 to view:
- `val_wer` - Word Error Rate (should decrease)
- `train_loss` - Training loss
- `lr` - Learning rate schedule

### Step 7: Upload to HuggingFace

```bash
export HF_TOKEN='your_huggingface_token'
make upload MODEL=models/full.nemo REPO=parakeet-tdt-0.6b-malay
```

## Project Structure

```
train_parakeet_tdt/
├── configs/
│   ├── parakeet_stage1.yaml    # Stage 1: Initial fine-tuning
│   └── parakeet_stage2.yaml    # Stage 2: Continued training
├── data/
│   ├── raw/                    # Downloaded dataset
│   └── manifests/              # NeMo manifest files
├── models/                     # Saved models
├── outputs/                    # Training outputs & checkpoints
├── src/
│   ├── prepare_data.py         # Data preparation script
│   ├── train.py                # Main training script
│   ├── upload_model.py         # HuggingFace upload utility
│   ├── check_tensorboard.py    # TensorBoard event checker
│   └── unzip_data.py           # Data extraction utility
├── Makefile                    # Automation commands
├── requirements.txt            # Python dependencies
└── README.md
```

## Make Commands

```bash
make help           # Show all commands

# Installation (Step 1)
make install        # Install dependencies
make install-8bit   # Install 8-bit optimizer
make check-gpu      # Check GPU availability

# Data Management (Step 2)
make download       # Download Malaysian-STT dataset
make unzip          # Extract zip files
make prepare        # Prepare full dataset
make prepare-small  # Prepare small subset (10k samples)

# Training (Step 3)
make train          # Train with stage1 config (default)
make train-stage1   # Train stage 1
make train-stage2   # Train stage 2
make train-custom CONFIG=path  # Custom config

# Monitoring
make tensorboard    # Start TensorBoard
make check-events   # Check event files

# Model Upload
make upload MODEL=path REPO=name  # Upload to HuggingFace

# Cleanup
make clean-outputs  # Remove training outputs
make clean-data     # Remove downloaded data
make clean-all      # Remove all generated files
```

## Training Configurations

### Stage 1: Initial Fine-tuning

For training on the full Malaysian-STT dataset (~5.2M samples):

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2.0e-4 |
| Batch Size | 8 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 128 |
| Optimizer | AdamW 8-bit |
| Precision | bfloat16 |
| Estimated Time | ~4 days on A100 |

### Stage 2: Continued Training

For additional fine-tuning on new data:

| Parameter | Value |
|-----------|-------|
| Base Model | Stage 1 checkpoint |
| Learning Rate | 2.0e-4 |
| Epochs | 1 |

## Hardware Requirements

### Minimum (Consumer GPUs)
- **GPU**: 8GB VRAM (RTX 3060, RTX 4060)
- **RAM**: 16GB
- **Batch Size**: 4-8

### Recommended (Professional GPUs)
- **GPU**: 24GB+ VRAM (RTX 4090, A5000)
- **RAM**: 32GB
- **Batch Size**: 16-32

### High-Performance (Data Center)
- **GPU**: 40GB+ VRAM (A100, H100)
- **RAM**: 64GB
- **Batch Size**: 32-64+

## Troubleshooting

### CUDA Out of Memory

1. Reduce batch size in config:
   ```yaml
   per_device_train_batch_size: 4
   ```

2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 32
   ```

3. Enable gradient checkpointing:
   ```yaml
   model:
     gradient_checkpointing: true
   ```

### ModuleNotFoundError: No module named 'nemo'

```bash
make install
```

### Training is too slow

1. Increase batch size if VRAM allows
2. Reduce `dataloader_num_workers` if CPU is bottleneck
3. Use SSD for audio files
4. Consider multi-GPU training

### Validation WER not improving

1. Check transcription accuracy
2. Try lower learning rate: `1e-5`
3. May need more epochs
4. Verify audio quality (16kHz+)

## License

Training scripts are provided as-is. Check NVIDIA's model license for usage terms.

## Acknowledgements

- NVIDIA NeMo Team for the excellent framework
- NVIDIA for the base Parakeet TDT model
- mesolitica for the Malaysian-STT dataset
