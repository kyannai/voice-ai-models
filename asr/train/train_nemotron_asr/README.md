# Nemotron Speech Streaming ASR Training

Fine-tune NVIDIA Nemotron Speech Streaming 0.6B for Malay language automatic speech recognition (ASR).

## Overview

**Why Nemotron Speech Streaming?**
- **Low-Latency Streaming**: Cache-aware architecture for real-time ASR
- **Configurable Chunk Sizes**: 80ms, 160ms, 560ms, 1120ms
- **Auto-Punctuation**: Built-in punctuation and capitalization
- **High Quality**: Excellent English transcription quality
- **Lightweight**: Only 0.6B parameters (~4-6GB VRAM)

**Model**: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)  
**Framework**: NVIDIA NeMo

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (A100 recommended)
- ~50GB disk space for training outputs
- Training data manifests (use `train_parakeet_tdt` for data preparation)

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
make install
make install-8bit  # Recommended

# 3. Check GPU availability
make check-gpu

# 4. Prepare data (use manifests from train_parakeet_tdt)
ln -s ../train_parakeet_tdt/data/manifests data/manifests

# 5. Start training
make train

# 6. Monitor training
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

### Step 2: Prepare Data

This project shares data preparation with `train_parakeet_tdt`. Either:

**Option A: Symlink manifests**
```bash
ln -s ../train_parakeet_tdt/data/manifests data/manifests
```

**Option B: Copy manifests**
```bash
cp -r ../train_parakeet_tdt/data/manifests data/
```

**Option C: Prepare fresh data**
```bash
cd ../train_parakeet_tdt
make download && make unzip && make prepare
cd ../train_nemotron_asr
ln -s ../train_parakeet_tdt/data/manifests data/manifests
```

### Step 3: Configure Training

Edit `configs/nemotron_streaming.yaml`:

```yaml
model:
  name: "nvidia/nemotron-speech-streaming-en-0.6b"
  gradient_checkpointing: true

data:
  train_manifest: "./data/manifests/train_manifest.json"
  val_manifest: "./data/manifests/val_manifest.json"

training:
  num_train_epochs: 1
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4
```

### Step 4: Start Training

```bash
# Train with default config
make train

# Train with custom config
make train-custom CONFIG=path/to/config.yaml

# Resume from checkpoint
make train-resume CHECKPOINT=path/to/checkpoint.ckpt
```

### Step 5: Monitor Training

```bash
make tensorboard
```

Open http://localhost:6006 to view:
- `val_wer` - Word Error Rate (should decrease)
- `train_loss` - Training loss
- `lr` - Learning rate schedule

## Project Structure

```
train_nemotron_asr/
├── configs/
│   └── nemotron_streaming.yaml   # Training configuration
├── data/
│   └── manifests/                # Symlink to train_parakeet_tdt manifests
├── models/                       # Saved models
├── outputs/                      # Training outputs & checkpoints
├── src/
│   └── train.py                  # Main training script
├── Makefile                      # Automation commands
├── requirements.txt              # Python dependencies
└── README.md
```

## Make Commands

```bash
make help           # Show all commands

# Installation (Step 1)
make install        # Install dependencies
make install-8bit   # Install 8-bit optimizer
make check-gpu      # Check GPU availability

# Training (Step 2)
make train          # Train with default config
make train-custom CONFIG=path  # Custom config
make train-resume CHECKPOINT=path  # Resume training

# Monitoring
make tensorboard    # Start TensorBoard

# Cleanup
make clean-outputs  # Remove training outputs
```

## Data Format

NeMo manifest files in JSONL format:

```jsonl
{"audio_filepath": "/path/to/audio1.wav", "text": "transcription one", "duration": 2.5}
{"audio_filepath": "/path/to/audio2.wav", "text": "transcription two", "duration": 3.2}
```

## Technical Notes

- Nemotron uses cache-aware FastConformer encoder with RNNT decoder
- Streaming capability is inference-only (training is standard NeMo ASR)
- Native support for punctuation and capitalization
- bf16 recommended for A100/H100 GPUs

## Model Card: Nemotron Speech Streaming Malay ASR

**Status**: Not yet trained

### Planned Training

| Field | Value |
|-------|-------|
| Base Model | [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) |
| Architecture | Cache-aware FastConformer + RNNT |
| Parameters | 0.6B |
| Framework | NVIDIA NeMo |

### Training Configuration

Refer to `train_parakeet_tdt` for training parameters:
- Dataset: [mesolitica/Malaysian-STT-Whisper](https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper)
- Config template: `../train_parakeet_tdt/configs/parakeet_stage1.yaml`

### Notes

- Uses same data pipeline as Parakeet TDT
- Streaming inference capability for real-time ASR
- Update this section after training is complete

## License

Training scripts are provided as-is. Check NVIDIA's model license for usage terms.
