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

## Model Card: Parakeet TDT 0.6B Malay ASR

This section documents how the Malay ASR model was trained for future maintainers.

### Model Information

| Field | Value |
|-------|-------|
| Base Model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer + TDT (Token-and-Duration Transducer) |
| Parameters | 0.6B |
| Framework | NVIDIA NeMo |
| Training Hardware | NVIDIA A100-SXM4-80GB |

### Training Pipeline Overview

The model was trained in two stages:

```
Stage 1: Malaysian-STT-Whisper (~5.2M samples)
    ↓
Stage 2: 5K Synthetic names & numbers data
```

---

### Stage 1: Initial Fine-tuning

**Dataset**: [mesolitica/Malaysian-STT-Whisper](https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper)

| Field | Value |
|-------|-------|
| Total Samples | ~5.2M audio-text pairs |
| Subsets Used | `malaysian_context_v2`, `extra` |
| Audio Format | 16kHz WAV |
| Languages | Malay (primary), some English |
| Train/Val Split | 95% / 5% |

**Config**: `configs/parakeet_stage1.yaml`

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 8 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 128 |
| Learning Rate | 2.0e-4 |
| LR Scheduler | CosineAnnealing |
| Warmup Steps | 100 |
| Optimizer | AdamW 8-bit |
| Precision | bfloat16 |
| Total Steps | ~659,506 |
| Duration | ~4 days on A100-80GB |
| Checkpoints | Every 41,000 steps (~6 hours) |

---

### Stage 2: 5K Synthetic Names & Numbers

**Dataset**: Synthetic data generated using TTS

Data location: `../training_data/5k_v3/`

| Field | Value |
|-------|-------|
| Total Samples | ~5,000 synthetic audio-text pairs |
| Content | Malaysian names, numbers, addresses |
| Generation | TTS synthesis (`synthetic_data_generation`) |
| Purpose | Improve recognition of names and numbers |

**Config**: `configs/parakeet_5k.yaml`

| Parameter | Value |
|-----------|-------|
| Base Model | Stage 1 checkpoint (`./models/full.nemo`) |
| Epochs | 1 |
| Batch Size | 20 |
| Gradient Accumulation | 6 |
| Effective Batch Size | 120 |
| Learning Rate | 5.0e-5 (lower for fine-tuning) |
| Warmup Steps | 10 |
| Steps per Epoch | ~38 |

**Synthetic Data Generation**:
```bash
cd ../training_data/5k_v3/src/
./prepare_synthetic_data.sh  # Generate TTS audio
python prepare_synthetic_manifests.py  # Create manifests
```

---

### Memory Optimizations

To fit large datasets on A100-80GB:
- 8-bit AdamW optimizer (75% optimizer memory reduction)
- Gradient checkpointing (30-50% activation memory reduction)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Reduced batch size with higher gradient accumulation

### Model Outputs

The trained model includes:
- Auto-punctuation and capitalization
- Word-level timestamps
- Support for audio up to 24 minutes

### Checkpoint Locations

```
# Stage 1: Malaysian-STT-Whisper
outputs/parakeet-tdt-malay-asr/
└── parakeet-tdt-malay-finetuning/
    └── YYYY-MM-DD_HH-MM-SS/checkpoints/

# Stage 2: 5K Synthetic
outputs/parakeet-tdt-5k-v3/
└── parakeet-tdt-5k-v3/
    └── YYYY-MM-DD_HH-MM-SS/checkpoints/
```

### Usage

```python
import nemo.collections.asr as nemo_asr

# Load trained model
model = nemo_asr.models.ASRModel.restore_from("path/to/checkpoint.nemo")

# Transcribe audio
transcription = model.transcribe(["audio.wav"])
print(transcription)
```

## License

Training scripts are provided as-is. Check NVIDIA's model license for usage terms.

## Acknowledgements

- NVIDIA NeMo Team for the excellent framework
- NVIDIA for the base Parakeet TDT model
- mesolitica for the Malaysian-STT dataset
