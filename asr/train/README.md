# ASR Model Training

Training pipelines for ASR models.

## Directory Structure

```
asr/train/
├── training_data/        # Dataset preparation (Malaysian STT, KeSpeech, etc.)
├── train_parakeet_tdt/   # NVIDIA Parakeet TDT training
├── train_parakeet_ctc/   # NVIDIA Parakeet CTC training
├── train_nemotron_asr/   # Nemotron ASR training
└── common/               # Shared utilities (tokenizers, normalizers)
```

## Quick Start

```bash
# 1. Prepare training data
cd training_data && make help

# 2. Train a model
cd train_parakeet_tdt && make help
```