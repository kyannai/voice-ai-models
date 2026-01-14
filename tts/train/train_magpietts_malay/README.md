# MagpieTTS Malay Fine-tuning

Fine-tune NVIDIA's MagpieTTS multilingual model to add Malay language support using the `mesolitica/Malaysian-TTS` dataset.

## Overview

This project adds Malay as a new language to the pretrained `nvidia/magpie_tts_multilingual_357m` model through continual fine-tuning. The training uses IPA phonemes (via espeak-ng) for better pronunciation quality.

**Model**: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)  
**Dataset**: [mesolitica/Malaysian-TTS](https://huggingface.co/datasets/mesolitica/Malaysian-TTS) (~500k samples, 5 speakers)

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (A100 recommended)
- ~100GB disk space for full dataset
- espeak-ng for phoneme conversion

```bash
# Install espeak-ng (required for phoneme conversion)
sudo apt install espeak-ng  # Ubuntu/Debian
brew install espeak         # macOS
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
make install

# 3. Download and prepare data (full dataset)
make download
make prepare

# 4. Start training
make train

# 5. Test synthesis with trained model
make synth TEXT="Selamat pagi" MODEL_PATH=experiments/magpietts_malay/magpietts_malay_finetune/checkpoints/last.nemo
```

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
make install
```

This installs:
- NeMo toolkit (from GitHub main branch)
- PyTorch Lightning
- Audio processing libraries (librosa, soundfile)
- Phonemizer with espeak-ng support

### Step 2: Download Dataset

```bash
# Full dataset (~500k samples, ~100GB)
make download

# OR small subset for testing (~10k samples)
make download-small
```

The dataset includes 5 Malaysian speakers:
- `anwar_ibrahim` - Male, political speech
- `husein` - Male, conversational
- `kp_ms` - Female, news
- `kp_zh` - Female, news (Mandarin-accented)
- `shafiqah_idayu` - Female, conversational

### Step 3: Explore Dataset (Optional)

```bash
make explore
```

Shows dataset statistics, sample texts, and speaker distribution.

### Step 4: Prepare Training Data

```bash
# Full dataset (recommended)
make prepare

# OR small subset for testing
make prepare-small
```

This step:
1. Converts audio to 22.05kHz WAV format
2. Converts text to IPA phonemes using espeak-ng
3. Creates NeMo-compatible JSON manifest files
4. Splits into 95% train / 5% validation

**Output:**
- `data/audio/` - Processed WAV files
- `data/manifests/train_manifest.json` - Training manifest
- `data/manifests/val_manifest.json` - Validation manifest

### Step 5: Start Training

```bash
# Train with 1 GPU (default)
make train

# Train with multiple GPUs
make train GPUS=4

# Resume from checkpoint
make train-resume CHECKPOINT=experiments/magpietts_malay/magpietts_malay_finetune/checkpoints/last.nemo
```

**Training Configuration** (`configs/magpietts_malay.yaml`):
- Batch size: 8
- Learning rate: 2e-4 (higher for new language)
- Max epochs: 100
- Precision: 16-bit mixed

**Estimated Time** (A100 GPU):
- ~8 hours per epoch
- Full training: 10-20 epochs recommended

### Step 6: Monitor Training

```bash
# Start TensorBoard
make tensorboard
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Learning rate schedule
- Audio samples (if configured)

### Step 7: Test Synthesis

```bash
# Test with trained model (uses phonemes)
make synth TEXT="Selamat pagi, apa khabar?" MODEL_PATH=experiments/magpietts_malay/magpietts_malay_finetune/checkpoints/last.nemo

# Test with original pretrained model (no phonemes)
make synth-chars TEXT="Hello world" MODEL_PATH=nvidia/magpie_tts_multilingual_357m

# Synthesize from file
make synth-file INPUT=sentences.txt MODEL_PATH=path/to/model.nemo

# List available speakers
make list-speakers
```

**Synthesis Options:**
- `TEXT` - Text to synthesize
- `MODEL_PATH` - Path to .nemo checkpoint
- `SPEAKER` - Speaker index (0-4)
- `LANGUAGE` - Language code (default: ms)

## Project Structure

```
train_magpietts_malay/
├── configs/
│   └── magpietts_malay.yaml    # Training configuration
├── data/
│   ├── raw/                    # Downloaded dataset
│   ├── audio/                  # Processed WAV files
│   └── manifests/              # Training manifests
├── experiments/                # Training outputs & checkpoints
├── output/                     # Synthesized audio
├── src/
│   ├── download_dataset.py     # Dataset downloader
│   ├── explore_dataset.py      # Dataset explorer
│   ├── malay_phonemizer.py     # Text-to-phoneme converter
│   ├── prepare_data.py         # Data preparation
│   ├── synthesize.py           # Inference script
│   └── train.py                # Training script
├── Makefile                    # Automation commands
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Checkpoints

Checkpoints are saved to `experiments/magpietts_malay/magpietts_malay_finetune/checkpoints/`:

| File | Description |
|------|-------------|
| `last.nemo` | Latest checkpoint (saved every epoch) |
| `magpietts_malay-{epoch}-{val_loss}.nemo` | Best checkpoints by validation loss |

The config keeps the top 3 best checkpoints plus the latest.

## Phoneme Format

Text is converted to IPA phonemes using espeak-ng:

```
Input:  "Selamat pagi, apa khabar?"
Output: "səlamat paɡi, apə xabar?"
```

For code-switching (Malay-English), the phonemizer detects English words and uses English phonemes:

```
Input:  "Okay, meeting kita postpone ke next week."
Output: "oʊkeɪ, miːɾɪŋ kitə postpone kə nɛkst wɛəʔ."
```

## Configuration

Edit `configs/magpietts_malay.yaml` to customize:

```yaml
model:
  batch_size: 8           # Reduce to 4 if OOM
  learning_rate: 2e-4     # Higher LR for new language

trainer:
  max_epochs: 100         # Reduce for faster experiments
  precision: '16-mixed'   # Use '32' if numerical issues

exp_manager:
  exp_dir: experiments/magpietts_malay
  checkpoint_callback_params:
    save_top_k: 3         # Keep top N checkpoints
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
model:
  batch_size: 4
```

### Sequence Length Error
The model has a max sequence length of 2048 tokens. Long texts are automatically filtered during training.

### espeak-ng Not Found
Install espeak-ng:
```bash
sudo apt install espeak-ng
```

### NeMo Import Errors
Make sure you're using NeMo from GitHub main:
```bash
make install-fresh
```

## Make Commands Reference

```bash
make help           # Show all commands
make install        # Install dependencies
make install-fresh  # Reinstall NeMo from scratch
make download       # Download full dataset
make download-small # Download small subset
make explore        # Show dataset statistics
make prepare        # Prepare full dataset
make prepare-small  # Prepare small subset
make train          # Start training
make train GPUS=4   # Train with multiple GPUs
make tensorboard    # Start TensorBoard
make synth TEXT="..." MODEL_PATH=...  # Synthesize speech
make clean          # Remove generated files
make clean-all      # Remove everything
```

## License

This project uses:
- NVIDIA NeMo: Apache 2.0 License
- MagpieTTS model: NVIDIA License
- Malaysian-TTS dataset: Check dataset license on HuggingFace
