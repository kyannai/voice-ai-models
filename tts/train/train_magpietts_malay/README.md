# MagpieTTS Malay Two-Phase Training

Fine-tune NVIDIA's MagpieTTS multilingual model to add Malay language support using a two-phase training pipeline.

## Overview

This project implements a two-phase approach to adding Malay as a new language:

- **Phase 1: Language Training** - Teaches the model Malay pronunciation (G2P) and prosody
- **Phase 2: Voice Cloning** - Fine-tunes with specific speaker voices

After training, the model can synthesize Malay speech directly from raw text - no external phonemizer needed at inference time.

**Model**: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)  
**Dataset**: [mesolitica/Malaysian-TTS](https://huggingface.co/datasets/mesolitica/Malaysian-TTS) (~500k samples, 5 speakers)

## Technical Implementation

### Fresh Malay Tokenizer with Custom IPA Vocabulary

MagpieTTS has hardcoded language support for 7 languages. To add Malay, we:

1. **Repurpose the Spanish slot** - Use the Spanish tokenizer's position in the model (indices 96-199)
2. **Create fresh Malay IPA vocabulary** - Extract 61 unique IPA symbols from our G2P dictionary (includes Malay-specific phonemes like `ə`, `ŋ` that Spanish doesn't have)
3. **Replace Spanish tokenizer entirely** - Create a new `IPATokenizer` with Malay vocabulary and G2P
4. **Reset token embeddings** - Reinitialize embeddings at indices 96-199 with Xavier initialization, removing all Spanish phoneme knowledge

This approach ensures:
- Malay learns from scratch without Spanish bias
- All Malay IPA phonemes are in the vocabulary (no "unknown phoneme" errors)
- Pretrained encoder/decoder weights are preserved for transfer learning

### Training Data Format

Training data uses `language='es'` to route through the Spanish slot (which now contains our Malay tokenizer):

```json
{"audio_filepath": "audio/sample.wav", "text": "Selamat pagi", "duration": 2.5, "speaker": 0, "language": "es"}
```

### Inference

Use `--language ms` for user-friendly interface (internally mapped to `es`):

```bash
python src/synthesize.py --text "Selamat pagi" --language ms --model-path models/malay_base.nemo
```

## Architecture

```
Phase 1: Language Training (Run Once)
┌──────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ nvidia/magpie_tts    │ ──▶ │ + Malay G2P     │ ──▶ │ Malay Base Model │
│ (pretrained)         │     │ + Fresh IPA     │     │ (malay_base.nemo)│
│                      │     │ + Reset Embeds  │     │                  │
└──────────────────────┘     └─────────────────┘     └──────────────────┘
                                                             │
                                                             ▼
Phase 2: Voice Cloning (Per Speaker)
┌──────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Malay Base Model     │ ──▶ │ + New Speaker   │ ──▶ │ Production Model │
│                      │     │   Voice Data    │     │                  │
└──────────────────────┘     └─────────────────┘     └──────────────────┘
                                                             │
                                                             ▼
Inference (No phonemizer needed!)
┌──────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ "Selamat pagi"       │ ──▶ │ Model (has G2P) │ ──▶ │ Audio            │
│ (raw Malay text)     │     │                 │     │                  │
└──────────────────────┘     └─────────────────┘     └──────────────────┘
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (A100 recommended)
- ~100GB disk space for full dataset
- espeak-ng (for Phase 1 G2P dictionary generation only)

```bash
# Install espeak-ng (required for Phase 1 only)
sudo apt install espeak-ng  # Ubuntu/Debian
brew install espeak         # macOS
```

## Quick Start

### Phase 1: Language Training (One-Time)

```bash
# 1. Install dependencies
make install

# 2. Prepare data and generate G2P dictionary
make phase1-prepare

# 3. Train Malay language model
make phase1-train

# 4. Export base model
make phase1-export
```

### Phase 2: Voice Cloning (Per Speaker)

```bash
# 1. Prepare speaker data
make phase2-prepare

# 2. Fine-tune with new voice
make phase2-train
```

### Synthesis (No Phonemizer Needed!)

```bash
# Synthesize raw Malay text
make synth TEXT="Selamat pagi, apa khabar?" MODEL_PATH=models/malay_base.nemo

# With specific speaker (0-4)
make synth TEXT="Terima kasih" MODEL_PATH=models/malay_base.nemo SPEAKER=2
```

## Detailed Guide

### Phase 1: Language Training

Phase 1 teaches the model to understand Malay as a new language. This involves:
1. Generating a G2P (Grapheme-to-Phoneme) dictionary from the training corpus
2. Adding a Malay tokenizer to the model
3. Training the model on Malay audio with raw text

**Data Preparation:**
```bash
make phase1-prepare
```

This will:
- Download the Malaysian-TTS dataset
- Process audio files (resample to 22.05kHz)
- Create manifest files with raw text (not phonemes)
- Generate G2P dictionary using espeak-ng

**Training:**
```bash
# Train with all available GPUs
make phase1-train

# Train with specific number of GPUs
make phase1-train GPUS=2
```

**Export Model:**
```bash
make phase1-export
# Output: models/malay_base.nemo
```

### Phase 2: Voice Cloning

Phase 2 fine-tunes the Malay-capable model with new speaker voices. The model already has Malay G2P built-in, so no phonemizer is needed.

**Prepare New Speaker Data:**
```bash
make phase2-prepare
```

**Fine-tune:**
```bash
make phase2-train
```

### Configuration

Config files are in `configs/`:
- `phase1_language.yaml` - Language training settings
- `phase2_voiceclone.yaml` - Voice cloning settings

Key parameters:
```yaml
# Phase 1
batch_size: 32
learning_rate: 2e-4
max_epochs: 100

# Phase 2
batch_size: 32
learning_rate: 1e-4  # Lower for fine-tuning
max_epochs: 30
```

### Available Speakers

The Malaysian-TTS dataset includes 5 speakers:

| ID | Name            | Gender | Style          |
|----|-----------------|--------|----------------|
| 0  | anwar_ibrahim   | Male   | Political      |
| 1  | husein          | Male   | Conversational |
| 2  | kp_ms           | Female | News           |
| 3  | shafiqah_idayu  | Female | Conversational |

## Project Structure

```
train_magpietts_malay/
├── configs/
│   ├── phase1_language.yaml    # Phase 1 config (language training)
│   └── phase2_voiceclone.yaml  # Phase 2 config (voice cloning)
├── src/
│   ├── download_dataset.py     # Dataset downloader
│   ├── prepare_data.py         # Data preparation + G2P generation
│   ├── train.py                # Training script
│   └── synthesize.py           # Inference script
├── data/
│   ├── raw/                    # Downloaded dataset
│   ├── audio/                  # Processed audio
│   ├── manifests/              # Training manifests (language='es')
│   └── g2p/                    # G2P dictionary (ipa_malay_dict.txt)
├── models/
│   └── malay_base.nemo         # Exported base model
├── experiments/                # Training checkpoints
├── Makefile                    # Build automation
└── README.md
```

## Make Commands

### Data Management (Step 1)
```bash
make install           # Install dependencies
make download          # Download full dataset
make download-small    # Download small subset
make explore           # Show dataset statistics
```

### Phase 1: Language Training (Step 2)
```bash
make phase1-prepare    # Prepare data + generate G2P dictionary
make phase1-train      # Train language model
make phase1-export     # Export to models/malay_base.nemo
```

### Phase 2: Voice Cloning (Step 3)
```bash
make phase2-prepare    # Prepare new speaker data
make phase2-train      # Fine-tune with new voice
```

### Synthesis
```bash
make synth TEXT="..."              # Synthesize text
make synth-file INPUT=file.txt     # Synthesize from file
make list-speakers                 # Show available speakers
```

### Utilities
```bash
make check-gpu         # Check GPU availability
make tensorboard       # Start TensorBoard
make list-checkpoints  # List training checkpoints
make convert           # Convert checkpoint to .pt
make clean             # Remove generated files
```

## Troubleshooting

### Phase 1: G2P dictionary not found
```bash
# Make sure to run prepare first
make phase1-prepare
```

### Phase 2: Base model not found
```bash
# Export the Phase 1 model first
make phase1-export
```

### CUDA out of memory
Reduce batch size in the config file:
```yaml
model:
  batch_size: 16  # Reduce from 32
```

### espeak-ng not found
```bash
sudo apt install espeak-ng  # Ubuntu/Debian
brew install espeak         # macOS
```

## License

This project uses:
- NVIDIA MagpieTTS (Apache 2.0)
- mesolitica/Malaysian-TTS dataset
- NeMo Framework (Apache 2.0)
