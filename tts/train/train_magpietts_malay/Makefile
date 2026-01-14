# MagpieTTS Malay Training Data Preparation
# ==========================================
# Prepares mesolitica/Malaysian-TTS dataset for MagpieTTS fine-tuning
#
# Usage:
#   make install        - Install dependencies
#   make download-small - Download small subset (10k samples)
#   make all-small      - Full pipeline with small subset
#   make all            - Full pipeline with complete dataset
#   make help           - Show all commands

# Configuration
DATASET_NAME := mesolitica/Malaysian-TTS
MAX_SAMPLES_SMALL := 10000
TRAIN_SPLIT := 0.95
TARGET_SAMPLE_RATE := 22050

# Directories
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
AUDIO_DIR := $(DATA_DIR)/audio
MANIFEST_DIR := $(DATA_DIR)/manifests
SRC_DIR := src

.PHONY: help install install-fresh download download-small explore prepare prepare-small \
        all all-small clean clean-all

help:
	@echo "MagpieTTS Malay Training Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Quick Start (recommended):"
	@echo "  make install       - Install Python dependencies"
	@echo "  make all-small     - Prepare data with small subset (10k samples)"
	@echo "  make train         - Start training with GPU"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make download      - Download full dataset from HuggingFace"
	@echo "  make download-small- Download small subset for testing"
	@echo "  make explore       - Explore dataset and print statistics"
	@echo "  make prepare       - Prepare full dataset (convert audio, create manifests)"
	@echo "  make prepare-small - Prepare small subset for testing"
	@echo ""
	@echo "Training (GPU/CUDA):"
	@echo "  make check-gpu     - Check GPU availability"
	@echo "  make train         - Train with 1 GPU (default)"
	@echo "  make train GPUS=4  - Train with 4 GPUs"
	@echo "  make train-multi   - Train with all available GPUs"
	@echo "  make train-resume CHECKPOINT=path/to/ckpt - Resume training"
	@echo "  make tensorboard   - Start TensorBoard for monitoring"
	@echo ""
	@echo "Synthesis / Inference:"
	@echo "  make synth TEXT=\"...\"           - Synthesize text to audio"
	@echo "  make synth-file INPUT=file.txt  - Synthesize from text file"
	@echo "  make synth-test                 - Test with pretrained model"
	@echo "  make list-speakers              - Show available speakers"
	@echo ""
	@echo "  Options (append to synth commands):"
	@echo "    MODEL_PATH=path/to/model.nemo - Use custom model"
	@echo "    SPEAKER=0                     - Speaker index (0-4)"
	@echo "    LANGUAGE=ms                   - Language code"
	@echo ""
	@echo "Full Pipelines:"
	@echo "  make all-small     - Data prep with small subset"
	@echo "  make all           - Data prep with complete dataset"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove generated files (audio, manifests)"
	@echo "  make clean-all     - Remove everything including downloaded data"
	@echo "  make clean-output  - Remove synthesized audio files"
	@echo ""
	@echo "Configuration:"
	@echo "  DATASET_NAME       = $(DATASET_NAME)"
	@echo "  MAX_SAMPLES_SMALL  = $(MAX_SAMPLES_SMALL)"
	@echo "  TARGET_SAMPLE_RATE = $(TARGET_SAMPLE_RATE) Hz"
	@echo "  CONFIG_FILE        = $(CONFIG_FILE)"
	@echo "  GPUS               = $(GPUS)"

install:
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo "Note: Also install espeak-ng: sudo apt install espeak-ng"

install-fresh:
	@echo "Removing old NeMo and reinstalling from GitHub main..."
	-pip uninstall nemo_toolkit -y 2>/dev/null || true
	uv pip install -r requirements.txt --reinstall
	@echo ""
	@echo "Fresh install complete!"
	@echo "Note: Also install espeak-ng: sudo apt install espeak-ng"

# Download commands
download:
	@echo "Downloading full dataset from HuggingFace..."
	python $(SRC_DIR)/download_dataset.py \
		--dataset $(DATASET_NAME) \
		--output-dir $(RAW_DIR)
	@echo "Download complete!"

download-small:
	@echo "Downloading small subset ($(MAX_SAMPLES_SMALL) samples)..."
	python $(SRC_DIR)/download_dataset.py \
		--dataset $(DATASET_NAME) \
		--output-dir $(RAW_DIR) \
		--max-samples $(MAX_SAMPLES_SMALL)
	@echo "Download complete!"

# Explore dataset
explore:
	@echo "Exploring dataset..."
	python $(SRC_DIR)/explore_dataset.py \
		--data-dir $(RAW_DIR)

# Test download (metadata only, no audio - for testing)
download-test:
	@echo "Downloading metadata only (for testing)..."
	python $(SRC_DIR)/download_dataset.py \
		--output-dir $(RAW_DIR) \
		--max-samples 1000 \
		--skip-audio
	@echo "Download complete (metadata only)!"

# Prepare data commands
# Default: use phonemes for better quality
prepare:
	@echo "Preparing full dataset with phonemes..."
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT) \
		--use-phonemes
	@echo "Data preparation complete!"
	@echo "Manifests saved to $(MANIFEST_DIR)/"

prepare-small:
	@echo "Preparing small subset with phonemes..."
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT) \
		--max-samples $(MAX_SAMPLES_SMALL) \
		--use-phonemes
	@echo "Data preparation complete!"
	@echo "Manifests saved to $(MANIFEST_DIR)/"

# Character-level versions (without phonemization)
prepare-chars:
	@echo "Preparing full dataset (character-level, no phonemes)..."
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT)
	@echo "Data preparation complete!"

prepare-small-chars:
	@echo "Preparing small subset (character-level, no phonemes)..."
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT) \
		--max-samples $(MAX_SAMPLES_SMALL)
	@echo "Data preparation complete!"

# Test phonemizer
test-phonemizer:
	@echo "Testing Malay phonemizer..."
	python $(SRC_DIR)/malay_phonemizer.py

# Full pipelines
all-small: install download-small explore prepare-small
	@echo ""
	@echo "========================================"
	@echo "Pipeline complete (small subset)!"
	@echo "========================================"
	@echo "Train manifest: $(MANIFEST_DIR)/train_manifest.json"
	@echo "Val manifest:   $(MANIFEST_DIR)/val_manifest.json"
	@echo "Audio files:    $(AUDIO_DIR)/"

all: install download explore prepare
	@echo ""
	@echo "========================================"
	@echo "Pipeline complete (full dataset)!"
	@echo "========================================"
	@echo "Train manifest: $(MANIFEST_DIR)/train_manifest.json"
	@echo "Val manifest:   $(MANIFEST_DIR)/val_manifest.json"
	@echo "Audio files:    $(AUDIO_DIR)/"

# ============================================================
# Training (GPU/CUDA)
# ============================================================

# Training configuration
CONFIG_FILE := configs/magpietts_malay.yaml
GPUS ?= -1  # -1 means use all available GPUs
EXPERIMENT_DIR := experiments

# Train the model with GPU
train:
	@echo "Starting MagpieTTS fine-tuning with $(GPUS) GPU(s)..."
	@echo "Config: $(CONFIG_FILE)"
	python $(SRC_DIR)/train.py \
		--config $(CONFIG_FILE) \
		--gpus $(GPUS)

# Train with multiple GPUs
train-multi:
ifndef GPUS
	@echo "Using all available GPUs..."
	python $(SRC_DIR)/train.py \
		--config $(CONFIG_FILE) \
		--gpus -1
else
	@echo "Training with $(GPUS) GPUs..."
	python $(SRC_DIR)/train.py \
		--config $(CONFIG_FILE) \
		--gpus $(GPUS)
endif

# Resume training from checkpoint
train-resume:
ifndef CHECKPOINT
	@echo "Error: CHECKPOINT is required. Usage: make train-resume CHECKPOINT=path/to/checkpoint.ckpt"
	@exit 1
endif
	@echo "Resuming training from: $(CHECKPOINT)"
	python $(SRC_DIR)/train.py \
		--config $(CONFIG_FILE) \
		--gpus $(GPUS) \
		--resume $(CHECKPOINT)

# Check GPU availability
check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# Monitor training with TensorBoard
tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir $(EXPERIMENT_DIR) --port 6006

# ============================================================
# Inference / Synthesis
# ============================================================

# Model path - set this to your trained model or use pretrained
MODEL_PATH ?= nvidia/magpie_tts_multilingual_357m
OUTPUT_DIR := output
LANGUAGE := ms
SPEAKER := 0

# Synthesize from command line text (with phonemes for trained model)
# Usage: make synth TEXT="Selamat pagi"
synth:
ifndef TEXT
	@echo "Error: TEXT is required. Usage: make synth TEXT=\"Your text here\""
	@exit 1
endif
	@echo "Synthesizing: $(TEXT)"
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODEL_PATH) \
		--text "$(TEXT)" \
		--output-dir $(OUTPUT_DIR) \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		--use-phonemes

# Synthesize without phonemes (for pretrained model or char-level trained)
synth-chars:
ifndef TEXT
	@echo "Error: TEXT is required. Usage: make synth-chars TEXT=\"Your text here\""
	@exit 1
endif
	@echo "Synthesizing (character-level): $(TEXT)"
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODEL_PATH) \
		--text "$(TEXT)" \
		--output-dir $(OUTPUT_DIR) \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER)

# Synthesize from a text file (one sentence per line)
# Usage: make synth-file INPUT=sentences.txt
synth-file:
ifndef INPUT
	@echo "Error: INPUT is required. Usage: make synth-file INPUT=sentences.txt"
	@exit 1
endif
	@echo "Synthesizing from file: $(INPUT)"
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODEL_PATH) \
		--input-file $(INPUT) \
		--output-dir $(OUTPUT_DIR) \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		--use-phonemes

# Synthesize with custom model path
# Usage: make synth-custom MODEL_PATH=checkpoints/best.nemo TEXT="Hello"
synth-custom:
ifndef TEXT
	@echo "Error: TEXT is required."
	@exit 1
endif
ifndef MODEL_PATH
	@echo "Error: MODEL_PATH is required."
	@exit 1
endif
	@echo "Synthesizing with model: $(MODEL_PATH)"
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODEL_PATH) \
		--text "$(TEXT)" \
		--output-dir $(OUTPUT_DIR) \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER)

# List available speakers
list-speakers:
	@python $(SRC_DIR)/synthesize.py --list-speakers

# Test synthesis with pretrained model
synth-test:
	@echo "Testing synthesis with pretrained model..."
	python $(SRC_DIR)/synthesize.py \
		--model-path nvidia/magpie_tts_multilingual_357m \
		--text "Selamat pagi, apa khabar hari ini?" \
		--output-dir $(OUTPUT_DIR) \
		--language ms \
		--speaker 0
	@echo "Test audio saved to $(OUTPUT_DIR)/"

# ============================================================
# Cleanup
# ============================================================

clean:
	@echo "Removing generated files..."
	rm -rf $(AUDIO_DIR)/*
	rm -rf $(MANIFEST_DIR)/*
	@echo "Cleaned generated files."

clean-all: clean
	@echo "Removing downloaded data..."
	rm -rf $(RAW_DIR)/*
	@echo "Cleaned all data."

clean-output:
	@echo "Removing synthesized audio..."
	rm -rf $(OUTPUT_DIR)/*
	@echo "Cleaned output files."
