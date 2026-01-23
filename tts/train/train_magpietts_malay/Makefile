# MagpieTTS Malay Two-Phase Training Pipeline
# ============================================
# Phase 1: Language Training - Teaches model Malay G2P (run once)
# Phase 2: Voice Cloning - Fine-tunes with new speaker voices
#
# Quick Start:
#   make download         - Download Malaysian-TTS dataset
#   make phase1-prepare   - Prepare data + generate G2P dictionary
#   make phase1-train     - Train Malay language model
#   make phase1-export    - Export base model
#   make synth TEXT="..." - Synthesize (no phonemizer needed!)

# ============================================================
# Configuration
# ============================================================

DATASET_NAME := mesolitica/Malaysian-TTS
MAX_SAMPLES_SMALL := 10000
TRAIN_SPLIT := 0.95
TARGET_SAMPLE_RATE := 22050

# Python command (override with: make ... PYTHON="uv run python")
PYTHON ?= python

# Directories
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
AUDIO_DIR := $(DATA_DIR)/audio
MANIFEST_DIR := $(DATA_DIR)/manifests
SMALL_AUDIO_DIR := $(DATA_DIR)/audio_small
SMALL_MANIFEST_DIR := $(DATA_DIR)/manifests_small
G2P_DIR := $(DATA_DIR)/g2p
SRC_DIR := src
MODELS_DIR := models
OUTPUT_DIR := output
EXPERIMENT_DIR := experiments

# Training config
G2P_DICT := $(G2P_DIR)/ipa_malay_dict.txt
PHASE1_CONFIG := configs/phase1_language.yaml
PHASE1_SMALL_CONFIG := configs/phase1_language_small.yaml
PHASE2_CONFIG := configs/phase2_voiceclone.yaml
MALAY_BASE_MODEL := $(MODELS_DIR)/malay_base.nemo
GPUS ?= -1  # -1 means use all available GPUs
MAX_SAMPLES ?= 1000
SPEAKERS ?=

# Synthesis config
MODEL_PATH ?= nvidia/magpie_tts_multilingual_357m
LANGUAGE := ms
SPEAKER := 0
CODE_SWITCH ?= 0

.PHONY: help install download download-small explore \
        phase1-prepare phase1-prepare-small phase1-train phase1-train-small phase1-export \
        phase2-prepare phase2-train \
        synth synthesize synth-file list-speakers synth-test synth-demo synth-train-samples \
        convert convert-latest convert-best list-checkpoints phase1-convert \
        check-gpu tensorboard clean clean-all clean-output

# ============================================================
# Help
# ============================================================

help:
	@echo "MagpieTTS Malay Two-Phase Training Pipeline"
	@echo "============================================"
	@echo ""
	@echo "=== Data Management (Step 1) ==="
	@echo "  make install         - Install dependencies"
	@echo "  make download        - Download full dataset from HuggingFace"
	@echo "  make download-small  - Download small subset for testing"
	@echo "  make explore         - Explore dataset and print statistics"
	@echo ""
	@echo "=== Phase 1: Language Training (Step 2, Run Once) ==="
	@echo "  make phase1-prepare  - Prepare data + generate G2P dictionary"
	@echo "  make phase1-prepare-small MAX_SAMPLES=1000 - Prepare small subset"
	@echo "  make phase1-train    - Train model to learn Malay language"
	@echo "  make phase1-train-small MAX_SAMPLES=1000 - Overfit on small subset"
	@echo "  make phase1-export   - Export Malay base model to models/"
	@echo ""
	@echo "=== Phase 2: Voice Cloning (Step 3, Per Speaker) ==="
	@echo "  make phase2-prepare  - Prepare new speaker data"
	@echo "  make phase2-train    - Fine-tune with new speaker voice"
	@echo ""
	@echo "=== Synthesis (No phonemizer needed!) ==="
	@echo "  make synth TEXT=\"...\"           - Synthesize raw Malay text"
	@echo "  make synth-demo                  - Convert latest + run demo synth"
	@echo "  make synth-train-samples         - Convert latest + synth from small manifest"
	@echo "  make synth-file INPUT=file.txt  - Synthesize from text file"
	@echo "  make list-speakers              - Show available speakers"
	@echo ""
	@echo "  Options:"
	@echo "    MODEL_PATH=path/to/model.nemo - Use custom model"
	@echo "    SPEAKER=0                     - Speaker index (0-4)"
	@echo "    LANGUAGE=ms                   - Language code"
	@echo ""
	@echo "=== Utilities ==="
	@echo "  make check-gpu                    - Check GPU availability"
	@echo "  make tensorboard                  - Start TensorBoard"
	@echo "  make list-checkpoints             - List available checkpoints"
	@echo "  make convert CKPT=<name>.ckpt     - Convert specific checkpoint"
	@echo "  make convert-latest               - Convert latest checkpoint"
	@echo "  make convert-best                 - Convert best (lowest val_loss)"
	@echo ""
	@echo "=== Cleanup ==="
	@echo "  make clean            - Remove generated files (audio, manifests)"
	@echo "  make clean-all        - Remove everything including downloaded data"
	@echo "  make clean-output     - Remove synthesized audio files"
	@echo ""
	@echo "Configuration:"
	@echo "  DATASET_NAME       = $(DATASET_NAME)"
	@echo "  MAX_SAMPLES_SMALL  = $(MAX_SAMPLES_SMALL)"
	@echo "  TARGET_SAMPLE_RATE = $(TARGET_SAMPLE_RATE) Hz"
	@echo "  GPUS               = $(GPUS)"

# ============================================================
# Installation
# ============================================================

install:
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo "Note: Also install espeak-ng: sudo apt install espeak-ng"

# ============================================================
# Data Management (Step 1)
# ============================================================

download:
	@echo "Downloading full dataset from HuggingFace..."
	python $(SRC_DIR)/download_dataset.py \
		--dataset $(DATASET_NAME) \
		--output-dir $(RAW_DIR) \
		$(if $(SPEAKERS),--speakers $(SPEAKERS),)
	@echo "Download complete!"

download-small:
	@echo "Downloading small subset ($(MAX_SAMPLES_SMALL) samples)..."
	python $(SRC_DIR)/download_dataset.py \
		--dataset $(DATASET_NAME) \
		--output-dir $(RAW_DIR) \
		--max-samples $(MAX_SAMPLES_SMALL) \
		$(if $(SPEAKERS),--speakers $(SPEAKERS),)
	@echo "Download complete!"

explore:
	@echo "Exploring dataset..."
	python $(SRC_DIR)/explore_dataset.py \
		--data-dir $(RAW_DIR)

# ============================================================
# Phase 1: Language Training (Step 2, Run Once)
# Creates a Malay-capable base model with built-in G2P
# Uses Spanish slot with fresh Malay IPA vocabulary
# ============================================================

phase1-prepare: install download
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Preparing data with G2P dictionary"
	@echo "========================================"
	@mkdir -p $(G2P_DIR)
	@mkdir -p $(MODELS_DIR)
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT) \
		--generate-g2p \
		--g2p-output $(G2P_DICT)
	@echo ""
	@echo "Phase 1 data preparation complete!"
	@echo "  Manifests: $(MANIFEST_DIR)/"
	@echo "  G2P dictionary: $(G2P_DICT)"

phase1-prepare-small: install
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Preparing SMALL data subset"
	@echo "========================================"
	@if [ ! -f "$(G2P_DICT)" ]; then \
		echo "Error: G2P dictionary not found at $(G2P_DICT)"; \
		echo "Run 'make phase1-prepare' first"; \
		exit 1; \
	fi
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(SMALL_AUDIO_DIR) \
		--manifest-output-dir $(SMALL_MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT) \
		--max-samples $(MAX_SAMPLES) \
		$(if $(SPEAKERS),--speakers $(SPEAKERS),)
	@echo ""
	@echo "Phase 1 small data preparation preparation complete!"
	@echo "  Manifests: $(SMALL_MANIFEST_DIR)/"

phase1-train:
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Training Malay Language Model"
	@echo "========================================"
	@echo "  - Fresh Malay IPA tokenizer (61 symbols)"
	@echo "  - Reset Spanish embeddings (indices 96-199)"
	@echo "  - Training data uses language='es'"
	@echo "========================================"
	@if [ ! -f "$(G2P_DICT)" ]; then \
		echo "Error: G2P dictionary not found at $(G2P_DICT)"; \
		echo "Run 'make phase1-prepare' first"; \
		exit 1; \
	fi
	python $(SRC_DIR)/train.py \
		--config $(PHASE1_CONFIG) \
		--gpus $(GPUS)

phase1-train-small: phase1-prepare-small
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Overfit SMALL dataset"
	@echo "========================================"
	@echo "  Max samples: $(MAX_SAMPLES)"
	@echo "  Goal: verify wiring by overfitting"
	@echo "========================================"
	python $(SRC_DIR)/train.py \
		--config $(PHASE1_SMALL_CONFIG) \
		--gpus $(GPUS)

phase1-export:
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Exporting Malay Base Model"
	@echo "========================================"
	@mkdir -p $(MODELS_DIR)
	@LATEST=$$(ls -t experiments/phase1_language/malay_language_training/checkpoints/*.nemo 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "Error: No .nemo checkpoints found in experiments/phase1_language/"; \
		echo "Run 'make phase1-train' first"; \
		exit 1; \
	fi; \
	echo "Copying $$LATEST to $(MALAY_BASE_MODEL)"; \
	cp "$$LATEST" "$(MALAY_BASE_MODEL)"; \
	echo ""; \
	echo "Malay base model exported to: $(MALAY_BASE_MODEL)"; \
	echo "This model can now be used for Phase 2 voice cloning."

# ============================================================
# Phase 2: Voice Cloning (Step 3, Per Speaker)
# Fine-tunes the Malay model with new speaker voices
# ============================================================

phase2-prepare:
	@echo ""
	@echo "========================================"
	@echo "Phase 2: Preparing Voice Cloning Data"
	@echo "========================================"
	@if [ ! -f "$(MALAY_BASE_MODEL)" ]; then \
		echo "Warning: Malay base model not found at $(MALAY_BASE_MODEL)"; \
		echo "Make sure to run Phase 1 first, or update phase2_voiceclone.yaml"; \
	fi
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR) \
		--audio-output-dir $(AUDIO_DIR) \
		--manifest-output-dir $(MANIFEST_DIR) \
		--target-sample-rate $(TARGET_SAMPLE_RATE) \
		--train-split $(TRAIN_SPLIT)
	@echo ""
	@echo "Phase 2 data preparation complete!"

phase2-train:
	@echo ""
	@echo "========================================"
	@echo "Phase 2: Voice Cloning Training"
	@echo "========================================"
	@if [ ! -f "$(MALAY_BASE_MODEL)" ]; then \
		echo "Error: Malay base model not found at $(MALAY_BASE_MODEL)"; \
		echo "Run 'make phase1-train' and 'make phase1-export' first"; \
		exit 1; \
	fi
	python $(SRC_DIR)/train.py \
		--config $(PHASE2_CONFIG) \
		--gpus $(GPUS)

# ============================================================
# Synthesis (No phonemizer needed!)
# ============================================================

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
		--speaker $(SPEAKER)

synthesize:
ifndef TEXT
	@echo "Error: TEXT is required. Usage: make synthesize TEXT=\"Your text here\" MODEL_PATH=path/to/model.pt LANGUAGE=ms CODE_SWITCH=1"
	@exit 1
endif
	@echo "Synthesizing: $(TEXT)"
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODEL_PATH) \
		--text "$(TEXT)" \
		--output-dir $(OUTPUT_DIR) \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(filter 1 true yes,$(CODE_SWITCH)),--code-switch,)

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
		--speaker $(SPEAKER)

list-speakers:
	@python $(SRC_DIR)/synthesize.py --list-speakers

synth-test:
	@echo "Testing synthesis with pretrained model..."
	python $(SRC_DIR)/synthesize.py \
		--model-path nvidia/magpie_tts_multilingual_357m \
		--text "Selamat pagi, apa khabar hari ini?" \
		--output-dir $(OUTPUT_DIR) \
		--language ms \
		--speaker 0
	@echo "Test audio saved to $(OUTPUT_DIR)/"

synth-demo:
	@echo "Converting latest checkpoint to .pt..."
	@$(MAKE) convert-latest
	@echo "Running demo synthesis..."
	@$(MAKE) synthesize \
		TEXT='["selamat pagi tuan tuan dan puan puan","Ini boleh dilakukan dengan mewujudkan proses pelantikan yang telus dan berasaskan merit, bebas daripada campur tangan politik.","Okay, so basically kita kena siapkan report ni sebelum Jumaat, kalau tak boss akan marah."]' \
		MODEL_PATH="$(MODELS_DIR)/malay_base.pt" \
		LANGUAGE=ms \
		CODE_SWITCH=1 \
		NAME=demo

synth-train-samples:
	@echo "Converting latest SMALL checkpoint to .pt..."
	@$(MAKE) convert-latest-small
	@echo "Synthesizing from training manifest..."
	python $(SRC_DIR)/synthesize.py \
		--model-path $(MODELS_DIR)/malay_base_small.pt \
		--manifest $(SMALL_MANIFEST_DIR)/train_manifest.json \
		--manifest-limit 3 \
		--output-dir $(OUTPUT_DIR) \
		--language ms \
		--speaker $(SPEAKER) \
		--name train_sample \
		$(if $(filter 1 true yes,$(CODE_SWITCH)),--code-switch,)

# ============================================================
# Utilities
# ============================================================

check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir $(EXPERIMENT_DIR) --port 6006

# ============================================================
# Checkpoint Conversion
# ============================================================

PHASE1_CHECKPOINT_DIR := experiments/phase1_language/malay_language_training/checkpoints
PHASE1_SMALL_CHECKPOINT_DIR := experiments/phase1_language_small/malay_language_overfit/checkpoints

list-checkpoints:
	@echo "Phase 1 checkpoints:"
	@ls -lh $(PHASE1_CHECKPOINT_DIR)/*.nemo 2>/dev/null || echo "  No Phase 1 checkpoints found"
	@echo ""
	@ls -lh $(PHASE1_CHECKPOINT_DIR)/*.ckpt 2>/dev/null || echo "  No .ckpt files found"

convert:
ifndef CKPT
	@echo "Usage: make convert CKPT=<checkpoint_name>"
	@echo ""
	@echo "Available checkpoints:"
	@ls -1 $(PHASE1_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | xargs -n1 basename | grep -v last || echo "  No checkpoints found"
	@echo ""
	@echo "Example:"
	@echo "  make convert CKPT=malay_lang-epoch=00-val_loss=7.3366.ckpt"
else
	@CKPT_PATH="$(PHASE1_CHECKPOINT_DIR)/$(CKPT)"; \
	echo "Converting checkpoint: $$CKPT_PATH"; \
	if [ ! -f "$$CKPT_PATH" ]; then \
		echo "Error: Checkpoint not found: $$CKPT_PATH"; \
		exit 1; \
	fi; \
	$(PYTHON) src/convert_checkpoint.py "$$CKPT_PATH" -o "$(MODELS_DIR)/malay_base.pt"
endif

convert-latest:
	@echo "Converting latest checkpoint to .pt state dict..."
	@LATEST=$$(ls -t $(PHASE1_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | grep -v last | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "Error: No checkpoints found in $(PHASE1_CHECKPOINT_DIR)"; \
		exit 1; \
	fi; \
	echo "Found: $$LATEST"; \
	$(PYTHON) src/convert_checkpoint.py "$$LATEST" -o "$(MODELS_DIR)/malay_base.pt"

convert-latest-small:
	@echo "Converting latest SMALL checkpoint to .pt state dict..."
	@LATEST=$$(ls -t $(PHASE1_SMALL_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | grep -v last | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "Error: No checkpoints found in $(PHASE1_SMALL_CHECKPOINT_DIR)"; \
		exit 1; \
	fi; \
	echo "Found: $$LATEST"; \
	$(PYTHON) src/convert_checkpoint.py "$$LATEST" -o "$(MODELS_DIR)/malay_base_small.pt"

convert-best:
	@echo "Finding best checkpoint by val_loss..."
	@BEST=$$(ls $(PHASE1_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | grep -v last | sort -t= -k3 -n | head -1); \
	if [ -z "$$BEST" ]; then \
		echo "Error: No checkpoints found"; \
		exit 1; \
	fi; \
	echo "Best: $$BEST"; \
	$(PYTHON) src/convert_checkpoint.py "$$BEST" -o "$(MODELS_DIR)/malay_best.pt"

convert-best-small:
	@echo "Finding best SMALL checkpoint by val_loss..."
	@BEST=$$(ls $(PHASE1_SMALL_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | grep -v last | sort -t= -k3 -n | head -1); \
	if [ -z "$$BEST" ]; then \
		echo "Error: No checkpoints found in $(PHASE1_SMALL_CHECKPOINT_DIR)"; \
		exit 1; \
	fi; \
	echo "Best: $$BEST"; \
	$(PYTHON) src/convert_checkpoint.py "$$BEST" -o "$(MODELS_DIR)/malay_best_small.pt"

phase1-convert:
	@echo "Converting latest Phase 1 checkpoint..."
	@LATEST=$$(ls -t $(PHASE1_CHECKPOINT_DIR)/*.ckpt 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "Error: No Phase 1 checkpoints found in $(PHASE1_CHECKPOINT_DIR)"; \
		exit 1; \
	fi; \
	echo "Found: $$LATEST"; \
	mkdir -p $(MODELS_DIR); \
	$(PYTHON) src/convert_checkpoint.py "$$LATEST" -o "$(MODELS_DIR)/malay_base.pt"

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
