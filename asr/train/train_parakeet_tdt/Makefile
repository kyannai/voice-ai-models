# Parakeet TDT ASR Training
# ==========================
# Fine-tune NVIDIA Parakeet TDT 0.6B v3 for Malay ASR
#
# Quick Start:
#   make install        - Install dependencies
#   make download       - Download Malaysian-STT dataset
#   make unzip          - Extract zip files
#   make prepare        - Prepare training data
#   make train          - Train with stage1 config
#   make tensorboard    - Monitor training

# ============================================================
# Configuration
# ============================================================

MODEL_NAME := nvidia/parakeet-tdt-0.6b-v3
DATASET_NAME := mesolitica/Malaysian-STT-Whisper
TRAIN_SPLIT := 0.95

# Config files
STAGE1_CONFIG := configs/parakeet_stage1.yaml
STAGE2_CONFIG := configs/parakeet_stage2.yaml

# Directories
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
MANIFEST_DIR := $(DATA_DIR)/manifests
MODELS_DIR := models
OUTPUTS_DIR := outputs
SRC_DIR := src

# CUDA settings for memory optimization
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: help install install-8bit check-gpu \
        download unzip prepare prepare-small \
        train train-stage1 train-stage2 train-custom train-resume \
        tensorboard check-events upload \
        clean clean-outputs clean-data clean-all

# ============================================================
# Help
# ============================================================

help:
	@echo "Parakeet TDT ASR Training Pipeline"
	@echo "==================================="
	@echo ""
	@echo "=== Installation (Step 1) ==="
	@echo "  make install       - Install Python dependencies"
	@echo "  make install-8bit  - Install 8-bit optimizer (recommended)"
	@echo "  make check-gpu     - Check GPU availability"
	@echo ""
	@echo "=== Data Management (Step 2) ==="
	@echo "  make download      - Download Malaysian-STT dataset from HuggingFace"
	@echo "  make unzip         - Extract downloaded zip files"
	@echo "  make prepare       - Prepare full dataset (create NeMo manifests)"
	@echo "  make prepare-small - Prepare small subset (10k samples) for testing"
	@echo ""
	@echo "=== Training (Step 3) ==="
	@echo "  make train         - Train with stage1 config (default)"
	@echo "  make train-stage1  - Train on Malaysian-STT dataset (stage 1)"
	@echo "  make train-stage2  - Continued training (stage 2)"
	@echo "  make train-custom CONFIG=path/to/config.yaml"
	@echo "  make train-resume CONFIG=... CHECKPOINT=..."
	@echo ""
	@echo "=== Monitoring ==="
	@echo "  make tensorboard   - Start TensorBoard (http://localhost:6006)"
	@echo "  make check-events  - Check TensorBoard event files"
	@echo ""
	@echo "=== Model Upload ==="
	@echo "  make upload MODEL=path/to/model.nemo REPO=repo-name"
	@echo ""
	@echo "=== Cleanup ==="
	@echo "  make clean-outputs - Remove training outputs"
	@echo "  make clean-data    - Remove downloaded data"
	@echo "  make clean-all     - Remove all generated files"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL_NAME   = $(MODEL_NAME)"
	@echo "  DATASET_NAME = $(DATASET_NAME)"
	@echo "  STAGE1_CONFIG = $(STAGE1_CONFIG)"
	@echo "  STAGE2_CONFIG = $(STAGE2_CONFIG)"

# ============================================================
# Installation (Step 1)
# ============================================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo ""
	@echo "Note: For 8-bit optimizer support (recommended for large datasets):"
	@echo "  make install-8bit"

install-8bit:
	@echo "Installing bitsandbytes for 8-bit AdamW optimizer..."
	pip install bitsandbytes
	@echo ""
	@echo "8-bit optimizer installed!"
	@echo ""
	@echo "Memory optimizations enabled:"
	@echo "  - 8-bit AdamW optimizer (75% memory reduction)"
	@echo "  - Gradient checkpointing (set in config)"
	@echo "  - CUDA memory fragmentation prevention"

check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'GPU count: {torch.cuda.device_count()}'); \
		[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)') \
			for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# ============================================================
# Data Management (Step 2)
# ============================================================

download:
	@echo "Downloading Malaysian-STT dataset from HuggingFace..."
	@echo "Dataset: $(DATASET_NAME)"
	@echo ""
	@mkdir -p $(RAW_DIR)
	huggingface-cli download --repo-type dataset \
		--local-dir '$(RAW_DIR)' \
		--max-workers 20 \
		$(DATASET_NAME)
	@echo ""
	@echo "Download complete! Files saved to $(RAW_DIR)"
	@echo "Next step: make unzip"

unzip:
	@echo "Unzipping all .zip files in data/raw..."
	cd $(RAW_DIR) && python ../../$(SRC_DIR)/unzip_data.py
	@echo ""
	@echo "Unzip complete!"
	@echo "Next step: make prepare"

prepare:
	@echo "Preparing full dataset..."
	@mkdir -p $(MANIFEST_DIR)
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR)/data \
		--audio-base-dir $(RAW_DIR) \
		--output-dir $(MANIFEST_DIR) \
		--train-split $(TRAIN_SPLIT) \
		--datasets malaysian_context_v2 extra
	@echo ""
	@echo "Data preparation complete!"
	@echo "Train manifest: $(MANIFEST_DIR)/train_manifest.json"
	@echo "Val manifest: $(MANIFEST_DIR)/val_manifest.json"
	@echo ""
	@echo "Next step: make train"

prepare-small:
	@echo "Preparing small subset (10k samples) for testing..."
	@mkdir -p $(MANIFEST_DIR)
	python $(SRC_DIR)/prepare_data.py \
		--data-dir $(RAW_DIR)/data \
		--audio-base-dir $(RAW_DIR) \
		--output-dir $(MANIFEST_DIR) \
		--train-split $(TRAIN_SPLIT) \
		--datasets malaysian_context_v2 extra \
		--max-samples 10000 \
		--validate-audio
	@echo ""
	@echo "Small subset preparation complete!"

# ============================================================
# Training (Step 3)
# ============================================================

train: train-stage1

train-stage1:
	@echo ""
	@echo "========================================"
	@echo "Parakeet TDT Training - Stage 1"
	@echo "========================================"
	@echo "Config: $(STAGE1_CONFIG)"
	@echo ""
	@echo "TIP: Monitor training in another terminal:"
	@echo "  make tensorboard"
	@echo "  Open: http://localhost:6006"
	@echo "========================================"
	@echo ""
	python $(SRC_DIR)/train.py --config $(STAGE1_CONFIG)

train-stage2:
	@echo ""
	@echo "========================================"
	@echo "Parakeet TDT Training - Stage 2"
	@echo "========================================"
	@echo "Config: $(STAGE2_CONFIG)"
	@echo "========================================"
	@echo ""
	python $(SRC_DIR)/train.py --config $(STAGE2_CONFIG)

train-custom:
ifndef CONFIG
	@echo "Error: CONFIG is required."
	@echo "Usage: make train-custom CONFIG=path/to/config.yaml"
	@exit 1
endif
	@echo "Starting training with custom config..."
	@echo "Config: $(CONFIG)"
	python $(SRC_DIR)/train.py --config $(CONFIG)

train-resume:
ifndef CHECKPOINT
	@echo "Error: CHECKPOINT is required."
	@echo "Usage: make train-resume CONFIG=configs/parakeet_stage1.yaml CHECKPOINT=path/to/checkpoint.ckpt"
	@exit 1
endif
ifndef CONFIG
	$(eval CONFIG := $(STAGE1_CONFIG))
endif
	@echo "Resuming training from checkpoint..."
	@echo "Config: $(CONFIG)"
	@echo "Checkpoint: $(CHECKPOINT)"
	python $(SRC_DIR)/train.py --config $(CONFIG)

# ============================================================
# Monitoring
# ============================================================

tensorboard:
	@echo "Starting TensorBoard..."
	@echo ""
	@echo "Open in browser: http://localhost:6006"
	@echo ""
	@echo "What to watch:"
	@echo "  - val_wer: Should DECREASE (lower is better)"
	@echo "  - train_loss: Should DECREASE smoothly"
	@echo "  - lr: Learning rate schedule"
	@echo ""
	tensorboard --logdir $(OUTPUTS_DIR) --reload_interval 5

check-events:
	@echo "Checking TensorBoard event files..."
	python $(SRC_DIR)/check_tensorboard.py

# ============================================================
# Model Upload
# ============================================================

upload:
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo "Usage: make upload MODEL=models/full.nemo REPO=parakeet-tdt-0.6b-malay"
	@exit 1
endif
ifndef REPO
	@echo "Error: REPO is required."
	@echo "Usage: make upload MODEL=models/full.nemo REPO=parakeet-tdt-0.6b-malay"
	@exit 1
endif
	@echo "Uploading model to HuggingFace..."
	python $(SRC_DIR)/upload_model.py --model $(MODEL) --repo-name $(REPO)

# ============================================================
# Cleanup
# ============================================================

clean-outputs:
	@echo "Removing training outputs..."
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Cleaned training outputs."

clean-data:
	@echo "Removing downloaded data..."
	rm -rf $(RAW_DIR)/*
	rm -rf $(MANIFEST_DIR)/*
	@echo "Cleaned data files."

clean: clean-outputs
	@echo "Cleaning complete (outputs removed, data preserved)."

clean-all: clean-outputs clean-data
	@echo "Cleaned all generated files."
