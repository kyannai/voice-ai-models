# Nemotron Speech Streaming ASR Training
# ========================================
# Fine-tune NVIDIA Nemotron Speech Streaming 0.6B for Malay ASR
#
# Quick Start:
#   make install        - Install dependencies
#   make check-gpu      - Check GPU availability
#   make train          - Start training
#   make tensorboard    - Monitor training

# ============================================================
# Configuration
# ============================================================

MODEL_NAME := nvidia/nemotron-speech-streaming-en-0.6b
DEFAULT_CONFIG := configs/nemotron_streaming.yaml

# Directories
DATA_DIR := data
MODELS_DIR := models
OUTPUTS_DIR := outputs
SRC_DIR := src

# CUDA settings for memory optimization
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: help install install-8bit check-gpu \
        train train-custom train-resume \
        tensorboard clean clean-outputs

# ============================================================
# Help
# ============================================================

help:
	@echo "Nemotron Speech Streaming ASR Training Pipeline"
	@echo "================================================"
	@echo ""
	@echo "=== Installation (Step 1) ==="
	@echo "  make install       - Install Python dependencies"
	@echo "  make install-8bit  - Install 8-bit optimizer (recommended)"
	@echo "  make check-gpu     - Check GPU availability"
	@echo ""
	@echo "=== Training (Step 2) ==="
	@echo "  make train         - Train with default config"
	@echo "  make train-custom CONFIG=path/to/config.yaml"
	@echo "  make train-resume CHECKPOINT=path/to/ckpt"
	@echo ""
	@echo "=== Monitoring ==="
	@echo "  make tensorboard   - Start TensorBoard (http://localhost:6006)"
	@echo ""
	@echo "=== Cleanup ==="
	@echo "  make clean-outputs - Remove training outputs"
	@echo "  make clean         - Same as clean-outputs"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL_NAME     = $(MODEL_NAME)"
	@echo "  DEFAULT_CONFIG = $(DEFAULT_CONFIG)"
	@echo ""
	@echo "Note: Data preparation uses the same pipeline as train_parakeet_tdt."
	@echo "      Copy or symlink manifests from ../train_parakeet_tdt/data/manifests/"

# ============================================================
# Installation (Step 1)
# ============================================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo ""
	@echo "Note: For 8-bit optimizer support (recommended):"
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
# Training (Step 2)
# ============================================================

train:
	@echo ""
	@echo "========================================"
	@echo "Nemotron Speech Streaming Training"
	@echo "========================================"
	@echo "Config: $(DEFAULT_CONFIG)"
	@echo ""
	@echo "TIP: Monitor training in another terminal:"
	@echo "  make tensorboard"
	@echo "  Open: http://localhost:6006"
	@echo "========================================"
	@echo ""
	python $(SRC_DIR)/train.py --config $(DEFAULT_CONFIG)

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
	@echo "Usage: make train-resume CHECKPOINT=path/to/checkpoint.ckpt"
	@exit 1
endif
ifndef CONFIG
	$(eval CONFIG := $(DEFAULT_CONFIG))
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

# ============================================================
# Cleanup
# ============================================================

clean-outputs:
	@echo "Removing training outputs..."
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Cleaned training outputs."

clean: clean-outputs
	@echo "Cleaning complete."
