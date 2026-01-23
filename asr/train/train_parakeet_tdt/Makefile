# Parakeet TDT ASR Training
# ==========================
# Fine-tune NVIDIA Parakeet TDT 0.6B v3 for Malay ASR
#
# Quick Start:
#   make install        - Install dependencies
#   make train CONFIG=configs/parakeet_stage1.yaml
#   make tensorboard    - Monitor training

# ============================================================
# Configuration
# ============================================================

# Directories
MODELS_DIR := models
OUTPUTS_DIR := outputs
SRC_DIR := src

# CUDA settings for memory optimization
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

# Python command (use venv if available)
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup install install-8bit check-gpu \
        train tensorboard check-events check-tokenizer upload \
        inference inference-ckpt \
        clean clean-outputs clean-all

# ============================================================
# Help
# ============================================================

help:
	@echo "Parakeet TDT ASR Training Pipeline"
	@echo "==================================="
	@echo ""
	@echo "=== Environment (auto-runs when needed) ==="
	@echo "  make setup         - Create venv and install all dependencies"
	@echo "  make check-gpu     - Check GPU availability"
	@echo ""
	@echo "=== Training ==="
	@echo "  make train CONFIG=path/to/config.yaml"
	@echo ""
	@echo "=== Inference ==="
	@echo "  make inference MODEL=model.nemo AUDIO=audio.wav"
	@echo "  make inference-ckpt CKPT=checkpoint.ckpt BASE=base.nemo AUDIO=audio.wav"
	@echo ""
	@echo "=== Monitoring & Testing ==="
	@echo "  make tensorboard   - Start TensorBoard (http://localhost:6006)"
	@echo "  make check-events  - Check TensorBoard event files"
	@echo "  make check-tokenizer MODEL=path/to/model"
	@echo ""
	@echo "=== Model Upload ==="
	@echo "  make upload MODEL=path/to/model.nemo REPO=repo-name"
	@echo ""
	@echo "=== Cleanup ==="
	@echo "  make clean         - Remove training outputs"

# ============================================================
# Installation (Step 1)
# ============================================================

# Setup venv and install all dependencies (called automatically by train targets)
setup:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.10; \
	fi
	@echo "Installing dependencies..."
	@. $(VENV_DIR)/bin/activate && uv pip install -q -r requirements.txt
	@echo "Installing bitsandbytes for 8-bit optimizer..."
	@. $(VENV_DIR)/bin/activate && uv pip install -q bitsandbytes
	@echo "Environment ready!"

install: setup
	@echo ""
	@echo "Dependencies installed successfully!"
	@echo ""
	@echo "Note: 8-bit optimizer (bitsandbytes) is included by default."

install-8bit: setup
	@echo "8-bit optimizer already installed via setup."

check-gpu: setup
	@echo "Checking GPU availability..."
	@$(PYTHON) -c "import torch; \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'GPU count: {torch.cuda.device_count()}'); \
		[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)') \
			for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# ============================================================
# Training
# ============================================================

train: setup
ifndef CONFIG
	@echo "Error: CONFIG is required."
	@echo "Usage: make train CONFIG=configs/parakeet_stage1.yaml"
	@exit 1
endif
	@echo ""
	@echo "========================================"
	@echo "Parakeet TDT Training"
	@echo "========================================"
	@echo "Config: $(CONFIG)"
	@echo ""
	@echo "TIP: Monitor training in another terminal:"
	@echo "  make tensorboard"
	@echo "  Open: http://localhost:6006"
	@echo "========================================"
	@echo ""
	$(PYTHON) $(SRC_DIR)/train.py --config $(CONFIG)

# ============================================================
# Monitoring
# ============================================================

tensorboard: setup
	@echo "Starting TensorBoard..."
	@echo ""
	@echo "Open in browser: http://localhost:6006"
	@echo ""
	@echo "What to watch:"
	@echo "  - val_wer: Should DECREASE (lower is better)"
	@echo "  - train_loss: Should DECREASE smoothly"
	@echo "  - lr: Learning rate schedule"
	@echo ""
	$(VENV_DIR)/bin/tensorboard --logdir $(OUTPUTS_DIR) --reload_interval 5

check-events: setup
	@echo "Checking TensorBoard event files..."
	$(PYTHON) $(SRC_DIR)/check_tensorboard.py

# Check tokenizer language support
TOKENIZER_TOOL := tools/check_tokenizer/check_tokenizer.py
TOKENIZER_RESULTS := tools/check_tokenizer/results

check-tokenizer: setup
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo "Usage: make check-tokenizer MODEL=nvidia/parakeet-tdt-0.6b-v3"
	@echo "       make check-tokenizer MODEL=path/to/model.nemo"
	@exit 1
endif
	@echo "Checking tokenizer language support..."
	$(PYTHON) $(TOKENIZER_TOOL) --model $(MODEL) --show-vocab 20 --output-dir $(TOKENIZER_RESULTS)

# ============================================================
# Inference
# ============================================================

# Inference with .nemo model
# Usage: make inference MODEL=path/to/model.nemo AUDIO=path/to/audio.wav
inference: setup
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo "Usage: make inference MODEL=path/to/model.nemo AUDIO=path/to/audio.wav"
	@exit 1
endif
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo "Usage: make inference MODEL=path/to/model.nemo AUDIO=path/to/audio.wav"
	@exit 1
endif
	$(PYTHON) $(SRC_DIR)/inference.py --model $(MODEL) --audio $(AUDIO)

# Inference with .ckpt checkpoint
# Usage: make inference-ckpt CKPT=path/to/checkpoint.ckpt BASE=path/to/base.nemo AUDIO=path/to/audio.wav
inference-ckpt: setup
ifndef CKPT
	@echo "Error: CKPT is required."
	@echo "Usage: make inference-ckpt CKPT=path/to/checkpoint.ckpt BASE=path/to/base.nemo AUDIO=path/to/audio.wav"
	@exit 1
endif
ifndef BASE
	@echo "Error: BASE is required (the base .nemo model for architecture)."
	@echo "Usage: make inference-ckpt CKPT=path/to/checkpoint.ckpt BASE=path/to/base.nemo AUDIO=path/to/audio.wav"
	@exit 1
endif
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo "Usage: make inference-ckpt CKPT=path/to/checkpoint.ckpt BASE=path/to/base.nemo AUDIO=path/to/audio.wav"
	@exit 1
endif
	$(PYTHON) $(SRC_DIR)/inference.py --checkpoint $(CKPT) --base-model $(BASE) --audio $(AUDIO)

# ============================================================
# Model Upload
# ============================================================

upload: setup
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
	$(PYTHON) $(SRC_DIR)/upload_model.py --model $(MODEL) --repo-name $(REPO)

# ============================================================
# Cleanup
# ============================================================

clean-outputs:
	@echo "Removing training outputs..."
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Cleaned training outputs."

clean: clean-outputs

clean-all: clean-outputs
