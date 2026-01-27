# Whisper Fine-tuning for Malay ASR
# ==================================
# Fine-tune OpenAI Whisper on Malaysian STT data
#
# Quick Start:
#   make setup           - Install dependencies
#   make prepare-data    - Prepare Malaysian-STT dataset
#   make train CONFIG=configs/whisper_malay.yaml
#   make tensorboard     - Monitor training

# ============================================================
# Configuration
# ============================================================

# Directories
MODELS_DIR := models
OUTPUTS_DIR := outputs
SRC_DIR := src
CONFIGS_DIR := configs
DATA_DIR := data

# Default model
MODEL ?= openai/whisper-large-v3-turbo

# Python command (use venv if available)
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Malaysian-STT dataset location (relative to this Makefile)
MALAYSIAN_STT_DIR := ../training_data/malaysian-stt

.PHONY: help setup check-gpu prepare-data train tensorboard \
        inference upload clean clean-outputs clean-all

# ============================================================
# Help
# ============================================================

help:
	@echo "Whisper Fine-tuning for Malay ASR"
	@echo "================================="
	@echo "Environment is auto-created on first run."
	@echo ""
	@echo "prepare-data - Prepare Malaysian-STT dataset"
	@echo "  DATA_DIR=path                  Output directory (default: ./data)"
	@echo "  MAX_SAMPLES=num                Limit samples (optional, for testing)"
	@echo "  Example: make prepare-data MAX_SAMPLES=10000"
	@echo ""
	@echo "train - Fine-tune Whisper model"
	@echo "  CONFIG=path                    Training config yaml (required)"
	@echo "  Example: make train CONFIG=configs/whisper_malay.yaml"
	@echo ""
	@echo "inference - Run inference with fine-tuned model"
	@echo "  MODEL=path  AUDIO=path         Model and audio file (required)"
	@echo "  LANGUAGE=ms|en|zh|ta           Force specific language (optional)"
	@echo "  TARGET_LANGS='en ms zh ta'     Restrict to target languages only (optional)"
	@echo "  USE_DEFAULTS=true              Use default targets: en ms zh ta"
	@echo ""
	@echo "  Example: make inference MODEL=./outputs/final AUDIO=test.wav"
	@echo "  Example: make inference MODEL=./outputs/final AUDIO=test.wav LANGUAGE=ms"
	@echo "  Example: make inference MODEL=./outputs/final AUDIO=test.wav USE_DEFAULTS=true"
	@echo "  Example: make inference MODEL=./outputs/final AUDIO=test.wav TARGET_LANGS='en ms zh ta'"
	@echo ""
	@echo "upload - Upload model to HuggingFace"
	@echo "  MODEL=path  REPO=name          Model path and repo name (required)"
	@echo "  Example: make upload MODEL=./outputs/final REPO=whisper-malay"
	@echo ""
	@echo "tensorboard    - Start TensorBoard (http://localhost:6006)"
	@echo "check-gpu      - Check GPU availability"
	@echo "clean          - Remove training outputs"

# ============================================================
# Setup (auto-runs as dependency)
# ============================================================

setup:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.10; \
	fi
	@echo "Installing dependencies..."
	@. $(VENV_DIR)/bin/activate && uv pip install -q -r requirements.txt
	@echo "Environment ready!"

check-gpu: setup
	@echo "Checking GPU availability..."
	@$(PYTHON) -c "import torch; \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'GPU count: {torch.cuda.device_count()}'); \
		[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)') \
			for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# ============================================================
# Data Preparation
# ============================================================

prepare-data: setup
	@echo ""
	@echo "========================================"
	@echo "Preparing Malaysian-STT Dataset"
	@echo "========================================"
	@echo "Output: $(DATA_DIR)"
	@if [ -n "$(MAX_SAMPLES)" ]; then echo "Max samples: $(MAX_SAMPLES)"; fi
	@echo "========================================"
	@echo ""
	@mkdir -p $(DATA_DIR)
	$(PYTHON) $(SRC_DIR)/prepare_data.py \
		--output-dir $(DATA_DIR) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES),) \
		$(if $(DATASETS),--datasets $(DATASETS),)

# ============================================================
# Training
# ============================================================

train: setup
ifndef CONFIG
	@echo "Error: CONFIG is required."
	@echo "Usage: make train CONFIG=configs/whisper_malay.yaml"
	@exit 1
endif
	@echo ""
	@echo "========================================"
	@echo "Whisper Fine-tuning"
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
# Inference
# ============================================================

# Default target languages for Malaysian context
DEFAULT_TARGET_LANGS := en ms zh ta

inference: setup
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo ""
	@echo "Usage: make inference MODEL=path/to/model AUDIO=test.wav [LANGUAGE=ms]"
	@echo "       make inference MODEL=path/to/model AUDIO=test.wav TARGET_LANGS='en ms zh ta'"
	@echo ""
	@echo "Options:"
	@echo "  MODEL=path              Model path or HuggingFace ID (required)"
	@echo "  AUDIO=path              Audio file (required)"
	@echo "  LANGUAGE=ms|en|zh|ta    Force specific language (optional)"
	@echo "  TARGET_LANGS='...'      Restrict to target languages (optional)"
	@echo "  USE_DEFAULTS=true       Use default targets: $(DEFAULT_TARGET_LANGS)"
	@echo ""
	@exit 1
endif
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo "Usage: make inference MODEL=path/to/model AUDIO=test.wav [LANGUAGE=ms]"
	@exit 1
endif
	@echo ""
	@echo "========================================"
	@echo "Whisper Inference"
	@echo "========================================"
	@echo "Model: $(MODEL)"
	@echo "Audio: $(AUDIO)"
	@if [ -n "$(LANGUAGE)" ]; then echo "Language: $(LANGUAGE) (forced)"; \
	elif [ -n "$(TARGET_LANGS)" ]; then echo "Target Languages: $(TARGET_LANGS)"; \
	elif [ "$(USE_DEFAULTS)" = "true" ]; then echo "Target Languages: $(DEFAULT_TARGET_LANGS) (defaults)"; \
	else echo "Language: auto-detect (all languages)"; fi
	@echo "========================================"
	@echo ""
	$(PYTHON) $(SRC_DIR)/inference.py \
		--model $(MODEL) \
		--audio $(AUDIO) \
		$(if $(LANGUAGE),--language $(LANGUAGE),) \
		$(if $(TARGET_LANGS),--target-languages $(TARGET_LANGS),) \
		$(if $(filter true,$(USE_DEFAULTS)),--use-defaults,)

# ============================================================
# Monitoring
# ============================================================

tensorboard: setup
	@echo "Starting TensorBoard..."
	@echo ""
	@echo "Open in browser: http://localhost:6006"
	@echo ""
	@echo "What to watch:"
	@echo "  - eval/wer: Should DECREASE (lower is better)"
	@echo "  - train/loss: Should DECREASE smoothly"
	@echo "  - train/learning_rate: Learning rate schedule"
	@echo ""
	$(VENV_DIR)/bin/tensorboard --logdir $(OUTPUTS_DIR) --reload_interval 5

# ============================================================
# Model Upload
# ============================================================

upload: setup
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo "Usage: make upload MODEL=./outputs/final REPO=whisper-malay"
	@exit 1
endif
ifndef REPO
	@echo "Error: REPO is required."
	@echo "Usage: make upload MODEL=./outputs/final REPO=whisper-malay"
	@exit 1
endif
	@echo "Uploading model to HuggingFace..."
	$(PYTHON) -c "from huggingface_hub import HfApi; \
		api = HfApi(); \
		api.upload_folder(folder_path='$(MODEL)', repo_id='$(REPO)', repo_type='model')"

# ============================================================
# Cleanup
# ============================================================

clean-outputs:
	@echo "Removing training outputs..."
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Cleaned training outputs."

clean-data:
	@echo "Removing prepared data..."
	rm -rf $(DATA_DIR)/*
	@echo "Cleaned prepared data."

clean: clean-outputs

clean-all: clean-outputs clean-data
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."
