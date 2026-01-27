# ASR Model Comparison Test
# ==========================
# Compare Whisper and Parakeet models for Malaysian ASR
#
# Quick Start:
#   make inference-exp-all AUDIO=test.wav

# ============================================================
# Configuration
# ============================================================

SRC_DIR := src

# Default model for single inference
MODEL ?= mesolitica/Malaysian-whisper-large-v3-turbo-v3

# Python command (use venv if available)
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup inference inference-exp-all clean

# ============================================================
# Help
# ============================================================

help:
	@echo "ASR Model Comparison Test"
	@echo "========================="
	@echo ""
	@echo "inference-exp-all - Compare all ASR models"
	@echo "  AUDIO=path                     Audio file (required)"
	@echo ""
	@echo "  Faster Whisper:"
	@echo "    1. faster-whisper-large-v3-turbo"
	@echo ""
	@echo "  Whisper Models:"
	@echo "    2. openai/whisper-large-v3-turbo"
	@echo "    3. mesolitica/Malaysian-whisper-large-v3-turbo-v3"
	@echo "    4. whisper-malay-finetuned (latest LoRA checkpoint)"
	@echo ""
	@echo "  Parakeet Models:"
	@echo "    5. nvidia/parakeet-tdt-0.6b-v3"
	@echo "    6. parakeet-tdt-5k-v3.nemo"
	@echo "    7. parakeet-malay (latest checkpoint + LID)"
	@echo "    8. parakeet-multilingual (latest checkpoint + LID)"
	@echo ""
	@echo "  Example: make inference-exp-all AUDIO=test.wav"
	@echo ""
	@echo "inference - Single model inference"
	@echo "  AUDIO=path                     Audio file (required)"
	@echo "  MODEL=model-id                 Model to use (optional)"
	@echo "  LANGUAGE=ms|en|zh              Language hint (optional)"
	@echo ""
	@echo "  Example: make inference AUDIO=test.wav LANGUAGE=ms"
	@echo ""
	@echo "setup      - Install dependencies"
	@echo "clean      - Remove virtual environment"

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

# ============================================================
# Single Model Inference
# ============================================================

inference: setup
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo ""
	@echo "Usage:"
	@echo "  make inference AUDIO=path/to/audio.wav"
	@echo "  make inference AUDIO=path/to/audio.wav LANGUAGE=ms"
	@echo ""
	@exit 1
endif
	@echo ""
	@echo "========================================"
	@echo "Single Model Inference"
	@echo "========================================"
	@echo "Model: $(MODEL)"
	@echo "Audio: $(AUDIO)"
	@if [ -n "$(LANGUAGE)" ]; then echo "Language: $(LANGUAGE)"; else echo "Language: auto-detect"; fi
	@echo "========================================"
	@echo ""
	$(PYTHON) $(SRC_DIR)/inference.py \
		--model $(MODEL) \
		--audio $(AUDIO) \
		$(if $(LANGUAGE),--language $(LANGUAGE),)

# ============================================================
# Multi-Model Comparison Experiment
# ============================================================
# Compares Whisper and Parakeet models with different configurations.
# Each model is loaded once and runs both default and target_langs configs.

inference-exp-all: setup
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo ""
	@echo "Usage: make inference-exp-all AUDIO=path/to/audio.wav"
	@echo ""
	@echo "This compares multiple ASR models:"
	@echo ""
	@echo "Whisper Models:"
	@echo "  1. openai/whisper-large-v3-turbo (default + target_langs)"
	@echo "  2. mesolitica/Malaysian-whisper-large-v3-turbo-v3"
	@echo "  3. whisper-malay-finetuned (latest LoRA checkpoint)"
	@echo ""
	@echo "Parakeet Models:"
	@echo "  4. nvidia/parakeet-tdt-0.6b-v3"
	@echo "  5. parakeet-tdt-5k-v3.nemo"
	@echo "  6. parakeet-tdt-multilingual (latest checkpoint + language detection)"
	@echo ""
	@echo "All models run with target languages: en, ms, zh"
	@echo ""
	@exit 1
endif
	@echo ""
	@echo "========================================"
	@echo "ASR Model Comparison Experiment"
	@echo "========================================"
	@echo "Audio: $(AUDIO)"
	@echo ""
	@echo "Faster Whisper:"
	@echo "  - faster-whisper-large-v3-turbo"
	@echo ""
	@echo "Whisper Models:"
	@echo "  - openai/whisper-large-v3-turbo"
	@echo "  - mesolitica/Malaysian-whisper-large-v3-turbo-v3"
	@echo "  - whisper-malay-finetuned (latest checkpoint)"
	@echo ""
	@echo "Parakeet Models:"
	@echo "  - nvidia/parakeet-tdt-0.6b-v3"
	@echo "  - parakeet-tdt-5k-v3.nemo"
	@echo "  - parakeet-malay (with LID)"
	@echo "  - parakeet-multilingual (with LID)"
	@echo ""
	@echo "Target languages: en, ms, zh"
	@echo "========================================"
	@echo ""
	$(PYTHON) $(SRC_DIR)/inference_exp.py --audio $(AUDIO)

# ============================================================
# Cleanup
# ============================================================

clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."
