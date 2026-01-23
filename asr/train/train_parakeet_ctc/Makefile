# Makefile for Parakeet CTC Multilingual Training
# ================================================

SHELL := /bin/bash
PYTHON := python3
VENV := .venv

# Model paths
BASE_MODEL := nvidia/parakeet-ctc-1.1b
MULTILINGUAL_MODEL := ./models/parakeet-ctc-1.1b-multilingual.nemo

# Data paths
TRAIN_MANIFEST ?= ../training_data/multilingual-overfit/manifests/train_manifest.json
CHINESE_MANIFEST ?= ../training_data/KeSpeech/data/manifests/train_manifest.json

# Default number of Chinese chars to add
MAX_CHINESE_CHARS ?= 4000

.PHONY: help setup expand-vocab train-overfit train clean

help:
	@echo "Parakeet CTC Multilingual Training"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Create virtual environment and install dependencies"
	@echo ""
	@echo "Vocabulary Expansion:"
	@echo "  make expand-vocab       - Expand vocabulary with Chinese characters"
	@echo "                            Options: CHINESE_MANIFEST=path MAX_CHINESE_CHARS=4000"
	@echo ""
	@echo "Training:"
	@echo "  make train-overfit      - Run overfit sanity check (tiny dataset)"
	@echo "  make train              - Run full multilingual training"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-model        - Inspect model architecture"
	@echo "  make clean              - Remove outputs and cache"

# ============================================================
# Setup
# ============================================================

setup:
	@echo "Creating virtual environment..."
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install uv
	. $(VENV)/bin/activate && uv pip install -r requirements.txt
	@echo "Setup complete!"

# ============================================================
# Vocabulary Expansion
# ============================================================

expand-vocab: setup
	@echo "============================================================"
	@echo "Expanding Vocabulary with Chinese Characters"
	@echo "============================================================"
	@mkdir -p models
	. $(VENV)/bin/activate && $(PYTHON) src/expand_vocabulary.py \
		--base-model $(BASE_MODEL) \
		--manifest $(CHINESE_MANIFEST) \
		--output $(MULTILINGUAL_MODEL) \
		--max-chinese-chars $(MAX_CHINESE_CHARS)

# ============================================================
# Training
# ============================================================

train-overfit: setup
	@echo "============================================================"
	@echo "Overfit Sanity Check (Parakeet CTC)"
	@echo "============================================================"
	@if [ ! -f $(MULTILINGUAL_MODEL) ]; then \
		echo "Error: Model not found at $(MULTILINGUAL_MODEL)"; \
		echo "Run 'make expand-vocab' first"; \
		exit 1; \
	fi
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	. $(VENV)/bin/activate && $(PYTHON) src/train.py --config configs/parakeet_overfit.yaml

train: setup
	@echo "============================================================"
	@echo "Full Multilingual Training (Parakeet CTC)"
	@echo "============================================================"
	@if [ ! -f $(MULTILINGUAL_MODEL) ]; then \
		echo "Error: Model not found at $(MULTILINGUAL_MODEL)"; \
		echo "Run 'make expand-vocab' first"; \
		exit 1; \
	fi
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	. $(VENV)/bin/activate && $(PYTHON) src/train.py --config configs/parakeet_multilingual.yaml

# ============================================================
# Utilities
# ============================================================

check-model: setup
	@echo "============================================================"
	@echo "Checking Model Architecture"
	@echo "============================================================"
	. $(VENV)/bin/activate && $(PYTHON) -c "\
import nemo.collections.asr as nemo_asr; \
import sys; \
model_path = sys.argv[1] if len(sys.argv) > 1 else '$(BASE_MODEL)'; \
print(f'Loading: {model_path}'); \
if model_path.endswith('.nemo'): \
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path); \
else: \
    model = nemo_asr.models.ASRModel.from_pretrained(model_path); \
print(f'Model type: {type(model).__name__}'); \
print(f'Vocab size: {model.tokenizer.vocab_size}'); \
print(f'Decoder output: {model.decoder._num_classes}'); \
print(f'Blank ID: {model.decoder._num_classes - 1}'); \
" $(MODEL)

clean:
	rm -rf outputs/
	rm -rf $(VENV)
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
