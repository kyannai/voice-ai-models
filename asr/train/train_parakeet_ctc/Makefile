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

# Default output path for expanded CTC model
EXPANDED_CTC_OUTPUT ?= ./models/parakeet-ctc-1.1b-expanded.nemo

.PHONY: help setup expand-vocab expand-ctc-tokenizer train-overfit train test-tokenizer-languages clean

help:
	@echo "Parakeet CTC Multilingual Training"
	@echo "==================================="
	@echo "Environment is auto-created on first run."
	@echo ""
	@echo "expand-ctc-tokenizer - Expand vocabulary with new characters"
	@echo "  MANIFESTS='path:max'           Manifest with optional char limit"
	@echo "  EXPANDED_CTC_OUTPUT=path       Output .nemo (default: models/parakeet-ctc-1.1b-expanded.nemo)"
	@echo "  Example: make expand-ctc-tokenizer MANIFESTS='zh.json:5000'"
	@echo "  Example: make expand-ctc-tokenizer MANIFESTS='zh.json:5000' EXPANDED_CTC_OUTPUT=models/ctc-zh.nemo"
	@echo ""
	@echo "train - Full multilingual training"
	@echo "  Example: make train"
	@echo ""
	@echo "train-overfit - Overfit sanity check (tiny dataset)"
	@echo "  Example: make train-overfit"
	@echo ""
	@echo "test-tokenizer-languages - Test language support"
	@echo "  MODEL=path                     Model to test (required)"
	@echo "  LANGUAGES=en,ms,cn             Languages to check (optional)"
	@echo "  Example: make test-tokenizer-languages MODEL=models/model.nemo"
	@echo ""
	@echo "check-model    - Inspect model architecture"
	@echo "clean          - Remove outputs and cache"

# ============================================================
# Setup (auto-runs as dependency)
# ============================================================

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.10; \
	fi
	@. $(VENV)/bin/activate && uv pip install -q -r requirements.txt

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
# Expand CTC Tokenizer (Multi-Manifest Support)
# ============================================================
# Supports multiple manifests with individual token limits.
# Format: MANIFESTS="path1:max1 path2:max2 ..."

expand-ctc-tokenizer: setup
	@echo "============================================================"
	@echo "Expanding nvidia/parakeet-ctc-1.1b Vocabulary"
	@echo "============================================================"
	@echo "CTC expansion parameters:"
	@echo "  MANIFESTS=$(MANIFESTS)"
	@echo "  EXPANDED_CTC_OUTPUT=$(EXPANDED_CTC_OUTPUT)"
	@echo "  MAX_TOTAL_CHARS=$(MAX_TOTAL_CHARS) (optional, limits total chars)"
	@echo "  TEST_AUDIO=$(TEST_AUDIO) (optional)"
	@if [ -z "$(MANIFESTS)" ]; then \
		echo ""; \
		echo "Usage:"; \
		echo "  # Single manifest:"; \
		echo "  make expand-ctc-tokenizer \\"; \
		echo "    MANIFESTS='chinese.json:5000' \\"; \
		echo "    EXPANDED_CTC_OUTPUT=./models/parakeet-ctc-1.1b-zh.nemo"; \
		echo ""; \
		echo "  # Multiple manifests (Chinese + Malay):"; \
		echo "  make expand-ctc-tokenizer \\"; \
		echo "    MANIFESTS='chinese.json:5000 malay.json:3000' \\"; \
		echo "    EXPANDED_CTC_OUTPUT=./models/parakeet-ctc-multilingual.nemo"; \
		echo ""; \
		echo "  Format: path:max_tokens (max_tokens is optional)"; \
		exit 1; \
	fi
	@mkdir -p models
	. $(VENV)/bin/activate && $(PYTHON) src/expand_ctc_tokenizer.py \
		--manifests $(MANIFESTS) \
		--output "$(EXPANDED_CTC_OUTPUT)" \
		$(if $(MAX_TOTAL_CHARS),--max-total-chars $(MAX_TOTAL_CHARS),) \
		$(if $(TEST_AUDIO),--test-audio "$(TEST_AUDIO)",)
	@echo ""
	@echo "Done: expanded CTC model saved to $(EXPANDED_CTC_OUTPUT)"
	@echo "Note: Fine-tune this model on target language data to train the new embeddings."

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

# ============================================================
# Test Tokenizer Language Support
# ============================================================

TOKENIZER_DIR := ../common/tokenizers

test-tokenizer-languages: setup
	@if [ -z "$(MODEL)" ]; then \
		echo ""; \
		echo "Test Tokenizer Language Support"; \
		echo "================================"; \
		echo "Usage: make test-tokenizer-languages MODEL=path/to/model.nemo"; \
		echo "       make test-tokenizer-languages MODEL=nvidia/parakeet-ctc-1.1b"; \
		echo ""; \
		echo "Options:"; \
		echo "  MODEL      - HuggingFace model name or path to .nemo file (required)"; \
		echo "  LANGUAGES  - Comma-separated languages to test (default: en,ms,cn)"; \
		echo "  VERBOSE    - Set to 1 for detailed output"; \
		echo ""; \
		exit 1; \
	fi
	. $(VENV)/bin/activate && python $(TOKENIZER_DIR)/test_tokenizer_languages.py \
		"$(MODEL)" \
		$(if $(filter 1,$(VERBOSE)),--verbose,) \
		$(if $(LANGUAGES),--languages $(LANGUAGES),)

clean:
	rm -rf outputs/
	rm -rf $(VENV)
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
