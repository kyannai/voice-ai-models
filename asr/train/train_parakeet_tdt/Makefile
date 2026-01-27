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

# Default output path for expanded TDT model
EXPANDED_TDT_OUTPUT ?= $(MODELS_DIR)/parakeet-tdt-0.6b-v3-expanded.nemo

# CUDA settings for memory optimization
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

# Python command (use venv if available)
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup check-gpu train tensorboard check-events upload \
        inference inference-ckpt expand-tdt-tokenizer \
        test-tokenizer-languages add-language-tags clean clean-outputs clean-all

# ============================================================
# Help
# ============================================================

help:
	@echo "Parakeet TDT ASR Training"
	@echo "========================="
	@echo "Environment is auto-created on first run."
	@echo ""
	@echo "expand-tdt-tokenizer - Expand vocabulary with new characters"
	@echo "  MANIFESTS='path:max'           Manifest with optional char limit"
	@echo "  EXPANDED_TDT_OUTPUT=path       Output .nemo (default: models/parakeet-tdt-0.6b-v3-expanded.nemo)"
	@echo "  Example: make expand-tdt-tokenizer MANIFESTS='zh.json:5000'"
	@echo "  Example: make expand-tdt-tokenizer MANIFESTS='zh.json:5000' EXPANDED_TDT_OUTPUT=models/tdt-zh.nemo"
	@echo ""
	@echo "train - Fine-tune model"
	@echo "  CONFIG=path                    Training config yaml (required)"
	@echo "  Example: make train CONFIG=configs/parakeet_stage1.yaml"
	@echo ""
	@echo "inference - Run inference with .nemo model"
	@echo "  MODEL=path  AUDIO=path         Model and audio file (required)"
	@echo "  LANGUAGE=en|ms|zh              Language mode (optional, applies soft bias)"
	@echo "  LANGUAGE_DETECTION=true        Auto-detect language using Whisper tiny (~10ms)"
	@echo "  BIAS_STRENGTH=float            Custom bias (e.g., -5.0 to suppress, +5.0 to boost)"
	@echo "  HARD_SUPPRESSION=true          Block Chinese completely (old behavior)"
	@echo ""
	@echo "  Soft Bias Presets: zh=+3.0 (boost Chinese), ms/en=-3.0 (prefer Latin)"
	@echo ""
	@echo "  Example: make inference MODEL=model.nemo AUDIO=test.wav LANGUAGE_DETECTION=true"
	@echo "  Example: make inference MODEL=model.nemo AUDIO=test.wav LANGUAGE=ms"
	@echo "  Example: make inference MODEL=model.nemo AUDIO=test.wav LANGUAGE=ms BIAS_STRENGTH=-5.0"
	@echo "  Example: make inference MODEL=model.nemo AUDIO=test.wav LANGUAGE=ms HARD_SUPPRESSION=true"
	@echo ""
	@echo "inference-ckpt - Run inference with checkpoint"
	@echo "  CKPT=path  BASE=path  AUDIO=path"
	@echo "  LANGUAGE=en|ms|zh              Language mode (optional, applies soft bias)"
	@echo "  LANGUAGE_DETECTION=true        Auto-detect language using Whisper tiny (~10ms)"
	@echo "  BIAS_STRENGTH=float            Custom bias (e.g., -5.0 to suppress, +5.0 to boost)"
	@echo "  HARD_SUPPRESSION=true          Block Chinese completely (old behavior)"
	@echo "  BASE can be a .nemo file or HuggingFace model ID (e.g., nvidia/parakeet-tdt-0.6b)"
	@echo ""
	@echo "  Example: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav LANGUAGE_DETECTION=true"
	@echo "  Example: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav LANGUAGE=ms BIAS_STRENGTH=-5.0"
	@echo ""
	@echo "test-tokenizer-languages - Test language support"
	@echo "  MODEL=path                     Model to test (required)"
	@echo "  LANGUAGES=en,ms,cn             Languages to check (optional)"
	@echo "  Example: make test-tokenizer-languages MODEL=models/model.nemo"
	@echo ""
	@echo "upload - Upload model to HuggingFace"
	@echo "  MODEL=path  REPO=name          Model path and repo name (required)"
	@echo "  Example: make upload MODEL=models/model.nemo REPO=parakeet-tdt-malay"
	@echo ""
	@echo "add-language-tags - Add language tags to manifests (fix language confusion)"
	@echo "  MANIFEST=path  OUTPUT=path  LANG=auto|en|ms|zh"
	@echo "  Example: make add-language-tags MANIFEST=train.json OUTPUT=train_tagged.json LANG=ms"
	@echo ""
	@echo "tensorboard    - Start TensorBoard (http://localhost:6006)"
	@echo "check-events   - Check TensorBoard event files"
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
	@echo "Installing bitsandbytes for 8-bit optimizer..."
	@. $(VENV_DIR)/bin/activate && uv pip install -q bitsandbytes
	@echo "Environment ready!"

check-gpu: setup
	@echo "Checking GPU availability..."
	@$(PYTHON) -c "import torch; \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'GPU count: {torch.cuda.device_count()}'); \
		[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)') \
			for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# ============================================================
# Vocabulary Expansion
# ============================================================
# Supports multiple manifests with individual token limits.
# Format: MANIFESTS="path1:max1 path2:max2 ..."

expand-tdt-tokenizer: setup
	@echo "============================================================"
	@echo "Expanding nvidia/parakeet-tdt-0.6b-v3 Vocabulary"
	@echo "============================================================"
	@echo "TDT expansion parameters:"
	@echo "  MANIFESTS=$(MANIFESTS)"
	@echo "  EXPANDED_TDT_OUTPUT=$(EXPANDED_TDT_OUTPUT)"
	@echo "  MAX_TOTAL_CHARS=$(MAX_TOTAL_CHARS) (optional, limits total chars)"
	@echo "  TEST_AUDIO=$(TEST_AUDIO) (optional)"
	@if [ -z "$(MANIFESTS)" ]; then \
		echo ""; \
		echo "Usage:"; \
		echo "  # Single manifest:"; \
		echo "  make expand-tdt-tokenizer \\"; \
		echo "    MANIFESTS='chinese.json:5000' \\"; \
		echo "    EXPANDED_TDT_OUTPUT=./models/parakeet-tdt-0.6b-v3-zh.nemo"; \
		echo ""; \
		echo "  # Multiple manifests (Chinese + Malay):"; \
		echo "  make expand-tdt-tokenizer \\"; \
		echo "    MANIFESTS='chinese.json:5000 malay.json:3000' \\"; \
		echo "    EXPANDED_TDT_OUTPUT=./models/parakeet-tdt-multilingual.nemo"; \
		echo ""; \
		echo "  Format: path:max_tokens (max_tokens is optional)"; \
		exit 1; \
	fi
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(SRC_DIR)/expand_tdt_tokenizer.py \
		--manifests $(MANIFESTS) \
		--output "$(EXPANDED_TDT_OUTPUT)" \
		$(if $(MAX_TOTAL_CHARS),--max-total-chars $(MAX_TOTAL_CHARS),) \
		$(if $(TEST_AUDIO),--test-audio "$(TEST_AUDIO)",)
	@echo ""
	@echo "Done: expanded TDT model saved to $(EXPANDED_TDT_OUTPUT)"
	@echo "Note: Fine-tune this model on target language data to train the new embeddings."

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

# ============================================================
# Inference
# ============================================================

# Inference with .nemo model
# Usage: make inference MODEL=path/to/model.nemo AUDIO=path/to/audio.wav [LANGUAGE=en|ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]
inference: setup
ifndef MODEL
	@echo "Error: MODEL is required."
	@echo "Usage: make inference MODEL=model.nemo AUDIO=test.wav [LANGUAGE=ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]"
	@exit 1
endif
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo "Usage: make inference MODEL=model.nemo AUDIO=test.wav [LANGUAGE=ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]"
	@exit 1
endif
	$(PYTHON) $(SRC_DIR)/inference.py --model $(MODEL) --audio $(AUDIO) \
		$(if $(LANGUAGE),--language $(LANGUAGE),) \
		$(if $(filter true,$(LANGUAGE_DETECTION)),--language-detection,) \
		$(if $(BIAS_STRENGTH),--bias-strength $(BIAS_STRENGTH),) \
		$(if $(filter true,$(HARD_SUPPRESSION)),--hard-suppression,)

# Inference with .ckpt checkpoint
# Usage: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav [LANGUAGE=en|ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]
inference-ckpt: setup
ifndef CKPT
	@echo "Error: CKPT is required."
	@echo "Usage: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav [LANGUAGE=ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]"
	@exit 1
endif
ifndef BASE
	@echo "Error: BASE is required (the base .nemo model for architecture)."
	@echo "Usage: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav [LANGUAGE=ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]"
	@exit 1
endif
ifndef AUDIO
	@echo "Error: AUDIO is required."
	@echo "Usage: make inference-ckpt CKPT=ckpt.ckpt BASE=base.nemo AUDIO=test.wav [LANGUAGE=ms|zh] [LANGUAGE_DETECTION=true] [BIAS_STRENGTH=-3.0] [HARD_SUPPRESSION=true]"
	@exit 1
endif
	$(PYTHON) $(SRC_DIR)/inference.py --checkpoint $(CKPT) --base-model $(BASE) --audio $(AUDIO) \
		$(if $(LANGUAGE),--language $(LANGUAGE),) \
		$(if $(filter true,$(LANGUAGE_DETECTION)),--language-detection,) \
		$(if $(BIAS_STRENGTH),--bias-strength $(BIAS_STRENGTH),) \
		$(if $(filter true,$(HARD_SUPPRESSION)),--hard-suppression,)

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
# Test Tokenizer Language Support
# ============================================================

TOKENIZER_DIR := ../common/tokenizers

test-tokenizer-languages: setup
	@if [ -z "$(MODEL)" ]; then \
		echo ""; \
		echo "Test Tokenizer Language Support"; \
		echo "================================"; \
		echo "Usage: make test-tokenizer-languages MODEL=path/to/model.nemo"; \
		echo "       make test-tokenizer-languages MODEL=nvidia/parakeet-tdt-0.6b-v3"; \
		echo ""; \
		echo "Options:"; \
		echo "  MODEL      - HuggingFace model name or path to .nemo file (required)"; \
		echo "  LANGUAGES  - Comma-separated languages to test (default: en,ms,cn)"; \
		echo "  VERBOSE    - Set to 1 for detailed output"; \
		echo ""; \
		exit 1; \
	fi
	$(PYTHON) $(TOKENIZER_DIR)/test_tokenizer_languages.py \
		"$(MODEL)" \
		$(if $(filter 1,$(VERBOSE)),--verbose,) \
		$(if $(LANGUAGES),--languages $(LANGUAGES),)

# ============================================================
# Language Tagging (Multilingual Training)
# ============================================================
# Add language tags to manifests for language-conditioned ASR.
# This solves the language confusion problem (e.g., Malay → Chinese).

add-language-tags: setup
	@if [ -z "$(MANIFEST)" ]; then \
		echo ""; \
		echo "Add Language Tags to Manifest"; \
		echo "=============================="; \
		echo "Solves: Language confusion in multilingual models"; \
		echo ""; \
		echo "Usage:"; \
		echo "  make add-language-tags MANIFEST=train.json OUTPUT=train_tagged.json LANG=auto"; \
		echo ""; \
		echo "Options:"; \
		echo "  MANIFEST  - Input manifest file (required)"; \
		echo "  OUTPUT    - Output manifest file (required)"; \
		echo "  LANG      - Language: auto, en, ms, zh (default: auto)"; \
		echo ""; \
		echo "Example workflow:"; \
		echo "  1. Tag Malay manifest:"; \
		echo "     make add-language-tags MANIFEST=malay.json OUTPUT=malay_tagged.json LANG=ms"; \
		echo "  2. Tag Chinese manifest:"; \
		echo "     make add-language-tags MANIFEST=chinese.json OUTPUT=chinese_tagged.json LANG=zh"; \
		echo "  3. Combine: cat malay_tagged.json chinese_tagged.json > train_multilingual.json"; \
		echo "  4. Retrain model with tagged data"; \
		echo "  5. At inference, prepend '<|ms|>' for Malay or '<|zh|>' for Chinese"; \
		echo ""; \
		exit 1; \
	fi
ifndef OUTPUT
	@echo "Error: OUTPUT is required."
	@echo "Usage: make add-language-tags MANIFEST=train.json OUTPUT=train_tagged.json LANG=auto"
	@exit 1
endif
	$(PYTHON) $(SRC_DIR)/add_language_tags.py \
		--manifest "$(MANIFEST)" \
		--output "$(OUTPUT)" \
		--lang $(or $(LANG),auto)
	@echo ""
	@echo "Tagged manifest saved to: $(OUTPUT)"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update training config to use tagged manifest"
	@echo "2. Retrain model"
	@echo "3. At inference, prepend language tag to condition output:"
	@echo "   - Malay:   '<|ms|>'"
	@echo "   - Chinese: '<|zh|>'"
	@echo "   - English: '<|en|>'"

# ============================================================
# Cleanup
# ============================================================

clean-outputs:
	@echo "Removing training outputs..."
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Cleaned training outputs."

clean: clean-outputs

clean-all: clean-outputs
