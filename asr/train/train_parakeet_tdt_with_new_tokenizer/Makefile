# Parakeet TDT with Multilingual Tokenizer
# =========================================
# Training Parakeet TDT 0.6B with new EN/MS/ZH tokenizer
#
# Workflow:
#   1. Prepare data: cd ../  && make prepare-malaysian-stt-normalized
#   2. Prepare data: cd ../  && make prepare-kespeech-full KESPEECH_DATA=/path/to/kespeech
#   3. Train tokenizer: cd .. && make train-tokenizer MS_MANIFEST=... ZH_MANIFEST=...
#   4. Combine manifests: cd .. && make prepare-multilingual MS_MANIFEST=... ZH_MANIFEST=...
#   5. Modify model: make modify-model
#   6. Train: make train
#
# See: make help

SHELL := /bin/bash
PYTHON := python3

# =============================================================================
# Configuration
# =============================================================================

# Paths
TOKENIZER_PATH := ../common/tokenizers/tokenizer_multilingual.model
BASE_MODEL := nvidia/parakeet-tdt-0.6b-v3
MODIFIED_MODEL := ./models/parakeet-tdt-0.6b-multilingual-init.nemo
CONFIG := configs/parakeet_multilingual.yaml
OVERFIT_CONFIG := configs/parakeet_overfit.yaml

# Virtual environment
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo "Parakeet TDT Multilingual Training"
	@echo "==================================="
	@echo ""
	@echo "Prerequisites (run from parent directory: asr/train/):"
	@echo "  1. make prepare-malaysian-stt-normalized  - Prepare MS/EN data"
	@echo "  2. make prepare-kespeech-full             - Prepare Chinese data"
	@echo "  3. make train-tokenizer                   - Train tokenizer"
	@echo "  4. make prepare-multilingual              - Combine manifests"
	@echo ""
	@echo "Vocabulary Swap (Quick Experiments):"
	@echo "  make swap-vocab                           - Swap vocab (random init weights)"
	@echo "  make swap-vocab-restore                   - Swap vocab and restore original weights"
	@echo ""
	@echo "Model Modification:"
	@echo "  make modify-model                         - Create model with new tokenizer"
	@echo "  make modify-model-dry-run                 - Dry run (show changes only)"
	@echo ""
	@echo "Training:"
	@echo "  make train                                - Start/resume training"
	@echo "  make train-debug                          - Train with verbose logging"
	@echo "  make train-overfit                        - Overfit tiny subset (sanity check)"
	@echo ""
	@echo "Utilities:"
	@echo "  make setup                                - Create venv and install deps"
	@echo "  make test-tokenizer                       - Test tokenizer on sample texts"
	@echo "  make test-inference                       - Test model on audio samples"
	@echo "  make clean                                - Remove outputs"
	@echo "  make tensorboard                          - Launch TensorBoard"
	@echo ""
	@echo "Parameters:"
	@echo "  CONFIG           - Config file (default: $(CONFIG))"
	@echo "  TOKENIZER_PATH   - Tokenizer model (default: $(TOKENIZER_PATH))"
	@echo "  BASE_MODEL       - Base model (default: $(BASE_MODEL))"
	@echo ""

# =============================================================================
# Setup
# =============================================================================

.PHONY: setup
setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.10; \
	fi
	@echo "Installing dependencies..."
	. $(VENV)/bin/activate && uv pip install -r requirements.txt
	@echo "Setup complete!"

# =============================================================================
# Vocabulary Swap (Quick Experiments)
# =============================================================================

# Source model for vocab swap
V2_MODEL := ../train_parakeet_tdt/models/v2.nemo
SWAP_TOKENIZER_DIR := ../common/tokenizers
SWAP_OUTPUT_NO_RESTORE := ./models/v2-swapped-vocab.nemo
SWAP_OUTPUT_RESTORED := ./models/v2-swapped-vocab-restored.nemo
TEST_AUDIO := ./data/sample01.mp3

.PHONY: swap-vocab swap-vocab-restore

swap-vocab: setup
	@echo "============================================================"
	@echo "Swap Vocabulary (WITHOUT restoring original weights)"
	@echo "============================================================"
	@if [ ! -f "$(V2_MODEL)" ]; then \
		echo "ERROR: Model not found at $(V2_MODEL)"; \
		exit 1; \
	fi
	@mkdir -p models
	. $(VENV)/bin/activate && $(PYTHON) src/swap_vocabulary.py \
		--model-path $(V2_MODEL) \
		--tokenizer-dir $(SWAP_TOKENIZER_DIR) \
		--output-path $(SWAP_OUTPUT_NO_RESTORE) \
		--transcribe $(TEST_AUDIO)
	@echo ""
	@echo "Model saved to: $(SWAP_OUTPUT_NO_RESTORE)"

swap-vocab-restore: setup
	@echo "============================================================"
	@echo "Swap Vocabulary (WITH restoring original weights)"
	@echo "============================================================"
	@if [ ! -f "$(V2_MODEL)" ]; then \
		echo "ERROR: Model not found at $(V2_MODEL)"; \
		exit 1; \
	fi
	@mkdir -p models
	. $(VENV)/bin/activate && $(PYTHON) src/swap_vocabulary.py \
		--model-path $(V2_MODEL) \
		--tokenizer-dir $(SWAP_TOKENIZER_DIR) \
		--output-path $(SWAP_OUTPUT_RESTORED) \
		--restore-weights \
		--transcribe $(TEST_AUDIO)
	@echo ""
	@echo "Model saved to: $(SWAP_OUTPUT_RESTORED)"

# =============================================================================
# Model Modification
# =============================================================================

.PHONY: modify-model modify-model-dry-run

modify-model: setup
	@echo "============================================================"
	@echo "Modifying Parakeet TDT for Multilingual Tokenizer"
	@echo "============================================================"
	@if [ ! -f "$(TOKENIZER_PATH)" ]; then \
		echo "ERROR: Tokenizer not found at $(TOKENIZER_PATH)"; \
		echo "Run 'cd .. && make train-tokenizer' first."; \
		exit 1; \
	fi
	@mkdir -p models
	. $(VENV)/bin/activate && $(PYTHON) src/modify_architecture.py \
		--base-model $(BASE_MODEL) \
		--tokenizer-path $(TOKENIZER_PATH) \
		--output-path $(MODIFIED_MODEL)
	@echo ""
	@echo "Modified model saved to: $(MODIFIED_MODEL)"
	@echo "Next step: make train"

modify-model-dry-run: setup
	@echo "Dry run: checking what would change..."
	. $(VENV)/bin/activate && $(PYTHON) src/modify_architecture.py \
		--base-model $(BASE_MODEL) \
		--tokenizer-path $(TOKENIZER_PATH) \
		--output-path $(MODIFIED_MODEL) \
		--dry-run

# =============================================================================
# Training
# =============================================================================

.PHONY: train train-debug

train: setup
	@echo "============================================================"
	@echo "Starting Multilingual Training"
	@echo "============================================================"
	@if [ ! -f "$(MODIFIED_MODEL)" ]; then \
		echo "ERROR: Modified model not found at $(MODIFIED_MODEL)"; \
		echo "Run 'make modify-model' first."; \
		exit 1; \
	fi
	@if [ ! -f "../training_data/multilingual/manifests/train_manifest.json" ]; then \
		echo "ERROR: Multilingual manifests not found."; \
		echo "Run 'cd .. && make prepare-multilingual' first."; \
		exit 1; \
	fi
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	. $(VENV)/bin/activate && $(PYTHON) src/train.py --config $(CONFIG)

train-debug: setup
	@echo "Training with debug logging..."
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	NEMO_DEBUG=1 \
	. $(VENV)/bin/activate && $(PYTHON) src/train.py --config $(CONFIG)

train-overfit: setup
	@echo "============================================================"
	@echo "Overfit Tiny Subset (Sanity Check)"
	@echo "============================================================"
	@if [ ! -f "$(MODIFIED_MODEL)" ]; then \
		echo "ERROR: Modified model not found at $(MODIFIED_MODEL)"; \
		echo "Run 'make modify-model' first."; \
		exit 1; \
	fi
	@if [ ! -f "../training_data/multilingual/manifests/train_manifest.json" ]; then \
		echo "ERROR: Multilingual manifests not found."; \
		echo "Run 'cd .. && make prepare-multilingual' first."; \
		exit 1; \
	fi
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	. $(VENV)/bin/activate && $(PYTHON) src/train.py --config $(OVERFIT_CONFIG)

# =============================================================================
# Testing
# =============================================================================

.PHONY: test-tokenizer test-inference

test-tokenizer: setup
	@echo "Testing multilingual tokenizer..."
	@if [ ! -f "$(TOKENIZER_PATH)" ]; then \
		echo "ERROR: Tokenizer not found at $(TOKENIZER_PATH)"; \
		exit 1; \
	fi
	. $(VENV)/bin/activate && $(PYTHON) ../common/tokenizers/train_tokenizer.py \
		--validate-only $(TOKENIZER_PATH)

test-inference: setup
	@echo "Testing model inference..."
	@echo "Note: Requires trained model and validation manifests"
	. $(VENV)/bin/activate && $(PYTHON) -c " \
import torch \
import nemo.collections.asr as nemo_asr \
import json \
\
print('Loading model...') \
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$(MODIFIED_MODEL)') \
model.eval() \
model.cuda() \
\
print('Loading validation samples...') \
with open('../training_data/multilingual/manifests/val_manifest.json') as f: \
    samples = [json.loads(line) for line in f][:5] \
\
print('\\n--- Inference Results ---') \
for s in samples: \
    pred = model.transcribe([s['audio_filepath']])[0] \
    print(f'Reference: {s[\"text\"]}') \
    print(f'Predicted: {pred}') \
    print('---') \
"

# =============================================================================
# Utilities
# =============================================================================

.PHONY: tensorboard clean

tensorboard:
	@echo "Starting TensorBoard..."
	. $(VENV)/bin/activate && tensorboard --logdir outputs/

clean:
	@echo "Removing outputs..."
	rm -rf outputs/
	@echo "Removing modified models..."
	rm -f models/*.nemo
	@echo "Done."

# =============================================================================
# Full Workflow (convenience target)
# =============================================================================

.PHONY: all

all: modify-model train
	@echo ""
	@echo "============================================================"
	@echo "Multilingual training complete!"
	@echo "============================================================"
