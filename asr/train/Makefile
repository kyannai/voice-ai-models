# ASR Training Data Management
# =============================
# Centralized dataset preparation for ASR model training
#
# Usage:
#   make list-datasets         - List all available datasets with status
#   make prepare-all           - Prepare all datasets
#   make prepare-<dataset>     - Prepare a specific dataset
#
# See: make help

SHELL := /bin/bash
PYTHON := python3

# =============================================================================
# Configuration
# =============================================================================

# Directories
TRAINING_DATA_DIR := training_data
COMMON_DIR := common

# Dataset directories
MALAYSIAN_STT_DIR := $(TRAINING_DATA_DIR)/malaysian-stt
MALAYSIAN_STT_STAGE2_DIR := $(TRAINING_DATA_DIR)/malaysian-stt-stage2
CHINESE_MANDARIN_DIR := $(TRAINING_DATA_DIR)/chinese-mandarin
KESPEECH_DIR := $(TRAINING_DATA_DIR)/KeSpeech
SYNTHETIC_5K_DIR := $(TRAINING_DATA_DIR)/5k_v3

# Virtual environment
COMMON_VENV := $(COMMON_DIR)/.venv
COMMON_PYTHON := $(COMMON_VENV)/bin/python

# Stamp files (tracks prepared state)
STAMP_DIR := $(TRAINING_DATA_DIR)
STAMP_MALAYSIAN_STT := $(STAMP_DIR)/.malaysian-stt_prepared
STAMP_MALAYSIAN_STT_STAGE2 := $(STAMP_DIR)/.malaysian-stt-stage2_prepared
STAMP_CHINESE_MANDARIN := $(STAMP_DIR)/.chinese-mandarin_prepared
STAMP_KESPEECH := $(STAMP_DIR)/.kespeech_prepared
STAMP_SYNTHETIC_5K := $(STAMP_DIR)/.synthetic-5k_prepared

# Optional parameters
MAX_SAMPLES ?=
DATASET ?=

ifdef MAX_SAMPLES
MAX_SAMPLES_ARG := --max-samples $(MAX_SAMPLES)
else
MAX_SAMPLES_ARG :=
endif

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo "ASR Training Data Management"
	@echo "============================"
	@echo ""
	@echo "Dataset Management:"
	@echo "  make list-datasets           - List all datasets with status"
	@echo "  make prepare-all             - Prepare ALL datasets"
	@echo ""
	@echo "Individual Dataset Preparation (auto-downloads if needed):"
	@echo "  make prepare-malaysian-stt       - Malaysian STT (Stage 1, ~1000h, ASR preprocessed)"
	@echo "  make prepare-malaysian-stt-stage2 - Malaysian STT Stage 2 (~500h)"
	@echo "  make prepare-chinese-mandarin    - Chinese LiPS (~100h)"
	@echo "  make prepare-kespeech            - KeSpeech Mandarin (~1500h local)"
	@echo "  make prepare-synthetic-5k        - Synthetic Malay data (~1.5h)"
	@echo ""
	@echo "Multilingual Training Data (EN/MS/ZH):"
	@echo "  make prepare-kespeech-full             - Alias for prepare-kespeech"
	@echo "  make expand-ctc-tokenizer              - Expand parakeet-ctc-1.1b vocabulary"
	@echo "  make expand-tdt-tokenizer              - Expand parakeet-tdt-0.6b-v3 vocabulary"
	@echo "  make prepare-multilingual              - Combine manifests for multilingual training"
	@echo "  make prepare-multilingual-overfit      - Small manifests for overfit sanity check"
	@echo ""
	@echo "Tokenizer Testing:"
	@echo "  make test-tokenizer-languages          - Test if model supports cn/ms/en"
	@echo ""
	@echo "Training (use separate packages):"
	@echo "  cd train_parakeet_tdt && make train"
	@echo "  cd train_parakeet_tdt_with_new_tokenizer && make train"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean                   - Remove prepared data (keep raw)"
	@echo "  make clean-all               - Remove all data (including raw)"
	@echo "  make status                  - Show dataset status summary"
	@echo ""
	@echo "Parameters:"
	@echo "  MAX_SAMPLES            - Limit samples for testing (e.g., MAX_SAMPLES=1000)"
	@echo "  MANIFESTS              - Manifest specs: 'path:max path2:max2' (for expand-*-chinese)"
	@echo "  MAX_TOTAL_CHARS        - Optional total char limit across all manifests"
	@echo "  EXPANDED_CTC_OUTPUT    - Output path for expanded CTC model"
	@echo "  EXPANDED_TDT_OUTPUT    - Output path for expanded TDT model"
	@echo "  TEST_AUDIO             - Audio file to test before/after vocab expansion"
	@echo "  OVERFIT_SAMPLES        - Samples for overfit manifests (default: 100, same for train/val)"
	@echo "  MODEL                  - HuggingFace model or .nemo path (for test-tokenizer-languages)"
	@echo "  LANGUAGES              - Comma-separated languages to test (default: en,ms,cn)"
	@echo "  VERBOSE                - Set to 1 for detailed tokenizer test output"
	@echo ""
	@echo "Examples:"
	@echo "  make prepare-chinese-mandarin MAX_SAMPLES=1000  # Small test"
	@echo "  make prepare-malaysian-stt                      # Full dataset"
	@echo "  make expand-ctc-tokenizer MANIFESTS='chinese.json:5000 malay.json:3000'"
	@echo "  make expand-tdt-tokenizer MANIFESTS='chinese.json:5000 malay.json:3000' \\"
	@echo "       EXPANDED_TDT_OUTPUT=./models/parakeet-tdt-multilingual.nemo"
	@echo ""

# =============================================================================
# Internal: Environment Setup (auto-creates venv with uv)
# =============================================================================

.PHONY: _setup-common _setup-kespeech _setup-chinese-mandarin _setup-synthetic-5k

_setup-common:
	@if [ ! -d "$(COMMON_VENV)" ]; then \
		echo "Creating common venv..."; \
		cd $(COMMON_DIR) && uv venv --python 3.10; \
	fi
	@cd $(COMMON_DIR) && . .venv/bin/activate && uv pip install -q -r requirements.txt

_setup-kespeech:
	@if [ ! -d "$(KESPEECH_DIR)/.venv" ]; then \
		echo "Creating kespeech venv..."; \
		cd $(KESPEECH_DIR) && uv venv --python 3.10; \
	fi
	@cd $(KESPEECH_DIR) && . .venv/bin/activate && uv pip install -q -r requirements.txt

_setup-chinese-mandarin:
	@if [ ! -d "$(CHINESE_MANDARIN_DIR)/.venv" ]; then \
		echo "Creating chinese-mandarin venv..."; \
		cd $(CHINESE_MANDARIN_DIR) && uv venv --python 3.10; \
	fi
	@cd $(CHINESE_MANDARIN_DIR) && . .venv/bin/activate && uv pip install -q -r requirements.txt

_setup-synthetic-5k:
	@if [ ! -d "$(SYNTHETIC_5K_DIR)/.venv" ]; then \
		echo "Creating synthetic-5k venv..."; \
		cd $(SYNTHETIC_5K_DIR) && uv venv --python 3.10; \
	fi
	@if [ -f "$(SYNTHETIC_5K_DIR)/requirements.txt" ]; then \
		cd $(SYNTHETIC_5K_DIR) && . .venv/bin/activate && uv pip install -q -r requirements.txt; \
	else \
		cd $(SYNTHETIC_5K_DIR) && . .venv/bin/activate && uv pip install -q librosa tqdm; \
	fi

# =============================================================================
# Dataset Listing
# =============================================================================

.PHONY: list-datasets status

list-datasets: _setup-common
	@cd $(COMMON_DIR) && . .venv/bin/activate && $(PYTHON) datasets.py --training-data-dir ../$(TRAINING_DATA_DIR)

status: list-datasets

# =============================================================================
# Prepare All Datasets
# =============================================================================

.PHONY: prepare-all

prepare-all: prepare-malaysian-stt prepare-malaysian-stt-stage2 prepare-chinese-mandarin prepare-kespeech prepare-synthetic-5k
	@echo ""
	@echo "============================================================"
	@echo "All datasets prepared!"
	@echo "============================================================"
	@$(MAKE) list-datasets

# =============================================================================
# Malaysian STT (EN/MS only, with audio filtering)
# =============================================================================

.PHONY: prepare-malaysian-stt

prepare-malaysian-stt: _setup-common
	@echo "Preparing Malaysian STT dataset (EN/MS only)..."
	@if [ ! -d "$(MALAYSIAN_STT_DIR)/data/raw" ] || [ -z "$$(ls -A $(MALAYSIAN_STT_DIR)/data/raw 2>/dev/null)" ]; then \
		echo "Raw data not found. Downloading..."; \
		echo "Dataset: mesolitica/Malaysian-STT-Whisper"; \
		mkdir -p $(MALAYSIAN_STT_DIR)/data/raw; \
		huggingface-cli download --repo-type dataset \
			--local-dir '$(MALAYSIAN_STT_DIR)/data/raw' \
			--max-workers 20 \
			mesolitica/Malaysian-STT-Whisper; \
	fi
	@echo "Unzipping data files..."
	cd $(MALAYSIAN_STT_DIR)/data/raw && \
		for f in *.zip; do [ -f "$$f" ] && unzip -o -q "$$f" || true; done
	@echo "Creating manifests (EN/MS only, with ASR preprocessing)..."
	cd $(MALAYSIAN_STT_DIR) && ../../$(COMMON_DIR)/.venv/bin/python src/prepare_data.py \
		--data-dir data/raw/data \
		--audio-base-dir data/raw \
		--output-dir data/manifests \
		--train-split 0.95 \
		--include-languages en ms \
		--datasets extra imda malaysian_context_v2 science_context science_en science_ms \
		--filter-existing \
		--preprocess-asr \
		--clean-for-asr \
		--filter-invalid-chars \
		$(MAX_SAMPLES_ARG)
	@touch $(STAMP_MALAYSIAN_STT)
	@echo "Done: malaysian-stt prepared (EN/MS only, ASR preprocessed, cleaned)!"

# =============================================================================
# Malaysian STT Stage 2
# =============================================================================

.PHONY: prepare-malaysian-stt-stage2

prepare-malaysian-stt-stage2:
	@echo "Preparing Malaysian STT Stage 2 dataset..."
	@if [ ! -d "$(MALAYSIAN_STT_STAGE2_DIR)/data/raw" ] && [ ! -d "$(MALAYSIAN_STT_STAGE2_DIR)/data/data" ]; then \
		echo "Raw data not found. Downloading..."; \
		echo "Dataset: mesolitica/Malaysian-STT-Whisper-Stage2"; \
		mkdir -p $(MALAYSIAN_STT_STAGE2_DIR)/data/raw; \
		huggingface-cli download --repo-type dataset \
			--local-dir '$(MALAYSIAN_STT_STAGE2_DIR)/data/raw' \
			--max-workers 20 \
			mesolitica/Malaysian-STT-Whisper-Stage2; \
	fi
	cd $(MALAYSIAN_STT_STAGE2_DIR) && $(PYTHON) src/prepare_data.py \
		--data-dir data/data \
		--audio-base-dir data \
		--output-dir data/manifests \
		--train-split 0.95 \
		$(MAX_SAMPLES_ARG)
	@touch $(STAMP_MALAYSIAN_STT_STAGE2)
	@echo "Done: malaysian-stt-stage2 prepared!"

# =============================================================================
# Chinese Mandarin (Chinese-LiPS + ChildMandarin)
# =============================================================================

.PHONY: prepare-chinese-mandarin

prepare-chinese-mandarin: _setup-chinese-mandarin
	@echo "Preparing Chinese Mandarin datasets..."
	@if [ ! -d "$(CHINESE_MANDARIN_DIR)/data/raw" ] || [ -z "$$(ls -A $(CHINESE_MANDARIN_DIR)/data/raw 2>/dev/null)" ]; then \
		echo "Raw data not found. Downloading..."; \
		cd $(CHINESE_MANDARIN_DIR) && . .venv/bin/activate && $(MAKE) download; \
	fi
	cd $(CHINESE_MANDARIN_DIR) && . .venv/bin/activate && $(MAKE) prepare $(if $(MAX_SAMPLES_ARG),MAX_SAMPLES=$(MAX_SAMPLES),)
	@touch $(STAMP_CHINESE_MANDARIN)
	@echo "Done: chinese-mandarin prepared!"

# =============================================================================
# KeSpeech (Chinese Mandarin from HuggingFace)
# =============================================================================

.PHONY: prepare-kespeech

prepare-kespeech: _setup-kespeech
	@echo "Preparing KeSpeech dataset (Mandarin ~1500h)..."
	@echo "Processing local data from $(KESPEECH_DIR)"
	@mkdir -p $(KESPEECH_DIR)/data
	cd $(KESPEECH_DIR) && . .venv/bin/activate && $(PYTHON) src/prepare_kespeech.py \
		--data-dir . \
		--output-dir data \
		--subsets train_phase1 train_phase2 dev_phase1 dev_phase2 test \
		--normalize-text \
		$(MAX_SAMPLES_ARG)
	@touch $(STAMP_KESPEECH)
	@echo "Done: kespeech prepared!"

# =============================================================================
# Synthetic 5k (Malay)
# =============================================================================

.PHONY: prepare-synthetic-5k

prepare-synthetic-5k: _setup-synthetic-5k
	@echo "Preparing Synthetic 5k dataset..."
	@if [ ! -f "$(SYNTHETIC_5K_DIR)/data/synthesized.json" ]; then \
		echo "ERROR: Synthesized data not found at $(SYNTHETIC_5K_DIR)/data/synthesized.json"; \
		echo "This dataset requires running the synthesis pipeline first."; \
		exit 1; \
	fi
	cd $(SYNTHETIC_5K_DIR) && . .venv/bin/activate && $(PYTHON) src/prepare_synthetic_manifests.py \
		--input data/synthesized.json \
		--audio-base-dir data \
		--output-dir data/manifests \
		--train-split 0.9 \
		$(MAX_SAMPLES_ARG)
	@touch $(STAMP_SYNTHETIC_5K)
	@echo "Done: synthetic-5k prepared!"

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean clean-stamps clean-manifests clean-all

clean-stamps:
	@echo "Removing stamp files..."
	@rm -f $(STAMP_MALAYSIAN_STT) $(STAMP_MALAYSIAN_STT_STAGE2) $(STAMP_CHINESE_MANDARIN) $(STAMP_KESPEECH) $(STAMP_SYNTHETIC_5K)
	@echo "Done."

clean-manifests:
	@echo "Removing manifest files..."
	@rm -rf $(MALAYSIAN_STT_DIR)/data/manifests
	@rm -rf $(MALAYSIAN_STT_STAGE2_DIR)/data/manifests
	@rm -rf $(CHINESE_MANDARIN_DIR)/data/processed/manifests
	@rm -rf $(KESPEECH_DIR)/data/manifests
	@rm -rf $(SYNTHETIC_5K_DIR)/data/manifests
	@echo "Done."

clean: clean-stamps clean-manifests
	@echo "Cleaned prepared data (raw data preserved)."

clean-all: clean
	@echo "Removing all data (including raw downloads)..."
	@rm -rf $(MALAYSIAN_STT_DIR)/data
	@rm -rf $(MALAYSIAN_STT_STAGE2_DIR)/data
	cd $(CHINESE_MANDARIN_DIR) && $(MAKE) clean-all 2>/dev/null || true
	@echo "Done. All data removed."

# =============================================================================
# Multilingual Training Data Preparation
# =============================================================================

# Paths for multilingual data
MULTILINGUAL_DIR := $(TRAINING_DATA_DIR)/multilingual
MULTILINGUAL_OVERFIT_DIR := $(TRAINING_DATA_DIR)/multilingual-overfit

# Default manifest paths for multilingual targets
MS_MANIFEST ?= $(MALAYSIAN_STT_DIR)/data/manifests/train_manifest.json
ZH_MANIFEST ?= $(KESPEECH_DIR)/data/manifests/train_manifest.json
TOKENIZER_DIR := $(COMMON_DIR)/tokenizers
KESPEECH_FULL_DIR := $(TRAINING_DATA_DIR)/kespeech-full

.PHONY: prepare-kespeech-full prepare-multilingual prepare-multilingual-overfit

prepare-kespeech-full: prepare-kespeech
	@echo "prepare-kespeech-full is now an alias for prepare-kespeech"
	@echo "The KeSpeech data is processed from $(KESPEECH_DIR)"

# Default max Chinese characters for vocabulary expansion
MAX_CHINESE_CHARS ?= 5000

# =============================================================================
# Test Tokenizer Language Support
# =============================================================================

.PHONY: test-tokenizer-languages

# Test if a model's tokenizer supports cn, ms, en
# Usage: make test-tokenizer-languages MODEL=nvidia/parakeet-tdt-0.6b-v3
#        make test-tokenizer-languages MODEL=/path/to/model.nemo
MODEL ?=

test-tokenizer-languages: _setup-common
	@if [ -z "$(MODEL)" ]; then \
		echo ""; \
		echo "Test Tokenizer Language Support"; \
		echo "================================"; \
		echo ""; \
		echo "Usage:"; \
		echo "  make test-tokenizer-languages MODEL=nvidia/parakeet-tdt-0.6b-v3"; \
		echo "  make test-tokenizer-languages MODEL=/path/to/model.nemo"; \
		echo ""; \
		echo "Options:"; \
		echo "  MODEL            HuggingFace model name or path to .nemo file (required)"; \
		echo "  LANGUAGES        Comma-separated languages to test (default: en,ms,cn)"; \
		echo "  VERBOSE          Set to 1 for detailed output"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make test-tokenizer-languages MODEL=nvidia/parakeet-tdt-0.6b-v3"; \
		echo "  make test-tokenizer-languages MODEL=./models/my-model.nemo VERBOSE=1"; \
		echo "  make test-tokenizer-languages MODEL=nvidia/parakeet-tdt-0.6b-v3 LANGUAGES=en,cn"; \
		echo ""; \
		exit 1; \
	fi
	@if [ ! -d "$(TOKENIZER_DIR)/.venv" ]; then \
		echo "Creating tokenizer venv..."; \
		cd $(TOKENIZER_DIR) && uv venv --python 3.10; \
	fi
	@cd $(TOKENIZER_DIR) && . .venv/bin/activate && uv pip install -q sentencepiece huggingface_hub
	cd $(TOKENIZER_DIR) && . .venv/bin/activate && $(PYTHON) test_tokenizer_languages.py \
		"$(MODEL)" \
		$(if $(filter 1,$(VERBOSE)),--verbose,) \
		$(if $(LANGUAGES),--languages $(LANGUAGES),)

# =============================================================================
# Expand CTC Model Vocabulary (for parakeet-ctc-1.1b)
# =============================================================================
# Supports multiple manifests with individual token limits.
# Format: MANIFESTS="path1:max1 path2:max2 ..."

.PHONY: expand-ctc-tokenizer

# Default output path for expanded CTC model
EXPANDED_CTC_OUTPUT ?= $(TOKENIZER_DIR)/parakeet-ctc-1.1b-expanded.nemo

expand-ctc-tokenizer: _setup-common
	@echo "Expanding nvidia/parakeet-ctc-1.1b vocabulary..."
	@mkdir -p $(TOKENIZER_DIR)
	@if [ ! -d "$(TOKENIZER_DIR)/.venv" ]; then \
		echo "Creating tokenizer venv..."; \
		cd $(TOKENIZER_DIR) && uv venv --python 3.10; \
	fi
	@cd $(TOKENIZER_DIR) && . .venv/bin/activate && uv pip install -q "nemo_toolkit[asr]" sentencepiece tqdm pyyaml
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
	cd $(TOKENIZER_DIR) && . .venv/bin/activate && $(PYTHON) expand_ctc_tokenizer.py \
		--manifests $(MANIFESTS) \
		--output "$(CURDIR)/$(EXPANDED_CTC_OUTPUT)" \
		$(if $(MAX_TOTAL_CHARS),--max-total-chars $(MAX_TOTAL_CHARS),) \
		$(if $(TEST_AUDIO),--test-audio "$(TEST_AUDIO)",)
	@echo ""
	@echo "Done: expanded CTC model saved to $(EXPANDED_CTC_OUTPUT)"
	@echo "Note: Fine-tune this model on target language data to train the new embeddings."

# ===== TDT Model Vocabulary Expansion =====
# Supports multiple manifests with individual token limits.
# Format: MANIFESTS="path1:max1 path2:max2 ..."

.PHONY: expand-tdt-tokenizer

# Default output path for expanded TDT model
EXPANDED_TDT_OUTPUT ?= $(TOKENIZER_DIR)/parakeet-tdt-0.6b-v3-expanded.nemo

expand-tdt-tokenizer: _setup-common
	@echo "Expanding nvidia/parakeet-tdt-0.6b-v3 vocabulary..."
	@mkdir -p $(TOKENIZER_DIR)
	@if [ ! -d "$(TOKENIZER_DIR)/.venv" ]; then \
		echo "Creating tokenizer venv..."; \
		cd $(TOKENIZER_DIR) && uv venv --python 3.10; \
	fi
	@cd $(TOKENIZER_DIR) && . .venv/bin/activate && uv pip install -q "nemo_toolkit[asr]" sentencepiece tqdm pyyaml librosa
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
	cd $(TOKENIZER_DIR) && . .venv/bin/activate && $(PYTHON) expand_tdt_tokenizer.py \
		--manifests $(MANIFESTS) \
		--output "$(CURDIR)/$(EXPANDED_TDT_OUTPUT)" \
		$(if $(MAX_TOTAL_CHARS),--max-total-chars $(MAX_TOTAL_CHARS),) \
		$(if $(TEST_AUDIO),--test-audio "$(TEST_AUDIO)",)
	@echo ""
	@echo "Done: expanded TDT model saved to $(EXPANDED_TDT_OUTPUT)"
	@echo "Note: Fine-tune this model on target language data to train the new embeddings."

prepare-multilingual:
	@echo "Creating combined multilingual manifests..."
	@if [ ! -f "$(MS_MANIFEST)" ]; then \
		echo "Error: Malaysian manifest not found: $(MS_MANIFEST)"; \
		echo "Run 'make prepare-malaysian-stt' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(ZH_MANIFEST)" ]; then \
		echo "Error: Chinese manifest not found: $(ZH_MANIFEST)"; \
		echo "Run 'make prepare-kespeech' first."; \
		exit 1; \
	fi
	@mkdir -p $(MULTILINGUAL_DIR)/manifests
	@echo "Combining train manifests..."
	@cat $(MS_MANIFEST) $(ZH_MANIFEST) > $(MULTILINGUAL_DIR)/manifests/train_manifest.json
	@echo "Combining val manifests..."
	@MS_VAL=$$(dirname $(MS_MANIFEST))/val_manifest.json; \
	 ZH_VAL=$$(dirname $(ZH_MANIFEST))/val_manifest.json; \
	 cat "$$MS_VAL" "$$ZH_VAL" > $(MULTILINGUAL_DIR)/manifests/val_manifest.json
	@echo "Shuffling combined manifests..."
	@shuf $(MULTILINGUAL_DIR)/manifests/train_manifest.json -o $(MULTILINGUAL_DIR)/manifests/train_manifest.json.tmp && \
		mv $(MULTILINGUAL_DIR)/manifests/train_manifest.json.tmp $(MULTILINGUAL_DIR)/manifests/train_manifest.json
	@echo ""
	@echo "Combined manifest stats:"
	@echo "  Train: $$(wc -l < $(MULTILINGUAL_DIR)/manifests/train_manifest.json) samples"
	@echo "  Val: $$(wc -l < $(MULTILINGUAL_DIR)/manifests/val_manifest.json) samples"
	@echo ""
	@echo "Done: multilingual manifests at $(MULTILINGUAL_DIR)/manifests/"

OVERFIT_SAMPLES ?= 100

# Python script for balanced sampling (half from each manifest)
define OVERFIT_SAMPLE_SCRIPT
import random
from pathlib import Path
import sys

ms_manifest = Path(sys.argv[1])
zh_manifest = Path(sys.argv[2])
out_path = Path(sys.argv[3])
target = int(sys.argv[4])

random.seed(42)

def reservoir_sample(path, k):
    """Reservoir sample k lines from a single file."""
    sample = []
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if len(sample) < k:
                sample.append(line)
            else:
                j = random.randrange(n)
                if j < k:
                    sample[j] = line
    return sample

# Sample half from each manifest
half = target // 2
ms_lines = reservoir_sample(ms_manifest, half)
zh_lines = reservoir_sample(zh_manifest, target - half)  # handles odd numbers

lines = ms_lines + zh_lines
random.shuffle(lines)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("".join(lines), encoding="utf-8")
print(f"  Wrote {len(lines)} samples ({len(ms_lines)} MS + {len(zh_lines)} ZH) to {out_path}")
endef
export OVERFIT_SAMPLE_SCRIPT

prepare-multilingual-overfit:
	@echo "Creating small multilingual manifests for overfit..."
	@echo "Note: Train and val use IDENTICAL samples for true overfit testing."
	@if [ ! -f "$(MS_MANIFEST)" ]; then \
		echo "Error: Malaysian manifest not found: $(MS_MANIFEST)"; \
		echo "Run 'make prepare-malaysian-stt' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(ZH_MANIFEST)" ]; then \
		echo "Error: Chinese manifest not found: $(ZH_MANIFEST)"; \
		echo "Run 'make prepare-kespeech' first."; \
		exit 1; \
	fi
	@mkdir -p $(MULTILINGUAL_OVERFIT_DIR)/manifests
	@echo "Building overfit manifest ($(OVERFIT_SAMPLES) samples)..."
	@echo "$$OVERFIT_SAMPLE_SCRIPT" | $(PYTHON) - \
		"$(MS_MANIFEST)" \
		"$(ZH_MANIFEST)" \
		"$(MULTILINGUAL_OVERFIT_DIR)/manifests/train_manifest.json" \
		"$(OVERFIT_SAMPLES)"
	@echo "Copying train manifest to val (identical samples for overfit)..."
	@cp $(MULTILINGUAL_OVERFIT_DIR)/manifests/train_manifest.json \
		$(MULTILINGUAL_OVERFIT_DIR)/manifests/val_manifest.json
	@echo ""
	@echo "Overfit manifest stats:"
	@echo "  Train: $$(wc -l < $(MULTILINGUAL_OVERFIT_DIR)/manifests/train_manifest.json) samples"
	@echo "  Val: $$(wc -l < $(MULTILINGUAL_OVERFIT_DIR)/manifests/val_manifest.json) samples (IDENTICAL to train)"
	@echo ""
	@echo "Done: overfit manifests at $(MULTILINGUAL_OVERFIT_DIR)/manifests/"

# =============================================================================
# Info
# =============================================================================

.PHONY: info

info:
	@echo "Training Data Overview"
	@echo "======================"
	@echo ""
	@echo "Malaysian (Malay):"
	@echo "  malaysian-stt:        ~1000h from mesolitica/Malaysian-STT-Whisper"
	@echo "  malaysian-stt-stage2: ~500h  additional training data"
	@echo "  synthetic-5k:         ~1.5h  ElevenLabs synthesized (names/numbers)"
	@echo ""
	@echo "Chinese (Mandarin):"
	@echo "  chinese-mandarin:     ~100h  from BAAI/Chinese-LiPS"
	@echo "  kespeech:             ~1500h from OpenSLR (local full dataset)"
	@echo ""
	@echo "Directory Structure:"
	@echo "  training_data/"
	@echo "    ├── malaysian-stt/"
	@echo "    ├── malaysian-stt-stage2/"
	@echo "    ├── chinese-mandarin/"
	@echo "    ├── kespeech/"
	@echo "    └── 5k_v3/  (synthetic)"
	@echo ""
	@echo "Manifest Output:"
	@echo "  Each dataset produces train_manifest.json and val_manifest.json"
	@echo "  in NeMo format: {\"audio_filepath\": ..., \"text\": ..., \"duration\": ...}"
