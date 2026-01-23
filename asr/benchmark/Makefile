# ASR Benchmark Makefile
# Usage: make <target>

SHELL := /bin/bash
PYTHON := python

# Default target
.PHONY: help
help:
	@echo "ASR Benchmark - Available targets:"
	@echo ""
	@echo "Dataset Preparation:"
	@echo "  make list-datasets         - List all available test datasets"
	@echo "  make prepare-test-data-all - Download and prepare ALL datasets"
	@echo "  make prepare-ytl-testsets  - Prepare YTL Malay testsets (from tar)"
	@echo "  make prepare-kespeech      - Download and prepare KeSpeech (zh)"
	@echo "  make prepare-aishell2      - Prepare AISHELL-2 (zh) (Not available yet)"
	@echo "  make prepare-childmandarin - Prepare ChildMandarin (zh)"
	@echo "  make prepare-chinese-lips  - Prepare Chinese-LiPS (zh)"
	@echo ""
	@echo "Run Benchmarks:"
	@echo "  make run-all               - Run all benchmarks on all datasets"
	@echo "  make run-api               - Run YTL API benchmark"
	@echo "  make run-whisper           - Run both Whisper benchmarks"
	@echo "  make run-whisper-faster    - Run Faster Whisper benchmark"
	@echo "  make run-whisper-mesolitica - Run Mesolitica Whisper benchmark"
	@echo "  make run-parakeet          - Run Parakeet benchmark"
	@echo ""
	@echo "Single File Tests:"
	@echo "  make run-api-single AUDIO=/path/to/audio.wav"
	@echo "  make run-parakeet-single AUDIO=/path/to/audio.wav"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean                 - Clean output directories"
	@echo "  make clean-venvs           - Remove virtual environments"
	@echo "  make clean-test-data       - Remove prepared test data"
	@echo ""
	@echo "Parameters:"
	@echo "  MODEL          - API model name (default: bukit-tinggi-v2)"
	@echo "  PARAKEET_MODEL - Parakeet model (default: nvidia/parakeet-tdt-0.6b-v2)"
	@echo "  ENV            - API environment: staging|production (default: production)"
	@echo "  DATASET        - Run specific dataset only (see 'make list-datasets')"
	@echo "  MAX_SAMPLES    - Limit samples for prepare/run (optional)"
	@echo "  BATCH_SIZE     - Batch size for Parakeet (default: 16)"
	@echo "  WORKERS        - Number of workers (default: 1)"
	@echo ""
	@echo "Examples:"
	@echo "  # Prepare datasets"
	@echo "  make prepare-test-data-all"
	@echo "  make prepare-kespeech MAX_SAMPLES=1000"
	@echo ""
	@echo "  # Run benchmarks"
	@echo "  make run-api MODEL=bukit-tinggi-v2 DATASET=fleurs_test"
	@echo "  make run-api MODEL=ilmu-preview-asr DATASET=kespeech MAX_SAMPLES=100"
	@echo "  make run-parakeet PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2 DATASET=kespeech"
	@echo "  make run-whisper-faster DATASET=malay_scripted"
	@echo ""

# =============================================================================
# Configuration
# =============================================================================

# Configurable parameters
MODEL ?= bukit-tinggi-v2
PARAKEET_MODEL ?= nvidia/parakeet-tdt-0.6b-v3
ENV ?= production
BATCH_SIZE ?= 16
WORKERS ?= 1
DATASET ?=
MAX_SAMPLES ?=
AISHELL2_DATA_DIR ?=

# Build optional arguments
ifdef DATASET
DATASET_ARG := --dataset $(DATASET)
else
DATASET_ARG :=
endif

ifdef MAX_SAMPLES
MAX_SAMPLES_ARG := --max-samples $(MAX_SAMPLES)
else
MAX_SAMPLES_ARG :=
endif

ifdef AISHELL2_DATA_DIR
AISHELL2_DATA_ARG := --data-dir $(AISHELL2_DATA_DIR)
else
AISHELL2_DATA_ARG :=
endif

# Paths
TEST_DATA_DIR := test_data
TEST_DATA_VENV := $(TEST_DATA_DIR)/.venv
TEST_DATA_PYTHON := $(TEST_DATA_VENV)/bin/python
TEST_DATA_TAR := $(TEST_DATA_DIR)/YTL_testsets.tar

# Stamp files to track prepared state
STAMP_YTL := $(TEST_DATA_DIR)/.ytl_testsets_prepared
STAMP_KESPEECH := $(TEST_DATA_DIR)/.kespeech_prepared
STAMP_AISHELL2 := $(TEST_DATA_DIR)/.aishell2_prepared

# =============================================================================
# Internal: Environment Setup (called automatically by other targets)
# These are PHONY so they always run and check dependencies
# =============================================================================

.PHONY: _setup-common _setup-api _setup-whisper _setup-parakeet _setup-test-data

_setup-common:
	@if [ ! -d "common/.venv" ]; then echo "Creating common venv..."; cd common && uv venv --python 3.10; fi
	@cd common && . .venv/bin/activate && uv pip install -q -r requirements.txt

_setup-api: _setup-common
	@if [ ! -d "api/.venv" ]; then echo "Creating api venv..."; cd api && uv venv --python 3.10; fi
	@cd api && . .venv/bin/activate && uv pip install -q -r requirements.txt -r ../common/requirements.txt

_setup-whisper: _setup-common
	@if [ ! -d "whisper/.venv" ]; then echo "Creating whisper venv..."; cd whisper && uv venv --python 3.10; fi
	@cd whisper && . .venv/bin/activate && uv pip install -q -r requirements.txt -r ../common/requirements.txt

_setup-parakeet: _setup-common
	@if [ ! -d "parakeet/.venv" ]; then echo "Creating parakeet venv..."; cd parakeet && uv venv --python 3.10; fi
	@cd parakeet && . .venv/bin/activate && uv pip install -q -r requirements.txt -r ../common/requirements.txt

_setup-test-data:
	@if [ ! -d "$(TEST_DATA_VENV)" ]; then echo "Creating test_data venv..."; cd $(TEST_DATA_DIR) && uv venv --python 3.10; fi
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && uv pip install -q -r requirements.txt

# =============================================================================
# Dataset Preparation
# =============================================================================

.PHONY: list-datasets
list-datasets: _setup-common
	@cd common && . .venv/bin/activate && $(PYTHON) datasets.py

.PHONY: prepare-test-data-all
prepare-test-data-all: prepare-ytl-testsets prepare-kespeech prepare-aishell2 prepare-childmandarin prepare-chinese-lips
	@echo ""
	@echo "============================================================"
	@echo "All datasets prepared!"
	@echo "============================================================"
	@$(MAKE) list-datasets

# YTL Testsets (Malay: fleurs_test, malay_conversational, malay_scripted)
.PHONY: prepare-ytl-testsets
prepare-ytl-testsets: _setup-test-data
	@echo "Preparing YTL testsets (Malay)..."
	@if [ ! -f "$(TEST_DATA_TAR)" ]; then \
		echo "ERROR: $(TEST_DATA_TAR) not found."; \
		echo "Please download YTL_testsets.tar and place it in $(TEST_DATA_DIR)/"; \
		exit 1; \
	fi
	@tar -xf $(TEST_DATA_TAR) -C $(TEST_DATA_DIR)
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_ytl_testsets.py
	@touch $(STAMP_YTL)
	@echo "Done: fleurs_test, malay_conversational, malay_scripted"

# Backward compatibility alias
.PHONY: prepare-test-data
prepare-test-data: prepare-ytl-testsets

# KeSpeech (Chinese Mandarin from HuggingFace)
.PHONY: prepare-kespeech
prepare-kespeech: _setup-test-data
	@echo "Preparing KeSpeech dataset (Chinese)..."
	@echo "Downloading from HuggingFace: TwinkStart/KeSpeech"
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_kespeech.py \
		--output-dir kespeech $(MAX_SAMPLES_ARG)
	@touch $(STAMP_KESPEECH)
	@echo "Done: kespeech"

# AISHELL-2 (Chinese Mandarin)
.PHONY: prepare-aishell2
prepare-aishell2: _setup-test-data
	@echo "Preparing AISHELL-2 dataset (Chinese)..."
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_aishell2.py \
		--output-dir aishell2 $(AISHELL2_DATA_ARG) $(MAX_SAMPLES_ARG)
	@touch $(STAMP_AISHELL2)
	@echo "Done: aishell2"

# ChildMandarin (Chinese Mandarin - Children aged 3-5, from HuggingFace)
.PHONY: prepare-childmandarin
prepare-childmandarin: _setup-test-data
	@echo "Preparing ChildMandarin dataset (Chinese - Children)..."
	@echo "Note: Requires HF_TOKEN and accepting terms at https://huggingface.co/datasets/BAAI/ChildMandarin"
	@echo "Downloading from HuggingFace: BAAI/ChildMandarin"
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_childmandarin.py \
		--output-dir childmandarin $(MAX_SAMPLES_ARG)
	@echo "Done: childmandarin"

# Chinese-LiPS (Chinese Mandarin, from HuggingFace)
.PHONY: prepare-chinese-lips
prepare-chinese-lips: _setup-test-data
	@echo "Preparing Chinese-LiPS dataset (Chinese)..."
	@echo "Downloading from HuggingFace: BAAI/Chinese-LiPS"
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_chinese_lips.py \
		--output-dir chinese_lips $(MAX_SAMPLES_ARG)
	@echo "Done: chinese_lips"

# =============================================================================
# Run Benchmarks
# =============================================================================

.PHONY: run-all
run-all: run-api run-whisper run-parakeet
	@echo "All benchmarks complete."

.PHONY: run-api
run-api: _setup-api
	@echo "Running YTL API benchmark..."
	@echo "  Model: $(MODEL)"
	@echo "  Environment: $(ENV)"
ifdef DATASET
	@echo "  Dataset: $(DATASET)"
	@rm -rf api/ytl_labs/$(DATASET)
else
	@rm -rf api/ytl_labs
endif
ifdef MAX_SAMPLES
	@echo "  Max samples: $(MAX_SAMPLES)"
endif
	@cd api && . .venv/bin/activate && $(PYTHON) ytl_api_test.py --model $(MODEL) --env $(ENV) $(DATASET_ARG) $(MAX_SAMPLES_ARG)

.PHONY: run-api-single
run-api-single: _setup-api
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make run-api-single AUDIO=/path/to/audio.wav"; exit 1; fi
	@cd api && . .venv/bin/activate && $(PYTHON) ytl_api_test_single.py -a $(AUDIO) -m $(MODEL) --env $(ENV)

.PHONY: run-whisper
run-whisper: run-whisper-faster run-whisper-mesolitica
	@echo "All Whisper benchmarks complete."

.PHONY: run-whisper-faster
run-whisper-faster: _setup-whisper
	@echo "Running Faster Whisper benchmark..."
ifdef DATASET
	@echo "  Dataset: $(DATASET)"
endif
	@cd whisper && . .venv/bin/activate && $(PYTHON) ytl_faster_whisper_test.py $(DATASET_ARG)

.PHONY: run-whisper-mesolitica
run-whisper-mesolitica: _setup-whisper
	@echo "Running Mesolitica Whisper benchmark..."
ifdef DATASET
	@echo "  Dataset: $(DATASET)"
endif
	@cd whisper && . .venv/bin/activate && $(PYTHON) ytl_mesolitica_whisper_test.py $(DATASET_ARG)

.PHONY: run-parakeet
run-parakeet: _setup-parakeet
	@echo "Running Parakeet benchmark..."
	@echo "  Model: $(PARAKEET_MODEL)"
	@echo "  Batch size: $(BATCH_SIZE)"
	@echo "  Workers: $(WORKERS)"
ifdef DATASET
	@echo "  Dataset: $(DATASET)"
	@rm -rf parakeet/ytl_parakeet/$(DATASET)
else
	@rm -rf parakeet/ytl_parakeet
endif
ifdef MAX_SAMPLES
	@echo "  Limited to $(MAX_SAMPLES) samples"
endif
	$(eval PARAKEET_MODEL_ABS := $(shell if echo "$(PARAKEET_MODEL)" | grep -q '\.nemo$$' && [ -f "$(PARAKEET_MODEL)" ]; then realpath "$(PARAKEET_MODEL)"; else echo "$(PARAKEET_MODEL)"; fi))
	@cd parakeet && . .venv/bin/activate && $(PYTHON) ytl_parakeet_test.py -m $(PARAKEET_MODEL_ABS) -b $(BATCH_SIZE) -w $(WORKERS) $(DATASET_ARG) $(MAX_SAMPLES_ARG)

.PHONY: run-parakeet-single
run-parakeet-single: _setup-parakeet
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make run-parakeet-single AUDIO=/path/to/audio.wav"; exit 1; fi
	$(eval PARAKEET_MODEL_ABS := $(shell if echo "$(PARAKEET_MODEL)" | grep -q '\.nemo$$' && [ -f "$(PARAKEET_MODEL)" ]; then realpath "$(PARAKEET_MODEL)"; else echo "$(PARAKEET_MODEL)"; fi))
	@cd parakeet && . .venv/bin/activate && $(PYTHON) ytl_parakeet_test_single.py -a $(AUDIO) -m $(PARAKEET_MODEL_ABS)

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	@echo "Cleaning output directories..."
	@rm -rf api/ytl_labs whisper/ytl_labs whisper/faster_whisper parakeet/ytl_parakeet
	@echo "Done."

.PHONY: clean-venvs
clean-venvs:
	@echo "Removing virtual environments..."
	@rm -rf api/.venv whisper/.venv parakeet/.venv common/.venv $(TEST_DATA_VENV)
	@echo "Done."

.PHONY: clean-test-data
clean-test-data:
	@echo "Removing prepared test data..."
	@rm -rf $(TEST_DATA_DIR)/YTL_testsets $(TEST_DATA_DIR)/kespeech $(TEST_DATA_DIR)/aishell2 \
		$(TEST_DATA_DIR)/childmandarin $(TEST_DATA_DIR)/chinese_lips
	@rm -f $(STAMP_YTL) $(STAMP_KESPEECH) $(STAMP_AISHELL2)
	@echo "Done."
