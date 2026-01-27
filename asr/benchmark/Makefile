# ASR Benchmark Makefile
# Usage: make <target>

SHELL := /bin/bash
PYTHON := python

# =============================================================================
# Configuration (Default values)
# =============================================================================

MODEL ?= bukit-tinggi-v2
PARAKEET_MODEL ?= nvidia/parakeet-tdt-0.6b-v3
ENV ?= production
BATCH_SIZE ?= 16
WORKERS ?= 1
DATASET ?=
MAX_SAMPLES ?=

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

# Paths
TEST_DATA_DIR := test_data
TEST_DATA_VENV := $(TEST_DATA_DIR)/.venv
TEST_DATA_PYTHON := $(TEST_DATA_VENV)/bin/python
TEST_DATA_TAR := $(TEST_DATA_DIR)/YTL_testsets.tar

# Stamp files to track prepared state
STAMP_YTL := $(TEST_DATA_DIR)/.ytl_testsets_prepared
STAMP_KESPEECH := $(TEST_DATA_DIR)/.kespeech_prepared

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo "ASR Benchmark Makefile"
	@echo "======================"
	@echo ""
	@echo "==============================================================================="
	@echo "DATASET PREPARATION"
	@echo "==============================================================================="
	@echo ""
	@echo "list-datasets"
	@echo "  List all available test datasets."
	@echo "  Example: make list-datasets"
	@echo ""
	@echo "prepare-test-data-all"
	@echo "  Download and prepare ALL datasets."
	@echo "  Parameters:"
	@echo "    MAX_SAMPLES - Limit samples per dataset (optional)"
	@echo "  Examples:"
	@echo "    make prepare-test-data-all"
	@echo "    make prepare-test-data-all MAX_SAMPLES=500"
	@echo ""
	@echo "prepare-ytl-testsets"
	@echo "  Prepare YTL Malay testsets (fleurs_test, malay_conversational, malay_scripted)."
	@echo "  Requires: YTL_testsets.tar in test_data/"
	@echo "  Example: make prepare-ytl-testsets"
	@echo ""
	@echo "prepare-kespeech"
	@echo "  Download and prepare KeSpeech dataset (Chinese Mandarin)."
	@echo "  Parameters:"
	@echo "    MAX_SAMPLES - Limit number of samples (optional)"
	@echo "  Examples:"
	@echo "    make prepare-kespeech"
	@echo "    make prepare-kespeech MAX_SAMPLES=1000"
	@echo ""
	@echo "prepare-chinese-lips"
	@echo "  Download Chinese-LiPS dataset (Chinese Mandarin)."
	@echo "  Parameters:"
	@echo "    MAX_SAMPLES - Limit number of samples (optional)"
	@echo "  Examples:"
	@echo "    make prepare-chinese-lips"
	@echo "    make prepare-chinese-lips MAX_SAMPLES=300"
	@echo ""
	@echo "prepare-supa"
	@echo "  Prepare SUPA banking voice assistant dataset (Malay/English)."
	@echo "  Requires: supa/BenchmarkPOC_ASR_20251223/ in test_data/"
	@echo "  Example: make prepare-supa"
	@echo ""
	@echo "==============================================================================="
	@echo "RUN BENCHMARKS"
	@echo "==============================================================================="
	@echo ""
	@echo "run-all"
	@echo "  Run ALL models on a dataset with WER results."
	@echo "  Models: faster-whisper, whisper-openai, whisper-mesolitica, whisper-finetuned,"
	@echo "          parakeet-v3, parakeet-5k, parakeet-multilingual, parakeet-malay"
	@echo "  Parameters:"
	@echo "    DATASET     - Dataset to run (required)"
	@echo "    MAX_SAMPLES - Limit samples (optional)"
	@echo "  Examples:"
	@echo "    make run-all DATASET=supa"
	@echo "    make run-all DATASET=fleurs_test MAX_SAMPLES=100"
	@echo "  Output: benchmark_results/<timestamp>/wer_results.csv"
	@echo ""
	@echo "run-api"
	@echo "  Run YTL API benchmark."
	@echo "  Parameters:"
	@echo "    MODEL       - API model (default: bukit-tinggi-v2)"
	@echo "                  Options: bukit-tinggi-v2, ilmu-preview-asr"
	@echo "    ENV         - Environment (default: production)"
	@echo "                  Options: staging, production"
	@echo "    DATASET     - Specific dataset to run (optional)"
	@echo "    MAX_SAMPLES - Limit samples (optional)"
	@echo "  Examples:"
	@echo "    make run-api"
	@echo "    make run-api MODEL=bukit-tinggi-v2 DATASET=fleurs_test"
	@echo "    make run-api MODEL=ilmu-preview-asr DATASET=kespeech MAX_SAMPLES=100"
	@echo "    make run-api ENV=staging DATASET=malay_scripted"
	@echo ""
	@echo "run-api-single"
	@echo "  Run YTL API on a single audio file."
	@echo "  Parameters:"
	@echo "    AUDIO - Path to audio file (required)"
	@echo "    MODEL - API model (default: bukit-tinggi-v2)"
	@echo "    ENV   - Environment (default: production)"
	@echo "  Examples:"
	@echo "    make run-api-single AUDIO=/path/to/audio.wav"
	@echo "    make run-api-single AUDIO=test.mp3 MODEL=ilmu-preview-asr"
	@echo "    make run-api-single AUDIO=sample.wav ENV=staging"
	@echo ""
	@echo "run-whisper"
	@echo "  Run both Whisper benchmarks (Faster Whisper + Mesolitica)."
	@echo "  Parameters:"
	@echo "    DATASET - Specific dataset to run (optional)"
	@echo "  Examples:"
	@echo "    make run-whisper"
	@echo "    make run-whisper DATASET=malay_scripted"
	@echo ""
	@echo "run-whisper-faster"
	@echo "  Run Faster Whisper benchmark."
	@echo "  Parameters:"
	@echo "    DATASET - Specific dataset to run (optional)"
	@echo "  Examples:"
	@echo "    make run-whisper-faster"
	@echo "    make run-whisper-faster DATASET=fleurs_test"
	@echo ""
	@echo "run-whisper-mesolitica"
	@echo "  Run Mesolitica Whisper benchmark."
	@echo "  Parameters:"
	@echo "    DATASET - Specific dataset to run (optional)"
	@echo "  Examples:"
	@echo "    make run-whisper-mesolitica"
	@echo "    make run-whisper-mesolitica DATASET=malay_scripted"
	@echo ""
	@echo "run-parakeet"
	@echo "  Run NVIDIA Parakeet ASR benchmark."
	@echo "  Parameters:"
	@echo "    PARAKEET_MODEL - Model name or .nemo path (default: nvidia/parakeet-tdt-0.6b-v3)"
	@echo "    BATCH_SIZE     - Batch size (default: 16)"
	@echo "    WORKERS        - Data loader workers (default: 1)"
	@echo "    DATASET        - Specific dataset to run (optional)"
	@echo "    MAX_SAMPLES    - Limit samples (optional)"
	@echo "  Examples:"
	@echo "    make run-parakeet"
	@echo "    make run-parakeet PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2"
	@echo "    make run-parakeet DATASET=kespeech BATCH_SIZE=32"
	@echo "    make run-parakeet PARAKEET_MODEL=./models/custom.nemo MAX_SAMPLES=100"
	@echo ""
	@echo "run-parakeet-single"
	@echo "  Run Parakeet on a single audio file."
	@echo "  Parameters:"
	@echo "    AUDIO          - Path to audio file (required)"
	@echo "    PARAKEET_MODEL - Model name or .nemo path (default: nvidia/parakeet-tdt-0.6b-v3)"
	@echo "  Examples:"
	@echo "    make run-parakeet-single AUDIO=/path/to/audio.wav"
	@echo "    make run-parakeet-single AUDIO=test.mp3 PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2"
	@echo ""
	@echo "==============================================================================="
	@echo "UTILITIES"
	@echo "==============================================================================="
	@echo ""
	@echo "clean"
	@echo "  Remove all benchmark output directories."
	@echo "  Example: make clean"
	@echo ""
	@echo "clean-venvs"
	@echo "  Remove all Python virtual environments."
	@echo "  Example: make clean-venvs"
	@echo ""
	@echo "clean-test-data"
	@echo "  Remove all prepared/downloaded test datasets."
	@echo "  Example: make clean-test-data"
	@echo ""

# =============================================================================
# Internal: Environment Setup (called automatically by other targets)
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
prepare-test-data-all: prepare-ytl-testsets prepare-kespeech prepare-chinese-lips prepare-supa
	@echo ""
	@echo "============================================================"
	@echo "All datasets prepared!"
	@echo "============================================================"
	@$(MAKE) list-datasets

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

.PHONY: prepare-kespeech
prepare-kespeech: _setup-test-data
	@echo "Preparing KeSpeech dataset (Chinese)..."
	@echo "Downloading from HuggingFace: TwinkStart/KeSpeech"
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_kespeech.py \
		--output-dir kespeech $(MAX_SAMPLES_ARG)
	@touch $(STAMP_KESPEECH)
	@echo "Done: kespeech"

.PHONY: prepare-chinese-lips
prepare-chinese-lips: _setup-test-data
	@echo "Preparing Chinese-LiPS dataset (Chinese)..."
	@echo "Downloading from HuggingFace: BAAI/Chinese-LiPS"
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_chinese_lips.py \
		--output-dir chinese_lips $(MAX_SAMPLES_ARG)
	@echo "Done: chinese_lips"

.PHONY: prepare-supa
prepare-supa: _setup-test-data
	@echo "Preparing SUPA banking voice assistant dataset..."
	@cd $(TEST_DATA_DIR) && . .venv/bin/activate && $(PYTHON) ../scripts/prepare_supa.py \
		--output-dir supa
	@echo "Done: supa"

# =============================================================================
# Run Benchmarks
# =============================================================================

.PHONY: run-all
run-all: _setup-parakeet
	@if [ -z "$(DATASET)" ]; then \
		echo "ERROR: DATASET parameter is required"; \
		echo ""; \
		echo "Usage: make run-all DATASET=<dataset_name>"; \
		echo ""; \
		echo "Available datasets:"; \
		echo "  Malay: fleurs_test, malay_conversational, malay_scripted, supa"; \
		echo "  Chinese: kespeech, chinese_lips"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make run-all DATASET=supa"; \
		echo "  make run-all DATASET=fleurs_test MAX_SAMPLES=100"; \
		exit 1; \
	fi
	@echo "Running ALL models on dataset: $(DATASET)"
	@echo "  Models: faster-whisper, whisper-openai, whisper-mesolitica,"
	@echo "          whisper-finetuned, parakeet-v3, parakeet-5k, parakeet-multilingual, parakeet-malay"
ifdef MAX_SAMPLES
	@echo "  Max samples: $(MAX_SAMPLES)"
endif
	@cd parakeet && . .venv/bin/activate && uv pip install -q peft malaya faster-whisper && \
		$(PYTHON) ../scripts/run_all_benchmarks.py --output-dir ../benchmark_results --datasets $(DATASET) $(MAX_SAMPLES_ARG)
	@echo "Benchmark complete. Results in benchmark_results/"

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
	@if [ -z "$(AUDIO)" ]; then \
		echo "ERROR: AUDIO parameter is required"; \
		echo ""; \
		echo "Usage: make run-api-single AUDIO=/path/to/audio.wav"; \
		echo ""; \
		echo "Parameters:"; \
		echo "  AUDIO - Path to audio file (required)"; \
		echo "  MODEL - API model (default: bukit-tinggi-v2)"; \
		echo "  ENV   - Environment (default: production)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make run-api-single AUDIO=/path/to/audio.wav"; \
		echo "  make run-api-single AUDIO=test.mp3 MODEL=ilmu-preview-asr"; \
		exit 1; \
	fi
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
	@if [ -z "$(AUDIO)" ]; then \
		echo "ERROR: AUDIO parameter is required"; \
		echo ""; \
		echo "Usage: make run-parakeet-single AUDIO=/path/to/audio.wav"; \
		echo ""; \
		echo "Parameters:"; \
		echo "  AUDIO          - Path to audio file (required)"; \
		echo "  PARAKEET_MODEL - Model name or .nemo path (default: nvidia/parakeet-tdt-0.6b-v3)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make run-parakeet-single AUDIO=/path/to/audio.wav"; \
		echo "  make run-parakeet-single AUDIO=test.mp3 PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2"; \
		exit 1; \
	fi
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
	@rm -rf $(TEST_DATA_DIR)/YTL_testsets $(TEST_DATA_DIR)/kespeech \
		$(TEST_DATA_DIR)/chinese_lips $(TEST_DATA_DIR)/supa
	@rm -f $(STAMP_YTL) $(STAMP_KESPEECH)
	@echo "Done."
