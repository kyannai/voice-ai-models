# ASR Benchmark Makefile
# Usage: make <target>

SHELL := /bin/bash
PYTHON := python

# Default target
.PHONY: help
help:
	@echo "ASR Benchmark - Available targets:"
	@echo ""
	@echo "Utilities:"
	@echo "  make prepare-test-data - Unpack and validate test datasets"
	@echo "  make clean           - Clean output directories"
	@echo "  make clean-venvs     - Remove virtual environments"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-all       - Setup all benchmark environments"
	@echo "  make setup-api       - Setup API benchmark environment"
	@echo "  make setup-whisper   - Setup Whisper benchmark environment"
	@echo "  make setup-parakeet  - Setup Parakeet benchmark environment"
	@echo ""
	@echo "Run benchmarks:"
	@echo "  make run-all         - Run all benchmarks"
	@echo "  make run-api         - Run YTL API benchmark"
	@echo "    e.g. make run-api MODEL=bukit-tinggi ENV=staging"
	@echo "  make run-whisper     - Run Whisper benchmarks"
	@echo "  make run-parakeet    - Run Parakeet benchmark"
	@echo ""
	@echo "Single file tests:"
	@echo "  make run-api-single AUDIO=/path/to/audio.wav"
	@echo "  make run-parakeet-single AUDIO=/path/to/audio.wav"
	@echo ""

# =============================================================================
# Environment Setup
# =============================================================================

.PHONY: setup-all
setup-all: setup-api setup-whisper setup-parakeet
	@echo "All environments setup complete."

.PHONY: setup-api
setup-api:
	@echo "Setting up API benchmark environment..."
	cd api && uv venv --python 3.10 && . .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Done. Activate with: cd api && source .venv/bin/activate"

.PHONY: setup-whisper
setup-whisper:
	@echo "Setting up Whisper benchmark environment..."
	cd whisper && uv venv --python 3.10 && . .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Done. Activate with: cd whisper && source .venv/bin/activate"

.PHONY: setup-parakeet
setup-parakeet:
	@echo "Setting up Parakeet benchmark environment..."
	cd parakeet && uv venv --python 3.10 && . .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Done. Activate with: cd parakeet && source .venv/bin/activate"

# =============================================================================
# Run Benchmarks
# =============================================================================

# Parakeet model (required for parakeet benchmarks)
MODEL ?= nvidia/parakeet-tdt-0.6b
ENV ?= production
BATCH_SIZE ?= 16
WORKERS ?= 1

# Test data
TEST_DATA_DIR := test_data
TEST_DATA_TAR := $(TEST_DATA_DIR)/YTL_testsets.tar
TEST_DATA_EXTRACT_DIR := $(TEST_DATA_DIR)/YTL_testsets
TEST_DATA_STAMP := $(TEST_DATA_DIR)/.ytl_testsets_prepared
TEST_DATA_VENV := $(TEST_DATA_DIR)/.venv
TEST_DATA_PYTHON := $(TEST_DATA_VENV)/bin/python

.PHONY: run-all
run-all: run-api run-whisper run-parakeet
	@echo "All benchmarks complete."

.PHONY: run-api
run-api: setup-api
	@echo "Running YTL API benchmark..."
	cd api && . .venv/bin/activate && $(PYTHON) ytl_api_test.py --model $(MODEL) --env $(ENV)

.PHONY: run-api-single
run-api-single: setup-api
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make run-api-single AUDIO=/path/to/audio.wav"; exit 1; fi
	cd api && . .venv/bin/activate && $(PYTHON) ytl_api_test_single.py -a $(AUDIO) -m $(MODEL) --env $(ENV)

.PHONY: run-whisper
run-whisper: run-whisper-faster run-whisper-mesolitica
	@echo "All Whisper benchmarks complete."

.PHONY: run-whisper-faster
run-whisper-faster: setup-whisper
	@echo "Running Faster Whisper benchmark..."
	cd whisper && . .venv/bin/activate && $(PYTHON) ytl_faster_whisper_test.py

.PHONY: run-whisper-mesolitica
run-whisper-mesolitica: setup-whisper
	@echo "Running Mesolitica Whisper benchmark..."
	cd whisper && . .venv/bin/activate && $(PYTHON) ytl_mesolitica_whisper_test.py

.PHONY: run-parakeet
run-parakeet: setup-parakeet
	@echo "Running Parakeet benchmark with model: $(MODEL)"
	cd parakeet && . .venv/bin/activate && $(PYTHON) ytl_parakeet_test.py -m $(MODEL) -b $(BATCH_SIZE) -w $(WORKERS)

.PHONY: run-parakeet-single
run-parakeet-single: setup-parakeet
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make run-parakeet-single AUDIO=/path/to/audio.wav"; exit 1; fi
	cd parakeet && . .venv/bin/activate && $(PYTHON) ytl_parakeet_test_single.py -a $(AUDIO)

# =============================================================================
# Test Data
# =============================================================================

.PHONY: prepare-test-data
prepare-test-data: $(TEST_DATA_VENV) $(TEST_DATA_STAMP)
	@echo "Test data ready."

$(TEST_DATA_VENV): $(TEST_DATA_DIR)/requirements.txt
	@echo "Setting up test data environment..."
	cd $(TEST_DATA_DIR) && uv venv --python 3.10
	cd $(TEST_DATA_DIR) && . .venv/bin/activate && uv pip install -r requirements.txt

$(TEST_DATA_STAMP): $(TEST_DATA_TAR)
	@echo "Preparing test data from $(TEST_DATA_TAR)..."
	@mkdir -p $(TEST_DATA_DIR)
	@tar -xf $(TEST_DATA_TAR) -C $(TEST_DATA_DIR)
	@$(TEST_DATA_PYTHON) scripts/prepare_test_data.py
	@touch $(TEST_DATA_STAMP)

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	@echo "Cleaning output directories..."
	rm -rf api/ytl_labs whisper/ytl_labs whisper/faster_whisper parakeet/ytl_parakeet
	@echo "Done."

.PHONY: clean-venvs
clean-venvs:
	@echo "Removing virtual environments..."
	rm -rf api/.venv whisper/.venv parakeet/.venv
	@echo "Done."
