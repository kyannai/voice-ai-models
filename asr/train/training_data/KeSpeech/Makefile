# KeSpeech Dataset Preparation
# ============================
# Full 1500h KeSpeech Mandarin dataset
#
# Data is stored in split tar.gz files:
#   KeSpeech.splits/KeSpeech.tar.gz.aa ... .ai
#
# Usage:
#   make prepare          - Extract and prepare full dataset
#   make prepare-quick    - Prepare with limited samples (for testing)
#   make info             - Show data info

SHELL := /bin/bash
PYTHON := python3

VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: help setup prepare prepare-quick extract info clean

help:
	@echo "KeSpeech Dataset Preparation"
	@echo "============================"
	@echo ""
	@echo "Commands:"
	@echo "  make setup          - Create venv and install dependencies"
	@echo "  make prepare        - Extract and prepare full dataset (~1500h)"
	@echo "  make prepare-quick  - Prepare with 1000 samples (for testing)"
	@echo "  make extract        - Only extract archives (skip processing)"
	@echo "  make info           - Show dataset info"
	@echo "  make clean          - Remove generated files"
	@echo ""

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.10; \
	fi
	@echo "Installing dependencies..."
	. $(VENV)/bin/activate && uv pip install -r requirements.txt
	@echo "Setup complete!"

extract: setup
	@echo "Extracting KeSpeech archives..."
	@if [ -d "KeSpeech.splits" ] && [ ! -d "Tasks" ]; then \
		echo "Combining split archives..."; \
		cat KeSpeech.splits/KeSpeech.tar.gz.* > KeSpeech.tar.gz; \
		echo "Extracting (this will take a while)..."; \
		tar -xzf KeSpeech.tar.gz; \
		echo "Done!"; \
	else \
		echo "Already extracted or no split files found."; \
	fi

prepare: setup
	@echo "Preparing KeSpeech dataset..."
	@mkdir -p data
	. $(VENV)/bin/activate && $(PYTHON) src/prepare_kespeech.py \
		--data-dir . \
		--output-dir data \
		--subsets train_phase1 train_phase2 dev_phase1 dev_phase2 test
	@echo ""
	@echo "Done! Manifests saved to data/manifests/"

prepare-quick: setup
	@echo "Preparing KeSpeech dataset (quick test - 1000 samples)..."
	@mkdir -p data
	. $(VENV)/bin/activate && $(PYTHON) src/prepare_kespeech.py \
		--data-dir . \
		--output-dir data \
		--max-samples 1000 \
		--subsets train_phase1 dev_phase1

info:
	@echo "KeSpeech Dataset Info"
	@echo "====================="
	@echo ""
	@if [ -d "KeSpeech.splits" ]; then \
		echo "Split archives:"; \
		ls -lh KeSpeech.splits/; \
		echo ""; \
	fi
	@if [ -d "Tasks/ASR" ]; then \
		echo "Extracted subsets:"; \
		for d in Tasks/ASR/*/; do \
			echo "  $$(basename $$d): $$(wc -l < $$d/text 2>/dev/null || echo 0) utterances"; \
		done; \
	fi
	@if [ -f "data/manifests/train_manifest.json" ]; then \
		echo ""; \
		echo "Prepared manifests:"; \
		echo "  Train: $$(wc -l < data/manifests/train_manifest.json) samples"; \
		echo "  Val: $$(wc -l < data/manifests/val_manifest.json) samples"; \
	fi

clean:
	@echo "Removing generated files..."
	rm -rf data/
	rm -f KeSpeech.tar.gz
	@echo "Done. (Raw data in KeSpeech.splits/ and Tasks/ preserved)"

clean-all: clean
	@echo "Removing all data (including extracted)..."
	rm -rf Tasks/
	rm -rf KeSpeech/
	@echo "Done."
