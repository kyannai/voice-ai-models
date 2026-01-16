# Makefile for ElevenLabs Pronunciation Dictionary Tests
#
# Setup:
#   1. Create .env file with ELEVENLABS_API_KEY=your_key
#
# Usage:
#   make setup                          # Install dependencies
#   make test                           # Run Malay dictionary test
#   make clean                          # Remove output files

SHELL := /bin/bash
PYTHON := python3
PIP := pip3

# Output directory
OUTPUT_DIR ?= ./output

.PHONY: setup test clean help

# Install dependencies
setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✓ Setup complete"

# Run Malay dictionary test (with/without comparison)
test:
	@echo "Running Malaysian pronunciation dictionary test..."
	$(PYTHON) test_malay_dictionary.py

# Clean output files
clean:
	@echo "Cleaning output directory..."
	rm -rf $(OUTPUT_DIR)/*.mp3
	rm -rf $(OUTPUT_DIR)/*.json
	@echo "✓ Clean complete"

# Help
help:
	@echo "ElevenLabs Pronunciation Dictionary Tests"
	@echo ""
	@echo "Setup:"
	@echo "  1. Create .env file with: ELEVENLABS_API_KEY=your_api_key"
	@echo "  2. Run: make setup"
	@echo ""
	@echo "Usage:"
	@echo "  make setup    Install dependencies"
	@echo "  make test     Run Malay dictionary test (without/with comparison)"
	@echo "  make clean    Remove output files"
	@echo ""
	@echo "Note: Phoneme tags only work with eleven_turbo_v2, eleven_flash_v2,"
	@echo "      and eleven_monolingual_v1 models."
