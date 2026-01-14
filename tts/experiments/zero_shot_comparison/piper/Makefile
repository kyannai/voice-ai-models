# Piper TTS Makefile
# ==================
# Fast and local neural text-to-speech engine
# https://github.com/OHF-Voice/piper1-gpl
#
# NOTE: This model uses PRE-TRAINED voices, not zero-shot voice cloning.
#       You can choose from many available voices.
#
# Usage:
#   make install    - Install dependencies
#   make synthesize TEXT="..." - Synthesize speech
#   make run        - Run with default settings
#   make clean      - Remove output files

# Default values
TEXT ?= Hello, this is a test of Piper text to speech.
MODEL ?= en_US-lessac-medium
OUTPUT ?= output/piper.wav

# Common voices:
#   English: en_US-lessac-medium, en_US-amy-medium, en_US-ryan-medium, en_GB-alba-medium
#   Chinese: zh_CN-huayan-medium
#   Spanish: es_ES-davefx-medium
#   French:  fr_FR-upmc-medium
#   German:  de_DE-thorsten-medium
# More: https://huggingface.co/rhasspy/piper-voices

.PHONY: install setup run synthesize download-model list-voices clean help

help:
	@echo "Piper TTS Synthesizer"
	@echo ""
	@echo "NOTE: This model uses PRE-TRAINED voices, not zero-shot voice cloning."
	@echo ""
	@echo "Usage:"
	@echo "  make install                  - Install dependencies"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make list-voices              - List common voice models"
	@echo "  make run                      - Run with default settings"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT   - Text to synthesize"
	@echo "  MODEL  - Voice model (default: en_US-lessac-medium)"
	@echo "  OUTPUT - Output file path (default: output/piper.wav)"
	@echo ""
	@echo "Examples:"
	@echo "  make synthesize TEXT='Hello world'"
	@echo "  make synthesize TEXT='Hello' MODEL=en_US-amy-medium"
	@echo "  make synthesize TEXT='Bonjour' MODEL=fr_FR-upmc-medium"

install: setup
	@echo ""
	@echo "Installation complete! Run 'make synthesize TEXT=\"...\"' to generate."

setup:
	uv pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed."

run: synthesize

synthesize:
	@mkdir -p output
	python synthesize_piper.py \
		--text '$(TEXT)' \
		--model $(MODEL) \
		--output '$(OUTPUT)'
	@echo ""
	@echo "Output saved to $(OUTPUT)"

download-model:
	@mkdir -p models
	python synthesize_piper.py --text "test" --model $(MODEL) --output /tmp/test.wav || true
	@echo "Model $(MODEL) downloaded to models/"

list-voices:
	@python synthesize_piper.py --list-voices

clean:
	rm -rf output
	rm -rf __pycache__
	@echo "Cleaned up. Note: models/ not removed."

clean-all: clean
	rm -rf models
	@echo "Removed all models."
