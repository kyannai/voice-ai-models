# MagpieTTS Makefile
# ==================
# NVIDIA MagpieTTS: Multilingual TTS with fixed speakers
# https://huggingface.co/nvidia/magpie_tts_multilingual_357m
#
# NOTE: This model uses FIXED speakers (Sofia, Aria, Jason, Leo, John),
#       NOT zero-shot voice cloning from reference audio.
#
# Usage:
#   make install    - Install dependencies
#   make synthesize TEXT="..." - Synthesize speech
#   make run        - Run with default settings
#   make clean      - Remove output files

# Default values
TEXT ?= Hello, this is a test of MagpieTTS text to speech.
LANGUAGE ?= en
SPEAKER ?= Sofia
OUTPUT ?= output/magpietts.wav

# Available speakers: John, Sofia, Aria, Jason, Leo
# Available languages: en, es, de, fr, vi, it, zh

.PHONY: install setup run synthesize clean help

help:
	@echo "MagpieTTS Synthesizer (NVIDIA NeMo)"
	@echo ""
	@echo "NOTE: This model uses FIXED speakers, not zero-shot voice cloning."
	@echo ""
	@echo "Usage:"
	@echo "  make install                  - Install dependencies"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make run                      - Run with default settings"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT     - Text to synthesize"
	@echo "  LANGUAGE - Language code: en, es, de, fr, vi, it, zh (default: en)"
	@echo "  SPEAKER  - Speaker name: John, Sofia, Aria, Jason, Leo (default: Sofia)"
	@echo "  OUTPUT   - Output file path (default: output/magpietts.wav)"
	@echo ""
	@echo "Examples:"
	@echo "  make synthesize TEXT='Hello world'"
	@echo "  make synthesize TEXT='Bonjour le monde' LANGUAGE=fr SPEAKER=Aria"
	@echo "  make synthesize TEXT='Hola mundo' LANGUAGE=es SPEAKER=Jason"

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
	python synthesize_magpietts.py \
		--text '$(TEXT)' \
		--language $(LANGUAGE) \
		--speaker $(SPEAKER) \
		--output '$(OUTPUT)'
	@echo ""
	@echo "Output saved to $(OUTPUT)"

clean:
	rm -rf output
	rm -rf __pycache__
	@echo "Cleaned up."
