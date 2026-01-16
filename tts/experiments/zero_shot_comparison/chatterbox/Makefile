# Chatterbox-Turbo Makefile
# =========================
#
# Usage:
#   make setup      - Install dependencies with uv
#   make run        - Run synthesis with default settings
#   make synthesize TEXT="Hello world" SPEAKER=ref.wav OUTPUT=out.wav
#   make clean      - Remove output files

# Default values
TEXT ?= "Hello there, how are you doing today?"
SPEAKER ?= ../speakers/output/elevenlabs_UcqZLa941Kkt8ZhEEybf.wav
OUTPUT ?= output/chatterbox.wav

.PHONY: setup run synthesize clean help

help:
	@echo "Chatterbox-Turbo Synthesizer"
	@echo ""
	@echo "Usage:"
	@echo "  make setup                    - Install dependencies with uv"
	@echo "  make run                      - Run with default text"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT     - Text to synthesize (supports [laugh], [chuckle] tags)"
	@echo "  SPEAKER  - Speaker reference WAV file (~10s recommended)"
	@echo "  OUTPUT   - Output WAV file path"

setup:
	uv pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Run 'make run' to synthesize."

run: synthesize

synthesize:
	@mkdir -p output
	python synthesize_chatterbox.py \
		--text "$(TEXT)" \
		--speaker-wav "$(SPEAKER)" \
		--output "$(OUTPUT)"

clean:
	rm -rf output
	rm -rf __pycache__
	@echo "Cleaned up."
