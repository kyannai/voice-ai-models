# XTTS v2 Makefile
# =================
#
# Usage:
#   make setup      - Install dependencies with uv
#   make run        - Run synthesis with default settings
#   make synthesize TEXT="Hello world" SPEAKER=ref.wav OUTPUT=out.wav
#   make clean      - Remove output files

# Default values
TEXT ?= "Okay, so basically kita kena siapkan report ni sebelum Jumaat."
SPEAKER ?= ../speakers/output/elevenlabs_UcqZLa941Kkt8ZhEEybf.wav
OUTPUT ?= output/xtts.wav
LANGUAGE ?= ms

.PHONY: setup run synthesize clean help

help:
	@echo "XTTS v2 Synthesizer"
	@echo ""
	@echo "Usage:"
	@echo "  make setup                    - Install dependencies with uv"
	@echo "  make run                      - Run with default text"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT     - Text to synthesize (default: sample Malay)"
	@echo "  SPEAKER  - Speaker reference WAV file"
	@echo "  OUTPUT   - Output WAV file path"
	@echo "  LANGUAGE - Language code (default: ms)"

setup:
	uv pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Run 'make run' to synthesize."

run: synthesize

synthesize:
	@mkdir -p output
	python synthesize_xtts.py \
		--text $(TEXT) \
		--speaker-wav $(SPEAKER) \
		--output $(OUTPUT) \
		--language $(LANGUAGE)

clean:
	rm -rf output
	rm -rf __pycache__
	@echo "Cleaned up."
