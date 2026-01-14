# IndexTTS2 Makefile
# ==================
#
# Usage:
#   make install    - Full install (clone repo + download model)
#   make setup      - Install dependencies with uv
#   make run        - Run synthesis with default settings
#   make synthesize TEXT="Hello world" SPEAKER=ref.wav OUTPUT=out.wav
#   make clean      - Remove output files

# Default values
TEXT ?= "Okay, so basically kita kena siapkan report ni sebelum Jumaat."
SPEAKER ?= ../speakers/output/elevenlabs_UcqZLa941Kkt8ZhEEybf.wav
OUTPUT ?= output/indextts.wav
MODEL_DIR ?= checkpoints

.PHONY: install setup run synthesize clone-repo download-model clean help

help:
	@echo "IndexTTS2 Synthesizer"
	@echo ""
	@echo "Usage:"
	@echo "  make install                  - Full install (clone + model + deps)"
	@echo "  make setup                    - Install dependencies only"
	@echo "  make run                      - Run with default text"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT      - Text to synthesize (default: sample Malay)"
	@echo "  SPEAKER   - Speaker reference WAV file"
	@echo "  OUTPUT    - Output WAV file path"
	@echo "  MODEL_DIR - Path to model checkpoints"

install: clone-repo download-model setup
	@echo ""
	@echo "Installation complete! Run 'make run' to synthesize."

clone-repo:
	@if [ ! -d "index-tts" ]; then \
		git clone https://github.com/index-tts/index-tts.git; \
		cd index-tts && git lfs pull; \
	else \
		echo "index-tts already cloned"; \
	fi
	uv pip install -e index-tts
	@echo "IndexTTS repo cloned and installed."

setup:
	uv pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed."

download-model:
	uv pip install "huggingface-hub[cli,hf_xet]"
	huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=$(MODEL_DIR)
	@echo "Model downloaded to $(MODEL_DIR)"

run: synthesize

synthesize:
	@mkdir -p output
	python synthesize_indextts.py \
		--text $(TEXT) \
		--speaker-wav $(SPEAKER) \
		--output $(OUTPUT) \
		--model-dir $(MODEL_DIR)

clean:
	rm -rf output
	rm -rf __pycache__
	@echo "Cleaned up. Note: checkpoints/ not removed."
