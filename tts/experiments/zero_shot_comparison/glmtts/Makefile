# GLM-TTS Makefile
# =================
# GLM-TTS: Controllable & Emotion-Expressive Zero-shot TTS
# https://github.com/zai-org/GLM-TTS
#
# Usage:
#   make install    - Full install (clone repo + download model + deps)
#   make setup      - Install dependencies only
#   make synthesize TEXT="..." SPEAKER=... - Synthesize with custom text
#   make run        - Run synthesis with default settings
#   make clean      - Remove output files

# Default values for synthesis
TEXT ?= Hello, this is a test of GLM-TTS text to speech.
SPEAKER ?= ../speakers/output/elevenlabs_UcqZLa941Kkt8ZhEEybf.wav
PROMPT_TEXT ?= 
OUTPUT ?= output/glmtts.wav

# For data-file based synthesis (legacy)
DATA ?= example_zh
EXP_NAME ?= custom

.PHONY: install setup run synthesize synthesize-data clone-repo download-model web clean help

help:
	@echo "GLM-TTS Synthesizer (zai-org/GLM-TTS)"
	@echo ""
	@echo "Usage:"
	@echo "  make install                  - Full install (clone + model + deps)"
	@echo "  make setup                    - Install dependencies only"
	@echo "  make synthesize TEXT=\"...\"    - Synthesize custom text"
	@echo "  make synthesize-data DATA=... - Synthesize from JSONL data file"
	@echo "  make web                      - Launch Gradio web interface"
	@echo "  make clean                    - Remove outputs"
	@echo ""
	@echo "Options:"
	@echo "  TEXT       - Text to synthesize"
	@echo "  SPEAKER    - Path to reference audio for voice cloning"
	@echo "  PROMPT_TEXT- Text spoken in reference audio (optional)"
	@echo "  OUTPUT     - Output file path (default: output/glmtts.wav)"
	@echo ""
	@echo "Examples:"
	@echo "  make synthesize TEXT='Hello world' SPEAKER=ref.wav"
	@echo "  make synthesize TEXT='Okay, kita kena siapkan report ni.' SPEAKER=../speakers/output/speaker.wav"

install: clone-repo download-model setup
	@echo ""
	@echo "Installation complete! Run 'make synthesize TEXT=\"...\"' to generate."

clone-repo:
	@if [ ! -d "GLM-TTS" ]; then \
		git clone https://github.com/zai-org/GLM-TTS.git; \
	else \
		echo "GLM-TTS already cloned"; \
	fi
	@if [ -f "GLM-TTS/requirements.txt" ]; then \
		uv pip install -r GLM-TTS/requirements.txt; \
	fi
	@echo "GLM-TTS repo cloned and dependencies installed."

download-model:
	@mkdir -p GLM-TTS/ckpt
	uv pip install huggingface-hub
	huggingface-cli download zai-org/GLM-TTS --local-dir GLM-TTS/ckpt
	@echo "Model downloaded to GLM-TTS/ckpt"

setup:
	uv pip install -r requirements.txt
	@echo ""
	@echo "Dependencies installed."

run: synthesize

synthesize:
	@mkdir -p output
	@mkdir -p GLM-TTS/examples
	@# Convert SPEAKER path to absolute path and create JSONL
	@# GLM-TTS expects: uttid, prompt_text, prompt_speech, syn_text
	@SPEAKER_ABS=$$(realpath "$(SPEAKER)") && \
		python3 -c "import json; print(json.dumps({'uttid': 'custom_0', 'prompt_speech': '$$SPEAKER_ABS', 'prompt_text': '''$(PROMPT_TEXT)''', 'syn_text': '''$(TEXT)'''}))" > GLM-TTS/examples/custom_input.jsonl
	@echo "Created input file: GLM-TTS/examples/custom_input.jsonl"
	@cat GLM-TTS/examples/custom_input.jsonl
	cd GLM-TTS && python glmtts_inference.py \
		--data=custom_input \
		--exp_name=$(EXP_NAME) \
		--use_cache
	@# Copy output to output folder
	@if [ -f "GLM-TTS/outputs/$(EXP_NAME)/custom_input/custom_0.wav" ]; then \
		cp "GLM-TTS/outputs/$(EXP_NAME)/custom_input/custom_0.wav" "$(OUTPUT)"; \
		echo "Output saved to $(OUTPUT)"; \
	else \
		echo "Warning: Expected output not found. Check GLM-TTS/outputs/$(EXP_NAME)/"; \
		find GLM-TTS/outputs -name "*.wav" -mmin -5 2>/dev/null | head -5; \
	fi

synthesize-data:
	cd GLM-TTS && python glmtts_inference.py \
		--data=$(DATA) \
		--exp_name=$(EXP_NAME) \
		--use_cache

web:
	cd GLM-TTS && python -m tools.gradio_app

clean:
	rm -rf output
	rm -rf GLM-TTS/examples/custom_input.jsonl
	rm -rf __pycache__
	@echo "Cleaned up. Note: GLM-TTS/ and ckpt/ not removed."
