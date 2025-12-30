#!/bin/bash
#
# TTS Synthetic Data Generation Pipeline
# Generates XTTS v2 training data using ElevenLabs
#
# Prerequisites:
#   1. Set GOOGLE_API_KEY for Gemini text generation
#   2. Set ELEVENLABS_API_KEY for audio synthesis
#   3. Update config.yaml with ElevenLabs voice IDs
#
# Usage:
#   ./run_pipeline.sh              # Run full pipeline
#   ./run_pipeline.sh --step text  # Run only text generation
#   ./run_pipeline.sh --step audio # Run only audio synthesis
#   ./run_pipeline.sh --step prep  # Run only dataset preparation
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default step (all)
STEP="all"
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --step STEP    Run only this step (text, audio, prep, all)"
            echo "  --resume       Resume from checkpoint (for audio step)"
            echo "  --config FILE  Use specified config file"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  TTS Synthetic Data Generation Pipeline   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check environment variables
check_env() {
    local missing=0
    
    if [[ -z "${GOOGLE_API_KEY}" && ("${STEP}" == "all" || "${STEP}" == "text") ]]; then
        echo -e "${YELLOW}⚠ GOOGLE_API_KEY not set - text generation will use fallback templates${NC}"
    fi
    
    if [[ -z "${ELEVENLABS_API_KEY}" ]]; then
        echo -e "${RED}✗ ELEVENLABS_API_KEY not set${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ ELEVENLABS_API_KEY is set${NC}"
    fi
    
    return $missing
}

# Step 1: Generate text with Gemini
step_text() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 1: Generate Text with Gemini 2.5 Flash${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python "${SCRIPT_DIR}/scripts/generate_text.py" \
        --config "${CONFIG}" \
        --output "${OUTPUT_DIR}"
    
    echo -e "${GREEN}✓ Text generation complete${NC}"
}

# Step 2: Synthesize audio with ElevenLabs
step_audio() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 2: Synthesize Audio with ElevenLabs${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python "${SCRIPT_DIR}/scripts/synthesize_elevenlabs.py" \
        --config "${CONFIG}" \
        ${RESUME}
    
    echo -e "${GREEN}✓ Audio synthesis complete${NC}"
}

# Step 3: Prepare XTTS dataset
step_prep() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 3: Prepare XTTS Training Dataset${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    python "${SCRIPT_DIR}/scripts/prepare_xtts_dataset.py" \
        --config "${CONFIG}"
    
    echo -e "${GREEN}✓ Dataset preparation complete${NC}"
}

# Main execution
main() {
    # Check environment
    if [[ "${STEP}" != "text" ]]; then
        if ! check_env; then
            echo ""
            echo -e "${RED}Please set required environment variables:${NC}"
            echo "  export ELEVENLABS_API_KEY='your-key-here'"
            echo "  export GOOGLE_API_KEY='your-key-here' (optional)"
            exit 1
        fi
    fi
    
    # Check config exists
    if [[ ! -f "${CONFIG}" ]]; then
        echo -e "${RED}Config not found: ${CONFIG}${NC}"
        exit 1
    fi
    
    # Run steps
    case ${STEP} in
        all)
            step_text
            step_audio
            step_prep
            ;;
        text)
            step_text
            ;;
        audio)
            step_audio
            ;;
        prep)
            step_prep
            ;;
        *)
            echo -e "${RED}Unknown step: ${STEP}${NC}"
            echo "Valid steps: all, text, audio, prep"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Pipeline Complete!                        ${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Check outputs/ for generated data"
    echo "  2. Review train_manifest.json and val_manifest.json"
    echo "  3. Use the data to fine-tune XTTS v2"
}

main

