#!/bin/bash
# TTS Model Comparison Script
# ============================
#
# Runs a single sentence through multiple TTS models.
# Each model folder has its own .venv which is sourced before running.
# All outputs are collected in output/<model>.wav for easy comparison.
#
# Usage:
#   ./run.sh "Your sentence here"
#   ./run.sh "Your sentence here" --speaker path/to/speaker.wav
#   ./run.sh "Your sentence here" --models xtts glmtts
#   ./run.sh --help

# Don't use set -e as it causes issues with arithmetic expressions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
TEXT=""
SPEAKER="speakers/output/elevenlabs_UcqZLa941Kkt8ZhEEybf.wav"
OUTPUT_DIR="output"
MODELS=""

# Zero-shot models (support voice cloning)
ZERO_SHOT_MODELS="xtts indextts glmtts"
# Fixed-voice models
FIXED_VOICE_MODELS="magpietts piper"
# All models
ALL_MODELS="$ZERO_SHOT_MODELS $FIXED_VOICE_MODELS"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo "TTS Model Comparison Script"
    echo ""
    echo "Usage:"
    echo "  $0 \"Your sentence here\""
    echo "  $0 \"Your sentence here\" --speaker path/to/speaker.wav"
    echo "  $0 \"Your sentence here\" --models xtts glmtts"
    echo ""
    echo "Arguments:"
    echo "  TEXT                   The sentence to synthesize (required, first positional arg)"
    echo ""
    echo "Options:"
    echo "  --speaker FILE         Reference audio for zero-shot voice cloning"
    echo "                         (default: speakers/output/elevenlabs_*.wav)"
    echo "  --models MODEL...      Models to run (default: all)"
    echo "                         Zero-shot: xtts, indextts, glmtts"
    echo "                         Fixed-voice: magpietts, piper"
    echo "  --output-dir DIR       Output directory (default: output)"
    echo "  --help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 \"Hello, how are you today?\""
    echo "  $0 \"Okay, kita kena siapkan report ni.\" --speaker ref.wav"
    echo "  $0 \"Quick test\" --models xtts glmtts"
    echo ""
    echo "Output:"
    echo "  All audio files are saved to: output/<model>.wav"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_help
            exit 0
            ;;
        --speaker)
            SPEAKER="$2"
            shift 2
            ;;
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_help
            exit 1
            ;;
        *)
            # First positional argument is the text
            if [ -z "$TEXT" ]; then
                TEXT="$1"
            fi
            shift
            ;;
    esac
done

# Validate text
if [ -z "$TEXT" ]; then
    echo -e "${RED}Error: No text provided${NC}"
    echo ""
    print_help
    exit 1
fi

# Use all models if none specified
if [ -z "$MODELS" ]; then
    MODELS="$ALL_MODELS"
fi

# Convert speaker path to absolute
if [[ ! "$SPEAKER" = /* ]]; then
    SPEAKER="$SCRIPT_DIR/$SPEAKER"
fi

# Create central output directory
CENTRAL_OUTPUT="$SCRIPT_DIR/$OUTPUT_DIR"
mkdir -p "$CENTRAL_OUTPUT"

echo "========================================"
echo -e "${BLUE}TTS Model Comparison${NC}"
echo "========================================"
echo -e "Text:    ${GREEN}$TEXT${NC}"
echo -e "Speaker: $SPEAKER"
echo -e "Models:  $MODELS"
echo "========================================"
echo ""

# Function to run a model
run_model() {
    local model_name="$1"
    local model_dir="$SCRIPT_DIR/$model_name"
    local venv_path="$model_dir/.venv/bin/activate"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running: $model_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if model directory exists
    if [ ! -d "$model_dir" ]; then
        echo -e "${YELLOW}⚠ Skipping $model_name: directory not found${NC}"
        return 1
    fi
    
    # Check if venv exists
    if [ ! -f "$venv_path" ]; then
        echo -e "${YELLOW}⚠ Skipping $model_name: .venv not found${NC}"
        echo "  Run 'cd $model_name && python -m venv .venv && source .venv/bin/activate && make setup'"
        return 1
    fi
    
    # Run in a subshell to isolate the venv
    (
        cd "$model_dir"
        source "$venv_path"
        
        echo "Using Python: $(which python)"
        mkdir -p output
        
        # Run python directly to avoid make quoting issues
        case "$model_name" in
            xtts)
                python synthesize_xtts.py \
                    --text "$TEXT" \
                    --speaker-wav "$SPEAKER" \
                    --output output/xtts.wav \
                    --language ms
                ;;
            indextts)
                python synthesize_indextts.py \
                    --text "$TEXT" \
                    --speaker-wav "$SPEAKER" \
                    --output output/indextts.wav \
                    --model-dir checkpoints
                ;;
            cosyvoice)
                PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS python synthesize_cosyvoice.py \
                    --text "$TEXT" \
                    --speaker-wav "$SPEAKER" \
                    --output output/cosyvoice.wav \
                    --mode zero_shot
                ;;
            glmtts)
                # GLM-TTS uses JSONL input format
                mkdir -p GLM-TTS/examples
                SPEAKER_ABS=$(realpath "$SPEAKER")
                
                # Try to get prompt_text from metadata.json if it exists
                SPEAKER_DIR=$(dirname "$SPEAKER_ABS")
                METADATA_FILE="$SPEAKER_DIR/metadata.json"
                if [ -f "$METADATA_FILE" ]; then
                    PROMPT_TEXT=$(python3 -c "import json; print(json.load(open('$METADATA_FILE')).get('text', ''))" 2>/dev/null || echo "")
                else
                    PROMPT_TEXT=""
                fi
                
                # Use environment variables to avoid shell escaping issues
                GLMTTS_TEXT="$TEXT" GLMTTS_SPEAKER="$SPEAKER_ABS" GLMTTS_PROMPT="$PROMPT_TEXT" python3 -c '
import json, os
data = {
    "uttid": "custom_0",
    "prompt_speech": os.environ["GLMTTS_SPEAKER"],
    "prompt_text": os.environ.get("GLMTTS_PROMPT", ""),
    "syn_text": os.environ["GLMTTS_TEXT"]
}
print(json.dumps(data, ensure_ascii=False))
' > GLM-TTS/examples/custom_input.jsonl
                echo "GLM-TTS input: $(cat GLM-TTS/examples/custom_input.jsonl)"
                # Don't use --use_cache to avoid stale cached results
                cd GLM-TTS && python glmtts_inference.py --data=custom_input --exp_name=custom
                cd ..
                # Find and copy the output file
                if [ -f "GLM-TTS/outputs/custom/custom_input/custom_0.wav" ]; then
                    cp "GLM-TTS/outputs/custom/custom_input/custom_0.wav" "output/glmtts.wav"
                    echo "Output saved to: output/glmtts.wav"
                else
                    # Try to find any recent wav file
                    GLMTTS_OUTPUT=$(find GLM-TTS/outputs -name "*.wav" -mmin -5 2>/dev/null | head -1)
                    if [ -n "$GLMTTS_OUTPUT" ]; then
                        cp "$GLMTTS_OUTPUT" "output/glmtts.wav"
                        echo "Output saved to: output/glmtts.wav (from $GLMTTS_OUTPUT)"
                    else
                        echo "Warning: GLM-TTS output file not found"
                        find GLM-TTS/outputs -name "*.wav" 2>/dev/null | head -5
                        exit 1
                    fi
                fi
                ;;
            magpietts)
                python synthesize_magpietts.py \
                    --text "$TEXT" \
                    --language en \
                    --speaker Sofia \
                    --output output/magpietts.wav
                ;;
            piper)
                python synthesize_piper.py \
                    --text "$TEXT" \
                    --model en_US-lessac-medium \
                    --output output/piper.wav
                ;;
            *)
                echo -e "${YELLOW}Unknown model: $model_name${NC}"
                return 1
                ;;
        esac
    )
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ $model_name completed${NC}"
        
        # Copy output to central output folder
        local src_file=""
        case "$model_name" in
            glmtts)
                src_file="$model_dir/output/glmtts.wav"
                ;;
            *)
                src_file="$model_dir/output/${model_name}.wav"
                ;;
        esac
        
        if [ -f "$src_file" ]; then
            cp "$src_file" "$CENTRAL_OUTPUT/${model_name}.wav"
            echo -e "  Copied to: ${GREEN}$OUTPUT_DIR/${model_name}.wav${NC}"
        fi
    else
        echo -e "${RED}✗ $model_name failed${NC}"
    fi
    
    return $status
}

# Track results
declare -A RESULTS
SUCCESS_COUNT=0
FAIL_COUNT=0

# Run each model
for model in $MODELS; do
    if run_model "$model"; then
        RESULTS[$model]="success"
        ((SUCCESS_COUNT++))
    else
        RESULTS[$model]="failed"
        ((FAIL_COUNT++))
    fi
done

# Print summary
echo ""
echo "========================================"
echo -e "${BLUE}Summary${NC}"
echo "========================================"
echo -e "Text: ${GREEN}$TEXT${NC}"
echo -e "Output folder: ${GREEN}$OUTPUT_DIR/${NC}"
echo ""

for model in $MODELS; do
    central_file="$CENTRAL_OUTPUT/${model}.wav"
    if [ "${RESULTS[$model]}" = "success" ] && [ -f "$central_file" ]; then
        echo -e "  ${GREEN}✓${NC} $model → $OUTPUT_DIR/${model}.wav"
    else
        echo -e "  ${RED}✗${NC} $model"
    fi
done

echo ""
echo -e "Completed: ${GREEN}$SUCCESS_COUNT${NC} succeeded, ${RED}$FAIL_COUNT${NC} failed"
echo "========================================"
