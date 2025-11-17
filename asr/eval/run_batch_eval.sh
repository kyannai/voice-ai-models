#!/bin/bash
# Batch Evaluation: Compare 4 ASR Models
# 
# Models:
#   1. OpenAI Whisper Large V3 Turbo
#   2. Malaysian Whisper Large V3 Turbo
#   3. NVIDIA Parakeet TDT 0.6B v3 (base)
#   4. Fine-tuned Parakeet TDT 0.6B (Malay)
#
# Usage:
#   ./run_batch_eval.sh [OPTIONS]
#
# Options:
#   --test-dataset NAME   Test dataset name (default: meso-malaya-test)
#                         Available: meso-malaya-test, ytl-malay-test, seacrowd-asr-malcsc
#   --device DEVICE       Device to use: auto/cuda/cpu (default: auto)
#   --language LANG       Language code (default: ms)
#   --output PATH         Output comparison JSON file (default: outputs/comparison_4models.json)
#   --analysis-dir PATH   Analysis output directory (default: outputs/analysis_4models)
#
# Examples:
#   # Use default (meso-malaya-test)
#   ./run_batch_eval.sh
#
#   # Use a different dataset
#   ./run_batch_eval.sh --test-dataset ytl-malay-test
#
#   # Specify output location
#   ./run_batch_eval.sh --test-dataset seacrowd-asr-malcsc --output outputs/seacrowd_comparison.json

set -e

# Default configuration
TEST_DATASET="meso-malaya-test"
DEVICE="auto"
LANGUAGE="ms"
OUTPUT_FILE="outputs/comparison_4models.json"
ANALYSIS_DIR="outputs/analysis_4models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-dataset)
            TEST_DATASET="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --analysis-dir)
            ANALYSIS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ASR Batch Evaluation - 4 Models"
echo "=========================================="
echo ""

echo "Configuration:"
echo "  Test dataset: $TEST_DATASET"
echo "  Device:       $DEVICE"
echo "  Language:     $LANGUAGE"
echo "  Output:       $OUTPUT_FILE"
echo "  Analysis:     $ANALYSIS_DIR"
echo ""
echo "Models to compare:"
echo "  1. openai/whisper-large-v3-turbo"
echo "  2. mesolitica/Malaysian-whisper-large-v3-turbo-v3"
echo "  3. nvidia/parakeet-tdt-0.6b-v3 (base)"
echo "  4. Fine-tuned Parakeet TDT 0.6B (Malay)"
echo ""

# Check which models need evaluation
echo "Checking existing evaluations..."
echo ""

NEED_EVAL=false
MODELS_TO_EVAL=()

# Check Whisper Base (use more specific pattern to avoid matching Malaysian Whisper)
if ! find outputs -type f -path "*Whisper_whisper-large-v3-turbo_${TEST_DATASET}*/evaluation_results.json" 2>/dev/null | grep -q .; then
    echo "⚠ Whisper Base needs evaluation"
    MODELS_TO_EVAL+=("openai/whisper-large-v3-turbo")
    NEED_EVAL=true
else
    echo "✓ Whisper Base already evaluated"
fi

# Check Malaysian Whisper
if ! find outputs -type f -path "*Malaysian-whisper-large-v3-turbo-v3*${TEST_DATASET}*/evaluation_results.json" 2>/dev/null | grep -q .; then
    echo "⚠ Malaysian Whisper needs evaluation"
    MODELS_TO_EVAL+=("mesolitica/Malaysian-whisper-large-v3-turbo-v3")
    NEED_EVAL=true
else
    echo "✓ Malaysian Whisper already evaluated"
fi

# Check Parakeet (base)
if ! find outputs -type f -path "*parakeet-tdt-0.6b-v3*${TEST_DATASET}*/evaluation_results.json" 2>/dev/null | grep -q .; then
    echo "⚠ Parakeet Base needs evaluation"
    MODELS_TO_EVAL+=("nvidia/parakeet-tdt-0.6b-v3")
    NEED_EVAL=true
else
    echo "✓ Parakeet Base already evaluated"
fi

# Check Fine-tuned Parakeet (directory will contain "final_model.nemo")
if ! find outputs -type f -path "*final_model.nemo*${TEST_DATASET}*/evaluation_results.json" 2>/dev/null | grep -q .; then
    echo "⚠ Fine-tuned Parakeet needs evaluation"
    MODELS_TO_EVAL+=("../train/train_parakeet_tdt/outputs.bak/parakeet-tdt-malay-asr/final_model.nemo")
    NEED_EVAL=true
else
    echo "✓ Fine-tuned Parakeet already evaluated"
fi

echo ""

# Run evaluations if needed
if [ "$NEED_EVAL" = true ]; then
    echo "=========================================="
    echo "Running Evaluations"
    echo "=========================================="
    echo ""
    
    for model in "${MODELS_TO_EVAL[@]}"; do
        echo "Evaluating: $model"
        echo ""
        
        python evaluate.py \
            --model "$model" \
            --test-dataset "$TEST_DATASET" \
            --device "$DEVICE" \
            --language "$LANGUAGE"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ $model completed"
            echo ""
        else
            echo ""
            echo "❌ $model failed"
            echo ""
            echo "Continuing with other models..."
            echo ""
        fi
    done
else
    echo "All models already evaluated. Using existing results."
fi

echo ""
echo "=========================================="
echo "Merging Results"
echo "=========================================="
echo ""

# Merge results (skip evaluation, use existing)
python batch_evaluate.py \
    --test-dataset "$TEST_DATASET" \
    --skip-eval \
    --models whisper-large-v3-turbo malaysian-whisper-large-v3-turbo-v3 parakeet-tdt-0.6b final_model.nemo \
    --output "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to merge results"
    echo ""
    echo "This may happen if some models failed. Check logs above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Analyzing Results"
echo "=========================================="
echo ""

python analyze_comparison.py \
    --comparison "$OUTPUT_FILE" \
    --output-dir "$ANALYSIS_DIR" \
    --top-n 20

echo ""
echo "=========================================="
echo "✅ Batch Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved:"
echo "  📊 Comparison JSON:  $OUTPUT_FILE"
echo "  📈 Summary CSV:      $ANALYSIS_DIR/summary_metrics.csv"
echo "  📄 Detailed CSV:     $ANALYSIS_DIR/detailed_comparison.csv"
echo "  📝 Report:           $ANALYSIS_DIR/comparison_report.md"
echo ""
echo "Quick view:"
echo ""
cat "$ANALYSIS_DIR/summary_metrics.csv"
echo ""
echo "For detailed analysis:"
echo "  cat $ANALYSIS_DIR/comparison_report.md"
echo ""

