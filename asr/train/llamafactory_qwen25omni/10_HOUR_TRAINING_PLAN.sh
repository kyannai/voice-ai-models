#!/bin/bash
# 10-Hour Training Plan for Qwen2.5-Omni ASR Fine-tuning
# This plan validates the approach first, then scales up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "🚀 10-HOUR TRAINING PLAN"
echo "========================================"
echo ""
echo "Timeline:"
echo "  Phase 1: Quick validation (1-2 hours)"
echo "  Phase 2: Full training (6-8 hours)"
echo "  Phase 3: Validation & comparison (1 hour)"
echo ""
echo "Start time: $(date)"
echo ""

# ========================================
# PHASE 1: Quick Validation (1-2 hours)
# ========================================
echo -e "${BLUE}========================================"
echo "PHASE 1: Quick Validation (1-2 hours)"
echo -e "========================================${NC}"
echo ""
echo "Testing with 10K samples to validate fixes..."
echo ""

cd ~/voice-ai/asr/train/llamafactory_qwen25omni

# Prepare data (if not already done)
if [ ! -f "LLaMA-Factory/data/malaysian_asr_train.json" ]; then
    echo "📝 Preparing data with simplified prompt..."
    python prepare_data.py
else
    echo "✓ Data already prepared"
fi

echo ""
echo "🏋️ Starting Phase 1 training (10K samples, ~1-2 hours)..."
echo ""

cd LLaMA-Factory

# Train with conservative config (10K samples)
llamafactory-cli train ../qwen25omni_asr_qlora_v2_conservative.yaml

echo ""
echo -e "${GREEN}✓ Phase 1 complete!${NC}"
echo ""

# Quick validation of Phase 1
echo "🔍 Quick validation of Phase 1 checkpoints..."
cd ~/voice-ai/asr/eval

# Find latest checkpoint from Phase 1
PHASE1_CHECKPOINT=$(ls -td ~/voice-ai/asr/train/llamafactory_qwen25omni/LLaMA-Factory/outputs/qwen25omni-malaysian-asr-qlora-v2-conservative/checkpoint-* 2>/dev/null | head -1)

if [ -n "$PHASE1_CHECKPOINT" ]; then
    echo "Evaluating: $PHASE1_CHECKPOINT"
    
    python evaluate.py \
        --model "$PHASE1_CHECKPOINT" \
        --test-data test_data/malaya-test/malaya-malay-test-set.json \
        --audio-dir test_data/malaya-test \
        --output-dir outputs/phase1_validation
    
    # Extract WER
    PHASE1_WER=$(python3 -c "import json; print(json.load(open('outputs/phase1_validation/evaluation_results.json'))['wer']['wer'])")
    
    echo ""
    echo -e "${YELLOW}========================================"
    echo "PHASE 1 RESULTS:"
    echo "========================================${NC}"
    echo "Checkpoint: $(basename $PHASE1_CHECKPOINT)"
    echo "WER: ${PHASE1_WER}%"
    echo "Base model WER: 22.87%"
    echo ""
    
    # Check if we should continue
    if (( $(echo "$PHASE1_WER < 25" | bc -l) )); then
        echo -e "${GREEN}✅ Phase 1 looks good! WER < 25%${NC}"
        echo -e "${GREEN}Proceeding to Phase 2 (full training)...${NC}"
        echo ""
        CONTINUE_PHASE2=true
    else
        echo -e "${RED}⚠️  Phase 1 WER >= 25% (not better than checkpoint-704)${NC}"
        echo "Options:"
        echo "  1. Continue anyway (maybe needs more data)"
        echo "  2. Stop and investigate"
        echo ""
        read -p "Continue to Phase 2? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            CONTINUE_PHASE2=true
        else
            CONTINUE_PHASE2=false
        fi
    fi
else
    echo -e "${YELLOW}⚠️  No Phase 1 checkpoint found, proceeding to Phase 2 anyway${NC}"
    CONTINUE_PHASE2=true
fi

if [ "$CONTINUE_PHASE2" = false ]; then
    echo -e "${RED}Stopping at Phase 1. Please investigate before continuing.${NC}"
    exit 0
fi

# ========================================
# PHASE 2: Full Training (6-8 hours)
# ========================================
echo ""
echo -e "${BLUE}========================================"
echo "PHASE 2: Full Training (6-8 hours)"
echo -e "========================================${NC}"
echo ""
echo "Training with 200K samples (optimal for 10 hours)..."
echo "Current time: $(date)"
echo ""

cd ~/voice-ai/asr/train/llamafactory_qwen25omni/LLaMA-Factory

# Train with 10-hour plan config (200K samples)
llamafactory-cli train ../qwen25omni_asr_qlora_v2_10hour_plan.yaml

echo ""
echo -e "${GREEN}✓ Phase 2 complete!${NC}"
echo "Current time: $(date)"
echo ""

# ========================================
# PHASE 3: Validation & Comparison (1 hour)
# ========================================
echo ""
echo -e "${BLUE}========================================"
echo "PHASE 3: Validation & Comparison (1 hour)"
echo -e "========================================${NC}"
echo ""

cd ~/voice-ai/asr/train/llamafactory_qwen25omni

# Update validation script to check correct directory
sed -i.bak 's|qwen25omni-malaysian-asr-qlora-v2-conservative|qwen25omni-malaysian-asr-qlora-v2-10hour|g' validate_checkpoints.sh

echo "🔍 Validating all Phase 2 checkpoints..."
./validate_checkpoints.sh

# Restore original validation script
mv validate_checkpoints.sh.bak validate_checkpoints.sh

echo ""
echo -e "${GREEN}========================================"
echo "✅ 10-HOUR TRAINING PLAN COMPLETE!"
echo "========================================${NC}"
echo ""
echo "End time: $(date)"
echo ""

# Print summary
if [ -f "checkpoint_validation_results.txt" ]; then
    echo "📊 Checkpoint Validation Results:"
    cat checkpoint_validation_results.txt
    echo ""
    
    # Find best checkpoint
    BEST_LINE=$(tail -n +2 checkpoint_validation_results.txt | sort -t',' -k2 -n | head -1)
    BEST_CHECKPOINT=$(echo $BEST_LINE | cut -d',' -f1)
    BEST_WER=$(echo $BEST_LINE | cut -d',' -f2)
    
    echo -e "${GREEN}🏆 BEST CHECKPOINT: $BEST_CHECKPOINT${NC}"
    echo -e "${GREEN}🎯 WER: ${BEST_WER}%${NC}"
    echo ""
    
    if (( $(echo "$BEST_WER < 22.87" | bc -l) )); then
        IMPROVEMENT=$(echo "22.87 - $BEST_WER" | bc -l)
        echo -e "${GREEN}✅ SUCCESS! Better than base model!${NC}"
        echo -e "${GREEN}   Improvement: ${IMPROVEMENT}% WER reduction${NC}"
    else
        REGRESSION=$(echo "$BEST_WER - 22.87" | bc -l)
        echo -e "${YELLOW}⚠️  Still worse than base model${NC}"
        echo -e "${YELLOW}   Regression: +${REGRESSION}% WER${NC}"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Review checkpoint_validation_results.txt"
echo "  2. Compare best checkpoint with base model:"
echo "     cd ~/voice-ai/asr/eval"
echo "     python compare_models.py \\"
echo "       --result1 outputs/.../base_model/evaluation_results.json \\"
echo "       --result2 outputs/validation_checkpoint-XXX/evaluation_results.json \\"
echo "       --output final_comparison.md"
echo ""

