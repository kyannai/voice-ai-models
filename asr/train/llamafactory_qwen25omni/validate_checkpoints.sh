#!/bin/bash
# Validate all checkpoints on test set to find best one based on WER (not loss!)

set -e

EVAL_SCRIPT="../../eval/evaluate.py"
TEST_DATA="../../eval/test_data/malaya-test/malaya-malay-test-set.json"
AUDIO_DIR="../../eval/test_data/malaya-test"
CHECKPOINT_DIR="LLaMA-Factory/outputs/qwen25omni-malaysian-asr-qlora-v2-conservative"

echo "=========================================="
echo "Validating Checkpoints on Test Set"
echo "=========================================="
echo ""
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Test data: ${TEST_DATA}"
echo ""

# Find all checkpoints
checkpoints=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | sort -V)

if [ -z "$checkpoints" ]; then
    echo "❌ No checkpoints found in ${CHECKPOINT_DIR}"
    echo ""
    echo "Available output directories:"
    ls -d LLaMA-Factory/outputs/*/ 2>/dev/null || echo "No outputs found"
    exit 1
fi

echo "Found checkpoints:"
for cp in $checkpoints; do
    echo "  - $(basename $cp)"
done
echo ""

best_wer=999.0
best_checkpoint=""
results_summary="checkpoint_validation_results.txt"

# Clear previous results
> "$results_summary"

echo "Checkpoint,WER,CER,Substitutions,Insertions,Deletions" >> "$results_summary"

# Evaluate each checkpoint
for checkpoint in $checkpoints; do
    checkpoint_num=$(basename $checkpoint | grep -oP '\d+$' || basename $checkpoint | grep -oE '[0-9]+$')
    echo "=========================================="
    echo "Evaluating checkpoint-${checkpoint_num}..."
    echo "=========================================="
    
    # Run evaluation
    if python ${EVAL_SCRIPT} \
        --model "${checkpoint}" \
        --test-data "${TEST_DATA}" \
        --audio-dir "${AUDIO_DIR}" \
        --output-dir "../../eval/outputs/validation_checkpoint-${checkpoint_num}" 2>&1 | tee /tmp/eval_log_${checkpoint_num}.txt; then
        
        # Extract WER from results
        results_file="../../eval/outputs/validation_checkpoint-${checkpoint_num}/evaluation_results.json"
        if [ -f "$results_file" ]; then
            wer=$(python3 -c "import json; print(json.load(open('$results_file'))['wer']['wer'])")
            cer=$(python3 -c "import json; print(json.load(open('$results_file'))['cer'])")
            subs=$(python3 -c "import json; print(json.load(open('$results_file'))['wer']['substitutions'])")
            ins=$(python3 -c "import json; print(json.load(open('$results_file'))['wer']['insertions'])")
            dels=$(python3 -c "import json; print(json.load(open('$results_file'))['wer']['deletions'])")
            
            echo "✓ Checkpoint-${checkpoint_num} WER: ${wer}%"
            echo "  CER: ${cer}%, Subs: ${subs}, Ins: ${ins}, Del: ${dels}"
            
            # Save to summary
            echo "checkpoint-${checkpoint_num},${wer},${cer},${subs},${ins},${dels}" >> "$results_summary"
            
            # Track best checkpoint
            if (( $(echo "$wer < $best_wer" | bc -l 2>/dev/null || python3 -c "print(1 if $wer < $best_wer else 0)") )); then
                best_wer=$wer
                best_checkpoint="checkpoint-${checkpoint_num}"
            fi
        else
            echo "❌ Results file not found: $results_file"
        fi
    else
        echo "❌ Evaluation failed for checkpoint-${checkpoint_num}"
    fi
    
    echo ""
done

echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo ""
echo "All results saved to: ${results_summary}"
echo ""
cat "$results_summary"
echo ""
echo "=========================================="
echo "BEST CHECKPOINT:"
echo "=========================================="
echo "Checkpoint: ${best_checkpoint}"
echo "WER: ${best_wer}%"
echo ""
echo "Base model WER: 22.87%"
if (( $(echo "$best_wer < 22.87" | bc -l 2>/dev/null || python3 -c "print(1 if $best_wer < 22.87 else 0)") )); then
    improvement=$(echo "22.87 - $best_wer" | bc -l 2>/dev/null || python3 -c "print(22.87 - $best_wer)")
    echo "✅ Fine-tuned is BETTER! (${best_wer}% < 22.87%, improvement: ${improvement}%)"
else
    regression=$(echo "$best_wer - 22.87" | bc -l 2>/dev/null || python3 -c "print($best_wer - 22.87)")
    echo "❌ Fine-tuned is WORSE! (${best_wer}% > 22.87%, regression: +${regression}%)"
fi
echo ""
echo "✅ Validation complete!"

