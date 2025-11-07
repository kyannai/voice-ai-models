#!/bin/bash
# Quick reference commands for transcribing with Qwen2.5-Omni
# Choose the appropriate command based on your use case

# =============================================================================
# SETUP
# =============================================================================
# Navigate to transcribe directory
cd ~/voice-ai/asr/eval/transcribe

# =============================================================================
# OPTION 1: Quick Test (10 samples) - Use this FIRST to verify everything works
# =============================================================================
python transcribe_qwen25omni.py \
  --model ~/voice-ai/asr/train/train_qwen25omni/outputs/qwen25omni-malay-asr/checkpoint-1000 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/checkpoint-1000-test \
  --device cuda \
  --max-samples 10 \
  --asr-prompt "Transcribe this Malay audio to text:"

# =============================================================================
# OPTION 2: Full Evaluation with Training Checkpoint (LoRA adapter)
# =============================================================================
python transcribe_qwen25omni.py \
  --model ~/voice-ai/asr/train/train_qwen25omni/outputs/qwen25omni-malay-asr/checkpoint-1000 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/checkpoint-1000 \
  --device cuda \
  --asr-prompt "Transcribe this Malay audio to text:"

# =============================================================================
# OPTION 3: Full Evaluation with Final Merged Model (Recommended for production)
# =============================================================================
python transcribe_qwen25omni.py \
  --model ~/voice-ai/asr/train/train_qwen25omni/outputs/qwen25omni-malay-asr/final_model \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/final-model \
  --device cuda \
  --asr-prompt "Transcribe this Malay audio to text:"

# =============================================================================
# OPTION 4: Evaluate Base Model (For comparison)
# =============================================================================
python transcribe_qwen25omni.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/base-model \
  --device cuda \
  --asr-prompt "Transcribe this Malay audio to text:"

# =============================================================================
# OPTION 5: With Manual Base Model Specification (if auto-detect fails)
# =============================================================================
python transcribe_qwen25omni.py \
  --model ~/voice-ai/asr/train/train_qwen25omni/outputs/qwen25omni-malay-asr/checkpoint-1000 \
  --base-model Qwen/Qwen2.5-Omni-7B \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/checkpoint-1000 \
  --device cuda \
  --asr-prompt "Transcribe this Malay audio to text:"

# =============================================================================
# AFTER TRANSCRIPTION: Calculate Metrics
# =============================================================================
cd ../shared
python calculate_metrics.py ../transcribe/results/checkpoint-1000/predictions.json

# =============================================================================
# COMPARE TWO MODELS
# =============================================================================
# 1. Transcribe with both models (run commands above for both)
# 2. Compare results
cd ../shared
python analyze_results.py \
  --result1 ../transcribe/results/base-model/predictions.json \
  --result2 ../transcribe/results/checkpoint-1000/predictions.json \
  --output-dir ./comparison

# =============================================================================
# KEY NOTES:
# =============================================================================
# 1. Always use the SAME --asr-prompt that you used during training
# 2. Start with --max-samples 10 to test first
# 3. Use --device cuda for faster processing (10-20x faster than CPU)
# 4. Checkpoint paths may vary - check your training output directory
# 5. LoRA checkpoints (checkpoint-XXX) are auto-detected and merged
# 6. final_model is already merged and ready for production use
# =============================================================================

