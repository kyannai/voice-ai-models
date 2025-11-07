#!/bin/bash
# Quick reference commands for transcribing with NVIDIA Parakeet models
# Choose the appropriate command based on your use case

# =============================================================================
# SETUP
# =============================================================================
# Navigate to transcribe directory
cd ~/voice-ai/asr/eval/transcribe

# Install NeMo toolkit if not already installed
# pip install nemo_toolkit[asr]

# =============================================================================
# OPTION 1: Quick Test (10 samples) - Use this FIRST to verify everything works
# =============================================================================
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-tdt-0.6b-v3-test \
  --device cuda \
  --max-samples 10

# =============================================================================
# OPTION 2: Full Evaluation with Parakeet TDT 0.6B v3 (Recommended)
# =============================================================================
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-tdt-0.6b-v3 \
  --device cuda

# =============================================================================
# OPTION 3: With Custom Batch Size (for faster processing)
# =============================================================================
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-tdt-0.6b-v3 \
  --device cuda \
  --batch-size 4

# =============================================================================
# OPTION 4: CPU Mode (if CUDA not available)
# =============================================================================
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-tdt-0.6b-v3 \
  --device cpu

# =============================================================================
# OPTION 5: Other Parakeet Models
# =============================================================================
# Parakeet TDT 1.1B
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-1.1b \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-tdt-1.1b \
  --device cuda

# Parakeet RNNT 0.6B
python transcribe_parakeet.py \
  --model nvidia/parakeet-rnnt-0.6b \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-rnnt-0.6b \
  --device cuda

# =============================================================================
# OPTION 6: Local Fine-tuned Model
# =============================================================================
python transcribe_parakeet.py \
  --model ~/voice-ai/asr/train/parakeet-finetuned/final_model \
  --test-data ~/voice-ai/asr/eval/test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ~/voice-ai/asr/eval/test_data/ytl-malay-test \
  --output-dir ./results/parakeet-finetuned \
  --device cuda

# =============================================================================
# AFTER TRANSCRIPTION: Calculate Metrics
# =============================================================================
cd ../shared
python calculate_metrics.py ../transcribe/results/parakeet-tdt-0.6b-v3/predictions.json

# =============================================================================
# COMPARE WITH OTHER MODELS
# =============================================================================
# 1. Transcribe with both models (e.g., Parakeet vs Whisper)
# 2. Compare results
cd ../shared
python analyze_results.py \
  --result1 ../transcribe/results/whisper-small/predictions.json \
  --result2 ../transcribe/results/parakeet-tdt-0.6b-v3/predictions.json \
  --output-dir ./comparison

# =============================================================================
# KEY NOTES:
# =============================================================================
# 1. Parakeet models are built on NeMo framework - install nemo_toolkit[asr]
# 2. Start with --max-samples 10 to test first
# 3. Use --device cuda for faster processing (GPU recommended)
# 4. TDT (Token-and-Duration Transducer) models provide automatic punctuation
# 5. Parakeet TDT 0.6B v3 is optimized for speed and accuracy
# 6. Batch size can be increased for faster processing with more VRAM
# 7. Models automatically download from HuggingFace on first use
# =============================================================================

