# Evaluating Parakeet TDT Models - Complete Guide

## Overview

After training your Parakeet TDT model, you can evaluate it against test datasets to measure Word Error Rate (WER) and Character Error Rate (CER).

The evaluation scripts now support both:
1. ✅ **Base model** from HuggingFace (e.g., `nvidia/parakeet-tdt-0.6b-v3`)
2. ✅ **Fine-tuned model** from local path (your trained `.nemo` file)

---

## Quick Start

### Option 1: Evaluate Base Model (Baseline)

```bash
cd ../../eval

python evaluate.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto
```

### Option 2: Evaluate Your Fine-Tuned Model

```bash
cd ../../eval

# Using final_model.nemo
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --name finetuned-100k
```

### Option 3: Evaluate Specific Checkpoint

```bash
cd ../../eval

# Using a specific checkpoint (has WER in filename)
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/checkpoints/parakeet-tdt--epoch=00-step=5000-val_wer=0.1234.ckpt \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --name checkpoint-5000
```

**Note:** Checkpoints use `.ckpt` extension but can be loaded the same way!

---

## Full Command Reference

### Evaluate Base Model

```bash
cd /path/to/voice-ai/asr/eval

python evaluate.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --name baseline
```

**Output:**
- WER/CER metrics for baseline model
- Results saved to: `outputs/baseline_Parakeet_*/`

### Evaluate Fine-Tuned Model (Final)

```bash
cd /path/to/voice-ai/asr/eval

python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --name finetuned-final
```

**Output:**
- WER/CER metrics for your fine-tuned model
- Results saved to: `outputs/finetuned-final_Parakeet-FineTuned_*/`

### Evaluate Fine-Tuned Model (Directory)

If you provide a directory, it will automatically find the `.nemo` file:

```bash
cd /path/to/voice-ai/asr/eval

python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto
```

### Evaluate on Different Test Sets

#### YTL Malay Test Set

```bash
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test/audio \
  --device auto \
  --name finetuned-ytl
```

#### Malaya Test Set (Larger)

```bash
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --name finetuned-malaya
```

### Quick Testing (Limited Samples)

```bash
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --max-samples 100 \
  --name quick-test
```

---

## Understanding Model Paths

### Where is My Fine-Tuned Model?

After training completes, your models are saved here:

```
train_parakeet_tdt/
└── outputs/
    └── parakeet-tdt-malay-asr/
        ├── final_model.nemo              ← Use this for evaluation
        └── parakeet-tdt-malay-finetuning/
            └── 2024-11-06_20-30-45/
                └── checkpoints/
                    ├── parakeet-tdt--epoch=00-step=1000-val_wer=0.4523.ckpt
                    ├── parakeet-tdt--epoch=00-step=2000-val_wer=0.3214.ckpt
                    └── parakeet-tdt--epoch=00-step=3000-val_wer=0.2876.ckpt
```

### Which Model Should I Use?

#### For Final Evaluation (Recommended):
```bash
--model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo
```

This is the model saved at the end of training (line 500-502 in train_parakeet_tdt.py).

#### For Checkpoint Evaluation:
```bash
# Best checkpoint (lowest WER)
--model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/checkpoints/*-val_wer=0.XXXX.ckpt
```

Find the checkpoint with the lowest `val_wer` value in the filename!

#### For Directory (Auto-detect):
```bash
--model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr
```

Will automatically find and use `final_model.nemo`.

---

## Expected WER Improvements

### Baseline (No Fine-Tuning)

| Dataset | Expected WER |
|---------|--------------|
| Malaya Malay Test | ~25-35% |
| YTL Malay Test | ~30-40% |

### After 100k Samples Fine-Tuning

| Dataset | Expected WER | Improvement |
|---------|--------------|-------------|
| Malaya Malay Test | ~15-25% | ✅ 10-15% better |
| YTL Malay Test | ~18-28% | ✅ 12-15% better |

### After Full Dataset (5.2M) Fine-Tuning

| Dataset | Expected WER | Improvement |
|---------|--------------|-------------|
| Malaya Malay Test | ~8-15% | ✅ 15-25% better |
| YTL Malay Test | ~10-18% | ✅ 20-25% better |

**Note:** Actual results depend on data quality, domain match, and training configuration.

---

## Comparing Multiple Models

### Script: Compare Base vs Fine-Tuned

```bash
cd /path/to/voice-ai/asr/eval

# 1. Evaluate baseline
python evaluate.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --name baseline

# 2. Evaluate fine-tuned
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --name finetuned-100k

# 3. Compare results
python compare_models.py \
  --model1 outputs/baseline_Parakeet_*/evaluation_results.json \
  --model2 outputs/finetuned-100k_Parakeet-FineTuned_*/evaluation_results.json
```

### Manual Comparison

Check the results:

```bash
# Baseline WER
cat outputs/baseline_Parakeet_*/evaluation_results.json | grep -i wer

# Fine-tuned WER
cat outputs/finetuned-100k_Parakeet-FineTuned_*/evaluation_results.json | grep -i wer
```

---

## Output Files

After evaluation, you'll get these files:

```
eval/outputs/
└── finetuned-100k_Parakeet-FineTuned_malaya-malay-test-set_auto_20241106_203045/
    ├── config.json                  # Evaluation configuration
    ├── predictions.json             # All transcriptions
    ├── evaluation_results.json      # WER/CER metrics
    ├── evaluation.log               # Full log
    └── detailed_metrics.json        # Per-sample metrics
```

### Key Metrics in `evaluation_results.json`

```json
{
  "num_samples": 765,
  "wer": {
    "wer": 18.5,           // Overall WER (lower is better)
    "substitutions": 12.3,
    "insertions": 3.2,
    "deletions": 3.0
  },
  "cer": 8.7,              // Character Error Rate
  "avg_rtf": 0.15,         // Real-Time Factor (speed)
  "total_duration": 1234.5,
  "total_time": 185.7
}
```

---

## Troubleshooting

### Error: "No .nemo file found in directory"

**Solution:** Provide the full path to `final_model.nemo`:

```bash
--model /full/path/to/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo
```

### Error: "Model not found"

**Check if training completed:**
```bash
ls -lh train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/
```

You should see `final_model.nemo` (created at end of training).

### Error: "CUDA out of memory" During Evaluation

**Reduce batch size** in `transcribe_parakeet.py` line 39:
```python
batch_size: int = 1,  # Already at minimum
```

Or **use CPU:**
```bash
--device cpu
```

### Model Loads But WER is Bad

**Check:**
1. Did training complete successfully?
2. Check training logs for final validation WER
3. Make sure you're using the same audio format as training data

---

## Batch Evaluation (Multiple Checkpoints)

### Evaluate All Checkpoints

```bash
cd /path/to/voice-ai/asr/eval

# Find all checkpoints
CHECKPOINTS_DIR="../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/checkpoints"

# Evaluate each one
for ckpt in $CHECKPOINTS_DIR/*.ckpt; do
    STEP=$(basename $ckpt | grep -oP 'step=\K[0-9]+')
    echo "Evaluating checkpoint at step $STEP..."
    
    python evaluate.py \
      --model "$ckpt" \
      --test-data test_data/malaya-test/malaya-malay-test-set.json \
      --audio-dir test_data/malaya-test \
      --name "checkpoint-step-$STEP" \
      --max-samples 100  # Quick test on 100 samples
done

echo "✓ All checkpoints evaluated!"
```

---

## Performance Benchmarking

### Speed Comparison

```bash
# Baseline
time python evaluate.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --max-samples 100

# Fine-tuned
time python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --max-samples 100
```

**Expected:** Similar speed (~0.1-0.2 RTF on GPU)

---

## Best Practices

### 1. Always Evaluate on Multiple Test Sets

```bash
# Internal test set (similar to training data)
python evaluate.py --model MODEL --test-data test_data/malaya-test/...

# External test set (different domain)
python evaluate.py --model MODEL --test-data test_data/ytl-malay-test/...
```

### 2. Compare Against Baseline

Always evaluate both base model and fine-tuned model on the same test set for fair comparison.

### 3. Use Meaningful Names

```bash
--name finetuned-100k-epoch1-baseline-comparison
```

Helps organize results in `outputs/` directory.

### 4. Quick Tests First

Use `--max-samples 100` for quick verification before running full evaluation.

---

## Summary Commands

### 📊 Standard Evaluation Workflow

```bash
# Step 1: Go to eval directory
cd /path/to/voice-ai/asr/eval

# Step 2: Evaluate baseline (for comparison)
python evaluate.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --name baseline

# Step 3: Evaluate your fine-tuned model
python evaluate.py \
  --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --name finetuned-100k

# Step 4: Compare results
echo "Baseline WER:"
cat outputs/baseline_Parakeet_*/evaluation_results.json | grep -A 5 '"wer"'

echo "Fine-tuned WER:"
cat outputs/finetuned-100k_Parakeet-FineTuned_*/evaluation_results.json | grep -A 5 '"wer"'
```

---

## What Changed in the Scripts?

### ✅ Updated: `transcribe_parakeet.py`

**Before:** Only supported HuggingFace models
```python
self.model = ASRModel.from_pretrained(model_name=model_name)
```

**After:** Supports both HuggingFace and local `.nemo` files
```python
if model_path.exists() and model_path.suffix == '.nemo':
    self.model = ASRModel.restore_from(restore_path=str(model_path))
else:
    self.model = ASRModel.from_pretrained(model_name=model_name)
```

### ✅ Updated: `evaluate.py`

**Added detection for local `.nemo` files:**
```python
# Check for local .nemo files (fine-tuned Parakeet models)
if model_path.exists() and model_path.suffix == '.nemo':
    return ("transcribe_parakeet.py", "Parakeet-FineTuned")
```

---

## Next Steps

1. **Train your model:**
   ```bash
   cd train_parakeet_tdt
   bash run_training.sh
   ```

2. **Wait for training to complete** (~2 hours for 100k samples)

3. **Evaluate:**
   ```bash
   cd ../../eval
   python evaluate.py --model ../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo ...
   ```

4. **Compare results** and iterate!

---

## Need Help?

- **Training not working?** See `TRAINING_MODES.md`
- **Evaluation errors?** Check `eval/README.md`
- **Low WER improvement?** Try full dataset or check data quality

Good luck with your fine-tuning! 🚀

