# Parakeet TDT Training Modes

## Quick Reference: Which Config Should I Use?

### Current Setup (100k samples) - Testing ✅
**Config:** `config.yaml` (current)
- **Dataset Size:** 100k samples  
- **Memory:** ~45-50GB
- **Training Time:** ~2 hours
- **Use Case:** Quick testing, debugging, proof-of-concept
- **Command:** `bash run_training.sh`

### Full Dataset (5.2M samples) - Production 🚀
**Config:** `config_full_dataset.yaml`
- **Dataset Size:** 5.2M samples (full)
- **Memory:** ~50-55GB (with optimizations)
- **Training Time:** ~24 hours
- **Use Case:** Production model, best accuracy
- **Setup:**
  ```bash
  bash setup_full_dataset.sh  # Install bitsandbytes
  cp config_full_dataset.yaml config.yaml
  bash run_training.sh
  ```

---

## Memory Optimization Comparison

| Config | Samples | Optimizer | Grad Checkpoint | Batch Size | Memory | Speed |
|--------|---------|-----------|-----------------|------------|--------|-------|
| `config.yaml` (current) | 100k | AdamW | No | 8 | ~45GB | 100% |
| `config_full_dataset.yaml` | 5.2M | 8-bit AdamW | Yes | 20 | ~55GB | 85% |

---

## What Was Added for Full Dataset Training?

### 1. Memory Optimizations in Code (`train_parakeet_tdt.py`)

**8-bit Optimizer Support:**
- Reduces optimizer memory by 75%
- Uses `bitsandbytes` library
- Automatically falls back to regular AdamW if unavailable

**Gradient Checkpointing Support:**
- Reduces activation memory by 30-50%
- Trades compute for memory (15% slower)
- Enabled via config flag

**CUDA Memory Management:**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `run_training.sh`
- Prevents fragmentation (the 16GB reserved but unallocated issue)

### 2. New Configuration Files

**`config_full_dataset.yaml`** - Optimized for 5.2M samples:
```yaml
model:
  gradient_checkpointing: true  # 30-50% memory savings

data:
  max_samples: -1  # Use all samples

training:
  optimizer: "adamw_8bit"  # 75% optimizer memory savings
  per_device_train_batch_size: 20
  gradient_accumulation_steps: 6
  # Effective batch: 20 * 6 = 120
```

**`FULL_DATASET_TRAINING.md`** - Comprehensive guide with:
- 5 different strategies for full dataset training
- Memory calculations and expected usage
- Multi-GPU setup instructions
- Troubleshooting guide
- Progressive training strategy

### 3. Setup Scripts

**`setup_full_dataset.sh`** - One-command setup:
- Installs `bitsandbytes`
- Provides usage instructions
- Shows expected memory/time

---

## When to Use Each Mode

### Use `config.yaml` (100k samples) when:
- ✅ Testing the training pipeline
- ✅ Debugging issues
- ✅ Experimenting with hyperparameters
- ✅ Quick iteration cycles
- ✅ Limited time/resources

### Use `config_full_dataset.yaml` (5.2M samples) when:
- ✅ Training production model
- ✅ Maximizing accuracy
- ✅ Have full GPU access (A100-80GB)
- ✅ Can afford 24+ hour training
- ✅ Need comprehensive language coverage

---

## Progressive Training Strategy (Recommended)

If you're unsure or want to validate quality incrementally:

```bash
# Step 1: Test with 100k samples
cp config.yaml config.yaml.backup
sed -i 's/max_samples:.*/max_samples: 100000/' config.yaml
bash run_training.sh

# Step 2: Scale to 500k samples
sed -i 's/max_samples:.*/max_samples: 500000/' config.yaml
sed -i 's/resume_from_checkpoint: false/resume_from_checkpoint: true/' config.yaml
bash run_training.sh

# Step 3: Scale to 1M samples
sed -i 's/max_samples:.*/max_samples: 1000000/' config.yaml
bash run_training.sh

# Step 4: Full dataset (5.2M)
# Install optimizations first
bash setup_full_dataset.sh
cp config_full_dataset.yaml config.yaml
bash run_training.sh
```

This approach:
- ✅ Validates training works at each scale
- ✅ Catches issues early
- ✅ Allows quality checks between stages
- ✅ Can stop at any stage if satisfied

---

## Troubleshooting

### Still Getting OOM?

1. **Reduce batch size further:**
   ```yaml
   per_device_train_batch_size: 12
   gradient_accumulation_steps: 10
   ```

2. **Enable both optimizations:**
   ```yaml
   model:
     gradient_checkpointing: true
   training:
     optimizer: "adamw_8bit"
   ```

3. **Reduce eval batch size:**
   ```yaml
   per_device_eval_batch_size: 2
   ```

4. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Optimizer Not Working?

If you see warning about `adamw_8bit` not available:
```bash
pip install bitsandbytes
```

Or run:
```bash
bash setup_full_dataset.sh
```

### Training Too Slow?

1. **Use multi-GPU if available:**
   ```yaml
   training:
     num_gpus: 2
   ```

2. **Increase dataloader workers:**
   ```yaml
   training:
     dataloader_num_workers: 16
   ```

3. **Disable gradient checkpointing:**
   ```yaml
   model:
     gradient_checkpointing: false
   ```
   (But will need to reduce batch size to avoid OOM)

---

## Summary

**For your current situation:**

1. ✅ **Current config (`config.yaml`)** works fine for 100k samples with batch size 8
2. 🚀 **To train on full dataset**, use `config_full_dataset.yaml`:
   ```bash
   bash setup_full_dataset.sh
   cp config_full_dataset.yaml config.yaml
   bash run_training.sh
   ```

**Memory optimizations added:**
- 8-bit AdamW optimizer (75% optimizer memory reduction)
- Gradient checkpointing (30-50% activation memory reduction)
- CUDA memory fragmentation prevention
- **Total savings: ~25-30GB**

**Expected result:**
- Full 5.2M dataset training in ~50-55GB memory
- ~24 hours on A100-80GB
- Same accuracy as baseline (no quality loss from optimizations)

See `FULL_DATASET_TRAINING.md` for detailed strategies and options!

