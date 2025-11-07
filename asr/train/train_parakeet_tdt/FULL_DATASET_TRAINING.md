# Training Parakeet TDT on Full Dataset (5.2M Samples)

## Problem
Training on the full dataset (5.2M samples) causes OOM on A100-80GB due to:
- Large optimizer states (AdamW stores momentum + variance for all parameters)
- Gradient accumulation memory
- Model activations and intermediate tensors

## Solutions (In Order of Preference)

### Solution 1: Use 8-bit Optimizers (Recommended) ⭐

**Benefits:**
- 75% reduction in optimizer memory (8-bit vs 32-bit)
- Minimal accuracy impact
- Easy to implement

**Implementation:**

1. Install `bitsandbytes`:
```bash
pip install bitsandbytes
```

2. Update `config.yaml`:
```yaml
training:
  optimizer: "adamw_8bit"  # Change from "adamw" to "adamw_8bit"
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 8
```

3. Modify `train_parakeet_tdt.py` line 206:
```python
'optim': {
    'name': '8bit_adamw' if config['training'].get('optimizer') == 'adamw_8bit' else config['training'].get('optimizer', 'adamw'),
    ...
}
```

**Expected Memory Savings:** ~20-25GB on A100-80GB

---

### Solution 2: Gradient Checkpointing (Trade Speed for Memory)

**Benefits:**
- 30-50% reduction in activation memory
- No accuracy loss
- ~20-30% slower training

**Implementation:**

1. Add to `config.yaml`:
```yaml
model:
  gradient_checkpointing: true  # Add this line

training:
  per_device_train_batch_size: 24  # Can use larger batch
  per_device_eval_batch_size: 12
  gradient_accumulation_steps: 5
```

2. Modify `train_parakeet_tdt.py` in `create_nemo_config()` around line 180:
```python
def create_nemo_config(config: Dict) -> OmegaConf:
    nemo_config = {
        'model': {
            'init_from_pretrained_model': config['model']['name'],
            
            # Add gradient checkpointing
            'encoder': {
                'gradient_checkpointing': config['model'].get('gradient_checkpointing', False),
            },
            'decoder': {
                'gradient_checkpointing': config['model'].get('gradient_checkpointing', False),
            },
            
            'train_ds': {
                ...
```

**Expected Memory Savings:** ~15-20GB

---

### Solution 3: CPU Offloading (Slowest but Most Memory-Efficient)

**Benefits:**
- Can train unlimited dataset size
- No code changes needed
- 2-3x slower training

**Implementation:**

Use DeepSpeed ZeRO Stage 2 with CPU offloading:

1. Install DeepSpeed:
```bash
pip install deepspeed
```

2. Create `deepspeed_config.json`:
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```

3. Modify `run_training.sh`:
```bash
# Replace: python train_parakeet_tdt.py --config config.yaml
# With:
deepspeed --num_gpus=1 train_parakeet_tdt.py --config config.yaml
```

**Expected Memory Savings:** ~30-40GB (but much slower)

---

### Solution 4: Multi-GPU Training (If Available)

**Benefits:**
- Linear memory scaling (2 GPUs = 2x memory)
- Faster training with data parallelism
- No accuracy loss

**Implementation:**

1. Update `config.yaml`:
```yaml
training:
  num_gpus: 2  # Or 4, 8, etc.
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
```

2. Run with:
```bash
bash run_training.sh
```

NeMo automatically handles DDP (Distributed Data Parallel).

**Memory per GPU:** ~40GB on 2x A100-80GB

---

### Solution 5: Progressive Training (Recommended for Limited Hardware)

**Benefits:**
- Train incrementally on full dataset
- Can verify quality at each stage
- Flexible and safe

**Implementation:**

```bash
# Stage 1: 100k samples
sed -i 's/max_samples:.*/max_samples: 100000/' config.yaml
bash run_training.sh

# Stage 2: Resume and train on 500k samples
sed -i 's/max_samples:.*/max_samples: 500000/' config.yaml
sed -i 's/resume_from_checkpoint: false/resume_from_checkpoint: true/' config.yaml
bash run_training.sh

# Stage 3: Resume and train on 1M samples
sed -i 's/max_samples:.*/max_samples: 1000000/' config.yaml
bash run_training.sh

# Stage 4: Full dataset (5.2M)
sed -i 's/max_samples:.*/max_samples: -1/' config.yaml
bash run_training.sh
```

---

## Recommended Combination ⚡

For best results, combine multiple strategies:

```yaml
# config.yaml - Optimized for Full Dataset
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  gradient_checkpointing: true  # Add this

data:
  max_samples: -1  # Full dataset

training:
  optimizer: "adamw_8bit"  # Use 8-bit optimizer
  per_device_train_batch_size: 20
  per_device_eval_batch_size: 10
  gradient_accumulation_steps: 6
  # Effective batch size: 20 * 6 = 120
```

**Expected Total Memory:** ~50-55GB (fits comfortably in A100-80GB)
**Training Speed:** ~85% of baseline (minimal slowdown)

---

## Memory Calculation Reference

For **AdamW** optimizer on **0.6B parameter** model:

| Component | FP32 | BF16 | 8-bit | Notes |
|-----------|------|------|-------|-------|
| Model Parameters | 2.4 GB | 1.2 GB | 1.2 GB | Frozen during training |
| Gradients | 2.4 GB | 1.2 GB | 1.2 GB | Same dtype as params |
| Optimizer State (momentum) | 2.4 GB | 2.4 GB | 0.6 GB | Always FP32 normally |
| Optimizer State (variance) | 2.4 GB | 2.4 GB | 0.6 GB | Always FP32 normally |
| Activations (batch=16) | ~15 GB | ~8 GB | ~6 GB | With gradient checkpointing |
| **Total Base** | ~25 GB | ~15 GB | ~10 GB | |

With **1M samples**, optimizer state stays in memory throughout training, adding significant overhead.

---

## Quick Decision Guide

**Choose based on your constraints:**

1. **Have 1 A100-80GB?** → Use 8-bit optimizer + gradient checkpointing (Solution 1 + 2)
2. **Have 2+ GPUs?** → Multi-GPU training (Solution 4)
3. **Speed not critical?** → DeepSpeed with CPU offload (Solution 3)
4. **Want gradual validation?** → Progressive training (Solution 5)

---

## Testing Your Configuration

Before full training, test memory usage:

```bash
# Add to config.yaml temporarily:
training:
  max_steps: 100  # Only 100 steps

# Run and monitor memory:
watch -n 1 nvidia-smi

# If memory stable for 100 steps, remove max_steps and run full training
```

---

## Expected Training Times (A100-80GB)

| Strategy | Memory | Speed | Full Dataset (5.2M) |
|----------|--------|-------|---------------------|
| Baseline (OOM) | 80GB+ | 100% | ❌ Crashes |
| 8-bit Optimizer | ~55GB | 95% | ~24 hours |
| + Grad Checkpoint | ~45GB | 80% | ~30 hours |
| DeepSpeed CPU Offload | ~35GB | 40% | ~60 hours |
| 2x GPU (DDP) | ~40GB each | 180% | ~13 hours |

---

## Troubleshooting

### Still Getting OOM?

1. **Check memory fragmentation:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

2. **Reduce eval batch size:**
```yaml
per_device_eval_batch_size: 2  # Minimal
```

3. **Clear cache between epochs:**
Add to `train_parakeet_tdt.py` (around line 350):
```python
torch.cuda.empty_cache()
```

4. **Use mixed precision more aggressively:**
```yaml
training:
  bf16: true
  fp16: false
```

### Training Too Slow?

1. Increase `dataloader_num_workers` (default: 8)
2. Use faster storage (NVMe SSD) for audio files
3. Pre-compute features (not recommended for NeMo ASR)

---

## Summary

**Minimum viable setup for full dataset:**
- 8-bit optimizer: **Required**
- Gradient checkpointing: **Recommended**
- Batch size: 16-24
- Memory: ~50GB

**Optimal setup:**
- 2x A100-80GB with DDP
- 8-bit optimizer
- Batch size: 32 per GPU
- Training time: ~13 hours

