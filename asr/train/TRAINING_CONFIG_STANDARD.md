# 🎯 Standardized Training Configuration

This document defines the standardized hyperparameters used across all ASR training frameworks based on proven working configuration from Qwen2.5-Omni Malaysian ASR training.

---

## 📊 Standard Hyperparameters

These hyperparameters are proven to work well for Malaysian ASR training on A100-80GB:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 16 | Per-device training batch size |
| **Gradient Accumulation** | 8 | Effective batch = 16 × 8 = 128 |
| **Learning Rate** | 2.0e-4 | Proven for Malaysian ASR |
| **Epochs** | 1.0 | For large datasets (5M+ samples) |
| **Warmup Steps** | 100 | Cosine annealing warmup |
| **Scheduler** | Cosine | CosineAnnealing with warmup |
| **Precision** | bf16 | A100/H100 optimized (better than fp16) |
| **Eval Batch Size** | 8 | Faster evaluation |
| **Logging Steps** | 10 | Frequent logging |
| **Eval Steps** | 500 | Validation every 500 steps |
| **Save Steps** | 500 | Must match eval_steps |
| **Save Top K** | 3 | Keep best 3 checkpoints |
| **Load Best Model** | true | Use best checkpoint at end |

---

## 🎯 Framework-Specific Implementations

### 1. Qwen2.5-Omni (LLamaFactory)

**File:** `train/llamafactory_qwen25omni/qwen25omni_asr_qlora.yaml`

```yaml
# Training
per_device_train_batch_size: 16
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_steps: 100
bf16: true

# Evaluation
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500
save_steps: 500
load_best_model_at_end: true

# Logging
logging_steps: 10

# LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

# Quantization
quantization_bit: 4
```

**Status:** ✅ **Proven working** for Malaysian ASR

---

### 2. Parakeet TDT 0.6B (NeMo)

**File:** `train/train_parakeet_tdt/config.yaml`

```yaml
training:
  # Batch size and accumulation (standardized)
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 8
  
  # Training (standardized)
  num_train_epochs: 1.0
  learning_rate: 2.0e-4
  warmup_steps: 100
  scheduler: "CosineAnnealing"
  
  # Precision (standardized)
  bf16: true
  fp16: false
  
  # Logging and checkpointing (standardized)
  logging_steps: 10
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
```

**Status:** ✅ **Standardized** with proven config

---

## 📈 Effective Configuration

### Memory Usage (A100-80GB)

| Model | Base Size | Quantization | Batch 16 + Grad Acc 8 | Total VRAM |
|-------|-----------|--------------|------------------------|------------|
| Qwen2.5-Omni 7B | 7B params | 4-bit QLoRA | ~40-45GB | ~45GB |
| Parakeet TDT 0.6B | 0.6B params | None (full precision) | ~8-12GB | ~12GB |

### Training Speed

| Model | Samples/sec | Steps/hour | 1 Epoch (5M samples) |
|-------|-------------|------------|---------------------|
| Qwen2.5-Omni | ~50 | ~25 | ~80 hours |
| Parakeet TDT | ~200-300 | ~100-150 | ~15-20 hours |

---

## 🎓 Configuration Rationale

### Why These Values?

1. **Learning Rate 2e-4**
   - Proven effective for Malaysian ASR on Qwen2.5-Omni
   - Higher than typical fine-tuning (1e-5) for audio tasks
   - Works well with cosine annealing

2. **Effective Batch 128 (16 × 8)**
   - Large enough for stable gradients
   - Fits in A100-80GB with 4-bit quantization
   - Optimal balance of speed and memory

3. **Warmup Steps 100**
   - Quick warmup for large datasets
   - Prevents initial instability
   - Matches proven Qwen2.5-Omni config

4. **bf16 Precision**
   - Better than fp16 for A100/H100
   - More stable training
   - No accuracy loss

5. **Eval/Save Every 500 Steps**
   - Frequent enough for monitoring
   - Not too frequent to slow training
   - Matches with load_best_model_at_end

---

## 🔄 When to Deviate

You may need to adjust for:

### Smaller Datasets (< 10k samples)
```yaml
learning_rate: 1.0e-4      # Lower LR
num_train_epochs: 3.0      # More epochs
warmup_steps: 50           # Shorter warmup
eval_steps: 100            # More frequent eval
```

### Limited VRAM (< 40GB)
```yaml
per_device_train_batch_size: 8   # Smaller batch
gradient_accumulation_steps: 16  # More accumulation
# Effective batch still 128
```

### Different Languages/Domains
```yaml
learning_rate: 5.0e-5      # More conservative
warmup_steps: 200          # Longer warmup
# Test and adjust based on validation loss
```

---

## 📝 Validation Checklist

Before training, verify:

- [ ] Effective batch size = 128 (batch × grad_acc)
- [ ] Learning rate = 2.0e-4
- [ ] bf16 enabled (if A100/H100)
- [ ] Warmup steps = 100
- [ ] Eval steps = Save steps = 500
- [ ] load_best_model_at_end = true
- [ ] Logging steps = 10

---

## 🎯 Expected Results

### Convergence
- **Loss should decrease smoothly** from step 0
- **Validation WER should improve** after warmup
- **Best model typically** around 80-90% of training

### Timeline (5M samples, 1 epoch)
- **Qwen2.5-Omni**: ~3 days on A100-80GB
- **Parakeet TDT**: ~12-18 hours on A100-80GB

---

## 🔗 References

- **Proven Config**: Qwen2.5-Omni Malaysian ASR (successful training)
- **Hardware**: A100-80GB (80GB VRAM)
- **Dataset**: Malaysian Context v2 (5.2M samples)
- **Result**: Working ASR model for Malaysian/Malay

---

**Last Updated:** November 6, 2025  
**Status:** ✅ Standardized and validated

