# Qwen3-Omni-30B Fine-tuning with LLaMA-Factory

Fine-tune Qwen3-Omni-30B for Malay ASR using LLaMA-Factory's QLoRA implementation.

## Why Qwen3-Omni-30B-Instruct?

- **Largest Qwen Omni model**: 30B parameters (A3B architecture)
- **Best ASR performance**: State-of-the-art for multimodal understanding
- **Instruct-tuned**: Optimized for direct instruction following
- **Memory efficient**: QLoRA 4-bit fits in A100-80GB

## Quick Start

### 1. Setup (One-time)

```bash
cd ~/voice-ai/asr/train/llamafactory_qwen3omni

# Activate training environment
source ../.venv/bin/activate

# Setup LLaMA-Factory (shares with qwen25omni if already installed)
bash setup_llamafactory.sh
```

### 2. Prepare Data

```bash
# Convert ASR data to LLaMA-Factory format
python prepare_data.py
```

### 3. Start Training

```bash
cd LLaMA-Factory

# Start training
llamafactory-cli train ../qwen3omni_asr_qlora.yaml

# Or for background:
nohup llamafactory-cli train ../qwen3omni_asr_qlora.yaml > ../training.log 2>&1 &
```

### 4. Monitor Training

```bash
# Watch logs
tail -f training.log

# Or TensorBoard
tensorboard --logdir outputs/qwen3omni-malaysian-asr-qlora/logs --port 6006
```

## Configuration

Key settings for Qwen3-Omni-30B:

```yaml
### Model
model_name_or_path: Qwen/Qwen3-Omni-30B-A3B-Instruct

### Training (optimized for A100-80GB)
per_device_train_batch_size: 1         # 30B model needs smaller batch
gradient_accumulation_steps: 128       # Effective batch size = 128
learning_rate: 2.0e-4
num_train_epochs: 3.0

### LoRA
lora_rank: 16
lora_alpha: 32

### Quantization (QLoRA 4-bit)
quantization_bit: 4                    # Essential for 30B model
```

## GPU Memory Requirements

| Config | Model Size | Batch | GPU Memory | A100-80GB |
|--------|-----------|-------|------------|-----------|
| QLoRA (4-bit) | 30B | 1 | ~45-50GB | ✅ Fits |
| QLoRA (4-bit) | 30B | 2 | ~70GB | ❌ OOM |

**Your A100-80GB:** Batch size of 1 is optimal for 30B model.

## Training Time Estimate

With 5.2M training samples and A100-80GB:
- **QLoRA (batch_size=1)**: ~5-7 days
- Slower than 7B models but better quality

## Comparison: Qwen3-Omni vs Qwen2.5-Omni

| Feature | Qwen2.5-Omni-7B | Qwen3-Omni-30B-Instruct |
|---------|----------------|------------------------|
| **Parameters** | 7B | 30B (A3B) |
| **Training Speed** | Fast (~2-3 days) | Slower (~5-7 days) |
| **GPU Memory** | ~11-15GB | ~45-50GB |
| **ASR Quality** | Excellent | Best |
| **Variant** | Base | Instruct-tuned |

## Troubleshooting

### OOM (Out of Memory)?
```yaml
# Already at minimum batch size (1)
# If still OOM, reduce max_samples for testing
max_samples: 10000
```

### Training too slow?
- 30B models are inherently slower
- Consider using Qwen2.5-Omni-7B instead for faster iteration
- Or increase GPU count if available

### Dataset issues?
- Same format as qwen25omni
- Ensure audio paths are absolute
- Run `prepare_data.py` to regenerate

## Evaluate Trained Model

After training:

```bash
cd ~/voice-ai/asr/eval

# Evaluate the LoRA adapter
python evaluate.py \
  --model ~/voice-ai/asr/train/llamafactory_qwen3omni/LLaMA-Factory/outputs/qwen3omni-malaysian-asr-qlora \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto
```

## Multi-GPU Training

Qwen3-Omni-30B can benefit from multi-GPU:

```bash
# Automatic multi-GPU (LLaMA-Factory detects all GPUs)
llamafactory-cli train ../qwen3omni_asr_qlora.yaml

# Manual specification
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train ../qwen3omni_asr_qlora.yaml
```

## When to Use Qwen3-Omni-30B?

**Use Qwen3-Omni-30B if:**
- ✅ You need the absolute best ASR quality
- ✅ You have time for longer training (~1 week)
- ✅ You have 80GB GPU available
- ✅ Model size isn't a deployment concern

**Use Qwen2.5-Omni-7B if:**
- ✅ You need faster training (~2-3 days)
- ✅ You want smaller deployment size
- ✅ Good ASR quality is sufficient (only 0.2% WER difference)
- ✅ You want to iterate quickly

## Resources

- [Qwen3-Omni Model Card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen Documentation](https://qwen.readthedocs.io/)

## Expected WER Performance

Based on LibriSpeech benchmarks:
- **Qwen2.5-Omni-7B**: 1.6/3.4 (test-clean/test-other)
- **Qwen3-Omni-30B**: Expected slightly better
- **After fine-tuning**: Significant improvement on Malaysian Malay

