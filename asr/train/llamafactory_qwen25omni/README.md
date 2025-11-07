# Qwen2.5-Omni-7B Fine-tuning with LLaMA-Factory

Fine-tune Qwen2.5-Omni-7B for Malay ASR using LLaMA-Factory's QLoRA implementation.

## Why LLaMA-Factory?

- ✅ **Officially supports Qwen2.5-Omni** (unlike Unsloth/standard PEFT)
- ✅ **QLoRA support**: 4-bit quantization + LoRA (~11GB GPU memory)
- ✅ **Production-ready**: Battle-tested on many models
- ✅ **Easy to use**: Simple YAML configuration
- ✅ **Active development**: Regular updates and bug fixes

## Quick Start

### 1. Setup (One-time)

```bash
cd ~/voice-ai/asr/train/llamafactory_qwen25omni

# Activate your training environment
source ../.venv/bin/activate

# Run setup script
bash setup_llamafactory.sh
```

This will:
- Clone LLaMA-Factory repository
- Install all dependencies
- Install DeepSpeed for distributed training

### 2. Prepare Data

```bash
# Convert your ASR data to LLaMA-Factory format
python prepare_data.py
```

This converts your Malaysian ASR dataset to LLaMA-Factory's conversation format with audio tags.

### 3. Start Training

```bash
cd LLaMA-Factory

# Start training with QLoRA
llamafactory-cli train ../qwen25omni_asr_qlora.yaml

# Or for background training
nohup llamafactory-cli train ../qwen25omni_asr_qlora.yaml > ../training.log 2>&1 &
```

### 4. Monitor Training

```bash
# Watch logs
tail -f training.log

# Or use TensorBoard
tensorboard --logdir outputs/qwen25omni-malaysian-asr-qlora/logs --port 6006
```

## Configuration

Key settings in `qwen25omni_asr_qlora.yaml`:

```yaml
### Model
model_name_or_path: Qwen/Qwen2.5-Omni-7B

### Training
per_device_train_batch_size: 2         # Adjust for your GPU
gradient_accumulation_steps: 64        # Effective batch size = 128
learning_rate: 2.0e-4
num_train_epochs: 3.0

### LoRA
lora_rank: 16
lora_alpha: 32

### Quantization (QLoRA)
quantization_bit: 4                    # 4-bit for ~11GB memory
```

## GPU Memory Requirements

| Config | Batch Size | Gradient Accum | GPU Memory |
|--------|------------|----------------|------------|
| QLoRA (4-bit) | 2 | 64 | ~11GB |
| QLoRA (4-bit) | 4 | 32 | ~15GB |
| QLoRA (4-bit) | 8 | 16 | ~22GB |
| LoRA (16-bit) | 2 | 64 | ~20GB |

**Your A100-80GB can easily handle larger batches for faster training!**

Recommended for A100-80GB:
```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 16
# Effective batch size = 128 (same total)
```

## Training Time Estimate

With 5.2M training samples and A100-80GB:
- **QLoRA (batch_size=2)**: ~4-5 days
- **QLoRA (batch_size=8)**: ~2-3 days (recommended)
- **Full LoRA**: ~1-2 days (needs more memory)

## Troubleshooting

### OOM (Out of Memory)?
```yaml
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 128
```

### Training too slow?
```yaml
# Increase batch size (if GPU memory allows)
per_device_train_batch_size: 8
gradient_accumulation_steps: 16

# Enable Flash-Attention
flash_attn: fa2  # Requires installation
```

### Dataset not found?
- Check paths in `prepare_data.py`
- Ensure audio files are accessible
- Run `prepare_data.py` again

## Evaluate Trained Model

After training completes:

```bash
cd ~/voice-ai/asr/eval

# Evaluate the LoRA adapter
python evaluate.py \
  --model ~/voice-ai/asr/train/llamafactory_qwen25omni/LLaMA-Factory/outputs/qwen25omni-malaysian-asr-qlora \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto
```

## Advanced Options

### Multi-GPU Training

LLaMA-Factory automatically uses all available GPUs:

```bash
# Will use all GPUs automatically
llamafactory-cli train ../qwen25omni_asr_qlora.yaml
```

### Resume Training

If training is interrupted:

```bash
# Add resume_from_checkpoint to config
llamafactory-cli train ../qwen25omni_asr_qlora.yaml \
  --resume_from_checkpoint outputs/qwen25omni-malaysian-asr-qlora/checkpoint-2000
```

### Export Merged Model

After training, merge LoRA weights with base model:

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --adapter_name_or_path outputs/qwen25omni-malaysian-asr-qlora \
  --template qwen2_5_omni \
  --finetuning_type lora \
  --export_dir outputs/qwen25omni-malaysian-asr-merged \
  --export_size 4 \
  --export_device cpu
```

## Resources

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory/wiki)
- [Qwen2.5-Omni Model Card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [Qwen Documentation](https://qwen.readthedocs.io/en/v3.0/training/llama_factory.html)

## Comparison with Other Approaches

| Method | Status | GPU Memory | Training Speed | Complexity |
|--------|--------|------------|----------------|------------|
| **LLaMA-Factory** | ✅ Works | ~11GB (QLoRA) | Normal | Low |
| Unsloth | ❌ No support | N/A | 2x faster | Low |
| Standard PEFT | ❌ Incompatible | N/A | Normal | Medium |
| Align-Anything | ⚠️ Untested | Unknown | Unknown | High |
| Qwen2-Audio | ✅ Works | ~12GB | Normal | Low |

**Verdict**: LLaMA-Factory is your best bet for Qwen2.5-Omni right now! 🎉

