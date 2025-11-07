# 🚀 Quick Start - Qwen3-Omni-30B-Instruct Training

## Step 1: Upload to Server

```bash
# From your local machine
cd ~/data/swprojects/ytl/voice-ai/asr/train
scp -r llamafactory_qwen3omni kyan@node1:~/voice-ai/asr/train/
```

## Step 2: Setup (if not already done)

```bash
cd ~/voice-ai/asr/train/llamafactory_qwen3omni
source ../.venv/bin/activate

# If LLaMA-Factory not installed yet
bash setup_llamafactory.sh
```

## Step 3: Prepare Data & Train

```bash
# Convert data (shares dataset with qwen25omni)
python prepare_data.py

# Start training
cd LLaMA-Factory
nohup llamafactory-cli train ../qwen3omni_asr_qlora.yaml > ../training.log 2>&1 &

# Monitor
tail -f ../training.log
```

## Key Differences from Qwen2.5-Omni

| Aspect | Qwen2.5-Omni-7B | Qwen3-Omni-30B |
|--------|----------------|----------------|
| **Model Size** | 7B | 30B |
| **Template** | `qwen2_omni` | `qwen3_omni` |
| **Batch Size** | 2 | 1 |
| **Grad Accum** | 64 | 128 |
| **GPU Memory** | ~15GB | ~50GB |
| **Training Time** | 2-3 days | 5-7 days |

## Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

Expected GPU usage:
- **Loading**: ~20GB (model download)
- **Training**: ~45-50GB (steady state)

## Quick Test (Optional)

Before full training, test with small dataset:

```yaml
# Edit qwen3omni_asr_qlora.yaml
max_samples: 1000  # Test with 1K samples
```

## Troubleshooting

**OOM Error?**
- Batch size already at minimum (1)
- Kill other GPU processes (VLLM)
- Check: `nvidia-smi`

**Slow preprocessing?**
- Normal for 100K samples with 1 worker
- Wait ~20-30 minutes for tokenization
- Or reduce `max_samples` for testing

**Template error?**
- Make sure using `qwen3_omni` template
- Latest LLaMA-Factory should support it

## Ready to Train! 🎉

Your training should start automatically once preprocessing completes.

Expected output:
```
Running tokenizer on dataset: 100%
Loading checkpoint shards: 100%
trainable params: 14,024,704 || all params: 30,XXX,XXX,XXX
Starting training...
```

