# 🚀 Quick Start Guide - LLaMA-Factory for Qwen2.5-Omni

Complete setup in 3 steps!

## Step 1: Upload Files to Server

From your local machine:

```bash
# Upload the entire llamafactory_qwen25omni folder
scp -r llamafactory_qwen25omni kyan@node1:~/voice-ai/asr/train/
```

## Step 2: Setup Environment

On your server:

```bash
cd ~/voice-ai/asr/train/llamafactory_qwen25omni

# Activate training environment
source ../.venv/bin/activate

# Run setup (installs LLaMA-Factory)
bash setup_llamafactory.sh
```

**Setup time**: ~5-10 minutes

## Step 3: Prepare Data & Train

```bash
# Convert ASR data to LLaMA-Factory format
python prepare_data.py

# Start training
cd LLaMA-Factory
llamafactory-cli train ../qwen25omni_asr_qlora.yaml
```

**Or for background training:**

```bash
cd LLaMA-Factory
nohup llamafactory-cli train ../qwen25omni_asr_qlora.yaml > ../training.log 2>&1 &

# Monitor
tail -f ../training.log
```

## Expected Output

```
🦥 LLaMA Factory CLI
Loading model: Qwen/Qwen2.5-Omni-7B
Using QLoRA (4-bit quantization)
LoRA rank: 16, alpha: 32
Training samples: 5,276,046
Validation samples: 586,228
Starting training...
```

## Training Time

With A100-80GB:
- **~2-3 days** for 3 epochs
- Checkpoints saved every 2000 steps
- Best model selected automatically

## Monitor Progress

### Option 1: Watch logs
```bash
tail -f training.log
```

### Option 2: TensorBoard
```bash
tensorboard --logdir outputs/qwen25omni-malaysian-asr-qlora/logs --port 6006
# Access: http://your-server-ip:6006
```

### Option 3: GPU usage
```bash
watch -n 1 nvidia-smi
```

## After Training

Evaluate your model:

```bash
cd ~/voice-ai/asr/eval
python evaluate.py \
  --model ~/voice-ai/asr/train/llamafactory_qwen25omni/LLaMA-Factory/outputs/qwen25omni-malaysian-asr-qlora \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto
```

## Troubleshooting

**"Dataset not found"**
- Check paths in `prepare_data.py` (lines 96-98)
- Make sure audio files are in correct location

**Out of Memory**
- Edit `qwen25omni_asr_qlora.yaml`
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 128

**Training too slow**
- Increase `per_device_train_batch_size` to 8 (A100-80GB can handle it)
- Reduce `gradient_accumulation_steps` to 16

## Why LLaMA-Factory?

✅ **Only framework that works** with Qwen2.5-Omni right now
✅ **Memory efficient**: QLoRA uses only ~11GB
✅ **Production ready**: Used by thousands of researchers
✅ **Easy to configure**: Simple YAML files
✅ **Active support**: Regular updates

You're all set! 🎉

