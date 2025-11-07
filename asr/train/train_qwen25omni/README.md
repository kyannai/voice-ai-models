# Qwen2.5-Omni-7B Fine-tuning for Malay ASR

Fine-tune Qwen2.5-Omni-7B with LoRA for improved Malay ASR. Better performance than Qwen2-Audio (1.6/3.4 vs 1.6/3.6 WER on LibriSpeech).

## Quick Start

### 1. Install Dependencies

```bash
# From train/ root directory
cd /path/to/train

# Option 1: Use setup script (recommended)
./setup_env.sh

# Option 2: Manual install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ⭐ REQUIRED: Install Unsloth for Qwen2.5-Omni training
pip install "unsloth[cu121_ampere_torch220] @ git+https://github.com/unslothai/unsloth.git"

# Optional: Flash-Attention 2 for 2-3x speedup
pip install flash-attn --no-build-isolation
```

**Why Unsloth?**
- ✅ Required for Qwen2.5-Omni (PEFT compatibility issues)
- ✅ 2x faster training than standard PEFT
- ✅ 30% less memory usage
- ✅ Optimized gradient checkpointing

### 2. Prepare Data

JSON format with `audio_path` and `text` fields:
```json
[
  {"audio_path": "audio/file1.wav", "text": "transcription 1"},
  {"audio_path": "audio/file2.wav", "text": "transcription 2"}
]
```

### 3. Configure

Edit `config.yaml` with your data paths:
```yaml
data:
  train_json: "/path/to/train.json"
  val_json: "/path/to/val.json"
  audio_base_dir: "/path/to/audio/"
```

### 4. Train

```bash
cd train_qwen25omni

# Interactive
bash run_training.sh

# Or direct
python train_qwen25omni.py

# Background
nohup python train_qwen25omni.py > training.log 2>&1 &
```

### 5. Monitor

```bash
tensorboard --logdir outputs/qwen25omni-malaysian-stt/logs --port 6006
```

## Configuration

Key settings in `config.yaml`:

```yaml
training:
  per_device_train_batch_size: 128  # Optimized for A100-80GB
  num_train_epochs: 3
  learning_rate: 2.0e-4

lora:
  r: 16                # LoRA rank
  lora_alpha: 32

quantization:
  load_in_4bit: true   # Fits in 24GB VRAM
```

## GPU Memory Adjustments

| GPU VRAM | Batch Size | Gradient Accum | GPU Usage |
|----------|------------|----------------|-----------|
| 24GB | 16 | 4 | ~20GB |
| 40GB | 32 | 2 | ~35GB |
| 80GB | 128 | 1 | ~50GB |

## Key Features

- **Unsloth Acceleration**: 2x faster, 30% less memory vs standard PEFT
- **Better ASR**: 1.6/3.4 WER vs 1.6/3.6 (Qwen2-Audio)
- **Memory Efficient**: ~12GB GPU (talker disabled saves ~2GB)
- **Fast Training**: Flash-Attention 2 support (optional)
- **LoRA**: Parameter-efficient fine-tuning
- **4-bit Quantization**: Fits in 24GB VRAM

## Troubleshooting

**Out of Memory?**
- Reduce `per_device_train_batch_size` to 16
- Increase `gradient_accumulation_steps` to 4

**Slow Training?**
- Install Flash-Attention 2: `pip install flash-attn --no-build-isolation`
- Increase `dataloader_num_workers` in config

**Loss Not Decreasing?**
- Reduce `learning_rate` to 1.0e-4
- Check data quality and paths

## Evaluate Model

```bash
cd ../../eval
python evaluate.py \
  --model ../train/train_qwen25omni/outputs/qwen25omni-malaysian-stt/checkpoint-6000 \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto
```

## Resources

- [Qwen2.5-Omni Model Card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Docs](https://huggingface.co/docs/peft)
