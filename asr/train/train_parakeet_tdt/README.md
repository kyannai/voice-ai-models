# 🚀 NVIDIA Parakeet TDT 0.6B v3 Training

Complete guide for fine-tuning NVIDIA Parakeet TDT (Token-and-Duration Transducer) models on custom ASR datasets using **NVIDIA NeMo** framework.

---

## 📋 Overview

**Why Parakeet TDT?**
- ⚡ **Lightning-Fast**: 60 minutes of audio in ~1 second
- 🎯 **High Accuracy**: 98% on long audio files (up to 24 minutes)
- 📝 **Auto-Punctuation**: Built-in punctuation and capitalization
- 🕐 **Word Timestamps**: Precise word-level timing
- 💾 **Lightweight**: Only 0.6B parameters (~4-6GB VRAM)
- 🔧 **Easy to Fine-tune**: No quantization needed

**Framework: NVIDIA NeMo**
- Official framework for NVIDIA speech AI models
- Native support for Parakeet TDT architecture
- Robust training pipelines with distributed training support
- Integrated experiment management and logging

---

## 🎯 Quick Start

### 1. Install Dependencies

```bash
# Install all training dependencies from main requirements file
cd ~/voice-ai/asr/train
pip install -r requirements.txt

# Or just install NeMo toolkit if you already have other deps
pip install nemo_toolkit[asr]
```

### 2. Prepare Your Data

Convert your training data to NeMo manifest format:

```bash
python prepare_data.py \
  --train-data /path/to/train.json \
  --val-data /path/to/val.json \
  --audio-base-dir /path/to/audio/files \
  --output-dir ./data
```

This will create:
- `./data/train_manifest.json` - Training manifest
- `./data/val_manifest.json` - Validation manifest

### 3. Configure Training

Edit `config.yaml` to customize training parameters:

```yaml
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"

data:
  train_manifest: "./data/train_manifest.json"
  val_manifest: "./data/val_manifest.json"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2.0e-5
```

### 4. Start Training

```bash
# Option A: Use the launcher script (recommended)
bash run_training.sh

# Option B: Run directly
python train_parakeet_tdt.py --config config.yaml
```

### 5. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir ./outputs/parakeet-tdt-malay-asr
```

---

## 📊 Data Format

### Input Data Format (Flexible)

Your input data can be in JSON or CSV format:

**JSON Format:**
```json
[
  {
    "audio_path": "audio1.wav",
    "text": "this is a test",
    "duration": 2.5
  },
  {
    "audio_path": "audio2.wav",
    "transcription": "another example",
    "duration": 3.2
  }
]
```

**CSV Format:**
```csv
audio_path,text,duration
audio1.wav,this is a test,2.5
audio2.wav,another example,3.2
```

### NeMo Manifest Format (JSONL)

The `prepare_data.py` script converts your data to NeMo's required format:

```jsonl
{"audio_filepath": "/absolute/path/to/audio1.wav", "text": "this is a test", "duration": 2.5}
{"audio_filepath": "/absolute/path/to/audio2.wav", "text": "another example", "duration": 3.2}
```

**Required Fields:**
- `audio_filepath`: Absolute path to audio file
- `text`: Transcription text (plain text, Parakeet adds punctuation automatically)
- `duration`: Audio duration in seconds

---

## ⚙️ Training Configuration

### Model Selection

```yaml
model:
  name: "nvidia/parakeet-tdt-0.6b-v3"  # Recommended
  # Alternatives:
  # - "nvidia/parakeet-tdt-1.1b" (more accurate, 2x slower)
  # - "nvidia/parakeet-rnnt-0.6b" (streaming-capable)
```

### Batch Size Guidelines

Parakeet TDT 0.6B is very memory-efficient:

| GPU VRAM | Batch Size | Gradient Accum | Effective Batch |
|----------|------------|----------------|-----------------|
| 8GB      | 4          | 4              | 16              |
| 16GB     | 8          | 2              | 16              |
| 24GB     | 16         | 1              | 16              |
| 40GB+    | 32         | 1              | 32              |

**No quantization needed!** The model is already lightweight.

### Learning Rate

```yaml
training:
  # Fine-tuning (recommended for custom data < 10k hours)
  learning_rate: 2.0e-5
  warmup_steps: 500
  
  # Extensive pre-training (for large datasets > 10k hours)
  # learning_rate: 1.0e-3
  # warmup_steps: 15000
```

### Training Stages (Following Official Parakeet Training)

**Stage 1: Pre-training (Optional, for large datasets)**
```yaml
training:
  learning_rate: 1.0e-3
  warmup_steps: 15000
  max_steps: 150000
  scheduler: "CosineAnnealing"
```

**Stage 2: Fine-tuning (Recommended for most use cases)**
```yaml
training:
  learning_rate: 2.0e-5  # or 1.0e-5
  warmup_steps: 500      # or 0
  max_steps: 5000        # or num_train_epochs: 3
  scheduler: "CosineAnnealing"
```

---

## 📦 Hardware Requirements

### Minimum (Consumer GPUs)
- **GPU**: 8GB VRAM (RTX 3060, RTX 4060)
- **RAM**: 16GB
- **Disk**: 10GB for model + data
- **Batch Size**: 4-8

### Recommended (Professional GPUs)
- **GPU**: 16GB+ VRAM (RTX 4090, A5000)
- **RAM**: 32GB
- **Disk**: 50GB+ for large datasets
- **Batch Size**: 16-32

### High-Performance (Data Center)
- **GPU**: 40GB+ VRAM (A100, H100)
- **RAM**: 64GB+
- **Multi-GPU**: Supported (distributed training)
- **Batch Size**: 32-64+

**Note**: Parakeet TDT is much more memory-efficient than LLM-based models (Qwen, etc.)

---

## 🎓 Training Best Practices

### For Small Datasets (< 100 hours)
- Start with pre-trained model (don't train from scratch)
- Use learning rate: `2e-5`
- Train for 3-5 epochs
- Monitor validation loss closely (watch for overfitting)
- Consider data augmentation

### For Medium Datasets (100-1000 hours)
- Learning rate: `2e-5` or `1e-5`
- Train for 2-3 epochs
- Use cosine annealing scheduler
- Evaluate every 1000 steps

### For Large Datasets (> 1000 hours)
- Two-stage training (optional):
  1. Pre-train: lr=`1e-3`, 150k steps
  2. Fine-tune: lr=`1e-5`, 5k steps
- Use distributed training for speed
- Monitor WER on validation set

### Data Quality Tips
- **Audio Quality**: 16kHz sample rate minimum, clean recordings
- **Transcription**: Accurate transcriptions (Parakeet adds punctuation automatically)
- **Duration**: 0.1s - 30s per sample (avoid very short/long clips)
- **Language**: Works best on English (can be fine-tuned for other languages)
- **Diversity**: Include various speakers, accents, recording conditions

---

## 🔍 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./outputs/parakeet-tdt-malay-asr
```

**Key Metrics to Watch:**
- `val_wer` - Word Error Rate (lower is better)
- `val_loss` - Validation loss (should decrease)
- `train_loss` - Training loss (should decrease smoothly)
- `learning_rate` - Should follow scheduler curve

### Checkpoints

Checkpoints are saved automatically in:
```
./outputs/parakeet-tdt-malay-asr/
├── checkpoints/
│   ├── parakeet-tdt--epoch=0-val_wer=0.1234.ckpt
│   ├── parakeet-tdt--epoch=1-val_wer=0.0987.ckpt
│   └── parakeet-tdt--epoch=2-val_wer=0.0856.ckpt
└── logs/
    └── tensorboard events
```

Best 3 checkpoints are kept (configurable via `save_total_limit`).

---

## 🧪 After Training

### 1. Export Final Model

The final model is automatically saved to:
```
./outputs/parakeet-tdt-malay-asr/final_model.nemo
```

### 2. Test the Model

Use the transcription script:

```bash
cd ../../eval/transcribe

python transcribe_parakeet.py \
  --model ../../train/train_parakeet_tdt/outputs/parakeet-tdt-malay-asr/final_model.nemo \
  --test-data /path/to/test.json \
  --output-dir ./results/parakeet-finetuned \
  --device cuda
```

### 3. Calculate Metrics

```bash
cd ../shared
python calculate_metrics.py ../transcribe/results/parakeet-finetuned/predictions.json
```

### 4. Compare with Base Model

```bash
# Transcribe with base model
python transcribe_parakeet.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --test-data /path/to/test.json \
  --output-dir ./results/parakeet-base

# Compare results
python analyze_results.py \
  --result1 ./results/parakeet-base/predictions.json \
  --result2 ./results/parakeet-finetuned/predictions.json \
  --output-dir ./comparison
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'nemo'"

**Solution:**
```bash
# Install from main requirements (recommended)
cd ~/voice-ai/asr/train
pip install -r requirements.txt

# Or just NeMo
pip install nemo_toolkit[asr]
```

### "CUDA Out of Memory"

**Solutions:**
1. Reduce batch size in `config.yaml`:
   ```yaml
   per_device_train_batch_size: 4  # or 2
   ```

2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 4
   ```

3. Ensure no other processes are using GPU:
   ```bash
   nvidia-smi
   ```

### "Manifest file not found"

**Solution:**
1. Run `prepare_data.py` first to create manifests
2. Check paths in `config.yaml` are correct
3. Ensure audio files exist at the paths specified in manifest

### "Training is too slow"

**Solutions:**
1. Increase batch size (if you have VRAM)
2. Reduce `dataloader_num_workers` if CPU is bottleneck
3. Enable `fp16` training
4. Use faster storage (SSD) for audio files
5. Consider multi-GPU training

### "Validation WER not improving"

**Checks:**
1. **Overfitting**: Reduce learning rate or add regularization
2. **Data quality**: Check transcription accuracy
3. **Learning rate**: Try 1e-5 instead of 2e-5
4. **Training duration**: May need more epochs
5. **Data size**: Small datasets may not improve much

### "Model outputs are not punctuated"

**Note:** Parakeet TDT automatically adds punctuation during inference, not during training. The model learns to predict punctuation from the training data structure.

---

## 📖 Advanced Topics

### Multi-GPU Training

Edit `config.yaml`:
```yaml
training:
  num_gpus: 2  # or 4, 8, etc.
```

NeMo automatically handles distributed training with PyTorch Lightning.

### Custom Learning Rate Schedule

```yaml
training:
  scheduler: "CosineAnnealing"  # or "WarmupAnnealing", "PolynomialDecayAnnealing"
  warmup_steps: 1000
  min_learning_rate: 1.0e-6
```

### Resume from Checkpoint

```yaml
training:
  resume_from_checkpoint: true
```

NeMo will automatically find and resume from the last checkpoint.

### Weights & Biases Integration

```yaml
wandb:
  enabled: true
  project: "parakeet-tdt-malay"
  entity: "your-username"
```

---

## 📚 Additional Resources

- **NVIDIA NeMo Docs**: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- **Parakeet Paper**: https://arxiv.org/abs/2408.xxxxx
- **Parakeet TDT 0.6B v3**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- **NeMo ASR Tutorial**: https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr
- **Training Configurations**: Check model card for official training hyperparameters

---

## 🔄 Comparison with Other Frameworks

| Framework | Best For | Pros | Cons |
|-----------|----------|------|------|
| **NeMo** ✓ | Parakeet models | Native support, robust, distributed training | Learning curve |
| LLamaFactory | LLM-based ASR | Easy to use, flexible | Not optimized for TDT |
| HuggingFace Trainer | Transformer models | Familiar API | Limited ASR-specific features |
| Custom PyTorch | Full control | Maximum flexibility | More code to write |

**For Parakeet TDT, NeMo is the clear winner** - it's the official framework and provides the best support.

---

## ❓ FAQ

**Q: Do I need quantization for Parakeet TDT 0.6B?**  
A: No! The model is already lightweight (~4-6GB VRAM). Quantization is not necessary.

**Q: Can I fine-tune on non-English languages?**  
A: Yes, but the base model is English-optimized. Fine-tuning on other languages works but may require more data.

**Q: How much data do I need?**  
A: Minimum 10 hours, recommended 100+ hours for good results on a new domain.

**Q: Should I use TDT or RNNT?**  
A: Use **TDT** (0.6b-v3) for best accuracy and automatic punctuation. Use **RNNT** only if you need streaming.

**Q: Can I train from scratch?**  
A: Possible but not recommended. Fine-tuning from pre-trained model is much faster and achieves better results.

---

## 📄 License

Training scripts are provided as-is. Check NVIDIA's model license for usage terms.

---

**Happy Training! 🚀**

