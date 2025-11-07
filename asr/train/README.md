# ASR Model Training

This directory contains training scripts and configurations for fine-tuning various ASR models on custom datasets.

## 🗂️ Directory Structure

```
asr/train/
├── requirements.txt          # Base training dependencies (common)
├── setup_env.sh              # Unified environment setup
├── README.md                 # This file
│
├── train_qwen25omni/         # ⭐ Qwen2.5-Omni fine-tuning (LLM-based ASR)
│   ├── train_qwen25omni.py  # Main training script
│   ├── config.yaml          # Training configuration
│   ├── run_training.sh      # Training launcher
│   └── README.md            # Documentation
│
├── train_parakeet_tdt/       # 🚀 NVIDIA Parakeet TDT (Fast & Lightweight)
│   ├── train_parakeet_tdt.py # Main training script
│   ├── prepare_data.py      # Data preparation (NeMo format)
│   ├── config.yaml          # Training configuration
│   ├── run_training.sh      # Training launcher
│   └── README.md            # Documentation
│
├── funasr/                   # Qwen2-Audio fine-tuning (legacy)
│   ├── train_qwen2audio.py  # Main training script
│   ├── prepare_data.py      # Data validation & preparation
│   ├── config.yaml          # Training configuration
│   ├── requirements.txt     # FunASR-specific deps (LoRA, quantization)
│   ├── run_training.sh      # Convenience script
│   └── README.md            # Detailed documentation
│
└── whisper/                  # Whisper fine-tuning (future)
    └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies

**Option A: Interactive Setup (Recommended)**
```bash
cd /path/to/train
./setup_env.sh
```

This will:
- Create virtual environment (`.venv`)
- Install all dependencies
- Optionally install Flash-Attention 2

**Option B: Manual Install**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Flash-Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

### 2. Choose Your Framework

**⭐ For Qwen2.5-Omni (LLM-based ASR - Best Accuracy):**
```bash
cd train_qwen25omni
bash run_training.sh
```

See [train_qwen25omni/README.md](train_qwen25omni/README.md) for instructions.

**🚀 For NVIDIA Parakeet TDT (Fast & Lightweight - Recommended for Production):**
```bash
cd train_parakeet_tdt
bash run_training.sh
```

See [train_parakeet_tdt/README.md](train_parakeet_tdt/README.md) for instructions.

**For Qwen2-Audio (Legacy):**
```bash
cd funasr
bash run_training.sh
```

See [funasr/README.md](funasr/README.md) for detailed instructions.

## 📦 What Gets Installed

### Base Dependencies (required for all)
- `torch`, `torchaudio` - Deep learning framework
- `transformers` - Model architectures
- `accelerate` - Training optimization
- `librosa`, `soundfile` - Audio processing
- `datasets` - Data loading
- `tensorboard` - Training visualization
- `pyyaml` - Configuration management

### Framework-Specific

**FunASR (Qwen2-Audio):**
- `peft` - LoRA for parameter-efficient fine-tuning
- `bitsandbytes` - Quantization (4-bit/8-bit)
- `deepspeed` - Distributed training (optional)

**Whisper (future):**
- TBD

## 🎯 Available Training Methods

### 1. Qwen2.5-Omni Fine-tuning ⭐ BEST ACCURACY
- **Location**: `train_qwen25omni/`
- **Model**: Qwen2.5-Omni-7B (LLM-based ASR)
- **Method**: LoRA (Low-Rank Adaptation)
- **Features**:
  - **Superior ASR**: 1.6/3.4 WER on LibriSpeech (vs 1.6/3.6 for Qwen2-Audio)
  - Parameter-efficient (only ~1% of weights trained)
  - 4-bit quantization support
  - Memory-efficient (fits in 24GB VRAM, **~2GB less than Qwen2-Audio**)
  - Talker module disabled (saves ~2GB GPU memory)
  - Flash-Attention 2 support (2-3x faster)
  - Fast training on small datasets
- **Use Case**: Fine-tune for Malay ASR with best accuracy
- **Documentation**: [train_qwen25omni/README.md](train_qwen25omni/README.md)

### 2. NVIDIA Parakeet TDT 🚀 RECOMMENDED FOR PRODUCTION
- **Location**: `train_parakeet_tdt/`
- **Model**: Parakeet TDT 0.6B v3 (Token-and-Duration Transducer)
- **Framework**: NVIDIA NeMo
- **Features**:
  - **Lightning-Fast**: 60 minutes audio in ~1 second (10-20x faster than LLM-based)
  - **Lightweight**: Only 0.6B parameters (~4-6GB VRAM, **no quantization needed**)
  - **Auto-Punctuation**: Built-in punctuation and capitalization
  - **High Accuracy**: 98% on long audio files
  - **Word Timestamps**: Precise word-level timing
  - No quantization needed (already efficient)
  - Easy to deploy (small model size)
- **Use Case**: Production ASR with speed and efficiency requirements
- **Documentation**: [train_parakeet_tdt/README.md](train_parakeet_tdt/README.md)

### 3. Qwen2-Audio Fine-tuning (Legacy)
- **Location**: `funasr/`
- **Model**: Qwen2-Audio-7B-Instruct
- **Method**: LoRA (Low-Rank Adaptation)
- **Features**:
  - Parameter-efficient (only ~1% of weights trained)
  - 4-bit quantization support
  - Memory-efficient (fits in 24GB VRAM)
  - Fast training on small datasets
- **Use Case**: Fine-tune for Malay ASR (older model)
- **Note**: Consider using Qwen2.5-Omni instead for better results
- **Documentation**: [funasr/README.md](funasr/README.md)

### 4. Whisper Fine-tuning (Coming Soon)
- **Location**: `whisper/` (future)
- **Method**: Full or LoRA fine-tuning
- **Use Case**: Fine-tune Whisper models for domain-specific ASR

## 🔧 Training Configurations

Each framework has its own `config.yaml` for customization:

```yaml
# Example: funasr/config.yaml
model:
  name: "Qwen/Qwen2-Audio-7B-Instruct"

lora:
  r: 16              # LoRA rank
  lora_alpha: 32     # LoRA alpha

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 1
```

## 📊 Model Comparison

| Feature | Qwen2.5-Omni | Parakeet TDT 0.6B | Qwen2-Audio |
|---------|--------------|-------------------|-------------|
| **Model Size** | 7B params | 0.6B params | 7B params |
| **VRAM (Training)** | ~22GB (4-bit) | ~4-6GB (no quant) | ~24GB (4-bit) |
| **Training Speed** | Slow (LLM-based) | Fast (efficient arch) | Slow (LLM-based) |
| **Inference Speed** | Slow (~1-2 RTF) | Very Fast (~0.05 RTF) | Slow (~1-2 RTF) |
| **WER (LibriSpeech)** | 1.6/3.4 | ~2.0/4.0 | 1.6/3.6 |
| **Auto-Punctuation** | ❌ | ✅ Built-in | ❌ |
| **Word Timestamps** | ❌ | ✅ Precise | ❌ |
| **Production Ready** | Moderate | ✅ Excellent | Moderate |
| **Best For** | Highest accuracy | Speed & efficiency | Legacy projects |

**Recommendation:**
- **Need best accuracy?** → Qwen2.5-Omni
- **Need speed & production deployment?** → Parakeet TDT
- **Legacy projects?** → Qwen2-Audio (consider upgrading)

---

## 📊 Hardware Requirements

### Minimum (Consumer GPUs)
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB
- **Method**: 4-bit quantization + LoRA

### Recommended (Professional GPUs)
- **GPU**: 40GB+ VRAM (A100, A6000)
- **RAM**: 64GB
- **Method**: 8-bit quantization or full precision + LoRA

### Multi-GPU
- Use DeepSpeed or FSDP for distributed training
- See framework-specific READMEs for setup

## 📚 Training Workflows

### Typical Fine-tuning Pipeline

1. **Prepare Data**
   ```bash
   python prepare_data.py --config config.yaml
   ```
   - Validates audio files
   - Checks transcriptions
   - Splits train/val sets
   - Computes statistics

2. **Configure Training**
   - Edit `config.yaml`
   - Adjust hyperparameters
   - Set data paths

3. **Start Training**
   ```bash
   python train_*.py --config config.yaml
   ```

4. **Monitor Progress**
   ```bash
   tensorboard --logdir ./outputs/*/logs
   ```

5. **Evaluate & Deploy**
   - Test on validation set
   - Calculate WER/CER
   - Deploy to production

## 🎓 Training Best Practices

### For Small Datasets (< 1000 samples)
- Use LoRA or other PEFT methods
- Increase epochs (5-10)
- Lower learning rate (1e-4 or 5e-5)
- Use data augmentation
- Monitor for overfitting

### For Medium Datasets (1k-10k samples)
- LoRA or full fine-tuning
- Standard epochs (3-5)
- Standard learning rate (2e-4)
- Use validation set for early stopping

### For Large Datasets (> 10k samples)
- Full fine-tuning possible
- Use distributed training
- Implement learning rate scheduling
- Use multiple checkpoints

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./outputs/
```

### Weights & Biases (Optional)
```yaml
# In config.yaml
wandb:
  enabled: true
  project: "my-asr-project"
```

## 🐛 Troubleshooting

### Out of Memory
1. Enable 4-bit quantization
2. Reduce batch size to 1
3. Increase gradient accumulation steps
4. Enable gradient checkpointing
5. Reduce model size or LoRA rank

### Slow Training
1. Use mixed precision (bf16/fp16)
2. Increase batch size if possible
3. Use faster data loaders
4. Disable unnecessary callbacks

### Poor Results
1. Increase training data
2. Train for more epochs
3. Adjust learning rate
4. Try different LoRA configurations
5. Check data quality

## 📖 Additional Resources

- [Qwen2-Audio Paper](https://arxiv.org/abs/2407.10759)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/training)

## 📄 License

Training scripts are provided as-is. Check individual model licenses for usage terms.

