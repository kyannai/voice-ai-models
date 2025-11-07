# Training Strategy & Guide
# Malaysian Multilingual ASR System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** ML Engineering Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Unsloth Setup & Configuration](#3-unsloth-setup--configuration)
4. [Training Pipeline](#4-training-pipeline)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Multi-Stage Training](#6-multi-stage-training)
7. [Monitoring & Debugging](#7-monitoring--debugging)
8. [Training Timeline](#8-training-timeline)
9. [Common Issues & Solutions](#9-common-issues--solutions)
10. [Optimization Techniques](#10-optimization-techniques)

---

## 1. Overview

### 1.1 Training Objectives

**Primary Goal:** Fine-tune Whisper-large v3 to achieve <15% WER on Malaysian multilingual speech

**Specific Objectives:**
1. **Code-Switching Accuracy**: >85% correct language identification per word
2. **Particle Recognition**: >80% recall on Malaysian discourse particles
3. **Robustness**: Handle noisy audio (SNR 10-15dB) with <20% WER
4. **Efficiency**: Train on 50 hours of data in <12 hours on A100 GPU (with Unsloth)

### 1.2 Why Unsloth?

**Unsloth Benefits:**
- ⚡ **4-5x faster training** than standard HuggingFace Trainer
- 💾 **80% less VRAM** usage with QLoRA (train on 24GB GPU instead of 80GB)
- 🎯 **Native Whisper support** (optimized for Whisper architecture)
- 🔧 **Easy integration** with HuggingFace ecosystem
- 💰 **Cost savings**: Use cheaper GPUs (RTX 4090 instead of A100)

**Performance Comparison:**

| Method | GPU | VRAM Usage | Training Time (50hrs data) | Cost |
|--------|-----|------------|----------------------------|------|
| **Full Fine-Tuning** | A100 80GB | 75GB | 48 hours | $1,575 |
| **LoRA (HF Trainer)** | A100 40GB | 35GB | 24 hours | $788 |
| **Unsloth + LoRA** | A100 40GB | 28GB | 12 hours | $394 |
| **Unsloth + QLoRA** | RTX 4090 24GB | 20GB | 16 hours | $160 |

**Recommended:** Unsloth + LoRA on A100 (best speed-cost balance)

### 1.3 Training Strategy Overview

```
Phase 1: Environment Setup (Day 1)
├── Install Unsloth, PyTorch, dependencies
├── Prepare dataset (HuggingFace format)
└── Verify GPU setup

Phase 2: Initial Training (Days 2-3)
├── Load Whisper-large v3
├── Add LoRA adapters
├── Train on 10-hour subset (validation)
└── Evaluate WER (~18-22%)

Phase 3: Full Training (Days 4-6)
├── Train on full dataset (50 hours)
├── Curriculum learning (clean → noisy data)
├── Monitor validation WER
└── Save checkpoints every 500 steps

Phase 4: Fine-Tuning & Optimization (Days 7-9)
├── Learning rate tuning
├── Particle-focused fine-tuning
├── Code-switching augmentation
└── Final evaluation (target <15% WER)

Phase 5: Deployment Prep (Day 10)
├── Merge LoRA weights with base model
├── Export to ONNX/TensorRT (optional)
├── Benchmark inference speed (RTF < 0.3)
└── Package model for deployment
```

---

## 2. Environment Setup

### 2.1 Hardware Requirements

**Minimum (Training):**
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: 16 cores
- RAM: 64GB
- Storage: 500GB SSD

**Recommended (Training):**
- GPU: NVIDIA A100 (40GB VRAM) or H100
- CPU: 32 cores
- RAM: 128GB
- Storage: 1TB NVMe SSD

**Minimum (Inference):**
- GPU: NVIDIA T4 (16GB VRAM) or CPU (slower)
- CPU: 8 cores
- RAM: 16GB

### 2.2 Software Requirements

**Operating System:**
- Ubuntu 22.04 LTS (recommended)
- Windows 11 with WSL2 (acceptable)
- macOS (CPU training only, slow)

**CUDA & Drivers:**
```bash
# Check NVIDIA driver
nvidia-smi

# Required: CUDA 11.8+ or CUDA 12.1+
nvcc --version
```

### 2.3 Python Environment Setup

**Step 1: Create Conda Environment**
```bash
# Create new environment
conda create -n whisper-malaysian python=3.10 -y
conda activate whisper-malaysian

# Verify Python version
python --version  # Should be 3.10.x
```

**Step 2: Install PyTorch**
```bash
# For CUDA 11.8
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # Should print your GPU name
```

**Step 3: Install Unsloth**
```bash
# Install Unsloth with all dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Alternative: Install from PyPI (if available)
# pip install unsloth

# Verify installation
python -c "from unsloth import FastWhisperModel; print('Unsloth installed successfully!')"
```

**Step 4: Install Other Dependencies**
```bash
# HuggingFace libraries
pip install transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0

# Audio processing
pip install librosa==0.10.1 soundfile==0.12.1

# Evaluation metrics
pip install jiwer==3.0.3 evaluate==0.4.1

# Training utilities
pip install wandb==0.16.0  # For experiment tracking
pip install tensorboard  # Alternative tracking

# Data processing
pip install pandas numpy scipy

# Optional: Audio augmentation
pip install torch-audiomentations==0.11.0
```

**Step 5: Verify Installation**
```bash
# Test script
python << EOF
import torch
import torchaudio
from transformers import WhisperProcessor
from unsloth import FastWhisperModel
import librosa
import jiwer

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("✓ Transformers:", transformers.__version__)
print("✓ Librosa:", librosa.__version__)
print("✓ Unsloth: Ready")
print("\nAll dependencies installed successfully!")
EOF
```

### 2.4 Dataset Preparation

**Download and Organize Dataset:**
```bash
# Create directory structure
mkdir -p ./data/train/audio
mkdir -p ./data/validation/audio
mkdir -p ./data/test/audio

# If using HuggingFace dataset
python << EOF
from datasets import load_dataset, Audio

# Load your custom dataset
dataset = load_dataset("path/to/your/dataset")

# Or load from local files
from datasets import Dataset
import json

def load_local_dataset(transcripts_path, audio_dir):
    with open(transcripts_path) as f:
        data = json.load(f)
    
    dataset_dict = {"audio": [], "text": []}
    for item_id, item in data.items():
        dataset_dict["audio"].append(f"{audio_dir}/{item['audio_filepath']}")
        dataset_dict["text"].append(item["text"])
    
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

train_dataset = load_local_dataset("./data/train/transcripts.json", "./data/train")
print(f"Loaded {len(train_dataset)} training samples")
EOF
```

---

## 3. Unsloth Setup & Configuration

### 3.1 Loading Whisper with Unsloth

```python
from unsloth import FastWhisperModel
import torch

# Load Whisper-large v3 with Unsloth optimizations
model, tokenizer = FastWhisperModel.from_pretrained(
    model_name="openai/whisper-large-v3",
    max_seq_length=448,  # 30 seconds of audio → 448 mel-spectrogram frames
    dtype=torch.bfloat16,  # Use bfloat16 on A100; float16 on older GPUs
    load_in_4bit=False,  # Set True for QLoRA (24GB GPUs)
)

# Model info
print(f"Model: {model.config.model_type}")
print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f}GB")
```

**Output:**
```
Model: whisper
Parameters: 1.55B
Memory footprint: 6.20GB (bfloat16)
```

### 3.2 Adding LoRA Adapters

**LoRA (Low-Rank Adaptation):**
- Freezes base Whisper weights
- Adds small trainable adapter matrices
- Trains only 1-2% of total parameters

**Configuration:**
```python
from unsloth import FastWhisperModel

# Add LoRA adapters to Whisper
model = FastWhisperModel.get_peft_model(
    model,
    r=64,              # LoRA rank (higher = more capacity, slower training)
                       # Recommended: 32-64 for Whisper-large
    
    lora_alpha=128,    # LoRA scaling factor (typically 2x rank)
    
    lora_dropout=0.05, # Dropout for regularization (0.05-0.1)
    
    target_modules=[   # Which layers to apply LoRA
        "q_proj",      # Query projection in attention
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "out_proj",    # Output projection
        "fc1",         # Feed-forward layer 1
        "fc2",         # Feed-forward layer 2
    ],
    
    bias="none",       # Don't train biases (saves memory)
    
    use_gradient_checkpointing=True,  # Trade compute for memory
    
    random_state=42,   # For reproducibility
)

# Print trainable parameters
model.print_trainable_parameters()
```

**Output:**
```
trainable params: 25,165,824 || all params: 1,575,165,824 || trainable%: 1.60%
```

**LoRA Rank Selection:**
- **r=32**: Fastest, least capacity (~12M params) - good for >100 hours data
- **r=64**: Balanced (default, ~25M params) - recommended for 50-100 hours
- **r=128**: High capacity (~50M params) - for <50 hours or complex tasks

### 3.3 QLoRA (Quantized LoRA)

**For 24GB GPUs (RTX 4090, RTX 3090):**
```python
# Load with 4-bit quantization
model, tokenizer = FastWhisperModel.from_pretrained(
    "openai/whisper-large-v3",
    max_seq_length=448,
    dtype=torch.float16,
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # NormalFloat4 (best for neural networks)
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

# Add LoRA (same as before)
model = FastWhisperModel.get_peft_model(model, r=64, lora_alpha=128)

# Memory usage: ~18-20GB (vs 35GB without quantization)
```

**QLoRA Trade-offs:**
- ✅ Fits on cheaper GPUs (RTX 4090 @ $0.10/hr vs A100 @ $1.20/hr)
- ✅ ~20% slower training than LoRA (but still 3-4x faster than standard)
- ⚠️ Slightly lower quality (~0.5% WER increase, usually acceptable)

---

## 4. Training Pipeline

### 4.1 Data Preprocessing

```python
from transformers import WhisperProcessor
from datasets import Audio

# Load processor (combines feature extractor + tokenizer)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

def prepare_dataset(batch):
    """Preprocess audio and text for Whisper."""
    
    # Process audio → mel-spectrogram
    audio = batch["audio"]
    
    # Whisper expects 16kHz mono audio
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    batch["input_features"] = inputs.input_features[0]
    
    # Tokenize transcription
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

# Apply to dataset
train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names,
    num_proc=4,  # Parallel processing
)
```

### 4.2 Data Collator

```python
from transformers import DataCollatorSpeechSeq2SeqWithPadding

# Data collator handles batching and padding
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
```

### 4.3 Training Arguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    # Output
    output_dir="./whisper-malaysian-finetuned",
    
    # Training
    num_train_epochs=3,                    # 3 epochs usually sufficient
    per_device_train_batch_size=4,         # Adjust based on GPU memory
    gradient_accumulation_steps=8,         # Effective batch size = 32
    
    # Learning rate
    learning_rate=5e-5,                    # LoRA learning rate (higher than full fine-tuning)
    warmup_steps=500,                      # Linear warmup
    lr_scheduler_type="cosine",            # Cosine decay
    
    # Optimization
    optim="adamw_torch",                   # AdamW optimizer
    # optim="adamw_8bit",                  # Alternative: 8-bit AdamW (saves memory)
    weight_decay=0.01,                     # Regularization
    max_grad_norm=1.0,                     # Gradient clipping
    
    # Precision
    bf16=True,                             # Use bfloat16 on A100
    # fp16=True,                           # Use float16 on older GPUs (V100, T4)
    
    # Memory
    gradient_checkpointing=True,           # Save memory at cost of speed
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,                        # Evaluate every 500 steps
    per_device_eval_batch_size=8,
    predict_with_generate=True,            # Generate predictions for WER calculation
    generation_max_length=225,             # Max tokens to generate
    
    # Logging
    logging_steps=50,
    logging_dir="./logs",
    report_to=["tensorboard"],             # Or "wandb" for Weights & Biases
    
    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,                    # Keep only 3 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,               # Lower WER is better
    
    # Misc
    push_to_hub=False,                     # Set True to upload to HuggingFace Hub
    seed=42,
)
```

### 4.4 Evaluation Metric (WER)

```python
import evaluate
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Load WER metric
wer_metric = evaluate.load("wer")

# Text normalizer (optional, for fair comparison)
normalizer = BasicTextNormalizer()

def compute_metrics(pred):
    """Compute Word Error Rate (WER) during evaluation."""
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 (padding) with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize text (lowercase, remove punctuation)
    pred_str = [normalizer(text) for text in pred_str]
    label_str = [normalizer(text) for text in label_str]
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}
```

### 4.5 Training Script

```python
from transformers import Seq2SeqTrainer

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("./whisper-malaysian-final")
processor.save_pretrained("./whisper-malaysian-final")

print("Training complete!")
```

**Expected Output:**
```
Training Progress:
Step 100/5000 | Loss: 0.45 | WER: 25.3%
Step 500/5000 | Loss: 0.32 | WER: 18.7%
Step 1000/5000 | Loss: 0.28 | WER: 16.2%
Step 2000/5000 | Loss: 0.24 | WER: 14.8%
Step 5000/5000 | Loss: 0.22 | WER: 13.5%
Training complete! Best WER: 13.5%
```

### 4.6 Complete Training Script

```python
#!/usr/bin/env python3
"""
Complete training script for Malaysian Whisper fine-tuning with Unsloth
"""

import torch
from unsloth import FastWhisperModel
from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk, Audio
import evaluate

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "openai/whisper-large-v3",
    "dataset_path": "./data/malaysian_asr_dataset",
    "output_dir": "./whisper-malaysian-finetuned",
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "lora_r": 64,
    "lora_alpha": 128,
    "use_4bit": False,  # Set True for QLoRA
}

# ============================================================================
# 2. LOAD MODEL
# ============================================================================

print("Loading Whisper-large v3 with Unsloth...")
model, tokenizer = FastWhisperModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=448,
    dtype=torch.bfloat16,
    load_in_4bit=CONFIG["use_4bit"],
)

# Add LoRA adapters
model = FastWhisperModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

model.print_trainable_parameters()

# Load processor
processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])

# ============================================================================
# 3. LOAD & PREPROCESS DATASET
# ============================================================================

print("Loading dataset...")
dataset = load_from_disk(CONFIG["dataset_path"])
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

def prepare_dataset(batch):
    audio = batch["audio"]
    inputs = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    )
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

print("Preprocessing data...")
train_dataset = train_dataset.map(
    prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4
)
eval_dataset = eval_dataset.map(
    prepare_dataset, remove_columns=eval_dataset.column_names, num_proc=4
)

# ============================================================================
# 4. DATA COLLATOR & METRICS
# ============================================================================

from transformers import DataCollatorSpeechSeq2SeqWithPadding

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ============================================================================
# 5. TRAINING
# ============================================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=500,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to=["tensorboard"],
    seed=42,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

# ============================================================================
# 6. SAVE MODEL
# ============================================================================

print("Saving model...")
trainer.save_model(CONFIG["output_dir"])
processor.save_pretrained(CONFIG["output_dir"])

print(f"Training complete! Model saved to {CONFIG['output_dir']}")
```

**Run training:**
```bash
python train_whisper_malaysian.py
```

---

## 5. Hyperparameter Tuning

### 5.1 Key Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Learning Rate** | 5e-5 | 1e-5 to 1e-4 | Too high: unstable; too low: slow |
| **LoRA Rank (r)** | 64 | 32 to 128 | Higher: more capacity, slower |
| **Batch Size** | 4 × 8 = 32 | 16 to 64 | Larger: faster, more memory |
| **Epochs** | 3 | 2 to 5 | More: better fit, risk overfitting |
| **Warmup Steps** | 500 | 100 to 1000 | Stabilizes early training |
| **Weight Decay** | 0.01 | 0.001 to 0.1 | Regularization strength |

### 5.2 Learning Rate Finder

```python
from torch_lr_finder import LRFinder

# Initialize LR finder
lr_finder = LRFinder(model, optimizer, criterion)

# Run range test
lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=500)

# Plot results
lr_finder.plot()  # Suggests optimal LR

# Recommended: Choose LR at steepest decline
optimal_lr = 5e-5  # Example result
```

### 5.3 Grid Search (Small Scale)

```python
import itertools

# Define hyperparameter grid
param_grid = {
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "lora_r": [32, 64, 128],
    "batch_size": [16, 32],
}

best_wer = float("inf")
best_params = None

# Grid search (run on small subset of data for speed)
for lr, r, bs in itertools.product(
    param_grid["learning_rate"], param_grid["lora_r"], param_grid["batch_size"]
):
    print(f"Testing: LR={lr}, r={r}, BS={bs}")
    
    # Train model with these hyperparameters
    model, trainer = train_model(learning_rate=lr, lora_r=r, batch_size=bs)
    
    # Evaluate
    metrics = trainer.evaluate()
    wer = metrics["eval_wer"]
    
    if wer < best_wer:
        best_wer = wer
        best_params = {"lr": lr, "r": r, "bs": bs}
        print(f"New best! WER: {wer:.2f}%")

print(f"Best hyperparameters: {best_params}, WER: {best_wer:.2f}%")
```

---

## 6. Multi-Stage Training

### 6.1 Stage 1: Clean Data (Weeks 1-2)

**Objective:** Learn basic Malaysian speech patterns

```python
# Train on high-quality studio recordings (SNR > 25dB)
clean_dataset = dataset.filter(lambda x: x["quality_score"] >= 4)

trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=clean_dataset,
    eval_dataset=eval_dataset,
    # ... other args
)

trainer.train()
```

**Expected:** WER ~15-18% on clean test set

### 6.2 Stage 2: Add Code-Switching Focus (Week 3)

**Objective:** Improve code-switching accuracy

```python
# Oversample code-switched examples
codeswitched_dataset = dataset.filter(lambda x: x["language"] == "mixed")

# Combine with clean data (70% clean, 30% code-switched)
from datasets import concatenate_datasets

combined_dataset = concatenate_datasets([
    clean_dataset.shuffle().select(range(7000)),
    codeswitched_dataset.shuffle().select(range(3000)),
])

trainer.train()
```

**Expected:** Code-switching F1 improves to >85%

### 6.3 Stage 3: Noisy Data & Augmentation (Week 4)

**Objective:** Robust to real-world conditions

```python
# Add noisy data + augmented samples
full_dataset = concatenate_datasets([combined_dataset, noisy_dataset])

trainer.train()
```

**Expected:** WER on noisy test set < 20%

---

## 7. Monitoring & Debugging

### 7.1 TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# View at http://localhost:6006
```

**Key Metrics to Watch:**
- **Training Loss**: Should decrease steadily (target: <0.25)
- **Validation WER**: Should decrease (target: <15%)
- **Learning Rate**: Should decay according to schedule
- **Gradient Norm**: Should be < 10 (if higher, training unstable)

### 7.2 Weights & Biases (Alternative)

```python
import wandb

# Initialize W&B
wandb.init(project="whisper-malaysian", config=CONFIG)

# Modify training args
training_args = Seq2SeqTrainingArguments(
    # ... other args
    report_to=["wandb"],
)

# Automatic logging during training
```

### 7.3 Common Training Issues

**Issue 1: Out of Memory (OOM)**
```python
# Solutions:
# 1. Reduce batch size
per_device_train_batch_size=2  # Instead of 4

# 2. Increase gradient accumulation
gradient_accumulation_steps=16  # Instead of 8

# 3. Enable gradient checkpointing (already on)

# 4. Use QLoRA instead of LoRA
load_in_4bit=True

# 5. Reduce LoRA rank
r=32  # Instead of 64
```

**Issue 2: NaN Loss**
```python
# Causes: Learning rate too high, fp16 overflow

# Solutions:
# 1. Lower learning rate
learning_rate=1e-5  # Instead of 5e-5

# 2. Use bfloat16 instead of fp16
bf16=True
fp16=False

# 3. Clip gradients (already on)
max_grad_norm=1.0
```

**Issue 3: Overfitting (Train WER << Val WER)**
```python
# Solutions:
# 1. Increase dropout
lora_dropout=0.1  # Instead of 0.05

# 2. Add weight decay
weight_decay=0.05  # Instead of 0.01

# 3. Reduce training epochs
num_train_epochs=2  # Instead of 3

# 4. Add more training data
```

---

## 8. Training Timeline

### 8.1 Expected Training Time

**Dataset: 50 hours of Malaysian speech**

| GPU | Configuration | Time | Cost (AWS) |
|-----|---------------|------|------------|
| **A100 80GB** | LoRA (r=64, bf16) | 8-10 hours | $262-328 |
| **A100 40GB** | LoRA (r=64, bf16) | 10-12 hours | $328-394 |
| **RTX 4090** | QLoRA (r=64, fp16) | 14-16 hours | $140-160 |
| **V100 32GB** | LoRA (r=32, fp16) | 18-24 hours | $468-624 |

**Breakdown (A100 40GB, 12 hours):**
- Environment setup: 30 min
- Data preprocessing: 1 hour
- Training (3 epochs): 9 hours
- Evaluation: 30 min
- Model export: 30 min

### 8.2 Checkpointing Strategy

```python
# Save checkpoints every 500 steps
save_steps=500

# Keep only 3 best checkpoints (by WER)
save_total_limit=3
load_best_model_at_end=True
metric_for_best_model="wer"
```

**Resume from checkpoint:**
```python
# If training interrupted
trainer.train(resume_from_checkpoint="./whisper-malaysian-finetuned/checkpoint-2500")
```

---

## 9. Common Issues & Solutions

### 9.1 Issue: Particles Not Recognized

**Symptom:** Model omits "lah", "leh", "loh" in transcriptions

**Solutions:**
1. **Increase particle representation in training data**
   ```python
   # Oversample sentences with particles
   particle_dataset = dataset.filter(lambda x: any(p in x["text"] for p in ["lah", "leh", "loh"]))
   ```

2. **Add particle-specific tokens (advanced)**
   ```python
   new_tokens = ["<|lah|>", "<|leh|>", "<|loh|>"]
   tokenizer.add_tokens(new_tokens)
   model.resize_token_embeddings(len(tokenizer))
   ```

3. **Post-processing rule-based correction**
   ```python
   def restore_particles(text: str, audio_features) -> str:
       # Use prosody features to detect particle locations
       # Add particles back if missing
       pass
   ```

### 9.2 Issue: Poor Code-Switching Detection

**Symptom:** WER good overall, but language boundaries wrong

**Solutions:**
1. **Annotate language tags during training**
   ```python
   # Include language tokens in transcription
   # "Can <|en|> you <|en|> tolong <|ms|> check <|en|>"
   ```

2. **Train separate language ID model**
   ```python
   # Post-processing step: classify each word's language
   from transformers import pipeline
   
   lang_classifier = pipeline("text-classification", model="facebook/fasttext-language-identification")
   ```

### 9.3 Issue: Slow Inference

**Symptom:** RTF > 0.5 (too slow for production)

**Solutions:**
1. **Use smaller Whisper model**
   ```python
   # Whisper-medium (769M params) instead of large (1.5B)
   model_name = "openai/whisper-medium"
   ```

2. **Quantize to INT8**
   ```python
   from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
   
   model = ORTModelForSpeechSeq2Seq.from_pretrained(
       "whisper-malaysian-finetuned",
       provider="CUDAExecutionProvider",
   )
   ```

3. **Use TensorRT (NVIDIA GPUs)**
   ```python
   # Convert to TensorRT (5-10x speedup)
   # See Deployment Guide for details
   ```

---

## 10. Optimization Techniques

### 10.1 Mixed Precision Training

```python
# Automatic Mixed Precision (AMP)
# Already enabled in training args:
bf16=True  # On A100
# or
fp16=True  # On older GPUs
```

**Benefits:**
- 2x faster training
- 50% less memory
- Minimal accuracy loss (<0.5% WER)

### 10.2 Gradient Accumulation

```python
# Simulate larger batch size without memory increase
gradient_accumulation_steps=8

# Effective batch size = per_device_batch_size × num_gpus × accumulation_steps
# Example: 4 × 1 × 8 = 32
```

### 10.3 Multi-GPU Training

```python
# Automatic with accelerate
# Run with:
accelerate launch --multi_gpu --num_processes=4 train_whisper_malaysian.py

# Or use torchrun:
torchrun --nproc_per_node=4 train_whisper_malaysian.py
```

**Expected Speedup:**
- 2 GPUs: 1.8x faster
- 4 GPUs: 3.2x faster
- 8 GPUs: 5.5x faster

---

## 11. Appendix

### 11.1 Full Training Command

```bash
#!/bin/bash
# train.sh - Complete training pipeline

# 1. Activate environment
conda activate whisper-malaysian

# 2. Run training
python train_whisper_malaysian.py \
  --model_name openai/whisper-large-v3 \
  --dataset_path ./data/malaysian_asr_dataset \
  --output_dir ./whisper-malaysian-finetuned \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --lora_r 64 \
  --lora_alpha 128 \
  --use_4bit False

# 3. Evaluate on test set
python evaluate_model.py \
  --model_path ./whisper-malaysian-finetuned \
  --test_dataset ./data/malaysian_asr_dataset/test

echo "Training complete!"
```

### 11.2 Troubleshooting Checklist

- [ ] CUDA available: `torch.cuda.is_available()` returns `True`
- [ ] Dataset loaded: `len(train_dataset) > 0`
- [ ] Model on GPU: `model.device == 'cuda'`
- [ ] Correct dtype: `model.dtype == torch.bfloat16` or `torch.float16`
- [ ] LoRA enabled: `model.print_trainable_parameters()` shows <5% trainable
- [ ] Checkpointing works: `trainer.save_model()` runs without errors
- [ ] WER computes: `compute_metrics()` returns valid number

### 11.3 Resources

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Whisper Paper**: https://arxiv.org/abs/2212.04356
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **HuggingFace Transformers Docs**: https://huggingface.co/docs/transformers

---

**End of Training Strategy Guide**

*For evaluation procedures, see Evaluation Methodology (05).*  
*For deployment, see Deployment Guide (06).*

