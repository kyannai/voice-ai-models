# ASR Evaluation Framework

Comprehensive evaluation tools for ASR (Automatic Speech Recognition) models on Malaysian multilingual audio.

## 📁 Structure

```
eval/
├── evaluate.py               ← Main entry point (unified interface)
├── test_installation.py      ← Test all dependencies
├── requirements.txt          ← ALL dependencies (consolidated)
├── setup_env.sh              ← Interactive setup script
├── README.md                 ← This file (complete documentation)
├── outputs/                  ← All evaluation outputs (auto-generated)
├── calculate_metrics/        ← Shared evaluation scripts
│   ├── calculate_metrics.py
│   └── analyze_results.py
├── transcribe/               ← Framework-specific scripts (internal use)
│   ├── whisper/
│   │   └── transcribe_whisper.py
│   └── funasr/
│       ├── utils.py                     ← Shared utilities
│       ├── transcribe_qwen25omni.py     ← Qwen2.5-Omni models (⭐ RECOMMENDED!)
│       ├── transcribe_qwen3omni.py      ← Qwen3-Omni models
│       ├── transcribe_qwen2audio.py     ← Qwen2-Audio + LoRA
│       ├── transcribe_paraformer.py     ← Traditional FunASR
│       └── transcribe_funasr.py         ← [DEPRECATED] Router
└── test_data/                ← Test datasets
    ├── malaya-test/
    └── ytl-malay-test/
```

**🆕 NEW: Modular Architecture**
The FunASR framework now uses specialized transcriber modules:
- **`utils.py`**: Shared utilities (data loading, output cleaning)
- **`transcribe_qwen25omni.py`**: For Qwen2.5-Omni-7B models (⭐ **RECOMMENDED for ASR**)
- **`transcribe_qwen3omni.py`**: For Qwen3-Omni-30B models
- **`transcribe_qwen2audio.py`**: For Qwen2-Audio models & LoRA checkpoints
- **`transcribe_paraformer.py`**: For traditional FunASR models (Paraformer, etc.)
- **`transcribe_funasr.py`**: Legacy router for backwards compatibility (deprecated)

## 🚀 Quick Start

### 1. Install Dependencies

**All dependencies are consolidated in one file!**

```bash
# Option 1: Interactive setup (recommended)
cd /path/to/eval
./setup_env.sh

# Option 2: Manual install (installs all frameworks)
pip install -r requirements.txt

# Option 3: Using uv (faster)
uv pip install -r requirements.txt
```

**Verify installation:**
```bash
python test_installation.py
```

This tests all dependencies and confirms everything is ready.

**What gets installed:**
- **Core**: torch, torchaudio, librosa, numpy, pandas
- **Metrics**: jiwer, scikit-learn
- **Utilities**: tqdm
- **Whisper**: transformers, python-dotenv
- **FunASR/Qwen2-Audio**: funasr, modelscope, accelerate, onnxruntime

### 2. Run Evaluation (Unified Interface)

**NEW: Single command for evaluation + metrics!**

```bash
# Whisper evaluation
python evaluate.py \
  --framework whisper \
  --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto

# Paraformer evaluation
python evaluate.py \
  --framework funasr \
  --model paraformer-multilingual \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device cuda

# Qwen2-Audio evaluation
python evaluate.py \
  --framework qwen2audio \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device cuda \
  --hub hf \
  --asr-prompt "Transcribe this Malay audio accurately."
```

**Features:**
- ✅ Automatic transcription + metrics calculation
- ✅ Organized outputs with timestamps
- ✅ Complete logging to file
- ✅ Standardized output format
- ✅ All results in `outputs/` directory

### 3. Prepare Test Data

Create a CSV or JSON file with your test data containing:
- `audio_path`: Path to audio file
- `text`: Reference transcription (ground truth)

**CSV format:**
```csv
audio_path,text
audio/sample1.wav,Can you tolong check the system lah
audio/sample2.wav,Selamat pagi, saya nak buat appointment
```

**JSON format:**
```json
[
  {
    "audio_path": "audio/sample1.wav",
    "text": "Can you tolong check the system lah"
  },
  {
    "audio_path": "audio/sample2.wav",
    "text": "Selamat pagi, saya nak buat appointment"
  }
]
```

### 4. View Results

```bash
# All outputs are in the outputs/ directory
ls outputs/

# Example output structure:
# outputs/whisper_Malaysian-whisper-large-v3-turbo-v3_asr_ground_truths_auto_20250102_143022/
#   ├── config.json              # Run configuration
#   ├── evaluation.log           # Complete log file
#   ├── predictions.json         # Predictions with metadata
#   ├── predictions.csv          # Human-readable predictions
#   ├── evaluation_results.json  # Detailed metrics
#   └── evaluation_summary.csv   # Quick summary
```

---

## 🎯 Unified Evaluation Interface

### Main Entry Point: `evaluate.py`

The `evaluate.py` script provides a single, unified interface for all ASR frameworks.

**Key Features:**
- **Single Command**: Run transcription + metrics in one go
- **Auto-naming**: Output directories automatically named with parameters
- **Full Logging**: Complete log saved to `evaluation.log` in each run
- **Organized Outputs**: All results in `outputs/` with timestamp
- **Framework Agnostic**: Same interface for Whisper, FunASR, Qwen2-Audio

**Basic Usage:**

```bash
python evaluate.py \
  --framework <framework> \
  --model <model-id-or-path> \
  --test-data <test-data-file> \
  --device <device>
```

**Options:**

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--framework` | ✅ | ASR framework | `whisper`, `funasr`, `qwen2audio` |
| `--model` | ✅ | Model ID or path | `mesolitica/whisper-small-malaysian-v2` |
| `--test-data` | ✅ | Test data file | `test_data/ytl-malay-test/asr_ground_truths.json` |
| `--model-source` | | Model source | `hf` (default), `local` |
| `--audio-dir` | | Audio base directory | `test_data/ytl-malay-test` |
| `--device` | | Device | `auto` (default), `cuda`, `mps`, `cpu` |
| `--name` | | Custom run name | `baseline_experiment` |
| `--language` | | Language (Whisper) | `ms` (default), `en`, `auto` |
| `--hub` | | Model hub (FunASR) | `hf` (default), `ms` |
| `--asr-prompt` | | Prompt (Qwen2-Audio) | Custom transcription prompt |
| `--workers` | | Worker threads | `4` (for FunASR) |
| `--streaming` | | Enable streaming | Future feature |

**Output Directory Naming:**

```
outputs/[name_]<framework>_<model>_<dataset>_<device>_<timestamp>/
```

Examples:
- `outputs/whisper_Malaysian-whisper-large-v3-turbo-v3_asr_ground_truths_auto_20250102_143022/`
- `outputs/baseline_funasr_paraformer-multilingual_malaya-malay-test-set_cuda_20250102_150015/`

**Complete Examples:**

```bash
# 1. Whisper with custom name
python evaluate.py \
  --framework whisper \
  --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device auto \
  --language ms \
  --name production_baseline

# 2. Local model evaluation
python evaluate.py \
  --framework whisper \
  --model-source local \
  --model /path/to/finetuned/model \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --device cuda

# 3. Qwen2-Audio with custom prompt
python evaluate.py \
  --framework qwen2audio \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test/malay-test \
  --device cuda \
  --hub hf \
  --asr-prompt "Transcribe this Malay audio accurately, preserving all discourse particles." \
  --name qwen_with_particles

# 4. Qwen3-Omni (latest SOTA model)
python evaluate.py \
  --framework funasr \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --asr-prompt "Transcribe the audio into text." \
  --name qwen3-omni-baseline

# 5. Fine-tuned LoRA checkpoint evaluation
python evaluate.py \
  --framework funasr \
  --model /path/to/training/outputs/qwen2audio-malaysian-stt/checkpoint-4000 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --model-source local \
  --asr-prompt "Transcribe this Malay audio accurately, preserving all English words and discourse particles." \
  --name checkpoint-4000-eval

# 6. Batch evaluation script
for model in whisper-small whisper-base whisper-medium; do
  python evaluate.py \
    --framework whisper \
    --model mesolitica/${model}-malaysian-v2 \
    --test-data test_data/ytl-malay-test/asr_ground_truths.json \
    --audio-dir test_data/ytl-malay-test \
    --device auto \
    --name ${model}_comparison
done
```

---

## 📊 Supported Models

### 1. Whisper (OpenAI/Malaysia-AI)

**Models:**
- `mesolitica/Malaysian-whisper-large-v3-turbo-v3` (recommended)
- `mesolitica/whisper-small-malaysian-v2`
- `mesolitica/whisper-base-malaysian`
- `openai/whisper-large-v3` (baseline comparison)

**Best for:** General purpose, proven accuracy, well-tested

**Usage:**
```bash
cd transcribe/whisper

python transcribe_whisper.py \
  --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/whisper-turbo-v3 \
  --device auto \
  --language ms
```

**Options:**
- `--device`: `auto`, `cuda`, `mps` (Apple Silicon), `cpu`
- `--language`: `ms` (Malay), `en` (English), `auto` (auto-detect)
- `--hf-token`: HuggingFace token for gated models

### 2. Paraformer (FunASR)

**Models:**
- `paraformer-multilingual` (recommended for Malaysian audio)
- `paraformer-zh` (Chinese)
- `paraformer-en` (English)

**Best for:** Fast inference, production deployment, low latency

**Usage:**
```bash
cd transcribe/funasr

python transcribe_funasr.py \
  --model paraformer-multilingual \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/paraformer \
  --device auto \
  --workers 4
```

**Options:**
- `--device`: `auto`, `cuda`, `cpu`
- `--hub`: `ms` (ModelScope), `hf` (HuggingFace)
- `--workers`: Number of concurrent workers for faster processing

### 3. Qwen2-Audio (Alibaba Audio LLM)

**Models:**
- `Qwen/Qwen2-Audio-7B-Instruct` (instruction-tuned, recommended)
- `Qwen/Qwen2-Audio-7B` (base model)

**Best for:** Highest accuracy, custom prompting, complex audio understanding

**Requirements:**
- GPU with ≥14GB VRAM (FP16) recommended
- CUDA ≥11.8
- Can run on CPU but very slow

**Usage:**
```bash
cd transcribe/funasr

python transcribe_funasr.py \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/qwen2-audio \
  --device cuda \
  --hub hf \
  --asr-prompt "Transcribe this Malay audio accurately, preserving discourse particles."
```

**Prompt Examples:**
- General: `"Transcribe the audio."`
- Malay with particles: `"Transcribe this Malay audio accurately, preserving all discourse particles like lah, leh, and lor."`
- Code-switching: `"Transcribe this Malay-English conversation accurately, preserving code-switching."`
- Formal: `"Transcribe this formal Malay speech accurately with proper punctuation."`

### 4. Qwen2.5-Omni (⭐ RECOMMENDED for ASR) 🆕

**🏆 Best Choice for ASR: Smaller, Faster, Better Accuracy**

**Models:**
- `Qwen/Qwen2.5-Omni-7B` (**⭐ HIGHLY RECOMMENDED for ASR**)

**Why Qwen2.5-Omni for ASR?**

According to [HuggingFace model card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), Qwen2.5-Omni achieves:
- **Superior ASR Performance**: Outperforms Qwen2-Audio on Common Voice benchmark
- **LibriSpeech Scores**: 1.6 WER (dev-clean) / 3.4 WER (test-clean) - excellent!
- **Smaller & Faster**: 7B params vs 30B (Qwen3-Omni) = **5x faster inference**
- **Memory Efficient**: ~14GB GPU vs ~60GB (Qwen3-Omni)
- **Audio Output Disabled**: Uses `disable_talker()` to save ~2GB GPU (ASR doesn't need speech synthesis)

**Key Features:**
- **🚀 Optimized for Speed**: Uses Flash-Attention 2 by default
- **💾 Memory Efficient**: Talker module disabled (no audio output generation)
- **🎯 Better ASR**: Specifically improved for speech recognition
- **⚡ Fast Inference**: ~0.2-0.5s/sample on A100 (vs 5-15s for Qwen3-Omni-30B)
- **🌍 Multilingual**: Supports speech input in multiple languages

**Requirements:**
- GPU with ≥16GB VRAM (BF16)
- CUDA ≥11.8
- Works well on: A100-40GB, A100-80GB, RTX 4090, A6000
- **Optional but recommended**: Flash-Attention 2 for 2-3x faster inference

**Install Flash-Attention 2 (Optional but Recommended):**
```bash
pip install flash-attn --no-build-isolation
```

The script will automatically detect and use Flash-Attention 2 if available. If not installed, it will fall back to standard attention (slower but still works).

**Usage:**
```bash
# Via evaluate.py (recommended)
python evaluate.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --asr-prompt "Transcribe this Malay audio accurately, preserving all English words and discourse particles. Return only the transcribed text with no preamble or explanation." \
  --name qwen25-omni-eval

# Quick test with 10 samples
python evaluate.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --asr-prompt "Transcribe the audio into text." \
  --max-samples 10 \
  --name qwen25-omni-test

# Direct script usage
python transcribe/funasr/transcribe_qwen25omni.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --output-dir outputs/qwen25-omni \
  --device auto \
  --asr-prompt "Transcribe the audio into text."
```

**Recommended Prompt for Malay ASR:**
```
"Transcribe this Malay audio accurately, preserving all English words and discourse particles. Return only the transcribed text with no preamble or explanation."
```

**Performance Comparison:**

| Model | Size | LibriSpeech WER | GPU Memory | Speed (A100) | Best For |
|-------|------|-----------------|------------|--------------|----------|
| **Qwen2.5-Omni-7B** | 7B | **1.6/3.4** | ~14GB | ⚡⚡ 0.2-0.5s | **ASR** ⭐ |
| Qwen3-Omni-30B | 30B | N/A | ~60GB | 🐢 5-15s | Multimodal |
| Qwen2-Audio-7B | 7B | 1.6/3.6 | ~14GB | ⚡ 1-2s | ASR/General |
| Whisper Large V3 | ~1.5B | 1.8/3.6 | ~6GB | ⚡⚡⚡ 0.1-0.3s | ASR |

**Memory Optimizations:**
```python
# Our implementation includes:
model.disable_talker()  # Saves ~2GB (no audio output needed)
return_audio=False      # Skip audio generation
attn_implementation="flash_attention_2"  # Faster inference
torch_dtype=torch.bfloat16  # Optimal precision
```

**When to Use:**
- ✅ **Primary ASR task** (speech-to-text only)
- ✅ Need **best accuracy** with reasonable speed
- ✅ Have **16GB+ GPU** available
- ✅ Want **better WER than Qwen2-Audio**
- ❌ Need real-time streaming (use Whisper instead)
- ❌ Limited to <16GB GPU (use Whisper or quantized models)

### 5. Qwen3-Omni (Latest Multimodal Foundation Model) 🆕

**✨ NEW: State-of-the-art Omni-modal Model**

**Models:**
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` (instruction-tuned, **recommended for ASR**)
- `Qwen/Qwen3-Omni-30B-A3B-Thinking` (for complex reasoning tasks)
- `Qwen/Qwen3-Omni-Flash-Instruct` (faster, lighter variant)

**Key Features:**
- **🌍 Native Malay Support**: Explicitly supports Malay for speech input (19 languages total)
- **🏆 SOTA Performance**: Comparable to Gemini 2.5 Pro on ASR benchmarks
- **🎯 Better Accuracy**: 30B parameters, outperforms Qwen2-Audio-7B
- **🔄 Omni-modal**: Handles audio, video, image, and text
- **⚡ MoE Architecture**: Efficient inference despite larger size

**Requirements:**
- GPU with ≥60GB VRAM (FP16) or ≥30GB (INT4 quantization)
- CUDA ≥11.8
- A100-80GB recommended

**Usage:**
```bash
# Via evaluate.py (recommended)
python evaluate.py \
  --framework funasr \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --hub hf \
  --asr-prompt "Transcribe the audio into text." \
  --name qwen3-omni-eval

# Quick test with 10 samples
python evaluate.py \
  --framework funasr \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --asr-prompt "Transcribe the audio into text." \
  --max-samples 10 \
  --name qwen3-omni-test
```

**Recommended Prompt for Malay ASR:**
```
"Transcribe the audio into text."
```

The model automatically detects the language (Malay) and transcribes accordingly.

**Performance Notes:**
- **Accuracy**: Expected to outperform Qwen2-Audio-7B due to larger size
- **Speed**: ~5-15s/sample (slower than 7B models, but better accuracy)
- **Memory**: Uses bfloat16 by default (more stable than float16)

**Comparison:**

| Model | Params | Languages | Malay Support | Expected WER |
|-------|--------|-----------|---------------|--------------|
| Qwen2-Audio-7B | 7B | Multiple | ✅ | ~84% (baseline) |
| **Qwen3-Omni-30B** | 30B | **19 input** | ✅ **Native** | **Better** 🎯 |
| Fine-tuned Checkpoint | 7B | Custom | ✅ | Varies |

### 5. Fine-tuned Qwen2-Audio (LoRA Checkpoints)

**✨ NEW: Automatic LoRA Detection**

The evaluation script now **automatically detects and loads LoRA checkpoints**! No need for separate scripts.

**Usage (Recommended - via evaluate.py):**
```bash
# Evaluate any training checkpoint directly
python evaluate.py \
  --framework funasr \
  --model /path/to/outputs/qwen2audio-malaysian-stt/checkpoint-4000 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --device auto \
  --model-source local \
  --asr-prompt "Transcribe this Malay audio accurately." \
  --name checkpoint-4000-eval

# The script will:
# 1. Detect it's a LoRA checkpoint (adapter_config.json + adapter_model.safetensors)
# 2. Read the base model from adapter_config.json
# 3. Load base model + apply LoRA adapter
# 4. Run full evaluation with metrics
```

**LoRA Checkpoint Structure:**
A valid LoRA checkpoint must contain:
- ✅ `adapter_config.json` - LoRA configuration with base model path
- ✅ `adapter_model.safetensors` - LoRA weights
- Optional: `optimizer.pt`, `scheduler.pt`, `trainer_state.json`

**Legacy Method (Still Supported):**
```bash
cd transcribe/funasr

python transcribe_finetuned.py \
  --base-model Qwen/Qwen2-Audio-7B-Instruct \
  --adapter-path ../../train/funasr/outputs/qwen2audio-malay-asr/final_model \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/qwen2-finetuned \
  --device cuda \
  --asr-prompt "Transcribe:"
```

---

## 🏗️ Modular Architecture

**NEW in v2.0:** The FunASR framework now uses a clean, modular architecture with specialized transcriber scripts.

### Why the Split?

Previously, all FunASR models (Paraformer, Qwen2-Audio, Qwen3-Omni) were handled by a single monolithic `transcribe_funasr.py` script. This caused:
- ❌ Complex conditional logic (`if is_qwen3_omni`, `if is_qwen2_audio`)
- ❌ Hard to maintain (841 lines in one file)
- ❌ Different APIs mixed together
- ❌ Difficult to add new models

**New Architecture:**
- ✅ Clean separation by model type
- ✅ Each file ~200-300 lines (manageable)
- ✅ No complex conditionals
- ✅ Shared utilities for common tasks
- ✅ Easy to add new models

### Module Overview

```
transcribe/funasr/
├── utils.py                     # Shared utilities
├── transcribe_qwen3omni.py      # Qwen3-Omni specialization
├── transcribe_qwen2audio.py     # Qwen2-Audio + LoRA
├── transcribe_paraformer.py     # Traditional FunASR
└── transcribe_funasr.py         # [DEPRECATED] Router
```

#### **1. `utils.py` - Shared Utilities**

Contains common functionality used across all transcribers:

- **`load_test_data()`**: Load test data from JSON/CSV
- **`validate_samples()`**: Validate and filter test samples
- **`clean_qwen_output()`**: Clean Qwen model outputs (remove preambles)
- **`save_predictions()`**: Save predictions to JSON/CSV with statistics

**Example:**
```python
from utils import load_test_data, clean_qwen_output

# Load test data
test_data = load_test_data("test.json", audio_dir="audio/")

# Clean Qwen output
text = clean_qwen_output("The transcription is: 'Hello world'")
# Returns: "Hello world"
```

#### **2. `transcribe_qwen3omni.py` - Qwen3-Omni**

Specialized for Qwen3-Omni models (30B+ parameter multimodal LLMs):

**Features:**
- Uses `Qwen3OmniMoeForConditionalGeneration` (MoE architecture)
- Uses `Qwen3OmniMoeProcessor` for multimodal processing
- Handles tuple outputs `(text_ids, audio)`
- Input format: `audio=[audio_array]` (list)
- BFloat16 precision (better for large models)
- Optimized for inference (greedy decoding, no sampling)

**Usage:**
```bash
python transcribe/funasr/transcribe_qwen3omni.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --output-dir outputs/qwen3omni \
  --device auto \
  --asr-prompt "Transcribe the audio into text."
```

#### **3. `transcribe_qwen2audio.py` - Qwen2-Audio + LoRA**

Specialized for Qwen2-Audio models (7B parameter audio LLMs):

**Features:**
- Uses `Qwen2AudioForConditionalGeneration`
- Uses `AutoProcessor` for audio processing
- **LoRA Detection**: Automatically detects and loads LoRA checkpoints
- **LoRA Merging**: Merges LoRA weights for faster inference
- Input format: `audio=audio_array, sampling_rate=16000`
- Float16 precision (good balance for 7B models)
- Optimized for inference (greedy decoding, KV cache)

**Usage (Base Model):**
```bash
python transcribe/funasr/transcribe_qwen2audio.py \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --output-dir outputs/qwen2audio \
  --device cuda \
  --asr-prompt "Transcribe this Malay audio accurately."
```

**Usage (LoRA Checkpoint):**
```bash
python transcribe/funasr/transcribe_qwen2audio.py \
  --model /path/to/checkpoint-4000 \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --output-dir outputs/lora-eval \
  --device cuda \
  --asr-prompt "Transcribe:"
```

**LoRA Checkpoint Detection:**
The script automatically detects LoRA checkpoints by checking for:
1. `adapter_config.json` (contains base model path)
2. `adapter_model.safetensors` (LoRA weights)

Then it:
1. Loads the base model from the config
2. Applies the LoRA adapter
3. **Merges weights** for faster inference (removes PeftModel wrapper)
4. Runs optimized inference

#### **4. `transcribe_paraformer.py` - Traditional FunASR**

Specialized for traditional FunASR models (Paraformer, etc.):

**Features:**
- Uses FunASR `AutoModel` API
- Supports ModelScope and HuggingFace hubs
- Simple, fast inference
- Batch processing for efficiency

**Supported Models:**
- `paraformer-multilingual` (recommended)
- `paraformer-zh` (Chinese)
- `paraformer-en` (English)
- Other FunASR models from ModelScope

**Usage:**
```bash
python transcribe/funasr/transcribe_paraformer.py \
  --model paraformer-multilingual \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --output-dir outputs/paraformer \
  --device auto \
  --hub ms
```

#### **5. `transcribe_funasr.py` - [DEPRECATED]**

**⚠️ DEPRECATED:** This file is kept only for backwards compatibility.

It's now a simple router that:
1. Detects your model type
2. Forwards to the correct specialized script
3. Prints a deprecation warning

**Please use:**
- `evaluate.py` (recommended - automatically routes)
- Or the specialized scripts directly

### Automatic Routing in `evaluate.py`

The `evaluate.py` script automatically detects your model type and routes to the correct script:

```python
# Detection logic
if "qwen3-omni" in model_name:
    script = "transcribe_qwen3omni.py"
elif "qwen2-audio" in model_name:
    script = "transcribe_qwen2audio.py"
elif is_lora_checkpoint(model_path):
    script = "transcribe_qwen2audio.py"  # LoRA uses Qwen2-Audio transcriber
else:
    script = "transcribe_paraformer.py"  # Traditional FunASR
```

**You don't need to specify which script to use** - just provide the model name/path:

```bash
# Qwen3-Omni - auto-detected
python evaluate.py --framework funasr --model Qwen/Qwen3-Omni-30B-A3B-Instruct ...

# Qwen2-Audio - auto-detected
python evaluate.py --framework funasr --model Qwen/Qwen2-Audio-7B-Instruct ...

# LoRA checkpoint - auto-detected
python evaluate.py --framework funasr --model /path/to/checkpoint-4000 ...

# Paraformer - auto-detected
python evaluate.py --framework funasr --model paraformer-multilingual ...
```

### Migration Guide

**If you're using the old `transcribe_funasr.py`:**

```bash
# Old (still works but deprecated)
python transcribe/funasr/transcribe_funasr.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct ...

# New (recommended)
python evaluate.py --framework funasr --model Qwen/Qwen3-Omni-30B-A3B-Instruct ...

# Or use specialized scripts directly
python transcribe/funasr/transcribe_qwen3omni.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct ...
```

**Benefits of Migration:**
- ✅ Cleaner, more maintainable code
- ✅ Faster development (easier to add features)
- ✅ Better performance (optimized per model type)
- ✅ Clear separation of concerns
- ✅ Future-proof (easy to add new models)

---

## 📈 Model Comparison

| Model | Speed (RTF) | Accuracy | GPU Memory | Best Use Case |
|-------|-------------|----------|------------|---------------|
| **Whisper** | ~0.2-0.3 | Good | ~4-8GB | General purpose, reliable |
| **Paraformer** | ~0.1-0.2 (2-3x faster) | Good | ~2GB | Production, low latency |
| **Qwen2-Audio** | ~0.3-0.5 (slower) | Best | ~14GB | Highest accuracy, complex audio |
| **Fine-tuned** | ~0.3-0.5 | Best for domain | ~14GB | Domain-specific, custom data |

**RTF (Real-Time Factor):**
- RTF < 1.0 = Faster than real-time
- RTF = 1.0 = Real-time processing
- RTF > 1.0 = Slower than real-time

---

## 📊 Evaluation Metrics

All evaluations calculate these metrics:

### 1. WER (Word Error Rate)
```
WER = (Substitutions + Insertions + Deletions) / Total Words × 100%
```
- **Target:** < 15% (world-class: < 12%)
- Measures word-level accuracy

### 2. CER (Character Error Rate)
- **Target:** < 8% (world-class: < 5%)
- More forgiving than WER for morphological variations

### 3. RTF (Real-Time Factor)
```
RTF = Processing Time / Audio Duration
```
- **Target:** < 0.3 (process 1 minute in 18 seconds)

### 4. Malaysian Particle Recognition
Recall, precision, and F1 for: lah, leh, loh, meh, lor, wor, hor, mah
- **Target:** Recall > 80%

---

## 🔄 Complete Evaluation Workflow

### Step 1: Transcribe Audio

**Choose your framework:**

```bash
# Option A: Whisper
cd transcribe/whisper
python transcribe_whisper.py \
  --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/whisper-model \
  --device auto

# Option B: Paraformer
cd transcribe/funasr
python transcribe_funasr.py \
  --model paraformer-multilingual \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/paraformer \
  --device auto

# Option C: Qwen2-Audio
cd transcribe/funasr
python transcribe_funasr.py \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/qwen2-audio \
  --device cuda \
  --hub hf \
  --asr-prompt "Transcribe this Malay audio accurately."
```

### Step 2: Calculate Metrics

```bash
cd ../../calculate_metrics

python calculate_metrics.py \
  --predictions ../transcribe/whisper/results/whisper-model/predictions.json \
  --output-dir ../transcribe/whisper/results/whisper-model
```

**Metrics Output:**
- `evaluation_results.json`: Detailed metrics with per-sample breakdown
- `evaluation_summary.csv`: Summary table for quick review

### Step 3: Compare Models (Optional)

```bash
python analyze_results.py \
  --predictions ../transcribe/whisper/results/whisper-model/predictions.json \
  --predictions ../transcribe/funasr/results/paraformer/predictions.json \
  --output-dir ./comparison
```

---

## 📁 Output Format

All frameworks produce standardized output:

**`predictions.json`:**
```json
{
  "model": "model-name",
  "num_samples": 100,
  "timing": {
    "total_audio_duration": 320.5,
    "total_processing_time": 85.3,
    "average_rtf": 0.266
  },
  "predictions": [
    {
      "audio_path": "audio/sample1.wav",
      "reference": "ground truth text",
      "hypothesis": "predicted text",
      "audio_duration": 2.5,
      "processing_time": 0.6,
      "rtf": 0.24
    }
  ]
}
```

**`predictions.csv`:**
```csv
audio_path,reference,hypothesis,audio_duration,processing_time,rtf
audio/sample1.wav,ground truth,prediction,2.5,0.6,0.24
```

---

## 🎯 Standard Test Datasets

### Malaya-Test Dataset
- **Location:** `test_data/malaya-test/`
- **Samples:** 765 audio files
- **Format:** JSON (`malaya-malay-test-set.json`)
- **Language:** Malay
- **Source:** Malaya Speech corpus

**Usage:**
```bash
python transcribe_*.py \
  --test-data ../../test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir ../../test_data/malaya-test/malay-test \
  --output-dir ./results/malaya-test
```

### YTL-Malay-Test Dataset
- **Location:** `test_data/ytl-malay-test/`
- **Samples:** 200 audio files
- **Format:** JSON or CSV (`asr_ground_truths.json`)
- **Language:** Malaysian multilingual (Malay-English code-switching)
- **Domain:** Business/call center

**Usage:**
```bash
python transcribe_*.py \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/ytl-test
```

---

## 🎓 Advanced Usage

### Batch Processing Multiple Models

```bash
#!/bin/bash
# evaluate_all.sh

models=(
  "mesolitica/Malaysian-whisper-large-v3-turbo-v3"
  "mesolitica/whisper-small-malaysian-v2"
  "openai/whisper-large-v3"
)

for model in "${models[@]}"; do
  name=$(basename $model)
  python transcribe/whisper/transcribe_whisper.py \
    --model $model \
    --test-data test_data/ytl-malay-test/asr_ground_truths.json \
    --audio-dir test_data/ytl-malay-test \
    --output-dir results/$name
done
```

### Custom Analysis

```python
import json
import pandas as pd

# Load results
with open("results/predictions.json") as f:
    results = json.load(f)

df = pd.DataFrame(results["predictions"])

# Find worst predictions
from jiwer import wer
df["wer"] = df.apply(lambda x: wer(x["reference"], x["hypothesis"]), axis=1)
worst_10 = df.nlargest(10, "wer")
print(worst_10[["reference", "hypothesis", "wer"]])

# Analyze by duration
import matplotlib.pyplot as plt
plt.scatter(df["audio_duration"], df["wer"])
plt.xlabel("Audio Duration (s)")
plt.ylabel("WER")
plt.savefig("wer_vs_duration.png")
```

### Comparing Fine-tuned vs Baseline

```bash
# 1. Evaluate baseline
cd transcribe/funasr
python transcribe_funasr.py \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/baseline \
  --device cuda \
  --hub hf

# 2. Evaluate fine-tuned
python transcribe_finetuned.py \
  --base-model Qwen/Qwen2-Audio-7B-Instruct \
  --adapter-path ../../train/funasr/outputs/qwen2audio-malay-asr/final_model \
  --test-data ../../test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir ../../test_data/ytl-malay-test \
  --output-dir ./results/finetuned \
  --device cuda

# 3. Compare
cd ../../calculate_metrics
python analyze_results.py \
  --predictions ../transcribe/funasr/results/baseline/predictions.json \
  --predictions ../transcribe/funasr/results/finetuned/predictions.json \
  --output-dir ./comparison
```

---

## 💡 Tips for Best Results

### Audio Quality
- Use 16kHz WAV files for best compatibility
- Clean audio (SNR > 20dB) gives better results
- Avoid heavy background noise or music

### Test Set Design
- Include diverse speakers (gender, age, accent)
- Balance language mix (English, Malay, code-switching)
- Include various domains (casual, business, call center)
- Minimum 5 hours (500-1000 samples) for reliable metrics

### Ground Truth Transcriptions
- Use professional transcribers
- Include all discourse particles (lah, leh, etc.)
- Maintain consistent spelling
- Mark code-switching boundaries if needed

### Model Selection
- **Whisper Small**: Fastest, good for real-time applications
- **Whisper Base**: Balance of speed and accuracy
- **Whisper Large**: Best accuracy, slower processing
- **Paraformer**: Fastest inference, production deployment
- **Qwen2-Audio**: Highest accuracy, complex audio, requires GPU

---

## 🐛 Troubleshooting

### Installation Issues

```bash
# Test your installation first
python test_installation.py

# If installation fails, try upgrading pip
pip install --upgrade pip

# If torch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If funasr fails
pip install funasr modelscope -U

# If transformers fails
pip install transformers accelerate -U

# Reinstall everything
pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Use CPU if CUDA unavailable
python transcribe_*.py --device cpu ...

# For Apple Silicon (M1/M2/M3)
python transcribe_whisper.py --device mps ...
```

### Memory Issues

```bash
# Reduce memory usage - use CPU
python transcribe_*.py --device cpu ...

# Or use smaller model
python transcribe_whisper.py --model mesolitica/whisper-small-malaysian-v2 ...

# For Qwen2-Audio, close other GPU applications
```

### Audio File Not Found

```bash
# Check audio path in test data
cat test_data/ytl-malay-test/asr_ground_truths.json | head

# Verify audio files exist
ls test_data/ytl-malay-test/audio/ | head

# Use --audio-dir to specify base directory
python transcribe_*.py --audio-dir /absolute/path/to/audio ...
```

### Poor WER Results

- Verify ground truth transcriptions are correct
- Check audio quality (sample rate, noise level)
- Try different model sizes or frameworks
- Ensure language setting matches your audio
- Check for consistent spelling in ground truth

### Model Download Issues

```bash
# FunASR models download from ModelScope
# If issues, try HuggingFace hub
python transcribe_funasr.py --hub hf ...

# For gated HuggingFace models, provide token
python transcribe_whisper.py --hf-token hf_your_token ...

# Manually download model
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
```

---

## 📚 Documentation & Resources

### Related Documentation
- [PRD Document](../docs/01_PRD_Product_Requirements.md) - Malaysian-specific requirements
- [Evaluation Methodology](../docs/05_Evaluation_Methodology.md) - Detailed methodology
- [Training Guide](../train/funasr/README.md) - How to fine-tune models

### External Resources
- **Whisper**: https://github.com/openai/whisper
- **Malaysia-AI**: https://github.com/mesolitica
- **FunASR**: https://github.com/modelscope/FunASR
- **Qwen2-Audio**: https://github.com/QwenLM/Qwen2-Audio

---

## 🤝 Contributing

When adding new ASR frameworks:

1. Create new subfolder: `eval/transcribe/my_model/`
2. Follow standardized output format (see Output Format section)
3. Use shared metrics scripts from `calculate_metrics/`
4. Add dependencies to root `requirements.txt`
5. Update this README with usage examples

### Output Format Requirements

Your transcription script must produce:
- `predictions.json` with required fields
- `predictions.csv` for human readability
- Compatible with `calculate_metrics.py`

---

## 📞 Support

For issues:
- Check troubleshooting section above
- Verify installation: `pip list | grep -E "torch|transformers|funasr"`
- Test with minimal example first
- Check GPU availability if using CUDA

For Malaysian-specific considerations, see project documentation in `../docs/`.

---

## 📝 License

This evaluation framework is part of the YTL Voice AI project.
