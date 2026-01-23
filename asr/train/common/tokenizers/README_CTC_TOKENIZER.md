# Expanding NVIDIA Parakeet CTC Model Vocabulary with Chinese Characters

This document explains the `expand_ctc_tokenizer.py` script, covering the CTC (Connectionist Temporal Classification) model architecture, the vocabulary expansion process, and the key differences from TDT expansion.

## Table of Contents

1. [Overview](#overview)
2. [CTC Model Architecture](#ctc-model-architecture)
   - [High-Level Architecture](#high-level-architecture)
   - [Encoder (Acoustic Model)](#encoder-acoustic-model)
   - [Decoder (Output Layer)](#decoder-output-layer)
   - [Vocabulary Layout](#vocabulary-layout)
3. [CTC vs TDT: Key Differences](#ctc-vs-tdt-key-differences)
4. [Step-by-Step Expansion Process](#step-by-step-expansion-process)
   - [Step 1: Extract Chinese Characters](#step-1-extract-chinese-characters)
   - [Step 2: Load the Base Model](#step-2-load-the-base-model)
   - [Step 3: Preserve Original Weights](#step-3-preserve-original-weights)
   - [Step 4: Expand the SentencePiece Tokenizer](#step-4-expand-the-sentencepiece-tokenizer)
   - [Step 5: Apply New Tokenizer and Restore Weights](#step-5-apply-new-tokenizer-and-restore-weights)
   - [Step 6: Save the Expanded Model](#step-6-save-the-expanded-model)
5. [Usage](#usage)
6. [Verification](#verification)
7. [Fine-Tuning the Expanded Model](#fine-tuning-the-expanded-model)

---

## Overview

The `expand_ctc_tokenizer.py` script expands the vocabulary of the pre-trained `nvidia/parakeet-ctc-1.1b` English ASR model by adding Chinese characters.

**Goals**:
1. **Preserve English performance**: Identical transcriptions for English audio
2. **Add Chinese capability**: New tokens ready for fine-tuning on Chinese data

CTC models have a simpler architecture than TDT, making vocabulary expansion more straightforward.

---

## CTC Model Architecture

### High-Level Architecture

CTC (Connectionist Temporal Classification) models have a simple encoder-decoder structure:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CTC ASR Model Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────┐                                              │
│   │   Audio Input        │                                              │
│   │   (waveform)         │                                              │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────┐                                              │
│   │   PREPROCESSOR       │  Mel spectrogram extraction                  │
│   │   (AudioToMel)       │  80 mel bins, 16kHz sample rate              │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────┐                                              │
│   │   ENCODER            │  FastConformer: 24 layers                    │
│   │   (Acoustic Model)   │  1024 hidden dim, 8 attention heads          │
│   │                      │  Output: [B, T, 1024]                        │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────┐                                              │
│   │   DECODER            │  Single Conv1D layer                         │
│   │   (Output Layer)     │  1024 → vocab_size + 1                       │
│   │                      │  Output: [B, T, vocab_size + 1]              │
│   └──────────┬───────────┘  ◄── Only layer to expand                    │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────┐                                              │
│   │   CTC DECODING       │  Greedy or beam search                       │
│   │                      │  Collapse repeated tokens                    │
│   │                      │  Remove blank tokens                         │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────┐                                              │
│   │   Text Output        │                                              │
│   │   (transcription)    │                                              │
│   └──────────────────────┘                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Encoder (Acoustic Model)

**Purpose**: Converts audio features into high-level acoustic representations.

**Architecture**: FastConformer (larger than TDT version)

**Key Properties**:
- Input: Mel spectrogram features [B, T_mel, 80]
- Output: Encoded representations [B, T_enc, 1024]
- 24 Conformer blocks
- 1024 hidden dimension (larger than TDT's 640)

**Vocabulary Independence**: The encoder is completely independent of vocabulary size.

### Decoder (Output Layer)

**Purpose**: Projects encoder output to vocabulary logits.

**Architecture**: Single convolutional layer

```
┌────────────────────────────────────────────────────────────┐
│                  CTC DECODER (Output Layer)                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Input: Encoder output [B, T, 1024]                       │
│                                                            │
│   ┌──────────────────────────────────────────────────┐     │
│   │  DECODER LAYER (decoder.decoder_layers[0])       │     │
│   │                                                  │     │
│   │  Conv1D: in_channels=1024, out_channels=vocab+1  │     │
│   │                                                  │     │
│   │  For original model (vocab_size = 1024):         │     │
│   │  • Weight shape: [1025, 1024, 1]                 │     │
│   │  • Bias shape: [1025]                            │     │
│   │                                                  │     │
│   │  Output layout:                                  │     │
│   │  • [0:1023]  → BPE token logits                  │     │
│   │  • [1024]    → Blank token (⌀)                   │     │
│   │                                                  │     │
│   │  For expanded model (vocab_size = 6024):         │     │
│   │  • Weight shape: [6025, 1024, 1]                 │     │
│   │  • Bias shape: [6025]                            │     │
│   │                                                  │     │
│   │  Output layout:                                  │     │
│   │  • [0:1023]     → Original BPE tokens            │     │
│   │  • [1024:6023]  → New Chinese tokens (5000)      │     │
│   │  • [6024]       → Blank token (⌀)                │     │
│   └──────────────────────────────────────────────────┘     │
│                                                            │
│   Output: [B, T, vocab_size + 1] - Logits per frame        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Why Conv1D instead of Linear?**
- Processes each time frame independently
- More memory efficient for long sequences
- Functionally equivalent to Linear but operates on [B, C, T] format

### Vocabulary Layout

CTC has a simpler vocabulary layout than TDT (no duration tokens):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CTC DECODER OUTPUT LAYOUT                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ORIGINAL MODEL (vocab_size = 1024):                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Index    │  Content                                              │  │
│  ├───────────┼───────────────────────────────────────────────────────┤  │
│  │  0-1023   │  BPE vocabulary tokens (English)                      │  │
│  │  1024     │  Blank token (⌀) - CTC special token                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Total output size: 1025                                                │
│                                                                         │
│  EXPANDED MODEL (vocab_size = 6024):                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Index       │  Content                                           │  │
│  ├─────────────┼────────────────────────────────────────────────────┤  │
│  │  0-1023      │  Original BPE tokens (English)                     │  │
│  │  1024-6023   │  NEW Chinese character tokens (5000)               │  │
│  │  6024        │  Blank token (⌀)                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Total output size: 6025                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Blank Token**:
- Used by CTC algorithm to represent "no output at this time step"
- Always positioned at the end of the vocabulary
- Must be moved when vocabulary is expanded

---

## CTC vs TDT: Key Differences

Understanding why CTC expansion is simpler than TDT:

| Aspect | CTC | TDT |
|--------|-----|-----|
| **Decoder** | Single Conv1D layer | Embedding + LSTM + complex structure |
| **Joint Network** | None | Encoder proj + Pred proj + Output layer |
| **Special Tokens** | 1 (blank only) | 6 (5 durations + blank) |
| **Weights to Preserve** | 1 layer (decoder) | ~15 layers (decoder + joint) |
| **Complexity** | Low | High |

### Why CTC is Simpler

```
CTC Model:
┌──────────────────────────────────────────────────────────┐
│  Encoder → Decoder (1 layer) → CTC Loss                  │
└──────────────────────────────────────────────────────────┘
            ↑
            Only this layer needs expansion

TDT Model:
┌──────────────────────────────────────────────────────────┐
│  Encoder ─┬─────────────────────────────────────┐        │
│           │                                     ▼        │
│           │                              ┌─────────────┐ │
│           │                              │ Joint Net   │ │
│           │                              │ (3 layers)  │ │
│           │                              └──────┬──────┘ │
│           │                                     │        │
│           ▼                                     │        │
│  ┌────────────────┐                             │        │
│  │ Decoder        │◄────────────────────────────┘        │
│  │ (Embed + LSTM) │                                      │
│  └────────────────┘                                      │
└──────────────────────────────────────────────────────────┘
   ↑                    ↑
   Embedding needs      Joint output needs
   expansion            expansion
   LSTM must be         Projections must
   preserved            be preserved
```

### CTC Advantages for Expansion

1. **Single vocabulary-dependent layer**: Only `decoder.decoder_layers[0]`
2. **No autoregressive component**: No embedding or LSTM to worry about
3. **Simpler weight layout**: Just vocab tokens + blank (no durations)
4. **Less risk of breaking**: Fewer layers = fewer things to get wrong

---

## Step-by-Step Expansion Process

### Step 1: Extract Chinese Characters

Same as TDT - extract unique Chinese characters from manifest, ordered by frequency.

```python
def extract_chinese_chars_from_manifest(manifest_path: str, limit: int = None) -> List[str]:
    """Extract unique Chinese characters, ordered by frequency."""
    char_freq = Counter()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            for c in text:
                if '\u4e00' <= c <= '\u9fff':  # CJK Unified Ideographs
                    char_freq[c] += 1
    
    return [char for char, _ in char_freq.most_common()]
```

### Step 2: Load the Base Model

```python
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
```

For CTC, the model class is `EncDecCTCModelBPE`.

### Step 3: Preserve Original Weights

**Much simpler than TDT** - only one layer to save:

```python
ori_decoder_weights = asr_model.decoder.decoder_layers[0].weight.clone()
ori_decoder_bias = asr_model.decoder.decoder_layers[0].bias.clone()
# Weight shape: [1025, 1024, 1] = [vocab+1, encoder_dim, kernel_size]
# Bias shape: [1025]
```

That's it! No LSTM, no joint network, no projections.

### Step 4: Expand the SentencePiece Tokenizer

Same as TDT - use protobuf manipulation to add new characters:

```python
def expand_sentencepiece_model(original_model_path, new_chars, output_dir):
    """Add new characters as USER_DEFINED tokens with low scores."""
    
    # Load original tokenizer
    sp = spm.SentencePieceProcessor(model_file=original_model_path)
    model_proto = sp_pb2_model.ModelProto()
    model_proto.ParseFromString(sp.serialized_model_proto())
    
    # Add new characters
    for i, char in enumerate(new_chars):
        new_piece = model_proto.pieces.add()
        new_piece.piece = char
        new_piece.score = -1000.0 - i * 0.001
        new_piece.type = sp_pb2_model.ModelProto.SentencePiece.Type.USER_DEFINED
    
    # Save expanded tokenizer
    with open(output_dir / "tokenizer.model", 'wb') as f:
        f.write(model_proto.SerializeToString())
```

### Step 5: Apply New Tokenizer and Restore Weights

```python
# Apply new tokenizer (resizes decoder layer)
asr_model.change_vocabulary(new_tokenizer_dir, "bpe")

# Restore weights
with torch.no_grad():
    current_decoder = asr_model.decoder.decoder_layers[0]
    
    # Copy original vocab tokens (0 to old_vocab_size-1)
    current_decoder.weight[:old_vocab_size].copy_(ori_decoder_weights[:old_vocab_size])
    current_decoder.bias[:old_vocab_size].copy_(ori_decoder_bias[:old_vocab_size])
    
    # Move blank token to new last position
    current_decoder.weight[new_vocab_size].copy_(ori_decoder_weights[old_vocab_size])
    current_decoder.bias[new_vocab_size].copy_(ori_decoder_bias[old_vocab_size])
```

**Weight transformation**:
```
Original [1025 rows]:        Expanded [6025 rows]:
┌──────────────────┐         ┌──────────────────┐
│ Token 0          │  ────►  │ Token 0          │  (copied)
│ Token 1          │  ────►  │ Token 1          │  (copied)
│ ...              │  ────►  │ ...              │  (copied)
│ Token 1023       │  ────►  │ Token 1023       │  (copied)
│ Blank (1024)     │  ───┐   │ Chinese 0 (1024) │  (Xavier init)
└──────────────────┘     │   │ Chinese 1 (1025) │  (Xavier init)
                         │   │ ...              │  (Xavier init)
                         │   │ Chinese 4999     │  (Xavier init)
                         └──►│ Blank (6024)     │  (moved)
                             └──────────────────┘
```

### Step 6: Save the Expanded Model

```python
asr_model.save_to(str(output_path))
```

---

## Why CTC Doesn't Need Zero Initialization

**Important difference from TDT**: CTC expansion works without explicitly zeroing new token weights.

### Reason

In CTC, the decoder is a simple projection:
```
logits = Conv1D(encoder_output)
```

The encoder output is trained to produce values that, when projected through the original weights, give appropriate logits. New tokens with random Xavier-initialized weights will typically produce:
- Similar magnitude to trained weights
- No systematic bias toward being selected

In contrast, TDT's joint network combines two sources (encoder + decoder), and the interaction can amplify random weights unpredictably.

### However, for Safety

You could still add zero initialization for new tokens:
```python
# Optional: zero-init for extra safety
current_decoder.weight[old_vocab_size:new_vocab_size].zero_()
current_decoder.bias[old_vocab_size:new_vocab_size].fill_(-10.0)
```

This ensures new tokens have low probability, but empirically isn't required for CTC.

---

## Usage

### Via Makefile

```bash
make expand-ctc-chinese \
    ZH_MANIFEST=./training_data/KeSpeech/data/manifests/train_manifest.json \
    EXPANDED_CTC_OUTPUT=./models/parakeet-ctc-1.1b-zh.nemo \
    MAX_CHINESE_CHARS=5000
```

### Direct Python

```bash
python expand_ctc_tokenizer.py \
    --base-model nvidia/parakeet-ctc-1.1b \
    --zh-manifest /path/to/chinese_manifest.json \
    --output ./models/parakeet-ctc-1.1b-zh.nemo \
    --max-chinese-chars 5000 \
    --test-audio /path/to/english_test.wav
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--base-model` | No | `nvidia/parakeet-ctc-1.1b` | Base model name or .nemo path |
| `--zh-manifest` | Yes | - | Path to Chinese manifest (JSONL) |
| `--output` | Yes | - | Output path for expanded model |
| `--max-chinese-chars` | No | 5000 | Maximum Chinese characters to add |
| `--manifest-limit` | No | None | Limit manifest lines (for testing) |
| `--test-audio` | No | None | Audio file to transcribe before/after |

---

## Verification

### Test Identical English Performance

```bash
# Transcribe with original model
make run-parakeet-single \
    PARAKEET_MODEL=nvidia/parakeet-ctc-1.1b \
    AUDIO=/path/to/english_test.wav

# Transcribe with expanded model  
make run-parakeet-single \
    PARAKEET_MODEL=./models/parakeet-ctc-1.1b-zh.nemo \
    AUDIO=/path/to/english_test.wav
```

**Expected result**: Identical transcriptions.

### Check Vocabulary Size

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from("./models/parakeet-ctc-1.1b-zh.nemo")
print(f"Vocabulary size: {model.tokenizer.vocab_size}")
# Should show: 6024 (1024 original + 5000 Chinese)
```

---

## Fine-Tuning the Expanded Model

After expansion:
- ✅ English transcription works perfectly
- ❌ Chinese transcription doesn't work (untrained weights)

### Fine-Tuning Configuration

```yaml
# NeMo CTC fine-tuning config
model:
  train_ds:
    manifest_filepath: /path/to/chinese_train_manifest.json
    sample_rate: 16000
    batch_size: 32
  
  optim:
    name: adamw
    lr: 1e-4  # Lower LR to preserve English
    weight_decay: 0.001
    
  # Optional: freeze encoder initially
  encoder:
    freeze: true  # Preserve acoustic features
```

### Fine-Tuning Steps

1. **Prepare Chinese data**: Create manifest with `audio_filepath` and `text` fields
2. **Configure training**: Use low learning rate to preserve English
3. **Run fine-tuning**: NeMo's training scripts work out of the box
4. **Evaluate**: Test on both English and Chinese data

---

## Comparison with TDT Expansion

| Aspect | CTC Script | TDT Script |
|--------|------------|------------|
| Lines of code | ~350 | ~450 |
| Layers to preserve | 1 | ~15 |
| Weight restoration | Simple copy | Complex multi-layer |
| Blank token handling | End → new end | End → new end |
| Duration tokens | None | 5 tokens to shift |
| Zero initialization | Not required | Critical for correctness |
| Debugging difficulty | Easy | Hard (see JOURNEY_TDT.md) |

---

## Summary

CTC vocabulary expansion is straightforward because:

1. **Simple architecture**: Single output layer
2. **No autoregressive decoder**: No embedding/LSTM to worry about  
3. **Minimal special tokens**: Only blank to relocate
4. **Robust to random init**: New tokens don't interfere with decoding

The script:
1. Extracts Chinese characters from training data
2. Expands SentencePiece tokenizer via protobuf
3. Saves original decoder weights
4. Applies new tokenizer (which resizes decoder)
5. Restores original weights + moves blank token
6. Saves expanded model

Result: Identical English performance, ready for Chinese fine-tuning.
