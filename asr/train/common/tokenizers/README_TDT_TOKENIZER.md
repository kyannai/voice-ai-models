# Expanding NVIDIA Parakeet TDT Model Vocabulary with Chinese Characters

This document provides an in-depth explanation of the `expand_tdt_tokenizer.py` script, covering the TDT (Transducer with Duration Tokens) model architecture, the step-by-step vocabulary expansion process, and the reasoning behind each modification.

## Table of Contents

1. [Overview](#overview)
2. [TDT Model Architecture](#tdt-model-architecture)
   - [High-Level Architecture](#high-level-architecture)
   - [Encoder (Acoustic Model)](#encoder-acoustic-model)
   - [Decoder (Prediction Network)](#decoder-prediction-network)
   - [Joint Network](#joint-network)
   - [Vocabulary Layout and Special Tokens](#vocabulary-layout-and-special-tokens)
3. [Why Vocabulary Expansion is Non-Trivial](#why-vocabulary-expansion-is-non-trivial)
4. [Step-by-Step Expansion Process](#step-by-step-expansion-process)
   - [Step 1: Extract Chinese Characters](#step-1-extract-chinese-characters)
   - [Step 2: Load the Base Model](#step-2-load-the-base-model)
   - [Step 3: Preserve Original Weights](#step-3-preserve-original-weights)
   - [Step 4: Expand the SentencePiece Tokenizer](#step-4-expand-the-sentencepiece-tokenizer)
   - [Step 5: Apply New Tokenizer and Restore Weights](#step-5-apply-new-tokenizer-and-restore-weights)
   - [Step 6: Save the Expanded Model](#step-6-save-the-expanded-model)
5. [Critical Implementation Details](#critical-implementation-details)
   - [Weight Layout Transformation](#weight-layout-transformation)
   - [New Token Initialization](#new-token-initialization)
6. [Usage](#usage)
7. [Verification](#verification)
8. [Fine-Tuning the Expanded Model](#fine-tuning-the-expanded-model)

---

## Overview

The `expand_tdt_tokenizer.py` script expands the vocabulary of the pre-trained `nvidia/parakeet-tdt-0.6b-v3` English ASR model by adding Chinese characters. The key challenge is to:

1. **Preserve English performance**: The expanded model must produce **identical** transcriptions for English audio as the original model.
2. **Add Chinese capability**: New Chinese character tokens are added to the vocabulary, ready for fine-tuning on Chinese speech data.

This is achieved by carefully manipulating the model's:
- SentencePiece tokenizer (protobuf manipulation)
- Decoder embedding layer
- Joint network output layer

---

## TDT Model Architecture

### High-Level Architecture

The TDT (Transducer with Duration Tokens) model is an RNNT (Recurrent Neural Network Transducer) variant used for end-to-end automatic speech recognition. It consists of three main components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TDT ASR Model Architecture                       │
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
│   │   ENCODER            │  FastConformer: 17 layers                    │
│   │   (Acoustic Model)   │  640 hidden dim, 8 attention heads           │
│   │                      │  Output: [B, T, 640]                         │
│   └──────────┬───────────┘                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌──────────────────────────────────────────────┐                      │
│   │               JOINT NETWORK                   │                      │
│   │  ┌────────────────┐     ┌────────────────┐   │                      │
│   │  │ Encoder Output │     │ Decoder Output │   │                      │
│   │  │   [B,T,640]    │     │   [B,U,640]    │   │                      │
│   │  └───────┬────────┘     └───────┬────────┘   │                      │
│   │          │                      │            │                      │
│   │          ▼                      ▼            │                      │
│   │  ┌────────────────┐     ┌────────────────┐   │                      │
│   │  │  enc_proj      │     │  pred_proj     │   │                      │
│   │  │  640 → 640     │     │  640 → 640     │   │                      │
│   │  └───────┬────────┘     └───────┬────────┘   │                      │
│   │          │                      │            │                      │
│   │          └──────────┬───────────┘            │                      │
│   │                     │ (addition)             │                      │
│   │                     ▼                        │                      │
│   │          ┌────────────────────┐              │                      │
│   │          │   Activation       │              │                      │
│   │          │   (ReLU/Tanh)      │              │                      │
│   │          └─────────┬──────────┘              │                      │
│   │                    │                         │                      │
│   │                    ▼                         │                      │
│   │          ┌────────────────────┐              │                      │
│   │          │   Output Layer    │              │                      │
│   │          │  640 → vocab+6    │   ◄── Expanded layer               │
│   │          └─────────┬──────────┘              │                      │
│   │                    │                         │                      │
│   └────────────────────┼─────────────────────────┘                      │
│                        ▼                                                │
│            ┌────────────────────┐                                       │
│            │  DECODING          │                                       │
│            │  (Greedy/Beam)     │                                       │
│            └─────────┬──────────┘                                       │
│                      │                                                  │
│                      ▼                                                  │
│   ┌──────────────────────────────────────────────┐                      │
│   │                                              │                      │
│   │  ┌──────────────────────┐                    │                      │
│   │  │   DECODER            │  LSTM-based        │                      │
│   │  │   (Prediction Net)   │  Predicts next     │                      │
│   │  │                      │  token             │                      │
│   │  │  Embedding ◄── Expanded layer            │                      │
│   │  │  [vocab+1, 640]      │                    │                      │
│   │  │                      │                    │                      │
│   │  │  LSTM Layers         │  Hidden: 640      │                      │
│   │  │  (2 layers)          │                    │                      │
│   │  └──────────────────────┘                    │                      │
│   │                                              │                      │
│   └──────────────────────────────────────────────┘                      │
│                                                                         │
│                        ▼                                                │
│            ┌────────────────────┐                                       │
│            │   Text Output      │                                       │
│            │   (transcription)  │                                       │
│            └────────────────────┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Encoder (Acoustic Model)

**Purpose**: Converts audio features into high-level acoustic representations.

**Architecture**: FastConformer (a variant of Conformer optimized for efficiency)

**Key Properties**:
- Input: Mel spectrogram features [B, T_mel, 80]
- Output: Encoded representations [B, T_enc, 640]
- 17 Conformer blocks with:
  - Self-attention (8 heads)
  - Convolution modules
  - Feed-forward layers
- Subsampling factor: 8x (every 8 mel frames → 1 encoder frame)

**Vocabulary Independence**: The encoder processes audio and produces acoustic embeddings. It is **completely independent of the vocabulary size** and does not need modification during vocabulary expansion.

### Decoder (Prediction Network)

**Purpose**: Predicts the next output token based on previously emitted tokens. This is the "language model" component of the transducer.

**Architecture**:

```
┌────────────────────────────────────────────────────────────┐
│                  DECODER (Prediction Network)               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Input: Previous token IDs [B, U]                         │
│                                                            │
│   ┌──────────────────────────────────────────────────┐     │
│   │  EMBEDDING LAYER (decoder.prediction.embed)      │     │
│   │  Shape: [vocab_size + 1, 640]                    │     │
│   │                                                  │     │
│   │  • Indices 0 to 8191: BPE tokens                 │     │
│   │  • Index 8192: Blank token (⌀)                   │     │
│   │                                                  │     │
│   │  For expanded model (5000 Chinese chars):        │     │
│   │  • Indices 0 to 8191: Original BPE tokens        │     │
│   │  • Indices 8192 to 13191: New Chinese tokens     │     │
│   │  • Index 13192: Blank token (⌀)                  │     │
│   └──────────────────────────────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│   ┌──────────────────────────────────────────────────┐     │
│   │  LSTM LAYERS (decoder.prediction.dec_rnn.lstm)   │     │
│   │                                                  │     │
│   │  • 2 LSTM layers                                 │     │
│   │  • Hidden size: 640                              │     │
│   │  • Bidirectional: No (causal for autoregressive) │     │
│   │                                                  │     │
│   │  Weights to preserve:                            │     │
│   │  • weight_ih_l0, weight_hh_l0, bias_ih_l0, etc. │     │
│   └──────────────────────────────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│   Output: [B, U, 640] - Prediction representations         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Vocabulary-Dependent Components**:
- **Embedding Layer**: `decoder.prediction.embed.weight` has shape `[vocab_size + 1, 640]`
  - The `+1` is for the blank token, placed at the end
  - Original: `[8193, 640]` → Expanded: `[13193, 640]`

**Vocabulary-Independent Components**:
- **LSTM Layers**: The LSTM weights don't depend on vocabulary size
  - They process 640-dimensional embeddings regardless of vocabulary
  - Must be preserved exactly to maintain English performance

### Joint Network

**Purpose**: Combines encoder output (acoustic information) and decoder output (linguistic information) to produce probabilities over the vocabulary + special tokens.

**Architecture**:

```
┌────────────────────────────────────────────────────────────┐
│                      JOINT NETWORK                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Inputs:                                                  │
│   • Encoder output: [B, T, 640]                            │
│   • Decoder output: [B, U, 640]                            │
│                                                            │
│   ┌──────────────────────────────────────────────────┐     │
│   │  ENCODER PROJECTION (joint.enc.*)                │     │
│   │  Linear: 640 → 640                               │     │
│   │  + LayerNorm                                     │     │
│   └──────────────────────────────────────────────────┘     │
│                          │                                 │
│   ┌──────────────────────────────────────────────────┐     │
│   │  PREDICTION PROJECTION (joint.pred.*)            │     │
│   │  Linear: 640 → 640                               │     │
│   │  + LayerNorm                                     │     │
│   └──────────────────────────────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│   ┌──────────────────────────────────────────────────┐     │
│   │  JOINT NET (joint.joint_net)                     │     │
│   │                                                  │     │
│   │  joint_net[0]: Linear(640, 640)                  │     │
│   │  joint_net[1]: Activation (ReLU)                 │     │
│   │  joint_net[2]: Linear(640, vocab_size + 6)       │ ◄── OUTPUT LAYER   │
│   │                                                  │     │
│   │  Output layout for original (8192 vocab):        │     │
│   │  • [0:8191]     → BPE token logits               │     │
│   │  • [8192]       → Duration 0                     │     │
│   │  • [8193]       → Duration 1                     │     │
│   │  • [8194]       → Duration 2                     │     │
│   │  • [8195]       → Duration 3                     │     │
│   │  • [8196]       → Duration 4                     │     │
│   │  • [8197]       → Blank token                    │     │
│   │                                                  │     │
│   │  Output layout for expanded (13192 vocab):       │     │
│   │  • [0:8191]     → Original BPE token logits      │     │
│   │  • [8192:13191] → New Chinese token logits       │     │
│   │  • [13192]      → Duration 0                     │     │
│   │  • [13193]      → Duration 1                     │     │
│   │  • [13194]      → Duration 2                     │     │
│   │  • [13195]      → Duration 3                     │     │
│   │  • [13196]      → Duration 4                     │     │
│   │  • [13197]      → Blank token                    │     │
│   └──────────────────────────────────────────────────┘     │
│                          │                                 │
│   Output: [B, T, U, vocab_size + 6] - Logits               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Vocabulary-Dependent Components**:
- **Output Layer** (`joint.joint_net.2`):
  - Weight: `[vocab_size + 6, 640]` (6 = 5 durations + 1 blank)
  - Bias: `[vocab_size + 6]`
  - Original: `[8198, 640]` → Expanded: `[13198, 640]`

**Vocabulary-Independent Components** (must be preserved):
- **Encoder Projection**: `joint.enc.*`
- **Prediction Projection**: `joint.pred.*`
- **First hidden layer**: `joint.joint_net.0`

### Vocabulary Layout and Special Tokens

TDT models have a specific layout for the output layer that includes both vocabulary tokens and special tokens:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JOINT OUTPUT LAYER LAYOUT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ORIGINAL MODEL (vocab_size = 8192):                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Index    │  Content                                              │  │
│  ├───────────┼───────────────────────────────────────────────────────┤  │
│  │  0-8191   │  BPE vocabulary tokens (English)                      │  │
│  │  8192     │  Duration 0 (emit token, advance 0 frames)            │  │
│  │  8193     │  Duration 1 (emit token, advance 1 frame)             │  │
│  │  8194     │  Duration 2 (emit token, advance 2 frames)            │  │
│  │  8195     │  Duration 3 (emit token, advance 3 frames)            │  │
│  │  8196     │  Duration 4 (emit token, advance 4 frames)            │  │
│  │  8197     │  Blank (⌀) - emit nothing, advance 1 frame            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Total output size: 8198                                                │
│                                                                         │
│  EXPANDED MODEL (vocab_size = 13192):                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Index       │  Content                                           │  │
│  ├─────────────┼────────────────────────────────────────────────────┤  │
│  │  0-8191      │  Original BPE tokens (English)                     │  │
│  │  8192-13191  │  NEW Chinese character tokens (5000)               │  │
│  │  13192       │  Duration 0                                        │  │
│  │  13193       │  Duration 1                                        │  │
│  │  13194       │  Duration 2                                        │  │
│  │  13195       │  Duration 3                                        │  │
│  │  13196       │  Duration 4                                        │  │
│  │  13197       │  Blank (⌀)                                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Total output size: 13198                                               │
│                                                                         │
│  DECODER EMBEDDING LAYOUT:                                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Original: [vocab (0-8191), blank (8192)]                         │  │
│  │  Expanded: [vocab (0-8191), new (8192-13191), blank (13192)]      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Note: Decoder embedding has vocab+1 (includes blank but not durations) │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Duration Tokens** are a TDT-specific feature:
- In standard RNNT, a token is emitted and advances by 1 frame
- In TDT, duration tokens allow predicting how many frames to advance (0-4)
- This enables more efficient decoding for variable-length tokens

---

## Why Vocabulary Expansion is Non-Trivial

### The `change_vocabulary()` Problem

NeMo provides a `change_vocabulary()` method that updates the tokenizer and resizes vocabulary-dependent layers. However, **it reinitializes ALL decoder and joint network weights**, not just the newly added dimensions.

```python
# What change_vocabulary() does internally:
# 1. Loads new tokenizer ✓
# 2. Updates config ✓
# 3. REINITIALIZES entire decoder ✗ (loses trained weights!)
# 4. REINITIALIZES entire joint network ✗ (loses trained weights!)
```

This means:
- LSTM weights are reinitialized (random)
- Encoder/Prediction projections are reinitialized (random)
- Hidden layers are reinitialized (random)
- Even the original vocabulary embeddings are reinitialized!

**Result**: After naive `change_vocabulary()`, the model outputs garbage even for English.

### Our Solution

We must:
1. **Save** all decoder and joint weights **before** calling `change_vocabulary()`
2. Call `change_vocabulary()` to resize the layers and load new tokenizer
3. **Restore** original weights to their correct positions
4. **Initialize** new token positions carefully (zero weights + very negative bias)

---

## Step-by-Step Expansion Process

### Step 1: Extract Chinese Characters

```python
def extract_chinese_chars_from_manifest(manifest_path: str, limit: int = None) -> List[str]:
```

**Purpose**: Extract unique Chinese characters from a training manifest file, ordered by frequency.

**Process**:
1. Read the JSONL manifest file line by line
2. For each line, parse the JSON and extract the `text` field
3. Filter for Chinese characters (CJK Unified Ideographs: `\u4e00-\u9fff`, Extension A: `\u3400-\u4dbf`)
4. Count character frequencies
5. Return characters sorted by frequency (most common first)

**Why frequency ordering?**
- More common characters get lower indices
- During fine-tuning, common characters will be learned first
- Helps with BPE score assignment (common = higher priority)

### Step 2: Load the Base Model

```python
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
```

**Purpose**: Load the pre-trained model and extract original vocabulary information.

**What we extract**:
- `old_vocab_size`: Original vocabulary size (8192)
- `vocab_list`: List of original BPE tokens
- Original model for transcription comparison (optional)

### Step 3: Preserve Original Weights

This is the **most critical step**. We save all weights that will be affected by `change_vocabulary()`.

#### Decoder Embedding

```python
ori_embed_weight = asr_model.decoder.prediction.embed.weight.clone()
# Shape: [8193, 640] = [vocab_size + 1, hidden_dim]
# Layout: [vocab tokens (0-8191), blank token (8192)]
```

**Why save it?**
- Contains learned embeddings for all English BPE tokens
- Blank token embedding at position 8192 must be moved to position 13192

#### Decoder LSTM Weights

```python
ori_decoder_state = {}
for name, param in asr_model.decoder.prediction.named_parameters():
    if 'embed' not in name:  # Skip embedding
        ori_decoder_state[name] = param.clone()
```

**Weights saved**:
- `dec_rnn.lstm.weight_ih_l0`: Input-hidden weights layer 0
- `dec_rnn.lstm.weight_hh_l0`: Hidden-hidden weights layer 0
- `dec_rnn.lstm.bias_ih_l0`, `dec_rnn.lstm.bias_hh_l0`
- `dec_rnn.lstm.weight_ih_l1`, etc. (layer 1)

**Why save them?**
- LSTM weights encode the learned language model
- They are independent of vocabulary size but are reinitialized by `change_vocabulary()`

#### Joint Output Layer

```python
ori_joint_weight = asr_model.joint.joint_net[2].weight.clone()
ori_joint_bias = asr_model.joint.joint_net[2].bias.clone()
# Weight shape: [8198, 640] = [vocab + 6, hidden_dim]
# Bias shape: [8198]
```

**Layout analysis**:
```
Positions 0-8191:     Vocabulary token weights
Positions 8192-8196:  Duration token weights (0-4)
Position 8197:        Blank token weight
```

**Why save it?**
- Contains learned output mappings for all English tokens
- Duration and blank weights must be moved to new positions

#### Other Joint Network Weights

```python
ori_joint_state = {}
for name, param in asr_model.joint.named_parameters():
    if 'joint_net.2' not in name:  # Skip output layer
        ori_joint_state[name] = param.clone()
```

**Weights saved**:
- `enc.0.weight`, `enc.0.bias`: Encoder projection
- `enc.1.weight`, `enc.1.bias`: Encoder LayerNorm
- `pred.0.weight`, `pred.0.bias`: Prediction projection
- `pred.1.weight`, `pred.1.bias`: Prediction LayerNorm
- `joint_net.0.weight`, `joint_net.0.bias`: Hidden layer

**Why save them?**
- These are vocabulary-independent but still reinitialized
- Critical for maintaining the learned acoustic-linguistic mapping

### Step 4: Expand the SentencePiece Tokenizer

```python
def expand_sentencepiece_model(original_model_path, new_chars, output_dir):
```

**Purpose**: Add new Chinese characters to the SentencePiece tokenizer using protobuf manipulation.

**Process**:
1. Load original `tokenizer.model` as protobuf
2. Parse existing pieces to avoid duplicates
3. Add new characters as `USER_DEFINED` type with low scores (-1000 - i*0.001)
4. Update `trainer_spec.vocab_size`
5. Save `tokenizer.model`, `tokenizer.vocab`, `vocab.txt`

**Why protobuf manipulation instead of retraining?**
- Preserves exact original tokenization for English text
- Ensures token IDs 0-8191 remain unchanged
- New tokens get IDs 8192+ without affecting original mapping

**Why USER_DEFINED type with low scores?**
- `USER_DEFINED` tokens are always kept during SentencePiece encoding
- Low scores (-1000) ensure they're only used when explicitly needed
- Prevents interference with existing English BPE tokenization

### Step 5: Apply New Tokenizer and Restore Weights

#### Apply New Tokenizer

```python
asr_model.change_vocabulary(
    new_tokenizer_dir=str(tmpdir),
    new_tokenizer_type="bpe"
)
```

This:
- Loads the expanded tokenizer
- Resizes decoder embedding: `[8193, 640]` → `[13193, 640]`
- Resizes joint output: `[8198, 640]` → `[13198, 640]`
- **Reinitializes all weights to random values!**

#### Restore Decoder Embedding

```python
with torch.no_grad():
    new_embed = asr_model.decoder.prediction.embed
    
    # Copy original vocab embeddings (0 to old_vocab_size-1)
    new_embed.weight[:old_vocab_size].copy_(ori_embed_weight[:old_vocab_size])
    
    # Move blank token from old position to new position
    new_embed.weight[new_vocab_size].copy_(ori_embed_weight[old_vocab_size])
```

**Weight transformation**:
```
Original [8193 rows]:        Expanded [13193 rows]:
┌──────────────────┐         ┌──────────────────┐
│ Token 0          │  ────►  │ Token 0          │  (copied)
│ Token 1          │  ────►  │ Token 1          │  (copied)
│ ...              │  ────►  │ ...              │  (copied)
│ Token 8191       │  ────►  │ Token 8191       │  (copied)
│ Blank (8192)     │  ───┐   │ Chinese 0 (8192) │  (zero-init)
└──────────────────┘     │   │ Chinese 1 (8193) │  (zero-init)
                         │   │ ...              │  (zero-init)
                         │   │ Chinese 4999     │  (zero-init)
                         └──►│ Blank (13192)    │  (moved)
                             └──────────────────┘
```

#### Restore LSTM Weights

```python
decoder_params = dict(asr_model.decoder.prediction.named_parameters())
for name, ori_param in ori_decoder_state.items():
    if name in decoder_params:
        decoder_params[name].copy_(ori_param)
```

LSTM weights are vocabulary-independent, so we copy them exactly.

#### Restore Joint Output Layer

```python
new_joint = asr_model.joint.joint_net[2]

# Copy original vocab weights (0 to old_vocab_size-1)
new_joint.weight[:old_vocab_size].copy_(ori_joint_weight[:old_vocab_size])
new_joint.bias[:old_vocab_size].copy_(ori_joint_bias[:old_vocab_size])

# Move duration + blank tokens to new positions
num_special_tokens = 6  # 5 durations + 1 blank
new_joint.weight[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
    ori_joint_weight[old_vocab_size:old_vocab_size + num_special_tokens]
)
new_joint.bias[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
    ori_joint_bias[old_vocab_size:old_vocab_size + num_special_tokens]
)
```

**Weight transformation**:
```
Original [8198 rows]:              Expanded [13198 rows]:
┌─────────────────────────┐        ┌─────────────────────────┐
│ Token 0                 │  ────► │ Token 0                 │
│ Token 1                 │  ────► │ Token 1                 │
│ ...                     │  ────► │ ...                     │
│ Token 8191              │  ────► │ Token 8191              │
│ Duration 0 (8192)       │  ───┐  │ Chinese 0 (8192)        │  (zero, -1000 bias)
│ Duration 1 (8193)       │  ──┐│  │ Chinese 1 (8193)        │  (zero, -1000 bias)
│ Duration 2 (8194)       │  ─┐││  │ ...                     │  ...
│ Duration 3 (8195)       │  ┐│││  │ Chinese 4999 (13191)    │  (zero, -1000 bias)
│ Duration 4 (8196)       │  ││││  │ Duration 0 (13192)      │  ◄─── (moved)
│ Blank (8197)            │  │││└► │ Duration 1 (13193)      │  ◄─── (moved)
└─────────────────────────┘  ││└──►│ Duration 2 (13194)      │  ◄─── (moved)
                             │└───►│ Duration 3 (13195)      │  ◄─── (moved)
                             └────►│ Duration 4 (13196)      │  ◄─── (moved)
                              └───►│ Blank (13197)           │  ◄─── (moved)
                                   └─────────────────────────┘
```

#### Initialize New Tokens (Critical!)

```python
# Get statistics from existing tokens for smart initialization
existing_weight_std = ori_joint_weight[:old_vocab_size].std()
existing_bias_mean = ori_joint_bias[:old_vocab_size].mean()
existing_embed_std = ori_embed_weight[:old_vocab_size].std()

# Initialize new joint weights with small random values
new_joint.weight[old_vocab_size:new_vocab_size].normal_(
    mean=0.0,
    std=existing_weight_std * 0.01  # Small random init
)
# Initialize bias slightly below mean (not extreme)
new_joint.bias[old_vocab_size:new_vocab_size].fill_(
    existing_bias_mean - 5.0  # Slightly negative, but trainable
)

# Initialize new embeddings with small random values
new_embed.weight[old_vocab_size:new_vocab_size].normal_(
    mean=0.0,
    std=existing_embed_std * 0.01  # Small random init
)
```

**Why this initialization is critical**:

The initialization must balance two requirements:

1. **Inference**: New tokens shouldn't interfere with English transcription
2. **Training**: Gradients must flow to enable learning of new tokens

**The Problem with Extreme Initialization**:

An earlier approach used `-1000` bias, which:
- ✅ Preserved English perfectly (new tokens never selected)
- ❌ Blocked gradients during training (new tokens couldn't learn!)

**Our Balanced Solution**:
- **Small random weights**: Non-zero but small, allows gradients to flow
- **Slightly negative bias (mean - 5)**: Low probability during inference, but gradients can still update the weights
- **Result**: English is preserved AND new tokens can be learned during fine-tuning

The key insight is that the softmax gradient is essentially zero when logits are at `-1000`, creating a "gradient desert" that prevents learning.

### Step 6: Save the Expanded Model

```python
asr_model.save_to(str(output_path))
```

NeMo's `.nemo` format packages:
- Updated `model_config.yaml` with new vocab size
- Updated tokenizer files (`tokenizer.model`, `tokenizer.vocab`, `vocab.txt`)
- Updated model checkpoint with expanded weights

---

## Critical Implementation Details

### Weight Layout Transformation

Understanding the layout transformation is crucial:

| Component | Original Layout | Expanded Layout | Transformation |
|-----------|-----------------|-----------------|----------------|
| Decoder Embed | `[vocab, blank]` | `[vocab, new, blank]` | Insert new tokens, shift blank |
| Joint Output | `[vocab, dur0-4, blank]` | `[vocab, new, dur0-4, blank]` | Insert new tokens, shift specials |

### New Token Initialization

The initialization strategy must balance inference correctness AND training capability:

| Layer | New Token Init | Value | Reason |
|-------|----------------|-------|--------|
| Decoder Embed | Normal | std * 0.01 | Small random values, allows gradient flow |
| Joint Weight | Normal | std * 0.01 | Small random values, allows gradient flow |
| Joint Bias | Fill | mean - 5.0 | Slightly negative, but gradients can still flow |

This ensures:
1. Softmax(logits) for new tokens is low (English preserved)
2. Gradients can flow during training (new tokens can learn)
3. English transcription is unchanged during inference

**Why Not -1000 Bias?**

An extreme `-1000` bias was tried initially, but it blocked gradient flow completely:
- The softmax derivative approaches zero for extreme logits
- New tokens could never be updated during training
- Chinese characters would output repetitive patterns like `的的的的的的`

The balanced initialization (`mean - 5.0`) provides enough suppression for inference while allowing learning during fine-tuning.

---

## Usage

```bash
# Via Makefile
make expand-tdt-chinese \
    ZH_MANIFEST=./training_data/KeSpeech/data/manifests/train_manifest.json \
    EXPANDED_TDT_OUTPUT=./models/parakeet-tdt-0.6b-v3-zh.nemo \
    MAX_CHINESE_CHARS=5000

# Direct Python
python expand_tdt_tokenizer.py \
    --base-model nvidia/parakeet-tdt-0.6b-v3 \
    --zh-manifest /path/to/chinese_manifest.json \
    --output ./models/parakeet-tdt-0.6b-v3-zh.nemo \
    --max-chinese-chars 5000 \
    --test-audio /path/to/english_test.wav
```

---

## Verification

To verify the expansion preserved English performance:

```bash
# Transcribe with original model
make run-parakeet-single \
    PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3 \
    AUDIO=/path/to/english_test.wav

# Transcribe with expanded model  
make run-parakeet-single \
    PARAKEET_MODEL=./models/parakeet-tdt-0.6b-v3-zh.nemo \
    AUDIO=/path/to/english_test.wav
```

**Expected**: Identical transcription results for English audio.

---

## Fine-Tuning the Expanded Model

After expansion, the model:
- ✅ Transcribes English perfectly (preserved weights)
- ❌ Cannot transcribe Chinese (new tokens have zero weights)

To enable Chinese transcription, fine-tune on Chinese ASR data:

```yaml
# NeMo fine-tuning config
model:
  train_ds:
    manifest_filepath: /path/to/chinese_train_manifest.json
  
  optim:
    lr: 1e-4  # Lower LR to preserve English
    
  # Consider freezing encoder initially
  encoder:
    freeze: true  # Optional: preserve acoustic features
```

The fine-tuning process will:
1. Learn embeddings for Chinese characters
2. Learn joint output weights for Chinese characters
3. Adapt LSTM for Chinese language patterns
4. (If unfrozen) Adapt encoder for Chinese acoustics

---

## Summary

The `expand_tdt_tokenizer.py` script successfully expands the Parakeet TDT model vocabulary by:

1. **Carefully preserving** all original model weights
2. **Strategically initializing** new tokens with a balanced approach:
   - Small random weights for gradient flow during training
   - Slightly negative bias (mean - 5) for low probability during inference
3. **Maintaining exact** English transcription performance
4. **Enabling** new language learning through fine-tuning

The key insights are:
- NeMo's `change_vocabulary()` reinitializes too many weights, requiring manual save/restore
- Token index shifts for blank and duration tokens must be handled carefully
- **Initialization must balance inference AND training needs** — extreme values (-1000 bias) block gradient flow
