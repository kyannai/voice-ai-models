# The TDT Vocabulary Expansion Journey

*A debugging story: How we discovered why expanding a TDT model's vocabulary broke English transcription, and the surprising fix.*

---

## The Goal

Expand the `nvidia/parakeet-tdt-0.6b-v3` English ASR model to support Chinese characters, while preserving **identical** English transcription performance.

Simple, right? We already had a working CTC expansion script. TDT should be similar...

---

## Chapter 1: The Naive Approach

### Initial Assumption

"TDT is just like CTC but with a decoder. We just need to expand the tokenizer and the output layer."

Based on the CTC script, the initial approach was:
1. Expand the SentencePiece tokenizer with Chinese characters
2. Call `model.change_vocabulary()` to resize the model
3. Copy original weights to the new layers
4. Done!

### First Implementation

```python
# Save original joint output weights
ori_joint_weight = asr_model.joint.joint_net[2].weight.clone()
ori_joint_bias = asr_model.joint.joint_net[2].bias.clone()

# Change vocabulary (resize layers)
asr_model.change_vocabulary(new_tokenizer_dir, "bpe")

# Restore original weights
asr_model.joint.joint_net[2].weight[:old_vocab_size].copy_(ori_joint_weight[:old_vocab_size])
asr_model.joint.joint_net[2].bias[:old_vocab_size].copy_(ori_joint_bias[:old_vocab_size])
```

### First Test

```bash
$ make run-parakeet-single PARAKEET_MODEL=./expanded-model.nemo AUDIO=english_test.wav
```

**Result**: Complete garbage. Not even recognizable as language.

```
Original: "the quick brown fox jumps over the lazy dog"
Expanded: "的的的一一一是是是在在在有有有不不不了了了"
```

**Reaction**: 😱 The model was outputting Chinese characters for English audio!

---

## Chapter 2: Understanding TDT Architecture

### Investigation: What's Different About TDT?

CTC models have a simple architecture:
- Encoder → Linear output layer

TDT models are more complex:
- Encoder → processes audio
- Decoder (Prediction Network) → predicts next token autoregressively
- Joint Network → combines encoder + decoder outputs

### Key Discovery #1: The Decoder Has an Embedding Layer

```python
>>> print(asr_model.decoder.prediction)
StatelessTransducerDecoder(
  (embed): Embedding(8193, 640)  # <-- Vocab-dependent!
  (dec_rnn): LSTM(640, 640, num_layers=2, batch_first=True)
)
```

The decoder has an **embedding layer** of shape `[vocab_size + 1, hidden_dim]`!

This wasn't being saved or restored.

### Fix Attempt #2: Save Decoder Embedding Too

```python
# Save decoder embedding
ori_embed_weight = asr_model.decoder.prediction.embed.weight.clone()

# After change_vocabulary...
asr_model.decoder.prediction.embed.weight[:old_vocab_size].copy_(
    ori_embed_weight[:old_vocab_size]
)
```

### Second Test

**Result**: Still garbage, but different garbage.

```
Expanded: "ththththehehehe ququququickickickick"
```

**Observation**: It's trying to say English words but stuttering horribly.

---

## Chapter 3: The LSTM Revelation

### Investigation: What Else Is Being Reset?

Let's check what `change_vocabulary()` actually does...

```python
# Before change_vocabulary
for name, param in asr_model.decoder.named_parameters():
    print(f"{name}: {param.mean():.4f}")

# dec_rnn.lstm.weight_ih_l0: 0.0012
# dec_rnn.lstm.weight_hh_l0: -0.0008
# ...

asr_model.change_vocabulary(...)

# After change_vocabulary
for name, param in asr_model.decoder.named_parameters():
    print(f"{name}: {param.mean():.4f}")

# dec_rnn.lstm.weight_ih_l0: 0.0000  <-- DIFFERENT!
# dec_rnn.lstm.weight_hh_l0: 0.0001  <-- DIFFERENT!
```

**Discovery #2**: `change_vocabulary()` reinitializes the **entire decoder**, including the LSTM layers!

The LSTM weights are vocabulary-independent (they process 640-dim embeddings regardless of vocab size), but NeMo reinitializes them anyway.

### Fix Attempt #3: Save ALL Decoder Weights

```python
# Save entire decoder state
ori_decoder_state = {}
for name, param in asr_model.decoder.prediction.named_parameters():
    ori_decoder_state[name] = param.clone()

# After change_vocabulary, restore LSTM weights
for name, ori_param in ori_decoder_state.items():
    if 'embed' not in name:  # Handle embedding separately
        new_param = dict(asr_model.decoder.prediction.named_parameters())[name]
        new_param.copy_(ori_param)
```

### Third Test

**Result**: Better! English words are recognizable now.

```
Original: "the quick brown fox jumps over the lazy dog"
Expanded: "the quick brown fox jumps over the lazy dog 的一是"
```

**Observation**: The English is mostly correct, but there are random Chinese characters at the end!

---

## Chapter 4: The Joint Network Mystery

### Investigation: What About the Joint Network?

```python
>>> print(asr_model.joint)
RNNTJoint(
  (pred): Sequential(
    (0): Linear(in_features=640, out_features=640, bias=True)
    (1): LayerNorm((640,), ...)
  )
  (enc): Sequential(
    (0): Linear(in_features=640, out_features=640, bias=True)
    (1): LayerNorm((640,), ...)
  )
  (joint_net): Sequential(
    (0): Linear(in_features=640, out_features=640, bias=True)
    (1): ReLU()
    (2): Linear(in_features=640, out_features=8198, bias=True)  # Output layer
  )
)
```

The joint network has:
- `enc`: Encoder projection (vocabulary-independent)
- `pred`: Prediction projection (vocabulary-independent)
- `joint_net[0]`: Hidden layer (vocabulary-independent)
- `joint_net[2]`: Output layer (vocabulary-DEPENDENT)

**Discovery #3**: Even the vocabulary-independent layers were being reset!

### Fix Attempt #4: Save ALL Joint Network Weights

```python
ori_joint_state = {}
for name, param in asr_model.joint.named_parameters():
    ori_joint_state[name] = param.clone()

# After change_vocabulary...
for name, ori_param in ori_joint_state.items():
    if 'joint_net.2' not in name:  # Handle output layer separately
        new_param = dict(asr_model.joint.named_parameters())[name]
        new_param.copy_(ori_param)
```

### Fourth Test

**Result**: English is now correct most of the time!

```
Original: "the quick brown fox jumps over the lazy dog"
Expanded: "the quick brown fox jumps over the lazy 的"
```

But there's still an occasional Chinese character sneaking in.

---

## Chapter 5: The Blank Token Shift

### Investigation: Special Token Positions

TDT has special tokens at the end of the vocabulary:
- Positions 8192-8196: Duration tokens (0-4)
- Position 8197: Blank token

When we expand the vocabulary:
- New Chinese chars go to positions 8192-13191
- Duration tokens should move to 13192-13196
- Blank token should move to 13197

But were we handling this correctly?

```python
# Original code - WRONG!
new_joint.weight[:old_vocab_size].copy_(ori_joint_weight[:old_vocab_size])
# This copies positions 0-8191, but what about 8192-8197?
```

**Discovery #4**: Duration and blank tokens weren't being moved to their new positions!

### Fix Attempt #5: Handle Special Token Shift

```python
num_special_tokens = 6  # 5 durations + 1 blank

# Copy original vocab (0 to old_vocab_size-1)
new_joint.weight[:old_vocab_size].copy_(ori_joint_weight[:old_vocab_size])
new_joint.bias[:old_vocab_size].copy_(ori_joint_bias[:old_vocab_size])

# Move special tokens to new positions
new_joint.weight[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
    ori_joint_weight[old_vocab_size:old_vocab_size + num_special_tokens]
)
new_joint.bias[new_vocab_size:new_vocab_size + num_special_tokens].copy_(
    ori_joint_bias[old_vocab_size:old_vocab_size + num_special_tokens]
)
```

Same for decoder embedding (blank token at the end).

### Fifth Test

**Result**: Still Chinese characters appearing!

```
Original: "the quick brown fox jumps over the lazy dog"
Expanded: "the quick 的 brown fox 一 jumps over 是 the lazy dog"
```

The Chinese characters were appearing **in the middle** of English words now.

---

## Chapter 6: The Logit Investigation

### Deep Debugging: What's Happening at Inference Time?

Time to look at the actual logits during decoding.

```python
# Hook into the joint network output
def debug_forward(self, encoder_output, prediction_output):
    logits = self.original_forward(encoder_output, prediction_output)
    
    # Check logits for English vs Chinese tokens
    english_logits = logits[..., :8192].max(dim=-1).values
    chinese_logits = logits[..., 8192:13192].max(dim=-1).values
    
    print(f"English max: {english_logits.mean():.2f}")
    print(f"Chinese max: {chinese_logits.mean():.2f}")
    
    return logits
```

**Output**:
```
English max: 2.34
Chinese max: 4.89  <-- HIGHER!
```

**Discovery #5**: Chinese token logits were **HIGHER** than English token logits!

### Why Were Chinese Logits Higher?

The new Chinese tokens (positions 8192-13191) had:
- **Random weights** from Xavier initialization
- These random weights produced non-zero activations
- By chance, many of these activations were positive and relatively large

Meanwhile, the original English tokens:
- Had trained weights that produced appropriate logits
- But the model was tuned to produce moderate logit values

The random Chinese logits were simply overwhelming the trained English logits!

---

## Chapter 7: The Eureka Moment

### The Solution: Silence the New Tokens

If random weights produce too-high logits, we need to ensure new tokens produce **very low** logits until they're properly trained.

```python
# Initialize new token weights to ZERO
new_joint.weight[old_vocab_size:new_vocab_size].zero_()

# Set bias to very negative value (-1000)
new_joint.bias[old_vocab_size:new_vocab_size].fill_(-1000.0)

# Similarly for decoder embeddings
new_embed.weight[old_vocab_size:new_vocab_size].zero_()
```

**The Math**:
- Output logit = weight · input + bias
- With zero weights: logit = 0 · input + (-1000) = -1000
- After softmax: exp(-1000) ≈ 0

The new tokens effectively have **zero probability** of being selected!

### Final Test

```bash
$ # Original model
$ make run-parakeet-single PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3 AUDIO=test.wav
Result: "the quick brown fox jumps over the lazy dog"

$ # Expanded model
$ make run-parakeet-single PARAKEET_MODEL=./expanded-model.nemo AUDIO=test.wav
Result: "the quick brown fox jumps over the lazy dog"
```

**IDENTICAL!** 🎉

---

## Chapter 8: The Complete Picture

### Summary of All Issues Found

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| #1 | Complete garbage | Joint output not preserved | Save & restore joint weights |
| #2 | Stuttering | Decoder embedding not preserved | Save & restore embedding |
| #3 | Different garbage | LSTM weights reset | Save & restore all decoder params |
| #4 | Mostly works | Joint projections reset | Save & restore all joint params |
| #5 | Wrong special tokens | Blank/duration not shifted | Shift special tokens to new positions |
| #6 | Random Chinese chars | New tokens have high logits | Zero weights, -1000 bias |

### The Final Weight Preservation Strategy

```
COMPONENTS TO SAVE AND RESTORE:
├── Decoder (decoder.prediction)
│   ├── embed.weight [vocab+1, 640]      → Restore + shift blank
│   ├── dec_rnn.lstm.weight_ih_l0        → Restore exactly
│   ├── dec_rnn.lstm.weight_hh_l0        → Restore exactly
│   ├── dec_rnn.lstm.bias_ih_l0          → Restore exactly
│   ├── dec_rnn.lstm.bias_hh_l0          → Restore exactly
│   └── (layer 1 weights...)             → Restore exactly
│
└── Joint (joint)
    ├── enc.0.weight, enc.0.bias         → Restore exactly
    ├── enc.1.weight, enc.1.bias         → Restore exactly
    ├── pred.0.weight, pred.0.bias       → Restore exactly
    ├── pred.1.weight, pred.1.bias       → Restore exactly
    ├── joint_net.0.weight, .bias        → Restore exactly
    └── joint_net.2.weight, .bias        → Restore + shift specials + zero new
```

### New Token Initialization Strategy

```
Position Range          | Weight Init | Bias Init | Reason
------------------------|-------------|-----------|--------
0 to old_vocab-1       | Restored    | Restored  | Original English tokens
old_vocab to new_vocab-1| Zero        | -1000     | New Chinese (silent until trained)
new_vocab to end       | Restored    | Restored  | Shifted duration/blank tokens
```

---

## Chapter 9: The Training Surprise

### The Setup

With English transcription working perfectly, it was time to fine-tune on Chinese data. We ran an overfitting test on a small dataset (32 samples) with Chinese, Malay, and English.

### Expected Result

After 100 epochs of overfitting:
- English: Perfect (overfit) ✅
- Malay: Perfect (overfit) ✅
- Chinese: Perfect (overfit) ❓

### Actual Result (First Attempt)

```
Epoch 50:
WER reference: 最糟的老婆很可能是很好的女人
WER predicted: 的的的的的的的的的的的的的的

Epoch 99:
WER reference: 最糟的老婆很可能是很好的女人  
WER predicted: 使了的的的的的的的的的的的的
```

**Problem**: Chinese was outputting repetitive characters like `的的的的的` instead of learning the correct transcriptions!

Malay and English were overfitting correctly, but Chinese was stuck in a loop.

### Investigation: Why Was Chinese Not Learning?

Looking back at our initialization:

```python
# Our "solution" for inference
new_joint.bias[old_vocab_size:new_vocab_size].fill_(-1000.0)
```

**The problem**: A bias of `-1000` makes the logit so negative that:
1. The token has ~zero probability during inference ✅ (this was intentional)
2. The gradient is essentially **zero** during training ❌ (this was NOT intended!)

**The math**:
```
logit = weight · input + (-1000)
probability = softmax(logit) ≈ 0

gradient = target_prob - predicted_prob
         ≈ target_prob - 0
         = target_prob

But the gradient for the bias is multiplied by the softmax derivative,
which is essentially 0 when the logit is at -1000!
```

This created a **gradient desert** — the new tokens couldn't learn because no gradients were flowing to them.

### Why Malay Worked But Chinese Didn't

**Malay** uses Latin script with many overlapping characters with English:
- `a`, `e`, `i`, `o`, `u`, `k`, `s`, `t`, etc. already existed in vocab
- The model could use existing English tokens for most Malay words
- Only a few new characters needed to be learned

**Chinese** uses entirely new characters (CJK ideographs):
- All 5000+ Chinese tokens were NEW (positions 8192+)
- Every character had a `-1000` bias
- The model **couldn't produce any Chinese output**

### The Fix: Balanced Initialization

The solution was to initialize new tokens in a way that:
1. **Doesn't interfere with English inference** (slightly negative)
2. **Allows gradients to flow during training** (not too negative)

```python
# Get statistics from existing tokens
existing_weight_std = ori_joint_weight[:old_vocab_size].std()
existing_bias_mean = ori_joint_bias[:old_vocab_size].mean()

# Initialize new tokens with small random weights
new_joint.weight[old_vocab_size:new_vocab_size].normal_(
    mean=0.0, 
    std=existing_weight_std * 0.01  # Small but non-zero
)

# Initialize bias to slightly below average (not extreme)
new_joint.bias[old_vocab_size:new_vocab_size].fill_(
    existing_bias_mean - 5.0  # About -5 below mean, not -1000!
)

# Same for decoder embeddings
existing_embed_std = ori_embed_weight[:old_vocab_size].std()
new_embed.weight[old_vocab_size:new_vocab_size].normal_(
    mean=0.0,
    std=existing_embed_std * 0.01
)
```

### The Training Config Adjustments

We also increased the learning rate for the overfit test:

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `learning_rate` | `1e-4` | `5e-4` | Stronger gradients for new tokens |
| `batch_size` | `4` | `2` | More gradient updates |
| `warmup_steps` | `50` | `10` | Reach full LR faster |

### Result After Fix

```
Epoch 31:
WER reference: 最糟的老婆很可能是很好的女人
WER predicted: 最糟的老婆很可能是很好的女人  ✅ PERFECT!

WER reference: 中牧总公司拟延期履行相关承诺
WER predicted: 中牧总公司拟延期履行相关承诺  ✅ PERFECT!

WER reference: 尽管乐视电动车未能现身舞台
WER predicted: 尽管乐视电动车未能现身舞台  ✅ PERFECT!
```

**Chinese was now learning correctly!** 🎉

---

## The Complete Picture (Updated)

### Summary of All Issues Found

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| #1 | Complete garbage | Joint output not preserved | Save & restore joint weights |
| #2 | Stuttering | Decoder embedding not preserved | Save & restore embedding |
| #3 | Different garbage | LSTM weights reset | Save & restore all decoder params |
| #4 | Mostly works | Joint projections reset | Save & restore all joint params |
| #5 | Wrong special tokens | Blank/duration not shifted | Shift special tokens to new positions |
| #6 | Random Chinese chars | New tokens have high logits | Zero weights, negative bias |
| **#7** | **Chinese not training** | **Bias too negative (-1000)** | **Use mean-5.0 bias, small random weights** |

### The Final Initialization Strategy (Updated)

```
Position Range           | Weight Init              | Bias Init           | Purpose
------------------------|--------------------------|---------------------|--------
0 to old_vocab-1        | Restored from original   | Restored            | English tokens (unchanged)
old_vocab to new_vocab-1| Normal(0, std*0.01)      | existing_mean - 5.0 | New tokens (trainable)
new_vocab to end        | Restored from original   | Restored            | Duration/blank (shifted)
```

---

## Lessons Learned

### 1. Never Trust Framework "Convenience" Methods
`change_vocabulary()` sounds helpful, but it does too much. Always inspect what framework methods actually do to your model weights.

### 2. Debug Layer by Layer
When output is wrong, systematically check each layer. Compare weights before/after, check intermediate activations.

### 3. New Parameters Need Careful Initialization
Random initialization for vocabulary expansion can produce unexpected behavior. New tokens should be "silent" until trained.

### 4. TDT is More Complex Than CTC
The decoder (prediction network) adds significant complexity. Every vocabulary-dependent and vocabulary-adjacent component must be handled.

### 5. Special Tokens Are Easy to Forget
Duration tokens and blank tokens exist at specific positions. When shifting the vocabulary, these must move too.

### 6. Inference vs Training Requirements Differ
What works for inference may break training! A `-1000` bias preserves English perfectly but prevents new tokens from ever learning. The initialization must balance both needs.

### 7. Test Training, Not Just Inference
Always run an overfitting test on a small dataset before full training. This catches issues like blocked gradients that wouldn't show up in inference testing.

---

## Timeline

| Step | Action | Time Spent | Outcome |
|------|--------|------------|---------|
| 1 | Initial implementation (like CTC) | 1 hour | Complete failure |
| 2 | Discovered decoder embedding | 30 min | Still broken |
| 3 | Found LSTM reset issue | 1 hour | Better but wrong |
| 4 | Found joint projection reset | 45 min | Mostly working |
| 5 | Fixed special token positions | 30 min | Random Chinese chars |
| 6 | Investigated logits | 2 hours | Found the real issue |
| 7 | Implemented zero init + negative bias | 15 min | **Inference works!** |
| 8 | Ran overfitting test | 30 min | Chinese not learning |
| 9 | Analyzed training dynamics | 1 hour | Found gradient blockage |
| 10 | Implemented balanced initialization | 15 min | **Training works!** |

**Total debugging time**: ~8 hours

---

## The Moral of the Story

What seemed like a simple vocabulary expansion turned into a deep dive into the TDT architecture. The key insights were:

1. **Preserving weights isn't enough** — new tokens must be initialized carefully
2. **Inference and training have different requirements** — what works for one may break the other

The final solution balances both needs:
- Small random weights = gradients can flow during training
- Slightly negative bias (mean - 5) = low probability but not blocked
- Result = new tokens don't interfere with English AND can be learned

Sometimes the fix is simple, but finding it requires understanding the entire system — including the training dynamics, not just inference behavior.

---

*Document written: January 2026*
*Updated: January 2026 (added training fix)*
*Script: expand_tdt_tokenizer.py*
*Model: nvidia/parakeet-tdt-0.6b-v3*
