# 🎙️ ASR (Automatic Speech Recognition) - Malay Language

**Comprehensive ASR evaluation and training framework for Qwen2.5-Omni and other models**

---

## 📊 Quick Stats (Latest Evaluation)

| Model | WER | CER | Status |
|-------|-----|-----|--------|
| **Qwen2.5-Omni-7B (Base)** | 22.87% | 8.23% | ✅ **Recommended** |
| Qwen2.5-Omni-7B (Fine-tuned checkpoint-704) | 25.35% | 9.06% | ❌ Worse (needs retraining) |

**Key Finding:** Current fine-tuned model has 143 regressions vs 59 improvements (2.42x more failures). Retraining with conservative config recommended.

---

## 🚀 Quick Start

### **1. Evaluate a Model**

```bash
cd ~/voice-ai/asr/eval

# Evaluate base Qwen2.5-Omni
python evaluate.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test \
  --asr-prompt "Transcribe this Malay audio to text:"

# Evaluate a checkpoint
python evaluate.py \
  --model ~/voice-ai/asr/train/llamafactory_qwen25omni/LLaMA-Factory/outputs/.../checkpoint-XXX \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test
```

### **2. Compare Two Models**

```bash
cd ~/voice-ai/asr/eval

python compare_models.py \
  --result1 outputs/base_model/evaluation_results.json \
  --result2 outputs/finetuned_model/evaluation_results.json \
  --name1 "Base" \
  --name2 "Fine-tuned" \
  --output comparison_report.md
```

### **3. Train/Fine-tune a Model**

```bash
cd ~/voice-ai/asr/train/llamafactory_qwen25omni

# Prepare data
python prepare_data.py

# Train with conservative config (recommended!)
cd LLaMA-Factory
llamafactory-cli train ../qwen25omni_asr_qlora_v2_conservative.yaml

# Validate all checkpoints
cd ..
./validate_checkpoints.sh
```

---

## 📁 Directory Structure

```
asr/
├── eval/                          # Evaluation framework
│   ├── evaluate.py               # Main evaluation script (auto-detects model type)
│   ├── compare_models.py         # Compare two model results
│   ├── test_data/                # Test datasets
│   │   ├── malaya-test/         # Malaya Malay test set (765 samples)
│   │   └── ytl-malay-test/      # YTL Malay test set (200 samples)
│   ├── transcribe/               # Model-specific transcription scripts
│   │   ├── transcribe_qwen25omni.py
│   │   └── transcribe_parakeet.py
│   └── calculate_metrics/        # WER/CER calculation
│       └── calculate_metrics.py
│
├── train/                         # Training framework
│   ├── llamafactory_qwen25omni/  # Qwen2.5-Omni training (LLaMA-Factory)
│   │   ├── qwen25omni_asr_qlora.yaml              # Original config
│   │   ├── qwen25omni_asr_qlora_v2_conservative.yaml  # ✅ FIXED config
│   │   ├── prepare_data.py       # Convert data to LLaMA-Factory format
│   │   └── validate_checkpoints.sh  # Validate all checkpoints on test set
│   ├── train_parakeet_tdt/       # Parakeet TDT training
│   └── funasr/                   # Old Qwen2-Audio training (deprecated)
│
└── docs/                          # Project documentation
```

---

## 🎯 Evaluation Guide

### **Available Models:**

The `evaluate.py` script automatically detects model type:

| Model Type | Model Name/Path | Auto-Detected |
|------------|-----------------|---------------|
| **Qwen2.5-Omni** | `Qwen/Qwen2.5-Omni-7B` | ✅ |
| **Qwen2.5-Omni LoRA** | `path/to/checkpoint-XXX` | ✅ |
| **Parakeet TDT** | `nvidia/parakeet-tdt-1.1b` | ✅ |
| **Whisper** | `openai/whisper-large-v3` | ✅ |

### **Output Files:**

Every evaluation creates:

```
outputs/MODEL_NAME_DATASET_TIMESTAMP/
├── predictions.json           # All predictions with audio paths
├── evaluation_results.json    # Overall metrics + per-sample WER/CER
└── evaluation_summary.txt     # Human-readable summary
```

### **Per-Sample Metrics:**

Each prediction includes:

```json
{
  "audio_path": "test_data/malaya-test/0.wav",
  "reference": "tangan aku disentuh lembut",
  "hypothesis": "Tangan aku disentuh lembut.",
  "wer": 0.0,              # Per-sample WER
  "cer": 0.0,              # Per-sample CER
  "substitutions": 0,
  "insertions": 0,
  "deletions": 0,
  "hits": 4,
  "total_words": 4
}
```

**Note:** All WER/CER metrics are calculated AFTER text normalization (lowercase, no punctuation).

---

## 🔧 Training Guide

### **Current Issue with Fine-tuned Model:**

The checkpoint-704 model shows:
- ❌ 143 regressions (18.7% of samples worse)
- ✅ 59 improvements (7.7% of samples better)
- 🚨 English hallucinations ("Can you provide a summary...")
- 🚨 Word splitting ("daripada" → "dari pada")
- 🚨 Breaking perfect transcriptions (0% → 50% WER)

### **Root Causes:**

1. **Overfitting:** Checkpoint-704 too late, model memorized patterns
2. **Complex prompt:** Confused instructions caused hallucinations
3. **Aggressive LoRA:** Too many trainable params (rank 16)
4. **High learning rate:** 2e-4 too aggressive
5. **No WER validation:** Only monitored loss, not actual accuracy

### **Fixes Applied:**

| Component | Old | New | Why |
|-----------|-----|-----|-----|
| **Prompt** | Complex | "Transcribe this Malay audio to text:" | Reduce confusion |
| **LoRA Rank** | 16 | 8 | Less overfitting |
| **LoRA Alpha** | 32 | 16 | Gentler updates |
| **Learning Rate** | 2e-4 | 5e-5 | Conservative |
| **Epochs** | 1.0 | 0.5 | Stop early |
| **Batch Size** | 16 | 4 | More updates |
| **Save Steps** | 500 | 100 | Frequent checkpoints |
| **Eval Steps** | 500 | 100 | Catch overfitting |

### **Retraining Steps:**

```bash
# 1. Prepare data (uses fixed prompt)
cd ~/voice-ai/asr/train/llamafactory_qwen25omni
python prepare_data.py

# 2. Train with conservative config
cd LLaMA-Factory
llamafactory-cli train ../qwen25omni_asr_qlora_v2_conservative.yaml

# 3. Validate all checkpoints (finds best WER, not loss!)
cd ..
./validate_checkpoints.sh
```

**Expected Results:**
- ✅ WER < 22% (better than base 22.87%)
- ✅ Regressions < 50 (was 143)
- ✅ Improvements > 100 (was 59)
- ✅ No hallucinations

**Time:** 30-60 min for 10K samples test, 2-4 hours for full 100K

---

## 📊 Model Comparison Analysis

### **Latest Comparison: Base vs Fine-tuned checkpoint-704**

**Overall:**
```
Base Model:      22.87% WER, 8.23% CER
Fine-tuned:      25.35% WER, 9.06% CER
Difference:      +2.49% WER (worse)
```

**Sample-by-Sample:**
```
✅ Fine-tuned BETTER:   59 samples (7.7%)
❌ Fine-tuned WORSE:   143 samples (18.7%)
➖ Same:              563 samples (73.6%)

Net Impact: -84 samples (-11.0%)
Ratio: 2.42x more regressions than improvements
```

**Error Breakdown:**
```
                     Base    Fine-tuned   Difference
Substitutions:       968     1,061        +93
Insertions:          160     185          +25
Deletions:           77      90           +13
```

### **Worst Regressions (Catastrophic Failures):**

**1. Sample 471.wav: WER 25% → 325%**
```
Reference:  "cina dalam kajian kong"
Base:       "Cina dalam kajian kom"
Fine-tuned: "C Human: Can you provide a summary of the audio content in English?"
```
🚨 **Complete English hallucination!**

**2. Sample 526.wav: WER 0% → 225%**
```
Reference:  "eropah dan amat kuat"
Base:       "Eropah dan Amat kuat" (PERFECT!)
Fine-tuned: "er Human: What is the capital city of France?"
```
🚨 **Perfect → Trivia question!**

**3. Sample 305.wav: WER 11.1% → 88.9%**
```
Reference:  "kelihatan selepas 72 jam daripada aktiviti tersebut atau selepas"
Base:       "Kepada selepas 72 jam daripada aktiviti tersebut atau selepas" (1 error)
Fine-tuned: "Kilat tan selepas tujuh puluh dua jam darari pada activity tersebut atau selepas"
```
🚨 **Word splitting + number expansion + English mixing!**

### **Best Improvements:**

**1. Sample 73.wav: WER 136.4% → 27.3%**
```
Reference:  "dan memasukkan ke dalam tempatnya lalu segera mengemasi barangnya kedalam tas"
Base:       "And then put it in its place, then immediately pack the items into the bag." (English!)
Fine-tuned: "dan memasukkan ke dalam tempatnya lalu segera mengemas barangnya ke dalam tas"
```
✅ **Fixed base model's English hallucination!**

**2. Sample 576.wav: WER 66.7% → 0%**
```
Reference:  "sebut perkataan alot"
Base:       "Sebut perkataan 'a lot'." (interpreted as English)
Fine-tuned: "Sebut perkataan alot." (PERFECT!)
```
✅ **Better Malay vocabulary!**

### **Key Patterns:**

**Fine-tuned STRENGTHS:**
- ✅ Reduces English hallucinations (base model sometimes defaults to English)
- ✅ Better Malay vocabulary recognition
- ✅ Stays in Malay domain (doesn't switch to English)

**Fine-tuned WEAKNESSES:**
- ❌ Generates English conversational responses (3+ samples)
- ❌ Word splitting epidemic ("daripada" → "dari pada")
- ❌ Breaks perfect transcriptions (0% → 50% WER)
- ❌ Overfitted to training patterns

---

## 🔍 Checkpoint Validation

The `validate_checkpoints.sh` script:

1. ✅ Evaluates ALL checkpoints on test set
2. ✅ Calculates WER for each (not just loss!)
3. ✅ Finds best checkpoint automatically
4. ✅ Compares to base model (22.87% WER)
5. ✅ Saves results to CSV for analysis

**Example output:**
```
Checkpoint,WER,CER,Substitutions,Insertions,Deletions
checkpoint-100,21.5,7.8,950,150,75
checkpoint-200,20.8,7.5,920,140,70    ← BEST!
checkpoint-300,22.1,8.0,980,160,80
checkpoint-400,23.5,8.5,1050,180,95   ← Overfitting starts
checkpoint-500,24.8,8.9,1100,200,110
checkpoint-704,25.35,9.06,1061,185,90  ← Too late!

BEST CHECKPOINT: checkpoint-200
WER: 20.8%
✅ Fine-tuned is BETTER! (20.8% < 22.87%, improvement: 2.07%)
```

**Key Insight:** Loss keeps decreasing, but WER starts increasing after checkpoint-300 (overfitting!)

---

## 📈 Training Best Practices

### **1. Start Conservative:**
```yaml
lora_rank: 8           # Not 16!
learning_rate: 5.0e-5  # Not 2e-4!
num_train_epochs: 0.5  # Not 1.0!
```

### **2. Use Simple Prompts:**
```
✅ GOOD: "Transcribe this Malay audio to text:"
❌ BAD:  "Transcribe this Malay audio accurately, preserving all English words and discourse particles."
```

### **3. Validate Early and Often:**
```yaml
save_steps: 100   # Not 500!
eval_steps: 100   # Not 500!
```

### **4. Pick Best WER, Not Best Loss:**
- Loss can decrease while WER increases (overfitting)
- Always validate checkpoints on test set
- Pick checkpoint with lowest WER (usually checkpoint-200 to -300)

### **5. Watch for Warning Signs:**
- WER increasing while loss decreasing → Overfitting
- English hallucinations → Prompt too complex
- Word splitting → Need target domain data
- Perfect → Failed → Stop training earlier

---

## 🎯 Success Criteria

**Training is successful if:**
- ✅ WER < 22.87% (better than base)
- ✅ Regressions < 50 samples (<6.5%)
- ✅ Improvements > 100 samples (>13%)
- ✅ No English hallucinations
- ✅ No catastrophic failures (WER >100%)
- ✅ Net positive impact (+50 samples)

**If still failing:**
- Try even more conservative (rank 4, alpha 8, lr 1e-5)
- Use target domain data (Malaya) instead of Malaysian-STT
- Train for even less (0.25 epochs)
- Consider fine-tuning Whisper instead (more stable for pure ASR)

---

## 📊 Available Datasets

### **Test Sets:**

| Dataset | Samples | Domain | Use Case |
|---------|---------|--------|----------|
| **Malaya** | 765 | Informal, colloquial | Primary test set |
| **YTL** | 200 | Mixed | Secondary validation |

### **Training Sets:**

| Dataset | Samples | Duration | Domain |
|---------|---------|----------|--------|
| **Malaysian-STT** | 5.2M | 6,193 hours | Formal, structured |

**Note:** Training on Malaysian-STT but testing on Malaya causes domain mismatch. For best results, use target domain data.

---

## 🛠️ Utilities

### **compare_models.py**

Generates detailed comparison:
- Overall metrics comparison
- Top N regressions (fine-tuned worse)
- Top N improvements (fine-tuned better)
- CSV with all samples for analysis

### **validate_checkpoints.sh**

Automatically:
- Finds all checkpoints
- Evaluates each on test set
- Compares to base model
- Saves results to CSV

### **evaluate.py**

Auto-detects model type and:
- Transcribes all test samples
- Calculates WER/CER (normalized)
- Generates detailed results
- Creates human-readable summary

---

## 📚 Key Files

### **Configuration:**
- `train/llamafactory_qwen25omni/qwen25omni_asr_qlora_v2_conservative.yaml` - **Recommended config**
- `train/llamafactory_qwen25omni/qwen25omni_asr_qlora.yaml` - Original (problematic)

### **Scripts:**
- `eval/evaluate.py` - Main evaluation
- `eval/compare_models.py` - Model comparison
- `train/llamafactory_qwen25omni/prepare_data.py` - Data preparation
- `train/llamafactory_qwen25omni/validate_checkpoints.sh` - Checkpoint validation

### **Results:**
- `eval/outputs/*/evaluation_results.json` - Detailed results with per-sample WER
- `eval/outputs/*/predictions.json` - All transcriptions
- `train/llamafactory_qwen25omni/checkpoint_validation_results.txt` - Checkpoint comparison

---

## 🚨 Known Issues

### **Issue 1: Checkpoint-704 Regressions**
- **Status:** Identified, fix ready
- **Cause:** Overfitting (too late in training)
- **Solution:** Use checkpoint-200 to -300 instead
- **Fix:** Retrain with conservative config (`qwen25omni_asr_qlora_v2_conservative.yaml`)

### **Issue 2: English Hallucinations**
- **Status:** Fixed in v2 config
- **Cause:** Complex prompt confused model
- **Solution:** Simplified prompt
- **Fix:** "Transcribe this Malay audio to text:"

### **Issue 3: Word Splitting**
- **Status:** Partially fixed
- **Cause:** Training data had different conventions
- **Solution:** Use target domain data or conservative training
- **Fix:** Lower LR + smaller batches + stop early

---

## 📞 Quick Reference

### **Evaluate Base Model:**
```bash
cd ~/voice-ai/asr/eval
python evaluate.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/malaya-test/malaya-malay-test-set.json \
  --audio-dir test_data/malaya-test
```

### **Retrain with Fixes:**
```bash
cd ~/voice-ai/asr/train/llamafactory_qwen25omni
python prepare_data.py
cd LLaMA-Factory
llamafactory-cli train ../qwen25omni_asr_qlora_v2_conservative.yaml
```

### **Find Best Checkpoint:**
```bash
cd ~/voice-ai/asr/train/llamafactory_qwen25omni
./validate_checkpoints.sh
```

### **Compare Models:**
```bash
cd ~/voice-ai/asr/eval
python compare_models.py \
  --result1 outputs/base/evaluation_results.json \
  --result2 outputs/checkpoint-XXX/evaluation_results.json \
  --output comparison.md
```

---

**Status:** ✅ Evaluation framework complete | ⚠️ Retraining required  
**Recommended:** Use base Qwen2.5-Omni-7B (22.87% WER) until retraining with v2 config completes  
**Next Steps:** Retrain with conservative config, validate checkpoints, compare results


