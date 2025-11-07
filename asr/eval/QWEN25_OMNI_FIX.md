# 🔧 Qwen2.5-Omni Fix Applied

## ❌ Error Encountered
```
ERROR - Error transcribing: not enough values to unpack (expected 2, got 1)
```

## 🔍 Root Cause

The official Qwen2.5-Omni code returns a tuple:
```python
text_ids, audio = model.generate(**inputs)  # Returns (text, audio)
```

But when we call `model.disable_talker()` (to save ~2GB GPU memory), the model no longer generates audio output, so `generate()` only returns:
```python
text_ids = model.generate(**inputs)  # Returns just text_ids
```

## ✅ Fix Applied

Updated `transcribe_qwen25omni.py` to handle both cases:

```python
# NEW: Handle both with/without talker
output = self.model.generate(**inputs, ...)

if isinstance(output, tuple):
    # Talker enabled: (text_ids, audio)
    text_ids, audio_output = output
else:
    # Talker disabled: just text_ids
    text_ids = output
```

## 📋 What Changed

**File:** `transcribe/transcribe_qwen25omni.py`
- Lines 182-203: Updated generation and unpacking logic
- Now gracefully handles both talker-enabled and talker-disabled modes
- Keeps the ~2GB memory savings from `disable_talker()`

## ⚠️ Expected Warnings (Safe to Ignore)

You'll see this warning - **it's harmless**:
```
WARNING - System prompt modified, audio output may not work as expected.
```

**Why:** We're using a custom ASR prompt instead of the default system prompt. Since we disabled audio output (ASR only needs text), this warning doesn't affect us.

## 🚀 Next Steps

1. **Upload the fixed file** to your server:
   - `transcribe/transcribe_qwen25omni.py` (updated)

2. **Run evaluation again:**
```bash
python evaluate.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --test-data test_data/ytl-malay-test/asr_ground_truths.json \
  --audio-dir test_data/ytl-malay-test \
  --asr-prompt "Transcribe this Malay audio accurately, preserving all English words and discourse particles. Return only the transcribed text with no preamble or explanation." \
  --max-samples 10  # Quick test first!
```

## ✨ Benefits Retained

✅ `disable_talker()` still works - saves ~2GB GPU memory  
✅ Flash-Attention 2 auto-detection still works  
✅ All optimizations intact  
✅ No functionality lost  

The fix just makes the code more robust to handle the disabled talker case!
