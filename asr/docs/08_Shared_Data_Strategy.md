# Shared Data Collection Strategy
# One Recording Session for Both ASR & TTS

**Author:** AI Assistant  
**Date:** October 12, 2025  
**Status:** Recommended Strategy ✅

---

## 🎯 Executive Summary

**The Big Idea:** Instead of recording voice data separately for ASR and TTS projects, we record ONCE at highest quality (48kHz/24-bit) and process it differently for each use case.

**Result:**
- **Save $4,500-8,000** (43-50% cost reduction on recording)
- **Save 15-25 hours** of studio time
- **Same timeline** (4 months total)
- **Better consistency** across models

---

## ✅ Why This Works

### Technical Compatibility

| Specification | ASR Requirement | TTS Requirement | Recording Strategy |
|---------------|-----------------|-----------------|-------------------|
| **Sample Rate** | 16 kHz (resamples from 8-48kHz) | 22.05-44.1 kHz | Record at **48 kHz** → auto-downsample |
| **Bit Depth** | 16-bit minimum, 24-bit preferred | 24-bit required | Record at **24-bit** |
| **Format** | WAV, MP3, FLAC, etc. | WAV uncompressed | Record **WAV** |
| **Duration** | 1-30 seconds per clip | 2-20 seconds per clip | Record **3-20 seconds** |
| **Content** | Malaysian multilingual speech | Malaysian multilingual speech | **Identical!** ✓ |
| **Transcripts** | Required (99%+ accuracy) | Required (exact match) | **Same transcripts!** ✓ |
| **Speakers** | 3-5 diverse voices | 3-5 diverse voices | **Same speakers!** ✓ |

**Bottom Line:** All requirements are compatible! Recording at 48kHz/24-bit satisfies both ASR and TTS needs.

---

## 💰 Cost-Benefit Analysis

### Original Plan (Separate Recording Sessions)

```
📅 Month 1-2 (ASR Project):
├─ Record 30-40 hours for ASR
├─ Voice actors: $150/hour
├─ Total: $4,500-6,000
└─ Transcribe all recordings

📅 Month 3-4 (TTS Project):
├─ Record 15-25 hours for TTS (SAME SPEAKERS!)
├─ Voice actors: $400/hour (premium for TTS quality)
├─ Total: $6,000-10,000
└─ Transcribe all recordings (AGAIN!)

💸 Total Recording Cost: $10,500-16,000
⏰ Total Studio Time: 45-65 hours
❌ Problem: Paying voice actors TWICE for similar work!
```

### Optimized Plan (Shared Recording Session)

```
📅 Month 1 Only (Weeks 1-2):
├─ Record 30-40 hours at HIGHEST quality (48kHz/24-bit)
├─ Voice actors: $200/hour (bulk negotiated rate)
├─ Total: $6,000-8,000
└─ Transcribe ONCE

🤖 Automatic Processing (Week 2):
├─ Downsample to 16kHz for ASR → asr/data/
├─ Downsample to 22.05kHz for TTS → tts/data/
└─ Copy transcripts to both projects

📅 Month 2: ASR Training (uses 16kHz data)
📅 Month 3-4: TTS Training (uses 22.05kHz data)

💸 Total Recording Cost: $6,000-8,000
⏰ Total Studio Time: 30-40 hours
🎉 SAVINGS: $4,500-8,000 (43-50% reduction!)
```

---

## 📊 Impact on Project Budget

| Budget Category | Original | Optimized | Savings |
|-----------------|----------|-----------|---------|
| **ASR Recording** | $4,500-6,000 | $3,000-4,000 | $1,500-2,000 |
| **TTS Recording** | $6,000-10,000 | $0 (reuse!) | $6,000-10,000 |
| **Transcription** | 2× work | 1× work | 50% time saved |
| **Total Project (ASR + TTS)** | $105K-140K | $95K-125K | **$10K-15K** ⬇️ |

**Note:** TTS still has preprocessing costs, but no NEW recording costs!

---

## 🛠️ Implementation Guide

### Phase 1: Recording (Month 1, Weeks 1-2)

#### Week 1: Preparation

**Day 1-2: Sentence Generation**
```python
# Generate 10,000-12,000 Malaysian sentences
# (These will be used for BOTH ASR and TTS!)

import openai

def generate_sentences(count=10000):
    prompt = """
    Generate natural Malaysian English sentences with:
    - Code-switching (Malay + English)
    - Discourse particles (lah, leh, loh)
    - Length: 8-15 words
    - Diverse topics (casual, work, shopping, food)
    """
    # Generate sentences...
    return sentences

sentences = generate_sentences(10000)
# Save to sentences_master.txt
```

**Day 3-4: Voice Actor Recruitment**
```
Criteria:
- 3-5 Malaysian speakers
- Native fluency in Malay + English
- Professional recording experience
- Age: 25-45 years
- Gender balance: 50/50

Negotiation:
- Rate: $200/hour (bulk rate for 30-40 hours)
- Contract: 10-13 hours per speaker
- Payment: 50% upfront, 50% on completion
- Rights: Full commercial use
```

**Day 5-7: Studio Setup**
```yaml
Equipment Required:
  microphone: Large-diaphragm condenser (e.g., Neumann U87)
  audio_interface: 24-bit capable (e.g., Focusrite Clarett)
  recording_software: Pro Tools, Audacity, or Reaper
  monitoring: High-quality headphones

Recording Settings:
  sample_rate: 48000  # Hz (highest quality)
  bit_depth: 24       # bit (professional quality)
  format: WAV         # uncompressed
  channels: 1         # mono
  buffer_size: 512    # samples (low latency)
```

#### Week 2: Recording Sessions

**Daily Schedule (per voice actor):**
```
09:00-09:15  Warm-up & mic check
09:15-10:30  Record 100 sentences (Batch 1)
10:30-10:40  Break
10:40-12:00  Record 100 sentences (Batch 2)
12:00-13:00  Lunch
13:00-14:30  Record 100 sentences (Batch 3)
14:30-14:40  Break
14:40-16:00  Record 80-100 sentences (Batch 4)

Daily Output: 280-300 sentences = 1.5-2 hours of audio
Weekly Output per speaker: 8-10 hours of audio
Total (3 speakers × 2 weeks): 30-40 hours of audio ✅
```

**Recording Workflow:**
```
For each sentence:
1. Display sentence on screen
2. Actor reads naturally (1-2 takes)
3. Engineer marks best take
4. Save as: speaker_ID_sentence_ID.wav
5. Auto-log metadata (text, speaker, timestamp)
6. Next sentence

Quality Checks:
- No clipping (peaks < -3 dBFS)
- Clean pronunciation
- Natural code-switching
- Proper particle usage
```

---

### Phase 2: Processing (Month 1, End of Week 2)

**Automatic Audio Processing Script:**

```python
#!/usr/bin/env python3
# process_master_recordings.py

import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def process_recordings(master_dir, output_base):
    """
    Process master 48kHz recordings for both ASR and TTS
    
    Args:
        master_dir: Directory with master 48kHz/24-bit WAV files
        output_base: Base directory for processed outputs
    """
    master_files = sorted(Path(master_dir).glob("*.wav"))
    
    print(f"Found {len(master_files)} master recordings")
    print("Processing for ASR and TTS...")
    
    for master_file in tqdm(master_files):
        # Load master recording (48kHz/24-bit)
        audio, sr = librosa.load(str(master_file), sr=48000, mono=True)
        
        # === FOR ASR: Downsample to 16kHz ===
        asr_audio = librosa.resample(audio, orig_sr=48000, target_sr=16000)
        asr_path = Path(output_base) / "asr" / "data" / master_file.name
        asr_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(asr_path), asr_audio, 16000, subtype='PCM_16')
        
        # === FOR TTS: Downsample to 22.05kHz ===
        tts_audio = librosa.resample(audio, orig_sr=48000, target_sr=22050)
        tts_path = Path(output_base) / "tts" / "data" / master_file.name
        tts_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(tts_path), tts_audio, 22050, subtype='PCM_24')
    
    print(f"\n✅ Processed {len(master_files)} files")
    print(f"   ASR data: {output_base}/asr/data/ (16kHz)")
    print(f"   TTS data: {output_base}/tts/data/ (22.05kHz)")

if __name__ == "__main__":
    process_recordings(
        master_dir="recordings/master",
        output_base="recordings/processed"
    )
```

**Create Shared Metadata:**

```python
#!/usr/bin/env python3
# create_shared_metadata.py

import pandas as pd
from pathlib import Path

def create_metadata(sessions_csv, output_base):
    """
    Create metadata files for both ASR and TTS from single source
    
    Args:
        sessions_csv: CSV with columns: filename, text, speaker_id, duration
        output_base: Base directory for outputs
    """
    # Load master session data
    sessions = pd.read_csv(sessions_csv)
    
    print(f"Loaded {len(sessions)} recordings")
    
    # === ASR Metadata (HuggingFace Datasets format) ===
    asr_metadata = pd.DataFrame({
        'audio': sessions['filename'].apply(lambda x: f"data/{x}"),
        'text': sessions['text'],
        'speaker_id': sessions['speaker_id'],
        'duration': sessions['duration'],
        'sampling_rate': 16000,
    })
    
    asr_output = Path(output_base) / "asr" / "metadata.csv"
    asr_output.parent.mkdir(parents=True, exist_ok=True)
    asr_metadata.to_csv(asr_output, index=False)
    print(f"✅ ASR metadata: {asr_output} ({len(asr_metadata)} samples)")
    
    # === TTS Metadata (same content, different format) ===
    tts_metadata = pd.DataFrame({
        'audio_file': sessions['filename'].apply(lambda x: f"data/{x}"),
        'text': sessions['text'],
        'speaker_id': sessions['speaker_id'],
        'duration': sessions['duration'],
        'sample_rate': 22050,
        'language': sessions['text'].apply(detect_language_mix),  # Helper function
    })
    
    tts_output = Path(output_base) / "tts" / "metadata.csv"
    tts_output.parent.mkdir(parents=True, exist_ok=True)
    tts_metadata.to_csv(tts_output, index=False)
    print(f"✅ TTS metadata: {tts_output} ({len(tts_metadata)} samples)")

def detect_language_mix(text):
    """Simple heuristic to detect Malay/English mix"""
    malay_words = ['saya', 'nak', 'kita', 'ada', 'sudah']
    english_words = ['the', 'is', 'are', 'can', 'want']
    
    has_malay = any(word in text.lower() for word in malay_words)
    has_english = any(word in text.lower() for word in english_words)
    
    if has_malay and has_english:
        return 'ms-en'
    elif has_malay:
        return 'ms'
    else:
        return 'en'

if __name__ == "__main__":
    create_metadata(
        sessions_csv="recordings/master/sessions.csv",
        output_base="recordings/processed"
    )
```

**Run Processing:**

```bash
# After Week 2 recording completes:

# 1. Process audio files
python process_master_recordings.py
# Output:
#   recordings/processed/asr/data/*.wav (16kHz)
#   recordings/processed/tts/data/*.wav (22.05kHz)

# 2. Create metadata
python create_shared_metadata.py
# Output:
#   recordings/processed/asr/metadata.csv
#   recordings/processed/tts/metadata.csv

# 3. Verify processing
python verify_datasets.py
# Checks: file counts, sample rates, durations match
```

---

### Phase 3: Usage

**Month 2 (ASR Training):**
```python
# Load ASR dataset
from datasets import load_dataset

asr_dataset = load_dataset(
    'audiofolder',
    data_dir='recordings/processed/asr'
)

# Train Whisper-large v3 with Unsloth
# Uses 16kHz data automatically
```

**Months 3-4 (TTS Training):**
```python
# Load TTS dataset
import pandas as pd

tts_metadata = pd.read_csv('recordings/processed/tts/metadata.csv')

# Train XTTS v2
# Uses 22.05kHz data automatically
```

**No additional recording needed!** All data collected in Month 1. 🎉

---

## 📁 Final Directory Structure

```
recordings/
├── master/                     # Original recordings (48kHz/24-bit)
│   ├── SP001_0001.wav         # Keep these as backup/archive
│   ├── SP001_0002.wav
│   ├── ...
│   └── sessions.csv           # Master metadata
│
└── processed/
    ├── asr/                    # ASR-ready data (16kHz/16-bit)
    │   ├── data/
    │   │   ├── SP001_0001.wav
    │   │   └── ...
    │   └── metadata.csv
    │
    └── tts/                    # TTS-ready data (22.05kHz/24-bit)
        ├── data/
        │   ├── SP001_0001.wav
        │   └── ...
        └── metadata.csv

Storage:
- Master: ~10-15 GB (48kHz/24-bit)
- ASR: ~2-3 GB (16kHz/16-bit)
- TTS: ~4-6 GB (22.05kHz/24-bit)
- Total: ~16-24 GB (very manageable!)
```

---

## ✅ Quality Assurance

**Verification Checklist:**

```python
# verify_datasets.py

def verify_shared_datasets(base_dir):
    """Verify ASR and TTS datasets match"""
    
    import pandas as pd
    from pathlib import Path
    
    asr_meta = pd.read_csv(f"{base_dir}/asr/metadata.csv")
    tts_meta = pd.read_csv(f"{base_dir}/tts/metadata.csv")
    
    # Check 1: Same number of files
    assert len(asr_meta) == len(tts_meta), "File count mismatch!"
    
    # Check 2: Same text content
    assert asr_meta['text'].equals(tts_meta['text']), "Text mismatch!"
    
    # Check 3: Same speakers
    assert asr_meta['speaker_id'].equals(tts_meta['speaker_id']), "Speaker mismatch!"
    
    # Check 4: Sample rates correct
    assert asr_meta['sampling_rate'].iloc[0] == 16000, "ASR sample rate wrong!"
    assert tts_meta['sample_rate'].iloc[0] == 22050, "TTS sample rate wrong!"
    
    # Check 5: Files exist
    for file in asr_meta['audio']:
        assert Path(f"{base_dir}/asr/{file}").exists(), f"Missing: {file}"
    
    for file in tts_meta['audio_file']:
        assert Path(f"{base_dir}/tts/{file}").exists(), f"Missing: {file}"
    
    print("✅ All verification checks passed!")
    print(f"   Total recordings: {len(asr_meta)}")
    print(f"   ASR-ready: {len(asr_meta)} @ 16kHz")
    print(f"   TTS-ready: {len(tts_meta)} @ 22.05kHz")

verify_shared_datasets("recordings/processed")
```

---

## 🎯 Success Criteria

**End of Month 1 (Week 2):**
- ✅ 30-40 hours recorded at 48kHz/24-bit
- ✅ 10,000-12,000 sentences covered
- ✅ 3-5 diverse speakers
- ✅ All transcripts validated (99%+ accuracy)
- ✅ ASR dataset ready (16kHz)
- ✅ TTS dataset ready (22.05kHz)
- ✅ Budget: $6,000-8,000 (vs $10,500-16,000 original)

**Outcome:** BOTH ASR and TTS projects have high-quality data, ready to start training!

---

## 🚀 Summary

**What we're doing:**
1. Record ONCE at highest quality (48kHz/24-bit)
2. Process automatically for ASR (16kHz) and TTS (22.05kHz)
3. Use same transcripts for both
4. Train ASR in Month 2, TTS in Months 3-4

**Why it's brilliant:**
- **Save $4,500-8,000** on recording costs
- **Save 15-25 hours** of studio time
- **Better consistency** across models
- **Same timeline** (4 months total)
- **No quality compromise**

**The catch:** None! This is industry-standard practice. Recording at high quality and downsampling is exactly how professional studios work.

---

**Questions? See `/PROJECT_TIMELINE_SUMMARY.md` for full 4-month timeline.**

