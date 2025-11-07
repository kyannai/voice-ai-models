# Data Collection Quick Start Guide
# Single Recording Session for ASR & TTS

---

## 🎯 The Strategy (TL;DR)

**Record ONCE → Use for BOTH ASR and TTS**

```
┌─────────────────────────────────────────────────────────────┐
│  Month 1, Weeks 1-2: ONE Recording Session                  │
│  • Record 30-40 hours at 48kHz/24-bit                       │
│  • 3-5 voice actors × 10-13 hours each                      │
│  • Cost: $6,000-8,000 (bulk rate: $200/hr)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Automatic Processing (1 day)
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  ASR Dataset    │         │  TTS Dataset    │
    │  (16kHz/16-bit) │         │  (22.05kHz/24-bit)│
    │                 │         │                 │
    │  Month 2:       │         │  Months 3-4:    │
    │  Train Whisper  │         │  Train XTTS v2  │
    └─────────────────┘         └─────────────────┘
```

**Savings: $4,500-8,000 + 15-25 hours of studio time** 🎉

---

## ✅ Week-by-Week Checklist

### Week 1: Preparation

- [ ] **Day 1-2:** Generate 10,000-12,000 Malaysian sentences with GPT-4
- [ ] **Day 3-4:** Recruit 3-5 voice actors ($200/hr bulk rate)
- [ ] **Day 5-6:** Book studio (48kHz/24-bit recording capability)
- [ ] **Day 7:** Set up GPU environment, install Unsloth

### Week 2: Recording & Processing

- [ ] **Day 8-12:** Record with voice actors (5 days, 2-3 hrs/day each)
  - Target: 30-40 hours total audio
  - Quality: 48kHz/24-bit WAV mono
  - Each actor: 10-13 hours
  
- [ ] **Day 13:** Run automatic processing script
  ```bash
  python process_master_recordings.py
  # Creates: asr/data/ (16kHz) + tts/data/ (22.05kHz)
  ```

- [ ] **Day 14:** Verify datasets & backup
  ```bash
  python verify_datasets.py
  # Checks: file counts, sample rates, transcripts match
  ```

**✅ End of Week 2:** Ready to start ASR training (Month 2)

---

## 📊 Cost Comparison

| Approach | Recording Cost | Studio Time | Transcription | Total |
|----------|----------------|-------------|---------------|-------|
| **❌ Separate** | $10,500-16,000 | 45-65 hrs | 2× work | $10,500+ |
| **✅ Shared** | $6,000-8,000 | 30-40 hrs | 1× work | $6,000-8,000 |
| **💰 SAVINGS** | **$4,500-8,000** | **15-25 hrs** | **50% time** | **43-50%** |

---

## 🛠️ Tools You'll Need

### Week 1: Sentence Generation
```bash
# Install dependencies
pip install openai pandas

# Generate sentences
python generate_sentences.py --count 10000 --output sentences.txt
```

### Week 2: Recording
```yaml
Studio Equipment:
  - Large-diaphragm condenser microphone
  - 24-bit audio interface
  - Pro Tools, Audacity, or Reaper
  - Quiet recording booth

Recording Settings:
  sample_rate: 48000  # Hz
  bit_depth: 24       # bit
  format: WAV
  channels: 1         # mono
```

### Week 2: Processing
```bash
# Install processing libraries
pip install librosa soundfile tqdm

# Process recordings
python process_master_recordings.py \
  --master_dir recordings/master \
  --output_dir recordings/processed

# Create metadata
python create_shared_metadata.py \
  --sessions recordings/master/sessions.csv \
  --output_dir recordings/processed

# Verify
python verify_datasets.py \
  --base_dir recordings/processed
```

---

## 📁 Final Output Structure

```
recordings/
├── master/                      # Keep as backup
│   ├── SP001_0001.wav          # 48kHz/24-bit originals
│   └── sessions.csv             # Master metadata
│
└── processed/
    ├── asr/                     # Ready for ASR training
    │   ├── data/                # 16kHz/16-bit
    │   │   └── SP001_0001.wav
    │   └── metadata.csv
    │
    └── tts/                     # Ready for TTS training
        ├── data/                # 22.05kHz/24-bit
        │   └── SP001_0001.wav
        └── metadata.csv

Storage: ~16-24 GB total (very manageable!)
```

---

## ✅ Success Criteria

**After Week 2, you should have:**

- ✅ 30-40 hours of master recordings (48kHz/24-bit)
- ✅ 10,000-12,000 sentences covered
- ✅ 3-5 diverse Malaysian speakers
- ✅ ASR dataset ready: `processed/asr/` (16kHz)
- ✅ TTS dataset ready: `processed/tts/` (22.05kHz)
- ✅ Metadata files for both projects
- ✅ All transcripts validated (99%+ accuracy)
- ✅ Budget: $6,000-8,000 (vs $10,500-16,000 saved!)

**Ready for Month 2:** ASR training can start immediately  
**Ready for Month 3-4:** TTS training uses same data (no new recording!)

---

## 🚀 Next Steps

1. **Month 2 (Weeks 3-8):** Train ASR model
   - Use `processed/asr/` dataset
   - Fine-tune Whisper-large v3 with Unsloth
   - Target: WER < 15%

2. **Months 3-4 (Weeks 9-16):** Train TTS model
   - Use `processed/tts/` dataset (same recordings!)
   - Fine-tune XTTS v2
   - Target: MOS > 4.0

---

## 📚 Detailed Documentation

- **Full Strategy:** [SHARED_DATA_STRATEGY.md](SHARED_DATA_STRATEGY.md)
- **Project Timeline:** [PROJECT_TIMELINE_SUMMARY.md](PROJECT_TIMELINE_SUMMARY.md)
- **ASR Execution Plan:** [asr/docs/07_Project_Execution_Plan.md](asr/docs/07_Project_Execution_Plan.md)
- **TTS Execution Plan:** [tts/docs/07_Project_Execution_Plan.md](tts/docs/07_Project_Execution_Plan.md)

---

**Questions?** This is a standard industry practice. Recording at high quality and resampling for different use cases is how professional studios work!

