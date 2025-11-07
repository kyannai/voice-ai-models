# Fine-tuning Strategy & Guide
# Malaysian Multilingual TTS System
# **Using Sesame CSM-1B + Unsloth**

**Version:** 2.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** ML Engineering Team

---

## 1. Fine-tuning Overview

### 1.1 Training Philosophy

**🎯 Strategic Decision: Fine-tune Instead of Train from Scratch**

We will leverage **Sesame CSM-1B** as our base model and use **Unsloth** for efficient fine-tuning. This approach provides:

✅ **Faster Development**: 2-3 days instead of 2-3 months  
✅ **Less Data Required**: 10-30 hours instead of 100+ hours  
✅ **Lower Compute Cost**: Single GPU instead of multi-GPU cluster  
✅ **Better Initial Quality**: Start from state-of-the-art baseline  
✅ **Proven Architecture**: Code-switching built-in  

**Our Fine-tuning Principles:**

1. **Parameter-Efficient Fine-tuning (PEFT)**: Use LoRA/QLoRA to update only 1-5% of parameters
2. **Malaysian-Specific Adaptation**: Focus on accent, particles, and prosody
3. **Gradual Complexity**: Start with clean data, add code-switching progressively
4. **Quality Over Quantity**: 20 hours of high-quality data beats 100 hours of mediocre data
5. **Continuous Evaluation**: Monitor Malaysian-specific metrics throughout

### 1.2 Fine-tuning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

Phase 0: Setup & Preparation
    ├─ Download Sesame CSM-1B base model
    ├─ Install Unsloth framework
    ├─ Prepare Malaysian training data (10-30 hours)
    └─ Setup training environment (1x GPU)

Phase 1: Initial Fine-tuning
    ├─ Freeze base model, train LoRA adapters
    ├─ Start with single-language utterances
    ├─ Focus on Malaysian accent adaptation
    ├─ Training: 3000-5000 steps (~12-24 hours)
    └─ Target: Decent Malaysian accent

Phase 2: Code-Switching Fine-tuning
    ├─ Introduce code-switched data
    ├─ Fine-tune language transition smoothness
    ├─ Training: 2000-3000 additional steps (~8-12 hours)
    └─ Target: Natural code-switching

Phase 3: Particle Refinement
    ├─ Focus on particle pronunciation (lah, leh, loh)
    ├─ Increase weight on particle-containing utterances
    ├─ Training: 1000-2000 additional steps (~4-8 hours)
    └─ Target: Perfect particle intonation

Phase 4: Multi-Speaker Adaptation (Optional)
    ├─ Add speaker-specific adapters
    ├─ Voice cloning capability
    ├─ Training: 1000-2000 steps per speaker
    └─ Target: Multiple voice options

Total Timeline: 3-7 days (vs 8-12 weeks training from scratch)
Total Data: 10-30 hours (vs 100+ hours for scratch)
Total Compute: ~72-168 GPU-hours (vs 5000+ GPU-hours)
```

---

### 1.2.1 Sample Data for Each Phase (Detailed Examples)

**Important:** These are **progressive fine-tuning phases** - we continue training from the previous phase, gradually adding complexity. Think of it as teaching a child: simple words first, then sentences, then complex conversations.

---

#### **Phase 0: Setup & Preparation (Day 0)**

**Goal:** Download model, prepare environment, organize data

**No training yet - just preparation:**

```bash
# Download Sesame CSM-1B
huggingface-cli download facebook/sesame-csm-1b

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Organize your data
data/
├── phase1_single_language/     # 5-10 hours
├── phase2_code_switching/      # 5-10 hours
├── phase3_particles/           # 2-5 hours
└── phase4_multi_speaker/       # 1-3 hours per speaker
```

**Data Quality Requirements:**
- Sample rate: 22.05kHz or 44.1kHz (will resample)
- Format: WAV, FLAC, or MP3
- Noise: Minimal background noise
- Length: 2-10 seconds per utterance
- Silence: < 0.5 seconds at start/end

---

#### **Phase 1: Initial Fine-tuning - Malaysian Accent (Days 1-2)**

**Goal:** Teach the model Malaysian English and Malay pronunciation (single language per utterance)

**Data Composition: 5-10 hours (500-1000 samples)**
- 60% Malaysian English (clean sentences)
- 30% Standard Malay (clean sentences)
- 10% Malaysian Mandarin/Pinyin (optional)

**Sample Data (Malaysian English):**

```python
phase1_english_samples = [
    {
        "audio": "phase1/me_001.wav",
        "text": "Good morning, how are you today?",
        "language": "en",
        "accent": "malaysian_english",
        "speaker_id": "speaker_01",
        "duration": 3.2,
        "notes": "Malaysian English - lighter 'r', flatter intonation"
    },
    {
        "audio": "phase1/me_002.wav",
        "text": "I want to go shopping this weekend.",
        "language": "en",
        "accent": "malaysian_english",
        "speaker_id": "speaker_01",
        "duration": 3.5,
        "notes": "Focus on Malaysian 'r' pronunciation"
    },
    {
        "audio": "phase1/me_003.wav",
        "text": "Can you help me with this problem?",
        "language": "en",
        "accent": "malaysian_english",
        "speaker_id": "speaker_01",
        "duration": 2.8,
    },
    {
        "audio": "phase1/me_004.wav",
        "text": "The weather is very hot today.",
        "language": "en",
        "accent": "malaysian_english",
        "speaker_id": "speaker_01",
        "duration": 2.5,
    },
    # ... 300 more English samples
]

phase1_malay_samples = [
    {
        "audio": "phase1/ms_001.wav",
        "text": "Selamat pagi, apa khabar?",
        "language": "ms",
        "accent": "malaysian_malay",
        "speaker_id": "speaker_01",
        "duration": 2.8,
        "notes": "Standard Malay greeting"
    },
    {
        "audio": "phase1/ms_002.wav",
        "text": "Saya hendak pergi ke kedai.",
        "language": "ms",
        "accent": "malaysian_malay",
        "speaker_id": "speaker_01",
        "duration": 3.0,
        "notes": "Clean Malay sentence, no mixing"
    },
    {
        "audio": "phase1/ms_003.wav",
        "text": "Boleh tolong saya tak?",
        "language": "ms",
        "accent": "malaysian_malay",
        "speaker_id": "speaker_01",
        "duration": 2.3,
    },
    {
        "audio": "phase1/ms_004.wav",
        "text": "Cuaca hari ini sangat panas.",
        "language": "ms",
        "accent": "malaysian_malay",
        "speaker_id": "speaker_01",
        "duration": 2.7,
    },
    # ... 200 more Malay samples
]
```

**What the model learns in Phase 1:**
- ✅ Malaysian English pronunciation (lighter 'r', flatter tone)
- ✅ Malay pronunciation (phonetic consistency)
- ✅ Basic prosody for each language separately
- ❌ NOT learning code-switching yet
- ❌ NOT learning particles yet

**Training Configuration:**
```python
phase1_config = {
    "base_model": "facebook/sesame-csm-1b",
    "data_path": "data/phase1_single_language/",
    "batch_size": 4,
    "learning_rate": 2e-4,
    "steps": 3000-5000,
    "lora_r": 16,
    "focus": "accent_adaptation",
}
```

---

#### **Phase 2: Code-Switching Fine-tuning (Days 3-4)**

**Goal:** Teach smooth transitions between Malay and English in the same sentence

**Data Composition: 5-10 hours (500-1000 samples)**
- 100% code-switched sentences
- Mix of 2 languages (Malay + English)
- Gradually increase switching frequency

**Sample Data (Code-Switching):**

```python
phase2_samples = [
    # Light code-switching (1-2 switches per sentence)
    {
        "audio": "phase2/cs_001.wav",
        "text": "I want to pergi shopping.",
        "language_tags": ["en", "en", "en", "ms", "en"],
        "word_langs": ["en", "en", "en", "ms", "en"],
        "speaker_id": "speaker_01",
        "duration": 2.5,
        "notes": "Simple switch: English → Malay → English"
    },
    {
        "audio": "phase2/cs_002.wav",
        "text": "Boleh you help me tak?",
        "language_tags": ["ms", "en", "en", "en", "ms"],
        "word_langs": ["ms", "en", "en", "en", "ms"],
        "speaker_id": "speaker_01",
        "duration": 2.3,
        "notes": "Malay bookends with English center"
    },
    
    # Medium code-switching (3-4 switches)
    {
        "audio": "phase2/cs_003.wav",
        "text": "Saya nak go to the kedai.",
        "language_tags": ["ms", "ms", "en", "en", "en", "ms"],
        "word_langs": ["ms", "ms", "en", "en", "en", "ms"],
        "speaker_id": "speaker_01",
        "duration": 2.8,
        "notes": "Natural Malaysian code-switching"
    },
    {
        "audio": "phase2/cs_004.wav",
        "text": "Can you tolong ambil that book?",
        "language_tags": ["en", "en", "ms", "ms", "en", "en"],
        "word_langs": ["en", "en", "ms", "ms", "en", "en"],
        "speaker_id": "speaker_01",
        "duration": 2.6,
    },
    
    # Heavy code-switching (5+ switches)
    {
        "audio": "phase2/cs_005.wav",
        "text": "Saya nak go shopping dengan my friends hari ini.",
        "language_tags": ["ms", "ms", "en", "en", "ms", "en", "en", "ms", "ms"],
        "word_langs": ["ms", "ms", "en", "en", "ms", "en", "en", "ms", "ms"],
        "speaker_id": "speaker_01",
        "duration": 3.5,
        "notes": "Complex switching pattern - very Malaysian!"
    },
    {
        "audio": "phase2/cs_006.wav",
        "text": "This morning I pergi pasar untuk buy some sayur.",
        "language_tags": ["en", "en", "en", "ms", "ms", "en", "en", "en", "ms"],
        "word_langs": ["en", "en", "en", "ms", "ms", "en", "en", "en", "ms"],
        "speaker_id": "speaker_01",
        "duration": 3.8,
    },
    
    # Real-world conversations
    {
        "audio": "phase2/cs_007.wav",
        "text": "Jom kita go makan, I very hungry already.",
        "language_tags": ["ms", "ms", "en", "ms", "en", "en", "en", "en"],
        "word_langs": ["ms", "ms", "en", "ms", "en", "en", "en", "en"],
        "speaker_id": "speaker_01",
        "duration": 3.2,
        "notes": "'Jom' is uniquely Malaysian"
    },
    {
        "audio": "phase2/cs_008.wav",
        "text": "You dah makan or not yet?",
        "language_tags": ["en", "ms", "ms", "en", "en", "en"],
        "word_langs": ["en", "ms", "ms", "en", "en", "en"],
        "speaker_id": "speaker_01",
        "duration": 2.4,
        "notes": "'dah' = already (Malay contraction)"
    },
    
    # ... 500+ more code-switching samples
]
```

**What the model learns in Phase 2:**
- ✅ Smooth language transitions (no jarring switches)
- ✅ Natural code-switching patterns
- ✅ Maintain accent consistency across switches
- ✅ Handle "Manglish" (Malaysian English with Malay words)
- ❌ NOT focusing on particles yet (they'll come in Phase 3)

**Training Configuration:**
```python
phase2_config = {
    "base_model": "checkpoints/phase1_final/",  # Continue from Phase 1!
    "data_path": "data/phase2_code_switching/",
    "batch_size": 4,
    "learning_rate": 1.5e-4,  # Slightly lower than Phase 1
    "steps": 2000-3000,
    "lora_r": 16,
    "focus": "language_transitions",
    "special_loss_weight": {
        "transition_smoothness": 1.5  # Extra weight on language boundaries
    }
}
```

---

#### **Phase 3: Particle Refinement (Day 5)**

**Goal:** Perfect pronunciation of Malaysian particles with correct intonation and prosody

**Data Composition: 2-5 hours (200-500 samples)**
- 100% sentences containing particles
- Multiple particles per sentence
- Focus on prosody (pitch, emphasis, duration)

**Sample Data (Particles):**

```python
phase3_samples = [
    # "lah" - Emphasis/completion
    {
        "audio": "phase3/p_001.wav",
        "text": "Okay lah, I will go.",
        "language_tags": ["en", "particle", "en", "en", "en"],
        "particle_tags": ["none", "lah", "none", "none", "none"],
        "particle_type": "lah_emphasis",
        "speaker_id": "speaker_01",
        "duration": 2.5,
        "prosody_notes": "Rise-fall intonation on 'lah', slight pause after"
    },
    {
        "audio": "phase3/p_002.wav",
        "text": "Sudah lah, don't worry about it.",
        "language_tags": ["ms", "particle", "en", "en", "en", "en"],
        "particle_tags": ["none", "lah", "none", "none", "none", "none"],
        "particle_type": "lah_completion",
        "speaker_id": "speaker_01",
        "duration": 2.8,
        "prosody_notes": "'Sudah lah' = dismissive tone, falling pitch"
    },
    
    # "leh" - Suggestion/possibility
    {
        "audio": "phase3/p_003.wav",
        "text": "Can try leh, maybe it works.",
        "language_tags": ["en", "en", "particle", "en", "en", "en"],
        "particle_tags": ["none", "none", "leh", "none", "none", "none"],
        "particle_type": "leh_suggestion",
        "speaker_id": "speaker_01",
        "duration": 2.6,
        "prosody_notes": "Rising intonation on 'leh', questioning tone"
    },
    {
        "audio": "phase3/p_004.wav",
        "text": "Boleh leh?",
        "language_tags": ["ms", "particle"],
        "particle_tags": ["none", "leh"],
        "particle_type": "leh_question",
        "speaker_id": "speaker_01",
        "duration": 1.5,
        "prosody_notes": "High rising tone, very Malaysian!"
    },
    
    # "lor" - Resignation/obviousness
    {
        "audio": "phase3/p_005.wav",
        "text": "Cannot help lor, I told you already.",
        "language_tags": ["en", "en", "particle", "en", "en", "en", "en"],
        "particle_tags": ["none", "none", "lor", "none", "none", "none", "none"],
        "particle_type": "lor_resignation",
        "speaker_id": "speaker_01",
        "duration": 3.0,
        "prosody_notes": "Flat/falling tone on 'lor', matter-of-fact"
    },
    
    # "meh" - Doubt/question
    {
        "audio": "phase3/p_006.wav",
        "text": "Really meh? Are you sure?",
        "language_tags": ["en", "particle", "en", "en", "en"],
        "particle_tags": ["none", "meh", "none", "none", "none"],
        "particle_type": "meh_doubt",
        "speaker_id": "speaker_01",
        "duration": 2.4,
        "prosody_notes": "High rising pitch on 'meh', skeptical tone"
    },
    
    # "mah" - Stating the obvious
    {
        "audio": "phase3/p_007.wav",
        "text": "Of course I know mah!",
        "language_tags": ["en", "en", "en", "en", "particle"],
        "particle_tags": ["none", "none", "none", "none", "mah"],
        "particle_type": "mah_obvious",
        "speaker_id": "speaker_01",
        "duration": 2.2,
        "prosody_notes": "Slightly annoyed tone, quick 'mah'"
    },
    
    # Multiple particles in one sentence
    {
        "audio": "phase3/p_008.wav",
        "text": "Cannot lah, I busy leh.",
        "language_tags": ["en", "particle", "en", "en", "particle"],
        "particle_tags": ["none", "lah", "none", "none", "leh"],
        "particle_type": "multiple",
        "speaker_id": "speaker_01",
        "duration": 2.5,
        "prosody_notes": "Two particles - 'lah' falling, 'leh' rising"
    },
    {
        "audio": "phase3/p_009.wav",
        "text": "Why you like that one lah?",
        "language_tags": ["en", "en", "en", "en", "en", "particle"],
        "particle_tags": ["none", "none", "none", "none", "none", "lah"],
        "particle_type": "lah_complaint",
        "speaker_id": "speaker_01",
        "duration": 2.7,
        "prosody_notes": "Complaining tone, drawn out 'lah'"
    },
    
    # ... 200+ more particle samples
]
```

**Particle Prosody Patterns (What model must learn):**

| Particle | Pitch Pattern | Duration | Emphasis | Usage |
|----------|--------------|----------|----------|-------|
| **lah** | Rise → Fall | Long (1.2-1.5x) | Strong | Completion, emphasis |
| **leh** | Rising | Medium (1.0-1.2x) | Medium | Suggestion, question |
| **lor** | Flat/Falling | Medium (1.0x) | Weak | Resignation, obvious |
| **meh** | High Rising | Short (0.9-1.1x) | Medium | Doubt, question |
| **mah** | Quick Fall | Short (0.8-1.0x) | Medium | Obvious fact |
| **loh** | Falling | Medium (1.0-1.2x) | Medium | Realization |

**What the model learns in Phase 3:**
- ✅ Particle-specific pitch contours
- ✅ Correct duration for each particle
- ✅ Appropriate emphasis and stress
- ✅ Context-dependent particle meaning
- ✅ Multiple particles in one sentence

**Training Configuration:**
```python
phase3_config = {
    "base_model": "checkpoints/phase2_final/",  # Continue from Phase 2!
    "data_path": "data/phase3_particles/",
    "batch_size": 4,
    "learning_rate": 1e-4,  # Even lower for fine details
    "steps": 1000-2000,
    "lora_r": 16,
    "focus": "particle_prosody",
    "special_loss_weight": {
        "pitch_loss": 2.0,      # 2x weight on pitch for particles
        "duration_loss": 1.5,    # 1.5x weight on duration
        "particle_emphasis": 2.0 # 2x weight on particle tokens
    }
}
```

---

#### **Phase 4: Multi-Speaker Adaptation (Days 6-7, Optional)**

**Goal:** Add multiple voices/speakers for diversity

**Data Composition: 1-3 hours per speaker (100-300 samples each)**
- Same type of data as Phase 3 (mixed code-switching + particles)
- Different speakers (male, female, different ages)
- Speaker-specific LoRA adapters

**Sample Data (Multi-Speaker):**

```python
phase4_speaker1 = [
    {
        "audio": "phase4/speaker01/s1_001.wav",
        "text": "Saya nak pergi shopping lah.",
        "speaker_id": "speaker_01",
        "speaker_name": "Sarah (Female, 25, KL)",
        "voice_characteristics": {
            "gender": "female",
            "age_range": "20-30",
            "accent": "kuala_lumpur",
            "pitch_avg": 210,  # Hz
            "speaking_rate": "medium"
        },
        "duration": 2.8,
    },
    # ... 100+ samples for speaker 1
]

phase4_speaker2 = [
    {
        "audio": "phase4/speaker02/s2_001.wav",
        "text": "Boleh tolong saya tak?",
        "speaker_id": "speaker_02",
        "speaker_name": "Ahmad (Male, 35, Penang)",
        "voice_characteristics": {
            "gender": "male",
            "age_range": "30-40",
            "accent": "penang",
            "pitch_avg": 120,  # Hz
            "speaking_rate": "fast"
        },
        "duration": 2.3,
    },
    # ... 100+ samples for speaker 2
]

phase4_speaker3 = [
    {
        "audio": "phase4/speaker03/s3_001.wav",
        "text": "Can you help me leh?",
        "speaker_id": "speaker_03",
        "speaker_name": "Mei Ling (Female, 45, Johor)",
        "voice_characteristics": {
            "gender": "female",
            "age_range": "40-50",
            "accent": "johor",
            "pitch_avg": 195,  # Hz
            "speaking_rate": "slow"
        },
        "duration": 2.5,
    },
    # ... 100+ samples for speaker 3
]
```

**What the model learns in Phase 4:**
- ✅ Multiple voice characteristics
- ✅ Speaker-specific LoRA adapters
- ✅ Voice cloning capability
- ✅ Regional accent variations (KL, Penang, Johor)

**Training Configuration:**
```python
phase4_config = {
    "base_model": "checkpoints/phase3_final/",
    "data_path": "data/phase4_multi_speaker/",
    "batch_size": 4,
    "learning_rate": 1e-4,
    "steps_per_speaker": 1000-2000,
    "lora_r": 16,
    "focus": "speaker_adaptation",
    "multi_speaker": True,
    "speaker_lora": True,  # Separate LoRA per speaker
}
```

---

### 1.2.2 Summary: Progressive Fine-tuning Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE TRAINING                       │
└──────────────────────────────────────────────────────────────┘

Phase 1 (Days 1-2): Learn Malaysian Accent
├─ Data: 5-10 hours, single language
├─ Focus: Pronunciation, basic prosody
└─ Result: Model speaks Malaysian English & Malay ✓

       ↓ Continue training with Phase 1 weights

Phase 2 (Days 3-4): Learn Code-Switching
├─ Data: 5-10 hours, mixed languages
├─ Focus: Smooth transitions
└─ Result: Model switches languages naturally ✓

       ↓ Continue training with Phase 2 weights

Phase 3 (Day 5): Perfect Particles
├─ Data: 2-5 hours, particle-rich
├─ Focus: Prosody, intonation, emphasis
└─ Result: Model pronounces "lah, leh, lor" correctly ✓

       ↓ Continue training with Phase 3 weights

Phase 4 (Days 6-7): Add Voices (Optional)
├─ Data: 1-3 hours per speaker
├─ Focus: Voice diversity
└─ Result: Multiple voices available ✓

Final Model: Malaysian TTS with code-switching + particles + multiple voices!
```

**Key Points:**
1. ✅ **Cumulative learning**: Each phase builds on previous phase
2. ✅ **Curriculum learning**: Start simple, add complexity gradually
3. ✅ **No retraining**: We never start from scratch between phases
4. ✅ **Efficient**: 10-30 hours total data (not 100+)
5. ✅ **Fast**: 3-7 days total (not 3 months)

---

### 1.3 Fine-tuning Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING INFRASTRUCTURE                                  │
│                    (Sesame CSM-1B + Unsloth)                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BASE MODEL LAYER                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                      Sesame CSM-1B (Pre-trained)                          │  │
│  │  • 1B parameters                                                          │  │
│  │  • Pre-trained on English, Mandarin, Malay, Cantonese                    │  │
│  │  • Code-switching capabilities built-in                                   │  │
│  │  • Download: Hugging Face / Facebook Research                            │  │
│  │  • Model Size: 4GB (FP32), 2GB (BF16), 550MB (4-bit)                     │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MALAYSIAN DATA LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Raw Audio   │  │ Transcripts  │  │  Lang Tags   │  │  Metadata    │       │
│  │   (WAV)      │  │   (TXT)      │  │   (JSON)     │  │   (JSON)     │       │
│  │              │  │              │  │              │  │              │       │
│  │ 10-30 hours  │  │ Mixed lang   │  │  ms/en/zh    │  │  Speaker ID  │       │
│  │ 22.05kHz     │  │ Code-switch  │  │  per word    │  │  Particles   │       │
│  │ 16-bit PCM   │  │  + Particles │  │              │  │  Quality     │       │
│  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└──────────┼─────────────────┼──────────────────┼──────────────────┼──────────────┘
           │                 │                  │                  │
           └─────────────────┴──────────────────┴──────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING PIPELINE                                    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Stage 1: Audio Preprocessing                                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │Resample │→│Normalize│→│  Trim   │→│ Filter  │→│  Save   │      │  │
│  │  │22.05kHz │  │ Volume  │  │Silence  │  │ Noise   │  │  WAV    │      │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Stage 2: Feature Extraction                                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
│  │  │    Mel     │  │   Pitch    │  │   Energy   │  │  Duration  │        │  │
│  │  │Spectrogram │  │   (F0)     │  │   (RMS)    │  │  (Frames)  │        │  │
│  │  │  80 bins   │  │  PyWorld   │  │  Librosa   │  │    MFA     │        │  │
│  │  │[T x 80]    │  │   [T]      │  │   [T]      │  │   [P]      │        │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Stage 3: Text Processing                                                │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │Language │→│   G2P   │→│ Phoneme │→│Language │→│Particle │      │  │
│  │  │Detect   │  │Convert  │  │  IDs    │  │  Tags   │  │  Tags   │      │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Stage 4: Dataset Creation                                               │  │
│  │  • Combine all features into training samples                            │  │
│  │  • Create train/val/test splits (80/10/10)                               │  │
│  │  • Generate metadata files                                               │  │
│  │  • Compute statistics (mean/std for normalization)                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SINGLE GPU FINE-TUNING NODE                                 │
│                            (Unsloth Optimized)                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        Training Hardware                                  │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │  │
│  │  │                       SINGLE GPU NODE                             │   │  │
│  │  │                                                                   │   │  │
│  │  │  GPU Options (pick one):                                          │   │  │
│  │  │  • NVIDIA RTX 4090 (24GB) - Best price/performance               │   │  │
│  │  │  • NVIDIA A100 (40GB) - Enterprise option                        │   │  │
│  │  │  • NVIDIA V100 (32GB) - Cloud GPU                                │   │  │
│  │  │  • NVIDIA L4 (24GB) - Budget cloud option                        │   │  │
│  │  │                                                                   │   │  │
│  │  │  System:                                                          │   │  │
│  │  │  • 32-64GB RAM (system)                                           │   │  │
│  │  │  • 200GB SSD (dataset + checkpoints)                             │   │  │
│  │  │  • 8+ CPU cores (data loading)                                   │   │  │
│  │  │  • Ubuntu 22.04 LTS                                              │   │  │
│  │  │                                                                   │   │  │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │   │  │
│  │  │  │            Unsloth Fine-tuning Process                      │ │   │  │
│  │  │  │                                                             │ │   │  │
│  │  │  │  • Load Sesame CSM-1B (4-bit quantized)                    │ │   │  │
│  │  │  │  • Apply LoRA adapters (r=16, only 32MB trainable!)        │ │   │  │
│  │  │  │  • Gradient checkpointing (70% memory savings)             │ │   │  │
│  │  │  │  • Flash Attention 2 (4x faster)                           │ │   │  │
│  │  │  │  • 8-bit AdamW optimizer                                   │ │   │  │
│  │  │  │  • Mixed precision (bfloat16)                              │ │   │  │
│  │  │  │                                                             │ │   │  │
│  │  │  │  Memory Usage:                                             │ │   │  │
│  │  │  │  • Base model (4-bit): 550MB                               │ │   │  │
│  │  │  │  • LoRA params: 32MB                                       │ │   │  │
│  │  │  │  • Gradients: 64MB                                         │ │   │  │
│  │  │  │  • Optimizer states: 128MB                                 │ │   │  │
│  │  │  │  • Activations: ~1-2GB                                     │ │   │  │
│  │  │  │  • Total: ~2.5-3GB (fits in any modern GPU!)              │ │   │  │
│  │  │  └─────────────────────────────────────────────────────────────┘ │   │  │
│  │  └───────────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                           │
│                                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                FINE-TUNING LOOP (LoRA/QLoRA)                              │  │
│  │                                                                           │  │
│  │  Step 1: Load Batch (Batch size: 4-8, smaller than full training)       │  │
│  │    ├─ Text [B, T_text]: "Saya nak go to mall lah"                       │  │
│  │    ├─ Language tags [B, T_text]: [ms, ms, en, en, en, particle]         │  │
│  │    ├─ Particle markers [B, T_text]: [0, 0, 0, 0, 0, 1]                  │  │
│  │    └─ Target Audio features [B, T_mel, 80]                              │  │
│  │                                                                           │  │
│  │  Step 2: Forward Pass (through LoRA adapters only)                       │  │
│  │    ├─ Base model layers: FROZEN (no gradients)                          │  │
│  │    ├─ LoRA adapters: TRAINABLE (only 1-5% of parameters)                │  │
│  │    ├─ Input → Sesame encoder (with LoRA) → Hidden states                │  │
│  │    └─ Hidden → Sesame decoder (with LoRA) → Mel-spectrogram             │  │
│  │                                                                           │  │
│  │  Step 3: Compute Loss (Malaysian-specific)                               │  │
│  │    ├─ Reconstruction Loss (L1 + MSE on mel)                             │  │
│  │    ├─ Particle-weighted Loss (2x weight on particles)                   │  │
│  │    ├─ Code-switching smoothness penalty                                 │  │
│  │    └─ Accent consistency loss                                           │  │
│  │                                                                           │  │
│  │  Step 4: Backward Pass (only through LoRA adapters)                      │  │
│  │    ├─ Compute gradients (only for 32MB LoRA params)                     │  │
│  │    ├─ Gradient clipping (max_norm=0.3 for stability)                    │  │
│  │    ├─ 8-bit AdamW step (memory efficient)                               │  │
│  │    └─ Cosine LR schedule with warmup                                    │  │
│  │                                                                           │  │
│  │  Step 5: Logging & Validation                                            │  │
│  │    ├─ Log metrics every 50 steps (more frequent for fast iteration)     │  │
│  │    ├─ Generate samples every 500 steps                                  │  │
│  │    ├─ Run validation every 500 steps                                    │  │
│  │    └─ Save LoRA adapters every 1000 steps (only 32MB!)                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING & LOGGING                                     │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐             │
│  │  Weights & Biases│  │   TensorBoard    │  │   MLflow         │             │
│  │                  │  │                  │  │                  │             │
│  │ • Loss curves    │  │ • Scalar metrics │  │ • Experiments    │             │
│  │ • Audio samples  │  │ • Audio playback │  │ • Artifacts      │             │
│  │ • Metrics table  │  │ • Histograms     │  │ • Model registry │             │
│  │ • System monitor │  │ • Graphs         │  │ • Comparisons    │             │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘             │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        Logged Metrics                                     │  │
│  │  • Training loss (total, mel, duration, pitch, energy)                   │  │
│  │  • Validation loss                                                        │  │
│  │  • Learning rate                                                          │  │
│  │  • Gradient norm                                                          │  │
│  │  • GPU utilization & memory                                               │  │
│  │  • Samples per second (throughput)                                       │  │
│  │  • Audio samples (every 5k steps)                                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STORAGE & CHECKPOINTING                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         S3 / Cloud Storage                                │  │
│  │                                                                           │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │  Checkpoints    │  │  Training Logs  │  │  Audio Samples  │         │  │
│  │  │                 │  │                 │  │                 │         │  │
│  │  │ • model.pt      │  │ • train.log     │  │ • step_1k.wav   │         │  │
│  │  │ • optimizer.pt  │  │ • metrics.csv   │  │ • step_5k.wav   │         │  │
│  │  │ • scheduler.pt  │  │ • tensorboard/  │  │ • step_10k.wav  │         │  │
│  │  │ • config.yaml   │  │ • wandb/        │  │ • ...           │         │  │
│  │  │                 │  │                 │  │                 │         │  │
│  │  │ Every 10k steps │  │ Continuous      │  │ Every 5k steps  │         │  │
│  │  │ Keep best 5     │  │                 │  │ Keep last 20    │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                     │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    Automated Evaluation (Periodic)                        │  │
│  │                                                                           │  │
│  │  Every 10k steps:                                                        │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │  Objective      │  │   Listening     │  │   Selection     │         │  │
│  │  │  Metrics        │  │   Check         │  │   Criteria      │         │  │
│  │  │                 │  │                 │  │                 │         │  │
│  │  │ • Mel Loss      │  │ • Generate 10   │  │ • Best val loss │         │  │
│  │  │ • Duration MAE  │  │   samples       │  │ • Best MCD      │         │  │
│  │  │ • F0 RMSE       │  │ • Save to S3    │  │ • Convergence   │         │  │
│  │  │ • MCD           │  │ • Manual review │  │ • No divergence │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                    Manual Evaluation (Milestones)                         │  │
│  │                                                                           │  │
│  │  At 50k, 100k, 150k, 200k, 250k steps:                                  │  │
│  │  • Generate comprehensive test set                                       │  │
│  │  • Calculate MOS (internal team)                                         │  │
│  │  • Test code-switching quality                                           │  │
│  │  • Test particle pronunciation                                           │  │
│  │  • Compare with baseline/previous checkpoints                            │  │
│  │  • Decide: continue, adjust, or stop                                     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FINAL MODEL                                           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        Best Checkpoint                                    │  │
│  │                                                                           │  │
│  │  • Acoustic Model: fastspeech2_malaysian_300k.pt (~180MB)               │  │
│  │  • Vocoder: hifigan_malaysian.pt (~60MB)                                │  │
│  │  • Config: model_config.yaml                                             │  │
│  │  • Phoneme Vocab: phoneme_vocab.json                                     │  │
│  │  • Stats: mean_std_stats.npz                                             │  │
│  │                                                                           │  │
│  │  Ready for:                                                               │  │
│  │  ✓ Production deployment                                                 │  │
│  │  ✓ Further fine-tuning                                                   │  │
│  │  ✓ Model optimization (quantization, ONNX, TensorRT)                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘

Legend:
  [B] = Batch size (typically 16-32)
  [T_text] = Text sequence length (phonemes)
  [T_mel] = Mel-spectrogram time steps
  [H] = Hidden dimension (256-512)
  [P] = Number of phonemes in utterance
```

### 1.4 Understanding the Preprocessing Pipeline (Detailed)

This section explains **WHY** and **HOW** we process raw data before training.

---

#### 1.4.1 The Big Picture: What Are We Trying To Do?

**Training Goal:** Teach the model to predict "How should this text sound?"

```
INPUT:  Text = "Saya nak go to the mall lah"
OUTPUT: Audio waveform of someone saying this naturally
```

**The Challenge:** We can't directly teach a model "text → audio" because:
1. ❌ Audio waveforms are too complex (22,050 numbers per second!)
2. ❌ Model doesn't know what "sounds" correspond to letters
3. ❌ Model doesn't know language boundaries (Malay vs English)
4. ❌ Model doesn't know prosody (how to pronounce "lah" with right intonation)

**The Solution:** Break it into smaller, learnable steps

---

#### 1.4.2 Stage 1: Audio Preprocessing (Cleaning the Audio)

**What it does:** Prepare raw audio recordings for analysis

**Input Example:**
```
raw_recording.wav
- Recorded on different devices (phone, mic, laptop)
- Different volumes (some loud, some quiet)
- Different sample rates (44.1kHz, 48kHz, 16kHz)
- Has silence at start/end
- Background noise (fan, traffic, hum)
```

**Processing Steps:**

**1. Resample to 22.05kHz**
```python
# Why 22.05kHz?
# - Human speech is 80-8000 Hz
# - Nyquist theorem: need 2x highest frequency
# - 22.05kHz captures up to 11kHz (more than enough)
# - Lower than 44.1kHz = smaller files, faster training

import librosa
audio, sr = librosa.load("raw.wav", sr=22050)
# Input:  1.5 million samples (44.1kHz for 3 seconds)
# Output: 66,150 samples (22.05kHz for 3 seconds)
```

**2. Normalize Volume**
```python
# Problem: Some recordings loud (-3dB), some quiet (-20dB)
# Solution: Normalize to consistent level

from scipy.io import wavfile
audio = audio / np.max(np.abs(audio))  # Peak normalization
# OR
rms = np.sqrt(np.mean(audio**2))
audio = audio * (0.1 / rms)  # RMS normalization to -20dB
```

**3. Trim Silence**
```python
# Remove silence at start/end (saves training time)
audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
# Before: 3.0 seconds (0.5s silence + 2.0s speech + 0.5s silence)
# After:  2.0 seconds (pure speech)
```

**4. Filter Noise**
```python
# Remove background noise (optional but helpful)
import noisereduce as nr
audio_clean = nr.reduce_noise(y=audio, sr=22050)
```

**Result:** Clean, consistent audio ready for feature extraction

---

#### 1.4.3 Stage 2: Feature Extraction (Converting Audio to Numbers the Model Can Learn)

**Why we need this:**
- Raw audio: 22,050 numbers per second = Too much!
- Model can't learn from raw waveforms easily
- We need **meaningful representations** of sound

**What we extract:** 4 key features

---

##### **Feature 1: Mel-Spectrogram** (The Most Important!)

**What it is:** A visual representation of "what frequencies are present over time"

**How it works:**

```python
import librosa
import numpy as np

# Step 1: Divide audio into small chunks (frames)
audio = [... 66,150 samples ...]  # 3 seconds at 22.05kHz

# Use 1024 samples per frame, hop 256 samples
# Frame size = 1024/22050 = 46ms (human speech changes every 40-50ms)
# Hop size = 256/22050 = 11.6ms (smooth transitions)

# Step 2: Apply FFT (Fast Fourier Transform) to each frame
# Converts time domain → frequency domain
stft = librosa.stft(audio, n_fft=1024, hop_length=256)
# Output shape: [513 frequency bins, 258 time frames]

# Step 3: Convert to magnitude spectrogram
magnitude = np.abs(stft)
# Shows "how much energy" at each frequency

# Step 4: Apply Mel scale (human hearing is logarithmic)
mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
mel_spec = mel_basis @ magnitude
# Output shape: [80 mel bins, 258 time frames]

# Step 5: Convert to log scale (decibels)
log_mel = librosa.power_to_db(mel_spec)
# Final shape: [80, 258] or written as [T x 80]
```

**Visual Example:**

```
Mel-Spectrogram for "Saya nak go to the mall lah"

Frequency
(Mel bins)
    80 ┤ ░░▓▓▓░░░▓▓░░░▓▓▓▓░░▓▓░░▓░   ← High frequencies (s, sh, f sounds)
    60 ┤ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ← Mid frequencies (vowels)
    40 ┤ ████████████████████████    ← Low-mid (voice fundamental)
    20 ┤ ████████████████████████    ← Low frequencies (bass, pitch)
     0 ┤─────────────────────────
       Say-ya nak  go  to  mall lah
           Time frames (258 frames) →

Legend: 
░ = Low energy (quiet)
▓ = Medium energy
█ = High energy (loud)
```

**Why Mel-Spectrogram?**
- ✅ Captures "what sounds are present" (vowels, consonants)
- ✅ Captures "when sounds occur" (timing)
- ✅ Much smaller than raw audio (80 numbers per frame vs 256)
- ✅ Model can learn patterns easily

---

##### **Feature 2: Pitch (F0 - Fundamental Frequency)**

**What it is:** How high or low the voice sounds (Hz)

**Why we need it:**
- Pitch changes meaning (question vs statement)
- Pitch adds emotion (excitement vs bored)
- Malaysian particles need specific pitch patterns!

**How we extract it:**

```python
import pyworld as pw

# PyWorld is specialized for pitch extraction
f0, timeaxis = pw.dio(audio, sr=22050)
# Returns pitch for each frame

# Example output for "Saya nak go to mall lah"
f0 = [
    120, 125, 130, 135,  # "Sa-ya" (rising slightly)
    140, 140, 140,       # "nak" (steady)
    155, 160, 165,       # "go" (rising)
    150, 145, 140,       # "to the" (falling)
    160, 165, 170,       # "mall" (rising)
    180, 190, 200, 150   # "lah" (rise then fall - particle pattern!)
]
# Shape: [258] - one pitch value per frame
```

**Visual Example:**

```
Pitch Contour (F0):

Hz
200 ┤                          ╱╲     ← "lah" characteristic rise-fall
180 ┤                        ╱╯  ╲    
160 ┤          ╱╲          ╱      ╲
140 ┤  ╱─────╯  ╲────────╯        ╲_
120 ┤─╯
    └───────────────────────────────
     Saya nak go to the mall lah
```

**Important:** Pitch is often in **log scale** for training:
```python
log_f0 = np.log(f0 + 1e-8)  # Log makes learning easier
```

---

##### **Feature 3: Energy (RMS - Root Mean Square)**

**What it is:** How loud each part of the speech is

**Why we need it:**
- Emphasizes important words ("I REALLY want this")
- Shows speech rhythm and stress patterns
- Helps model learn natural volume variations

**How we extract it:**

```python
import librosa

# Calculate RMS energy for each frame
energy = librosa.feature.rms(
    y=audio, 
    frame_length=1024, 
    hop_length=256
)[0]

# Example for "Saya nak go to the mall lah"
energy = [
    0.15, 0.18, 0.20,    # "Sa-ya" (moderate)
    0.25, 0.25,          # "nak" (emphasized!)
    0.22, 0.20,          # "go to"
    0.18, 0.17,          # "the mall"
    0.30, 0.32, 0.28     # "lah" (strong ending!)
]
# Shape: [258] - one energy value per frame
```

**Visual Example:**

```
Energy (Volume):

Energy
0.30 ┤        ▄▄            ▄▄▄    ← Emphasized words
0.20 ┤    ▄▄▄▀▀ ▀▄▄▄▄▄    ▄▀▀▀
0.10 ┤ ▄▄▀            ▀▀▀▀
0.00 ┤─────────────────────────
     Saya nak go to the mall lah
```

---

##### **Feature 4: Duration (Phoneme Alignment)**

**What it is:** How long each sound lasts (in frames)

**Why we need it:**
- Teaches model speech rhythm
- "Hello" is 2 sounds: "he" + "lo" (how long is each?)
- Particles like "lah" are stretched differently than normal words

**How we get it: Montreal Forced Aligner (MFA)**

```bash
# MFA aligns text to audio automatically
mfa align audio.wav transcript.txt dictionary.txt output/

# Input:
# Audio: audio.wav (3 seconds)
# Text: "Saya nak go to the mall lah"

# Output: TextGrid file with timings
# Phoneme | Start | End  | Duration
# --------|-------|------|----------
# s       | 0.00  | 0.08 | 8 frames
# a       | 0.08  | 0.18 | 10 frames
# j       | 0.18  | 0.24 | 6 frames
# a       | 0.24  | 0.36 | 12 frames
# n       | 0.36  | 0.44 | 8 frames
# ...     | ...   | ...  | ...
# l       | 2.80  | 2.88 | 8 frames
# a       | 2.88  | 3.00 | 12 frames (longer for particle!)
```

**Visual Example:**

```
Duration Alignment:

Phoneme: s  a  j  a  | n  a  k | g  o  | ...  | l  a  h
Frames:  8  10 6  12 | 8  10 8 | 10 12 | ...  | 8  12 15
         └─ "Sa" ─┘  └ "nak"┘  └ "go"┘       └─ "lah"──┘
```

---

#### 1.4.4 Stage 3: Text Processing (Teaching the Model What to Say)

**Why we need this:**
- Model doesn't understand "letters" - needs phonemes (sounds)
- Model needs to know which language each word is
- Model needs special handling for particles

---

##### **Step 1: Language Detection**

**What it does:** Identify which language each word belongs to

```python
# Input text
text = "Saya nak go to the mall lah"

# Language detection
words = ["Saya", "nak", "go", "to", "the", "mall", "lah"]
languages = ["ms", "ms", "en", "en", "en", "en", "particle"]

# With probabilities
detection = [
    {"word": "Saya", "lang": "ms", "confidence": 0.99},
    {"word": "nak", "lang": "ms", "confidence": 0.95},
    {"word": "go", "lang": "en", "confidence": 0.97},
    {"word": "to", "lang": "en", "confidence": 0.99},
    {"word": "the", "lang": "en", "confidence": 0.99},
    {"word": "mall", "lang": "en", "confidence": 0.98},
    {"word": "lah", "lang": "particle", "confidence": 0.92},
]
```

**Why important:**
- Different languages use different phoneme sets
- Model needs to transition smoothly between languages
- Particles need special prosody treatment

---

##### **Step 2: G2P (Grapheme-to-Phoneme) Conversion**

**What it does:** Convert letters → sounds (phonemes)

**English Example:**
```python
# Word: "the"
# Graphemes (letters): t-h-e
# Phonemes (sounds):   ð-ə  (IPA notation)

# Word: "mall"
# Graphemes: m-a-l-l
# Phonemes:  m-ɔ-l    (IPA)

from g2p_en import G2p
g2p = G2p()

english_words = ["go", "to", "the", "mall"]
phonemes = {
    "go": ["g", "oʊ"],
    "to": ["t", "u"],
    "the": ["ð", "ə"],
    "mall": ["m", "ɔ", "l"],
}
```

**Malay Example:**
```python
# Malay is mostly phonetic (written = pronounced)

# Word: "Saya"
# Graphemes: S-a-y-a
# Phonemes:  s-a-j-a

# Word: "nak"
# Graphemes: n-a-k
# Phonemes:  n-a-k

malay_words = ["Saya", "nak"]
phonemes = {
    "Saya": ["s", "a", "j", "a"],
    "nak": ["n", "a", "k"],
}
```

**Chinese/Pinyin Example:**
```python
# Pinyin (without tone marks)
# Word: "qu" (去 - go)
# Phonemes: "q", "u" (ch-like + u sound)

# For Malaysian use: simplified phonemes
pinyin_words = ["ni", "hao"]  # 你好
phonemes = {
    "ni": ["n", "i"],
    "hao": ["h", "a", "o"],
}
```

**Complete Sentence:**
```python
# "Saya nak go to the mall lah"
full_phoneme_sequence = [
    # Saya (ms)
    "s", "a", "j", "a",
    # nak (ms)
    "n", "a", "k",
    # go (en)
    "g", "oʊ",
    # to (en)
    "t", "u",
    # the (en)
    "ð", "ə",
    # mall (en)
    "m", "ɔ", "l",
    # lah (particle)
    "l", "a", "h"
]
# Total: 18 phonemes
```

---

##### **Step 3: Phoneme to ID Conversion**

**What it does:** Convert phonemes to numbers (models work with numbers)

```python
# Create vocabulary
phoneme_vocab = {
    "<pad>": 0,    # Padding token
    "<unk>": 1,    # Unknown
    "a": 2,
    "b": 3,
    "d": 4,
    "ð": 5,        # "th" in "the"
    "ə": 6,        # schwa sound
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,       # "y" sound
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "ɔ": 16,       # "aw" sound
    "oʊ": 17,      # "oh" sound
    "s": 18,
    "t": 19,
    "u": 20,
    # ... more phonemes ~100-200 total
}

# Convert phonemes to IDs
phoneme_sequence = ["s", "a", "j", "a", "n", "a", "k", ...]
phoneme_ids = [18, 2, 10, 2, 14, 2, 11, ...]
# Shape: [18] - one ID per phoneme
```

---

##### **Step 4: Language Tags**

**What it does:** Mark which language each phoneme belongs to

```python
# Language vocabulary
lang_vocab = {
    "malay": 0,
    "english": 1,
    "chinese": 2,
    "particle": 3,
}

# "Saya nak go to the mall lah"
# Phonemes: s a j a | n a k | g oʊ | t u | ð ə | m ɔ l | l a h
language_tags = [
    0, 0, 0, 0,     # "Saya" - malay
    0, 0, 0,        # "nak" - malay
    1, 1,           # "go" - english
    1, 1,           # "to" - english
    1, 1,           # "the" - english
    1, 1, 1,        # "mall" - english
    3, 3, 3,        # "lah" - particle
]
# Shape: [18] - one tag per phoneme
```

---

##### **Step 5: Particle Tags**

**What it does:** Mark particle types for special prosody

```python
# Particle vocabulary
particle_vocab = {
    "none": 0,
    "lah": 1,      # Emphasis/completion
    "leh": 2,      # Suggestion
    "loh": 3,      # Obviousness
    "meh": 4,      # Question/doubt
    "lor": 5,      # Resignation
    "mah": 6,      # Stating the obvious
    # ... ~15 particles total
}

# "Saya nak go to the mall lah"
particle_tags = [
    0, 0, 0, 0,     # "Saya"
    0, 0, 0,        # "nak"
    0, 0,           # "go"
    0, 0,           # "to"
    0, 0,           # "the"
    0, 0, 0,        # "mall"
    1, 1, 1,        # "lah" - type 1
]
# Shape: [18] - one tag per phoneme
```

---

#### 1.4.5 Stage 4: Putting It All Together (Dataset Creation)

**Final Training Sample:**

```python
training_sample = {
    # Text information
    "phoneme_ids": [18, 2, 10, 2, 14, 2, 11, ...],     # [18] phonemes
    "language_tags": [0, 0, 0, 0, 0, 0, 0, ...],       # [18] lang tags
    "particle_tags": [0, 0, 0, 0, 0, 0, 0, ...],       # [18] particle tags
    
    # Audio features (targets to predict)
    "mel_spectrogram": [...],                           # [258, 80] mel
    "pitch": [...],                                     # [258] F0
    "energy": [...],                                    # [258] RMS
    "durations": [8, 10, 6, 12, 8, 10, 8, ...],        # [18] frames per phoneme
    
    # Metadata
    "text": "Saya nak go to the mall lah",
    "audio_path": "data/sample_001.wav",
    "speaker_id": 0,
    "duration_seconds": 3.0,
}
```

**What the model learns:**

```
INPUT:  Phoneme IDs + Language Tags + Particle Tags
        [18, 2, 10, 2, ...] + [0, 0, 0, 0, ...] + [0, 0, 0, 0, ...]
        
OUTPUT: Mel-spectrogram [258, 80] + Pitch [258] + Energy [258]
        (How it should sound!)
```

---

#### 1.4.6 Summary: The Complete Flow

```
┌──────────────────────────────────────────────────────────────┐
│ RAW DATA                                                     │
│ • Audio: sample.wav (3 seconds, messy)                       │
│ • Text: "Saya nak go to the mall lah"                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │ AUDIO PREPROCESSING │
          │ Clean & normalize   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────────┐
          │ FEATURE EXTRACTION      │
          │ • Mel: [258, 80]        │
          │ • Pitch: [258]          │
          │ • Energy: [258]         │
          │ • Duration: [18]        │
          └──────────┬──────────────┘
                     │
          ┌──────────▼──────────────┐
          │ TEXT PROCESSING         │
          │ • Phonemes: [18] IDs    │
          │ • Languages: [18] tags  │
          │ • Particles: [18] tags  │
          └──────────┬──────────────┘
                     │
          ┌──────────▼──────────────┐
          │ TRAINING SAMPLE         │
          │ Input: Phonemes+Tags    │
          │ Target: Mel+Pitch+Energy│
          └──────────┬──────────────┘
                     │
          ┌──────────▼──────────────┐
          │ MODEL TRAINING          │
          │ Learn: Text → Sound     │
          └─────────────────────────┘
```

**Key Takeaways:**

1. **Feature Extraction** = Converting audio into numbers the model can learn from
   - Mel-spectrogram (what frequencies)
   - Pitch (how high/low)
   - Energy (how loud)
   - Duration (how long)

2. **Text Processing** = Converting text into numbers the model can understand
   - Phonemes (sounds, not letters)
   - Language tags (Malay/English/Chinese)
   - Particle tags (special handling)

3. **Why we need both:**
   - Model learns: "Given these phonemes → produce this sound"
   - Can't learn from raw audio (too complex)
   - Can't learn from letters (ambiguous pronunciation)

---

### 1.5 Training Data Flow

```
Raw Audio (10-30 hours)
    │
    ├─→ Audio Preprocessing → Clean WAV files
    │
    ├─→ Feature Extraction → Mel/Pitch/Energy
    │
    ├─→ Text Processing → Phonemes/Tags
    │
    ├─→ Forced Alignment → Durations
    │
    └─→ Dataset Builder
            │
            ├─→ Train Set (80%): ~8-24 hours → Batch sampling → GPU Training
            │
            ├─→ Val Set (10%): ~1-3 hours → Validation loop → Metrics
            │
            └─→ Test Set (10%): ~1-3 hours → Final evaluation → MOS/WER
```

### 1.5 Fine-tuning Infrastructure Requirements

**For Fine-tuning with Unsloth (Much Simpler!):**

| Component | Specification | Purpose | Cost |
|-----------|--------------|---------|------|
| **GPU** | 1× NVIDIA RTX 4090 (24GB) | Fine-tuning | $1,600 |
| **Storage** | 200GB SSD | Dataset + checkpoints | $30 |
| **RAM** | 32-64GB DDR4 | Data loading | $100-200 |
| **CPU** | 8+ cores (Ryzen/Intel) | Preprocessing | $300 |
| **Motherboard** | Standard ATX | System | $200 |
| **PSU** | 850W 80+ Gold | Power | $150 |
| **Case** | Standard tower | Housing | $100 |
| **Monitor** | Any display | Setup/monitoring | $200 |

**Total Hardware Cost: ~$2,680 (vs $60,000+ for multi-GPU cluster)**

**Cloud Alternatives:**

| Provider | Instance | GPU | Price/hour | Total Cost (100 hours) |
|----------|----------|-----|------------|----------------------|
| **AWS** | g5.2xlarge | A10G (24GB) | $1.20 | $120 |
| **GCP** | n1-standard-8 + T4 | T4 (16GB) | $0.95 | $95 |
| **Lambda Labs** | gpu_1x_a6000 | A6000 (48GB) | $0.80 | $80 |
| **RunPod** | RTX 4090 | 4090 (24GB) | $0.69 | $69 |
| **Vast.ai** | RTX 4090 | 4090 (24GB) | $0.40-0.60 | $40-60 |

**Recommended Setup:**
- **For Quick Start**: Use RunPod/Vast.ai (~$50-100 total)
- **For Multiple Experiments**: Buy RTX 4090 (~$2,700 one-time)
- **For Enterprise**: Use AWS/GCP with spot instances (~$100-200 total)

**Comparison: Fine-tuning vs Training from Scratch**

| Metric | Fine-tuning (Unsloth) | Training from Scratch |
|--------|----------------------|---------------------|
| **GPUs Needed** | 1× RTX 4090 | 4× V100 or A100 |
| **Training Time** | 3-7 days | 8-12 weeks |
| **GPU Hours** | 72-168 hours | 5,000+ hours |
| **Data Required** | 10-30 hours | 100+ hours |
| **Total Cost (cloud)** | $50-200 | $35,000+ |
| **Total Cost (hardware)** | $2,700 | $60,000+ |
| **Memory per GPU** | ~3GB | ~28GB |
| **Expertise Required** | Medium | High |
| **Time to First Result** | 12-24 hours | 2-3 weeks |

**💡 Key Insight**: Fine-tuning with Unsloth is 100× cheaper and 10× faster!

---

## 2. Environment Setup

### 2.1 Hardware Requirements

#### Minimum Requirements
- **GPU**: 1x NVIDIA RTX 3090 (24GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 500GB SSD
- **CPU**: 8+ cores

#### Recommended Setup
- **GPU**: 4x NVIDIA RTX 3090 or 2x A6000 (48GB each)
- **RAM**: 64GB+ system RAM
- **Storage**: 1TB NVMe SSD
- **CPU**: 16+ cores
- **Network**: 10 Gbps (for distributed training)

#### Cloud Alternatives
- **AWS**: p3.8xlarge (4x V100) or p4d.24xlarge (8x A100)
- **GCP**: a2-highgpu-4g (4x A100)
- **Azure**: NC24ads A100 v4

### 2.2 Software Environment

```bash
# Create conda environment
conda create -n malaysian-tts python=3.10
conda activate malaysian-tts

# Install PyTorch with CUDA
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TTS dependencies
pip install -r requirements.txt

# requirements.txt content:
"""
torch==2.0.1
torchaudio==2.0.1
numpy==1.24.3
scipy==1.10.1
librosa==0.10.0
matplotlib==3.7.1
tensorboard==2.13.0
wandb==0.15.4
pydantic==2.0.2
pyyaml==6.0
tqdm==4.65.0
pyloudnorm==0.1.1
pyworld==0.3.2
g2p-en==2.1.0
montreal-forced-aligner==2.2.15
phonemizer==3.2.1
jiwer==3.0.1
pesq==0.0.4
pystoi==0.3.3
einops==0.6.1
"""
```

### 2.3 Directory Structure

```
project/
├── configs/
│   ├── base_config.yaml
│   ├── pretraining_config.yaml
│   ├── training_config.yaml
│   └── finetuning_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── fastspeech2/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── modules.py
│   │   └── loss.py
│   ├── hifigan/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── discriminator.py
│   └── text_processor.py
├── preprocessing/
│   ├── audio_preprocessing.py
│   ├── feature_extraction.py
│   └── build_dataset.py
├── training/
│   ├── train_acoustic.py
│   ├── train_vocoder.py
│   ├── trainer.py
│   └── lr_scheduler.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── generate_samples.py
├── checkpoints/
│   ├── acoustic/
│   └── vocoder/
├── logs/
│   ├── tensorboard/
│   └── wandb/
└── scripts/
    ├── prepare_data.sh
    ├── train.sh
    └── evaluate.sh
```

---

## 3. Data Preparation for Training

### 3.1 Dataset Builder

```python
# preprocessing/build_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path

class MalaysianTTSDataset(Dataset):
    """
    Dataset for Malaysian TTS training
    """
    def __init__(self, metadata_path, config):
        self.config = config
        self.metadata = self.load_metadata(metadata_path)
        
        # Load phoneme vocabulary
        self.phoneme_vocab = self.load_vocab(config.phoneme_vocab_path)
        self.lang_vocab = {'ms': 0, 'en': 1, 'zh': 2, 'particle': 3}
        self.particle_vocab = self.load_vocab(config.particle_vocab_path)
        
        # Filters
        self.min_duration = config.min_duration  # 1.0s
        self.max_duration = config.max_duration  # 20.0s
        
        self.filtered_data = self.filter_data()
        
    def load_metadata(self, path):
        """Load metadata file"""
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def filter_data(self):
        """Filter data by duration and quality"""
        filtered = []
        for key, item in self.metadata.items():
            if self.min_duration <= item['duration'] <= self.max_duration:
                filtered.append((key, item))
        return filtered
    
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            phoneme_ids: [seq_len]
            language_ids: [seq_len]
            particle_types: [seq_len]
            mel: [time, n_mels]
            pitch: [time]
            energy: [time]
            durations: [seq_len]
            speaker_id: int
        """
        key, item = self.filtered_data[idx]
        
        # Load features
        mel = np.load(f"{self.config.mel_dir}/{key}.npy")
        pitch = np.load(f"{self.config.pitch_dir}/{key}.npy")
        energy = np.load(f"{self.config.energy_dir}/{key}.npy")
        
        # Get phoneme sequence
        phoneme_ids = [
            self.phoneme_vocab.get(p, self.phoneme_vocab['<unk>']) 
            for p in item['phonemes']
        ]
        
        # Get language IDs
        language_ids = item['language_ids']
        
        # Get particle types
        particle_types = [
            self.particle_vocab.get(p, 0) 
            for p in item['particle_types']
        ]
        
        # Get durations from alignment
        durations = np.array(item['phoneme_durations']) / self.config.hop_length
        durations = np.round(durations).astype(np.int32)
        
        # Speaker ID
        speaker_id = item.get('speaker_id', 0)
        
        return {
            'phoneme_ids': torch.LongTensor(phoneme_ids),
            'language_ids': torch.LongTensor(language_ids),
            'particle_types': torch.LongTensor(particle_types),
            'mel': torch.FloatTensor(mel),
            'pitch': torch.FloatTensor(pitch),
            'energy': torch.FloatTensor(energy),
            'durations': torch.LongTensor(durations),
            'speaker_id': torch.LongTensor([speaker_id]),
            'text': item['text']
        }

def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads sequences to same length within batch
    """
    # Find max lengths
    max_phoneme_len = max([x['phoneme_ids'].shape[0] for x in batch])
    max_mel_len = max([x['mel'].shape[0] for x in batch])
    
    batch_size = len(batch)
    n_mels = batch[0]['mel'].shape[1]
    
    # Initialize padded tensors
    phoneme_ids_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    language_ids_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    particle_types_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, max_mel_len, n_mels)
    pitch_padded = torch.zeros(batch_size, max_mel_len)
    energy_padded = torch.zeros(batch_size, max_mel_len)
    durations_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    
    # Create masks
    phoneme_mask = torch.zeros(batch_size, max_phoneme_len, dtype=torch.bool)
    mel_mask = torch.zeros(batch_size, max_mel_len, dtype=torch.bool)
    
    speaker_ids = []
    texts = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        phoneme_len = item['phoneme_ids'].shape[0]
        mel_len = item['mel'].shape[0]
        
        phoneme_ids_padded[i, :phoneme_len] = item['phoneme_ids']
        language_ids_padded[i, :phoneme_len] = item['language_ids']
        particle_types_padded[i, :phoneme_len] = item['particle_types']
        mel_padded[i, :mel_len] = item['mel']
        pitch_padded[i, :mel_len] = item['pitch']
        energy_padded[i, :mel_len] = item['energy']
        durations_padded[i, :phoneme_len] = item['durations']
        
        phoneme_mask[i, :phoneme_len] = True
        mel_mask[i, :mel_len] = True
        
        speaker_ids.append(item['speaker_id'])
        texts.append(item['text'])
    
    return {
        'phoneme_ids': phoneme_ids_padded,
        'language_ids': language_ids_padded,
        'particle_types': particle_types_padded,
        'mel': mel_padded,
        'pitch': pitch_padded,
        'energy': energy_padded,
        'durations': durations_padded,
        'phoneme_mask': phoneme_mask,
        'mel_mask': mel_mask,
        'speaker_ids': torch.cat(speaker_ids),
        'texts': texts
    }
```

---

## 4. Phase 0: Pre-training (Optional)

### 4.1 Pre-training Strategy

**Objective**: Learn general phoneme-to-mel mappings from high-resource languages

**Data Sources**:
- LJSpeech (English, 24 hours, single speaker)
- VCTK (English, 44 hours, 109 speakers)
- AISHELL-3 (Mandarin, 85 hours, 218 speakers)
- LibriTTS (English, 585 hours, multi-speaker)

**Why Pre-train**:
1. Better initialization than random weights
2. Learn robust phoneme representations
3. Faster convergence on target language
4. Better generalization with limited data

### 4.2 Pre-training Configuration

```yaml
# configs/pretraining_config.yaml

# Model
model:
  name: "FastSpeech2"
  encoder_hidden: 256
  encoder_layers: 4
  encoder_heads: 2
  decoder_hidden: 256
  decoder_layers: 4
  decoder_heads: 2
  n_mel_channels: 80
  phoneme_vocab_size: 256
  num_speakers: 327  # Combined from all datasets
  dropout: 0.2

# Training
training:
  batch_size: 32
  gradient_accumulation: 2  # Effective batch size: 64
  max_steps: 200000
  warmup_steps: 4000
  learning_rate: 1e-3
  weight_decay: 1e-6
  grad_clip: 1.0
  
  # Learning rate schedule
  lr_scheduler: "noam"  # Transformer-style
  
  # Validation
  val_interval: 5000
  save_interval: 10000
  log_interval: 100
  
  # Early stopping
  patience: 50000  # steps
  
# Loss weights
loss:
  mel_loss: 1.0
  duration_loss: 1.0
  pitch_loss: 1.0
  energy_loss: 1.0
  
# Data
data:
  sample_rate: 22050
  hop_length: 256
  n_fft: 1024
  win_length: 1024
  mel_fmin: 0
  mel_fmax: 8000
  min_duration: 1.0
  max_duration: 20.0
  
# Hardware
hardware:
  num_gpus: 4
  mixed_precision: true  # FP16
  num_workers: 8
```

### 4.3 Pre-training Script

```python
# training/train_acoustic.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

from models.fastspeech2 import MalaysianFastSpeech2
from preprocessing.build_dataset import MalaysianTTSDataset, collate_fn
from training.lr_scheduler import get_noam_scheduler

def train_pretrain_phase(config):
    """
    Pre-training phase
    """
    # Initialize wandb
    wandb.init(
        project="malaysian-tts-pretraining",
        config=config
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = MalaysianTTSDataset(
        config.train_metadata,
        config
    )
    val_dataset = MalaysianTTSDataset(
        config.val_metadata,
        config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = MalaysianFastSpeech2(config)
    model = model.to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = get_noam_scheduler(
        optimizer,
        config.encoder_hidden,
        config.warmup_steps
    )
    
    # Mixed precision training
    scaler = GradScaler() if config.mixed_precision else None
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1000):  # Effectively infinite
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if config.mixed_precision:
                with autocast():
                    output = model(
                        phonemes=batch['phoneme_ids'],
                        language_ids=batch['language_ids'],
                        particle_types=batch['particle_types'],
                        speaker_ids=batch['speaker_ids'],
                        durations=batch['durations'],
                        pitch=batch['pitch'],
                        energy=batch['energy'],
                        mel_target=batch['mel'],
                        phoneme_mask=batch['phoneme_mask'],
                        mel_mask=batch['mel_mask']
                    )
                    loss = compute_loss(output, batch, config)
            else:
                output = model(...)
                loss = compute_loss(output, batch, config)
            
            # Backward pass
            optimizer.zero_grad()
            
            if config.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % config.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mel_loss': output['mel_loss'].item(),
                    'train/duration_loss': output['duration_loss'].item(),
                    'train/pitch_loss': output['pitch_loss'].item(),
                    'train/energy_loss': output['energy_loss'].item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'global_step': global_step
                })
            
            pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            if global_step % config.val_interval == 0:
                val_loss = validate(model, val_loader, device, config)
                
                wandb.log({
                    'val/loss': val_loss,
                    'global_step': global_step
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, global_step,
                        f"{config.checkpoint_dir}/best_pretrain.pt"
                    )
                    patience_counter = 0
                else:
                    patience_counter += config.val_interval
                
                # Early stopping
                if patience_counter >= config.patience:
                    print("Early stopping triggered")
                    return
            
            # Save checkpoint
            if global_step % config.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step,
                    f"{config.checkpoint_dir}/pretrain_step_{global_step}.pt"
                )
            
            # Max steps reached
            if global_step >= config.max_steps:
                print(f"Max steps ({config.max_steps}) reached")
                return

def compute_loss(output, batch, config):
    """
    Compute multi-component loss
    """
    # Mel loss (L1 + L2)
    mel_loss = nn.L1Loss()(output['mel_pred'], batch['mel'])
    mel_loss += nn.MSELoss()(output['mel_pred'], batch['mel'])
    
    # Duration loss
    duration_loss = nn.MSELoss()(
        output['duration_pred'],
        torch.log(batch['durations'].float() + 1)
    )
    
    # Pitch loss
    pitch_loss = nn.MSELoss()(output['pitch_pred'], batch['pitch'])
    
    # Energy loss
    energy_loss = nn.MSELoss()(output['energy_pred'], batch['energy'])
    
    # Total loss
    total_loss = (
        config.mel_loss_weight * mel_loss +
        config.duration_loss_weight * duration_loss +
        config.pitch_loss_weight * pitch_loss +
        config.energy_loss_weight * energy_loss
    )
    
    # Store individual losses for logging
    output['mel_loss'] = mel_loss
    output['duration_loss'] = duration_loss
    output['pitch_loss'] = pitch_loss
    output['energy_loss'] = energy_loss
    
    return total_loss

def validate(model, val_loader, device, config):
    """
    Validation loop
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            output = model(
                phonemes=batch['phoneme_ids'],
                language_ids=batch['language_ids'],
                particle_types=batch['particle_types'],
                speaker_ids=batch['speaker_ids'],
                durations=batch['durations'],
                pitch=batch['pitch'],
                energy=batch['energy'],
                mel_target=batch['mel']
            )
            
            loss = compute_loss(output, batch, config)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    model.train()
    
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, global_step, path):
    """
    Save training checkpoint
    """
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
```

---

## 5. Phase 1: Main Training

### 5.1 Training Configuration

```yaml
# configs/training_config.yaml

# Model (load from pre-training or initialize)
model:
  checkpoint: "checkpoints/acoustic/best_pretrain.pt"  # or null for random init
  freeze_encoder: false  # Set to true initially, then fine-tune
  
  # Malaysian-specific
  particle_types: 15
  language_types: 4  # ms, en, zh, particle

# Training
training:
  batch_size: 24  # Smaller for longer sequences
  gradient_accumulation: 3  # Effective batch size: 72
  max_steps: 300000
  warmup_steps: 8000
  learning_rate: 5e-4  # Lower than pre-training
  weight_decay: 1e-6
  grad_clip: 1.0
  
  # Curriculum learning
  curriculum:
    enabled: true
    phases:
      - name: "single_language"
        steps: 50000
        data_filter: "single_lang_only"
      - name: "two_language"
        steps: 100000
        data_filter: "max_2_languages"
      - name: "full_code_switching"
        steps: 150000
        data_filter: "all"

# Loss weights (adjusted for Malaysian features)
loss:
  mel_loss: 1.0
  duration_loss: 1.5  # Emphasize duration for particles
  pitch_loss: 2.0  # Emphasize pitch for particles
  energy_loss: 1.5
  
  # Auxiliary losses
  language_classification: 0.1
  particle_classification: 0.2

# Data augmentation
augmentation:
  speed_perturbation: 0.3  # 30% chance
  pitch_shift: 0.2  # 20% chance
  noise_injection: 0.1  # 10% chance
```

### 5.2 Curriculum Learning Implementation

```python
# training/curriculum.py

class CurriculumScheduler:
    """
    Manage curriculum learning phases
    """
    def __init__(self, config):
        self.phases = config.curriculum.phases
        self.current_phase_idx = 0
        self.phase_start_step = 0
    
    def get_current_phase(self, global_step):
        """
        Determine current curriculum phase
        """
        cumulative_steps = 0
        for i, phase in enumerate(self.phases):
            cumulative_steps += phase['steps']
            if global_step < cumulative_steps:
                if i != self.current_phase_idx:
                    self.current_phase_idx = i
                    self.phase_start_step = cumulative_steps - phase['steps']
                    print(f"\n{'='*60}")
                    print(f"Entering Phase {i+1}: {phase['name']}")
                    print(f"Steps: {self.phase_start_step} - {cumulative_steps}")
                    print(f"{'='*60}\n")
                return phase
        
        # Final phase
        return self.phases[-1]
    
    def filter_dataset(self, dataset, phase):
        """
        Filter dataset based on curriculum phase
        """
        data_filter = phase['data_filter']
        
        if data_filter == "single_lang_only":
            return [d for d in dataset if not d[1]['code_switching']]
        elif data_filter == "max_2_languages":
            return [d for d in dataset if d[1]['num_languages'] <= 2]
        else:  # "all"
            return dataset
```

### 5.3 Multi-Task Learning

```python
# models/fastspeech2/multitask.py

class MultiTaskFastSpeech2(nn.Module):
    """
    FastSpeech2 with auxiliary classification tasks
    """
    def __init__(self, config):
        super().__init__()
        
        # Main FastSpeech2 model
        self.fastspeech2 = MalaysianFastSpeech2(config)
        
        # Auxiliary task heads
        self.language_classifier = nn.Sequential(
            nn.Linear(config.encoder_hidden, config.encoder_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.encoder_hidden // 2, 4)  # 4 languages
        )
        
        self.particle_classifier = nn.Sequential(
            nn.Linear(config.encoder_hidden, config.encoder_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.encoder_hidden // 2, config.particle_types)
        )
    
    def forward(self, ...):
        # Main TTS forward
        output = self.fastspeech2(...)
        
        # Auxiliary predictions (from encoder output)
        encoder_output = output['encoder_output']
        
        lang_pred = self.language_classifier(encoder_output)
        particle_pred = self.particle_classifier(encoder_output)
        
        output['lang_pred'] = lang_pred
        output['particle_pred'] = particle_pred
        
        return output

def compute_multitask_loss(output, batch, config):
    """
    Compute loss with auxiliary tasks
    """
    # Main TTS loss
    main_loss = compute_loss(output, batch, config)
    
    # Language classification loss
    lang_loss = nn.CrossEntropyLoss()(
        output['lang_pred'].view(-1, 4),
        batch['language_ids'].view(-1)
    )
    
    # Particle classification loss
    particle_loss = nn.CrossEntropyLoss()(
        output['particle_pred'].view(-1, config.particle_types),
        batch['particle_types'].view(-1)
    )
    
    # Total loss
    total_loss = (
        main_loss +
        config.lang_loss_weight * lang_loss +
        config.particle_loss_weight * particle_loss
    )
    
    output['lang_loss'] = lang_loss
    output['particle_loss'] = particle_loss
    
    return total_loss
```

---

## 6. Phase 2: Fine-tuning

### 6.1 Fine-tuning Strategy

**Objective**: Refine model on highest-quality data and challenging cases

**Data**: 
- Top 20% highest quality recordings
- Particle-rich examples
- Difficult code-switching patterns

**Configuration**:
```yaml
# configs/finetuning_config.yaml

training:
  checkpoint: "checkpoints/acoustic/best_main_training.pt"
  
  # Lower learning rate
  learning_rate: 1e-4
  
  # Smaller batch for stability
  batch_size: 16
  
  # Shorter training
  max_steps: 100000
  
  # More frequent validation
  val_interval: 2000
  
  # Stricter early stopping
  patience: 20000

loss:
  # Emphasize prosody more
  pitch_loss: 3.0  # Increased
  energy_loss: 2.0  # Increased
  
  # Particle-specific loss
  particle_prosody_loss: 1.0  # New
```

### 6.2 Particle-Specific Loss

```python
# models/fastspeech2/loss.py

def particle_prosody_loss(output, batch, config):
    """
    Special loss for particle prosody
    
    Emphasize pitch and duration accuracy for particles
    """
    # Identify particle positions
    particle_mask = batch['particle_types'] > 0  # [B, T]
    
    if not particle_mask.any():
        return torch.tensor(0.0, device=output['pitch_pred'].device)
    
    # Pitch loss for particles only
    particle_pitch_pred = output['pitch_pred'][particle_mask.unsqueeze(-1).expand_as(output['pitch_pred'])]
    particle_pitch_target = batch['pitch'][particle_mask.unsqueeze(-1).expand_as(batch['pitch'])]
    
    pitch_loss = nn.MSELoss()(particle_pitch_pred, particle_pitch_target)
    
    # Duration loss for particles only
    phoneme_particle_mask = batch['particle_types'] > 0  # Phoneme-level
    if phoneme_particle_mask.any():
        particle_dur_pred = output['duration_pred'][phoneme_particle_mask]
        particle_dur_target = torch.log(batch['durations'][phoneme_particle_mask].float() + 1)
        duration_loss = nn.MSELoss()(particle_dur_pred, particle_dur_target)
    else:
        duration_loss = torch.tensor(0.0, device=pitch_loss.device)
    
    return pitch_loss + duration_loss
```

---

## 7. Vocoder Training

### 7.1 HiFi-GAN Training

**Training Strategy**:
1. Train on ground-truth mel-spectrograms first
2. Switch to predicted mels for fine-tuning
3. Use multi-scale/multi-period discriminators

### 7.2 HiFi-GAN Configuration

```yaml
# configs/hifigan_config.yaml

model:
  resblock: "1"
  upsample_rates: [8, 8, 2, 2]
  upsample_kernel_sizes: [16, 16, 4, 4]
  upsample_initial_channel: 512
  resblock_kernel_sizes: [3, 7, 11]
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]

training:
  batch_size: 16
  learning_rate: 2e-4
  adam_b1: 0.8
  adam_b2: 0.99
  lr_decay: 0.999
  segment_size: 8192  # Audio segment length
  max_steps: 2500000  # ~2-3 weeks
  
  # Loss weights
  fm_loss_weight: 2.0  # Feature matching
  mel_loss_weight: 45.0  # Mel reconstruction

discriminator:
  periods: [2, 3, 5, 7, 11]  # Multi-period
  num_scales: 3  # Multi-scale
```

### 7.3 Vocoder Training Script

```python
# training/train_vocoder.py

def train_hifigan(config):
    """
    Train HiFi-GAN vocoder
    """
    # Setup
    device = torch.device('cuda')
    
    # Models
    generator = HiFiGANGenerator(config).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # Optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_b1, config.adam_b2)
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config.learning_rate,
        betas=(config.adam_b1, config.adam_b2)
    )
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.lr_decay
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.lr_decay
    )
    
    # Dataset
    train_dataset = VocoderDataset(config.train_files, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    global_step = 0
    
    for epoch in range(10000):
        for batch in tqdm(train_loader):
            mel = batch['mel'].to(device)  # [B, T, n_mels]
            audio = batch['audio'].to(device)  # [B, T*hop_length]
            
            mel = mel.transpose(1, 2)  # [B, n_mels, T]
            audio = audio.unsqueeze(1)  # [B, 1, T_audio]
            
            # Train discriminator
            optim_d.zero_grad()
            
            # Generate audio
            audio_gen = generator(mel)
            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(audio, audio_gen.detach())
            loss_disc_f, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(audio, audio_gen.detach())
            loss_disc_s, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            
            loss_disc_all = loss_disc_s + loss_disc_f
            
            loss_disc_all.backward()
            optim_d.step()
            
            # Train generator
            optim_g.zero_grad()
            
            # L1 Mel loss
            mel_gen = mel_spectrogram(audio_gen.squeeze(1), config)
            loss_mel = nn.L1Loss()(mel, mel_gen) * config.mel_loss_weight
            
            # MPD
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio, audio_gen)
            # MSD
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(audio, audio_gen)
            
            # Generator adversarial loss
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            
            loss_gen_all = (
                loss_gen_s + loss_gen_f +
                config.fm_loss_weight * (loss_fm_s + loss_fm_f) +
                loss_mel
            )
            
            loss_gen_all.backward()
            optim_g.step()
            
            # Logging
            if global_step % 100 == 0:
                wandb.log({
                    'train/loss_disc': loss_disc_all.item(),
                    'train/loss_gen': loss_gen_all.item(),
                    'train/loss_mel': loss_mel.item(),
                    'global_step': global_step
                })
            
            global_step += 1
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
        
        # Save checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                generator, optim_g,
                f"checkpoints/vocoder/g_epoch_{epoch}.pt"
            )
```

---

## 8. Monitoring & Debugging

### 8.1 Key Metrics to Monitor

```python
# Metrics to track during training

metrics_to_track = {
    # Loss metrics
    'train/loss_total': 'Main training loss',
    'train/loss_mel': 'Mel-spectrogram reconstruction',
    'train/loss_duration': 'Duration prediction',
    'train/loss_pitch': 'Pitch prediction',
    'train/loss_energy': 'Energy prediction',
    'val/loss_total': 'Validation loss',
    
    # Auxiliary metrics
    'train/lang_accuracy': 'Language classification accuracy',
    'train/particle_accuracy': 'Particle classification accuracy',
    
    # Quality metrics (computed periodically)
    'val/mcd': 'Mel-cepstral distortion',
    'val/f0_rmse': 'Pitch RMSE',
    'val/duration_mae': 'Duration MAE',
    
    # System metrics
    'system/gpu_memory': 'GPU memory usage',
    'system/learning_rate': 'Current learning rate',
    'system/gradient_norm': 'Gradient norm (for debugging)',
    'system/samples_per_second': 'Training throughput'
}
```

### 8.2 Debugging Tips

**Common Issues & Solutions:**

1. **Loss explodes (NaN)**:
   - Lower learning rate
   - Increase gradient clipping
   - Check for data issues (extreme values)
   - Reduce batch size

2. **Model not learning (flat loss)**:
   - Check data preprocessing
   - Verify labels match predictions
   - Increase learning rate
   - Check if model too complex for data size

3. **Poor code-switching**:
   - Increase language embedding dimension
   - Add more code-switched examples
   - Use multi-task language classification
   - Check language tags are correct

4. **Particles sound wrong**:
   - Use particle-specific loss
   - Increase pitch/energy loss weight
   - Add more particle examples
   - Check particle annotations

5. **Slow training**:
   - Use mixed precision (FP16)
   - Increase batch size
   - Use more efficient data loading
   - Profile code to find bottlenecks

### 8.3 Tensorboard Visualization

```python
# Add to training loop for visualization

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/tensorboard')

# Log scalars
writer.add_scalar('Loss/train', loss.item(), global_step)

# Log mel-spectrograms (every N steps)
if global_step % 1000 == 0:
    writer.add_image(
        'Mel/predicted',
        plot_mel_spectrogram(output['mel_pred'][0]),
        global_step
    )
    writer.add_image(
        'Mel/target',
        plot_mel_spectrogram(batch['mel'][0]),
        global_step
    )

# Log audio samples (less frequent)
if global_step % 5000 == 0:
    audio_gen = vocoder(output['mel_pred'][0])
    writer.add_audio(
        'Audio/generated',
        audio_gen,
        global_step,
        sample_rate=22050
    )
```

---

## 9. Training Best Practices

### 9.1 General Tips

1. **Start Small**: Train on subset first (100 samples) to verify pipeline
2. **Monitor Overfitting**: Track train/val gap
3. **Save Often**: Checkpoints every 10k steps
4. **Document Everything**: Log hyperparameters, data versions
5. **Validate Frequently**: Generate samples to listen during training
6. **Use Version Control**: Git for code, DVC for data/models

### 9.2 Hyperparameter Tuning

**Priority Order**:
1. Learning rate (most important)
2. Batch size
3. Loss weights
4. Model size (hidden_dim, layers)
5. Dropout rate

**Suggested Ranges**:
```python
hyperparameter_ranges = {
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [16, 24, 32],
    'encoder_hidden': [256, 384, 512],
    'mel_loss_weight': [0.5, 1.0, 2.0],
    'pitch_loss_weight': [1.0, 2.0, 3.0]
}
```

### 9.3 Checkpointing Strategy

```python
# Save multiple types of checkpoints

checkpoint_strategy = {
    'best_val_loss': {
        'metric': 'val_loss',
        'mode': 'min',
        'keep': 3  # Keep top 3
    },
    'periodic': {
        'interval': 10000,  # Every 10k steps
        'keep': 'all'
    },
    'final': {
        'at_end': True,
        'keep': 1
    }
}
```

---

## 10. Expected Results & Timeline

### 10.1 Training Timeline

```
Pre-training (if used):       2-3 weeks
Main Training:                4-6 weeks
Fine-tuning:                  1-2 weeks
Vocoder Training:             2-3 weeks (can be parallel)
-------------------------------------------
Total:                        9-14 weeks
```

### 10.2 Expected Metrics at Each Phase

| Phase | MCD (dB) | F0 RMSE (Hz) | Duration MAE (ms) |
|-------|----------|--------------|-------------------|
| After Pre-training | 7-8 | 30-40 | 60-80 |
| After Main Training | 6-7 | 20-30 | 40-60 |
| After Fine-tuning | 5.5-6.5 | 15-25 | 30-50 |

**MOS Targets**:
- Main Training: 3.5-3.8
- Fine-tuning: 3.8-4.2
- Goal: >4.0

---

## 11. Troubleshooting Guide

See full troubleshooting matrix in Appendix A.

---

**Document Version:** 1.0  
**Last Updated:** October 12, 2025  
**Next Steps:** Begin pre-training or main training based on data availability

