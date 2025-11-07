# Data Preparation Guide
# Malaysian Multilingual ASR System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** Data Engineering Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset Requirements](#2-dataset-requirements)
3. [Data Collection Strategies](#3-data-collection-strategies)
4. [Recording Guidelines](#4-recording-guidelines)
5. [Transcription Guidelines](#5-transcription-guidelines)
6. [Data Annotation Schema](#6-data-annotation-schema)
7. [Quality Control](#7-quality-control)
8. [Data Augmentation](#8-data-augmentation)
9. [Dataset Organization](#9-dataset-organization)
10. [Tools & Automation](#10-tools--automation)
11. [Quick Start Guide](#11-quick-start-guide)

---

## 1. Overview

### 1.1 Purpose

This guide provides comprehensive instructions for collecting, annotating, and preparing Malaysian multilingual speech data for fine-tuning Whisper-large v3.

### 1.2 Key Challenges for Malaysian ASR

1. **Code-Switching**: Speakers mix Malay, English, and Mandarin mid-sentence
2. **Discourse Particles**: "lah", "leh", "loh" are grammatically optional but semantically important
3. **Accent Variation**: Malaysian English has unique phonetic characteristics
4. **Domain Diversity**: Formal vs informal, different age groups, regional accents
5. **Audio Quality**: Real-world recordings (phone calls, podcasts) have varying quality

### 1.3 Data Goals

**Quality Over Quantity:**
- **Minimum**: 10 hours of high-quality, transcribed Malaysian speech
- **Recommended**: 50+ hours for production-grade model
- **Ideal**: 100+ hours with diverse speakers, domains, and accents

**Diversity Requirements:**
- At least 50% code-switching samples (mixed Malay + English)
- Representation of all major discourse particles
- Balanced gender distribution (50/50 male/female)
- Age range: 18-60 years old
- Recording conditions: clean studio, conversational, noisy environments

---

## 2. Dataset Requirements

### 2.1 Audio Requirements

| Specification | Minimum | Recommended | Notes |
|---------------|---------|-------------|-------|
| **Format** | WAV (PCM) | WAV (PCM) / FLAC | Lossless preferred |
| **Sample Rate** | 16 kHz | 48 kHz | Will resample to 16kHz |
| **Bit Depth** | 16-bit | 24-bit | Higher depth = better quality |
| **Channels** | Mono | Mono | Stereo will be converted |
| **Duration** | 1-30 seconds | 3-20 seconds | Optimal for Whisper |
| **SNR (Signal-to-Noise)** | > 15 dB | > 25 dB | Clean speech |
| **File Size** | - | < 50 MB per file | For efficient processing |

**Audio Quality Checklist:**
- ✅ No clipping (audio peaks should be < 0 dBFS)
- ✅ Minimal background noise
- ✅ Clear speech (not muffled)
- ✅ No echo or reverb (unless intentional)
- ✅ Consistent volume levels

### 2.2 Transcription Requirements

**Transcription Quality:**
- **Accuracy**: 99%+ word-level accuracy
- **Formatting**: Consistent spelling, capitalization, punctuation
- **Completeness**: Every word, including particles, must be transcribed
- **Timing**: Optional but recommended for evaluation

**What to Include:**
- Spoken words (including fillers like "um", "uh")
- Code-switches (language changes)
- Discourse particles (lah, leh, loh, etc.)
- Hesitations and false starts (optional, depends on use case)

**What to Exclude:**
- Non-speech sounds (cough, laugh) - use tags like [cough]
- Music or sound effects
- Unintelligible speech - use [inaudible] tag

### 2.3 Dataset Size Guidelines

| Use Case | Training Hours | Speakers | Transcripts | Expected WER |
|----------|----------------|----------|-------------|--------------|
| **Proof of Concept** | 10-20 hours | 10-20 | 2,000-4,000 | 18-22% |
| **MVP / Beta** | 30-50 hours | 30-50 | 6,000-10,000 | 15-18% |
| **Production** | 100+ hours | 100+ | 20,000+ | 10-15% |
| **World-Class** | 500+ hours | 500+ | 100,000+ | <10% |

**Budget Estimation:**
- **Recording**: RM 50-150/hour (voice actors) or free (crowdsourcing)
- **Transcription**: RM 100-200/hour (professional) or RM 30-60/hour (crowdsourcing)
- **Total Cost** (50 hours): RM 7,500-17,500

---

## 3. Data Collection Strategies

### 3.1 Strategy 1: Voice Actor Recordings (Controlled)

**Advantages:**
- High audio quality (studio conditions)
- Controlled content (ensure code-switching coverage)
- Fast turnaround (professional actors)

**Disadvantages:**
- Expensive (RM 100-200/hour)
- May sound less natural (reading scripts)
- Limited speaker diversity

**Process:**
1. Write diverse scripts covering Malaysian topics
2. Hire 10-20 Malaysian voice actors (diverse demographics)
3. Record in quiet studio environment
4. Each actor records 3-5 hours of content

**Script Examples:**
```
# Casual conversation
"So last night I pergi that new restaurant in Bukit Bintang lah. 
The food memang sedap, but quite mahal also."

# Work scenario  
"Can you tolong check the report ah? I think ada some errors in 
the data analysis section."

# Shopping
"This baju how much ah? Got discount or not? I mau beli two pieces."
```

**Cost**: ~RM 5,000 for 10 actors × 3 hours = 30 hours of data

---

### 3.2 Strategy 2: Crowdsourcing (Scalable)

**Advantages:**
- Cheaper (RM 30-60/hour)
- Diverse speakers naturally
- Scalable to 100+ hours

**Disadvantages:**
- Variable audio quality
- Requires quality control
- Longer turnaround time

**Platforms:**
- [Appen](https://appen.com/) - Global crowdsourcing platform
- [Lionbridge](https://www.lionbridge.com/) - Supports Malaysian languages
- [Local university partnerships](https://www.um.edu.my/) - Student volunteers
- Custom mobile app for data collection

**Crowdsourcing Task Design:**

```markdown
# Task: Record Malaysian Speech

## Instructions:
1. Read the provided sentence naturally (as you would speak to a friend)
2. Use code-switching if it feels natural
3. Add particles (lah, leh, loh) where appropriate
4. Record in a quiet room using your smartphone
5. Submit recording + confirm you read correctly

## Sentence to read:
"Can you tolong explain how this works? I rasa a bit confused leh."

## Requirements:
- Must be Malaysian resident
- Fluent in Malay + English
- Age 18-60
- 30 seconds of recording
- Payment: RM 2 per valid recording
```

**Quality Control:**
- Each recording reviewed by 2 annotators
- Reject low-quality audio (high noise, incorrect pronunciation)
- Bonus for high-quality submissions (RM 3 instead of RM 2)

**Cost**: ~RM 3,000 for 1,500 recordings × 2 minutes = 50 hours of data

---

### 3.3 Strategy 3: Found Data (Existing Corpus)

**Advantages:**
- No recording cost
- Natural speech patterns
- Large volume available

**Disadvantages:**
- Copyright/licensing issues
- May lack transcriptions
- Variable quality

**Potential Sources:**

1. **Common Voice (Mozilla)**
   - Open-source speech dataset
   - Malay corpus available (~20 hours validated)
   - English corpus (Malaysian accent) - limited
   - License: CC0 (public domain)
   - URL: https://commonvoice.mozilla.org/ms

2. **MALAY-DATASET (GitHub)**
   - Community-collected Malaysian speech
   - ~10 hours of read speech
   - License: Various (check per dataset)
   - URL: https://github.com/huseinzol05/malaya-speech

3. **YouTube / Podcast Transcription**
   - Scrape Malaysian content creators
   - Use YouTube auto-captions as initial transcripts (requires heavy editing)
   - Example channels: Malaysian tech reviewers, news channels
   - **Important**: Get permission from creators!

4. **Call Center Recordings**
   - Partner with local call centers (e.g., e-commerce, telco)
   - Highly valuable (real-world code-switching)
   - **Critical**: Anonymize personal data (PDPA compliance)

**Licensing Checklist:**
- ✅ Obtain permission from copyright holders
- ✅ Ensure data can be used for commercial purposes
- ✅ Attribute sources properly
- ✅ Comply with Malaysia PDPA (Personal Data Protection Act)

---

### 3.4 Strategy 4: Synthetic Data Generation

**Use Cases:**
- Bootstrap initial dataset
- Augment code-switching examples
- Test edge cases

**Methods:**

**A) Text-to-Speech (TTS) + Manual Mixing:**
```python
# Generate synthetic code-switched audio
from gtts import gTTS
import pydub

# English segment
tts_en = gTTS("Can you check the system", lang='en')
tts_en.save("segment1.mp3")

# Malay segment  
tts_ms = gTTS("tolong", lang='ms')
tts_ms.save("segment2.mp3")

# Combine segments
audio1 = pydub.AudioSegment.from_mp3("segment1.mp3")
audio2 = pydub.AudioSegment.from_mp3("segment2.mp3")
combined = audio1[:2000] + audio2 + audio1[2000:]
combined.export("synthetic_codeswitched.wav", format="wav")
```

**B) Language Model-Generated Scripts:**
```python
# Use GPT-4 to generate realistic Malaysian scripts
prompt = """
Generate 10 natural Malaysian English sentences with code-switching 
between English and Malay. Include discourse particles (lah, leh, loh).
Examples:
- "Can you tolong pass me the remote lah?"
- "I think this one lagi cheaper than that one."
"""

# GPT-4 generates realistic sentences → record with voice actors
```

**Limitations:**
- TTS voices may sound robotic (fine for augmentation, not primary data)
- Doesn't capture natural prosody of code-switching
- Use as supplement, not replacement for real speech

---

## 4. Recording Guidelines

### 4.1 Equipment Recommendations

**Professional Setup (Studio Quality):**
- **Microphone**: Shure SM7B, Rode NT1-A ($300-400)
- **Audio Interface**: Focusrite Scarlett 2i2 ($150)
- **Headphones**: Audio-Technica ATH-M50x ($150)
- **Recording Software**: Audacity (free), Adobe Audition ($20/month)
- **Environment**: Soundproofed room or vocal booth

**Budget Setup (Crowdsourcing):**
- **Microphone**: Blue Yeti USB mic ($100) or smartphone headset
- **Recording Software**: Audacity (free), Voice Recorder apps
- **Environment**: Quiet room with soft furnishings (absorb echoes)

### 4.2 Recording Protocol

**Step 1: Environment Setup**
- Close windows (reduce external noise)
- Turn off fans, AC, or noisy appliances
- Use carpets/curtains to absorb echoes
- Record in a small room (less reverb)

**Step 2: Microphone Positioning**
- Distance: 6-12 inches from mouth
- Angle: Slightly off-axis (reduces plosives "p", "b")
- Use pop filter to reduce "p", "t", "k" sounds

**Step 3: Recording Settings**
```
Sample Rate: 48 kHz (or 44.1 kHz)
Bit Depth: 24-bit
Format: WAV (uncompressed)
Mono: Yes (single channel)
```

**Step 4: Recording Process**
1. Do a test recording (10 seconds)
2. Listen back for quality issues
3. Adjust microphone position/volume if needed
4. Record each sentence separately (easier to edit)
5. Save as: `speaker_001_sentence_001.wav`

**Step 5: Post-Recording QA**
- Listen to each file for errors
- Re-record if: clipping, noise, mispronunciation
- Trim silence at start/end (leave 0.5s padding)

### 4.3 Recording Session Best Practices

**For Voice Actors:**
- Warm up voice before session (vocal exercises)
- Hydrate well (water, no dairy/caffeine right before)
- Take 5-minute breaks every 30 minutes (prevent vocal fatigue)
- Record 2-3 takes of each sentence (choose best one)
- Vary intonation to sound natural

**For Crowdsourcing:**
- Provide clear instructions + video tutorial
- Give example recordings (good vs bad quality)
- Automatic quality check before submission
- Allow re-recordings if rejected

---

## 5. Transcription Guidelines

### 5.1 General Transcription Rules

**Rule 1: Verbatim Transcription**
Transcribe exactly what is spoken, including:
- False starts: "I think... I mean, I believe..."
- Fillers: "um", "uh", "er"
- Repetitions: "the the system"
- Particles: "lah", "leh", "loh"

**Rule 2: Spelling Conventions**
- **English**: Use Malaysian English spelling (British-leaning)
  - "colour" not "color"
  - "centre" not "center"
  - But: Allow Americanisms if commonly used in Malaysia ("okay" not "OK")

- **Malay**: Use standard Bahasa Malaysia orthography
  - "dengan" not "dgn"
  - "yang" not "yg"
  - "ada" not "ade"

- **Particles**: Consistent romanization
  - "lah" (not "la", "laa")
  - "leh" (not "lah" for the suggestion particle)
  - "loh" (not "lo", "lor" for obviousness)
  - "meh" (not "meh?" with punctuation)

**Rule 3: Capitalization**
- Capitalize first word of sentence
- Capitalize proper nouns (names, places, brands)
- Do NOT capitalize language switches mid-sentence

Example:
```
✓ Correct: "Yesterday I pergi Pavilion lah."
✗ Wrong: "Yesterday I Pergi Pavilion Lah."
```

**Rule 4: Punctuation**
- Use punctuation to indicate sentence boundaries
- Commas for natural pauses (optional, use sparingly)
- No punctuation for particles (they are part of the word)

Example:
```
✓ Correct: "Can you check lah, I think got problem."
✗ Wrong: "Can you check, lah. I think got problem."
```

### 5.2 Code-Switching Transcription

**Language Boundaries:**
Whisper will handle language detection, but annotators should mark languages for evaluation:

```json
{
  "text": "Can you tolong check the system lah",
  "words": [
    {"word": "Can", "language": "en"},
    {"word": "you", "language": "en"},
    {"word": "tolong", "language": "ms"},
    {"word": "check", "language": "en"},
    {"word": "the", "language": "en"},
    {"word": "system", "language": "en"},
    {"word": "lah", "language": "particle"}
  ]
}
```

**Ambiguous Cases:**

Some words exist in both English and Malay:
- "can" (English: able to / Malay: abbreviation of "boleh can")
- "best" (English: superlative / Malay slang: excellent)

**Decision Rule:** Use context to determine primary language. If unclear, default to the surrounding sentence's dominant language.

### 5.3 Discourse Particle Transcription

**Particle Inventory:**

| Particle | Spelling | Usage | Example |
|----------|----------|-------|---------|
| **lah** | lah | Emphasis, assertion | "Of course can lah!" |
| **leh** | leh | Suggestion, possibility | "We can try leh?" |
| **loh** | loh | Obviousness | "I told you already loh!" |
| **meh** | meh | Doubt, surprise | "Really meh?" |
| **lor** | lor | Resignation | "Cannot help it lor." |
| **wor** | wor | Concern | "How wor?" |
| **hor** | hor | Seeking agreement | "You know right hor?" |
| **mah** | mah | Explanation | "He's busy mah!" |

**Transcription Guidelines:**

1. **Attach to preceding word or keep separate?**
   - **Keep separate**: Easier for tokenization
   - Example: "can lah" (not "canlah")

2. **Multiple particles in sequence:**
   ```
   "You know lah hor" (two particles)
   "Cannot lah lor" (two particles, different functions)
   ```

3. **Particle vs similar words:**
   - "lah" (particle) ≠ "la" (shortened "already")
   - Context: "Done lah" (particle) vs "Done la" (done already)
   - **Decision**: Use "lah" for particle, "dah/already" for temporal marker

### 5.4 Handling Special Cases

**Non-Speech Sounds:**
```
[LAUGH]    - Laughter
[COUGH]    - Coughing
[NOISE]    - Background noise
[MUSIC]    - Music playing
[OVERLAP]  - Multiple speakers talking simultaneously
[INAUDIBLE] - Speech unclear/unintelligible
```

**Long Pauses:**
```
# Pause > 2 seconds
"I think... [PAUSE] ...maybe we should go."
```

**Incomplete Sentences:**
```
"Can you- can you pass me the-"  (speaker cuts off)
```

---

## 6. Data Annotation Schema

### 6.1 JSON Format

```json
{
  "audio_id": "MY_0001",
  "audio_filepath": "audio/MY_0001.wav",
  "text": "Can you tolong check the system lah",
  "duration": 3.2,
  "sample_rate": 16000,
  "speaker_id": "SPK_001",
  "speaker_gender": "female",
  "speaker_age": 28,
  "accent": "kuala_lumpur",
  "recording_environment": "studio",
  "language": "mixed",
  "code_switches": [
    {"start_word": 0, "end_word": 1, "language": "en"},
    {"start_word": 2, "end_word": 2, "language": "ms"},
    {"start_word": 3, "end_word": 5, "language": "en"},
    {"start_word": 6, "end_word": 6, "language": "particle"}
  ],
  "word_timestamps": [
    {"word": "Can", "start": 0.0, "end": 0.3, "confidence": 0.95},
    {"word": "you", "start": 0.3, "end": 0.5, "confidence": 0.97},
    {"word": "tolong", "start": 0.5, "end": 0.9, "confidence": 0.94},
    {"word": "check", "start": 0.9, "end": 1.2, "confidence": 0.96},
    {"word": "the", "start": 1.2, "end": 1.3, "confidence": 0.98},
    {"word": "system", "start": 1.3, "end": 1.8, "confidence": 0.95},
    {"word": "lah", "start": 1.8, "end": 2.0, "confidence": 0.89}
  ],
  "quality_score": 4.5,
  "transcriber_id": "TRANS_012",
  "verified": true,
  "created_at": "2025-10-12T10:30:00Z"
}
```

### 6.2 Metadata Fields

**Required Fields:**
- `audio_id`: Unique identifier
- `audio_filepath`: Path to audio file
- `text`: Transcription
- `duration`: Audio length in seconds

**Recommended Fields:**
- `speaker_id`: Speaker identifier (for speaker diversity analysis)
- `language`: "en", "ms", "mixed"
- `code_switches`: Language boundary annotations

**Optional Fields:**
- `word_timestamps`: For evaluation (forced alignment)
- `quality_score`: 1-5 rating by annotator
- `transcriber_id`: For quality control

---

## 7. Quality Control

### 7.1 Multi-Stage QC Process

**Stage 1: Automated Checks**
```python
def validate_audio(audio_path: str, metadata: dict):
    """Automated audio quality checks."""
    import librosa
    import numpy as np
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    checks = {
        "duration_match": False,
        "snr_acceptable": False,
        "no_clipping": False,
        "correct_sample_rate": False,
    }
    
    # Duration check
    actual_duration = len(audio) / sr
    expected_duration = metadata.get("duration", 0)
    if abs(actual_duration - expected_duration) < 0.1:
        checks["duration_match"] = True
    
    # SNR check (signal-to-noise ratio)
    # Simple method: compare RMS of loudest vs quietest segments
    rms = librosa.feature.rms(y=audio)[0]
    snr_estimate = 20 * np.log10(np.max(rms) / np.mean(rms))
    if snr_estimate > 15:
        checks["snr_acceptable"] = True
    
    # Clipping check
    if np.max(np.abs(audio)) < 0.99:
        checks["no_clipping"] = True
    
    # Sample rate check
    if sr >= 16000:
        checks["correct_sample_rate"] = True
    
    return checks

# Usage
result = validate_audio("audio/MY_0001.wav", metadata)
if not all(result.values()):
    print(f"Quality issues: {[k for k,v in result.items() if not v]}")
```

**Stage 2: Manual Review**
- Random sample 10% of dataset
- Listen to audio + read transcript
- Flag errors: wrong words, missing particles, mispronunciations

**Stage 3: Cross-Validation**
- Each audio reviewed by 2 independent transcribers
- Compare transcriptions
- If disagreement > 2 words, escalate to expert reviewer

**Stage 4: Test Set Curation**
- Human expert reviews 100% of test set
- Ensure gold-standard quality
- Test set must be representative of production use cases

### 7.2 Quality Metrics

**Transcription Agreement:**
```python
def inter_annotator_agreement(transcripts_1: list, transcripts_2: list):
    """Calculate agreement between two transcribers."""
    import jiwer
    
    wer = jiwer.wer(transcripts_1, transcripts_2)
    agreement = 1 - wer
    
    return agreement

# Target: > 95% agreement (WER < 5% between annotators)
```

**Audio Quality Score:**
- 5 = Studio quality (SNR > 35 dB, no noise)
- 4 = High quality (SNR 25-35 dB, minimal noise)
- 3 = Good quality (SNR 15-25 dB, acceptable for training)
- 2 = Fair quality (SNR 10-15 dB, use with caution)
- 1 = Poor quality (SNR < 10 dB, reject)

**Dataset Health Metrics:**
- % samples with quality score ≥ 3: Target > 95%
- % samples with verified transcriptions: Target 100% for test set, > 80% for train
- Inter-annotator agreement: Target > 95%

---

## 8. Data Augmentation

### 8.1 Audio Augmentation Techniques

**1. Additive Noise**
```python
import torch_audiomentations as tam

add_noise = tam.AddBackgroundNoise(
    sounds_path="./noise_samples/",  # Traffic, cafe, office sounds
    min_snr_in_db=5.0,
    max_snr_in_db=15.0,
    p=0.5,  # 50% probability
)

augmented_audio = add_noise(original_audio, sample_rate=16000)
```

**2. Room Impulse Response (Reverb)**
```python
apply_reverb = tam.ApplyImpulseResponse(
    ir_paths="./room_impulses/",  # Small room, large hall, etc.
    p=0.3,
)
```

**3. Pitch Shifting**
```python
pitch_shift = tam.PitchShift(
    min_transpose_semitones=-2,  # Lower pitch
    max_transpose_semitones=2,   # Higher pitch
    p=0.3,
)
```

**4. Time Stretching (Speed Perturbation)**
```python
time_stretch = tam.TimeStretch(
    min_rate=0.9,  # 10% slower
    max_rate=1.1,  # 10% faster
    p=0.3,
)
```

**5. Combining Augmentations**
```python
augmentation_pipeline = tam.Compose([
    add_noise,
    apply_reverb,
    pitch_shift,
    time_stretch,
])

# Apply to 30-50% of training data
```

**Best Practices:**
- Apply augmentations ONLY to training set (not validation/test)
- Don't over-augment (can degrade performance)
- Keep 50-70% of data un-augmented (pristine)

### 8.2 Text Augmentation (Synthetic Code-Switching)

**Technique: Back-Translation**
```python
from googletrans import Translator

def generate_code_switched_variants(text: str):
    """Generate code-switched variants of a sentence."""
    translator = Translator()
    
    # Translate parts to Malay and back
    words = text.split()
    variants = []
    
    # Replace random English words with Malay
    for i in range(len(words)):
        if random.random() < 0.3:  # 30% chance
            malay_word = translator.translate(words[i], src='en', dest='ms').text
            new_text = ' '.join(words[:i] + [malay_word] + words[i+1:])
            variants.append(new_text)
    
    return variants

# Example:
# Input: "Can you check the system"
# Output: ["Can you semak the system", "Can you check the sistem"]
```

---

## 9. Dataset Organization

### 9.1 Directory Structure

```
malaysian_asr_dataset/
├── README.md                 # Dataset documentation
├── metadata.json             # Overall dataset statistics
├── train/
│   ├── audio/
│   │   ├── MY_TRAIN_0001.wav
│   │   ├── MY_TRAIN_0002.wav
│   │   └── ...
│   ├── transcripts.json      # All transcripts + metadata
│   └── speakers.json         # Speaker metadata
├── validation/
│   ├── audio/
│   ├── transcripts.json
│   └── speakers.json
├── test/
│   ├── audio/
│   ├── transcripts.json
│   └── speakers.json
└── tools/
    ├── validate_data.py      # QC scripts
    ├── augment_audio.py
    └── analyze_dataset.py
```

### 9.2 Dataset Splits

**Split Ratios:**
- Training: 80% (e.g., 40 hours out of 50)
- Validation: 10% (5 hours)
- Test: 10% (5 hours)

**Important:** 
- **Speaker-independent splits**: Test speakers should NOT appear in training set
- **Domain balance**: Each split should have similar distribution of:
  - Code-switching density
  - Particle usage
  - Audio quality
  - Speaker demographics

**Stratified Splitting:**
```python
from sklearn.model_selection import train_test_split

def stratified_split(dataset, test_size=0.2, val_size=0.1):
    """Split dataset while maintaining speaker independence."""
    
    # Group by speaker
    speaker_groups = {}
    for item in dataset:
        speaker_id = item["speaker_id"]
        if speaker_id not in speaker_groups:
            speaker_groups[speaker_id] = []
        speaker_groups[speaker_id].append(item)
    
    # Split speakers (not samples)
    speakers = list(speaker_groups.keys())
    train_speakers, test_speakers = train_test_split(
        speakers, test_size=test_size, random_state=42
    )
    train_speakers, val_speakers = train_test_split(
        train_speakers, test_size=val_size, random_state=42
    )
    
    # Collect samples
    train_data = [item for spk in train_speakers for item in speaker_groups[spk]]
    val_data = [item for spk in val_speakers for item in speaker_groups[spk]]
    test_data = [item for spk in test_speakers for item in speaker_groups[spk]]
    
    return train_data, val_data, test_data
```

---

## 10. Tools & Automation

### 10.1 Transcription Tools

**Tool 1: Audacity (Free, Open-Source)**
- Good for: Manual transcription, audio editing
- Features: Playback speed control, spectrograms, labels
- URL: https://www.audacityteam.org/

**Tool 2: Transcriber (Free)**
- Good for: Keyboard shortcuts for fast transcription
- Features: Auto-repeat, timestamp insertion
- URL: https://transcribertools.com/

**Tool 3: ELAN (Free, Linguistic Research)**
- Good for: Detailed annotation with time-aligned tiers
- Features: Multiple annotation tiers, export to JSON/XML
- URL: https://archive.mpi.nl/tla/elan

**Tool 4: Label Studio (Free, ML-Focused)**
- Good for: Team collaboration, integration with ML pipelines
- Features: Web-based, customizable interface, quality review workflows
- URL: https://labelstud.io/

### 10.2 Forced Alignment (Automatic Timestamp Generation)

**Montreal Forced Aligner (MFA):**
```bash
# Install MFA
conda install -c conda-forge montreal-forced-aligner

# Download pre-trained models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Run alignment
mfa align \
  ./audio/ \
  ./transcripts.txt \
  english_us_arpa \
  english_us_arpa \
  ./output_aligned/
```

**Output: TextGrid files with word-level timestamps**

### 10.3 Automated QC Script

```python
#!/usr/bin/env python3
"""
Dataset Quality Control Script
Validates audio files and transcriptions
"""

import json
import librosa
import numpy as np
from pathlib import Path

def check_dataset_quality(data_dir: Path):
    """Run comprehensive QC on dataset."""
    
    issues = []
    
    # Load metadata
    with open(data_dir / "transcripts.json") as f:
        transcripts = json.load(f)
    
    for audio_id, metadata in transcripts.items():
        audio_path = data_dir / metadata["audio_filepath"]
        
        # Check 1: Audio file exists
        if not audio_path.exists():
            issues.append(f"{audio_id}: Audio file not found")
            continue
        
        # Check 2: Audio is readable
        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            issues.append(f"{audio_id}: Cannot read audio - {e}")
            continue
        
        # Check 3: Duration matches metadata
        actual_duration = len(audio) / sr
        expected_duration = metadata.get("duration", 0)
        if abs(actual_duration - expected_duration) > 0.5:
            issues.append(f"{audio_id}: Duration mismatch ({actual_duration:.1f}s vs {expected_duration:.1f}s)")
        
        # Check 4: Transcription not empty
        if not metadata.get("text", "").strip():
            issues.append(f"{audio_id}: Empty transcription")
        
        # Check 5: Audio quality (SNR)
        rms = librosa.feature.rms(y=audio)[0]
        snr_estimate = 20 * np.log10(np.max(rms) / np.mean(rms))
        if snr_estimate < 10:
            issues.append(f"{audio_id}: Low SNR ({snr_estimate:.1f} dB)")
        
        # Check 6: No clipping
        if np.max(np.abs(audio)) > 0.99:
            issues.append(f"{audio_id}: Audio clipping detected")
    
    # Report
    print(f"Checked {len(transcripts)} samples")
    print(f"Found {len(issues)} issues:")
    for issue in issues[:20]:  # Show first 20
        print(f"  - {issue}")
    
    return issues

if __name__ == "__main__":
    check_dataset_quality(Path("./train/"))
```

---

## 11. Quick Start Guide

### 11.1 Minimal Dataset (10 Hours)

**Week 1: Recording (5 days)**
1. Recruit 5 speakers (diverse age, gender)
2. Prepare 50 script prompts (mix of casual, formal, code-switched)
3. Each speaker records 2 hours (100 sentences × 1-2 min each)
4. Total: 10 hours of audio

**Week 2: Transcription (5 days)**
1. Hire 2 transcribers (native Malaysian)
2. Each transcribes 5 hours
3. Cross-check 20% of transcriptions for agreement
4. Fix errors and finalize

**Week 3: QC & Formatting (3 days)**
1. Run automated QC script
2. Fix flagged issues (re-record or re-transcribe)
3. Organize into train/val/test splits
4. Generate JSON metadata files

**Budget: ~RM 3,000**
- Recording: RM 500 (5 speakers × RM 100/person)
- Transcription: RM 2,000 (10 hours × RM 200/hour)
- QC: RM 500 (manual review)

### 11.2 One-Command Data Prep

```bash
#!/bin/bash
# prepare_dataset.sh

# Step 1: Validate audio files
python tools/validate_audio.py --input_dir ./raw_audio/ --output_dir ./validated_audio/

# Step 2: Generate transcripts (using Whisper for initial draft)
python tools/generate_draft_transcripts.py --audio_dir ./validated_audio/ --output transcripts_draft.json

# Step 3: Manual editing
# (Open transcripts_draft.json in text editor, fix errors)

# Step 4: Forced alignment (generate word timestamps)
mfa align ./validated_audio/ ./transcripts_edited.txt english_us_arpa english_us_arpa ./aligned/

# Step 5: Split into train/val/test
python tools/split_dataset.py --input ./aligned/ --output ./final_dataset/ --test_size 0.1 --val_size 0.1

# Step 6: Generate HuggingFace dataset
python tools/create_hf_dataset.py --input ./final_dataset/ --output ./malaysian_asr_hf/

echo "Dataset ready at: ./malaysian_asr_hf/"
```

### 11.3 Sample Dataset Download

**For experimentation, use a small sample:**

```python
from datasets import load_dataset

# Load Mozilla Common Voice (Malay subset)
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "ms", split="train[:100]")

# Or create your own minimal test set
# 20 recordings × 30 seconds = 10 minutes (enough to test pipeline)
```

---

## 12. Appendix

### 12.1 Particle Usage Statistics

Target distribution in dataset (based on Malaysian speech corpus analysis):

| Particle | Frequency | Target Count (per 10 hours) |
|----------|-----------|------------------------------|
| lah | 35% | ~700 occurrences |
| leh | 15% | ~300 |
| loh | 12% | ~240 |
| meh | 10% | ~200 |
| lor | 10% | ~200 |
| Others | 18% | ~360 |

### 12.2 Common Transcription Errors

| Error | Example | Correction |
|-------|---------|------------|
| Missing particle | "Can you check" | "Can you check lah" |
| Wrong particle | "Can you check lah?" | "Can you check leh?" (suggestion) |
| Merged words | "canlah" | "can lah" |
| Wrong language | "tolong" → "too long" | "tolong" (Malay for "help") |
| Missing code-switch | All English | Mark mixed-language segments |

### 12.3 Resources

**Linguistic References:**
- Baskaran, Loga. (2005). A Malaysian English primer: Aspects of Malaysian English features.
- Wong, Bee Eng. (2014). Malaysian English Discourse Particles

**Tools:**
- Audacity: https://www.audacityteam.org/
- Label Studio: https://labelstud.io/
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/

**Datasets:**
- Common Voice (Malay): https://commonvoice.mozilla.org/ms
- Malaya-Speech: https://github.com/huseinzol05/malaya-speech

---

**End of Data Preparation Guide**

*For training procedures, see Training Strategy Guide (04).*

