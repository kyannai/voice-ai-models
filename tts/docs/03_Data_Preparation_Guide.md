# Data Preparation Guide
# Malaysian Multilingual TTS System

**Version:** 1.0  
**Date:** October 12, 2025  
**Status:** Draft  
**Owner:** Data Team

---

## 1. Overview

This guide provides comprehensive instructions for collecting, annotating, and preparing training data for the Malaysian Multilingual TTS system.

### 1.1 Data Requirements Summary

| Category | Target | Priority |
|----------|--------|----------|
| Total Audio Duration | 50-75 hours | Critical |
| Training Set | 40-60 hours | Critical |
| Validation Set | 5-8 hours | Critical |
| Test Set | 3-5 hours | Critical |
| Number of Speakers | 5-10 | High |
| Gender Balance | 50/50 M/F | High |
| Code-Switching Coverage | 40%+ of data | Critical |
| Particle Examples | 5000+ instances | Critical |

---

## 2. Data Collection Strategies

### 2.1 Strategy 1: Professional Voice Actors (Primary)

#### 2.1.1 Recruitment Criteria

**Essential Requirements:**
- Native Malaysian speaker
- Fluent in Malay + English + basic Mandarin
- Natural code-switching ability
- Clear pronunciation and diction
- Professional recording experience
- Age: 25-45 years
- No strong regional accent (neutral Malaysian)

**Preferred Qualifications:**
- Voice acting or broadcasting experience
- Familiarity with IPA or phonetic transcription
- Previous TTS/dubbing work
- Ability to maintain consistent tone

#### 2.1.2 Voice Actor Profiles

**Target Distribution:**

| Profile | Count | Description |
|---------|-------|-------------|
| Young Female (25-35) | 2 | Clear, energetic voice |
| Young Male (25-35) | 2 | Clear, professional voice |
| Mature Female (35-45) | 1 | Authoritative, warm voice |
| Mature Male (35-45) | 1 | Deep, professional voice |
| Neutral/Androgynous | 1-2 | Flexible, character voices |

#### 2.1.3 Recording Specifications

**Audio Technical Requirements:**
```yaml
# Recording Configuration
sample_rate: 44100  # Hz (will downsample to 22050)
bit_depth: 24  # bit
format: WAV (uncompressed PCM)
channels: 1 (mono)
microphone: Large-diaphragm condenser
preamp: Clean, transparent (avoid heavy processing)
recording_software: Audacity, Reaper, Pro Tools, or similar
```

**Recording Environment:**
- Treated recording booth or very quiet room
- Background noise: < 30 dB SPL
- No reverb or echo (use acoustic treatment)
- No HVAC or electrical noise
- Consistent room (same location for all sessions)

**Session Guidelines:**
- Session length: 2-3 hours max (avoid vocal fatigue)
- Warm-up: 10 minutes before recording
- Breaks: 10 minutes every hour
- Hydration: Water available (no ice, no tea/coffee during recording)
- Distance from mic: 15-20 cm consistent
- Pop filter: Always use

#### 2.1.4 Script Preparation

**Script Composition:**

```python
# scripts/generate_recording_script.py

script_distribution = {
    # Pure languages (40%)
    'pure_malay': {
        'percentage': 15,
        'examples': 1500,
        'avg_length': 10,  # words
        'difficulty': ['simple', 'medium', 'complex']
    },
    'pure_english': {
        'percentage': 15,
        'examples': 1500,
        'avg_length': 10,
        'difficulty': ['simple', 'medium', 'complex']
    },
    'pure_chinese': {
        'percentage': 10,
        'examples': 1000,
        'avg_length': 8,
        'difficulty': ['simple', 'medium']  # Pinyin only
    },
    
    # Code-switched (50%)
    'malay_english': {
        'percentage': 25,
        'examples': 2500,
        'patterns': ['MS-EN', 'EN-MS', 'MS-EN-MS'],
        'avg_length': 12
    },
    'malay_chinese': {
        'percentage': 10,
        'examples': 1000,
        'patterns': ['MS-ZH', 'ZH-MS'],
        'avg_length': 10
    },
    'english_chinese': {
        'percentage': 10,
        'examples': 1000,
        'patterns': ['EN-ZH', 'ZH-EN'],
        'avg_length': 10
    },
    'triple_mix': {
        'percentage': 5,
        'examples': 500,
        'patterns': ['MS-EN-ZH', 'EN-MS-ZH'],
        'avg_length': 15
    },
    
    # Particle-focused (10%)
    'particle_rich': {
        'percentage': 10,
        'examples': 1000,
        'particles_per_sentence': 2,
        'contexts': ['emphatic', 'questioning', 'casual']
    }
}
```

**Example Sentences by Category:**

##### Pure Malay
```
1. Selamat pagi, saya mahu beli tiket ke Kuala Lumpur.
2. Restoran ini terkenal dengan masakan tradisional Malaysia.
3. Boleh tolong saya buatkan tempahan untuk lima orang?
4. Cuaca hari ini sangat panas dan lembap.
5. Saya suka membaca buku tentang sejarah Malaysia.
```

##### Pure English (Malaysian accent)
```
1. Good morning, I would like to order some nasi lemak for breakfast.
2. The traffic is very bad today because of the heavy rain.
3. Can you help me find the nearest petrol station?
4. This shopping mall has many interesting stores to visit.
5. I am planning to go to Penang for the long weekend.
```

##### Pure Chinese (Pinyin)
```
1. Ni hao, wo xiang mai yi bei kopi. (你好，我想买一杯咖啡)
2. Zhe ge hen hao chi, ni yao bu yao shi? (这个很好吃，你要不要试？)
3. Wo ming tian yao qu xin jia po. (我明天要去新加坡)
4. Zhe li de ren hen duo. (这里的人很多)
5. Qing wen, ze me zou dao ji chang? (请问，怎么走到机场？)
```

##### Code-Switched: Malay-English
```
1. Saya nak pergi shopping mall to buy some new clothes.
2. Boss kata kita kena complete this project by Friday.
3. Let's go makan at the mamak stall near our office.
4. Dia very pandai in mathematics and science subjects.
5. Kenapa you tak reply my message since yesterday?
```

##### Code-Switched: Malay-English-Chinese
```
1. Saya rasa this char kway teow hen hao chi!
2. Boss kata tomorrow we need to submit the bao gao.
3. Let's go to the pasar malam to buy xin xian food.
4. Dia belajar Mandarin at the zhong wen school near his house.
5. Mamak uncle very friendly, always say "xie xie" when we pay.
```

##### Particle-Rich Sentences
```
1. Can lah, no problem one!
2. You sure meh? Sounds too good to be true leh.
3. Aiyah, I told you already loh, don't do like that.
4. Wah, this one very expensive sia!
5. Must go early lor, otherwise sure cannot get parking.
6. Really ah? I don't believe leh.
7. Like that also can meh?
8. Faster lah, we're going to be late already!
9. He never listen one, so stubborn lor.
10. So nice hor, the weather today!
```

#### 2.1.5 Recording Workflow

**Step-by-Step Process:**

```
1. PRE-SESSION
   ├─ Voice actor receives script 24h advance (familiarization)
   ├─ Technical check: mic, levels, room noise
   ├─ Warm-up exercises (5-10 minutes)
   └─ Test recording (check quality)

2. RECORDING SESSION
   ├─ Record in batches of 50-100 sentences
   ├─ Director monitors quality in real-time
   ├─ Mark takes (take 1, 2, 3 if needed)
   ├─ Note any issues (stumbles, background noise)
   └─ Break every 60 minutes

3. POST-SESSION
   ├─ Quick quality check
   ├─ Backup recordings immediately
   ├─ Log session metadata
   └─ Schedule next session

4. POST-PRODUCTION
   ├─ Select best takes
   ├─ Trim silence (keep 0.2s before/after)
   ├─ Normalize loudness (-20 LUFS)
   ├─ High-pass filter (remove < 80 Hz)
   └─ Save as WAV 24-bit 44.1kHz
```

**Recording Script Template:**

```
SESSION: 2025-10-15_Speaker01_Session03
SPEAKER: SP001 (Female, 28 years old)
DURATION: 2.5 hours
SENTENCES: 001-250

---
ID: SP001_0001
TEXT: Saya nak pergi shopping mall to buy some new clothes.
LANGUAGE: MS-EN
PARTICLES: nak
TAKE: 1
STATUS: APPROVED
NOTES: Clear pronunciation, good code-switching

---
ID: SP001_0002
TEXT: Boss kata kita kena complete this project by Friday.
LANGUAGE: MS-EN
PARTICLES: kena
TAKE: 2
STATUS: APPROVED
NOTES: Retake - first take had lip smack

---
```

#### 2.1.6 Budget & Timeline

**Per Speaker Costs:**
```
Recording rate: RM 200-400 per hour
Hours per speaker: 10-12 hours
Total per speaker: RM 2,000-4,800

For 5-10 speakers:
Low estimate: RM 10,000 (5 speakers × RM 2,000)
High estimate: RM 48,000 (10 speakers × RM 4,800)
Recommended: RM 25,000-35,000 (7-8 speakers × RM 3,500)
```

**Additional Costs:**
- Studio rental: RM 100-300 per hour (if not included)
- Audio engineer: RM 150-300 per hour
- Post-production: RM 50-100 per hour
- Script development: RM 2,000-5,000
- Project management: RM 3,000-8,000

**Total Budget Estimate: RM 35,000-55,000**

**Timeline:**
- Script preparation: 2-3 weeks
- Voice actor recruitment: 2-3 weeks
- Recording sessions: 4-6 weeks (parallel)
- Post-production: 2-3 weeks
- Quality control: 1-2 weeks
- **Total: 10-14 weeks**

---

### 2.2 Strategy 2: Crowdsourcing (Supplementary)

#### 2.2.1 Platform Setup

**Technology Stack:**
- Frontend: React web app
- Backend: FastAPI
- Storage: S3 or MinIO
- Database: PostgreSQL
- Audio recorder: RecordRTC or similar

**Key Features:**
- Browser-based recording (no downloads)
- Sentence prompt display
- Playback before submission
- Quality feedback
- Gamification (points, leaderboards)
- Payment integration (if paid)

#### 2.2.2 Recording Interface

```javascript
// Simplified crowdsourcing interface

const RecordingInterface = () => {
  const [sentence, setSentence] = useState("");
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  
  const startRecording = async () => {
    // Request microphone permission
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    // ... recording logic
  };
  
  const submitRecording = async () => {
    // Upload to server
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('text', sentence);
    formData.append('user_id', userId);
    
    await fetch('/api/submit-recording', {
      method: 'POST',
      body: formData
    });
  };
  
  return (
    <div>
      <h2>Read this sentence:</h2>
      <p className="sentence-prompt">{sentence}</p>
      <button onClick={startRecording}>Record</button>
      <button onClick={stopRecording}>Stop</button>
      <button onClick={submitRecording}>Submit</button>
    </div>
  );
};
```

#### 2.2.3 Quality Control Pipeline

**Automated Checks:**
1. **Audio Quality:**
   - Duration check (2-20 seconds expected)
   - Volume level (not too quiet/loud)
   - Silence detection (not all silent)
   - Clipping detection
   - Background noise level

2. **Content Validation:**
   - ASR transcription match (>80% accuracy)
   - Language detection validation
   - Duplicate detection

**Manual Validation:**
- Random sample review (10%)
- Flagged recordings review (100%)
- Native speaker verification

**Rejection Criteria:**
- Poor audio quality (SNR < 20 dB)
- Wrong text read
- Strong background noise
- Obvious mispronunciation
- Non-native accent (if specified)

#### 2.2.4 Incentive Structure

**Payment Model:**

| Tier | Rate | Requirements |
|------|------|--------------|
| Beginner | RM 0.30/recording | First 100 recordings |
| Intermediate | RM 0.50/recording | 100-500 recordings, 90% approval |
| Advanced | RM 0.75/recording | 500+ recordings, 95% approval |
| Expert | RM 1.00/recording | 1000+ recordings, 98% approval |

**Bonuses:**
- Daily streak: +10% for 7-day streak
- Quality bonus: +20% if 98% approval rate
- Referral bonus: RM 10 per referred contributor

**Expected Costs:**
- Target: 10,000 recordings
- Average rate: RM 0.50
- Total: RM 5,000
- Platform overhead: +20%
- **Total Budget: RM 6,000-8,000**

---

### 2.3 Strategy 3: Found Data (Augmentation)

#### 2.3.1 Data Sources

**Potential Sources:**

1. **Malaysian Podcasts**
   - BFM Radio podcasts
   - The Podcast Network Asia
   - Local content creators
   - **Requirement:** Written permission + transcripts

2. **YouTube Channels**
   - Educational channels
   - News channels
   - Tech review channels
   - **Requirement:** Creative Commons license or permission

3. **Audiobooks**
   - Malaysian literature
   - Language learning materials
   - **Requirement:** Public domain or licensed

4. **Radio Broadcasts**
   - Public service announcements
   - News broadcasts
   - **Requirement:** Public domain or permission

#### 2.3.2 Legal Considerations

**Critical Requirements:**
- ✅ Explicit permission from content owner
- ✅ Commercial use rights
- ✅ Derivative works allowed
- ✅ No attribution requirement (preferred)
- ✅ Written agreement

**Licensing Template:**

```
DATA USAGE AGREEMENT

This agreement grants [Your Company] the right to:
1. Use audio recordings for training TTS models
2. Create derivative works (synthetic speech)
3. Commercial use of trained models
4. No obligation to provide attribution

Content Owner: [Name]
Date: [Date]
Signature: __________
```

#### 2.3.3 Processing Pipeline

**Step 1: Audio Extraction**
```bash
# Extract audio from video
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 1 output.wav
```

**Step 2: Speaker Diarization**
```python
# Identify different speakers
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("audio.wav")

# Output: speaker segments
# SPEAKER_00: [0.5-3.2s], [5.1-8.9s]
# SPEAKER_01: [3.2-5.1s], [8.9-12.3s]
```

**Step 3: Automatic Transcription**
```python
# Use Whisper or similar ASR
import whisper

model = whisper.load_model("large")
result = model.transcribe(
    "audio.wav",
    language="ms",  # or None for auto-detection
    task="transcribe"
)

transcription = result["text"]
segments = result["segments"]
```

**Step 4: Quality Filtering**
```python
def filter_quality(audio_segment, transcription):
    """
    Filter out low-quality segments
    """
    # Check duration
    if len(audio_segment) < 1.0 or len(audio_segment) > 20.0:
        return False
    
    # Check SNR
    if compute_snr(audio_segment) < 25:
        return False
    
    # Check transcription confidence
    if transcription['confidence'] < 0.8:
        return False
    
    # Check for music/noise
    if contains_music(audio_segment):
        return False
    
    return True
```

**Expected Yield:**
- Time investment: High (processing + legal)
- Cost: Low-Medium (mostly time)
- Quality: Variable
- Quantity: 10-30 hours if successful
- **Recommendation:** Lower priority, pursue if other sources insufficient

---

## 3. Data Annotation

### 3.1 Annotation Schema

#### 3.1.1 Basic Metadata

```json
{
  "audio_file": "SP001_0001.wav",
  "text": "Saya nak go to the mall lah",
  "speaker_id": "SP001",
  "speaker_gender": "female",
  "speaker_age_range": "25-35",
  "recording_date": "2025-10-15",
  "recording_quality": "high",
  "duration": 2.45,
  "sample_rate": 22050,
  "notes": ""
}
```

#### 3.1.2 Linguistic Annotation

```json
{
  "audio_file": "SP001_0001.wav",
  "text": "Saya nak go to the mall lah",
  "tokens": [
    {
      "word": "Saya",
      "language": "ms",
      "pos": "pronoun",
      "start_time": 0.0,
      "end_time": 0.35
    },
    {
      "word": "nak",
      "language": "ms",
      "pos": "verb",
      "start_time": 0.40,
      "end_time": 0.65,
      "notes": "colloquial form of 'hendak'"
    },
    {
      "word": "go",
      "language": "en",
      "pos": "verb",
      "start_time": 0.70,
      "end_time": 0.95
    },
    {
      "word": "to",
      "language": "en",
      "pos": "preposition",
      "start_time": 1.00,
      "end_time": 1.15
    },
    {
      "word": "the",
      "language": "en",
      "pos": "article",
      "start_time": 1.20,
      "end_time": 1.35
    },
    {
      "word": "mall",
      "language": "en",
      "pos": "noun",
      "start_time": 1.40,
      "end_time": 1.80
    },
    {
      "word": "lah",
      "language": "particle",
      "particle_type": "emphatic",
      "particle_function": "assertion",
      "pos": "particle",
      "start_time": 1.85,
      "end_time": 2.15,
      "intonation": "mid-high"
    }
  ],
  "code_switching_points": [
    {"position": 2, "from_lang": "ms", "to_lang": "en"}
  ],
  "sentence_type": "statement",
  "emotion": "neutral",
  "speaking_style": "casual"
}
```

#### 3.1.3 Phoneme-Level Annotation

```json
{
  "audio_file": "SP001_0001.wav",
  "text": "lah",
  "phonemes": [
    {
      "phoneme": "l",
      "ipa": "l",
      "start_time": 1.85,
      "end_time": 1.95,
      "duration": 0.10
    },
    {
      "phoneme": "a",
      "ipa": "ɑː",
      "start_time": 1.95,
      "end_time": 2.10,
      "duration": 0.15,
      "pitch_contour": [180, 200, 220, 210],  # Hz samples
      "energy": 0.75
    },
    {
      "phoneme": "h",
      "ipa": "h",
      "start_time": 2.10,
      "end_time": 2.15,
      "duration": 0.05
    }
  ]
}
```

### 3.2 Annotation Tools

#### 3.2.1 Recommended Tools

**1. Praat**
- Purpose: Phonetic analysis, alignment
- Cost: Free
- Features: Waveform, spectrogram, pitch tracking, formants
- Learning curve: Moderate

**2. Montreal Forced Aligner (MFA)**
- Purpose: Automatic phoneme alignment
- Cost: Free (open source)
- Features: Automatic word/phoneme alignment, multiple languages
- Learning curve: Moderate
- **Highly Recommended**

**3. ELAN**
- Purpose: Linguistic annotation
- Cost: Free
- Features: Multi-tier annotation, time-aligned transcription
- Learning curve: Easy-Moderate

**4. Label Studio**
- Purpose: Custom annotation interface
- Cost: Free (open source)
- Features: Web-based, customizable, team collaboration
- Learning curve: Easy

#### 3.2.2 Annotation Workflow

```
┌─────────────────────────────────────────────┐
│  1. AUDIO PREPROCESSING                     │
│  • Normalize volume                         │
│  • Trim silence                             │
│  • Resample to 22050 Hz                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  2. TRANSCRIPTION                           │
│  • Manual typing (if not from script)      │
│  • ASR-assisted (Whisper)                   │
│  • Quality check                            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  3. LANGUAGE TAGGING                        │
│  • Automatic detection (first pass)         │
│  • Manual correction                        │
│  • Particle identification                  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  4. FORCED ALIGNMENT                        │
│  • Montreal Forced Aligner                  │
│  • Generate phoneme boundaries              │
│  • Manual correction (if needed)            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  5. PROSODY ANNOTATION                      │
│  • Extract pitch (Praat/PyWorld)            │
│  • Extract energy                           │
│  • Mark stress/emphasis                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  6. QUALITY CONTROL                         │
│  • Validation script                        │
│  • Manual review (10% sample)               │
│  • Fix issues                               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  7. EXPORT                                  │
│  • Generate training format                 │
│  • Create metadata files                    │
│  • Dataset statistics                       │
└─────────────────────────────────────────────┘
```

### 3.3 Montreal Forced Aligner Setup

#### 3.3.1 Installation

```bash
# Create conda environment
conda create -n aligner -c conda-forge montreal-forced-aligner

# Activate environment
conda activate aligner

# Verify installation
mfa version
```

#### 3.3.2 Preparing Data for MFA

**Directory Structure:**
```
data/
├── corpus/
│   ├── speaker01/
│   │   ├── audio001.wav
│   │   ├── audio001.txt
│   │   ├── audio002.wav
│   │   ├── audio002.txt
│   │   └── ...
│   ├── speaker02/
│   │   └── ...
│   └── ...
```

**Text File Format (audio001.txt):**
```
Saya nak go to the mall lah
```

#### 3.3.3 Running MFA

```bash
# Download pretrained models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Or train your own
# Step 1: Validate corpus
mfa validate corpus/ dictionary.txt --clean

# Step 2: Train acoustic model
mfa train corpus/ dictionary.txt output/

# Step 3: Align
mfa align corpus/ dictionary.txt acoustic_model.zip output_alignments/

# Output: TextGrid files with word and phoneme boundaries
```

#### 3.3.4 Custom Dictionary for Malaysian

```python
# scripts/generate_mfa_dictionary.py

def create_malaysian_dictionary():
    """
    Create pronunciation dictionary for MFA
    Format: word  P H O N E M E S
    """
    dictionary = {
        # Malay words
        'saya': 's a j a',
        'nak': 'n a k',
        'makan': 'm a k a n',
        'pergi': 'p ə r ɡ i',
        
        # Particles
        'lah': 'l a h',
        'leh': 'l e h',
        'loh': 'l o h',
        'lor': 'l o r',
        'meh': 'm e h',
        
        # English (Malaysian pronunciation)
        'go': 'ɡ oʊ',
        'mall': 'm ɔ l',
        'shopping': 'ʃ ɑ p ɪ ŋ',
        
        # Pinyin
        'ni': 'n i',
        'hao': 'h a o',
        'hen': 'h ə n',
    }
    
    # Write to file
    with open('malaysian_dict.txt', 'w', encoding='utf-8') as f:
        for word, phones in dictionary.items():
            f.write(f"{word}\t{phones}\n")

create_malaysian_dictionary()
```

### 3.4 Annotation Quality Control

#### 3.4.1 Validation Scripts

```python
# scripts/validate_annotations.py

import json
import librosa
from pathlib import Path

def validate_dataset(data_dir):
    """
    Validate all annotations in dataset
    """
    errors = []
    warnings = []
    
    for json_file in Path(data_dir).glob("**/*.json"):
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        
        # Check 1: Audio file exists
        audio_path = Path(data_dir) / annotation['audio_file']
        if not audio_path.exists():
            errors.append(f"Missing audio: {audio_path}")
            continue
        
        # Check 2: Duration matches
        audio, sr = librosa.load(audio_path, sr=22050)
        actual_duration = len(audio) / sr
        annotated_duration = annotation['duration']
        
        if abs(actual_duration - annotated_duration) > 0.1:
            warnings.append(
                f"{json_file}: Duration mismatch "
                f"(actual: {actual_duration:.2f}s, "
                f"annotated: {annotated_duration:.2f}s)"
            )
        
        # Check 3: Token timings are valid
        if 'tokens' in annotation:
            for i, token in enumerate(annotation['tokens']):
                if token['end_time'] <= token['start_time']:
                    errors.append(
                        f"{json_file}: Invalid token timing for '{token['word']}'"
                    )
                
                if token['end_time'] > actual_duration:
                    errors.append(
                        f"{json_file}: Token end time exceeds audio duration"
                    )
                
                # Check timing sequence
                if i > 0:
                    prev_token = annotation['tokens'][i-1]
                    if token['start_time'] < prev_token['end_time']:
                        warnings.append(
                            f"{json_file}: Token overlap between "
                            f"'{prev_token['word']}' and '{token['word']}'"
                        )
        
        # Check 4: Text matches tokens
        if 'tokens' in annotation:
            reconstructed = ' '.join([t['word'] for t in annotation['tokens']])
            if reconstructed != annotation['text']:
                warnings.append(
                    f"{json_file}: Text mismatch\n"
                    f"  Original: {annotation['text']}\n"
                    f"  Reconstructed: {reconstructed}"
                )
        
        # Check 5: Language tags are valid
        valid_langs = {'ms', 'en', 'zh', 'particle'}
        for token in annotation.get('tokens', []):
            if token['language'] not in valid_langs:
                errors.append(
                    f"{json_file}: Invalid language tag '{token['language']}'"
                )
    
    # Print report
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total files checked: {len(list(Path(data_dir).glob('**/*.json')))}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print(f"\n{'ERRORS':=^60}")
        for error in errors[:20]:  # Show first 20
            print(f"  ❌ {error}")
    
    if warnings:
        print(f"\n{'WARNINGS':=^60}")
        for warning in warnings[:20]:
            print(f"  ⚠️  {warning}")
    
    return len(errors) == 0

if __name__ == "__main__":
    validate_dataset("data/processed")
```

---

## 4. Data Preprocessing

### 4.1 Audio Preprocessing Pipeline

```python
# preprocessing/audio_preprocessing.py

import librosa
import soundfile as sf
import numpy as np
from scipy import signal

class AudioPreprocessor:
    """
    Preprocess raw audio for TTS training
    """
    def __init__(self, config):
        self.target_sr = config.sample_rate  # 22050
        self.trim_top_db = config.trim_top_db  # 40
        self.normalize_method = config.normalize_method  # 'peak' or 'lufs'
        self.target_lufs = config.target_lufs  # -20
    
    def process(self, audio_path, output_path):
        """
        Complete preprocessing pipeline
        """
        # 1. Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 2. Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # 3. Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # 4. Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=self.trim_top_db)
        
        # 5. High-pass filter (remove DC offset and low rumble)
        audio = self.high_pass_filter(audio, cutoff=80, sr=self.target_sr)
        
        # 6. Denoise (optional, light denoising)
        # audio = self.denoise(audio)
        
        # 7. Normalize volume
        audio = self.normalize(audio)
        
        # 8. Add padding (0.2s before and after)
        padding = int(0.2 * self.target_sr)
        audio = np.pad(audio, (padding, padding), mode='constant')
        
        # 9. Save
        sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
        
        return audio
    
    def high_pass_filter(self, audio, cutoff=80, sr=22050, order=5):
        """Remove low-frequency noise"""
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def normalize(self, audio):
        """Normalize audio volume"""
        if self.normalize_method == 'peak':
            # Peak normalization to -3 dB
            peak = np.abs(audio).max()
            target_peak = 10 ** (-3.0 / 20)  # -3 dB
            if peak > 0:
                audio = audio * (target_peak / peak)
        
        elif self.normalize_method == 'lufs':
            # LUFS normalization (requires pyloudnorm)
            import pyloudnorm as pyln
            meter = pyln.Meter(self.target_sr)
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, self.target_lufs)
        
        return audio
```

### 4.2 Feature Extraction

```python
# preprocessing/feature_extraction.py

import librosa
import numpy as np
import pyworld as pw

class FeatureExtractor:
    """
    Extract acoustic features for TTS training
    """
    def __init__(self, config):
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_mels = config.n_mel_channels
        self.fmin = config.mel_fmin
        self.fmax = config.mel_fmax
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel-spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return mel_db.T  # [time, n_mels]
    
    def extract_pitch(self, audio):
        """Extract pitch (F0) using PyWorld"""
        # Convert to double precision
        audio = audio.astype(np.float64)
        
        # Extract F0, spectral envelope, aperiodicity
        f0, timeaxis = pw.dio(
            audio, 
            self.sr,
            frame_period=self.hop_length / self.sr * 1000
        )
        
        # Refine F0
        f0 = pw.stonemask(audio, f0, timeaxis, self.sr)
        
        return f0
    
    def extract_energy(self, audio):
        """Extract frame-level energy"""
        # RMS energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
        
        return energy
    
    def extract_all(self, audio_path):
        """Extract all features"""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # Extract features
        mel = self.extract_mel_spectrogram(audio)
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        
        # Ensure same length (sometimes off by 1 frame)
        min_len = min(len(mel), len(pitch), len(energy))
        mel = mel[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]
        
        return {
            'mel': mel,
            'pitch': pitch,
            'energy': energy,
            'duration': len(audio) / self.sr
        }
```

### 4.3 Dataset Format

#### 4.3.1 Directory Structure

```
data/
├── processed/
│   ├── audio/
│   │   ├── SP001_0001.wav
│   │   ├── SP001_0002.wav
│   │   └── ...
│   ├── mel/
│   │   ├── SP001_0001.npy
│   │   ├── SP001_0002.npy
│   │   └── ...
│   ├── pitch/
│   │   ├── SP001_0001.npy
│   │   └── ...
│   ├── energy/
│   │   ├── SP001_0001.npy
│   │   └── ...
│   ├── alignments/
│   │   ├── SP001_0001.TextGrid
│   │   └── ...
│   └── metadata/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── annotations.json
```

#### 4.3.2 Metadata File Format

**train.txt / val.txt / test.txt:**
```
SP001_0001|Saya nak go to the mall lah|SP001|2.45
SP001_0002|Boss kata kita kena complete this project|SP001|3.12
SP002_0001|Can lah, no problem one|SP002|1.87
...
```

Format: `audio_id|text|speaker_id|duration`

**annotations.json:**
```json
{
  "SP001_0001": {
    "text": "Saya nak go to the mall lah",
    "speaker_id": "SP001",
    "duration": 2.45,
    "phonemes": ["s", "a", "j", "a", "n", "a", "k", ...],
    "phoneme_durations": [0.08, 0.12, 0.06, ...],
    "language_ids": [0, 0, 0, 0, 0, 0, 0, 1, 1, ...],
    "particle_types": [0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 1],
    "pitch_mean": 180.5,
    "pitch_std": 45.2,
    "energy_mean": 0.65,
    "code_switching": true
  }
}
```

---

## 5. Data Augmentation

### 5.1 Audio Augmentation

```python
# preprocessing/augmentation.py

import numpy as np
import librosa

class AudioAugmenter:
    """
    Audio augmentation for data diversity
    """
    def __init__(self, config):
        self.speed_range = config.speed_range  # [0.9, 1.1]
        self.pitch_range = config.pitch_range  # [-2, 2] semitones
        self.noise_snr_range = config.noise_snr_range  # [20, 40] dB
    
    def speed_perturbation(self, audio, sr, speed_factor=None):
        """Change speed without changing pitch"""
        if speed_factor is None:
            speed_factor = np.random.uniform(*self.speed_range)
        
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def pitch_shift(self, audio, sr, n_steps=None):
        """Shift pitch"""
        if n_steps is None:
            n_steps = np.random.uniform(*self.pitch_range)
        
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def add_noise(self, audio, noise_audio, snr_db=None):
        """Add background noise at specified SNR"""
        if snr_db is None:
            snr_db = np.random.uniform(*self.noise_snr_range)
        
        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # Calculate required noise scaling
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        # Add noise
        noisy_audio = audio + noise_scale * noise_audio[:len(audio)]
        
        return noisy_audio
    
    def augment(self, audio, sr):
        """Apply random augmentation"""
        # Random choice of augmentation(s)
        augmentations = []
        
        if np.random.rand() < 0.5:  # 50% chance
            augmentations.append('speed')
        
        if np.random.rand() < 0.3:  # 30% chance
            augmentations.append('pitch')
        
        if np.random.rand() < 0.2:  # 20% chance
            augmentations.append('noise')
        
        # Apply augmentations
        augmented = audio.copy()
        
        if 'speed' in augmentations:
            augmented = self.speed_perturbation(augmented, sr)
        
        if 'pitch' in augmentations:
            augmented = self.pitch_shift(augmented, sr)
        
        if 'noise' in augmentations and hasattr(self, 'noise_samples'):
            noise = np.random.choice(self.noise_samples)
            augmented = self.add_noise(augmented, noise)
        
        return augmented
```

### 5.2 Text Augmentation

```python
# preprocessing/text_augmentation.py

class TextAugmenter:
    """
    Generate synthetic code-switched sentences
    """
    def __init__(self):
        self.malay_words = self.load_malay_vocabulary()
        self.english_words = self.load_english_vocabulary()
        self.particles = ['lah', 'leh', 'loh', 'lor', 'meh', 'mah']
    
    def generate_code_switched_sentence(self, template=None):
        """
        Generate code-switched sentence
        
        Templates:
        - MS-EN: Malay beginning, English end
        - EN-MS: English beginning, Malay end
        - MS-EN-MS: Malay-English-Malay sandwich
        """
        if template is None:
            template = np.random.choice(['MS-EN', 'EN-MS', 'MS-EN-MS'])
        
        if template == 'MS-EN':
            sentence = (
                f"{self.random_malay_phrase()} "
                f"{self.random_english_phrase()} "
                f"{self.random_particle()}"
            )
        elif template == 'EN-MS':
            sentence = (
                f"{self.random_english_phrase()} "
                f"{self.random_malay_phrase()} "
                f"{self.random_particle()}"
            )
        elif template == 'MS-EN-MS':
            sentence = (
                f"{self.random_malay_phrase()} "
                f"{self.random_english_phrase()} "
                f"{self.random_malay_phrase()} "
                f"{self.random_particle()}"
            )
        
        return sentence.strip()
```

---

## 6. Dataset Statistics & Quality Metrics

### 6.1 Key Statistics to Track

```python
# scripts/dataset_statistics.py

def compute_dataset_statistics(dataset_path):
    """
    Compute comprehensive dataset statistics
    """
    stats = {
        'total_utterances': 0,
        'total_duration': 0.0,
        'speakers': {},
        'languages': {'ms': 0, 'en': 0, 'zh': 0},
        'code_switching': 0,
        'particles': {},
        'duration_distribution': [],
        'phoneme_coverage': set(),
        'word_count': 0
    }
    
    # ... computation logic
    
    # Print report
    print(f"""
    ====================================
    DATASET STATISTICS
    ====================================
    Total Utterances: {stats['total_utterances']}
    Total Duration: {stats['total_duration']:.2f} hours
    
    Speakers: {len(stats['speakers'])}
    - Male: {sum(1 for s in stats['speakers'].values() if s['gender'] == 'male')}
    - Female: {sum(1 for s in stats['speakers'].values() if s['gender'] == 'female')}
    
    Language Distribution:
    - Malay: {stats['languages']['ms']} utterances
    - English: {stats['languages']['en']} utterances
    - Chinese: {stats['languages']['zh']} utterances
    - Code-switched: {stats['code_switching']} utterances
    
    Particle Occurrences: {sum(stats['particles'].values())}
    Most common: {sorted(stats['particles'].items(), key=lambda x: x[1], reverse=True)[:5]}
    
    Duration Statistics:
    - Mean: {np.mean(stats['duration_distribution']):.2f}s
    - Median: {np.median(stats['duration_distribution']):.2f}s
    - Min: {np.min(stats['duration_distribution']):.2f}s
    - Max: {np.max(stats['duration_distribution']):.2f}s
    
    Phoneme Coverage: {len(stats['phoneme_coverage'])} unique phonemes
    """)
    
    return stats
```

---

## 7. Conclusion

This data preparation guide provides a comprehensive approach to collecting and processing data for Malaysian TTS. The key to success is:

1. **Quality over Quantity**: 50 hours of high-quality data beats 200 hours of poor data
2. **Diversity**: Cover all code-switching patterns, particles, and speaking styles
3. **Consistency**: Maintain consistent recording quality across all speakers
4. **Thorough Annotation**: Invest time in accurate linguistic annotation
5. **Validation**: Rigorously validate data quality at every step

**Next Steps:**
- Begin professional voice actor recruitment
- Set up recording infrastructure
- Develop annotation tools and workflows
- Start pilot recording session (5-10 hours)
- Validate pipeline before scaling

---

**Document Version:** 1.0  
**Last Updated:** October 12, 2025  
**Next Review:** As pipeline is established

