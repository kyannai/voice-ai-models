# SEACrowd ASR MALCSC Dataset

## Overview

**Dataset Name**: SEACrowd ASR MALCSC (Malaysian Conversational Speech Corpus)  
**Language**: Malay (Malaysian)  
**Type**: Conversational speech  
**Number of Samples**: 20 audio files  
**License**: Copyright Beijing Magic Data Technology Co., Ltd.

## Dataset Structure

```
seacrowd-malcsc/
├── WAV/                    # Audio files (.wav format)
├── TXT/                    # Transcription files
├── AUDIOINFO.txt          # Audio metadata
├── SPKINFO.txt            # Speaker information
├── README.txt             # Original dataset documentation
├── seacrowd_malcsc.json   # Processed dataset (standard format)
└── prepare_dataset.py     # Conversion script
```

## Dataset Details

### Audio Files
- **Format**: WAV
- **Naming**: `GroupID_ConversationID_0_SpeakerID.wav`
- **Topics**: Health, Sports, Food, Culture, Work, Telecom, Holiday
- **Speakers**: 10 speakers (Mix of male and female)
- **Location**: Johor Bahru, Malaysia
- **Age Range**: 19-24 years old

### Speaker Information

| Speaker ID | Gender | Age | Device |
|------------|--------|-----|--------|
| G0216 | Male | 19 | iPhone 6s |
| G0369 | Female | 19 | HUAWEI DRA-LX9 |
| G0489 | Female | 19 | iPhone 6s |
| G0712 | Male | 19 | HUAWEI DRA-LX9 |
| G0397 | Female | 24 | Xiaomi Redmi Note 4 |
| G0474 | Male | 19 | HUAWEI LDN-LX2 |
| G0521 | Female | 19 | iPhone 6s |
| G0591 | Male | 19 | HUAWEI DRA-LX9 |
| G0598 | Female | 19 | HUAWEI CDY-NX9B |
| G0626 | Female | 19 | iPhone 6s |

### Conversation Topics

1. **Health** (8 files): COVID-19, Dengue fever, health issues
2. **Sports** (4 files): Football, swimming, badminton, rugby
3. **Food** (2 files): Food preferences and habits
4. **Culture** (2 files): Cultural discussions
5. **Work** (2 files): Work-related conversations
6. **Telecom** (2 files): Telecommunications topics
7. **Holiday** (2 files): Holiday discussions

## Transcription Format

The original TXT files contain timestamped transcriptions:
```
[start_time,end_time]	speaker_id	gender,location	transcript
```

### Annotations Removed in Processing
- `[*]` - Unintelligible words/long foreign language passages
- `[LAUGHTER]` - Laughter
- `[SONANT]` - Vocal system noises (cough, sneeze)
- `[MUSIC]` - Music/humming
- `[SYSTEM]` - System sounds
- `[ENS]` - Ambient noises
- `[UNK]` - Unintelligible words
- `+` - Overlapping speech markers

## Processed Dataset Format

The `seacrowd_malcsc.json` file contains samples in standard ASR evaluation format:

```json
{
  "audio_path": "WAV/A0004_S008_0_G0369.wav",
  "reference": "hai Syakirah macam mana kau sekarang...",
  "file_id": "A0004_S008_0_G0369"
}
```

## Usage

### With evaluate.py
```bash
python evaluate.py \
  --model openai/whisper-large-v3-turbo \
  --test-dataset seacrowd-asr-malcsc \
  --device auto
```

### With batch_evaluate.py
```bash
python batch_evaluate.py \
  --test-dataset seacrowd-asr-malcsc \
  --models whisper-large-v3-turbo parakeet-tdt-0.6b
```

### Direct loading (for custom scripts)
```python
from dataset_loader import load_dataset

samples = load_dataset("seacrowd-asr-malcsc")
print(f"Loaded {len(samples)} samples")
```

## Dataset Characteristics

### Conversational Nature
- **Natural speech**: Includes fillers, false starts, repetitions
- **Code-switching**: Some English-Malay mixing
- **Colloquial Malay**: Informal conversational style
- **Regional accent**: Johor Bahru dialect

### Challenges for ASR
1. **Disfluencies**: Frequent "ah", "em", "kan", repetitions
2. **Long utterances**: Average ~5000 characters per conversation
3. **Overlapping speech**: Some concurrent talking
4. **Background noise**: Some system sounds and ambient noise
5. **Informal vocabulary**: Colloquial expressions

## Statistics

- **Total samples**: 20 conversations
- **Average transcription length**: ~5000 characters
- **Shortest transcription**: 1492 characters
- **Longest transcription**: 8813 characters
- **Language**: Colloquial Malaysian Malay with some English code-switching

## Citation

If using this dataset, please cite:
- SEACrowd project
- Magic Data Technology Co., Ltd. (original corpus provider)

## Notes

- This is a TEST set - suitable for evaluation but not training
- Audio quality varies by recording device
- Conversations are natural and unscripted
- Some files contain multiple speakers (conversational format)
- Transcriptions have been cleaned of annotation markers for ASR evaluation

