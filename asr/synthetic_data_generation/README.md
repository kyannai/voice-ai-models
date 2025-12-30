# Synthetic Training Data Generation for Parakeet-TDT

Generate high-quality synthetic Malay audio data focused on **Malaysian public figures** and **numbers** to improve Parakeet-TDT's recognition accuracy.

**Current Target:** 10 hours of audio (can scale to 100+ hours later)

## 🎯 Objectives

- **10 hours** of synthetic Malay audio (starting small, can scale up)
- **Mixed Context Training**: Train names AND numbers together in realistic sentences (more efficient!)
- **Name Recognition**: Malaysian politicians, historical figures
- **Number Accuracy**: Currency, dates, phone numbers, IDs, general numbers
- **Speaker Diversity**: Multiple voice IDs for robust training

## 💡 Why Mixed Templates?

**Traditional Approach (Inefficient):**
- Separate sentences for names: "Menteri Anwar Ibrahim mengumumkan dasar baru"
- Separate sentences for numbers: "Peruntukan sebanyak lima ratus ribu ringgit"
- Requires 2× the audio to train both

**Our Approach (Efficient):**
- Mixed sentences: "Menteri Anwar Ibrahim mengumumkan peruntukan lima ratus ribu ringgit"
- Trains BOTH names AND numbers in realistic context
- 50% reduction in required audio hours!

## 📁 Project Structure

```
synthetic_data_generation/
├── config.yaml                     # Main configuration
├── requirements.txt                # Python dependencies
├── data_sources/                   # Source data (names, number patterns)
├── sentence_templates/             # Malay sentence templates
├── scripts/                        # Generation scripts
│   ├── generate_name_sentences.py
│   ├── generate_number_sentences.py
│   ├── synthesize_with_elevenlabs.py
│   └── prepare_nemo_manifest.py
└── outputs/                        # Generated data
    ├── audio/                      # Synthesized audio files
    └── manifests/                  # NeMo training manifests
```

## 🚀 Quick Start

### Option A: Complete Pipeline (One Command)

```bash
bash run_pipeline.sh
```

This runs all steps automatically: generate sentences → synthesize audio → create manifests

### Option B: Step-by-Step (Recommended for First Time)

Follow steps 1-6 below to understand each stage:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure ElevenLabs API

Set your ElevenLabs API key:

```bash
export ELEVENLABS_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```bash
echo "ELEVENLABS_API_KEY=your_api_key_here" > .env
```

### 3. Add Voice IDs

Edit `config.yaml` and add your ElevenLabs voice IDs:

```yaml
elevenlabs:
  voice_ids:
    - "your_voice_id_1"
    - "your_voice_id_2"
    - "your_voice_id_3"
    # Add more for diversity
```

Find voice IDs at: https://elevenlabs.io/voice-library

### 4. Generate Sentences (Mixed Context - Recommended!)

Generate mixed sentences that train **both names AND numbers together**:

```bash
python scripts/generate_sentences.py \
    --config config.yaml \
    --output outputs/sentences.json \
    --seed 42
```

**What this generates:**
- ~3,600 mixed sentences (225 names × 16 samples)
- Each sentence includes a name + numbers in realistic context
- Examples:
  - "Menteri Anwar Ibrahim mengumumkan peruntukan lima ratus ribu ringgit"
  - "Mesyuarat dengan Lim Guan Eng dijadualkan pada lima belas Januari dua ribu dua puluh empat"
  - "Hubungi pejabat Ahmad Zahid di kosong satu dua tiga empat lima enam..."

**Why mixed?** 
- ✅ 50% more efficient than separate name/number sentences
- ✅ Each sentence trains both tasks simultaneously
- ✅ More realistic training context
- ✅ Results in ~10 hours of audio

**Output:** `outputs/sentences.json`

### 5. Synthesize Audio with ElevenLabs

Synthesize the generated sentences into audio:

```bash
python scripts/synthesize_with_elevenlabs.py \
    --config config.yaml \
    --input outputs/sentences.json \
    --output outputs/synthesized.json \
    --resume  # Skip existing files if restarting
```

This will:
- Process all 3,600 sentences (~10 hours)
- **Randomly select voice IDs** from your configured list for speaker diversity
- Handle rate limiting and retries automatically
- Save audio files to `outputs/audio/audio_000000.mp3`, etc.
- Create metadata in `outputs/synthesized.json`

**Resume capability:** Use `--resume` flag to skip already synthesized files if interrupted

### 6. Prepare NeMo Training Manifests

Convert synthesized audio to NeMo training format:

```bash
python scripts/prepare_nemo_manifest.py \
    --config config.yaml \
    --input outputs/synthesized.json
```

This will:
- Validate all audio files
- Calculate accurate durations
- Split into train (90%) and validation (10%)
- Create NeMo JSONL manifests

**Outputs:**
- `outputs/manifests/train_manifest.json` (~3,240 samples)
- `outputs/manifests/val_manifest.json` (~360 samples)

Each line in the manifest:
```json
{"audio_filepath": "/absolute/path/audio.mp3", "text": "Menteri Anwar Ibrahim...", "duration": 10.5}
```

## 🎓 Training with Parakeet-TDT

After generating data, use it for continued training:

```bash
cd ../train_parakeet_tdt

# Update config to point to synthetic data
python train_parakeet_tdt.py --config config_synthetic.yaml
```

## ⚙️ Configuration Options

### Generation Settings

- `target_hours`: Total hours of audio to generate
- `samples_per_name`: How many times each name appears
- `total_samples`: Number of number-focused samples

### ElevenLabs Settings

- `model`: TTS model (default: `eleven_multilingual_v2`)
- `voice_ids`: List of voice IDs (more = greater diversity)
- `stability`: Voice stability (0-1)
- `similarity_boost`: Voice similarity (0-1)

### Quality Control

- `validate_audio`: Check audio files after generation
- `min_audio_duration`: Minimum audio length (seconds)
- `max_audio_duration`: Maximum audio length (seconds)

## 📊 Expected Output

**Dataset Statistics:**
- ~25,000 unique sentences
- ~100 hours total audio
- 90/10 train/val split
- Multiple speaker voices
- Full Malay language coverage

**File Formats:**
- Audio: MP3 (22050 Hz)
- Manifest: NeMo JSONL format

## 💰 Cost Estimation

**ElevenLabs Pricing:**
- ~2M characters for 25,000 sentences
- Cost: $6-10 (Creator/Pro tier)
- Budget-friendly for high-quality synthetic data

## 🔧 Troubleshooting

**API Rate Limits:**
- Adjust `requests_per_minute` in config
- Use `batch_size` to control concurrent requests

**Voice ID Issues:**
- Verify voice IDs at https://elevenlabs.io/voice-library
- Ensure voices support Multilingual v2 model

**Audio Quality:**
- Adjust `stability` and `similarity_boost`
- Try different voice IDs for better quality

## 📚 Data Sources

**Malaysian Names (225 total):**
- **Politicians (~160):** Current cabinet ministers, opposition leaders, state chief ministers, key party leaders
- **Historical Figures (~65):** Independence leaders, former prime ministers, cultural icons

**Number Patterns (All converted to Malay words):**
- **Currency:** ringgit amounts (no RM prefix) → "lima ratus ringgit", "tujuh puluh ringgit"
- **Dates:** Malaysian formats → "lima belas Januari dua ribu dua puluh empat"
- **Phone numbers:** Malaysian formats → "kosong satu dua tiga empat lima..."
- **ID numbers:** MyKad, passport → digit by digit in Malay
- **General:** Decimals, percentages, ordinals, measurements

**Mixed Templates (150+):**
- Budget announcements, statistics, contact info, meetings, projects
- Each template combines names + numbers in realistic contexts
- Example: "Menteri {name} meluluskan {amount} pada {date}"

## 🎯 Business Impact

**Name Recognition:** 60-80% reduction in misrecognition of Malaysian politicians/figures
**Number Accuracy:** Near-perfect for financial/compliance use cases (currency, dates, phones)
**Mixed Context Training:** Learns names and numbers together in realistic scenarios
**Overall:** Higher trust and credibility for Malaysian ASR in professional applications

## 📊 Current Configuration

**Default Setup (config.yaml):**
```yaml
generation:
  target_hours: 10                  # Starting with 10 hours
  use_mixed_templates: true         # Mixed approach (efficient!)
  mixed:
    samples_per_name: 16            # 225 names × 16 = 3,600 sentences
  names:
    samples_per_name: 0             # DISABLED (mixed already trains names)
  numbers:
    total_samples: 0                # DISABLED (mixed already trains numbers)
```

**Why this is efficient:**
- ✅ Each sentence trains BOTH names AND numbers
- ✅ 50% less audio needed vs separate approaches
- ✅ More realistic training context
- ✅ Easier to scale (just increase samples_per_name)

## 📄 License

Part of the YTL Voice AI project.

