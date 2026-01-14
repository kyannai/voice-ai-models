huggingface-cli download \
mesolitica/Malaysian-STT-Whisper-Stage2 \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/Malaysian-Multiturn-Chat-Assistant \
--include "*.zip" \
--exclude "voice/*.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/Malaysian-UltraChat-Speech-Multiturn-Instructions \
--include "ultrachat-speech-*.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-*.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-chat-assistant-*.zip" \
--exclude "*force-alignment.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-combine-*.zip" \
--exclude "*force-alignment.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-english-news-*.zip" \
--exclude "*force-alignment.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-malay-news-*.zip" \
--exclude "*force-alignment.zip" \
--repo-type "dataset" \
--local-dir './'

huggingface-cli download \
mesolitica/STT-Normalizer \
--include "prepare-dataset-normalizer-text-malay-news-part2-*.zip" \
--exclude "*force-alignment.zip" \
--repo-type "dataset" \
--local-dir './'


wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py

cd src

# First, list available datasets
python prepare_data.py --data-dir ../data/data --audio-base-dir ../data --output-dir ../output --list-datasets


python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
import os

data_dir = Path('../data/data')
audio_base = Path('../data')

# Get all unique audio folder prefixes
all_prefixes = set()
for pq_file in data_dir.glob('*_segments*.parquet'):
    df = pq.read_table(pq_file).to_pandas()
    prefixes = df['audio_filename'].str.split('/').str[0].unique()
    all_prefixes.update(prefixes)

print('=== Audio folders referenced in parquet files ===')
for prefix in sorted(all_prefixes):
    exists = (audio_base / prefix).exists()
    status = '✓' if exists else '✗ MISSING'
    if exists:
        count = len(list((audio_base / prefix).glob('*')))
        print(f'{status} {prefix}/ ({count} files)')
    else:
        print(f'{status} {prefix}/')
"
=== Audio folders referenced in parquet files ===
✓ Malaysian-Multiturn-Chat-Assistant/ (712711 files)
✓ Malaysian-Multiturn-Chat-Assistant-manglish/ (474310 files)
✓ malaymmlu-v2/ (39374 files)
✓ mallm-v3/ (12315 files)
✗ MISSING prepare-dataset-normalizer-text/
✗ MISSING prepare-dataset-normalizer-text-chat-assistant/
✗ MISSING prepare-dataset-normalizer-text-combine/
✗ MISSING prepare-dataset-normalizer-text-english-news/
✗ MISSING prepare-dataset-normalizer-text-malay-news/
✗ MISSING prepare-dataset-normalizer-text-malay-news-part2/
✓ prepared-hansard/ (456020 files)
✓ prepared-mixed-malaysian-instructions/ (508841 files)
✓ tatabahasa-v3/ (2481 files)
✗ MISSING ultrachat-speech/


# Test with small subset
python prepare_data.py \
  --data-dir ../data/data \
  --audio-base-dir ../data \
  --output-dir ./test_output \
  --max-samples 10000 \
  --validate-audio


# Full run
python prepare_data.py \
  --data-dir ../data/data \
  --audio-base-dir ../data \
  --output-dir ../output \
  --train-split 0.95

# or run with selected dataset
python prepare_data.py \
  --data-dir ../data/data \
  --audio-base-dir ../data \
  --output-dir ../output \
  --datasets malaysian_multiturn_chat_assistants \
             malaysian_multiturn_chat_assistants_manglish \
             malaysian_speech_instructions \
             malaysian_ultrachat \
             malaysian_qa \
             malaysia_hansard \
             mixed_malaysian_instructions \
             normalizer_text \
             text_chat_assistant \
             text_combined \
  --train-split 0.95 \
  --validate-audio



# check total number of audio 
python3 -c "
import json
train_file = '/home/kyan/voice-ai/asr/train/training_data/malaysian-stt-stage2/output/train_manifest.json'
val_file = '/home/kyan/voice-ai/asr/train/training_data/malaysian-stt-stage2/output/val_manifest.json'

train_hours = sum(json.loads(line)['duration'] for line in open(train_file)) / 3600
val_hours = sum(json.loads(line)['duration'] for line in open(val_file)) / 3600

print(f'Train: {train_hours:,.1f} hours')
print(f'Val: {val_hours:,.1f} hours')
print(f'Total: {train_hours + val_hours:,.1f} hours')
"
