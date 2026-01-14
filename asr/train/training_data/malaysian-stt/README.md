huggingface-cli download --repo-type dataset \
--local-dir './' \
--max-workers 20 \
mesolitica/Malaysian-STT-Whisper

wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py

cd src

# Test first with 10k samples
python prepare_data.py \
  --data-dir ../data/data \
  --audio-base-dir ../data \
  --output-dir ./test_output \
  --datasets malaysian_context_v2 extra \
  --max-samples 10000 \
  --validate-audio

# Full run (~5.87M samples) - output to train_parakeet_tdt/data
python prepare_data.py \
  --data-dir ../data/data \
  --audio-base-dir ../data \
  --output-dir ../output \
  --datasets malaysian_context_v2 extra \
  --train-split 0.95