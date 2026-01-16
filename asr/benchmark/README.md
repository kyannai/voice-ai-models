# ASR Benchmark

Benchmarking tools for evaluating ASR models on Malaysian language datasets.

## Directory Structure

```
benchmark/
├── Makefile              # Build and run automation
├── README.md
├── api/                  # YTL API-based ASR tests
│   ├── requirements.txt
│   ├── ytl_api_test.py          # Batch evaluation
│   └── ytl_api_test_single.py   # Single file test
├── whisper/              # Whisper model tests
│   ├── requirements.txt
│   ├── ytl_faster_whisper_test.py     # faster-whisper
│   └── ytl_mesolitica_whisper_test.py # Mesolitica Whisper
└── parakeet/             # NVIDIA Parakeet/NeMo tests
    ├── requirements.txt
    ├── ytl_parakeet_test.py          # Batch evaluation
    └── ytl_parakeet_test_single.py   # Single file test
```

## Quick Start

### Prerequisites

- Python 3.10.x
- [uv](https://github.com/astral-sh/uv) package manager
- API keys:
  - `ILMU_API_KEY` for YTL API
  - `HF_TOKEN` for HuggingFace models

### 1. Create .env file

```bash
cat > .env << 'EOF'
ILMU_STAGING_URL=https://api.staging.ytlailabs.tech/v1
ILMU_STAGING_API_KEY=your_staging_key_here
ILMU_PRODUCTION_URL=https://api.ytlailabs.tech/v1
ILMU_PRODUCTION_API_KEY=your_production_key_here
HF_TOKEN=your_huggingface_token_here
EOF
```

### 2. Prepare Test Data

Download `YTL_testsets.tar` and place it at `test_data/YTL_testsets.tar`, then run:

```bash
make prepare-test-data
```

### 3. Setup All Environments

```bash
make setup-all
```

Or setup individually:

```bash
make setup-api
make setup-whisper
make setup-parakeet
```

### 4. Run All Benchmarks

```bash
make run-all
```

Or run individually:

```bash
make run-api MODEL=bukit-tinggi ENV=staging
make run-whisper
make run-parakeet MODEL=nvidia/parakeet-tdt-0.6b
```

## Makefile Targets

```bash
# Utilities
make prepare-test-data  # Unpack and validate test datasets

# Setup
make setup-all          # Setup all environments
make setup-api          # Setup API environment
make setup-whisper      # Setup Whisper environment
make setup-parakeet     # Setup Parakeet environment

# Run benchmarks
make run-all            # Run all benchmarks
make run-api MODEL=... ENV=staging|production  # Run YTL API benchmark
make run-whisper        # Run both Whisper benchmarks
make run-whisper-faster # Run faster-whisper only
make run-whisper-mesolitica  # Run Mesolitica Whisper only
make run-parakeet MODEL=nvidia/parakeet-tdt-0.6b

# Single file tests
make run-api-single AUDIO=/path/to/audio.wav
make run-parakeet-single AUDIO=/path/to/audio.wav

```

