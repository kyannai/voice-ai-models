# Malaysian Voice AI - ASR & TTS Training

Training pipelines for Malay language **Automatic Speech Recognition (ASR)** and **Text-to-Speech (TTS)** models using NVIDIA NeMo framework.

## Overview

| Component | Model | Status |
|-----------|-------|--------|
| **ASR** | [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Trained |
| **ASR** | [Nemotron Speech Streaming 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | Planned |
| **TTS** | [MagpieTTS 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | In Progress |

## Project Structure

```
voice-ai/
├── asr/
│   ├── train/
│   │   ├── train_parakeet_tdt/       # Parakeet TDT fine-tuning
│   │   └── train_nemotron_asr/       # Nemotron streaming ASR
│   ├── eval/                         # Evaluation scripts
│   └── synthetic_data_generation/    # Synthetic data for ASR
│
├── tts/
│   ├── train/
│   │   └── train_magpietts_malay/    # MagpieTTS Malay fine-tuning
│   └── experiments/                  # TTS experiments
│
└── README.md
```

## Datasets

| Dataset | Used For |
|---------|----------|
| [mesolitica/Malaysian-STT-Whisper](https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper) | ASR Training |
| [mesolitica/Malaysian-TTS](https://huggingface.co/datasets/mesolitica/Malaysian-TTS) | TTS Training |

## Documentation

See README in each training directory:
- [ASR: Parakeet TDT](asr/train/train_parakeet_tdt/README.md)
- [ASR: Nemotron](asr/train/train_nemotron_asr/README.md)
- [TTS: MagpieTTS](tts/train/train_magpietts_malay/README.md)

## Acknowledgements

- NVIDIA NeMo Team
- mesolitica for Malaysian datasets
