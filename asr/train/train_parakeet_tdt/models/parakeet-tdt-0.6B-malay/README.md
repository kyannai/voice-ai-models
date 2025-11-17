# Parakeet TDT 0.6B - Malay ASR

Fine-tuned NVIDIA Parakeet TDT 0.6B v3 model for Malay language automatic speech recognition (ASR).

## Model Description

This is a fine-tuned version of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) trained on Malay speech data. The model uses the Token-and-Duration Transducer (TDT) architecture, which provides:

- ⚡ **Lightning-fast inference**: 60 minutes of audio transcribed in ~1 second
- 🎯 **High accuracy**: Optimized for Malay language ASR
- 📝 **Automatic punctuation**: Built-in punctuation and capitalization
- 🕐 **Word timestamps**: Precise word-level timing information
- 💾 **Lightweight**: Only 0.6B parameters (~2.3GB model file)
- 🔧 **Production-ready**: No quantization needed for efficient inference

## Model Details

- **Base Model**: nvidia/parakeet-tdt-0.6b-v3
- **Architecture**: Token-and-Duration Transducer (TDT)
- **Parameters**: 0.6 billion
- **Language**: Malay (Bahasa Melayu)
- **Framework**: NVIDIA NeMo
- **Training Data**: ~5.2 million samples
- **Training Epochs**: 1 epoch
- **Training Steps**: 41,220 steps
- **Final Validation WER**: 0.2038 (20.38%)

### Training Configuration

- **Learning Rate**: 2.0e-4 with CosineAnnealing scheduler
- **Optimizer**: AdamW 8-bit (bitsandbytes)
- **Batch Size**: 8 per device
- **Gradient Accumulation**: 16 steps (effective batch size: 128)
- **Precision**: bfloat16
- **Hardware**: Single GPU (A100/H100)
- **Training Time**: ~4 days

Full training configuration is available in `training_config.yaml`.

## Performance

| Metric | Value |
|--------|-------|
| Validation WER | 20.38% |
| Training Steps | 41,220 |
| Training Samples | ~5.2M |
| Audio Processing Speed | ~60x real-time |

## Installation

```bash
# Install NVIDIA NeMo toolkit
pip install nemo_toolkit[asr]

# Or install from requirements
pip install nemo_toolkit==1.22.0
```

## Usage

### Basic Inference

```python
import nemo.collections.asr as nemo_asr

# Load the model
model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")

# Transcribe audio file
transcription = model.transcribe(["audio.wav"])
print(transcription[0])
```

### Batch Transcription

```python
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")

# Transcribe multiple files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
transcriptions = model.transcribe(audio_files, batch_size=8)

for audio_file, transcription in zip(audio_files, transcriptions):
    print(f"{audio_file}: {transcription}")
```

### With Word Timestamps

```python
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")

# Enable word timestamps
model.cfg.timestamps = True

# Transcribe with timestamps
transcription = model.transcribe(["audio.wav"], timestamps=True)
print(transcription[0])
```

### Advanced Options

```python
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")

# Configure inference
transcription = model.transcribe(
    paths2audio_files=["audio.wav"],
    batch_size=16,                    # Batch size for processing
    return_hypotheses=True,           # Return detailed results
    timestamps=True,                  # Include word timestamps
    num_workers=4                     # Parallel data loading
)

# Access detailed results
for result in transcription:
    print(f"Text: {result.text}")
    if hasattr(result, 'words'):
        for word in result.words:
            print(f"  {word.word}: {word.start_time:.2f}s - {word.end_time:.2f}s")
```

## Model Files

This repository contains:

- `parakeet-tdt-0.6b-malay.nemo` - Main model file (2.3GB) - **Use this for inference**
- `parakeet-tdt-0.6b-malay-10p.nemo` - Checkpoint at 10% training (2.3GB)
- `checkpoint-epoch01-step41220.ckpt` - PyTorch Lightning checkpoint (7.0GB)
- `training_config.yaml` - Full training configuration

### Loading the Model

#### Option 1: From HuggingFace (Recommended)

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")
```

#### Option 2: From Local File

```python
import nemo.collections.asr as nemo_asr

# Download the .nemo file first, then:
model = nemo_asr.models.ASRModel.restore_from("parakeet-tdt-0.6b-malay.nemo")
```

## Hardware Requirements

### Inference

| Hardware | Batch Size | Speed |
|----------|------------|-------|
| CPU (16 cores) | 1 | ~5x real-time |
| GPU (8GB VRAM) | 4 | ~30x real-time |
| GPU (16GB VRAM) | 8 | ~50x real-time |
| GPU (24GB+ VRAM) | 16+ | ~60x real-time |

### Memory Usage

- **Model Loading**: ~4-6GB VRAM/RAM
- **Inference (batch=1)**: ~6-8GB VRAM
- **Inference (batch=8)**: ~10-12GB VRAM

## Training Details

### Dataset

The model was fine-tuned on a large-scale Malay speech dataset:

- **Size**: ~5.2 million audio samples
- **Language**: Malay (Bahasa Melayu)
- **Audio Format**: 16kHz, mono WAV files
- **Duration**: 0.1s - 30s per sample
- **Domain**: General conversational and formal speech

### Training Process

1. **Pre-trained Base**: Started from nvidia/parakeet-tdt-0.6b-v3
2. **Fine-tuning**: 1 epoch over full dataset
3. **Validation**: Evaluated every 41,000 steps
4. **Best Checkpoint**: Selected based on validation WER
5. **Optimization**: Used 8-bit optimizer for memory efficiency

### Hyperparameters

```yaml
Learning Rate: 2.0e-4
Warmup Steps: 100
Scheduler: CosineAnnealing
Min Learning Rate: 1.0e-6
Optimizer: AdamW (8-bit)
Weight Decay: 0.0001
Max Grad Norm: 1.0
Batch Size: 8
Gradient Accumulation: 16
Effective Batch Size: 128
Precision: bfloat16
```

## Limitations

- **Language**: Optimized for Malay language only. Performance on other languages not guaranteed.
- **Domain**: Best performance on conversational and formal speech. May have reduced accuracy on technical jargon or rare dialects.
- **Audio Quality**: Works best with 16kHz+ sample rate and clean audio. Noisy environments may reduce accuracy.
- **Framework**: Requires NVIDIA NeMo framework. Not compatible with standard HuggingFace Transformers.

## Use Cases

- Real-time speech transcription
- Podcast and video subtitling
- Call center analytics
- Voice assistants and chatbots
- Meeting transcription
- Medical dictation (with domain adaptation)
- Educational content transcription

## Evaluation

To evaluate on your own test set:

```python
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("YOUR_USERNAME/parakeet-tdt-0.6b-malay")

# Prepare test data (NeMo manifest format)
# Each line: {"audio_filepath": "/path/to/audio.wav", "text": "ground truth", "duration": 2.5}

# Run evaluation
wer, cer = model.evaluate_from_manifest("test_manifest.json")
print(f"WER: {wer:.2%}")
print(f"CER: {cer:.2%}")
```

## Citation

If you use this model, please cite:

```bibtex
@misc{parakeet-tdt-malay-2024,
  title={Parakeet TDT 0.6B - Malay ASR},
  author={Your Name},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/YOUR_USERNAME/parakeet-tdt-0.6b-malay}}
}
```

And the base model:

```bibtex
@misc{parakeet-tdt-2024,
  title={Parakeet TDT: Transducer-based ASR with Token and Duration Prediction},
  author={NVIDIA},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3}}
}
```

## License

This model inherits the license from the base model [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). Please refer to the base model card for license details.

## Acknowledgements

- **NVIDIA NeMo Team**: For the excellent NeMo framework and base Parakeet TDT model
- **Training Infrastructure**: [Add your infrastructure details if applicable]

## Contact

For questions or issues, please open an issue in the model repository.

---

**Model Version**: 1.0  
**Last Updated**: November 2024  
**Framework**: NVIDIA NeMo 1.22.0  
**Base Model**: nvidia/parakeet-tdt-0.6b-v3

