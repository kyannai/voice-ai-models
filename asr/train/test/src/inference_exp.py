#!/usr/bin/env python3
"""
Experimental inference comparison for ASR models.

Compares transcription results across Whisper and Parakeet models:

Whisper Models:
1. openai/whisper-large-v3-turbo
2. mesolitica/Malaysian-whisper-large-v3-turbo-v3
3. whisper-malay-finetuned (latest LoRA checkpoint)

Parakeet Models:
4. nvidia/parakeet-tdt-0.6b-v3
5. parakeet-tdt-5k-v3.nemo (local fine-tuned)
6. parakeet-tdt-multilingual (checkpoint + language detection)

Each model is loaded only once and runs relevant configurations.

Usage:
    python inference_exp.py --audio test.wav
    python inference_exp.py --audio file1.wav file2.wav
"""

import argparse
import glob
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Try to import PEFT for LoRA model loading
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Try to import NeMo for Parakeet models
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo not available. Parakeet models will be skipped.")

# Try to import Whisper for language detection
try:
    import whisper as openai_whisper
    WHISPER_LID_AVAILABLE = True
except ImportError:
    WHISPER_LID_AVAILABLE = False

# Try to import Faster Whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available.")


# Fine-tuned model directories
WHISPER_FINETUNED_DIR = "/home/kyan/voice-ai/asr/train/train_whisper/outputs/whisper-malay"
PARAKEET_MULTILINGUAL_CHECKPOINT_DIR = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/outputs/parakeet-tdt-multilingual/parakeet-tdt-multilingual-v1/checkpoints"
PARAKEET_MULTILINGUAL_BASE = "/home/kyan/voice-ai/asr/train/models/parakeet-tdt-multilingual-init.nemo"
PARAKEET_MALAY_CHECKPOINT_DIR = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/outputs/parakeet-tdt-malay/parakeet-tdt-malay/checkpoints"
PARAKEET_MALAY_BASE = "/home/kyan/voice-ai/asr/train/models/parakeet-tdt-multilingual-init.nemo"
PARAKEET_5K_MODEL = "/home/kyan/voice-ai/asr/train/train_parakeet_tdt/models/parakeet-tdt-5k-v3.nemo"

# Original vocabulary size for Parakeet (tokens >= this are Chinese)
PARAKEET_ORIGINAL_VOCAB_SIZE = 8192


def get_latest_checkpoint(output_dir: str, pattern: str = "checkpoint-*") -> Optional[str]:
    """Find the latest checkpoint in the output directory.
    
    Args:
        output_dir: Path to training output directory
        pattern: Glob pattern for checkpoints (default: "checkpoint-*")
        
    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_pattern = os.path.join(output_dir, pattern)
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract checkpoint numbers and find the latest
    def get_checkpoint_num(path):
        # Match patterns like checkpoint-100, model--step=1000.ckpt, etc.
        match = re.search(r'[-=](\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    latest = max(checkpoints, key=get_checkpoint_num)
    return latest


# Lazy-loaded Whisper model for language detection
_whisper_lid_model = None


def detect_language(audio_path: str) -> Tuple[str, float]:
    """Detect language from audio using Whisper tiny model.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (language_code, confidence)
        language_code: 'zh' for Chinese, 'ms' for Malay, 'en' for English/other
        confidence: Float between 0 and 1
    """
    global _whisper_lid_model
    
    if not WHISPER_LID_AVAILABLE:
        return 'en', 0.5  # Default to English if Whisper not available
    
    if _whisper_lid_model is None:
        print("[LID] Loading Whisper tiny model for language detection...")
        _whisper_lid_model = openai_whisper.load_model("tiny")
        print("[LID] Whisper tiny model loaded")
    
    # Load and preprocess audio for Whisper
    audio = openai_whisper.load_audio(audio_path)
    audio = openai_whisper.pad_or_trim(audio)
    mel = openai_whisper.log_mel_spectrogram(audio).to(_whisper_lid_model.device)
    
    # Detect language
    _, probs = _whisper_lid_model.detect_language(mel)
    detected = max(probs, key=probs.get)
    confidence = probs[detected]
    
    # Map to our language codes
    if detected in ['zh', 'yue']:  # Chinese variants
        return 'zh', confidence
    elif detected in ['ms', 'id']:  # Malay/Indonesian
        return 'ms', confidence
    else:
        return 'en', confidence


def preprocess_audio(audio_path: str, target_sr: int = 16000) -> Tuple[str, bool]:
    """Preprocess audio: convert stereo to mono and resample if needed.
    
    Returns:
        Tuple of (path_to_use, is_temp_file)
    """
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    
    needs_processing = False
    
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
        needs_processing = True
    
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        needs_processing = True
    
    if needs_processing:
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, waveform, target_sr)
        return temp_file.name, True
    
    return audio_path, False


def get_faster_whisper_models() -> List[Tuple[str, str]]:
    """Get list of Faster Whisper models to compare.
    
    Returns:
        List of (model_size, display_name) tuples
    """
    if not FASTER_WHISPER_AVAILABLE:
        return []
    
    return [
        ("large-v3-turbo", "faster-whisper-large-v3-turbo"),
    ]


def get_whisper_models() -> List[Tuple[str, str, str]]:
    """Get list of Whisper models to compare.
    
    Returns:
        List of (model_path, display_name, model_type) tuples
    """
    models = [
        ("openai/whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", "whisper"),
        ("mesolitica/Malaysian-whisper-large-v3-turbo-v3", "mesolitica/Malaysian-whisper-large-v3-turbo-v3", "whisper"),
    ]
    
    # Add fine-tuned Whisper LoRA model if available
    latest_checkpoint = get_latest_checkpoint(WHISPER_FINETUNED_DIR)
    if latest_checkpoint:
        checkpoint_name = os.path.basename(latest_checkpoint)
        display_name = f"whisper-malay-finetuned ({checkpoint_name})"
        models.append((latest_checkpoint, display_name, "whisper-lora"))
        print(f"Found Whisper fine-tuned model: {latest_checkpoint}")
    else:
        print(f"Warning: No Whisper checkpoint found in {WHISPER_FINETUNED_DIR}")
    
    return models


def get_parakeet_models() -> List[Tuple[str, str, str, Optional[str]]]:
    """Get list of Parakeet models to compare.
    
    Returns:
        List of (model_path, display_name, model_type, base_model) tuples
    """
    if not NEMO_AVAILABLE:
        return []
    
    models = [
        # HuggingFace model
        ("nvidia/parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "nemo-hf", None),
    ]
    
    # Add local .nemo model
    if os.path.exists(PARAKEET_5K_MODEL):
        models.append((PARAKEET_5K_MODEL, "parakeet-tdt-5k-v3", "nemo", None))
        print(f"Found Parakeet 5k model: {PARAKEET_5K_MODEL}")
    else:
        print(f"Warning: Parakeet 5k model not found: {PARAKEET_5K_MODEL}")
    
    # Add malay checkpoint model with language detection
    latest_ckpt = get_latest_checkpoint(PARAKEET_MALAY_CHECKPOINT_DIR, "*.ckpt")
    if latest_ckpt and os.path.exists(PARAKEET_MALAY_BASE):
        ckpt_name = os.path.basename(latest_ckpt)
        step_match = re.search(r'step=(\d+)', ckpt_name)
        wer_match = re.search(r'val_wer=([0-9.]+)', ckpt_name)
        step = step_match.group(1) if step_match else "?"
        wer = wer_match.group(1) if wer_match else "?"
        display_name = f"parakeet-malay (step={step}, wer={wer})"
        models.append((latest_ckpt, display_name, "nemo-ckpt", PARAKEET_MALAY_BASE))
        print(f"Found Parakeet malay checkpoint: {latest_ckpt}")
    else:
        if not os.path.exists(PARAKEET_MALAY_BASE):
            print(f"Warning: Parakeet malay base not found: {PARAKEET_MALAY_BASE}")
        else:
            print(f"Warning: No checkpoint found in {PARAKEET_MALAY_CHECKPOINT_DIR}")
    
    # Add multilingual checkpoint model with language detection
    latest_ckpt = get_latest_checkpoint(PARAKEET_MULTILINGUAL_CHECKPOINT_DIR, "*.ckpt")
    if latest_ckpt and os.path.exists(PARAKEET_MULTILINGUAL_BASE):
        ckpt_name = os.path.basename(latest_ckpt)
        # Extract step and WER from checkpoint name like "parakeet-tdt--epoch=00-step=30750-val_wer=0.0793-last.ckpt"
        step_match = re.search(r'step=(\d+)', ckpt_name)
        wer_match = re.search(r'val_wer=([0-9.]+)', ckpt_name)
        step = step_match.group(1) if step_match else "?"
        wer = wer_match.group(1) if wer_match else "?"
        display_name = f"parakeet-multilingual (step={step}, wer={wer})"
        models.append((latest_ckpt, display_name, "nemo-ckpt", PARAKEET_MULTILINGUAL_BASE))
        print(f"Found Parakeet multilingual checkpoint: {latest_ckpt}")
    else:
        if not os.path.exists(PARAKEET_MULTILINGUAL_BASE):
            print(f"Warning: Parakeet multilingual base not found: {PARAKEET_MULTILINGUAL_BASE}")
        else:
            print(f"Warning: No checkpoint found in {PARAKEET_MULTILINGUAL_CHECKPOINT_DIR}")
    
    return models

# Target languages for restricted inference
TARGET_LANGUAGES = ["en", "ms", "zh"]

# All Whisper language codes (99 languages)
ALL_WHISPER_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
]


def get_suppress_token_ids(tokenizer, target_languages: List[str]) -> List[int]:
    """Get token IDs to suppress (all non-target languages)."""
    target_set = set(target_languages)
    suppress_ids = []
    
    for lang in ALL_WHISPER_LANGUAGES:
        if lang not in target_set:
            lang_token = f"<|{lang}|>"
            try:
                token_id = tokenizer.convert_tokens_to_ids(lang_token)
                if token_id is not None and token_id != tokenizer.unk_token_id:
                    suppress_ids.append(token_id)
            except Exception:
                pass
    
    return suppress_ids


class WhisperExperiment:
    """Whisper model for experimental comparison."""
    
    def __init__(self, model_path: str, display_name: str = None, base_model: str = "openai/whisper-large-v3-turbo"):
        self.model_path = model_path
        self.display_name = display_name or model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_type = "whisper"
        
        print(f"\n{'='*60}")
        print(f"Loading: {self.display_name}")
        print(f"Path: {model_path}")
        print(f"Device: {self.device}")
        
        # Check if this is a LoRA checkpoint (has adapter_config.json)
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_lora:
            if not PEFT_AVAILABLE:
                raise RuntimeError("PEFT is required to load LoRA checkpoints. Install with: pip install peft")
            
            print(f"Detected LoRA checkpoint, loading base model: {base_model}")
            # Load processor from checkpoint (has tokenizer files)
            self.processor = WhisperProcessor.from_pretrained(model_path)
            
            # Load base model first
            base = WhisperForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            print("LoRA adapter merged successfully!")
        else:
            # Regular model loading
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-compute suppress token IDs for target languages
        self.suppress_ids = get_suppress_token_ids(
            self.processor.tokenizer, 
            TARGET_LANGUAGES
        )
        
        print(f"Loaded successfully!")
        print(f"Target language suppression: {len(self.suppress_ids)} tokens to suppress")
        print(f"{'='*60}")
    
    def transcribe(
        self,
        audio_path: str,
        waveform=None,
        use_target_langs: bool = False,
        detected_language: str = None,
    ) -> str:
        """Transcribe audio.
        
        Args:
            audio_path: Path to audio file (used for loading if waveform not provided)
            waveform: Audio waveform (numpy array, 16kHz mono), optional
            use_target_langs: If True, restrict to target languages only
            detected_language: Not used for Whisper (uses suppress_tokens instead)
        
        Returns:
            Transcribed text
        """
        if waveform is None:
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Prepare input features
        input_features = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        
        # Build generate kwargs - transcribe only
        generate_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "task": "transcribe",
        }
        
        # Add language suppression if using target languages
        if use_target_langs and self.suppress_ids:
            existing_suppress = list(self.model.generation_config.suppress_tokens or [])
            generate_kwargs["suppress_tokens"] = existing_suppress + self.suppress_ids
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()


class FasterWhisperExperiment:
    """Faster Whisper model (CTranslate2 backend) for experimental comparison."""
    
    def __init__(self, model_size: str = "large-v3-turbo", display_name: str = None):
        self.model_size = model_size
        self.display_name = display_name or f"faster-whisper-{model_size}"
        self.model_type = "faster-whisper"
        
        print(f"\n{'='*60}")
        print(f"Loading: {self.display_name}")
        print(f"Model: {model_size}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Device: {device}, Compute: {compute_type}")
        
        self.model = FasterWhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        
        print(f"Loaded successfully!")
        print(f"{'='*60}")
    
    def transcribe(
        self,
        audio_path: str,
        waveform=None,
        use_target_langs: bool = False,
        detected_language: str = None,
    ) -> str:
        """Transcribe audio.
        
        Args:
            audio_path: Path to audio file
            waveform: Not used (faster-whisper loads from file)
            use_target_langs: Not used (faster-whisper doesn't support token suppression)
            detected_language: Not used
        
        Returns:
            Transcribed text
        """
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            task="transcribe",
        )
        
        # Collect all segments
        text_parts = [segment.text for segment in segments]
        return "".join(text_parts).strip()


class ParakeetExperiment:
    """Parakeet/NeMo model for experimental comparison."""
    
    # Bias presets for different languages
    BIAS_PRESETS = {
        'zh': +3.0,    # Chinese: boost Chinese tokens
        'ms': -3.0,    # Malay: suppress Chinese tokens
        'en': -3.0,    # English: suppress Chinese tokens
    }
    
    def __init__(
        self, 
        model_path: str, 
        display_name: str = None,
        model_type: str = "nemo",
        base_model: str = None,
        use_language_detection: bool = False,
    ):
        self.model_path = model_path
        self.display_name = display_name or model_path
        self.model_type = "parakeet"
        self.use_language_detection = use_language_detection
        self.original_vocab_size = PARAKEET_ORIGINAL_VOCAB_SIZE
        
        print(f"\n{'='*60}")
        print(f"Loading: {self.display_name}")
        print(f"Path: {model_path}")
        print(f"Type: {model_type}")
        if use_language_detection:
            print("Language detection: ENABLED")
        
        if model_type == "nemo-hf":
            # HuggingFace model
            print(f"Loading from HuggingFace...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_path)
        elif model_type == "nemo-ckpt":
            # Checkpoint with base model
            if not base_model:
                raise ValueError("base_model required for nemo-ckpt type")
            print(f"Loading base model: {base_model}")
            self.model = nemo_asr.models.ASRModel.restore_from(base_model)
            print(f"Loading checkpoint weights: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Regular .nemo file
            print(f"Loading from .nemo file...")
            self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using GPU")
        
        # Check if model has Chinese tokens
        try:
            vocab_size = self.model.joint.joint_net[-1].bias.shape[0]
            self.has_chinese = vocab_size > self.original_vocab_size
            if self.has_chinese:
                print(f"Chinese tokens: {vocab_size - self.original_vocab_size}")
        except Exception:
            self.has_chinese = False
        
        print(f"Loaded successfully!")
        print(f"{'='*60}")
    
    def _apply_language_bias(self, language: str) -> Optional[torch.Tensor]:
        """Apply token bias based on language."""
        if not self.has_chinese:
            return None
        
        bias_strength = self.BIAS_PRESETS.get(language, 0.0)
        if bias_strength == 0.0:
            return None
        
        try:
            output_layer = self.model.joint.joint_net[-1]
            original_bias = output_layer.bias.data.clone()
            output_layer.bias.data[self.original_vocab_size:] += bias_strength
            return original_bias
        except Exception:
            return None
    
    def _restore_bias(self, original_bias: Optional[torch.Tensor]):
        """Restore original bias."""
        if original_bias is not None:
            try:
                self.model.joint.joint_net[-1].bias.data = original_bias
            except Exception:
                pass
    
    def transcribe(
        self,
        audio_path: str,
        waveform=None,
        use_target_langs: bool = False,
        detected_language: str = None,
    ) -> str:
        """Transcribe audio.
        
        Args:
            audio_path: Path to audio file
            waveform: Not used (Parakeet needs file path)
            use_target_langs: If True and has language detection, apply bias
            detected_language: Pre-detected language code
        
        Returns:
            Transcribed text
        """
        # Preprocess audio
        processed_path, is_temp = preprocess_audio(audio_path)
        
        original_bias = None
        try:
            # Apply language bias if needed
            if use_target_langs and self.has_chinese:
                # Use provided language or detect
                lang = detected_language
                if lang is None and self.use_language_detection:
                    lang, conf = detect_language(processed_path)
                    # Only apply bias for non-Chinese
                    if lang != 'zh':
                        original_bias = self._apply_language_bias(lang)
                elif lang and lang != 'zh':
                    original_bias = self._apply_language_bias(lang)
            
            # Transcribe
            result = self.model.transcribe([processed_path])
            
            if hasattr(result[0], 'text'):
                text = result[0].text
            else:
                text = str(result[0])
            
            return text.strip()
        finally:
            self._restore_bias(original_bias)
            if is_temp:
                try:
                    os.unlink(processed_path)
                except Exception:
                    pass


def run_experiment(audio_paths: List[str]) -> Tuple[Dict, List[Tuple[str, str]]]:
    """Run the full experiment across all models and configurations.
    
    Args:
        audio_paths: List of audio file paths
        
    Returns:
        Tuple of (results dict, list of (display_name, model_type))
    """
    results = {}
    
    # Get all models to compare
    faster_whisper_models = get_faster_whisper_models()
    whisper_models = get_whisper_models()
    parakeet_models = get_parakeet_models()
    
    # Load models
    models = {}
    model_info = []  # (display_name, model_type)
    
    # Load Faster Whisper models
    for model_size, display_name in faster_whisper_models:
        try:
            models[display_name] = FasterWhisperExperiment(model_size, display_name)
            model_info.append((display_name, "faster-whisper"))
        except Exception as e:
            print(f"Warning: Failed to load {display_name}: {e}")
    
    # Load Whisper models
    for model_path, display_name, model_type in whisper_models:
        try:
            models[display_name] = WhisperExperiment(model_path, display_name)
            model_info.append((display_name, "whisper"))
        except Exception as e:
            print(f"Warning: Failed to load {display_name}: {e}")
    
    # Load Parakeet models
    for model_path, display_name, model_type, base_model in parakeet_models:
        try:
            # Enable language detection for multilingual model
            use_lid = "multilingual" in display_name.lower()
            models[display_name] = ParakeetExperiment(
                model_path, 
                display_name, 
                model_type,
                base_model,
                use_language_detection=use_lid
            )
            model_info.append((display_name, "parakeet"))
        except Exception as e:
            print(f"Warning: Failed to load {display_name}: {e}")
    
    # Process each audio file
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"\n⚠️  File not found: {audio_path}")
            continue
        
        filename = os.path.basename(audio_path)
        print(f"\n{'='*70}")
        print(f"📁 Processing: {filename}")
        print(f"{'='*70}")
        
        # Load audio once for Whisper models
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Detect language once for all models
        detected_lang = None
        if WHISPER_LID_AVAILABLE:
            detected_lang, conf = detect_language(audio_path)
            print(f"[LID] Detected: {detected_lang} ({conf:.1%})")
        
        results[filename] = {}
        
        # Run each model with both configurations
        for display_name, model in models.items():
            model_short = display_name.split("/")[-1] if "/" in display_name else display_name.split()[0]
            
            try:
                # Target languages only (en, ms, zh)
                result = model.transcribe(
                    audio_path=audio_path,
                    waveform=waveform,
                    use_target_langs=True,
                    detected_language=detected_lang
                )
                results[filename][model_short] = result
            except Exception as e:
                print(f"Error with {display_name}: {e}")
                results[filename][model_short] = f"[ERROR: {e}]"
        
        # Print results for this file
        print(f"\n{'─'*70}")
        print(f"Results for: {filename}")
        print(f"{'─'*70}")
        
        for display_name in models.keys():
            model_short = display_name.split("/")[-1] if "/" in display_name else display_name.split()[0]
            print(f"\n🔹 {display_name}")
            print(f"   {results[filename].get(model_short, 'N/A')}")
    
    return results, model_info


def print_comparison_table(results: Dict, model_info: List[Tuple[str, str]]):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 130)
    print("COMPARISON SUMMARY")
    print("=" * 130)
    print(f"Target Languages: {', '.join(TARGET_LANGUAGES)}")
    print("=" * 130)
    
    for filename, file_results in results.items():
        print(f"\n📁 {filename}")
        print("-" * 130)
        
        # Table header
        print(f"{'Model':<55} | {'Type':<10} | Transcription")
        print("-" * 130)
        
        for display_name, model_type in model_info:
            model_short = display_name.split("/")[-1] if "/" in display_name else display_name.split()[0]
            
            if model_short in file_results:
                print(f"{display_name:<55} | {model_type:<10} | {file_results[model_short]}")
            
            print("-" * 130)
    
    print("\n" + "=" * 130)
    print(f"Target Languages: {', '.join(TARGET_LANGUAGES)}")
    print("  Whisper: suppress non-target language tokens")
    print("  Parakeet: language bias for Chinese tokens")
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ASR models (Whisper + Parakeet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script compares multiple ASR models:

Whisper Models:
  1. openai/whisper-large-v3-turbo
  2. mesolitica/Malaysian-whisper-large-v3-turbo-v3
  3. whisper-malay-finetuned (latest LoRA checkpoint)

Parakeet Models:
  4. nvidia/parakeet-tdt-0.6b-v3
  5. parakeet-tdt-5k-v3.nemo
  6. parakeet-tdt-multilingual (latest checkpoint + language detection)

Each model runs with default and target_langs configurations.
Target languages: {', '.join(TARGET_LANGUAGES)}

Examples:
  python inference_exp.py --audio test.wav
  python inference_exp.py --audio file1.wav file2.wav file3.wav
        """
    )
    
    parser.add_argument(
        "--audio", "-a",
        type=str,
        nargs='+',
        required=True,
        help="Audio file(s) to transcribe"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    
    args = parser.parse_args()
    
    # Count models
    faster_whisper_count = len(get_faster_whisper_models())
    whisper_count = len(get_whisper_models())
    parakeet_count = len(get_parakeet_models())
    total_models = faster_whisper_count + whisper_count + parakeet_count
    
    print("\n" + "=" * 70)
    print("🧪 ASR MODEL COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Audio files: {len(args.audio)}")
    print(f"Faster Whisper models: {faster_whisper_count}")
    print(f"Whisper models: {whisper_count}")
    print(f"Parakeet models: {parakeet_count}")
    print(f"Total models: {total_models}")
    print(f"Target languages: {', '.join(TARGET_LANGUAGES)}")
    print("=" * 70)
    
    # Run experiment
    results, model_info = run_experiment(args.audio)
    
    # Print comparison table
    print_comparison_table(results, model_info)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
