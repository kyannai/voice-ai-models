#!/usr/bin/env python3
"""
Transcription script for NVIDIA Parakeet models (NeMo)
Supports Parakeet-TDT-0.6B and related models
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Union, Optional
from threading import Lock

import librosa
from tqdm import tqdm

from utils import (
    load_test_data,
    validate_samples,
    save_predictions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParakeetTranscriber:
    """Transcriber for NVIDIA Parakeet ASR models (NeMo framework)"""
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "auto",
        batch_size: int = 1,
    ):
        """
        Initialize Parakeet transcriber
        
        Args:
            model_name: Model name or path (e.g., "nvidia/parakeet-tdt-0.6b-v3")
            device: Device to run on ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.lock = Lock()  # Thread lock for model access
        
        logger.info(f"Loading Parakeet model: {model_name}")
        logger.info(f"Device: {device}")
        
        # Import NeMo (lazy import to avoid dependency issues)
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for Parakeet models. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        
        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Suppress NeMo's internal progress bars
        # Set logging level to reduce verbosity during transcription
        import logging as nemo_logging
        nemo_logging.getLogger('nemo_logger').setLevel(nemo_logging.WARNING)
        
        # Load model through NeMo
        logger.info("Loading model through NeMo ASRModel...")
        
        # Check if model is a local .nemo file or HuggingFace model name
        model_path = Path(model_name)
        if model_path.exists() and (model_path.is_file() and model_path.suffix == '.nemo'):
            # Load from local .nemo file (fine-tuned model)
            logger.info(f"Loading from local .nemo file: {model_name}")
            self.model = ASRModel.restore_from(restore_path=str(model_path))
        elif model_path.exists() and model_path.is_dir():
            # Check if directory contains a .nemo file
            nemo_files = list(model_path.glob("*.nemo"))
            if nemo_files:
                logger.info(f"Found .nemo file in directory: {nemo_files[0]}")
                self.model = ASRModel.restore_from(restore_path=str(nemo_files[0]))
            else:
                raise FileNotFoundError(f"No .nemo file found in directory: {model_path}")
        else:
            # Load from HuggingFace (pretrained model)
            logger.info(f"Loading pretrained model from HuggingFace: {model_name}")
            self.model = ASRModel.from_pretrained(model_name=model_name)
        
        # Move model to device
        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        self.model.eval()
        
        logger.info("Parakeet model loaded successfully!")
    
    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and timing info
        """
        import tempfile
        import soundfile as sf
        import os
        
        # Load audio with forced mono conversion
        audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        audio_duration = len(audio_array) / sr
        
        # Start timing
        start_time = time.time()
        
        with self.lock:  # Thread-safe model access
            try:
                # NeMo expects file paths and loads audio internally
                # If audio is stereo, we need to convert it first
                # Create a temporary mono file for NeMo to read
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    # Write mono audio to temp file
                    sf.write(tmp_path, audio_array, sr)
                
                try:
                    # Generate transcription using temp mono file
                    # Disable NeMo's progress bar for each file
                    old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
                    os.environ['TQDM_DISABLE'] = '1'
                    
                    try:
                        result = self.model.transcribe([tmp_path])
                    finally:
                        # Restore previous TQDM setting
                        if old_tqdm_disable is None:
                            os.environ.pop('TQDM_DISABLE', None)
                        else:
                            os.environ['TQDM_DISABLE'] = old_tqdm_disable
                    
                    # Extract text from result
                    # NeMo returns list of Hypothesis objects with .text attribute
                    if isinstance(result, list) and len(result) > 0:
                        # Check if result is Hypothesis object (has .text attribute)
                        if hasattr(result[0], 'text'):
                            transcription = result[0].text
                        else:
                            transcription = str(result[0])
                    else:
                        transcription = str(result)
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                raise
        
        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        return {
            "text": transcription.strip(),
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "rtf": rtf,
        }


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using NVIDIA Parakeet models")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", 
                       help="Parakeet model name or path (default: nvidia/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON/CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--audio-dir", help="Base directory for audio files")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data, args.audio_dir)
    test_data = validate_samples(test_data)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(test_data):
        logger.info(f"Limiting to first {args.max_samples} samples (--max-samples)")
        test_data = test_data[:args.max_samples]
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize transcriber
    transcriber = ParakeetTranscriber(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Transcribe all samples
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting transcription on {len(test_data)} samples")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {transcriber.device}")
    logger.info(f"{'='*70}\n")
    
    predictions = []
    for idx, sample in enumerate(tqdm(test_data, desc="Transcribing")):
        try:
            result = transcriber.transcribe(sample['audio_path'])
            
            prediction = {
                "audio_path": sample['audio_path'],
                "reference": sample['reference'],
                "hypothesis": result['text'],
                "audio_duration": result['audio_duration'],
                "processing_time": result['processing_time'],
                "rtf": result['rtf'],
                "index": idx
            }
            predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Failed to transcribe sample {idx}: {e}")
            continue
    
    # Save results
    save_predictions(predictions, args.output_dir, args.model)
    logger.info(f"\n✓ Transcription completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

