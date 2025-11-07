#!/usr/bin/env python3
"""
Transcription script for traditional FunASR models
Handles Paraformer, Paraformer-streaming, and other FunASR models
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Union
from threading import Lock

from tqdm import tqdm
from funasr import AutoModel

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


class ParaformerTranscriber:
    """Transcriber for traditional FunASR models (Paraformer, etc.)"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        hub: str = "ms",  # modelscope or huggingface
    ):
        """
        Initialize Paraformer/FunASR transcriber
        
        Args:
            model_name: Model name or path
            device: Device to run on ('cuda', 'cpu', or 'auto')
            hub: Model hub ('ms' for ModelScope, 'hf' for HuggingFace)
        """
        self.model_name = model_name
        self.device = device
        self.hub = hub
        self.lock = Lock()  # Thread lock for model access
        
        logger.info(f"Loading FunASR model: {model_name}")
        logger.info(f"Hub: {hub}")
        logger.info(f"Device: {device}")
        
        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load model through FunASR
        logger.info("Loading model through FunASR AutoModel...")
        self.model = AutoModel(
            model=model_name,
            device=self.device,
            hub=hub,
        )
        
        logger.info("FunASR model loaded successfully!")
    
    def transcribe(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and timing info
        """
        # Get audio duration
        import librosa
        audio_array, sr = librosa.load(str(audio_path), sr=16000)
        audio_duration = len(audio_array) / sr
        
        # Start timing
        start_time = time.time()
        
        with self.lock:  # Thread-safe model access
            try:
                # Generate transcription
                result = self.model.generate(
                    input=str(audio_path),
                    batch_size_s=300,  # Process in 300s batches
                )
                
                # Extract text from result
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "text" in result[0]:
                        transcription = result[0]["text"]
                    else:
                        transcription = str(result[0])
                elif isinstance(result, dict) and "text" in result:
                    transcription = result["text"]
                else:
                    transcription = str(result)
                
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
    parser = argparse.ArgumentParser(description="Transcribe audio using FunASR/Paraformer")
    parser.add_argument("--model", required=True, help="FunASR model name or path")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON/CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--hub", default="ms", choices=["ms", "hf"],
                       help="Model hub (ms=ModelScope, hf=HuggingFace)")
    parser.add_argument("--audio-dir", help="Base directory for audio files")
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
    transcriber = ParaformerTranscriber(
        model_name=args.model,
        device=args.device,
        hub=args.hub
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

