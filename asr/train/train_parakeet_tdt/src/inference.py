#!/usr/bin/env python3
"""
Run inference on audio files using a trained model.

Supports:
- .nemo model files
- .ckpt checkpoint files (requires base model for config)

Usage:
    # With .nemo model
    python inference.py --model path/to/model.nemo --audio path/to/audio.wav
    
    # With .ckpt checkpoint (needs base model for architecture)
    python inference.py --checkpoint path/to/checkpoint.ckpt \\
                        --base-model path/to/base.nemo \\
                        --audio path/to/audio.wav
    
    # Multiple audio files
    python inference.py --model path/to/model.nemo --audio file1.wav file2.wav file3.wav
"""
import argparse
import os
import sys

import torch
import nemo.collections.asr as nemo_asr


def load_model_from_nemo(nemo_path: str):
    """Load model from .nemo file."""
    print(f"Loading model from: {nemo_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_path)
    return model


def load_model_from_checkpoint(ckpt_path: str, base_model_path: str):
    """Load model from .ckpt checkpoint file.
    
    Args:
        ckpt_path: Path to the checkpoint file (.ckpt)
        base_model_path: Path to the base model (.nemo) for architecture/config
    """
    print(f"Loading base model architecture from: {base_model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(base_model_path)
    
    print(f"Loading checkpoint weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # NeMo checkpoints store state_dict under 'state_dict' key
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model


def transcribe_audio(model, audio_paths: list[str]):
    """Transcribe audio files."""
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    results = []
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            results.append((audio_path, "[FILE NOT FOUND]"))
            continue
        
        try:
            # Transcribe
            transcription = model.transcribe([audio_path])
            
            # Handle different return types
            if hasattr(transcription[0], 'text'):
                text = transcription[0].text
            else:
                text = str(transcription[0])
            
            results.append((audio_path, text))
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            results.append((audio_path, f"[ERROR: {e}]"))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ASR inference on audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With .nemo model
  python inference.py --model model.nemo --audio test.wav
  
  # With checkpoint
  python inference.py --checkpoint ckpt.ckpt --base-model base.nemo --audio test.wav
  
  # Multiple files
  python inference.py --model model.nemo --audio *.wav
        """
    )
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", 
        type=str,
        help="Path to .nemo model file"
    )
    model_group.add_argument(
        "--checkpoint", 
        type=str,
        help="Path to .ckpt checkpoint file (requires --base-model)"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base .nemo model (required when using --checkpoint)"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        nargs='+',
        required=True,
        help="Audio file(s) to transcribe"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint and not args.base_model:
        parser.error("--base-model is required when using --checkpoint")
    
    # Load model
    if args.model:
        model = load_model_from_nemo(args.model)
    else:
        model = load_model_from_checkpoint(args.checkpoint, args.base_model)
    
    # Transcribe
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    print("=" * 60)
    
    results = transcribe_audio(model, args.audio)
    
    for audio_path, transcription in results:
        filename = os.path.basename(audio_path)
        print(f"\n📁 {filename}")
        print(f"   {transcription}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
