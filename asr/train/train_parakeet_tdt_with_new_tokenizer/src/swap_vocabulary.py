#!/usr/bin/env python3
"""
Swap vocabulary in a NeMo ASR model with optional weight restoration.

This script loads a model, changes its tokenizer/vocabulary, and optionally
restores the original layer weights for layers that were replaced during
the vocabulary change.

Usage:
    # Swap vocab only (weights get randomly initialized)
    python src/swap_vocabulary.py \
        --model-path /path/to/model.nemo \
        --tokenizer-dir ../common/tokenizers \
        --output-path ./models/swapped.nemo

    # Swap vocab AND restore original weights
    python src/swap_vocabulary.py \
        --model-path /path/to/model.nemo \
        --tokenizer-dir ../common/tokenizers \
        --output-path ./models/swapped.nemo \
        --restore-weights

    # Also transcribe a test audio file
    python src/swap_vocabulary.py \
        --model-path /path/to/model.nemo \
        --tokenizer-dir ../common/tokenizers \
        --output-path ./models/swapped.nemo \
        --restore-weights \
        --transcribe ./data/sample01.mp3
"""

import argparse
import logging
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def swap_vocabulary(
    model_path: str,
    tokenizer_dir: str,
    output_path: str,
    restore_weights: bool = False,
    transcribe_audio: str = None
):
    """
    Swap the vocabulary of a NeMo model.
    
    Args:
        model_path: Path to the original .nemo model
        tokenizer_dir: Directory containing tokenizer.model (or tokenizer_multilingual.model)
        output_path: Where to save the modified model
        restore_weights: If True, restore original layer weights after vocab swap
        transcribe_audio: Optional audio file to transcribe for testing
    """
    print("=" * 70)
    print("Swap Vocabulary in NeMo ASR Model")
    print("=" * 70)
    
    # Load model
    logger.info(f"\n📂 Loading model from: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    
    prev_vocab_size = model.tokenizer.vocab_size
    logger.info(f"   Original vocabulary size: {prev_vocab_size}")
    
    # Store original layer weights before vocabulary change
    logger.info("\n📝 Reserving original layer weights...")
    ori_decoder_prediction_embed = model.decoder.prediction.embed
    ori_decoder_prediction_dec_rnn = model.decoder.prediction.dec_rnn
    ori_joint_pred = model.joint.pred
    ori_joint_enc = model.joint.enc
    # The output Linear layer in the Joint Network (index 2 in the Sequential)
    ori_joint_joint_net_Linear = model.joint.joint_net[2]
    
    logger.info(f"   - decoder.prediction.embed: {ori_decoder_prediction_embed.weight.shape}")
    logger.info(f"   - decoder.prediction.dec_rnn: {type(ori_decoder_prediction_dec_rnn).__name__}")
    logger.info(f"   - joint.pred: {type(ori_joint_pred).__name__}")
    logger.info(f"   - joint.enc: {type(ori_joint_enc).__name__}")
    logger.info(f"   - joint.joint_net[2]: {ori_joint_joint_net_Linear.weight.shape}")
    
    # Find tokenizer model file
    tokenizer_dir_path = Path(tokenizer_dir)
    tokenizer_model = None
    for candidate in ['tokenizer.model', 'tokenizer_multilingual.model']:
        candidate_path = tokenizer_dir_path / candidate
        if candidate_path.exists():
            tokenizer_model = candidate_path
            break
    
    if tokenizer_model is None:
        raise FileNotFoundError(f"No tokenizer.model found in {tokenizer_dir}")
    
    logger.info(f"\n🔄 Changing vocabulary to: {tokenizer_model}")
    
    # Change vocabulary
    model.change_vocabulary(
        new_tokenizer_dir=str(tokenizer_dir_path),
        new_tokenizer_type="bpe"
    )
    
    new_vocab_size = model.tokenizer.vocab_size
    logger.info(f"   New vocabulary size: {new_vocab_size}")
    
    if restore_weights:
        logger.info("\n🔧 Restoring original layer weights...")
        
        with torch.no_grad():
            # 3.1 RNN-T Decoder Layers
            # Embedding: Copy weights for the first 1024 tokens (original vocab) and the last token (e.g., <sos>)
            logger.info("   - Restoring decoder.prediction.embed weights...")
            model.decoder.prediction.embed.weight[:prev_vocab_size] = ori_decoder_prediction_embed.weight[:prev_vocab_size]
            model.decoder.prediction.embed.weight[-1] = ori_decoder_prediction_embed.weight[-1]
            
            # Prediction RNN: Re-assign the original RNN layer
            logger.info("   - Restoring decoder.prediction.dec_rnn...")
            model.decoder.prediction.dec_rnn = ori_decoder_prediction_dec_rnn

            # 3.2 Joint Network Layers
            # Prediction/Encoder Nets: Re-assign the original layers
            logger.info("   - Restoring joint.pred...")
            model.joint.pred = ori_joint_pred 
            logger.info("   - Restoring joint.enc...")
            model.joint.enc = ori_joint_enc 

            # Joint Net Linear (Output): Copy weights/biases for original vocab ([:1024]) and special tokens ([-6:])
            # The range [-6:] accounts for 5 duration tokens + 1 padding/blank token, common in TDT models.
            logger.info("   - Restoring joint.joint_net[2] weights...")
            model.joint.joint_net[2].weight[:prev_vocab_size] = ori_joint_joint_net_Linear.weight[:prev_vocab_size]
            model.joint.joint_net[2].bias[:prev_vocab_size] = ori_joint_joint_net_Linear.bias[:prev_vocab_size]
            model.joint.joint_net[2].weight[-6:] = ori_joint_joint_net_Linear.weight[-6:]
            model.joint.joint_net[2].bias[-6:] = ori_joint_joint_net_Linear.bias[-6:]
        
        logger.info("   ✅ Original weights restored!")
    else:
        logger.info("\n⚠️  Skipping weight restoration (layers have randomly initialized weights)")
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving modified model to: {output_path}")
    model.save_to(str(output_path))
    logger.info("   ✅ Model saved!")
    
    # Transcribe test audio if requested
    if transcribe_audio:
        logger.info(f"\n🎤 Transcribing test audio: {transcribe_audio}")
        audio_path = Path(transcribe_audio)
        if not audio_path.exists():
            logger.error(f"   Audio file not found: {transcribe_audio}")
        else:
            import librosa
            
            model.eval()
            
            # Device selection: CUDA > MPS > CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
                logger.info("   Using CUDA")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                model = model.to(device)
                logger.info("   Using MPS (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("   Using CPU")
            
            try:
                # Load audio with librosa and pass tensor to avoid Lhotse dataloader issues
                audio, sr = librosa.load(str(audio_path), sr=16000)
                audio_tensor = torch.from_numpy(audio)
                
                with torch.inference_mode():
                    with torch.no_grad():
                        output = model.transcribe([audio_tensor])
                
                logger.info(f"\n📝 Transcription result:")
                
                # Handle tuple output (some models return (text, hypotheses))
                if isinstance(output, tuple):
                    output = output[0]
                
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    result = output[0]
                    if hasattr(result, 'text'):
                        logger.info(f"   {result.text}")
                    else:
                        logger.info(f"   {result}")
                else:
                    logger.info(f"   {output}")
            except Exception as e:
                logger.error(f"   Transcription failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Swap vocabulary in a NeMo ASR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Swap vocab only
  python src/swap_vocabulary.py --model-path ./v2.nemo --tokenizer-dir ../common/tokenizers --output-path ./models/swapped.nemo

  # Swap vocab and restore weights
  python src/swap_vocabulary.py --model-path ./v2.nemo --tokenizer-dir ../common/tokenizers --output-path ./models/swapped.nemo --restore-weights

  # Swap and transcribe
  python src/swap_vocabulary.py --model-path ./v2.nemo --tokenizer-dir ../common/tokenizers --output-path ./models/swapped.nemo --restore-weights --transcribe ./data/sample01.mp3
        """
    )
    
    parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to the original .nemo model"
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing the new tokenizer (tokenizer.model or tokenizer_multilingual.model)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save the modified model"
    )
    parser.add_argument(
        "--restore-weights",
        action="store_true",
        help="Restore original layer weights after vocabulary swap"
    )
    parser.add_argument(
        "--transcribe",
        default=None,
        help="Audio file to transcribe for testing (optional)"
    )
    
    args = parser.parse_args()
    
    swap_vocabulary(
        model_path=args.model_path,
        tokenizer_dir=args.tokenizer_dir,
        output_path=args.output_path,
        restore_weights=args.restore_weights,
        transcribe_audio=args.transcribe
    )


if __name__ == "__main__":
    main()
