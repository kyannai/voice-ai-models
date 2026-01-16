#!/usr/bin/env python3
"""
Fine-tune MagpieTTS on Malaysian-TTS dataset.

Two-Phase Training Pipeline:
- Phase 1 (Language): Teaches model Malay G2P from pretrained model
- Phase 2 (Voice Clone): Fine-tunes Malay model with new speaker voices

Usage:
    # Phase 1: Language training
    python train.py --config configs/phase1_language.yaml
    
    # Phase 2: Voice cloning
    python train.py --config configs/phase2_voiceclone.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

# Use NeMo's trainer for compatibility with NeMo models
try:
    from nemo.lightning import Trainer as NeMoTrainer
    USE_NEMO_TRAINER = True
except ImportError:
    NeMoTrainer = None
    USE_NEMO_TRAINER = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_malay_tokenizer(model, g2p_dict_path: str):
    """
    Add Malay tokenizer to MagpieTTS model by replacing Spanish tokenizer entirely.
    
    We create a fresh IPATokenizer with:
    - Malay-specific IPA vocabulary (extracted from G2P dictionary)
    - Malay G2P dictionary for word-to-phoneme conversion
    
    This ensures all Malay phonemes (like ə, ŋ) are in the vocabulary.
    Training data uses language='es' to route through the Spanish slot.
    
    Args:
        model: MagpieTTSModel instance
        g2p_dict_path: Path to the Malay G2P dictionary file
    """
    from pathlib import Path
    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPATokenizer
    from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
    
    g2p_dict_path = str(Path(g2p_dict_path).absolute())
    
    if not Path(g2p_dict_path).exists():
        raise FileNotFoundError(f"G2P dictionary not found: {g2p_dict_path}")
    
    logger.info(f"Creating Malay tokenizer with fresh IPA vocab: {g2p_dict_path}")
    
    agg_tok = model.tokenizer
    
    if 'spanish_phoneme' not in agg_tok.tokenizers:
        raise RuntimeError("Spanish tokenizer not found, cannot replace")
    
    # Create Malay G2P
    malay_g2p = IpaG2p(
        phoneme_dict=g2p_dict_path,
        heteronyms=None,
        phoneme_probability=0.8,
        ignore_ambiguous_words=False,
        use_chars=True,
        use_stresses=True,
    )
    logger.info(f"  Created Malay IpaG2p with {len(malay_g2p.phoneme_dict)} entries")
    
    # Extract all unique IPA symbols from the Malay G2P dictionary
    all_phonemes = set()
    for word, pronunciations in malay_g2p.phoneme_dict.items():
        for pron in pronunciations:
            all_phonemes.update(pron)
    
    # Add common punctuation
    punct_symbols = set(".,!?;:'-\"()[]{}…–—")
    all_phonemes.update(punct_symbols)
    
    logger.info(f"  Extracted {len(all_phonemes)} unique IPA symbols for Malay")
    
    # Create fresh IPATokenizer with Malay vocabulary
    malay_tokenizer = IPATokenizer(
        g2p=malay_g2p,
        punct=True,
        apostrophe=True,
        pad_with_space=False,
    )
    
    # Verify vocab
    if hasattr(malay_tokenizer, 'tokens'):
        tok_vocab = set(malay_tokenizer.tokens)
        missing = all_phonemes - tok_vocab - {' '}
        if missing:
            logger.warning(f"  Some phonemes not in tokenizer vocab: {list(missing)[:10]}")
    
    # Replace Spanish tokenizer entirely
    agg_tok.tokenizers['spanish_phoneme'] = malay_tokenizer
    logger.info("  Replaced Spanish tokenizer with fresh Malay tokenizer")
    logger.info("  Training data should use language='es'")
    
    logger.info("Malay tokenizer added successfully (using Spanish slot)")


def reset_language_embeddings(model, tokenizer_name: str = 'spanish_phoneme'):
    """
    Reinitialize token embeddings for a specific language slot.
    
    This gives the model a "clean slate" for learning a new language
    without bias from the original language's phoneme patterns.
    
    Args:
        model: MagpieTTSModel instance
        tokenizer_name: Name of the tokenizer slot to reset (default: 'spanish_phoneme')
    """
    import torch
    
    agg_tok = model.tokenizer
    
    if tokenizer_name not in agg_tok.tokenizer_offsets:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not found. Available: {list(agg_tok.tokenizer_offsets.keys())}")
    
    offset = agg_tok.tokenizer_offsets[tokenizer_name]
    size = agg_tok.num_tokens_per_tokenizer[tokenizer_name]
    
    logger.info(f"Resetting embeddings for '{tokenizer_name}' (indices {offset}:{offset+size})")
    
    # Check if model has text_embedding layer
    if not hasattr(model, 'text_embedding'):
        raise RuntimeError("Model does not have 'text_embedding' layer")
    
    # Reinitialize embeddings with small random values
    with torch.no_grad():
        embedding_dim = model.text_embedding.weight.shape[1]
        # Use Xavier/Glorot initialization scaled for embedding
        std = (2.0 / (size + embedding_dim)) ** 0.5
        model.text_embedding.weight[offset:offset+size].normal_(0, std)
    
    logger.info(f"  Reset {size} embeddings (dim={embedding_dim}) with std={std:.4f}")
    logger.info("  Spanish language knowledge cleared - ready for Malay training")


def setup_training(config_path: str, gpus: int = 1, resume_from: str | None = None):
    """
    Setup and run training.
    
    Args:
        config_path: Path to training config YAML
        gpus: Number of GPUs to use
        resume_from: Path to checkpoint to resume from
    """
    try:
        from nemo.collections.tts.models import MagpieTTSModel
        from nemo.utils.exp_manager import exp_manager
    except ImportError:
        logger.error("NeMo is not installed. Install with: pip install nemo_toolkit[tts]")
        raise
    
    # Load config
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA device count: {num_gpus}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Handle -1 (all GPUs) or specific number
        if gpus == -1:
            gpus = num_gpus
            logger.info(f"Using all {gpus} available GPU(s)")
        else:
            logger.info(f"Using {gpus} GPU(s)")
    else:
        logger.warning("CUDA not available! Training will be slow on CPU.")
        gpus = 0
    
    # Setup trainer
    # Use NeMo's Trainer for compatibility with NeMo models
    trainer_config = config.get('trainer', {})
    
    trainer_kwargs = dict(
        devices=gpus if gpus > 0 else 1,
        accelerator='gpu' if gpus > 0 else 'cpu',
        max_epochs=trainer_config.get('max_epochs', 100),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        precision=trainer_config.get('precision', '16-mixed'),
        strategy=trainer_config.get('strategy', 'auto'),
        enable_checkpointing=False,  # exp_manager handles checkpointing
        logger=False,  # exp_manager handles logging
    )
    
    if USE_NEMO_TRAINER and NeMoTrainer is not None:
        logger.info("Using NeMo Trainer for compatibility")
        trainer = NeMoTrainer(**trainer_kwargs)
    else:
        logger.info("Using PyTorch Lightning Trainer")
        trainer = pl.Trainer(**trainer_kwargs)
    
    # Setup experiment manager
    exp_config = config.get('exp_manager', {})
    exp_manager(trainer, exp_config)
    
    # Determine training phase
    phase = config.get('phase', 'language')  # Default to language training
    logger.info(f"Training phase: {phase}")
    
    # Load model based on phase and resume settings
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = MagpieTTSModel.load_from_checkpoint(resume_from)
    elif phase == 'voiceclone':
        # Phase 2: Voice cloning - load from Malay base model
        base_model = config.get('base_model')
        if not base_model:
            raise ValueError("Phase 2 (voiceclone) requires 'base_model' path in config")
        logger.info(f"Loading Malay base model for voice cloning: {base_model}")
        model = MagpieTTSModel.restore_from(base_model)
        logger.info("Malay tokenizer already present in base model")
    else:
        # Phase 1: Language training - load from pretrained and add Malay tokenizer
        pretrained_model = config.get('pretrained_model', 'nvidia/magpie_tts_multilingual_357m')
        logger.info(f"Loading pretrained model: {pretrained_model}")
        model = MagpieTTSModel.from_pretrained(pretrained_model)
        
        # Add Malay tokenizer with G2P dictionary (Phase 1 only)
        g2p_dict = config.get('g2p_dict') or config.get('model', {}).get('g2p_dict')
        if g2p_dict:
            add_malay_tokenizer(model, g2p_dict)
            
            # Reset Spanish embeddings to give Malay a clean slate
            # This removes Spanish phoneme bias while keeping shared encoder/decoder knowledge
            reset_language_embeddings(model, 'spanish_phoneme')
        else:
            logger.warning("No G2P dictionary specified - model won't have Malay G2P support")
            logger.warning("Use --generate-g2p in prepare_data.py to create one")
    
    # Get model config from our YAML
    model_config = config.get('model', {})
    train_manifest = model_config.get('train_manifest')
    val_manifest = model_config.get('val_manifest')
    batch_size = model_config.get('batch_size', 8)
    sample_rate = model_config.get('sample_rate', 22050)
    
    logger.info(f"Train manifest: {train_manifest}")
    logger.info(f"Val manifest: {val_manifest}")
    logger.info(f"Batch size: {batch_size}")
    
    # MagpieTTS has complex internal dataset setup that doesn't work with simple configs
    # We'll create our own dataloaders and attach them directly to the model
    # (OmegaConf is already imported at top of file)
    
    from nemo.collections.tts.data.dataset import TTSDataset
    from torch.utils.data import DataLoader
    import json
    
    # Update learning rate in model config
    lr = model_config.get('learning_rate', 2e-4)
    if hasattr(model.cfg, 'optim'):
        with open_dict(model.cfg):
            model.cfg.optim.lr = lr
        logger.info(f"Learning rate: {lr}")
    
    # Create a simple TTS dataset from our manifests
    logger.info("Creating training dataloader...")
    
    # Load manifests
    with open(train_manifest, 'r') as f:
        train_samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(train_samples)} training samples")
    
    with open(val_manifest, 'r') as f:
        val_samples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(val_samples)} validation samples")
    
    # Create dataset class for MagpieTTS
    from torch.utils.data import Dataset
    import librosa
    import numpy as np
    
    # Get the model's tokenizer for text processing
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
    if tokenizer:
        logger.info(f"Using model's tokenizer: {type(tokenizer)}")
    else:
        logger.warning("Model tokenizer not found - will pass raw text")
    
    # MagpieTTS max sequence length (from model's causal mask)
    MAX_TEXT_TOKENS = 2000  # Leave some margin below 2048
    # Shorter audio = shorter text = less likely to exceed token limit
    # 10 seconds is typically ~100-200 tokens for phonemes
    MAX_DURATION_FOR_TRAINING = 15.0  # Reduced from 20s to avoid long sequences
    
    def resolve_tokenizer(tokenizer_obj, lang_value: str):
        """Resolve the correct sub-tokenizer for a language code."""
        if not hasattr(tokenizer_obj, 'tokenizers'):
            return tokenizer_obj, lang_value
        
        # Direct match by language code
        if lang_value in tokenizer_obj.tokenizers:
            return tokenizer_obj, lang_value
        
        # Map Malay/Spanish language code to Spanish tokenizer slot
        if lang_value in ('es', 'ms') and 'spanish_phoneme' in tokenizer_obj.tokenizers:
            return tokenizer_obj.tokenizers['spanish_phoneme'], 'spanish_phoneme'
        
        # Map Malay code to explicit malay_phoneme if present
        if lang_value == 'ms' and 'malay_phoneme' in tokenizer_obj.tokenizers:
            return tokenizer_obj.tokenizers['malay_phoneme'], 'malay_phoneme'
        
        return tokenizer_obj, lang_value

    def encode_text_for_language(tokenizer_obj, text_value: str, lang_value: str) -> torch.LongTensor:
        """Encode text with language-aware tokenizer when available."""
        if tokenizer_obj is None:
            return torch.LongTensor([ord(c) % 256 for c in text_value])
        
        resolved_tokenizer, resolved_lang = resolve_tokenizer(tokenizer_obj, lang_value)

        # Prefer language-aware encoding for AggregateTokenizer
        try:
            return torch.LongTensor(resolved_tokenizer.encode(text_value, language=resolved_lang))
        except TypeError:
            pass
        try:
            return torch.LongTensor(resolved_tokenizer.encode(text_value, lang=resolved_lang))
        except TypeError:
            pass
        try:
            return torch.LongTensor(resolved_tokenizer.encode(text_value, resolved_lang))
        except TypeError:
            pass
        except Exception as e:
            logger.warning(f"Tokenizer encode failed for language='{lang_value}': {e}")
        
        # Fallbacks for older tokenizer APIs
        if hasattr(resolved_tokenizer, 'text_to_ids'):
            try:
                return torch.LongTensor(resolved_tokenizer.text_to_ids(text_value, resolved_lang))
            except TypeError:
                try:
                    return torch.LongTensor(resolved_tokenizer.text_to_ids(text_value))
                except Exception as e:
                    logger.warning(f"Tokenizer text_to_ids failed: {e}")
        
        # Last-resort fallback: character-level bytes
        return torch.LongTensor([ord(c) % 256 for c in text_value])

    class MagpieTTSDataset(Dataset):
        """Dataset for MagpieTTS that produces correctly formatted batches."""
        
        def __init__(self, samples, sample_rate=22050, max_duration=MAX_DURATION_FOR_TRAINING, tokenizer=None, max_text_tokens=MAX_TEXT_TOKENS, max_text_chars=1500):
            # Filter by duration and text length
            filtered = []
            skipped_duration = 0
            skipped_text = 0
            for s in samples:
                if s.get('duration', 0) > max_duration:
                    skipped_duration += 1
                    continue
                # Also filter by raw text character length as a proxy for token count
                # Phonemes are roughly 1:1 with characters
                if len(s.get('text', '')) > max_text_chars:
                    skipped_text += 1
                    continue
                filtered.append(s)
            
            self.samples = filtered
            self.sample_rate = sample_rate
            self.tokenizer = tokenizer
            self.max_text_tokens = max_text_tokens
            
            if skipped_duration > 0 or skipped_text > 0:
                logger.info(f"  Filtered: {skipped_duration} too long (>{max_duration}s), {skipped_text} text too long (>{max_text_chars} chars)")
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load audio
            audio, sr = librosa.load(sample['audio_filepath'], sr=self.sample_rate)
            audio = torch.FloatTensor(audio)
            
            text = sample['text']
            # Default to 'es' (Spanish slot) - we replace Spanish G2P with Malay G2P
            language = sample.get('language', 'es')
            
            # Tokenize text with language-aware tokenizer (critical for Malay slot)
            text_tokens = encode_text_for_language(self.tokenizer, text, language)
            
            # Safety truncation (should rarely happen after pre-filtering)
            if len(text_tokens) > self.max_text_tokens:
                text_tokens = text_tokens[:self.max_text_tokens]
            
            return {
                'audio': audio,
                'audio_len': len(audio),
                'text': text_tokens,
                'text_len': len(text_tokens),
                'raw_text': text,
                'speaker': sample.get('speaker', 0),
                'language': language,
            }
    
    def collate_fn(batch):
        """Collate function for MagpieTTS batch format."""
        # Find max lengths
        max_audio_len = max(item['audio'].shape[0] for item in batch)
        max_text_len = max(item['text'].shape[0] for item in batch)
        
        # Pad and collect
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        speakers = []
        languages = []
        
        for item in batch:
            # Pad audio
            audio = item['audio']
            audio_pad = max_audio_len - len(audio)
            if audio_pad > 0:
                audio = torch.nn.functional.pad(audio, (0, audio_pad))
            audios.append(audio)
            audio_lens.append(item['audio_len'])
            
            # Pad text
            text = item['text']
            text_pad = max_text_len - len(text)
            if text_pad > 0:
                text = torch.nn.functional.pad(text, (0, text_pad), value=0)
            texts.append(text)
            text_lens.append(item['text_len'])
            
            speakers.append(item['speaker'])
            languages.append(item['language'])
        
        return {
            'audio': torch.stack(audios),
            'audio_lens': torch.LongTensor(audio_lens),
            'text': torch.stack(texts),
            'text_lens': torch.LongTensor(text_lens),
            'speaker': torch.LongTensor(speakers),
            'language': languages,
        }
    
    train_dataset = MagpieTTSDataset(train_samples, sample_rate=sample_rate, tokenizer=tokenizer)
    val_dataset = MagpieTTSDataset(val_samples, sample_rate=sample_rate, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # Attach dataloaders to model
    model._train_dl = train_loader
    model._validation_dl = val_loader
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model)
    
    logger.info("Training complete!")
    
    # Save final model
    output_dir = exp_config.get('exp_dir', 'experiments')
    final_model_path = Path(output_dir) / 'final_model.nemo'
    model.save_to(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MagpieTTS on Malaysian-TTS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python train.py --config configs/magpietts_malay.yaml

    # Train with multiple GPUs
    python train.py --config configs/magpietts_malay.yaml --gpus 4

    # Resume from checkpoint
    python train.py --config configs/magpietts_malay.yaml --resume checkpoints/last.ckpt
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    setup_training(
        config_path=args.config,
        gpus=args.gpus,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
