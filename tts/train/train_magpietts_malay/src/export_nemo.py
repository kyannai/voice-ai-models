#!/usr/bin/env python3
"""
Export a Phase 1/2 checkpoint to a proper .nemo file with Malay tokenizer.

This creates a complete .nemo model that includes:
- The pretrained model config
- The Malay tokenizer configuration
- The fine-tuned weights
- All required artifact files (phoneme dicts, heteronyms)

Usage:
    python export_nemo.py checkpoint.ckpt -o models/malay_model.nemo
    python export_nemo.py checkpoint.ckpt -o models/malay_model.nemo --g2p-dict data/g2p/ipa_malay_dict.txt
"""

import argparse
import logging
import os
import shutil
import tempfile
import tarfile
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default G2P dictionary path
DEFAULT_G2P_DICT = "data/g2p/ipa_malay_dict.txt"


def find_hf_cache_file(filename: str) -> str | None:
    """Find a file in the HuggingFace cache."""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub",
    ]
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        # Search for the file in cache
        for path in cache_dir.rglob(filename):
            if path.is_file():
                return str(path)
    return None


def collect_artifact_files(model) -> dict:
    """Collect all artifact files referenced in the model config.
    
    Returns dict mapping config_path -> actual_file_path
    """
    artifacts = {}
    
    if not hasattr(model.cfg, 'text_tokenizers'):
        return artifacts
    
    for tokenizer_name in model.cfg.text_tokenizers:
        tokenizer_cfg = model.cfg.text_tokenizers[tokenizer_name]
        
        # Check G2P phoneme_dict
        if hasattr(tokenizer_cfg, 'g2p') and hasattr(tokenizer_cfg.g2p, 'phoneme_dict'):
            dict_path = tokenizer_cfg.g2p.phoneme_dict
            if dict_path and Path(dict_path).exists():
                artifacts[f"{tokenizer_name}_phoneme_dict"] = dict_path
            elif dict_path:
                # Try to find in HF cache
                filename = Path(dict_path).name
                cached = find_hf_cache_file(filename)
                if cached:
                    artifacts[f"{tokenizer_name}_phoneme_dict"] = cached
                    logger.info(f"  Found {filename} in HF cache")
        
        # Check heteronyms
        if hasattr(tokenizer_cfg, 'g2p') and hasattr(tokenizer_cfg.g2p, 'heteronyms'):
            het_path = tokenizer_cfg.g2p.heteronyms
            if het_path and Path(het_path).exists():
                artifacts[f"{tokenizer_name}_heteronyms"] = het_path
            elif het_path:
                filename = Path(het_path).name
                cached = find_hf_cache_file(filename)
                if cached:
                    artifacts[f"{tokenizer_name}_heteronyms"] = cached
    
    return artifacts


def add_malay_tokenizer(model, g2p_dict_path: str):
    """Add Malay tokenizer to MagpieTTS model config."""
    g2p_dict_path = str(Path(g2p_dict_path).absolute())
    
    if not Path(g2p_dict_path).exists():
        raise FileNotFoundError(f"G2P dictionary not found: {g2p_dict_path}")
    
    logger.info(f"Adding Malay tokenizer with G2P: {g2p_dict_path}")
    
    with open_dict(model.cfg):
        # Clone english_phoneme tokenizer for Malay
        if hasattr(model.cfg, 'text_tokenizers') and hasattr(model.cfg.text_tokenizers, 'english_phoneme'):
            english_cfg = OmegaConf.to_container(model.cfg.text_tokenizers.english_phoneme, resolve=True)
            
            # Update G2P dictionary path for Malay
            if 'g2p' in english_cfg and 'phoneme_dict' in english_cfg['g2p']:
                english_cfg['g2p']['phoneme_dict'] = g2p_dict_path
            
            # Remove heteronyms for Malay (we don't have a Malay heteronyms file)
            if 'g2p' in english_cfg and 'heteronyms' in english_cfg['g2p']:
                english_cfg['g2p']['heteronyms'] = None
            
            model.cfg.text_tokenizers.malay_phoneme = OmegaConf.create(english_cfg)
            logger.info("  Created malay_phoneme tokenizer config")
        else:
            logger.warning("  Could not find english_phoneme tokenizer to clone")
        
        # Add language-to-tokenizer mapping
        if hasattr(model.cfg, 'lang2tokenizer'):
            model.cfg.lang2tokenizer.ms = 'malay_phoneme'
        else:
            model.cfg.lang2tokenizer = OmegaConf.create({'ms': 'malay_phoneme'})
        logger.info("  Added lang2tokenizer mapping: ms -> malay_phoneme")
    
    logger.info("Malay tokenizer added successfully")


def export_nemo(
    checkpoint_path: str,
    output_path: str,
    g2p_dict_path: str = DEFAULT_G2P_DICT,
) -> str:
    """
    Export a .ckpt checkpoint to a proper .nemo file with Malay support.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Output .nemo path
        g2p_dict_path: Path to the Malay G2P dictionary
        
    Returns:
        Path to the saved .nemo file
    """
    try:
        from nemo.collections.tts.models import MagpieTTSModel
    except ImportError:
        logger.error("NeMo is not installed. Install with: pip install 'nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@main'")
        raise
    
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Ensure output has .nemo extension
    if not str(output_path).endswith('.nemo'):
        output_path = output_path.with_suffix('.nemo')
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load pretrained model
    logger.info("Loading pretrained model...")
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m", map_location="cpu")
    
    # Step 2: Add Malay tokenizer to config
    add_malay_tokenizer(model, g2p_dict_path)
    
    # Step 3: Load fine-tuned weights
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        global_step = checkpoint.get("global_step", "unknown")
        logger.info(f"Checkpoint from epoch {epoch}, step {global_step}")
    else:
        state_dict = checkpoint
    
    logger.info(f"Loading {len(state_dict)} weights...")
    model.load_state_dict(state_dict, strict=False)
    
    # Step 4: Collect and fix artifact paths
    logger.info("Collecting artifact files...")
    artifacts = collect_artifact_files(model)
    logger.info(f"  Found {len(artifacts)} artifact files")
    
    # Step 5: Save as .nemo with proper artifact bundling
    logger.info(f"Saving to: {output_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy artifact files to temp dir and update config paths
        artifact_mapping = {}  # old_path -> new_name
        
        with open_dict(model.cfg):
            for tokenizer_name in list(model.cfg.text_tokenizers.keys()):
                tokenizer_cfg = model.cfg.text_tokenizers[tokenizer_name]
                
                # Fix phoneme_dict path
                if hasattr(tokenizer_cfg, 'g2p') and hasattr(tokenizer_cfg.g2p, 'phoneme_dict'):
                    old_path = tokenizer_cfg.g2p.phoneme_dict
                    if old_path:
                        # Find the actual file
                        if Path(old_path).exists():
                            src_path = old_path
                        else:
                            src_path = artifacts.get(f"{tokenizer_name}_phoneme_dict")
                        
                        if src_path and Path(src_path).exists():
                            new_name = f"{tokenizer_name}_phoneme_dict.txt"
                            shutil.copy(src_path, tmpdir / new_name)
                            tokenizer_cfg.g2p.phoneme_dict = new_name
                            logger.info(f"  Bundling {new_name}")
                        else:
                            logger.warning(f"  Missing phoneme_dict for {tokenizer_name}, setting to None")
                            tokenizer_cfg.g2p.phoneme_dict = None
                
                # Fix heteronyms path
                if hasattr(tokenizer_cfg, 'g2p') and hasattr(tokenizer_cfg.g2p, 'heteronyms'):
                    old_path = tokenizer_cfg.g2p.heteronyms
                    if old_path:
                        if Path(old_path).exists():
                            src_path = old_path
                        else:
                            src_path = artifacts.get(f"{tokenizer_name}_heteronyms")
                        
                        if src_path and Path(src_path).exists():
                            new_name = f"{tokenizer_name}_heteronyms.txt"
                            shutil.copy(src_path, tmpdir / new_name)
                            tokenizer_cfg.g2p.heteronyms = new_name
                            logger.info(f"  Bundling {new_name}")
                        else:
                            # Set to None if not found (optional for most languages)
                            tokenizer_cfg.g2p.heteronyms = None
        
        # Save config
        config_path = tmpdir / "model_config.yaml"
        OmegaConf.save(model.cfg, config_path)
        logger.info("  Saved model_config.yaml")
        
        # Save weights
        weights_path = tmpdir / "model_weights.ckpt"
        torch.save(model.state_dict(), weights_path)
        logger.info("  Saved model_weights.ckpt")
        
        # Create .nemo tarball
        with tarfile.open(output_path, "w:gz") as tar:
            for file_path in tmpdir.iterdir():
                tar.add(file_path, arcname=file_path.name)
        
        logger.info(f"  Created .nemo archive with {len(list(tmpdir.iterdir()))} files")
    
    logger.info("")
    logger.info(f"Successfully exported to: {output_path}")
    logger.info("")
    logger.info("To use this model for inference:")
    logger.info("  from nemo.collections.tts.models import MagpieTTSModel")
    logger.info(f"  model = MagpieTTSModel.restore_from('{output_path}')")
    logger.info("  audio = model.synthesize(text='Selamat pagi', language='ms')")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export Phase 1/2 checkpoint to .nemo with Malay tokenizer"
    )
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output .nemo path"
    )
    parser.add_argument(
        "--g2p-dict",
        default=DEFAULT_G2P_DICT,
        help=f"Path to Malay G2P dictionary (default: {DEFAULT_G2P_DICT})"
    )
    
    args = parser.parse_args()
    export_nemo(args.checkpoint, args.output, args.g2p_dict)


if __name__ == "__main__":
    main()
