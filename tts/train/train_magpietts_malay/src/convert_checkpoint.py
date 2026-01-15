#!/usr/bin/env python3
"""Convert a .ckpt checkpoint to a loadable format.

Since NeMo's .nemo saving has issues with artifact paths from HuggingFace cache,
we save a .pt state dict that can be loaded at inference time.
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str | None = None,
) -> str:
    """Extract state dict from .ckpt and save as .pt file.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        output_path: Output .pt path (default: same as checkpoint with .pt extension)
        
    Returns:
        Path to the saved .pt file
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".pt")
    else:
        output_path = Path(output_path)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        global_step = checkpoint.get("global_step", "unknown")
        logger.info(f"Checkpoint from epoch {epoch}, step {global_step}")
    else:
        state_dict = checkpoint
    
    logger.info(f"State dict contains {len(state_dict)} keys")
    logger.info(f"Saving to: {output_path}")
    
    # Save state dict
    torch.save(state_dict, output_path)
    
    logger.info(f"Successfully saved state dict to: {output_path}")
    logger.info("")
    logger.info("To use this checkpoint for inference:")
    logger.info("  from nemo.collections.tts.models import MagpieTTSModel")
    logger.info("  model = MagpieTTSModel.from_pretrained('nvidia/magpie_tts_multilingual_357m')")
    logger.info(f"  state_dict = torch.load('{output_path}')")
    logger.info("  model.load_state_dict(state_dict, strict=False)")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert .ckpt to .pt state dict")
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument("-o", "--output", help="Output .pt path (default: same name as checkpoint)")
    
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
