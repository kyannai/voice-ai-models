#!/usr/bin/env python3
"""
Synthesizer for GLM-TTS (GLM-4-Voice)

GLM-4-Voice is a multimodal model from Zhipu AI that supports text-to-speech.
It can be used for high-quality speech synthesis with natural prosody.

Requirements:
    pip install transformers accelerate torch

Usage:
    python synthesize_glmtts.py \
        --test-dataset meso-malaya-test \
        --output-dir outputs/glmtts_eval \
        --model THUDM/glm-4-voice-9b
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from threading import Lock

import torch
import numpy as np
from tqdm import tqdm

from utils import (
    load_dataset_by_name,
    save_audio,
    save_synthesis_results,
    create_output_dir,
    calculate_rtf,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GLMTTSSynthesizer:
    """Synthesizer for GLM-4-Voice TTS model"""
    
    # Default model
    DEFAULT_MODEL = "THUDM/glm-4-voice-9b"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        """
        Initialize GLM-TTS synthesizer
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run on ('cuda', 'cpu', or 'auto')
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16', 'float32')
        """
        self.model_name = model_name
        self.lock = Lock()
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Determine dtype
        if torch_dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        
        logger.info(f"Loading GLM-TTS model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {self.torch_dtype}")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device == "cuda" else None,
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Default sample rate for GLM-4-Voice
            self.sample_rate = 22050
            
            logger.info("GLM-TTS model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers accelerate"
            )
        except Exception as e:
            logger.error(f"Failed to load GLM-TTS model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Synthesize speech from text
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Dictionary with synthesis results
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning silence")
            return {
                "audio": np.zeros(1000, dtype=np.float32),
                "synthesis_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }
        
        start_time = time.time()
        
        with self.lock:
            try:
                # Prepare input for TTS
                # GLM-4-Voice uses a specific prompt format for TTS
                tts_prompt = f"<|system|>\nYou are a text-to-speech assistant. Convert the following text to speech.\n<|user|>\n{text}\n<|assistant|>\n"
                
                inputs = self.tokenizer(
                    tts_prompt,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    # Generate audio
                    # Note: The exact API depends on the GLM-4-Voice implementation
                    # This is a general approach that may need adjustment
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        temperature=1.0,
                        return_audio=True,
                    )
                    
                    # Extract audio from outputs
                    if hasattr(outputs, 'audio'):
                        audio = outputs.audio.cpu().numpy().flatten()
                    elif isinstance(outputs, dict) and 'audio' in outputs:
                        audio = outputs['audio'].cpu().numpy().flatten()
                    elif isinstance(outputs, tuple) and len(outputs) > 1:
                        # Some models return (text_output, audio_output)
                        audio = outputs[1].cpu().numpy().flatten()
                    else:
                        # Fallback: try to decode audio tokens
                        audio_tokens = outputs[0] if isinstance(outputs, tuple) else outputs
                        if hasattr(self.model, 'decode_audio'):
                            audio = self.model.decode_audio(audio_tokens).cpu().numpy().flatten()
                        else:
                            raise ValueError("Could not extract audio from model outputs")
                
                audio = np.array(audio, dtype=np.float32)
                
                # Normalize audio
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
            except Exception as e:
                logger.error(f"Error synthesizing text: {e}")
                raise
        
        synthesis_time = time.time() - start_time
        
        # Calculate audio duration
        audio_duration = len(audio) / self.sample_rate
        rtf = calculate_rtf(synthesis_time, audio_duration)
        
        result = {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        
        # Save audio if output path provided
        if output_path:
            output_path = Path(output_path)
            save_audio(audio, output_path, sample_rate=self.sample_rate)
            result["audio_path"] = str(output_path)
        
        return result
    
    def synthesize_batch(
        self,
        texts: List[Dict],
        output_dir: Union[str, Path],
        filename_prefix: str = "tts",
    ) -> List[Dict]:
        """
        Synthesize batch of texts
        
        Args:
            texts: List of text dictionaries with 'id' and 'text' keys
            output_dir: Directory to save audio files
            filename_prefix: Prefix for audio filenames
            
        Returns:
            List of synthesis result dictionaries
        """
        output_dir = Path(output_dir)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for item in tqdm(texts, desc="Synthesizing"):
            text_id = item.get('id', len(results))
            text = item.get('text', '')
            
            # Generate output filename
            output_path = audio_dir / f"{filename_prefix}_{text_id:04d}.wav"
            
            try:
                result = self.synthesize(text, output_path)
                
                results.append({
                    "id": text_id,
                    "text": text,
                    "audio_path": str(output_path),
                    "synthesis_time": result["synthesis_time"],
                    "audio_duration": result["audio_duration"],
                    "rtf": result["rtf"],
                })
                
            except Exception as e:
                logger.error(f"Failed to synthesize sample {text_id}: {e}")
                results.append({
                    "id": text_id,
                    "text": text,
                    "error": str(e),
                })
        
        return results


class GLMTTSFallbackSynthesizer:
    """
    Fallback synthesizer using edge-tts or other alternatives
    when GLM-4-Voice is not available or too resource-intensive.
    """
    
    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        device: str = "auto",
    ):
        """
        Initialize fallback synthesizer using edge-tts
        
        Args:
            voice: Edge TTS voice name
            device: Ignored (edge-tts is CPU-only)
        """
        self.voice = voice
        self.lock = Lock()
        self.sample_rate = 24000
        
        logger.info(f"Initializing Edge-TTS fallback")
        logger.info(f"Voice: {voice}")
        
        try:
            import edge_tts
            self.edge_tts = edge_tts
            logger.info("Edge-TTS fallback ready")
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Install with: pip install edge-tts"
            )
    
    async def _synthesize_async(self, text: str, output_path: str) -> None:
        """Async synthesis using edge-tts"""
        communicate = self.edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Synthesize speech from text"""
        import asyncio
        import tempfile
        import soundfile as sf
        
        if not text or not text.strip():
            return {
                "audio": np.zeros(1000, dtype=np.float32),
                "synthesis_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }
        
        start_time = time.time()
        
        with self.lock:
            try:
                # Create temp file if no output path
                if output_path:
                    temp_path = str(output_path)
                else:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                
                # Run async synthesis
                asyncio.run(self._synthesize_async(text, temp_path))
                
                # Load audio
                audio, sr = sf.read(temp_path)
                audio = np.array(audio, dtype=np.float32)
                self.sample_rate = sr
                
                # Clean up temp file if created
                if not output_path:
                    Path(temp_path).unlink()
                
            except Exception as e:
                logger.error(f"Error synthesizing text: {e}")
                raise
        
        synthesis_time = time.time() - start_time
        audio_duration = len(audio) / self.sample_rate
        rtf = calculate_rtf(synthesis_time, audio_duration)
        
        result = {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        }
        
        if output_path:
            result["audio_path"] = str(output_path)
        
        return result
    
    def synthesize_batch(
        self,
        texts: List[Dict],
        output_dir: Union[str, Path],
        filename_prefix: str = "tts",
    ) -> List[Dict]:
        """Synthesize batch of texts"""
        output_dir = Path(output_dir)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for item in tqdm(texts, desc="Synthesizing"):
            text_id = item.get('id', len(results))
            text = item.get('text', '')
            output_path = audio_dir / f"{filename_prefix}_{text_id:04d}.wav"
            
            try:
                result = self.synthesize(text, output_path)
                results.append({
                    "id": text_id,
                    "text": text,
                    "audio_path": str(output_path),
                    "synthesis_time": result["synthesis_time"],
                    "audio_duration": result["audio_duration"],
                    "rtf": result["rtf"],
                })
            except Exception as e:
                logger.error(f"Failed to synthesize sample {text_id}: {e}")
                results.append({
                    "id": text_id,
                    "text": text,
                    "error": str(e),
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using GLM-TTS (GLM-4-Voice)"
    )
    
    parser.add_argument(
        "--model",
        default=GLMTTSSynthesizer.DEFAULT_MODEL,
        help="Model name or path"
    )
    parser.add_argument(
        "--test-dataset",
        required=True,
        help="Dataset name from registry (e.g., meso-malaya-test)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for synthesized audio"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model"
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use edge-tts fallback instead of GLM-4-Voice"
    )
    parser.add_argument(
        "--fallback-voice",
        default="zh-CN-XiaoxiaoNeural",
        help="Voice for edge-tts fallback"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to synthesize"
    )
    
    args = parser.parse_args()
    
    # Load text data
    logger.info(f"Loading dataset: {args.test_dataset}")
    texts = load_dataset_by_name(args.test_dataset, max_samples=args.max_samples)
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Create output directory
    model_name = "GLM-TTS" if not args.use_fallback else f"EdgeTTS-{args.fallback_voice}"
    output_dir = create_output_dir(
        args.output_dir,
        model_name,
        args.test_dataset,
    )
    
    # Initialize synthesizer
    if args.use_fallback:
        synthesizer = GLMTTSFallbackSynthesizer(
            voice=args.fallback_voice,
            device=args.device,
        )
    else:
        synthesizer = GLMTTSSynthesizer(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
        )
    
    # Synthesize all texts
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting synthesis on {len(texts)} samples")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*70}\n")
    
    results = synthesizer.synthesize_batch(texts, output_dir)
    
    # Save results
    save_synthesis_results(results, output_dir, model_name)
    
    # Print summary
    successful = [r for r in results if 'error' not in r]
    avg_rtf = sum(r['rtf'] for r in successful) / len(successful) if successful else 0
    total_audio = sum(r['audio_duration'] for r in successful)
    total_time = sum(r['synthesis_time'] for r in successful)
    
    logger.info(f"\n{'='*70}")
    logger.info("SYNTHESIS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    logger.info(f"Total audio duration: {total_audio:.2f}s")
    logger.info(f"Total synthesis time: {total_time:.2f}s")
    logger.info(f"Average RTF: {avg_rtf:.3f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()

