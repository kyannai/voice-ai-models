#!/usr/bin/env python3
"""
Synthesize audio using ElevenLabs API
Supports multiple voice IDs with random selection for speaker diversity
Includes retry logic and rate limiting
"""

import json
import os
import time
import argparse
import yaml
import random
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

# ElevenLabs SDK (v2.x API)
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logging.warning("ElevenLabs SDK not installed. Install with: pip install elevenlabs")

# Audio duration library
try:
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning("Mutagen not installed. Install with: pip install mutagen")

# Environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ElevenLabsSynthesizer:
    """Synthesize audio using ElevenLabs with retry and rate limiting"""
    
    def __init__(self, config_path: str):
        """Initialize synthesizer with configuration"""
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("ElevenLabs SDK not installed")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key from environment
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        
        # Initialize ElevenLabs client (v2.x API)
        self.client = ElevenLabs(api_key=api_key)
        
        # Get voice configuration
        self.voice_ids = self.config['elevenlabs']['voice_ids']
        self.model = self.config['elevenlabs']['model']
        
        # Voice settings
        self.voice_settings = VoiceSettings(
            stability=self.config['elevenlabs']['stability'],
            similarity_boost=self.config['elevenlabs']['similarity_boost'],
            style=self.config['elevenlabs'].get('style', 0.0),
            use_speaker_boost=self.config['elevenlabs'].get('use_speaker_boost', True)
        )
        
        # Rate limiting
        self.requests_per_minute = self.config['elevenlabs']['requests_per_minute']
        self.request_delay = 60.0 / self.requests_per_minute
        self.last_request_time = 0
        
        # Retry settings
        self.max_retries = self.config['elevenlabs']['max_retries']
        self.retry_delay = self.config['elevenlabs']['retry_delay']
        self.timeout = self.config['elevenlabs']['timeout']
        
        # Output settings
        self.audio_dir = Path(self.config['output']['audio_dir'])
        self.audio_format = self.config['output']['audio_format']
        
        # Progress tracking
        self.checkpoint_file = None
        self.total_duration = 0.0
        
        logger.info(f"Initialized ElevenLabs synthesizer")
        logger.info(f"Model: {self.model}")
        logger.info(f"Available voices: {len(self.voice_ids)}")
        logger.info(f"Rate limit: {self.requests_per_minute} requests/minute")
        
        if not MUTAGEN_AVAILABLE:
            logger.warning("Mutagen not installed - audio duration will not be calculated")
    
    def rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def synthesize_text(self, text: str, voice_id: str, retries: int = 0) -> Optional[bytes]:
        """
        Synthesize text to audio with retry logic
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            retries: Current retry count
            
        Returns:
            Audio bytes or None if failed
        """
        try:
            # Apply rate limiting
            self.rate_limit()
            
            # Generate audio using v2.x API
            # In v2.x, use text_to_speech.convert() method
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=self.model,
                voice_settings=self.voice_settings
            )
            
            # Collect audio bytes from generator
            audio_bytes = b"".join(audio_generator)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            
            # Retry logic
            if retries < self.max_retries:
                logger.warning(f"Retrying ({retries + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (retries + 1))  # Exponential backoff
                return self.synthesize_text(text, voice_id, retries + 1)
            else:
                logger.error(f"Failed after {self.max_retries} retries")
                return None
    
    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds or None if failed
        """
        if not MUTAGEN_AVAILABLE:
            return None
        
        try:
            if audio_path.suffix.lower() == '.mp3':
                audio = MP3(audio_path)
            elif audio_path.suffix.lower() in ['.wav', '.wave']:
                audio = WAVE(audio_path)
            else:
                return None
            
            return audio.info.length
        except Exception as e:
            logger.warning(f"Could not read audio duration from {audio_path}: {e}")
            return None
    
    def save_audio(self, audio: bytes, output_path: Path) -> bool:
        """
        Save audio to file
        
        Args:
            audio: Audio bytes
            output_path: Path to save audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Write audio bytes directly to file
            with open(output_path, 'wb') as f:
                f.write(audio)
            return True
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: Path) -> List[Dict]:
        """Load checkpoint file with previously synthesized results"""
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint with {len(data['results'])} existing results")
                self.total_duration = data.get('total_duration', 0.0)
                logger.info(f"Total duration so far: {self.total_duration/60:.2f} minutes")
                return data['results']
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return []
    
    def save_checkpoint(self, results: List[Dict], checkpoint_path: Path):
        """Save checkpoint file incrementally"""
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate total duration
            total_duration = sum(r.get('duration', 0) for r in results)
            
            checkpoint_data = {
                'results': results,
                'total_samples': len(results),
                'total_duration': total_duration,
                'total_duration_minutes': total_duration / 60,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            self.total_duration = total_duration
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def synthesize_batch(
        self, 
        sentences: List[Dict],
        start_idx: int = 0,
        resume: bool = False,
        checkpoint_path: Optional[Path] = None
    ) -> List[Dict]:
        """
        Synthesize a batch of sentences
        
        Args:
            sentences: List of sentence dicts with 'text' field
            start_idx: Starting index (for resume)
            resume: Whether to skip existing files
            checkpoint_path: Path to save incremental progress
            
        Returns:
            List of results with audio paths
        """
        # Load existing checkpoint if resuming
        if checkpoint_path and resume:
            results = self.load_checkpoint(checkpoint_path)
            # Find the last processed index
            if results:
                last_idx = max(r['index'] for r in results)
                start_idx = max(start_idx, last_idx + 1)
                logger.info(f"Resuming from index {start_idx}")
        else:
            results = []
        
        failed = []
        
        logger.info(f"Starting synthesis of {len(sentences)} sentences")
        logger.info(f"Using {len(self.voice_ids)} voice IDs for diversity")
        if checkpoint_path:
            logger.info(f"Checkpoint file: {checkpoint_path}")
        
        for idx, sentence in enumerate(tqdm(sentences[start_idx:], desc="Synthesizing"), start=start_idx):
            # Generate filename
            filename = f"audio_{idx:06d}.{self.audio_format}"
            output_path = self.audio_dir / filename
            
            # Use relative path for audio_path (relative to outputs dir)
            relative_audio_path = f"audio/{filename}"
            
            # Skip if resume and file exists
            if resume and output_path.exists():
                # Check if already in results
                if not any(r['index'] == idx for r in results):
                    logger.debug(f"Adding existing file: {filename}")
                    duration = self.get_audio_duration(output_path)
                    results.append({
                        **sentence,
                        'audio_path': relative_audio_path,
                        'voice_id': 'existing',
                        'index': idx,
                        'duration': duration or 0.0
                    })
                    
                    # Save checkpoint
                    if checkpoint_path:
                        self.save_checkpoint(results, checkpoint_path)
                continue
            
            # Randomly select voice ID for diversity
            voice_id = random.choice(self.voice_ids)
            
            # Synthesize with retry
            audio = self.synthesize_text(sentence['text'], voice_id)
            
            if audio:
                # Save audio
                if self.save_audio(audio, output_path):
                    # Get audio duration
                    duration = self.get_audio_duration(output_path)
                    
                    result = {
                        **sentence,
                        'audio_path': relative_audio_path,
                        'voice_id': voice_id,
                        'index': idx,
                        'duration': duration or 0.0
                    }
                    results.append(result)
                    
                    # Save checkpoint after each successful synthesis
                    if checkpoint_path:
                        self.save_checkpoint(results, checkpoint_path)
                    
                    # Log progress
                    if duration:
                        total_duration = sum(r.get('duration', 0) for r in results)
                        logger.info(f"[{idx+1}/{len(sentences)}] Duration: {duration:.2f}s | Total: {total_duration/60:.2f} min")
                else:
                    failed.append({
                        'index': idx,
                        'text': sentence['text'],
                        'error': 'Failed to save audio'
                    })
            else:
                failed.append({
                    'index': idx,
                    'text': sentence['text'],
                    'error': 'Failed to synthesize after retries'
                })
        
        # Calculate total duration
        total_duration = sum(r.get('duration', 0) for r in results)
        
        # Log statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"Synthesis completed!")
        logger.info(f"{'='*70}")
        logger.info(f"Successful: {len(results)}/{len(sentences)}")
        logger.info(f"Failed: {len(failed)}/{len(sentences)}")
        logger.info(f"\n📊 Audio Statistics:")
        logger.info(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        if results:
            avg_duration = total_duration / len(results)
            logger.info(f"  Average duration: {avg_duration:.2f} seconds per audio")
        
        if failed:
            logger.warning(f"\n⚠️  Failed synthesizations:")
            for fail in failed[:10]:  # Show first 10
                logger.warning(f"  [{fail['index']}] {fail['text'][:50]}... - {fail['error']}")
            
            if len(failed) > 10:
                logger.warning(f"  ... and {len(failed) - 10} more failures")
            
            # Save failed items
            failed_path = self.audio_dir.parent / 'failed_synthesis.json'
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            logger.warning(f"Failed items saved to: {failed_path}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save synthesis results with audio paths and statistics"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate total duration
        total_duration = sum(r.get('duration', 0) for r in results)
        
        # Create output data with metadata
        output_data = {
            'results': results,
            'metadata': {
                'total_samples': len(results),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'average_duration_seconds': total_duration / len(results) if results else 0,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_path}")
        
        # Voice distribution statistics
        voice_counts = {}
        for result in results:
            voice_id = result.get('voice_id', 'unknown')
            voice_counts[voice_id] = voice_counts.get(voice_id, 0) + 1
        
        logger.info(f"\n🎤 Voice distribution:")
        for voice_id, count in sorted(voice_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {voice_id}: {count} samples ({count/len(results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize audio using ElevenLabs API"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Name of the generation run (uses outputs/<name>/ folder). Auto-detects from --input if not specified.'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file with generated sentences (if not specified, uses outputs/<name>/sentences.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file with synthesis results (if not specified, uses outputs/<name>/synthesized.json)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index (for resume)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip existing audio files'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to synthesize'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file for incremental saving (default: outputs/synthesis_checkpoint.json)'
    )
    
    args = parser.parse_args()
    
    # Determine paths based on --name
    if args.name:
        run_dir = Path('outputs') / args.name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        input_file = args.input if args.input else run_dir / 'sentences.json'
        output_file = args.output if args.output else run_dir / 'synthesized.json'
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / 'synthesis_checkpoint.json'
        
        logger.info(f"Using run name: {args.name}")
        logger.info(f"Run directory: {run_dir}")
    else:
        # Use specified paths or defaults
        input_file = Path(args.input) if args.input else Path('outputs/sentences.json')
        output_file = Path(args.output) if args.output else Path('outputs/synthesized.json')
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path('outputs/synthesis_checkpoint.json')
    
    # Check input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Please generate sentences first or specify correct --input path")
        return 1
    
    # Load sentences
    logger.info(f"Loading sentences from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    logger.info(f"Loaded {len(sentences)} sentences")
    
    # Limit samples if specified
    if args.max_samples:
        sentences = sentences[:args.max_samples]
        logger.info(f"Limited to {len(sentences)} samples")
    
    # Initialize synthesizer (config will use its own audio_dir if set)
    synthesizer = ElevenLabsSynthesizer(args.config)
    
    # Override audio_dir if using --name
    if args.name:
        audio_dir = run_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        synthesizer.audio_dir = audio_dir
        logger.info(f"Audio output directory: {audio_dir}")
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Synthesize
    results = synthesizer.synthesize_batch(
        sentences,
        start_idx=args.start_idx,
        resume=args.resume,
        checkpoint_path=checkpoint_path
    )
    
    # Save final results
    synthesizer.save_results(results, str(output_file))
    
    # Calculate total duration
    total_duration = sum(r.get('duration', 0) for r in results)
    
    logger.info(f"\n✅ Synthesis completed successfully!")
    logger.info(f"Audio files saved to: {synthesizer.audio_dir}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total audio generated: {total_duration/60:.2f} minutes ({total_duration:.2f} seconds)")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    return 0


if __name__ == "__main__":
    main()

