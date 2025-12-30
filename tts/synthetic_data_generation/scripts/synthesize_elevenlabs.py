#!/usr/bin/env python3
"""
Synthesize audio using ElevenLabs API for TTS training data

Unlike ASR synthetic data (random voice selection), this script:
- Uses consistent voice_id per speaker
- Generates audio organized by speaker directory
- Creates XTTS-compatible output structure

Usage:
    python synthesize_elevenlabs.py --config ../config.yaml
"""

import json
import os
import time
import argparse
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ElevenLabs SDK
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

# Audio processing
try:
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ElevenLabsTTSSynthesizer:
    """Synthesize TTS training audio using ElevenLabs with per-speaker voices"""
    
    def __init__(self, config: Dict):
        """Initialize synthesizer with configuration"""
        if not ELEVENLABS_AVAILABLE:
            raise ImportError(
                "ElevenLabs SDK not installed. Install with: pip install elevenlabs"
            )
        
        self.config = config
        self.speakers = config['speakers']
        self.elevenlabs_config = config['elevenlabs']
        self.output_config = config['output']
        
        # Get API key from environment
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        
        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=api_key)
        
        # Voice settings
        self.voice_settings = VoiceSettings(
            stability=self.elevenlabs_config['stability'],
            similarity_boost=self.elevenlabs_config['similarity_boost'],
            style=self.elevenlabs_config.get('style', 0.0),
            use_speaker_boost=self.elevenlabs_config.get('use_speaker_boost', True)
        )
        
        # Model
        self.model = self.elevenlabs_config['model']
        
        # Rate limiting
        self.requests_per_minute = self.elevenlabs_config['requests_per_minute']
        self.request_delay = 60.0 / self.requests_per_minute
        self.last_request_time = 0
        
        # Retry settings
        self.max_retries = self.elevenlabs_config['max_retries']
        self.retry_delay = self.elevenlabs_config['retry_delay']
        
        # Output settings
        self.base_dir = Path(self.output_config['base_dir'])
        self.audio_format = self.output_config.get('audio_format', 'wav')
        self.target_sample_rate = self.output_config.get('sample_rate', 22050)
        
        logger.info(f"Initialized ElevenLabs TTS Synthesizer")
        logger.info(f"Model: {self.model}")
        logger.info(f"Speakers: {list(self.speakers.keys())}")
    
    def _validate_voice_ids(self) -> bool:
        """Check if voice IDs are configured (not placeholders)"""
        for speaker_id, speaker_config in self.speakers.items():
            voice_id = speaker_config.get('voice_id', '')
            if 'PLACEHOLDER' in voice_id or not voice_id:
                logger.warning(f"Speaker {speaker_id} has placeholder voice_id: {voice_id}")
                return False
        return True
    
    def rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def synthesize_text(
        self,
        text: str,
        voice_id: str,
        retries: int = 0
    ) -> Optional[bytes]:
        """
        Synthesize text to audio
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID for this speaker
            retries: Current retry count
            
        Returns:
            Audio bytes or None if failed
        """
        try:
            self.rate_limit()
            
            # Generate audio
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=self.model,
                voice_settings=self.voice_settings
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error synthesizing: {e}")
            
            if retries < self.max_retries:
                logger.warning(f"Retrying ({retries + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (retries + 1))
                return self.synthesize_text(text, voice_id, retries + 1)
            else:
                logger.error(f"Failed after {self.max_retries} retries")
                return None
    
    def save_audio(
        self,
        audio_bytes: bytes,
        output_path: Path,
        convert_to_wav: bool = True
    ) -> Optional[float]:
        """
        Save audio to file, optionally converting to WAV
        
        Args:
            audio_bytes: Audio data
            output_path: Path to save
            convert_to_wav: Whether to convert MP3 to WAV
            
        Returns:
            Duration in seconds or None if failed
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ElevenLabs returns MP3 by default
            if convert_to_wav and PYDUB_AVAILABLE:
                # Save temp MP3
                temp_mp3 = output_path.with_suffix('.mp3')
                with open(temp_mp3, 'wb') as f:
                    f.write(audio_bytes)
                
                # Convert to WAV
                audio = AudioSegment.from_mp3(temp_mp3)
                audio = audio.set_frame_rate(self.target_sample_rate)
                audio = audio.set_channels(1)  # Mono
                
                wav_path = output_path.with_suffix('.wav')
                audio.export(wav_path, format='wav')
                
                # Get duration
                duration = len(audio) / 1000.0  # ms to seconds
                
                # Remove temp MP3
                temp_mp3.unlink()
                
                return duration
            else:
                # Save as MP3
                mp3_path = output_path.with_suffix('.mp3')
                with open(mp3_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Get duration
                if MUTAGEN_AVAILABLE:
                    audio = MP3(mp3_path)
                    return audio.info.length
                return None
                
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            return None
    
    def synthesize_speaker(
        self,
        speaker_id: str,
        sentences: List[Dict],
        resume: bool = False
    ) -> List[Dict]:
        """
        Synthesize all sentences for a speaker
        
        Args:
            speaker_id: Speaker identifier
            sentences: List of sentence dicts with 'text' field
            resume: Whether to skip existing files
            
        Returns:
            List of synthesis results
        """
        speaker_config = self.speakers[speaker_id]
        voice_id = speaker_config['voice_id']
        
        if 'PLACEHOLDER' in voice_id:
            logger.error(f"Cannot synthesize {speaker_id}: placeholder voice_id")
            return []
        
        # Create speaker output directory
        speaker_dir = self.base_dir / speaker_id
        wavs_dir = speaker_dir / 'wavs'
        wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file for resuming
        checkpoint_file = speaker_dir / 'synthesis_checkpoint.json'
        
        # Load checkpoint if resuming
        results = []
        start_idx = 0
        
        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                if results:
                    start_idx = max(r['index'] for r in results) + 1
                    logger.info(f"Resuming {speaker_id} from index {start_idx}")
        
        logger.info(f"Synthesizing {len(sentences)} sentences for {speaker_id}")
        logger.info(f"Voice ID: {voice_id}")
        logger.info(f"Output: {wavs_dir}")
        
        for idx, sentence in enumerate(tqdm(sentences[start_idx:], desc=f"{speaker_id}"), start=start_idx):
            text = sentence.get('text', '')
            
            if not text:
                continue
            
            # Generate filename
            filename = f"{idx:04d}.wav"
            output_path = wavs_dir / filename
            
            # Skip if exists and resuming
            if resume and output_path.exists():
                continue
            
            # Synthesize
            audio_bytes = self.synthesize_text(text, voice_id)
            
            if audio_bytes:
                duration = self.save_audio(audio_bytes, output_path)
                
                result = {
                    'index': idx,
                    'text': text,
                    'audio_file': f"wavs/{filename}",
                    'speaker_id': speaker_id,
                    'language': sentence.get('language', 'unknown'),
                    'duration': duration or 0.0
                }
                results.append(result)
                
                # Save checkpoint
                self._save_checkpoint(checkpoint_file, results)
            else:
                logger.error(f"Failed to synthesize: {text[:50]}...")
        
        return results
    
    def _save_checkpoint(self, checkpoint_file: Path, results: List[Dict]):
        """Save checkpoint for resuming"""
        total_duration = sum(r.get('duration', 0) for r in results)
        
        checkpoint_data = {
            'results': results,
            'total_samples': len(results),
            'total_duration': total_duration,
            'total_duration_minutes': total_duration / 60,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def create_metadata_csv(self, speaker_id: str, results: List[Dict]):
        """
        Create XTTS-compatible metadata.csv for a speaker
        
        Format: audio_file|text|speaker_id
        """
        speaker_dir = self.base_dir / speaker_id
        metadata_file = speaker_dir / 'metadata.csv'
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for result in results:
                audio_file = result['audio_file']
                text = result['text'].replace('|', ' ')  # Remove pipe chars
                f.write(f"{audio_file}|{text}|{speaker_id}\n")
        
        logger.info(f"Created {metadata_file}")
    
    def synthesize_all(self, resume: bool = False) -> Dict[str, List[Dict]]:
        """
        Synthesize audio for all speakers
        
        Args:
            resume: Whether to resume from checkpoints
            
        Returns:
            Dict mapping speaker_id to list of results
        """
        # Validate voice IDs
        if not self._validate_voice_ids():
            logger.error("Please update config.yaml with valid ElevenLabs voice IDs")
            logger.error("Find voice IDs at: https://elevenlabs.io/voice-library")
            return {}
        
        all_results = {}
        
        for speaker_id in self.speakers.keys():
            # Load sentences for this speaker
            sentences_file = self.base_dir / speaker_id / 'sentences.json'
            
            if not sentences_file.exists():
                logger.warning(f"No sentences found for {speaker_id}: {sentences_file}")
                continue
            
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
            
            # Synthesize
            results = self.synthesize_speaker(speaker_id, sentences, resume=resume)
            
            if results:
                # Create metadata.csv for XTTS
                self.create_metadata_csv(speaker_id, results)
                all_results[speaker_id] = results
            
            # Print speaker statistics
            total_duration = sum(r.get('duration', 0) for r in results)
            logger.info(f"\n{speaker_id}: {len(results)} samples, {total_duration/60:.2f} minutes")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize TTS training audio using ElevenLabs"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--speaker',
        type=str,
        help='Synthesize only this speaker (optional)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint, skip existing files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without synthesizing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize synthesizer
    try:
        synthesizer = ElevenLabsTTSSynthesizer(config)
    except Exception as e:
        logger.error(f"Failed to initialize synthesizer: {e}")
        return 1
    
    # Dry run - just validate
    if args.dry_run:
        logger.info("Dry run - validating configuration...")
        
        valid = synthesizer._validate_voice_ids()
        if valid:
            logger.info("✅ All voice IDs are configured")
        else:
            logger.warning("⚠️  Some voice IDs are placeholders - update config.yaml")
        
        # Check sentences exist
        for speaker_id in config['speakers'].keys():
            sentences_file = Path(config['output']['base_dir']) / speaker_id / 'sentences.json'
            if sentences_file.exists():
                with open(sentences_file) as f:
                    sentences = json.load(f)
                logger.info(f"  {speaker_id}: {len(sentences)} sentences ready")
            else:
                logger.warning(f"  {speaker_id}: No sentences file found")
        
        return 0
    
    # Single speaker mode
    if args.speaker:
        if args.speaker not in config['speakers']:
            logger.error(f"Unknown speaker: {args.speaker}")
            return 1
        
        sentences_file = Path(config['output']['base_dir']) / args.speaker / 'sentences.json'
        if not sentences_file.exists():
            logger.error(f"Sentences not found: {sentences_file}")
            logger.error("Run generate_text.py first")
            return 1
        
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        results = synthesizer.synthesize_speaker(args.speaker, sentences, resume=args.resume)
        
        if results:
            synthesizer.create_metadata_csv(args.speaker, results)
            
            total_duration = sum(r.get('duration', 0) for r in results)
            logger.info(f"\n✅ Completed {args.speaker}:")
            logger.info(f"   Samples: {len(results)}")
            logger.info(f"   Duration: {total_duration/60:.2f} minutes")
        
        return 0
    
    # All speakers
    results = synthesizer.synthesize_all(resume=args.resume)
    
    if results:
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SYNTHESIS COMPLETE")
        logger.info("=" * 60)
        
        total_samples = 0
        total_duration = 0
        
        for speaker_id, speaker_results in results.items():
            samples = len(speaker_results)
            duration = sum(r.get('duration', 0) for r in speaker_results)
            total_samples += samples
            total_duration += duration
            
            logger.info(f"  {speaker_id}: {samples} samples, {duration/60:.2f} min")
        
        logger.info(f"\nTotal: {total_samples} samples, {total_duration/60:.2f} minutes")
        logger.info(f"Output: {config['output']['base_dir']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

