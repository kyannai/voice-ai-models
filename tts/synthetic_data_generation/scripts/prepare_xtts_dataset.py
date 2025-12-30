#!/usr/bin/env python3
"""
Prepare XTTS v2 Training Dataset

Validates and prepares the synthesized audio for XTTS fine-tuning:
- Validates audio duration (5-15 seconds optimal)
- Ensures WAV format at 22050Hz mono
- Creates metadata.csv in XTTS format
- Generates train/val split
- Creates combined manifest for multi-speaker training

Usage:
    python prepare_xtts_dataset.py --config ../config.yaml
"""

import argparse
import json
import os
import random
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Represents a validated audio sample"""
    audio_path: str
    text: str
    speaker_id: str
    duration: float
    sample_rate: int
    channels: int
    valid: bool
    error: str = ""


class XTTSDatasetPreparer:
    """Prepare and validate dataset for XTTS v2 fine-tuning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.speakers = config['speakers']
        self.output_config = config['output']
        self.quality_config = config.get('quality', {})
        
        self.base_dir = Path(self.output_config['base_dir'])
        self.target_sample_rate = self.output_config.get('sample_rate', 22050)
        
        # Quality thresholds
        self.min_duration = self.quality_config.get('min_audio_duration', 3.0)
        self.max_duration = self.quality_config.get('max_audio_duration', 15.0)
        self.min_text_length = self.quality_config.get('min_text_length', 20)
        self.max_text_length = self.quality_config.get('max_text_length', 200)
    
    def validate_audio(self, audio_path: Path) -> Tuple[bool, Dict]:
        """
        Validate audio file meets XTTS requirements
        
        Returns:
            (is_valid, metadata_dict)
        """
        try:
            info = sf.info(str(audio_path))
            
            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            
            errors = []
            
            # Check duration
            if duration < self.min_duration:
                errors.append(f"Too short: {duration:.2f}s < {self.min_duration}s")
            if duration > self.max_duration:
                errors.append(f"Too long: {duration:.2f}s > {self.max_duration}s")
            
            # Check sample rate (will resample if needed)
            if sample_rate != self.target_sample_rate:
                logger.debug(f"Sample rate mismatch: {sample_rate} != {self.target_sample_rate}")
            
            # Check channels (should be mono)
            if channels != 1:
                logger.debug(f"Not mono: {channels} channels")
            
            is_valid = len(errors) == 0
            
            return is_valid, {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'errors': errors
            }
            
        except Exception as e:
            return False, {
                'duration': 0,
                'sample_rate': 0,
                'channels': 0,
                'errors': [str(e)]
            }
    
    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text meets requirements"""
        errors = []
        
        if len(text) < self.min_text_length:
            errors.append(f"Text too short: {len(text)} < {self.min_text_length}")
        
        if len(text) > self.max_text_length:
            errors.append(f"Text too long: {len(text)} > {self.max_text_length}")
        
        # Check for problematic characters
        if '|' in text:
            errors.append("Text contains pipe character")
        
        return len(errors) == 0, errors
    
    def process_speaker(self, speaker_id: str) -> List[AudioSample]:
        """Process and validate all samples for a speaker"""
        
        speaker_dir = self.base_dir / speaker_id
        wavs_dir = speaker_dir / 'wavs'
        
        if not wavs_dir.exists():
            logger.warning(f"No wavs directory for {speaker_id}")
            return []
        
        # Load synthesis results or metadata
        checkpoint_file = speaker_dir / 'synthesis_checkpoint.json'
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                samples_data = checkpoint_data.get('results', [])
        else:
            # Try to load from sentences.json and infer audio files
            sentences_file = speaker_dir / 'sentences.json'
            if sentences_file.exists():
                with open(sentences_file, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                samples_data = [
                    {'text': s['text'], 'audio_file': f"wavs/{i:04d}.wav", 'index': i}
                    for i, s in enumerate(sentences)
                ]
            else:
                logger.warning(f"No data files found for {speaker_id}")
                return []
        
        valid_samples = []
        invalid_count = 0
        
        for sample in samples_data:
            text = sample.get('text', '')
            audio_file = sample.get('audio_file', '')
            
            if not audio_file:
                continue
            
            audio_path = speaker_dir / audio_file
            
            if not audio_path.exists():
                logger.debug(f"Audio not found: {audio_path}")
                invalid_count += 1
                continue
            
            # Validate audio
            audio_valid, audio_info = self.validate_audio(audio_path)
            
            # Validate text
            text_valid, text_errors = self.validate_text(text)
            
            if audio_valid and text_valid:
                valid_samples.append(AudioSample(
                    audio_path=str(audio_path.relative_to(speaker_dir)),
                    text=text.replace('|', ' '),  # Remove pipes
                    speaker_id=speaker_id,
                    duration=audio_info['duration'],
                    sample_rate=audio_info['sample_rate'],
                    channels=audio_info['channels'],
                    valid=True
                ))
            else:
                invalid_count += 1
                errors = audio_info.get('errors', []) + text_errors
                logger.debug(f"Invalid sample: {audio_file} - {errors}")
        
        logger.info(f"{speaker_id}: {len(valid_samples)} valid, {invalid_count} invalid")
        
        return valid_samples
    
    def create_metadata_csv(
        self,
        speaker_id: str,
        samples: List[AudioSample]
    ) -> Path:
        """
        Create XTTS metadata.csv for a speaker
        
        Format: audio_file|text|speaker_id
        """
        speaker_dir = self.base_dir / speaker_id
        metadata_file = speaker_dir / 'metadata.csv'
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(f"{sample.audio_path}|{sample.text}|{sample.speaker_id}\n")
        
        logger.info(f"Created {metadata_file} with {len(samples)} samples")
        
        return metadata_file
    
    def create_train_val_split(
        self,
        samples: List[AudioSample],
        train_ratio: float = 0.9
    ) -> Tuple[List[AudioSample], List[AudioSample]]:
        """Split samples into train and validation sets"""
        
        # Shuffle
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * train_ratio)
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]
        
        return train_samples, val_samples
    
    def create_combined_manifest(
        self,
        all_samples: Dict[str, List[AudioSample]],
        train_ratio: float = 0.9
    ):
        """
        Create combined train/val manifests for multi-speaker training
        """
        train_all = []
        val_all = []
        
        for speaker_id, samples in all_samples.items():
            train, val = self.create_train_val_split(samples, train_ratio)
            
            # Add full paths
            speaker_dir = self.base_dir / speaker_id
            
            for sample in train:
                train_all.append({
                    'audio_file': str(speaker_dir / sample.audio_path),
                    'text': sample.text,
                    'speaker_id': sample.speaker_id,
                    'duration': sample.duration
                })
            
            for sample in val:
                val_all.append({
                    'audio_file': str(speaker_dir / sample.audio_path),
                    'text': sample.text,
                    'speaker_id': sample.speaker_id,
                    'duration': sample.duration
                })
        
        # Save manifests
        train_file = self.base_dir / 'train_manifest.json'
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_all, f, ensure_ascii=False, indent=2)
        
        val_file = self.base_dir / 'val_manifest.json'
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_all, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Train manifest: {train_file} ({len(train_all)} samples)")
        logger.info(f"Val manifest: {val_file} ({len(val_all)} samples)")
        
        # Also create XTTS-style combined metadata
        combined_metadata = self.base_dir / 'metadata.csv'
        with open(combined_metadata, 'w', encoding='utf-8') as f:
            for item in train_all + val_all:
                audio_path = item['audio_file']
                text = item['text']
                speaker = item['speaker_id']
                f.write(f"{audio_path}|{text}|{speaker}\n")
        
        logger.info(f"Combined metadata: {combined_metadata}")
        
        return train_file, val_file
    
    def prepare_all(self) -> Dict[str, List[AudioSample]]:
        """Process and prepare all speakers"""
        
        all_samples = {}
        
        for speaker_id in self.speakers.keys():
            samples = self.process_speaker(speaker_id)
            
            if samples:
                # Create per-speaker metadata
                self.create_metadata_csv(speaker_id, samples)
                all_samples[speaker_id] = samples
        
        if all_samples:
            # Create combined manifests
            self.create_combined_manifest(all_samples)
        
        return all_samples
    
    def print_statistics(self, all_samples: Dict[str, List[AudioSample]]):
        """Print dataset statistics"""
        
        print("\n" + "=" * 60)
        print("XTTS DATASET STATISTICS")
        print("=" * 60)
        
        total_samples = 0
        total_duration = 0
        
        for speaker_id, samples in all_samples.items():
            n_samples = len(samples)
            duration = sum(s.duration for s in samples)
            
            total_samples += n_samples
            total_duration += duration
            
            print(f"\n{speaker_id}:")
            print(f"  Samples: {n_samples}")
            print(f"  Duration: {duration/60:.2f} minutes")
            print(f"  Avg duration: {duration/n_samples:.2f} seconds" if n_samples > 0 else "")
        
        print(f"\n{'='*60}")
        print(f"TOTAL:")
        print(f"  Speakers: {len(all_samples)}")
        print(f"  Samples: {total_samples}")
        print(f"  Duration: {total_duration/60:.2f} minutes ({total_duration/3600:.2f} hours)")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare XTTS v2 training dataset"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate, do not create manifests'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Train/val split ratio (default: 0.9)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preparer
    preparer = XTTSDatasetPreparer(config)
    
    if args.validate_only:
        logger.info("Validation mode - checking dataset...")
        
        all_samples = {}
        for speaker_id in config['speakers'].keys():
            samples = preparer.process_speaker(speaker_id)
            if samples:
                all_samples[speaker_id] = samples
        
        preparer.print_statistics(all_samples)
        return 0
    
    # Prepare full dataset
    logger.info("Preparing XTTS training dataset...")
    
    all_samples = preparer.prepare_all()
    
    if all_samples:
        preparer.print_statistics(all_samples)
        logger.info("\n✅ Dataset preparation complete!")
        logger.info(f"Output directory: {config['output']['base_dir']}")
    else:
        logger.error("No valid samples found")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

