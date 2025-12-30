#!/usr/bin/env python3
"""
Convert synthesized audio to NeMo manifest format
Validates audio files and creates train/validation splits
"""

import json
import argparse
import yaml
import random
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeMoManifestConverter:
    """Convert synthesized audio to NeMo manifest format"""
    
    def __init__(self, config_path: str):
        """Load configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.manifest_dir = Path(self.config['output']['manifest_dir'])
        self.train_split = self.config['output']['train_split']
        
        # Quality control settings
        self.min_duration = self.config['quality']['min_audio_duration']
        self.max_duration = self.config['quality']['max_audio_duration']
        self.validate_audio = self.config['quality']['validate_audio']
        
        logger.info("NeMo manifest converter initialized")
    
    def validate_audio_file(self, audio_path: Path) -> Dict:
        """
        Validate audio file and get metadata
        
        Returns:
            Dict with duration and validation status
        """
        try:
            # Load audio
            audio_array, sr = librosa.load(str(audio_path), sr=None)
            duration = len(audio_array) / sr
            
            # Validate duration
            if duration < self.min_duration:
                return {
                    'valid': False,
                    'error': f'Duration too short: {duration:.2f}s < {self.min_duration}s',
                    'duration': duration
                }
            
            if duration > self.max_duration:
                return {
                    'valid': False,
                    'error': f'Duration too long: {duration:.2f}s > {self.max_duration}s',
                    'duration': duration
                }
            
            # Check if audio is not empty
            if len(audio_array) == 0:
                return {
                    'valid': False,
                    'error': 'Empty audio file',
                    'duration': 0.0
                }
            
            return {
                'valid': True,
                'duration': duration,
                'sample_rate': sr
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Failed to load audio: {str(e)}',
                'duration': 0.0
            }
    
    def convert_to_nemo_format(self, results: List[Dict]) -> List[Dict]:
        """
        Convert synthesized results to NeMo manifest format
        
        NeMo format:
        {
            "audio_filepath": "/absolute/path/to/audio.mp3",
            "text": "transcription text",
            "duration": 2.5
        }
        """
        manifest_samples = []
        invalid_samples = []
        
        logger.info(f"Converting {len(results)} samples to NeMo format...")
        
        for result in tqdm(results, desc="Validating audio"):
            audio_path = Path(result['audio_path'])
            
            # Check if file exists
            if not audio_path.exists():
                invalid_samples.append({
                    **result,
                    'error': 'Audio file not found'
                })
                continue
            
            # Validate audio if enabled
            if self.validate_audio:
                validation = self.validate_audio_file(audio_path)
                
                if not validation['valid']:
                    invalid_samples.append({
                        **result,
                        'error': validation['error']
                    })
                    continue
                
                duration = validation['duration']
            else:
                # Quick duration check without full validation
                try:
                    audio_info = sf.info(str(audio_path))
                    duration = audio_info.duration
                except:
                    invalid_samples.append({
                        **result,
                        'error': 'Failed to get audio duration'
                    })
                    continue
            
            # Create NeMo manifest entry
            manifest_entry = {
                'audio_filepath': str(audio_path.absolute()),
                'text': result['text'],
                'duration': float(duration)
            }
            
            manifest_samples.append(manifest_entry)
        
        logger.info(f"Valid samples: {len(manifest_samples)}/{len(results)}")
        logger.info(f"Invalid samples: {len(invalid_samples)}/{len(results)}")
        
        if invalid_samples:
            # Save invalid samples log
            invalid_log = self.manifest_dir / 'invalid_samples.json'
            invalid_log.parent.mkdir(parents=True, exist_ok=True)
            with open(invalid_log, 'w', encoding='utf-8') as f:
                json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
            logger.warning(f"Invalid samples logged to: {invalid_log}")
        
        return manifest_samples
    
    def split_train_val(self, samples: List[Dict]) -> tuple:
        """Split samples into train and validation sets"""
        # Shuffle for random split
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        split_idx = int(len(shuffled) * self.train_split)
        
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
        logger.info(f"  Val: {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
        
        return train_samples, val_samples
    
    def save_manifest(self, samples: List[Dict], output_file: Path):
        """Save samples in NeMo JSONL format"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(samples)} samples to: {output_file}")
    
    def print_statistics(self, samples: List[Dict], name: str):
        """Print dataset statistics"""
        durations = [s['duration'] for s in samples]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Text length statistics
        text_lengths = [len(s['text']) for s in samples]
        avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        logger.info(f"\n{name} Statistics:")
        logger.info(f"  Samples: {len(samples):,}")
        logger.info(f"  Total Duration: {total_duration / 3600:.2f} hours")
        logger.info(f"  Avg Duration: {avg_duration:.2f}s")
        logger.info(f"  Min Duration: {min_duration:.2f}s")
        logger.info(f"  Max Duration: {max_duration:.2f}s")
        logger.info(f"  Avg Text Length: {avg_text_len:.1f} characters")
    
    def convert(self, input_file: str):
        """Main conversion pipeline"""
        # Load synthesis results
        logger.info(f"Loading synthesis results from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Loaded {len(results):,} results")
        
        # Convert to NeMo format
        logger.info("\n" + "="*70)
        logger.info("Converting to NeMo manifest format")
        logger.info("="*70)
        manifest_samples = self.convert_to_nemo_format(results)
        
        if not manifest_samples:
            logger.error("No valid samples to create manifest!")
            return
        
        # Split train/val
        train_samples, val_samples = self.split_train_val(manifest_samples)
        
        # Save manifests
        train_manifest = self.manifest_dir / 'train_manifest.json'
        val_manifest = self.manifest_dir / 'val_manifest.json'
        
        self.save_manifest(train_samples, train_manifest)
        self.save_manifest(val_samples, val_manifest)
        
        # Print statistics
        self.print_statistics(train_samples, "Training Set")
        self.print_statistics(val_samples, "Validation Set")
        
        # Sample entries
        logger.info(f"\nSample training entries:")
        for i, sample in enumerate(train_samples[:3], 1):
            logger.info(f"  {i}. Text: {sample['text']}")
            logger.info(f"     Duration: {sample['duration']:.2f}s")
            logger.info(f"     Audio: {Path(sample['audio_filepath']).name}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ NeMo manifest creation completed!")
        logger.info("="*70)
        logger.info(f"Train manifest: {train_manifest}")
        logger.info(f"Val manifest: {val_manifest}")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert synthesized audio to NeMo manifest format"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/synthesized.json',
        help='Input file with synthesis results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Convert to manifest
    converter = NeMoManifestConverter(args.config)
    converter.convert(args.input)
    
    logger.info("\n✅ Manifest preparation completed successfully!")


if __name__ == "__main__":
    main()

