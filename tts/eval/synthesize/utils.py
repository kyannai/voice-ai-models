#!/usr/bin/env python3
"""
Utility functions for TTS synthesis and evaluation

Provides common functions for:
- Loading text data from datasets
- Saving synthesized audio
- Managing output directories
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text_data(
    test_data_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load text data from dataset file (JSON, TSV, or CSV)
    
    For TTS evaluation, we extract the reference text from ASR datasets.
    The text field is used as input for TTS synthesis.
    
    Args:
        test_data_path: Path to test data file (JSON, TSV, or CSV)
        max_samples: Optional limit on number of samples to load
        
    Returns:
        List of dictionaries with 'text' and optionally 'audio_path' keys
    """
    test_data_path = Path(test_data_path)
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    suffix = test_data_path.suffix.lower()
    samples = []
    
    if suffix == '.json':
        samples = _load_json_data(test_data_path)
    elif suffix == '.tsv':
        samples = _load_tsv_data(test_data_path)
    elif suffix == '.csv':
        samples = _load_csv_data(test_data_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Limit samples if specified
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Loaded {len(samples)} text samples from {test_data_path.name}")
    
    return samples


def _load_json_data(json_path: Path) -> List[Dict]:
    """Load text data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, dict) and 'samples' in data:
        items = data['samples']
    else:
        # Assume it's a dict with items
        items = list(data.values()) if isinstance(data, dict) else [data]
    
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            # Try different field names for text
            text = (
                item.get('text') or 
                item.get('reference') or 
                item.get('transcription') or
                item.get('sentence') or
                item.get('transcript') or
                ''
            )
            audio_path = item.get('audio_path') or item.get('audio') or item.get('path')
        elif isinstance(item, str):
            text = item
            audio_path = None
        else:
            continue
        
        if text:
            sample = {
                'id': idx,
                'text': text.strip(),
            }
            if audio_path:
                sample['reference_audio'] = audio_path
            samples.append(sample)
    
    return samples


def _load_tsv_data(tsv_path: Path) -> List[Dict]:
    """Load text data from TSV file"""
    samples = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for idx, row in enumerate(reader):
            # Try different column names for text
            text = (
                row.get('text') or 
                row.get('reference') or 
                row.get('transcription') or
                row.get('sentence') or
                row.get('transcript') or
                ''
            )
            audio_path = row.get('audio_path') or row.get('audio') or row.get('path')
            
            if text:
                sample = {
                    'id': idx,
                    'text': text.strip(),
                }
                if audio_path:
                    sample['reference_audio'] = audio_path
                samples.append(sample)
    
    return samples


def _load_csv_data(csv_path: Path) -> List[Dict]:
    """Load text data from CSV file"""
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            text = (
                row.get('text') or 
                row.get('reference') or 
                row.get('transcription') or
                row.get('sentence') or
                row.get('transcript') or
                ''
            )
            audio_path = row.get('audio_path') or row.get('audio') or row.get('path')
            
            if text:
                sample = {
                    'id': idx,
                    'text': text.strip(),
                }
                if audio_path:
                    sample['reference_audio'] = audio_path
                samples.append(sample)
    
    return samples


def load_dataset_by_name(
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load text data by dataset name from registry
    
    Args:
        dataset_name: Name of dataset from registry (e.g., 'meso-malaya-test')
        max_samples: Optional limit on number of samples
        
    Returns:
        List of text sample dictionaries
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets_config import get_dataset_config
    
    config = get_dataset_config(dataset_name)
    test_data_path = config['test_data']
    
    return load_text_data(test_data_path, max_samples=max_samples)


def save_audio(
    audio_array: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = 22050,
) -> Path:
    """
    Save audio array to file
    
    Args:
        audio_array: Audio data as numpy array
        output_path: Path to save audio file
        sample_rate: Sample rate of audio (default: 22050)
        
    Returns:
        Path to saved audio file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio is float32 and normalized
    audio = np.asarray(audio_array, dtype=np.float32)
    
    # Normalize if needed
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    
    sf.write(str(output_path), audio, sample_rate)
    
    return output_path


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """
    Get duration of audio file in seconds
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    import librosa
    duration = librosa.get_duration(path=str(audio_path))
    return duration


def save_synthesis_results(
    results: List[Dict],
    output_dir: Union[str, Path],
    model_name: str,
) -> Path:
    """
    Save synthesis results to JSON file
    
    Args:
        results: List of synthesis result dictionaries
        output_dir: Output directory path
        model_name: Name of TTS model used
        
    Returns:
        Path to saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(results),
        'results': results,
    }
    
    output_file = output_dir / 'synthesis_results.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved synthesis results to {output_file}")
    
    return output_file


def create_output_dir(
    base_dir: Union[str, Path],
    model_name: str,
    dataset_name: str,
    custom_name: Optional[str] = None,
) -> Path:
    """
    Create timestamped output directory for synthesis results
    
    Args:
        base_dir: Base output directory
        model_name: Name of TTS model
        dataset_name: Name of dataset
        custom_name: Optional custom name prefix
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model name for directory
    model_clean = model_name.replace('/', '_').replace('\\', '_')
    
    # Build directory name
    parts = []
    if custom_name:
        parts.append(custom_name)
    parts.extend([model_clean, dataset_name, timestamp])
    
    dir_name = "_".join(parts)
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create audio subdirectory
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir


def calculate_rtf(synthesis_time: float, audio_duration: float) -> float:
    """
    Calculate Real-Time Factor (RTF)
    
    RTF = synthesis_time / audio_duration
    RTF < 1.0 means faster than real-time
    
    Args:
        synthesis_time: Time taken to synthesize (seconds)
        audio_duration: Duration of synthesized audio (seconds)
        
    Returns:
        RTF value
    """
    if audio_duration <= 0:
        return 0.0
    return synthesis_time / audio_duration

