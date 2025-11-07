#!/usr/bin/env python3
"""
Unified Installation Test Script

Tests that all required packages are installed correctly for all frameworks:
- Core dependencies (torch, librosa, pandas, etc.)
- Whisper framework
- FunASR framework
- Qwen2-Audio support
- Metrics calculation tools

Usage:
    python test_installation.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_core_dependencies():
    """Test core ML/Audio dependencies."""
    logger.info("")
    logger.info("="*70)
    logger.info("Testing Core Dependencies")
    logger.info("="*70)
    
    tests_passed = True
    
    # Test torch
    try:
        import torch
        logger.info(f"✓ torch: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info(f"  - MPS (Apple Silicon) available")
        else:
            logger.info("  - CPU only (no GPU acceleration)")
    except ImportError as e:
        logger.error(f"✗ torch import failed: {e}")
        tests_passed = False
    
    # Test torchaudio
    try:
        import torchaudio
        logger.info(f"✓ torchaudio: {torchaudio.__version__}")
    except ImportError as e:
        logger.error(f"✗ torchaudio import failed: {e}")
        tests_passed = False
    
    # Test librosa
    try:
        import librosa
        logger.info(f"✓ librosa: {librosa.__version__}")
    except ImportError as e:
        logger.error(f"✗ librosa import failed: {e}")
        tests_passed = False
    
    # Test numpy
    try:
        import numpy
        logger.info(f"✓ numpy: {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ numpy import failed: {e}")
        tests_passed = False
    
    # Test pandas
    try:
        import pandas
        logger.info(f"✓ pandas: {pandas.__version__}")
    except ImportError as e:
        logger.error(f"✗ pandas import failed: {e}")
        tests_passed = False
    
    # Test tqdm
    try:
        import tqdm
        logger.info(f"✓ tqdm: {tqdm.__version__}")
    except ImportError as e:
        logger.error(f"✗ tqdm import failed: {e}")
        tests_passed = False
    
    return tests_passed


def test_metrics_dependencies():
    """Test evaluation metrics dependencies."""
    logger.info("")
    logger.info("="*70)
    logger.info("Testing Metrics Dependencies")
    logger.info("="*70)
    
    tests_passed = True
    
    # Test jiwer
    try:
        import jiwer
        logger.info(f"✓ jiwer: {jiwer.__version__}")
    except ImportError as e:
        logger.error(f"✗ jiwer import failed: {e}")
        logger.error("  Required for WER/CER calculation")
        tests_passed = False
    
    # Test scikit-learn
    try:
        import sklearn
        logger.info(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        logger.error(f"✗ scikit-learn import failed: {e}")
        tests_passed = False
    
    return tests_passed


def test_whisper_dependencies():
    """Test Whisper framework dependencies."""
    logger.info("")
    logger.info("="*70)
    logger.info("Testing Whisper Dependencies")
    logger.info("="*70)
    
    tests_passed = True
    
    # Test transformers
    try:
        import transformers
        logger.info(f"✓ transformers: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ transformers import failed: {e}")
        logger.error("  Required for Whisper models")
        tests_passed = False
    
    # Test python-dotenv (optional)
    try:
        import dotenv
        logger.info(f"✓ python-dotenv: installed")
    except ImportError:
        logger.warning("⚠ python-dotenv not installed (optional, for .env file support)")
    
    return tests_passed


def test_funasr_dependencies():
    """Test FunASR framework dependencies."""
    logger.info("")
    logger.info("="*70)
    logger.info("Testing FunASR Dependencies")
    logger.info("="*70)
    
    tests_passed = True
    
    # Test funasr
    try:
        import funasr
        logger.info(f"✓ funasr: {funasr.__version__}")
    except ImportError as e:
        logger.error(f"✗ funasr import failed: {e}")
        logger.error("  Required for Paraformer models")
        tests_passed = False
    
    # Test modelscope
    try:
        import modelscope
        logger.info(f"✓ modelscope: {modelscope.__version__}")
    except ImportError as e:
        logger.error(f"✗ modelscope import failed: {e}")
        logger.error("  Required for FunASR models")
        tests_passed = False
    
    # Test onnxruntime
    try:
        import onnxruntime
        logger.info(f"✓ onnxruntime: {onnxruntime.__version__}")
    except ImportError as e:
        logger.error(f"✗ onnxruntime import failed: {e}")
        tests_passed = False
    
    # Test accelerate (for Qwen2-Audio)
    try:
        import accelerate
        logger.info(f"✓ accelerate: {accelerate.__version__}")
    except ImportError as e:
        logger.error(f"✗ accelerate import failed: {e}")
        logger.error("  Required for Qwen2-Audio models")
        tests_passed = False
    
    return tests_passed


def test_model_loading():
    """Test if we can load a small model (optional, downloads ~200MB)."""
    logger.info("")
    logger.info("="*70)
    logger.info("Testing Model Loading (Optional)")
    logger.info("="*70)
    logger.info("This will download a small model if not cached (~200MB)")
    logger.info("")
    
    response = input("Do you want to test model loading? (y/N): ").lower().strip()
    
    if response != 'y':
        logger.info("Skipping model loading test")
        return True
    
    try:
        from transformers import WhisperProcessor
        logger.info("Testing Whisper model loading...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        logger.info("✓ Successfully loaded Whisper model")
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        logger.error("")
        logger.error("This might be due to:")
        logger.error("1. Network connectivity issues")
        logger.error("2. HuggingFace hub access")
        logger.error("3. Insufficient disk space")
        return False


def print_summary(results):
    """Print test summary."""
    logger.info("")
    logger.info("="*70)
    logger.info("INSTALLATION TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = all(results.values())
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {component}")
    
    logger.info("")
    
    if all_passed:
        logger.info("="*70)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*70)
        logger.info("")
        logger.info("You're ready to run evaluations:")
        logger.info("  python evaluate.py --help")
        logger.info("")
    else:
        logger.info("="*70)
        logger.info("✗ SOME TESTS FAILED")
        logger.info("="*70)
        logger.info("")
        logger.info("Please install missing packages:")
        logger.info("  pip install -r requirements.txt")
        logger.info("")
        logger.info("Or use the interactive setup:")
        logger.info("  ./setup_env.sh")
        logger.info("")
    
    return all_passed


def main():
    """Run all installation tests."""
    logger.info("="*70)
    logger.info("ASR EVALUATION - INSTALLATION TEST")
    logger.info("="*70)
    logger.info("This script verifies that all required packages are installed.")
    logger.info("")
    
    results = {
        "Core Dependencies": test_core_dependencies(),
        "Metrics Dependencies": test_metrics_dependencies(),
        "Whisper Framework": test_whisper_dependencies(),
        "FunASR Framework": test_funasr_dependencies(),
    }
    
    # Optional model loading test
    # results["Model Loading"] = test_model_loading()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)

