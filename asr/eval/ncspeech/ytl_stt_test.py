import time
import openai
import tqdm
import re
import malaya
import random
import json
import argparse
import logging
import sys
import os
import multiprocessing as mp
import torch

from pathlib import Path
from torchmetrics.text import WordErrorRate
from abc import ABC, abstractmethod

# Add parent directory to path to import from eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets_config import list_datasets, get_dataset_config
from dataset_loader import load_dataset


lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model=lm)
# stemmer = malaya.stem.huggingface()
normalizer_mal = malaya.normalizer.rules.load(corrector, None)

chars_to_ignore_regex_normalise = r"""[\/:\\;"−*`‑―''""„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
pattern_normalise = re.compile(chars_to_ignore_regex_normalise, flags=re.UNICODE)


def detect_and_remove_repetition(text: str, max_repetitions: int = 3) -> tuple:
    """
    Detect and remove repetitive patterns in transcription (hallucination detection)
    
    Args:
        text: Input transcription text
        max_repetitions: Maximum allowed consecutive repetitions of a pattern
        
    Returns:
        Tuple of (cleaned_text, was_hallucination)
    """
    if not text:
        return text, False
    
    original_text = text
    
    # Check for repeated Unicode characters (e.g., Chinese "嗯嗯嗯嗯..." hallucination)
    # This must be checked FIRST before word splitting since CJK chars don't use spaces
    # Look for the same character repeated many times consecutively
    if len(text) > 20:
        # Check for repeated single characters (especially non-ASCII like CJK)
        char_counts = {}
        i = 0
        while i < len(text):
            char = text[i]
            # Count consecutive occurrences of this character
            count = 1
            j = i + 1
            while j < len(text) and text[j] == char:
                count += 1
                j += 1
            
            # If a single character repeats more than 10 times consecutively, likely hallucination
            # Especially for non-ASCII characters (Chinese, etc.)
            if count > 10 and (ord(char) > 127 or char in ['a', 'e', 'i', 'o', 'u', '.']):
                # Find where the excessive repetition starts
                # Keep text before the repetition
                cleaned = text[:i].strip()
                print(f"[WARNING] Detected character repetition hallucination: '{char}' repeated {count} times")
                return cleaned if cleaned else char, True
            
            i = j if j > i else i + 1
    
    # Check for repeated 2-4 character patterns (e.g., "嗯嗯" or "um um" patterns)
    # This catches patterns like "嗯嗯嗯嗯" or "uh uh uh uh"
    for pattern_len in range(1, 5):
        if len(text) >= pattern_len * 10:  # Need at least 10 repetitions to check
            # Try to find patterns that repeat
            for start_pos in range(min(len(text) - pattern_len * 5, 100)):  # Check first 100 chars
                pattern = text[start_pos:start_pos + pattern_len]
                
                # Count how many times this pattern repeats consecutively
                count = 0
                pos = start_pos
                while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == pattern:
                    count += 1
                    pos += pattern_len
                
                # If pattern repeats more than 10 times, likely hallucination
                if count > 10:
                    cleaned = text[:start_pos].strip()
                    print(f"[WARNING] Detected character pattern hallucination: '{pattern}' repeated {count} times")
                    return cleaned if cleaned else pattern, True
    
    # Check for comma-separated repetitions (e.g., "eh, eh, eh, eh, eh,")
    # This is a very common hallucination pattern - must check FIRST
    comma_pattern = re.findall(r'(\b\w{1,5}\b)(?:,\s*\1){5,}', text)
    if comma_pattern:
        # Found repeated words separated by commas
        for repeated_word in comma_pattern:
            # Find where the repetition starts
            repetition_start = text.find(repeated_word + ',')
            if repetition_start != -1:
                # Count how many times it repeats
                temp = text[repetition_start:]
                count = temp.count(repeated_word + ',')
                if count > max_repetitions:
                    # Truncate at the start of excessive repetition
                    cleaned = text[:repetition_start].strip()
                    if not cleaned:  # If nothing before repetition, keep one instance
                        cleaned = repeated_word
                    print(f"[WARNING] Detected comma-separated hallucination: '{repeated_word}' repeated {count} times")
                    return cleaned, True
    
    # Check for repeated words/phrases anywhere in the text (e.g., "Kategori. Kategori. Kategori...")
    words = text.split()
    if len(words) > 3:
        # Look for patterns where the same word/phrase repeats many times
        # Use a sliding window approach to detect repetitions efficiently
        idx = 0
        while idx < len(words):
            # Try different pattern lengths starting from this position
            for pattern_len in range(1, min(10, (len(words) - idx) // (max_repetitions + 1) + 1)):
                if idx + pattern_len > len(words):
                    break
                
                pattern = words[idx:idx + pattern_len]
                
                # Count consecutive repetitions from this position
                repetitions = 0
                check_idx = idx
                while check_idx + pattern_len <= len(words):
                    if words[check_idx:check_idx + pattern_len] == pattern:
                        repetitions += 1
                        check_idx += pattern_len
                    else:
                        break
                
                # If we find excessive repetition, truncate
                if repetitions > max_repetitions:
                    pattern_str = " ".join(pattern)
                    # Keep everything before the repetition + one instance of the pattern
                    cleaned = " ".join(words[:idx + pattern_len])
                    print(f"[WARNING] Detected word repetition hallucination at position {idx}: pattern '{pattern_str}' repeated {repetitions} times")
                    return cleaned, True
            
            idx += 1
    
    # Check for simple repeated words (case-insensitive, ignoring punctuation)
    # This catches patterns like "kata kata kata kata"
    cleaned_words = [re.sub(r'[^\w\s]', '', w.lower()) for w in words if w]
    if len(cleaned_words) > 5:
        # Count consecutive same words
        i = 0
        while i < len(cleaned_words):
            word = cleaned_words[i]
            if not word:
                i += 1
                continue
            
            # Count how many times this word repeats consecutively
            consecutive_count = 1
            j = i + 1
            while j < len(cleaned_words) and cleaned_words[j] == word:
                consecutive_count += 1
                j += 1
            
            # If same word repeats > threshold times
            if consecutive_count > max_repetitions * 2:  # More lenient for simple words
                # Truncate before the excessive repetition
                cleaned = " ".join(words[:i + 1])  # Keep 1 instance only
                print(f"[WARNING] Detected simple word hallucination: '{word}' repeated {consecutive_count} times")
                return cleaned, True
            
            i = j if j > i + 1 else i + 1
    
    # Check for repeated punctuation patterns (e.g., "...")
    # If more than 10 consecutive dots, it's likely hallucination
    if text.count('.') > 10:
        dot_match = re.search(r'\.{10,}', text)
        if dot_match:
            # Find where the excessive dots start and truncate
            cleaned = text[:dot_match.start()].strip()
            print(f"[WARNING] Detected punctuation hallucination: {text.count('.')} dots found")
            return cleaned, True
    
    return text, False


def postprocess_text_mal(texts):
    def normalize_superscripts(text: str) -> str:
        superscripts = {
            '¹': 'satu',
            '²': 'dua',
            '³': 'tiga',
        }
        for k, v in superscripts.items():
            text = text.replace(k, v)
        return text
    result_text = []
    for sentence in texts:
        sentence = normalize_superscripts(sentence).lower()
        sentence = pattern_normalise.sub(' ', sentence)
        sentence = normalizer_mal.normalize(sentence,
                                            normalize_url=True,
                                            normalize_email=True,
                                            normalize_time=False,
                                            normalize_emoji=False)['normalize']

        sentence = re.sub(r"[^\w\s'\-]", ' ', sentence)
        sentence = re.sub(r'(\s{2,})', ' ', re.sub('(\s+$)|(\A\s+)', '', sentence))
        result_text.append(sentence)
    return result_text


class ASRRecognizer(ABC):
    """Base class for ASR recognizers"""
    
    @abstractmethod
    def transcribe(self, wav_path):
        """Transcribe audio file and return text"""
        pass


class YTLASRRecognizer(ASRRecognizer):
    """YTL API-based ASR recognizer"""
    
    def __init__(self, model="ilmu-trial-asr"):
        api_key = os.environ.get('ILMU_API_KEY')
        
        if not api_key:
            raise ValueError(
                "ILMU_API_KEY environment variable is required for YTL provider. "
                "Please set it with: export ILMU_API_KEY=your_api_key_here"
            )

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.staging.ytlailabs.tech/v1"
        )
        self.model = model

    def transcribe(self, wav_path):
        with open(wav_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f
            )
        return response.text


class HuggingFaceASRRecognizer(ASRRecognizer):
    """HuggingFace model-based ASR recognizer"""
    
    def __init__(self, model_id="openai/whisper-large-v3-turbo"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import librosa
        
        self.model_id = model_id
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required for HuggingFace provider. "
                "Please set it with: export HF_TOKEN=your_token_here"
            )
        
        # Setup device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Determine if this is a Whisper model that needs hallucination detection
        self.is_whisper_turbo = any(model_name in model_id.lower() for model_name in [
            'whisper-large-v3-turbo',
            'malaysian-whisper-large-v3-turbo'
        ])
        
        print(f"Loading HuggingFace model: {model_id}")
        print(f"Device: {self.device}, dtype: {torch_dtype}")
        if self.is_whisper_turbo:
            print(f"Hallucination detection: ENABLED (Whisper turbo model detected)")
        
        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=hf_token
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )
        
        print(f"Model loaded successfully on {self.device}")
        
        # Track hallucination statistics
        self.hallucination_count = 0
        self.total_transcriptions = 0
    
    def transcribe(self, wav_path):
        import librosa
        
        # Load audio file
        audio, sr = librosa.load(wav_path, sr=16000)
        
        # Prepare generation kwargs with anti-hallucination parameters for Whisper turbo models
        generate_kwargs = {"language": "malay"}
        
        if self.is_whisper_turbo:
            # Anti-hallucination parameters from transcribe_whisper.py
            generate_kwargs.update({
                "task": "transcribe",
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
                "no_speech_threshold": 0.4,
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -0.5,
            })
        
        # Transcribe
        try:
            result = self.pipe(audio, generate_kwargs=generate_kwargs)
        except (TypeError, AttributeError) as e:
            # If threshold parameters cause issues, retry with minimal params
            if "NoneType" in str(e) or "not supported between" in str(e):
                print(f"[WARNING] Threshold parameters caused error, retrying with minimal params")
                result = self.pipe(audio, generate_kwargs={"language": "malay"})
            else:
                raise
        
        text = result["text"]
        
        # Apply hallucination detection for Whisper turbo models
        if self.is_whisper_turbo:
            self.total_transcriptions += 1
            cleaned_text, was_hallucination = detect_and_remove_repetition(text.strip())
            if was_hallucination:
                self.hallucination_count += 1
                print(f"[HALLUCINATION] Detected and cleaned repetition in: {wav_path}")
                print(f"  Original length: {len(text)} chars, Cleaned length: {len(cleaned_text)} chars")
            return cleaned_text
        
        return text


def create_recognizer(provider, model):
    """Factory function to create appropriate recognizer"""
    if provider == "ytl":
        return YTLASRRecognizer(model=model)
    elif provider == "huggingface":
        return HuggingFaceASRRecognizer(model_id=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def split_dict(data, num_splits):
    """Split dictionary into N roughly equal parts."""
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]


def job_audio(args):
    audio_dict_slice, out_dir, postprocessing, delay, provider, model, force = args
    recognizer = create_recognizer(provider=provider, model=model)

    for uid, wav_path in tqdm.tqdm(audio_dict_slice.items()):
        out_path = Path(out_dir) / f"{uid}.json"
        if out_path.exists() and not force:
            continue

        try:
            text = recognizer.transcribe(wav_path)
            result = {"text": text,
                      "text_norm": postprocessing([text])[0]}

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Only add delay for API-based providers to prevent rate limiting
            if provider == 'ytl':
                time.sleep(delay)

        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")
    
    # Return hallucination stats if available
    if hasattr(recognizer, 'hallucination_count'):
        return recognizer.hallucination_count, recognizer.total_transcriptions
    return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on test datasets with multiple providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{chr(10).join(f'  - {name}' for name in list_datasets())}

Examples:
  # YTL provider (API-based)
  export ILMU_API_KEY=your_api_key_here
  python ytl_stt_test.py --provider ytl --dataset fleurs-test --model ilmu-trial-asr --workers 1 --delay 0.5 --force
  
  # HuggingFace provider (local model)
  export HF_TOKEN=your_token_here
  python ytl_stt_test.py --provider huggingface --dataset malay-conversational --model openai/whisper-large-v3 --workers 8 --delay 0 --force
  python ytl_stt_test.py --provider huggingface --dataset malay-conversational --model mesolitica/Malaysian-whisper-large-v3-turbo-v3 --workers 8 --delay 0 --force
  
  # Other options
  python ytl_stt_test.py --dataset fleurs-test --workers 8
  python ytl_stt_test.py --dataset malay-scripted --max-samples 100
  python ytl_stt_test.py --dataset fleurs-test --delay 3.0  # slower for rate limits
  python ytl_stt_test.py --dataset fleurs-test --force  # re-transcribe all files
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name from registry (e.g., malay-conversational, fleurs-test)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='ytl',
        choices=['ytl', 'huggingface'],
        help='ASR provider to use: ytl (API-based) or huggingface (local model) (default: ytl)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ilmu-trial-asr',
        help='Model name/ID. For ytl: model name (default: ilmu-trial-asr). For huggingface: model ID (e.g., openai/whisper-large-v3-turbo)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers for transcription. Default: 4 for ytl, 1 for huggingface (GPU limited)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ytl_labs',
        help='Base directory for saving transcription results (default: ./ytl_labs)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay in seconds between API calls to avoid rate limiting (default: 2.0)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-transcription even if cached results exist'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # List datasets and exit if requested
    if args.list_datasets:
        print("\nAvailable datasets:")
        for dataset_name in list_datasets():
            config = get_dataset_config(dataset_name)
            print(f"\n  {dataset_name}")
            print(f"    Description: {config['description']}")
            print(f"    Language: {config['language']}")
        print()
        sys.exit(0)
    
    # Validate dataset argument
    if not args.dataset:
        parser.error("--dataset is required (or use --list-datasets to see available datasets)")
    
    # Set defaults based on provider
    if args.workers is None:
        args.workers = 4 if args.provider == 'ytl' else 1
    
    # Set default model based on provider if not specified
    if args.model == 'ilmu-trial-asr' and args.provider == 'huggingface':
        args.model = 'openai/whisper-large-v3-turbo'
        logger.info(f"Using default HuggingFace model: {args.model}")
    
    # Validate API keys/tokens based on provider
    if args.provider == 'ytl':
        if not os.environ.get('ILMU_API_KEY'):
            logger.error("ILMU_API_KEY environment variable is required for YTL provider")
            logger.error("Please set it with: export ILMU_API_KEY=your_api_key_here")
            sys.exit(1)
    elif args.provider == 'huggingface':
        if not os.environ.get('HF_TOKEN'):
            logger.error("HF_TOKEN environment variable is required for HuggingFace provider")
            logger.error("Please set it with: export HF_TOKEN=your_token_here")
            sys.exit(1)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        samples = load_dataset(args.dataset, max_samples=args.max_samples, validate=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    if len(samples) == 0:
        logger.error("No valid samples found in dataset")
        sys.exit(1)
    
    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"Using provider: {args.provider}")
    logger.info(f"Using model: {args.model}")
    
    # Setup output directory (include provider and model name to avoid cache conflicts)
    model_safe_name = args.model.replace('/', '_').replace('\\', '_')
    save_dir = Path(args.output_dir) / f"{args.dataset}_{args.provider}_{model_safe_name}"
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {save_dir}")
    if args.force:
        logger.info("Force mode enabled: will re-transcribe all files")
    
    # Prepare audio dict and reference transcripts
    audio_dict = {}
    ref_transcript = {}
    
    for sample in samples:
        audio_path = sample['audio_path']
        reference = sample['reference']
        uid = str(Path(audio_path).stem)
        audio_dict[uid] = audio_path
        ref_transcript[uid] = reference
    
    logger.info(f"Processing {len(audio_dict)} audio files with {args.workers} workers")
    if args.provider == 'ytl':
        logger.info(f"Delay between API calls: {args.delay} seconds")
    
    # Split work across workers
    audio_dict_slices = split_dict(audio_dict, args.workers)
    
    # Prepare args for multiprocessing
    lang_postprocessing = postprocess_text_mal
    mp_args = [(slice_dict, str(save_dir), lang_postprocessing, args.delay, args.provider, args.model, args.force) for slice_dict in audio_dict_slices]
    
    # Run transcription in parallel
    logger.info("Starting transcription...")
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(job_audio, mp_args)
    
    # Aggregate hallucination statistics from all workers
    total_hallucinations = sum(r[0] for r in results)
    total_processed = sum(r[1] for r in results)
    
    logger.info("Transcription complete. Computing WER...")
    
    # Calculate WER
    WER = WordErrorRate()
    json_files = {l.stem: str(l) for l in list(save_dir.glob('*.json'))}
    
    processed_count = 0
    error_count = 0
    
    for uid in json_files:
        try:
            with open(json_files[uid], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            hyp = data.get("text_norm", "")
            ref = ref_transcript.get(uid, "")
            
            if not ref:
                logger.warning(f"No reference found for {uid}")
                continue
            
            # Print sample predictions
            if processed_count % 50 == 0:
                logger.info(f"\nSample {uid}:")
                logger.info(f"  HYP: {hyp}")
                logger.info(f"  REF: {ref}")
            
            WER.update([hyp], [ref])
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {uid}: {e}")
            error_count += 1
    
    # Compute final WER
    if processed_count > 0:
        final_wer = WER.compute()
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Provider: {args.provider}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Processed: {processed_count} samples")
        if error_count > 0:
            logger.info(f"Errors: {error_count} samples")
        
        # Report hallucination statistics for Whisper turbo models
        if total_processed > 0:
            hallucination_rate = (total_hallucinations / total_processed) * 100
            logger.info(f"Hallucinations detected: {total_hallucinations}/{total_processed} ({hallucination_rate:.1f}%)")
        
        logger.info(f"Final WER: {final_wer:.3f} ({final_wer*100:.2f}%)")
        logger.info("=" * 70)
    else:
        logger.error("No samples were successfully processed")
        sys.exit(1)