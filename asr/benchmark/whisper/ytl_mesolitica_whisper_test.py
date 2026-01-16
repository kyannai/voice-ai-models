import pandas as pd
import tqdm
import re
import malaya
import torch
import torchaudio

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torchmetrics.text import WordErrorRate
from pathlib import Path

DATASETS = {
    "fleurs_test": "../test_data/YTL_testsets/fleurs_test.tsv",
    "malay_conversational": "../test_data/YTL_testsets/malay_conversational_meta.tsv",
    "malay_scripted": "../test_data/YTL_testsets/malay_scripted_meta.tsv",
}


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


# Track hallucination statistics
hallucination_count = 0
total_transcriptions = 0

lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model = lm)
normalizer_mal = malaya.normalizer.rules.load(corrector, None)


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
    chars_to_ignore_regex_normalise = r"""[\/:\\;"−*`‑―''""„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
    pattern_normalise = re.compile(chars_to_ignore_regex_normalise, flags=re.UNICODE)
    for sentence in texts:
        sentence = normalize_superscripts(sentence).lower()
        sentence = pattern_normalise.sub(' ', sentence)
        sentence = normalizer_mal.normalize(sentence,
                                            normalize_url=True,
                                            normalize_email=True,
                                            normalize_time=False,
                                            normalize_emoji=False)['normalize']

        sentence = re.sub(r'(?<!\w)-(?!\w)', ' ', sentence)
        sentence = re.sub(r"[^\w\s\-]", ' ', sentence)
        sentence = re.sub(r'(\s{2,})', ' ', re.sub('(\s+$)|(\A\s+)', '', sentence))
        result_text.append(sentence)
    return result_text


# Load the Mesolitica Malaysian Whisper model
model_id = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model {model_id} on {device}...")
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
model.to(device)
print("Model loaded successfully!")


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Mesolitica Malaysian Whisper model."""
    global hallucination_count, total_transcriptions
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Prepare input features
    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device, dtype=torch_dtype)
    
    # Generate transcription with deterministic settings
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="ms",
            task="transcribe",
            return_timestamps=True,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )
    
    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Apply hallucination detection
    total_transcriptions += 1
    cleaned_text, was_hallucination = detect_and_remove_repetition(transcription.strip())
    if was_hallucination:
        hallucination_count += 1
        print(f"[HALLUCINATION] Detected and cleaned repetition in: {audio_path}")
        print(f"  Original length: {len(transcription)} chars, Cleaned length: {len(cleaned_text)} chars")
    
    return cleaned_text


def compute_transcript():
    postprocessing = postprocess_text_mal

    all_wers = {}
    for d in DATASETS:
        print(f"\nProcessing dataset: {d}")

        all_data = pd.read_csv(DATASETS[d], sep='\t')
        tsv_dir = Path(DATASETS[d]).parent  # Get the directory containing the TSV file
        audio_dict = {}
        ref_transcript = {}
        sys_transcription = {}

        for idx, row in all_data.iterrows():
            audio_filepath = tsv_dir / row["path"]  # Construct full path relative to TSV location
            duration = float(row["duration"])
            audio_dict[str(Path(audio_filepath).stem)] = str(audio_filepath)
            ref_transcript[str(Path(audio_filepath).stem)] = row["sentence"]

        for audio_utt in tqdm.tqdm(audio_dict):
            text = transcribe_audio(audio_dict[audio_utt])
            sys_transcription[audio_utt] = postprocessing([text])[0]

        WER = WordErrorRate()
        for utt in sys_transcription:
            hyp = sys_transcription[utt]
            WER.update(hyp, ref_transcript[utt])

        _wer = WER.compute()
        all_wers[d] = float(_wer)
        print(f"WER for {d}: {_wer:.4f}")
        WER.reset()
    
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    for dataset, wer in all_wers.items():
        print(f"  {dataset}: {wer:.4f}")
    
    print("\n" + "-"*50)
    print("Hallucination Statistics:")
    print("-"*50)
    print(f"  Total transcriptions: {total_transcriptions}")
    print(f"  Hallucinations detected: {hallucination_count}")
    if total_transcriptions > 0:
        print(f"  Hallucination rate: {hallucination_count/total_transcriptions*100:.2f}%")


if __name__ == "__main__":
    compute_transcript()

