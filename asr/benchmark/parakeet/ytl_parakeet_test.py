import os
import time
import argparse
import tqdm
import re
import malaya
import random
import json
import pandas as pd
import multiprocessing as mp
import torch

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from torchmetrics.text import WordErrorRate

DATASETS = {
    "fleurs_test": "../test_data/YTL_testsets/fleurs_test.tsv",
    "malay_conversational": "../test_data/YTL_testsets/malay_conversational_meta.tsv",
    "malay_scripted": "../test_data/YTL_testsets/malay_scripted_meta.tsv",
}


lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model=lm)
normalizer_mal = malaya.normalizer.rules.load(corrector, None)

chars_to_ignore_regex_normalise = r"""[\/:\\;"−*`‑―''""„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
pattern_normalise = re.compile(chars_to_ignore_regex_normalise, flags=re.UNICODE)


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


class ParakeetRecognizer:
    """
    NVIDIA Parakeet model-based ASR recognizer using NeMo
    """
    
    def __init__(self, model_id="nvidia/parakeet-tdt-0.6b", batch_size=16):
        import nemo.collections.asr as nemo_asr
        
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NeMo Parakeet model: {model_id}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        
        is_local_nemo = model_id.endswith('.nemo') and os.path.exists(model_id)
        
        if is_local_nemo:
            print(f"Loading from local .nemo file: {model_id}")
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_id)
        else:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required for HuggingFace models. "
                    "Please add it to .env file or export it: export HF_TOKEN=your_token_here"
                )
            from huggingface_hub import login
            login(token=hf_token)
            print("Loading model from HuggingFace...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        
        print(f"Moving model to {self.device} and setting eval mode...")
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model.cfg, 'decoding'):
            if self.model.cfg.decoding.strategy != "beam":
                self.model.cfg.decoding.strategy = "greedy_batch"
                self.model.change_decoding_strategy(self.model.cfg.decoding)
        
        print(f"Model loaded successfully on {self.device}")
        self.total_transcriptions = 0
    
    def _audio_to_tensor(self, wav_path):
        import librosa
        audio, sr = librosa.load(wav_path, sr=16000)
        return torch.from_numpy(audio)
    
    def transcribe(self, wav_path):
        audio_tensor = self._audio_to_tensor(wav_path)
        
        with torch.inference_mode():
            with torch.no_grad():
                output = self.model.transcribe([audio_tensor])
        
        self.total_transcriptions += 1
        
        if isinstance(output, tuple):
            output = output[0]
        
        text = ""
        if isinstance(output, list) and len(output) > 0:
            first_result = output[0]
            if hasattr(first_result, 'text'):
                text = first_result.text.strip()
            else:
                text = str(first_result).strip()
        else:
            text = str(output).strip()
        
        return text
    
    def transcribe_batch(self, wav_paths):
        audio_tensors = []
        for wav_path in wav_paths:
            audio_tensor = self._audio_to_tensor(wav_path)
            audio_tensors.append(audio_tensor)
        
        with torch.inference_mode():
            with torch.no_grad():
                output = self.model.transcribe(audio_tensors)
        
        self.total_transcriptions += len(wav_paths)
        
        if isinstance(output, tuple):
            output = output[0]
        
        texts = []
        if isinstance(output, list):
            for result in output:
                if hasattr(result, 'text'):
                    texts.append(result.text.strip())
                else:
                    texts.append(str(result).strip())
        else:
            texts = [str(output).strip()]
        
        if len(texts) != len(wav_paths):
            raise ValueError(
                f"Expected {len(wav_paths)} transcriptions but got {len(texts)}. "
                f"Output format: {type(output)}"
            )
        
        return texts


def split_dict(data, num_splits):
    """Split dictionary into N roughly equal parts."""
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]


def job_audio(args):
    audio_dict_slice, out_dir, postprocessing, model_id, batch_size = args
    recognizer = ParakeetRecognizer(model_id=model_id, batch_size=batch_size)

    # Collect audio files that need processing
    audio_items = []
    for uid, wav_path in audio_dict_slice.items():
        out_path = Path(out_dir) / f"{uid}.json"
        if not out_path.exists():
            audio_items.append((uid, wav_path, out_path))

    if not audio_items:
        return

    # Process in batches for efficiency
    for i in tqdm.tqdm(range(0, len(audio_items), batch_size), desc="Batches"):
        batch = audio_items[i:i + batch_size]
        uids = [item[0] for item in batch]
        wav_paths = [item[1] for item in batch]
        out_paths = [item[2] for item in batch]

        try:
            # Batch transcription
            texts = recognizer.transcribe_batch(wav_paths)

            for uid, wav_path, out_path, text in zip(uids, wav_paths, out_paths, texts):
                result = {"text": text,
                          "text_norm": postprocessing([text])[0]}

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] Batch error: {e}")
            # Fall back to individual processing
            for uid, wav_path, out_path in batch:
                try:
                    text = recognizer.transcribe(wav_path)
                    result = {"text": text,
                              "text_norm": postprocessing([text])[0]}

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"[ERROR] {wav_path}: {e}")


def run_single_worker(audio_dict, out_dir, postprocessing, model_id, batch_size):
    """Run transcription with a single worker (recommended for GPU)."""
    recognizer = ParakeetRecognizer(model_id=model_id, batch_size=batch_size)

    # Collect audio files that need processing
    audio_items = []
    for uid, wav_path in audio_dict.items():
        out_path = Path(out_dir) / f"{uid}.json"
        if not out_path.exists():
            audio_items.append((uid, wav_path, out_path))

    if not audio_items:
        print("All files already processed.")
        return

    print(f"Processing {len(audio_items)} files...")

    # Process in batches for efficiency
    for i in tqdm.tqdm(range(0, len(audio_items), batch_size), desc="Batches"):
        batch = audio_items[i:i + batch_size]
        uids = [item[0] for item in batch]
        wav_paths = [item[1] for item in batch]
        out_paths = [item[2] for item in batch]

        try:
            # Batch transcription
            texts = recognizer.transcribe_batch(wav_paths)

            for uid, wav_path, out_path, text in zip(uids, wav_paths, out_paths, texts):
                result = {"text": text,
                          "text_norm": postprocessing([text])[0]}

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] Batch error: {e}")
            # Fall back to individual processing
            for uid, wav_path, out_path in batch:
                try:
                    text = recognizer.transcribe(wav_path)
                    result = {"text": text,
                              "text_norm": postprocessing([text])[0]}

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"[ERROR] {wav_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parakeet ASR benchmark on YTL test datasets")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to .nemo model file or HuggingFace model ID (e.g., nvidia/parakeet-tdt-0.6b)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of workers (default: 1, recommended for GPU)"
    )
    args = parser.parse_args()

    # Configuration from arguments
    workers = args.workers
    batch_size = args.batch_size
    model_id = args.model

    print(f"Model: {model_id}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")

    all_wers = {}
    for d in DATASETS:
        print(f"\n{'=' * 60}\nProcessing dataset: {d}\n{'=' * 60}")

        save_dir = f'./ytl_parakeet/{d}'
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        audio_dict = {}
        ref_transcript = {}

        all_data = pd.read_csv(DATASETS[d], sep='\t')
        tsv_dir = Path(DATASETS[d]).parent  # Get the directory containing the TSV file
        assert len(all_data['utterance_id'].tolist()) == all_data['utterance_id'].nunique()
        print(f'duration is {all_data["duration"].sum() / 3600:.2f} hours')
        lang_postprocessing = postprocess_text_mal
        for idx, row in all_data.iterrows():
            audio_filepath = tsv_dir / row["path"]  # Construct full path relative to TSV location
            audio_dict[str(Path(audio_filepath).stem)] = str(audio_filepath)
            ref_transcript[str(Path(audio_filepath).stem)] = row["sentence"]

        if workers == 1:
            # Single worker mode (recommended for GPU)
            run_single_worker(audio_dict, save_dir, lang_postprocessing, model_id, batch_size)
        else:
            # Multi-worker mode (for CPU or multi-GPU setups)
            audio_dict_slices = split_dict(audio_dict, workers)
            args = [(slice, save_dir, lang_postprocessing, model_id, batch_size) for slice in audio_dict_slices]

            with mp.Pool(processes=workers) as pool:
                pool.map(job_audio, args)

        WER = WordErrorRate()
        json_files = {l.stem: str(l) for l in list(Path(save_dir).glob('*.json'))}
        count = 0
        for utt in json_files:
            try:
                with open(json_files[utt], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                hyp = data.get("text_norm", "")
                if count % 50 == 0:
                    print(f'utt {utt} \n hyp {hyp} \n ref {ref_transcript[utt]}')
                WER.update([hyp], [ref_transcript[utt]])
                count += 1
            except Exception as e:
                print(f"Error processing {utt}: {e}")

        _wer = WER.compute()
        print(f"Final WER for {d}: {_wer:.3f}")
        all_wers[d] = float(_wer)
        WER.reset()

    print(f"\n{'=' * 60}\nAll Results:\n{'=' * 60}")
    for dataset_name, wer_value in all_wers.items():
        print(f"{dataset_name}: {wer_value:.3f}")
    print(all_wers)
