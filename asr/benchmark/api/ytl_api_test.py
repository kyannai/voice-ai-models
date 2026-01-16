import argparse
import json
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path

import malaya
import openai
import pandas as pd
import tqdm
from dotenv import load_dotenv
from torchmetrics.text import WordErrorRate

load_dotenv()

DATASETS = {
    "fleurs_test": "../test_data/YTL_testsets/fleurs_test.tsv",
    "malay_conversational": "../test_data/YTL_testsets/malay_conversational_meta.tsv",
    "malay_scripted": "../test_data/YTL_testsets/malay_scripted_meta.tsv",
}


lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model=lm)
# stemmer = malaya.stem.huggingface()
normalizer_mal = malaya.normalizer.rules.load(corrector, None)

chars_to_ignore_regex_normalise = r"""[\/:\\;"−*`‑―‘’“”„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
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


def resolve_api_config(env_name: str | None) -> tuple[str, str]:
    if env_name == "staging":
        api_key = os.getenv("ILMU_STAGING_API_KEY")
        base_url = os.getenv("ILMU_STAGING_URL")
    elif env_name == "production":
        api_key = os.getenv("ILMU_PRODUCTION_API_KEY")
        base_url = os.getenv("ILMU_PRODUCTION_URL")
    else:
        api_key = os.getenv("ILMU_API_KEY")
        base_url = os.getenv("ILMU_API_BASE_URL", "https://api.ytlailabs.tech/v1")

    if not api_key or not base_url:
        env_label = env_name or "default"
        raise ValueError(
            f"Missing API config for {env_label}. "
            "Set ILMU_STAGING_URL/ILMU_STAGING_API_KEY or "
            "ILMU_PRODUCTION_URL/ILMU_PRODUCTION_API_KEY in .env."
        )

    return api_key, base_url


class YTLASR:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def transcribe(self, wav_path):
        with open(wav_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f
            )
        return response.text


def split_dict(data, num_splits):
    """Split dictionary into N roughly equal parts."""
    keys = list(data.keys())
    random.shuffle(keys)
    split_keys = [keys[i::num_splits] for i in range(num_splits)]
    return [{k: data[k] for k in subset} for subset in split_keys]


def job_audio(args):
    audio_dict_slice, out_dir, postprocessing, api_key, base_url, model, sleep_s = args
    recognizer = YTLASR(api_key=api_key, base_url=base_url, model=model)

    for uid, wav_path in tqdm.tqdm(audio_dict_slice.items()):
        out_path = Path(out_dir) / f"{uid}.json"
        if out_path.exists():
            continue

        try:
            text = recognizer.transcribe(wav_path)
            result = {"text": text,
                      "text_norm": postprocessing([text])[0]}

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            time.sleep(sleep_s)  # prevent rate limit (429 errors)

        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch evaluation using YTL ASR API"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="bukit-tinggi-v2",
        help="Model name (default: bukit-tinggi-v2)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="production",
        choices=["staging", "production"],
        help="API environment (default: production)"
    )
    cli_args = parser.parse_args()

    try:
        api_key, base_url = resolve_api_config(cli_args.env)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if cli_args.env == "staging":
        workers = 1
        sleep_s = 0.5
    else:
        workers = 4
        sleep_s = 0.1

    all_wers = {}
    for d in DATASETS:
        print(f"\n{'=' * 60}\nProcessing dataset: {d}\n{'=' * 60}")

        save_dir = f'./ytl_labs/{d}'
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        audio_dict = {}
        ref_transcript = {}

        all_data = pd.read_csv(DATASETS[d], sep='\t')
        tsv_dir = Path(DATASETS[d]).parent  # Get the directory containing the TSV file
        assert len(all_data['utterance_id'].tolist()) == all_data['utterance_id'].nunique()
        print(f'duration is {all_data["duration"].sum() / 3600}')
        lang_postprocessing = postprocess_text_mal
        for idx, row in all_data.iterrows():
            audio_filepath = tsv_dir / row["path"]  # Construct full path relative to TSV location
            audio_dict[str(Path(audio_filepath).stem)] = str(audio_filepath)
            ref_transcript[str(Path(audio_filepath).stem)] = row["sentence"]

        audio_dict_slices = split_dict(audio_dict, workers)

        job_args = [
            (slice, save_dir, lang_postprocessing, api_key, base_url, cli_args.model, sleep_s)
            for slice in audio_dict_slices
        ]

        with mp.Pool(processes=workers) as pool:
            pool.map(job_audio, job_args)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())