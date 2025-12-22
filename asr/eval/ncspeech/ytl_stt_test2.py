import time
import openai
import tqdm
import re
import malaya
import random
import json
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from torchmetrics.text import WordErrorRate


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


class YTLASR:
    def __init__(self):
        api_key = 'API_KEY'

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.ytlailabs.tech/v1"
        )

    def transcribe(self, wav_path):
        with open(wav_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="ilmu-preview-asr",
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
    audio_dict_slice, out_dir, postprocessing = args
    recognizer = YTLASR()

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

            time.sleep(0.5)  # prevent spam API

        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")


if __name__ == "__main__":
    workers = 4

    datasets = {
        'fleurs_test': './YTL_testsets/fleurs_test.tsv',
        'malay_conversational': './YTL_testsets/malay_conversational_meta.tsv',
        'malay_scripted': './YTL_testsets/malay_scripted_meta.tsv',
    }

    all_wers = {}
    for d in datasets:
        print(f"\n{'=' * 60}\nProcessing dataset: {d}\n{'=' * 60}")

        save_dir = f'./ytl_labs/{d}'
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        audio_dict = {}
        ref_transcript = {}

        all_data = pd.read_csv(datasets[d], sep='\t')
        assert len(all_data['utterance_id'].tolist()) == all_data['utterance_id'].nunique()
        print(f'duration is {all_data["duration"].sum() / 3600}')
        lang_postprocessing = postprocess_text_mal
        for idx, row in all_data.iterrows():
            audio_filepath = row["path"]
            audio_dict[str(Path(audio_filepath).stem)] = audio_filepath
            ref_transcript[str(Path(audio_filepath).stem)] = row["sentence"]

        audio_dict_slices = split_dict(audio_dict, workers)

        args = [(slice, save_dir, lang_postprocessing) for slice in audio_dict_slices]

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
            except Exception as e:
                print(f"Error processing {utt}: {e}")

        _wer = WER.compute()
        print(f"Final WER: {_wer:.3f}")
        WER.reset()