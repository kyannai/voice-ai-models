import pandas as pd
import tqdm
import re
import malaya

from faster_whisper import WhisperModel
from torchmetrics.text import WordErrorRate
from pathlib import Path


lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model = lm)
# stemmer = malaya.stem.huggingface()
normalizer_mal = malaya.normalizer.rules.load(corrector, None)

DATASETS = {
    "fleurs_test": "../test_data/YTL_testsets/fleurs_test.tsv",
    "malay_conversational": "../test_data/YTL_testsets/malay_conversational_meta.tsv",
    "malay_scripted": "../test_data/YTL_testsets/malay_scripted_meta.tsv",
}


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
    chars_to_ignore_regex_normalise = r"""[\/:\\;"−*`‑―‘’“”„~«»–—…\[\]\(\)\t\r\n!?,\.]"""
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


download_root = './faster_whisper'
Path(download_root).mkdir(parents=True, exist_ok=True)
model = WhisperModel("large-v3-turbo",
                     download_root=download_root,
                     device="cuda",
                     compute_type="float16")


def compute_transcript():
    lang_id = 'ms'
    postprocessing = postprocess_text_mal

    all_wers = {}
    for d in DATASETS:

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
            # audio, sr = torchaudio.load(audio_files[audio_utt])
            segments, info = model.transcribe(audio_dict[audio_utt],
                                              language=lang_id,
                                              without_timestamps=False,
                                              vad_filter=True,
                                              beam_size=5)
            text = ''
            for segment in segments:
                text += segment.text + ' '
            sys_transcription[audio_utt] = postprocessing([text])[0]

        WER = WordErrorRate()
        for utt in sys_transcription:
            hyp = sys_transcription[utt]

            # print(f'utt {utt} \n hyp {hyp} \n ref {ref_transcript[utt]}')
            WER.update(hyp, ref_transcript[utt])

        _wer = WER.compute()
        print(f"Final WER: {_wer}")
        WER.reset()
    print(all_wers)


if __name__ == "__main__":
    compute_transcript()