from pathlib import Path
from typing import List

import datasets
import json
import os


_LANGUAGES = {
    "id": {
        "Language": "Indonesian",
        "Description": "Magic Data dataset for Indonesian",
        "Date": "2021",
    },
    "tr": {
        "Language": "Turkish",
        "Description": "Magic Data dataset for Turkish",
        "Date": "2021",
    }
}

_CITATION = """\
@inproceedings{lestari2006titmlidn,
  title={A large vocabulary continuous speech recognition system for Indonesian language},
  author={Lestari, Dessi Puji and Iwano, Koji and Furui, Sadaoki},
  booktitle={15th Indonesian Scientific Conference in Japan Proceedings},
  pages={17--22},
  year={2006}
}
"""

_DESCRIPTION = """\
TITML-IDN (Tokyo Institute of Technology Multilingual - Indonesian) is collected to build a pioneering Indonesian Large Vocabulary Continuous Speech Recognition (LVCSR) System. In order to build an LVCSR system, high accurate acoustic models and large-scale language models are essential. Since Indonesian speech corpus was not available yet, we tried to collect speech data from 20 Indonesian native speakers (11 males and 9 females) to construct a speech corpus for training the acoustic model based on Hidden Markov Models (HMMs). A text corpus which was collected by ILPS, Informatics Institute, University of Amsterdam, was used to build a 40K-vocabulary dictionary and a n-gram language model.
"""

_HOMEPAGE = "http://research.nii.ac.jp/src/en/TITML-IDN.html"

_LICENSE = "For research purposes only. If you use this corpus, you have to cite (Lestari et al, 2006)."

_URLs = {"titml-idn": "https://huggingface.co/datasets/holylovenia/TITML-IDN/resolve/main/IndoLVCSR.zip"}


class TitmlIdnConfig(datasets.BuilderConfig):
    """BuilderConfig for MagicData."""

    def __init__(self, name, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.language = kwargs.pop("language", None)
        self.description = kwargs.pop("description", None)
        self.date = kwargs.pop("date", None)
        super(TitmlIdnConfig, self).__init__(name=name, **kwargs)


class TitmlIdn(datasets.GeneratorBasedBuilder):
    """TITML-IDN is a speech recognition dataset containing Indonesian speech collected with transcriptions from newpaper and magazine articles."""


    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        TitmlIdnConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            description=_LANGUAGES[lang_id]["Description"],
            date=_LANGUAGES[lang_id]["Date"],
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "speaker_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=16_000),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath("/root/whisper_finetuning/local_dataset/titlm/")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir},
            ),
        ]

    def _generate_examples(self, filepath: Path, n_speakers=20):
        counter = -1
        if self.config.name in ["id", "tr"]:
            for speaker_id in range(1, n_speakers + 1):
                speaker_id = str(speaker_id).zfill(2)
                dir_path = os.path.join(filepath, speaker_id)
                transcription_path = os.path.join(dir_path, "script~")
                with open(transcription_path, "r+") as f:
                    for line in f:
                        counter += 1
                        audio_id = line[2:8]
                        text = line[9:].strip()
                        wav_path = os.path.join(dir_path, "{}.wav".format(audio_id))
                        if os.path.exists(wav_path):
                            result = {}
                            result["path"] = wav_path
                            with open(wav_path, "rb") as file:
                                result["audio"] = {"path": wav_path, "bytes": file.read()}
                            ex = {
                                "speaker_id": speaker_id,
                                "path": wav_path,
                                "audio": result,
                                "sentence": text,
                            }
                            yield counter, ex