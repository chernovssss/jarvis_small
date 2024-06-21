import sys

import numpy as np
import pyaudio


class FasterWhisperASR:
    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None):

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    @staticmethod
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            print(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
        model_size_or_path = model_dir

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio, init_prompt=""):

        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )

        return list(segments)

    @staticmethod
    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    @staticmethod
    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"
