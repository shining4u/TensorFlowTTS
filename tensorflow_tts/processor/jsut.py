# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for KSS dataset."""

import os
import re
import MeCab
import jaconv

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from tensorflow_tts.processor import BaseProcessor
from tensorflow_tts.utils import cleaners
from tensorflow_tts.utils.japanese import symbols as JSUT_SYMBOLS
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


@dataclass
class JSUTProcessor(BaseProcessor):
    """JSUT processor."""

    cleaner_names: str = None
    positions = {
        "wave_file": 0,
        "text": 1,
    }
    # train_f_name: str = "transcript_utf8.txt"
    recordings = [
        "basic5000",
        "countersuffix26",
        "loanword128",
        "onomatopee300",
        "precedent130",
        "repeat500",
        "travel1000",
        "utparaphrase512",
        "voiceactress100",
    ]
    transcripts = { rec_dir: os.path.join(rec_dir, "transcript_utf8.txt") for rec_dir in recordings }
    _tagger = None

    def create_items(self):
        if self.data_dir:
            self.items = []
            for rec_dir, transcript in self.transcripts.items():
                with open(os.path.join(self.data_dir, transcript), encoding='utf-8') as f:
                    self.items += [self.split_line(self.data_dir, rec_dir, line, ':') for line in f]
            self.items = list(filter(None, self.items))

    def split_line(self, data_dir, rec_dir, line, split):
        parts = line.strip().split(split)
        wav_file = parts[self.positions["wave_file"]] + ".wav"
        wav_path = os.path.join(data_dir, rec_dir, 'wav', wav_file)
        text = parts[self.positions['text']]
        speaker_name = "jsut"

        # test
        cleaned_text = self.clean_text(text)
        if not set(list(cleaned_text)).issubset(self.symbol_to_id):
            # missing = set(list(cleaned_text)) - set(self.symbol_to_id.keys())
            # missed = ''.join(missing)
            # if not re.match(r'\d+', missed):
            #     print(f'FAILED missing: {missing}, line: {text}, cleaned_text: {cleaned_text}')
            return None

        return text, wav_path, speaker_name

    def setup_eos_token(self):
        return "eos"

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        text = self.clean_text(text)
        return self._symbols_to_sequence(list(text)) + [self.eos_id]

    def clean_text(self, text):
        for c in [" ", "　", "「", "」", "『", "』", "・", "【", "】", "（", "）", "(", ")"]:
            text = text.replace(c, "")
        text = text.replace("!", "！")
        text = text.replace("?", "？")

        text = self.normalize_delimitor(text)
        text = jaconv.normalize(text)
        text = self.mix_pronunciation(text)
        text = jaconv.hira2kata(text)
        text = self.add_punctuation(text)
        return text

    #
    def _yomi(self, mecab_result):
        tokens = []
        yomis = []
        for line in mecab_result.split("\n")[:-1]:
            s = line.split("\t")
            if len(s) == 1:
                break
            token, rest = s
            rest = rest.split(",")
            tokens.append(token)
            yomi = rest[7] if len(rest) > 7 else None
            yomi = None if yomi == "*" else yomi
            yomis.append(yomi)
        return tokens, yomis

    def _mix_pronunciation(self, tokens, yomis):
        return "".join(
            yomis[idx] if yomis[idx] is not None else tokens[idx]
            for idx in range(len(tokens)))

    def mix_pronunciation(self, text):
        if self._tagger is None:
            # self._tagger = MeCab.Tagger("")
            # 7166
            self._tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
            # 7511
        tokens, yomis = self._yomi(self._tagger.parse(text))
        return self._mix_pronunciation(tokens, yomis)

    def add_punctuation(self, text):
        last = text[-1]
        if last not in [".", ",", "、", "。", "！", "？", "!", "?"]:
            text = text + "。"
        return text

    def normalize_delimitor(self, text):
        text = text.replace(",", "、")
        text = text.replace(".", "。")
        text = text.replace("，", "、")
        text = text.replace("．", "。")
        return text
    #

    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text

    def _symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id
