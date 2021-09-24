# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Dataset modules."""

import logging
import os

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files

import tensorflow_datasets as tfds
from tensorflow.python.lib.io import file_io
from io import BytesIO


def getgcsfile(src):
    f = BytesIO(file_io.read_file_to_string(src, binary_mode=True))
    return f


def getgcsnp(npf):
    return np.load(getgcsfile(npf))


def getflist(root_d, subfolder_name, query):
    the_files = list(tfds.as_numpy(tf.data.Dataset.list_files(os.path.join(root_d, subfolder_name, query))))
    the_files = [x.decode('utf8') for x in the_files]
    the_files = sorted(the_files)
    return the_files


class AudioMelDataset(AbstractDataset):
    """Tensorflow Audio Mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        mel_query="*-raw-feats.npy",
        audio_load_fn=np.load,
        mel_load_fn=np.load,
        audio_length_threshold=0,
        mel_length_threshold=0,
    ):
        """Initialize dataset.
        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
        """
        # find all of audio and mel files.
        self.tfrecord_files = getflist(root_dir, "tfrecords", "*.tfrecords")
        assert len(self.tfrecord_files) != 0, f"Not found any tfrecord files in ${root_dir}."

        # set global params
        maxlens_path = os.path.join(root_dir, "maxlens.npy")
        max_lens = getgcsnp(maxlens_path)
        self.len_dts = max_lens[2]
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.audio_length_threshold = audio_length_threshold
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            audio_file = self.audio_files[i]
            mel_file = self.mel_files[i]

            items = {
                "utt_ids": utt_id,
                "audio_files": audio_file,
                "mel_files": mel_file,
            }

            yield items

    def _parse_tfrecord(self, example_proto):
        parsed_ex = tf.io.parse_single_example(example_proto, self.feature_description)
        mel = tf.io.parse_tensor(parsed_ex["mel_gts"], tf.float32)
        mel = tf.reshape(mel, [parsed_ex["real_mel_lengths"], parsed_ex["num_mels"]])
        audio = tf.io.parse_tensor(parsed_ex["audios"], tf.float32)
        audio = tf.reshape(audio, [parsed_ex["audio_len"]])

        items = {
            "utt_ids": tf.cast(parsed_ex["utt_ids"], tf.int32),
            "mels": mel,
            "audios": audio,
            "mel_lengths": tf.cast(parsed_ex["mel_lengths"], tf.int32),
            "audio_lengths": parsed_ex["audio_len"],
        }
        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.TFRecordDataset(self.tfrecord_files)
        self.feature_description = {
            "utt_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "input_lengths": tf.io.FixedLenFeature([], tf.int64),
            "mel_gts": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "mel_lengths": tf.io.FixedLenFeature([], tf.int64),
            "real_mel_lengths": tf.io.FixedLenFeature([], tf.int64),
            "speaker_ids": tf.io.FixedLenFeature([], tf.int64),
            "num_mels": tf.io.FixedLenFeature([], tf.int64),
            "audios": tf.io.FixedLenFeature([], tf.string),
            "audio_len": tf.io.FixedLenFeature([], tf.int64),
        }
        datasets = datasets.map(self._parse_tfrecord, tf.data.experimental.AUTOTUNE)

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )
        datasets = datasets.filter(
            lambda x: x["audio_lengths"] > self.audio_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            datasets = datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        # define padded shapes
        padded_shapes = {
            "utt_ids": [],
            "audios": [None],
            "mels": [None, 80],
            "mel_lengths": [],
            "audio_lengths": [],
        }

        # define padded values
        padding_values = {
            "utt_ids": 0,
            "audios": 0.0,
            "mels": 0.0,
            "mel_lengths": 0,
            "audio_lengths": 0,
        }

        datasets = datasets.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "audio_files": tf.string,
            "mel_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return self.len_dts

    def __name__(self):
        return "AudioMelDataset"
