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
"""Tacotron Related Dataset modules."""

import itertools
import logging
import os
import random

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


class CharactorMelDataset(AbstractDataset):
    """Tensorflow Charactor Mel dataset."""

    def __init__(
        self,
        dataset,
        root_dir,
        charactor_query="*-ids.npy",
        mel_query="*-norm-feats.npy",
        align_query="",
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        mel_length_threshold=0,
        reduction_factor=1,
        mel_pad_value=0.0,
        char_pad_value=0,
        ga_pad_value=-1.0,
        g=0.2,
        use_fixed_shapes=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            align_query (str): Query to find FAL files in root_dir. If empty, we use stock guided attention loss
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            reduction_factor (int): Reduction factor on Tacotron-2 paper.
            mel_pad_value (float): Padding value for mel-spectrogram.
            char_pad_value (int): Padding value for charactor.
            ga_pad_value (float): Padding value for guided attention.
            g (float): G value for guided attention.
            use_fixed_shapes (bool): Use fixed shape for mel targets or not.
            max_char_length (int): maximum charactor length if use_fixed_shapes=True.
            max_mel_length (int): maximum mel length if use_fixed_shapes=True

        """
        # find all of charactor and mel files.
        self.tfrecord_files = getflist(root_dir, "tfrecords", "*.tfrecords")
        assert len(self.tfrecord_files) != 0, f"Not found any tfrecord files in ${root_dir}."

        # TODO: align_files
        # self.align_files = []
        #
        # if len(align_query) > 1:
        #     align_files = sorted(find_files(root_dir, align_query))
        #     assert len(align_files) == len(
        #         mel_files
        #     ), f"Number of align files ({len(align_files)}) and mel files ({len(mel_files)}) are different"
        #     logging.info("Using FAL loss")
        #     self.align_files = align_files
        # else:
        #     logging.info("Using guided attention loss")


        # set global params
        maxlens_path = os.path.join(root_dir, "maxlens.npy")

        max_lens = getgcsnp(maxlens_path)
        self.max_mel_length = max_lens[0]
        self.max_char_length = max_lens[1]
        self.len_dts = max_lens[2]

        self.reduction_factor = reduction_factor
        self.mel_length_threshold = mel_length_threshold
        self.mel_pad_value = mel_pad_value
        self.char_pad_value = char_pad_value
        self.ga_pad_value = ga_pad_value
        self.g = g
        self.use_fixed_shapes = use_fixed_shapes

        if self.max_mel_length % self.reduction_factor != 0:
            self.max_mel_length = (
                self.max_mel_length
                + self.reduction_factor
                - self.max_mel_length % self.reduction_factor
            )

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]
            align_file = self.align_files[i] if len(self.align_files) > 1 else ""

            items = {
                "utt_ids": utt_id,
                "mel_files": mel_file,
                "charactor_files": charactor_file,
                "align_files": align_file,
            }

            yield items

    def _parse_tfrecord(self, example_proto):
        parsed_ex = tf.io.parse_single_example(example_proto, self.feature_description)
        input_ids = tf.io.parse_tensor(parsed_ex["input_ids"], tf.int32)
        input_ids = tf.reshape(input_ids, [parsed_ex["input_lengths"]])
        mel = tf.io.parse_tensor(parsed_ex["mel_gts"], tf.float32)
        mel = tf.reshape(mel, [parsed_ex["real_mel_lengths"], parsed_ex["num_mels"]])
        # TODO:
        #     g_att = (
        #         tf.numpy_function(np.load, [items["align_files"]], tf.float32)
        #         if len(self.align_files) > 1
        #         else None
        #     )
        g_att = (
            None
        )

        # padding mel to make its length is multiple of reduction factor.
        # TODO:

        items = {
            "utt_ids": parsed_ex["utt_ids"],
            "input_ids": tf.cast(input_ids, tf.int64),
            "input_lengths": parsed_ex["input_lengths"],
            "speaker_ids": tf.cast(parsed_ex["speaker_ids"], tf.int32),
            "mel_gts": mel,
            "mel_lengths": tf.cast(parsed_ex["mel_lengths"], tf.int32),
            "real_mel_lengths": parsed_ex["real_mel_lengths"],
            "g_attentions": g_att,
        }
        return items

    def _guided_attention(self, items):
        """Guided attention. Refer to page 3 on the paper (https://arxiv.org/abs/1710.08969)."""
        items = items.copy()
        # mel len needs to be int32 for training not to err, but int64 here for calculating not to err.
        mel_len = tf.cast(items["mel_lengths"] // self.reduction_factor, tf.int64)
        char_len = items["input_lengths"]
        xv, yv = tf.meshgrid(tf.range(char_len), tf.range(mel_len), indexing="ij")
        f32_matrix = tf.cast(yv / mel_len - xv / char_len, tf.float32)
        items["g_attentions"] = 1.0 - tf.math.exp(
            -(f32_matrix ** 2) / (2 * self.g ** 2)
        )
        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
        drop_remainder=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.TFRecordDataset(self.tfrecord_files)
        self.feature_description = {
            "utt_ids": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "input_ids": tf.io.FixedLenFeature([], tf.string, default_value=""), # TODO: tf.string?
            "input_lengths": tf.io.FixedLenFeature([], tf.int64), # TODO: tf.int32?
            "mel_gts": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "mel_lengths": tf.io.FixedLenFeature([], tf.int64),
            "real_mel_lengths": tf.io.FixedLenFeature([], tf.int64),
            "speaker_ids": tf.io.FixedLenFeature([], tf.int64),
            "num_mels": tf.io.FixedLenFeature([], tf.int64),
        }

        # load data
        datasets = datasets.map(self._parse_tfrecord, tf.data.experimental.AUTOTUNE)

        # calculate guided attention
        if len(self.align_files) < 1:
            datasets = datasets.map(
                lambda items: self._guided_attention(items),
                tf.data.experimental.AUTOTUNE,
            )

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        print("dataset cardinality: " + str(self.len_dts))

        if is_shuffle:
            datasets = datasets.shuffle(
                self.len_dts,
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padding value.
        padding_values = {
            "utt_ids": " ",
            "input_ids": tf.cast(self.char_pad_value, tf.int64),
            "input_lengths": tf.cast(0, tf.int64),
            "speaker_ids": 0,
            "mel_gts": self.mel_pad_value,
            "mel_lengths": 0,
            "real_mel_lengths": tf.cast(0, tf.int64),
            "g_attentions": self.ga_pad_value,
        }

        # define padded shapes.
        padded_shapes = {
            "utt_ids": [],
            "input_ids": [None]
            if self.use_fixed_shapes is False
            else [self.max_char_length],
            "input_lengths": [],
            "speaker_ids": [],
            "mel_gts": [None, 80]
            if self.use_fixed_shapes is False
            else [self.max_mel_length, 80],
            "mel_lengths": [],
            "real_mel_lengths": [],
            "g_attentions": [None, None]
            if self.use_fixed_shapes is False
            else [self.max_char_length, self.max_mel_length // self.reduction_factor],
        }

        datasets = datasets.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder,
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mel_files": tf.string,
            "charactor_files": tf.string,
            "align_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorMelDataset"
