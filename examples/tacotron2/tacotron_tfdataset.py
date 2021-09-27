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
import yaml

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class CharactorMelTFDataset(AbstractDataset):
    """Tensorflow Charactor Mel dataset."""

    feature_description = {
        "utt_ids": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.VarLenFeature(tf.int64),
        "mel_gts": tf.io.VarLenFeature(tf.float32),
        "speaker_ids": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'g_attentions': tf.io.VarLenFeature(tf.float32),
        'mel_lengths': tf.io.FixedLenFeature([], tf.int64),
        'input_lengths': tf.io.FixedLenFeature([], tf.int64),
    }

    def __init__(
        self,
        dataset,
        root_dir,
        tfrecord_query="train-*.tfrec",
        stats_query="train_stats.yml",
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
            tfrecord_query (str):
            stats_query (str):
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

        self.tfrecord_files = tf.io.gfile.glob(os.path.join(root_dir, tfrecord_query))
        assert len(self.tfrecord_files) != 0, f"Not found any tfrecord files in ${root_dir}."

        stats_files = tf.io.gfile.glob(os.path.join(root_dir, stats_query))
        assert len(self.tfrecord_files) != 0, f"Not found any stats files in ${root_dir}."

        with tf.io.gfile.GFile(stats_files[0]) as f:
            self.stats = yaml.load(f, Loader=yaml.Loader)

        self.reduction_factor = reduction_factor
        self.mel_length_threshold = mel_length_threshold
        self.mel_pad_value = mel_pad_value
        self.char_pad_value = char_pad_value
        self.ga_pad_value = ga_pad_value
        self.g = g
        self.use_fixed_shapes = use_fixed_shapes

        self.len_dataset = self.stats['size']
        self.max_char_length = self.stats['max_char_length']
        self.max_mel_length = self.stats['max_mel_length']

        if self.max_mel_length % self.reduction_factor != 0:
            self.max_mel_length = (
                    self.max_mel_length
                    + self.reduction_factor
                    - self.max_mel_length % self.reduction_factor
            )

    def get_args(self):
        raise NotImplemented('never use this!')

    def generator(self, utt_ids):
        raise NotImplemented("Never use this!")

    def _parse_tfrecord(self, example_proto):
        parsed = tf.io.parse_single_example(example_proto, self.feature_description)
        utt_ids = tf.cast(parsed['utt_ids'], tf.int32)
        input_ids = tf.sparse.to_dense(parsed['input_ids'])
        input_ids = tf.cast(input_ids, tf.int32)
        mel_gts = tf.sparse.to_dense(parsed['mel_gts'])
        mel_gts = tf.reshape(mel_gts, (-1, 80))  # 80 is num_mel
        speaker_ids = tf.cast(parsed['speaker_ids'], tf.int32)
        mel_lengths = tf.cast(parsed['mel_lengths'], tf.int32)
        input_lengths = tf.cast(parsed['input_lengths'], tf.int32)
        g_attentions = tf.sparse.to_dense(parsed['g_attentions'])
        g_attentions = tf.reshape(g_attentions, (input_lengths, mel_lengths))

        items = {
            "utt_ids": utt_ids,
            "input_ids": input_ids,
            'input_lengths': input_lengths,
            "speaker_ids": speaker_ids,
            "mel_gts": mel_gts,
            'mel_lengths': mel_lengths,
            'real_mel_lengths': mel_lengths,
            'g_attentions': g_attentions,
        }
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

        datasets = tf.data.TFRecordDataset(self.tfrecord_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        datasets = datasets.map(self._parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padding value.
        padding_values = {
            "utt_ids": 0,
            "input_ids": self.char_pad_value,
            "input_lengths": 0,
            "speaker_ids": 0,
            "mel_gts": self.mel_pad_value,
            "mel_lengths": 0,
            "real_mel_lengths": 0,
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
        raise NotImplemented("Never use this!")

    def get_len_dataset(self):
        return self.len_dataset

    def __name__(self):
        return "CharactorMelTFDataset"
