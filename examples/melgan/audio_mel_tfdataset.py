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
import yaml

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class AudioMelTFDataset(AbstractDataset):
    """Tensorflow Audio Mel dataset."""

    feature_description = {
        "utt_ids": tf.io.FixedLenFeature([], tf.int64),
        "audios": tf.io.VarLenFeature(tf.float32),
        "mels": tf.io.VarLenFeature(tf.float32),
        'mel_lengths': tf.io.FixedLenFeature([], tf.int64),
        "audio_lengths": tf.io.FixedLenFeature([], tf.int64),
    }

    def __init__(
        self,
        root_dir,
        tfrecord_query="train-*.tfrec",
        stats_query="train_stats.yml",
        audio_length_threshold=0,
        mel_length_threshold=0,
    ):
        """Initialize dataset.
        Args:
            root_dir (str): Root directory including dumped files.
            tfrecord_query (str):
            stats_query (str):
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
        """

        self.tfrecord_files = tf.io.gfile.glob(os.path.join(root_dir, tfrecord_query))
        assert len(self.tfrecord_files) != 0, f"Not found any tfrecord files in ${root_dir}."

        stats_files = tf.io.gfile.glob(os.path.join(root_dir, stats_query))
        assert len(self.tfrecord_files) != 0, f"Not found any stats files in ${root_dir}."

        with tf.io.gfile.GFile(stats_files[0]) as f:
            self.stats = yaml.load(f, Loader=yaml.Loader)

        # set global params
        self.len_dataset = self.stats['size']
        self.audio_length_threshold = audio_length_threshold
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        raise NotImplemented('never use this!')

    def generator(self, utt_ids):
        raise NotImplemented('never use this!')

    def _parse_tfrecord(self, example_proto):
        parsed = tf.io.parse_single_example(example_proto, self.feature_description)
        utt_ids = tf.cast(parsed['utt_ids'], tf.int32)
        audios = tf.sparse.to_dense(parsed['audios'])
        mels = tf.sparse.to_dense(parsed['mels'])
        mels = tf.reshape(mels, (-1, 80))  # 80 is num_mel
        mel_lengths = tf.cast(parsed['mel_lengths'], tf.int32)
        audio_lengths = tf.cast(parsed['audio_lengths'], tf.int32)

        item = {
            "utt_ids": utt_ids,
            "audios": audios,
            "mels": mels,
            "mel_lengths": mel_lengths,
            "audio_lengths": audio_lengths,
        }
        return item

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""

        datasets = tf.data.TFRecordDataset(self.tfrecord_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        datasets = datasets.with_options(options)

        datasets = datasets.map(self._parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
        raise NotImplemented('never use this!')

    def get_len_dataset(self):
        return self.len_dataset

    def __name__(self):
        return "AudioMelTFDataset"
