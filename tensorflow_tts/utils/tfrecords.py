
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter

# feature_description for reading tfrecords
feature_description = {
    "utt_id": tf.io.FixedLenFeature([], tf.int64),
    "input_ids": tf.io.VarLenFeature(tf.int64),
    "raw_feat": tf.io.VarLenFeature(tf.float32),
    "norm_feat": tf.io.VarLenFeature(tf.float32),
    "speaker_ids": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "wave": tf.io.VarLenFeature(tf.float32),
    'raw_energy': tf.io.VarLenFeature(tf.float32),
    'raw_f0': tf.io.VarLenFeature(tf.float32),
}


def write_tfrecord(output_path, examples):
    with TFRecordWriter(output_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def create_tf_example(utt_id, input_ids, raw_feat, norm_feat, speaker_id, raw_energy, raw_f0, wave):
    return tf.train.Example(features=tf.train.Features(feature={
        'utt_id': int64_feature(utt_id),
        'input_ids': int64_list_feature(input_ids),
        'raw_feat': float_list_feature(raw_feat.reshape(-1)),
        'norm_feat': float_list_feature(norm_feat.reshape(-1)),
        'speaker_ids': int64_feature(speaker_id),
        'wave': float_list_feature(wave),
        'raw_energy': float_list_feature(raw_energy),
        'raw_f0': float_list_feature(raw_f0),
    }))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))