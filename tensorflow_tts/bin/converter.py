import argparse
import glob
import logging
import os
import re
import threading

import numpy as np

from tensorflow_tts.utils.tfrecords import create_tf_example
from tensorflow_tts.utils.tfrecords import write_tfrecord

os.environ["CUDA_VISIBLE_DEVICES"] = ""

utt_id_map = {}
next_utt_id = 1
utt_lock = threading.Lock()


def get_utt_id(utt):
    global utt_id_map
    global next_utt_id
    with utt_lock:
        if utt not in utt_id_map:
            utt_id_map[utt] = next_utt_id
            next_utt_id += 1
    return utt_id_map[utt]


def find_file(path, pred):
    return [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], pred))]


def np_load(path, pred):
    return np.load(find_file(path, pred)[0])


def parse_and_config():
    """Parse arguments and set configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and text features "
        "(See detail in tensorflow_tts/bin/preprocess_dataset.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="Directory containing the dataset files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
        help="Output directory where features will be saved.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO and WARNING, 2: INFO, WARNING, and ERROR",
    )
    args = parser.parse_args()

    # set logger
    FORMAT = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.WARNING, 2: logging.ERROR}
    logging.basicConfig(level=log_level[args.verbose], format=FORMAT)

    # load config
    config = {}
    config.update(vars(args))
    return config


def utt_id_list(basedir):
    target = os.path.join(basedir, 'ids')
    id_pat = re.compile(r'(.*?)-ids.npy')
    utt_ids = set()
    for filename in os.listdir(target):
        found = id_pat.findall(filename)
        if len(found) > 0:
            utt_ids.add(found[0])
    return utt_ids


def load_item(rootdir, utt_id_str):
    utt_id = get_utt_id(utt_id_str)
    input_ids = np_load(rootdir, f'{utt_id_str}-ids.npy')
    raw_feat = np_load(rootdir, f'{utt_id_str}-raw-feats.npy')
    norm_feat = np_load(rootdir, f'{utt_id_str}-norm-feats.npy')
    speaker_ids = 0
    wave = np_load(rootdir, f'{utt_id_str}-wave.npy')
    raw_energy = np_load(rootdir, f'{utt_id_str}-raw-energy.npy')
    raw_f0 = np_load(rootdir, f'{utt_id_str}-raw-f0.npy')
    return {
        'utt_id': utt_id,
        'input_ids': input_ids,
        'raw_feat': raw_feat,
        'norm_feat': norm_feat,
        'speaker_ids': speaker_ids,
        'wave': wave,
        'raw_energy': raw_energy,
        'raw_f0': raw_f0,
    }


def _convert_to_example(item):
    return create_tf_example(item['utt_id'], item['input_ids'], item['raw_feat'], item['norm_feat'],
                             item['speaker_ids'], item['wave'], item['raw_energy'], item['raw_f0'])


def convert():
    """Run preprocessing process and compute statistics for normalizing."""
    config = parse_and_config()

    # check output directories
    os.makedirs(config['outdir'], exist_ok=True)

    train_utt_ids = utt_id_list(os.path.join(config['rootdir'], 'train'))
    valid_utt_ids = utt_id_list(os.path.join(config['rootdir'], 'valid'))

    logging.info(f"Training items: {len(train_utt_ids)}")
    logging.info(f"Validation items: {len(valid_utt_ids)}")

    train_items = map(lambda id: load_item(config['rootdir'], id), train_utt_ids)
    train_examples = map(lambda item: _convert_to_example(item), train_items)

    train_tfrecord_path = os.path.join(config['outdir'], 'train.tfrecord')
    write_tfrecord(train_tfrecord_path, train_examples)

    valid_items = map(lambda id: load_item(config['rootdir'], id), valid_utt_ids)
    valid_examples = map(lambda item: _convert_to_example(item), valid_items)

    valid_tfrecord_path = os.path.join(config['outdir'], 'valid.tfrecord')
    write_tfrecord(valid_tfrecord_path, valid_examples)

    logging.info('converting completed.')


if __name__ == '__main__':
    convert()
