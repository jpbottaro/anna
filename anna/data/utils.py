"""Simple utilities to fetch datasets."""

import os
import re
import sys
import pickle
import random
import urllib.request
import tensorflow as tf
import numpy as np

TRAIN_PATH = "train.tfrecords"
TEST_PATH = "test.tfrecords"
UNUSED_PATH = "unused.tfrecords"
LABELS_PATH = "labels.pickle"

# Find numbers, examples: -1 | 123 | 1.324e10 | 1,234.24
number_finder = re.compile(r"[+-]?(\d+,?)+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def tokenize(text,
             remove="\"#()*+<=>@[\\]^_`{|}~\t\n",
             separate="?!/'%$&,.;:",
             number_token="1"):
    """
    Tokenizes the given `text`. Removes all tokens in `remove`, and splits
    the ones in `separate`.

    If `number_token` is not None, all numbers are modified to this token.

    Args:
        text (str): a piece of text to tokenize
        remove (str): chars that should be removed
        separate (str): chars that should separate tokens (and kept)
        number_token (str): token to use for all numbers

    Returns:
        tokens (list[str]): list of tokens from `text`
    """
    if not text:
        return []

    if number_token:
        text = number_finder.sub(number_token, text)

    remover = str.maketrans({c: " " for c in remove})
    separator = str.maketrans({c: " " + c for c in separate})
    text = text.translate(remover)
    text = text.translate(separator)

    return [t for t in text.split() if t]


def add_special_tokens(voc, emb):
    """
    Extends a vocabulary with special tokens, like padding or unknowns.

    Args:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`

    Returns:
        new_voc (list[str]): same as `voc`, with special tokens added (e.g.
          "_PAD_" and "_UNK_").
        new_emb (numpy.array): same as `emb`, with special tokens embeddings.
    """
    # Reserve 0 for special padding token, 1 for unknown
    voc = ["_PAD_", "_UNK_"] + voc

    # Make padding token be all zeros
    pad_emb = np.zeros_like(emb[0])
    pad_emb = pad_emb[np.newaxis, :]

    # And unknown be an average of all embeddings
    unk_emb = np.sum(emb, 0) / emb.shape[0]
    unk_emb = unk_emb[np.newaxis, :]

    return voc, np.concatenate([pad_emb, unk_emb, emb], axis=0)


def reporthook(blocknum, blocksize, totalsize):
    """
    A hook that conforms with 'urllib.request.urlretrieve()' interface.

    It reports in stdout the current progress of the download, including
    a progress bar.
    """

    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 100 / totalsize
        done = int(50 * percent / 100)
        s = "\r{:05.2f}% [{}{}] {:.2f}mb".format(percent,
                                                 '=' * done,
                                                 ' ' * (50 - done),
                                                 totalsize / 2 ** 20)
        sys.stdout.write(s)
        if readsofar >= totalsize:
            sys.stdout.write("\n")
        sys.stdout.flush()
    # total size is unknown
    else:
        sys.stdout.write("read {:.2f}mb\n".format(readsofar / 2 ** 20))


def urlretrieve(url, path):
    """
    Same as 'urllib.urlretrieve()', but with a nice reporthook to show
    a progress bar.

    If 'path' exists, doesn't download anything.
    """
    if os.path.exists(path):
        print("Skipping: " + url)
    else:
        print("Downloading: " + url)
        urllib.request.urlretrieve(url, path, reporthook)


def create_folder(folder):
    """
    Creates the given folder in the filesystem.

    The whole path is created, just like 'mkdir -p' would do.
    """

    try:
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise


def make_example(title, text, labels):
    """
    Serializes a doc into a TFRecord example.

    Args:
        title (list[str]): title of the example
        text (list[str]): text of the example
        labels (list[str]): the list of expected labels for the example

    Returns:
        ex (str): serialized TFRecord example
    """
    ex = tf.train.SequenceExample()

    title_list = ex.feature_lists.feature_list["title"]
    for token in title:
        title_list.feature.add().bytes_list.value.append(token.encode())

    text_list = ex.feature_lists.feature_list["text"]
    for token in text:
        text_list.feature.add().bytes_list.value.append(token.encode())

    labels_list = ex.feature_lists.feature_list["labels"]
    for label in labels:
        labels_list.feature.add().bytes_list.value.append(label.encode())

    return ex


def parse_example(ex):
    """
    Parses a single example of MLC from a TFRecord.

    Args:
        ex (str): serialized TFRecord example

    Returns:
        features: the dictionary of features from the example (title and text)
        labels: the list of expected labels for the example
    """
    features = {}
    sequence_features = {
        "title": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "text": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    with tf.name_scope("input_processing"):
        context, sequence = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=features,
            sequence_features=sequence_features
        )

        title = sequence["title"]
        text = sequence["text"]
        labels = sequence["labels"]

    return ({
        "title": title,
        "text": text,
    }, labels)


def mlc_tfrecords(folder, docs_creator):
    """
    Fetches and parses a TFRecord dataset for MLC. If the dataset doesn't
    exist, it's created with `docs_creator` and cached in a .tfrecords
    file.

    Args:
        folder (str): absolute path to the dir where datasets are stored
        docs_creator (function): generates the full list of Docs/labels to store

    Returns:
        dataset (tf.Dataset): the MLC dataset
        labels (list[str]): list of labels
    """
    train_path = os.path.join(folder, TRAIN_PATH)
    test_path = os.path.join(folder, TEST_PATH)
    unused_path = os.path.join(folder, UNUSED_PATH)
    labels_path = os.path.join(folder, LABELS_PATH)

    if not os.path.exists(train_path):
        train, test, unused, labels = docs_creator()

        with open(labels_path, "wb") as f:
            pickle.dump(labels, f)

        random.shuffle(train)
        for path, docs in [(train_path, train),
                           (test_path, test),
                           (unused_path, unused)]:
            with tf.python_io.TFRecordWriter(path) as writer:
                for doc in docs:
                    title = tokenize(doc.title)
                    text = tokenize(doc.text)
                    labels = doc.labels

                    example = make_example(title, text, labels)
                    writer.write(example.SerializeToString())

            del docs

    train = tf.data.TFRecordDataset([train_path]).map(parse_example)
    test = tf.data.TFRecordDataset([test_path]).map(parse_example)
    unused = tf.data.TFRecordDataset([unused_path]).map(parse_example)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    return train, test, unused, labels
