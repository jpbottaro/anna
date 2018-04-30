"""Reads the fasttext word embeddings

Visit: https://fasttext.cc/docs/en/english-vectors.html#format"""

import os
import zipfile
import numpy as np
import anna.data.utils as utils
from collections import defaultdict

DESTINATION = "fasttext"
NAME = "wiki-news-300d-1M-subword.vec"
ZIP_NAME = NAME + ".zip"
URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/" + ZIP_NAME


def fetch_and_parse(data_dir, voc_size=None):
    """
    Fetches and parses the fasttext word embeddings dataset. The dataset is
    also cached as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored
        voc_size (int): maximum size of the vocabulary, None for no limit

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    return parse(fetch(data_dir), voc_size)


def parse(fasttext_dir, voc_size):
    """
    Parses the fasttext word embeddings.

    Args:
        fastext_dir (str): absolute path to the extracted word embeddings
        voc_size (int): maximum size of the vocabulary, None for no limit

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    # Reserve 0 for special padding token, 1 for unknown, 2 for end of stream,
    # and default any token not in the vocabulary to unknown
    voc = ["_PAD_", "_UNK_", "_END_"]

    # Reserve first embeddings for special tokens
    emb = [[], [], []]

    first = True
    fasttext_path = os.path.join(fasttext_dir, NAME)
    with open(fasttext_path) as f:
        for line in f:
            # First line contains # words and embedding sizes, skip
            if first:
                first = False
                continue

            parts = line.split(" ")
            if parts[0] not in voc:
                voc.append(parts[0])
                emb.append([float(n) for n in parts[1:]])
            if len(emb) >= voc_size:
                break

    # We copy the last embedding for the special token
    emb[0] = emb[-1]
    emb[1] = emb[-1]
    emb[2] = emb[-1]

    return voc, np.array(emb)


def fetch(data_dir):
    """
    Fetches and extracts pretrained fastText word vectors.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        fasttext_dir (str): absolute path to the folder where datasets are stored
    """
    # Create folder
    fasttext_dir = os.path.join(data_dir, DESTINATION)
    utils.create_folder(fasttext_dir)

    # Extract annotations if not previously done
    fasttext_file = os.path.join(fasttext_dir, NAME)
    fasttext_zip = os.path.join(fasttext_dir, ZIP_NAME)
    if not os.path.exists(fasttext_file):
        utils.urlretrieve(URL, fasttext_zip)
        with zipfile.ZipFile(fasttext_file, "r") as fasttext:
            fasttext.extractall(fasttext_dir)

    return fasttext_dir
