"""Reads the fasttext word embeddings

Visit: https://fasttext.cc/docs/en/english-vectors.html#format"""

import os
import numpy as np
import dataset.utils as utils
from collections import defaultdict
from . import fetcher

NAME = "wiki-news-300d-1M.vec"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the fasttext word embeddings dataset. The dataset is
    also cached as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        voc (dict): bimap of word <-> id
        emb (numpy array): array of embeddings for each word in `voc`
    """
    fasttext_dir = fetcher.fetch(data_dir)

    return parse(fasttext_dir)


@utils.cache(2)
def parse(fasttext_dir):
    """
    Parses the fasttext word embeddings.

    Args:
        fastext_dir (str): absolute path to the extracted word embeddings

    Returns:
        voc (dict): bimap of word <-> id
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    # The first embedding is reserved for the UNK token
    emb = [[]]
    voc = defaultdict(int)
    voc[0] = "UNK"
    voc["UNK"] = 0

    first = True
    fasttext_path = os.path.join(fasttext_dir, NAME)
    with open(fasttext_path) as f:
        for line in f:
            # First line contains # words and embedding sizes, skip
            if first:
                first = False
                continue

            i = len(emb)
            parts = line.split(" ")
            voc[parts[0]] = i
            voc[i] = parts[0]
            emb.append([float(n) for n in parts[1:-1]])

    # We copy the last embedding for the UNK token
    emb[0] = emb[-1]
    return voc, np.array(emb)
