"""Reads the fasttext word embeddings

Visit: https://fasttext.cc/docs/en/english-vectors.html#format"""

import os
import numpy as np
import anna.data.utils as utils
from porter2stemmer import Porter2Stemmer

DESTINATION = "fasttext"
NAME = "wiki-news-300d-1M-subword.vec"
ZIP_NAME = NAME + ".zip"
URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/" + ZIP_NAME


def fetch_and_parse(data_dir, voc_size=None, lowercase=False, stem=False):
    """
    Fetches and parses the fasttext word embeddings dataset. The dataset is
    also cached as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored
        voc_size (int): maximum size of the vocabulary, None for no limit
        lowercase (bool): whether the vocabulary should be all lowercase
        stem (bool): whether the vocabulary should be stemmed

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    return parse(fetch(data_dir), voc_size, lowercase, stem)


def parse(fasttext_dir, voc_size, lowercase=False, stem=False):
    """
    Parses the fasttext word embeddings.

    Args:
        fasttext_dir (str): absolute path to the extracted word embeddings
        voc_size (int): maximum size of the vocabulary, None for no limit
        lowercase (bool): whether the vocabulary should be all lowercase
        stem (bool): whether the vocabulary should be stemmed

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    stemmer = Porter2Stemmer() if stem else lambda x: x

    voc = []
    emb = []
    words = set()
    first = True
    fasttext_path = os.path.join(fasttext_dir, NAME)
    with open(fasttext_path) as f:
        for line in f:
            # First line contains # words and embedding sizes, skip
            if first:
                first = False
                continue
            parts = line.split(" ")
            word = parts[0]

            if lowercase:
                word = word.lower()

            if stem:
                word = stemmer.stem(word)

            if word not in words:
                words.add(word)
                voc.append(word)
                emb.append([float(n) for n in parts[1:]])
                if len(voc) >= voc_size:
                    break

    return utils.add_special_tokens(voc, np.array(emb))


def fetch(data_dir):
    """
    Fetches and extracts pre-trained fastText word vectors.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        fasttext_dir (str): absolute path to the folder with fasttext data
    """
    file_path = os.path.join(data_dir, DESTINATION, ZIP_NAME)
    result_path = os.path.join(data_dir, DESTINATION, NAME)
    return utils.fetch(URL, file_path, result_path)
