"""Reads the GloVe word embeddings

Visit: https://nlp.stanford.edu/projects/glove/"""

import os
import numpy as np
import anna.data.utils as utils
from porter2stemmer import Porter2Stemmer

DESTINATION = "glove"
NAME = "glove.840B.300d"
TXT_NAME = NAME + ".txt"
ZIP_NAME = NAME + ".zip"
URL = "http://nlp.stanford.edu/data/" + ZIP_NAME


def fetch_and_parse(data_dir, voc_size=None, lowercase=False, stem=False):
    """
    Fetches and parses the GloVe word embeddings dataset. The dataset is
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


def parse(glove_dir, voc_size, lowercase=False, stem=False):
    """
    Parses the glove word embeddings.

    Args:
        glove_dir (str): absolute path to the extracted word embeddings
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
    glove_path = os.path.join(glove_dir, TXT_NAME)
    with open(glove_path) as f:
        for line in f:
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
                if len(words) >= voc_size:
                    break

    return utils.add_special_tokens(voc, np.array(emb))


def fetch(data_dir):
    """
    Fetches and extracts pre-trained GloVe word vectors.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        glove_dir (str): absolute path to the folder where glove is stored
    """
    file_path = os.path.join(data_dir, DESTINATION, ZIP_NAME)
    txt_path = os.path.join(data_dir, DESTINATION, TXT_NAME)
    return utils.fetch(URL, file_path, txt_path)
