"""Reads the GloVe word embeddings

Visit: https://nlp.stanford.edu/projects/glove/"""

import os
import zipfile
import numpy as np
import anna.data.utils as utils

DESTINATION = "glove"
NAME = "glove.840B.300d"
TXT_NAME = NAME + ".txt"
ZIP_NAME = NAME + ".zip"
URL = "http://nlp.stanford.edu/data/" + ZIP_NAME


def fetch_and_parse(data_dir, voc_size=None):
    """
    Fetches and parses the GloVe word embeddings dataset. The dataset is
    also cached as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored
        voc_size (int): maximum size of the vocabulary, None for no limit

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    return parse(fetch(data_dir), voc_size)


def parse(glove_dir, voc_size):
    """
    Parses the glove word embeddings.

    Args:
        glove_dir (str): absolute path to the extracted word embeddings
        voc_size (int): maximum size of the vocabulary, None for no limit

    Returns:
        voc (list[str]): list of words, matching the index in `emb`
        emb (numpy.array): array of embeddings for each word in `voc`
    """
    voc = []
    emb = []
    glove_path = os.path.join(glove_dir, TXT_NAME)
    with open(glove_path) as f:
        for line in f:
            parts = line.split(" ")
            if parts[0] not in voc:
                voc.append(parts[0])
                emb.append([float(n) for n in parts[1:]])
            if len(emb) >= voc_size:
                break

    return utils.add_special_tokens(voc, np.array(emb))


def fetch(data_dir):
    """
    Fetches and extracts pretrained GloVe word vectors.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        glove_dir (str): absolute path to the folder where datasets are stored
    """
    # Create folder
    glove_dir = os.path.join(data_dir, DESTINATION)
    utils.create_folder(glove_dir)

    # Extract annotations if not previously done
    glove_file = os.path.join(glove_dir, TXT_NAME)
    glove_zip = os.path.join(glove_dir, ZIP_NAME)
    if not os.path.exists(glove_file):
        utils.urlretrieve(URL, glove_zip)
        with zipfile.ZipFile(glove_zip, "r") as glove:
            glove.extractall(glove_dir)

    return glove_dir
