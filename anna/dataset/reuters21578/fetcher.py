"""Fetches the Reuters-21578 dataset for Multi-label Classification

Visit: https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection""" # noqa


import os
import tarfile
import dataset.utils as utils

NAME = "reuters21578"
FILE = NAME + ".tar.gz"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
    + "reuters21578-mld/" + FILE


def fetch(data_dir, dest="reuters21578"):
    """
    Fetches and extracts the Reuters-21578 dataset.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where Reuters-21578 will be extracted

    Returns:
        final_dir (str): absolute path where Reuters-21578 was extracted
    """

    # Create folder
    reuters_dir = os.path.join(data_dir, dest)
    utils.create_folder(reuters_dir)

    # Extract annotations if not previously done
    reuters_file = os.path.join(reuters_dir, FILE)
    if not os.path.exists(reuters_file):
        utils.urlretrieve(URL, reuters_file)
        with tarfile.open(reuters_file, "r:gz") as reuters:
            reuters.extractall(reuters_dir)

    return reuters_dir
