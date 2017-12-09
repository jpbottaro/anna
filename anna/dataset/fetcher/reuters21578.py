"""Fetches the Reuters-21578 dataset for Multi-label Classification

Visit: https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection""" # noqa


import os
import tarfile
from . import utils

FOLDER = "reuters21578"
FILE = FOLDER + ".tar.gz"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
    + "reuters21578-mld/" + FILE


def fetch(folder):
    """
    Fetches and extracts the Reuters-21578 dataset.

    Creates the folder if it doesn't exist.
    """

    target_folder = os.path.join(folder, FOLDER)
    reuters_file = os.path.join(target_folder, FILE)

    # Extract annotations if not previously done
    if not os.path.exists(target_folder):
        utils.create_folder(target_folder)
        utils.urlretrieve(URL, reuters_file)
        with tarfile.open(reuters_file, "r:gz") as reuters:
            reuters.extractall(target_folder)
