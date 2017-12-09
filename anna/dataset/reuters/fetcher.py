"""Fetches and processes the ConLL03

Visit: http://www.clips.uantwerpen.be/conll2003/ner"""

import os
import dataset.utils as utils

REUTERS_FILE = "rcv1.tar.xz"
REUTERS_TEXT = """To download the Reuters corpus, follow the instructions at:

    http://trec.nist.gov/data/reuters/reuters.html

Once you have the RC1 file, put it at:

    {}"""


def fetch(data_dir, dest="reuters"):
    """
    If the Reuters dataset is not available, prints instructions on how to
    get it. Otherwise, returns the folder with the file.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where Reuters should be extracted

    Returns:
        final_dir (str): absolute path where Reuters was extracted
    """

    # Create folder
    reuters_dir = os.path.join(data_dir, dest)
    if not os.path.exists(reuters_dir):
        utils.create_folder(reuters_dir)

    reuters_file = os.path.join(reuters_dir, REUTERS_FILE)
    if not os.path.exists(reuters_file):
        print(REUTERS_TEXT.format(reuters_file))
        exit(0)

    return reuters_dir
