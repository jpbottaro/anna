"""Fetches and processes the ConLL03

Visit: http://www.clips.uantwerpen.be/conll2003/ner"""

import os
import dataset.utils as utils

# TODO: Use data from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm # noqa


def fetch(data_dir, dest="rcv1-v2"):
    """
    Fetches the tokenized RCV1-v2 dataset.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where RCV1-v2 should be extracted

    Returns:
        final_dir (str): absolute path where RCV1-v2 was extracted
    """

    # Create folder
    rcv1_dir = os.path.join(data_dir, dest)
    utils.create_folder(rcv1_dir)

    return rcv1_dir
