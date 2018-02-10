"""Fetches all the freely available WMT14 data

Visit: http://www.statmt.org/wmt14/translation-task.html"""

import os
import anna.dataset.utils as utils

CORPORA = {
    "europarl-parallel.tgz":
    "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",

    "europarl-monolingual.tgz":
    "http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz",

    "commoncrawl.tgz":
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",

    "un.tgz":
    "http://www.statmt.org/wmt13/training-parallel-un.tgz",

    "nc-parallel.tgz":
    "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",

    "nc-monolingual.tgz":
    "http://www.statmt.org/wmt14/training-monolingual-nc-v9.tgz",

    "giga-fren.tar":
    "http://www.statmt.org/wmt10/training-giga-fren.tar",

    "dev.tgz": "http://www.statmt.org/wmt14/dev.tgz",

    "test.tgz": "http://www.statmt.org/wmt14/test-full.tgz"
}


def fetch(data_dir, dest="wmt14"):
    """
    Fetches most data from the WMT14 shared task.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored
        dest (str): name for dir where WMT14 datasets will be extracted

    Returns:
        final_dir (str): absolute path where WMT14 datasets were extracted
    """

    # Create folder
    wmt_dir = os.path.join(data_dir, dest)
    utils.create_folder(wmt_dir)

    # Download all datasets
    for f, url in CORPORA.items():
        utils.urlretrieve(url, os.path.join(wmt_dir, f))

    return wmt_dir
