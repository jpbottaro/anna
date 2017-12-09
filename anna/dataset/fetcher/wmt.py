"""Fetches all the freely available WMT14 data

Visit: http://www.statmt.org/wmt14/translation-task.html"""

import os
from . import utils

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


def fetch(folder):
    """
    Fetches most data from the WMT14 shared task .

    Creates the folder if it doesn't exist.
    """

    utils.create_folder(folder)
    for f, url in CORPORA.items():
        utils.urlretrieve(url, os.path.join(folder, f))
