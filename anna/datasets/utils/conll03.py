"""Fetches and processes the ConLL03

Visit: http://www.clips.uantwerpen.be/conll2003/ner"""

import os
import utils
import tarfile
import shutil
import subprocess

DATA_FOLDER = "data/conll03/"

CONLL_FOLDER = "ner"
CONLL_FILE = CONLL_FOLDER + ".tgz"
CONLL_URL = "http://www.clips.uantwerpen.be/conll2003/" + CONLL_FILE
CONLL_SCRIPT = "bin/make.eng.2016"

REUTERS_PATH = "data/reuters/rcv1.tar.xz"
REUTERS_TEXT = """To download the Reuters corpus, follow the instructions at:

    http://trec.nist.gov/data/reuters/reuters.html

Once you have the RC1 file, put it at:

    """ + REUTERS_PATH


def fetch(folder=DATA_FOLDER):
    """
    Fetches and extracts the CoNLL03 dataset. If the Reuters dataset is
    not available, it prints instructions on how to get it.

    Creates the folder if it doesn't exist.
    """

    # We need Reuters to produce CoNLL03
    if not os.path.exists(REUTERS_PATH):
        print(REUTERS_TEXT)
        exit(1)

    utils.create_folder(folder)
    utils.urlretrieve(CONLL_URL, folder + CONLL_FILE)

    # Extract annotations if not previously done
    if not os.path.exists(folder + CONLL_FOLDER):
        with tarfile.open(folder + CONLL_FILE, "r:gz") as conll:
            conll.extractall(folder)

    # Put Reuters data where CoNLL03 script expects it
    shutil.copy(REUTERS_PATH, folder + CONLL_FOLDER)

    # Run CoNLL script
    os.chdir(folder + CONLL_FOLDER)
    subprocess.call(CONLL_SCRIPT, shell=True)


if __name__ == "__main__":
    fetch()
