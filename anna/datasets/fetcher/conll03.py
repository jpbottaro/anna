"""Fetches and processes the ConLL03

Visit: http://www.clips.uantwerpen.be/conll2003/ner"""

import os
import tarfile
import shutil
import subprocess
from . import utils

FOLDER_NAME = "conll03"

CONLL_FOLDER = "ner"
CONLL_FILE = CONLL_FOLDER + ".tgz"
CONLL_URL = "http://www.clips.uantwerpen.be/conll2003/" + CONLL_FILE
CONLL_SCRIPT = "bin/make.eng.2016"
CONLL_TRAIN_FILE = "eng.train"

REUTERS_PATH = "reuters/rcv1.tar.xz"
REUTERS_TEXT = """To download the Reuters corpus, follow the instructions at:

    http://trec.nist.gov/data/reuters/reuters.html

Once you have the RC1 file, put it at:

    anna/datasets/data/""" + REUTERS_PATH


def fetch(folder):
    """
    Fetches and extracts the CoNLL03 dataset. If the Reuters dataset is
    not available, it prints instructions on how to get it.

    Creates the folder if it doesn't exist.
    """

    target_folder = os.path.join(folder, FOLDER_NAME)
    extracted_folder = os.path.join(target_folder, CONLL_FOLDER)
    conll_file = os.path.join(target_folder, CONLL_FILE)
    conll_train = os.path.join(extracted_folder, CONLL_TRAIN_FILE)

    if os.path.exists(conll_train):
        return

    # We need Reuters to produce CoNLL03
    reuters_path = os.path.join(folder, REUTERS_PATH)
    if not os.path.exists(reuters_path):
        print(REUTERS_TEXT)
        exit(0)

    # Extract annotations if not previously done
    if not os.path.exists(extracted_folder):
        utils.create_folder(target_folder)
        utils.urlretrieve(CONLL_URL, conll_file)
        with tarfile.open(conll_file, "r:gz") as conll:
            conll.extractall(target_folder)

    # Put Reuters data where CoNLL03 script expects it
    shutil.copy(reuters_path, extracted_folder)

    # Run CoNLL script
    os.chdir(extracted_folder)
    subprocess.call(CONLL_SCRIPT, shell=True)


if __name__ == "__main__":
    fetch()
