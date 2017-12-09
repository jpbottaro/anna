"""Fetches and processes the ConLL03

Visit: http://www.clips.uantwerpen.be/conll2003/ner"""

import os
import tarfile
import shutil
import subprocess
import dataset.utils as utils
import dataset.reuters.fetcher as reuters

NER_DIR = "ner"
CONLL_FILE = NER_DIR + ".tgz"
CONLL_URL = "http://www.clips.uantwerpen.be/conll2003/" + CONLL_FILE
CONLL_SCRIPT = "bin/make.eng.2016"
CONLL_TRAIN_FILE = "eng.train"
REUTERS_FILE = "rcv1.tar.xz"


def fetch(data_dir, dest="conll03"):
    """
    Fetches and extracts the CoNLL03 dataset. If the Reuters dataset is
    not available, it prints instructions on how to get it.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where CoNLL03 will be extracted

    Returns:
        final_dir (str): absolute path where CoNLL03 was extracted
    """

    # Get Reuters
    reuters_dir = reuters.fetch(data_dir)

    # Create folder
    conll_dir = os.path.join(data_dir, dest)
    utils.create_folder(conll_dir)

    # Download annotations
    conll_file = os.path.join(conll_dir, CONLL_FILE)
    if not os.path.exists(conll_file):
        utils.urlretrieve(CONLL_URL, conll_file)

    # Extract annotations if not previously done
    ner_dir = os.path.join(conll_dir, NER_DIR)
    if not os.path.exists(ner_dir):
        with tarfile.open(conll_file, "r:gz") as conll:
            conll.extractall(conll_dir)

    # Put Reuters data where CoNLL03 script expects it
    reuters_file = os.path.join(ner_dir, REUTERS_FILE)
    if not os.path.exists(reuters_file):
        reuters_file = os.path.join(reuters_dir, REUTERS_FILE)
        shutil.copy(reuters_file, ner_dir)

    # Run CoNLL script
    train_file = os.path.join(ner_dir, CONLL_TRAIN_FILE)
    if not os.path.exists(train_file):
        os.chdir(ner_dir)
        subprocess.call(CONLL_SCRIPT, shell=True)

    return ner_dir
