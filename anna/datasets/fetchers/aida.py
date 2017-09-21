"""Fetches and processes the AIDA dataset

Visit: https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/""" # noqa


import os
import zipfile
import subprocess
from . import utils
from . import conll03

FOLDER_NAME = "aida"

AIDA_FOLDER = "aida-yago2-dataset"
AIDA_FILE = AIDA_FOLDER + ".zip"
AIDA_URL = \
  "http://resources.mpi-inf.mpg.de/yago-naga/aida/download/" + AIDA_FILE
AIDA_SCRIPT = "java -jar aida-yago2-dataset.jar <<< '{}'"
AIDA_FINAL_FILE = "AIDA-YAGO2-dataset.tsv"

CONLL_FOLDER = "conll03/ner"


def fetch(folder):
    """
    Fetches and extracts the AIDA dataset. If the CoNLL dataset is
    not available, it tries to fecht it.

    Creates the folder if it doesn't exist.
    """

    target_folder = os.path.join(folder, FOLDER_NAME)
    extracted_folder = os.path.join(target_folder, AIDA_FOLDER)
    aida_file = os.path.join(target_folder, AIDA_FILE)
    aida_final_file = os.path.join(extracted_folder, AIDA_FINAL_FILE)

    if os.path.exists(aida_final_file):
        return

    # We need CoNLL03 to produce AIDA
    conll_path = os.path.join(folder, CONLL_FOLDER)
    if not os.path.exists(conll_path):
        conll03.fetch(folder)

    # Extract annotations if not previously done
    if not os.path.exists(extracted_folder):
        utils.create_folder(target_folder)
        utils.urlretrieve(AIDA_URL, aida_file)
        with zipfile.ZipFile(aida_file, "r") as aida:
            aida.extractall(target_folder)

    # Run CoNLL script
    os.chdir(extracted_folder)
    subprocess.call(AIDA_SCRIPT.format(conll_path), shell=True)


if __name__ == "__main__":
    fetch()
