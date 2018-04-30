"""Fetches and processes the AIDA dataset

Visit: https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/"""  # noqa

import os
import zipfile
import subprocess
import anna.data.utils as utils
import anna.data.dataset.conll03 as conll03

AIDA_NAME = "aida-yago2-dataset"
AIDA_FILE = AIDA_NAME + ".zip"
AIDA_URL = \
  "http://resources.mpi-inf.mpg.de/yago-naga/aida/download/" + AIDA_FILE
AIDA_SCRIPT = "java -jar aida-yago2-dataset.jar <<< '{}'"
AIDA_FINAL_FILE = "AIDA-YAGO2-dataset.tsv"


def fetch(data_dir, dest="aida"):
    """
    Fetches and extracts the AIDA dataset.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where AIDA will be extracted

    Returns:
        final_dir (str): absolute path where AIDA was extracted
    """

    # Get CoNLL03
    conll_dir = conll03.fetch(data_dir)

    # Create folder
    aida_dir = os.path.join(data_dir, dest)
    utils.create_folder(aida_dir)

    # Download AIDA
    aida_file = os.path.join(aida_dir, AIDA_FILE)
    if not os.path.exists(aida_file):
        utils.urlretrieve(AIDA_URL, aida_file)

    # Extract annotations
    final_dir = os.path.join(aida_dir, AIDA_NAME)
    if not os.path.exists(final_dir):
        with zipfile.ZipFile(aida_file, "r") as aida:
            aida.extractall(aida_dir)

    # Run AIDA script
    final_file = os.path.join(final_dir, AIDA_FINAL_FILE)
    if not os.path.exists(final_file):
        os.chdir(final_dir)
        subprocess.call(AIDA_SCRIPT.format(conll_dir), shell=True)

    return final_dir
