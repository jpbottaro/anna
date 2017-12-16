"""Fetches the pre-trained embeddings from fastText

Visit: https://fasttext.cc/docs/en/english-vectors.html"""


import os
import zipfile
import dataset.utils as utils


NAME = "wiki-news-300d-1M.vec.zip"
URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/" + NAME


def fetch(data_dir, dest="fasttext"):
    """
    Fetches and extracts pretrained fastText word vectors.

    Creates the `dest` if it doesn't exist.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored
        dest (str): name for dir where the embeddings will be extracted

    Returns:
        final_dir (str): absolute path where the embeddings were extracted
    """

    # Create folder
    fasttext_dir = os.path.join(data_dir, dest)
    utils.create_folder(fasttext_dir)

    # Extract annotations if not previously done
    fasttext_file = os.path.join(fasttext_dir, NAME)
    if not os.path.exists(fasttext_file):
        utils.urlretrieve(URL, fasttext_file)
        with zipfile.ZipFile(fasttext_file, "r") as fasttext:
            fasttext.extractall(fasttext_dir)

    return fasttext_dir
