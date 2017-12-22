"""Fetches and processes RCV1-v2 for Multi-label Clasiffication

Visit: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm"""  # noqa

import os
import gzip
import dataset.utils as utils

HOST = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/"
FILE_URL_FORMAT = HOST + "a12-token-files/{}"
TOPIC_URL_FORMAT = HOST + "a08-topic-qrels/{}"

TEST_FILES = ["lyrl2004_tokens_test_pt0.dat.gz",
              "lyrl2004_tokens_test_pt1.dat.gz",
              "lyrl2004_tokens_test_pt2.dat.gz",
              "lyrl2004_tokens_test_pt3.dat.gz"]
TRAIN_FILE = "lyrl2004_tokens_train.dat.gz"
TOPICS_FILE = "rcv1-v2.topics.qrels.gz"

FILE_URLS = {f: FILE_URL_FORMAT.format(f) for f in TEST_FILES + [TRAIN_FILE]}
FILE_URLS[TOPICS_FILE] = TOPIC_URL_FORMAT.format(TOPICS_FILE)

TEST_FINAL = "test.dat"
TRAIN_FINAL = "train.dat"
TOPICS_FINAL = "topics.dat"


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

    # Download all datasets
    for f, url in FILE_URLS.items():
        path = os.path.join(rcv1_dir, f)
        if not os.path.exists(path):
            utils.urlretrieve(url, path)

    # Extract topics
    path = os.path.join(rcv1_dir, TOPICS_FINAL)
    if not os.path.exists(path):
        src = os.path.join(rcv1_dir, TOPICS_FILE)
        with open(path, "wb") as o, gzip.open(src, "rb") as i:
            o.write(i.read())

    # Extract train
    path = os.path.join(rcv1_dir, TRAIN_FINAL)
    if not os.path.exists(path):
        src = os.path.join(rcv1_dir, TRAIN_FILE)
        with open(path, "wb") as o, gzip.open(src, "rb") as i:
            o.write(i.read())

    # Extract test
    path = os.path.join(rcv1_dir, TEST_FINAL)
    if not os.path.exists(path):
        with open(path, "wb") as o:
            for p in TEST_FILES:
                src = os.path.join(rcv1_dir, p)
                with gzip.open(src, "rb") as i:
                    o.write(i.read())

    return rcv1_dir
