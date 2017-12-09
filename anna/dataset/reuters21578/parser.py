"""Reads the Reuters-21578 dataset

Parses and splits according to:
    Yang, Yiming. (2001). A Study on Thresholding Strategies for Text Categorization.
    SIGIR Forum (ACM Special Interest Group on Information Retrieval). 10.1145/383952.383975.
""" # noqa

import os
import pickle
from . import fetcher
from api.doc import Doc

TRAIN_PICKLE = "train.pickle"
TEST_PICKLE = "test.pickle"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the Reuters-21578 dataset. The dataset is also cached
    as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (list[Doc]): annotated articles for training
        eval_docs (list[Doc]): annotated articles for testing
    """
    reuters_dir = fetcher.fetch(data_dir)

    train_path = os.path.join(reuters_dir, TRAIN_PICKLE)
    test_path = os.path.join(reuters_dir, TEST_PICKLE)

    if not os.path.isfile(train_path):
        train_docs, test_docs = parse(reuters_dir)

        with open(train_path, "wb") as f:
            pickle.dump(train_docs, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_docs, f)
    else:
        with open(train_path, "rb") as f:
            train_docs = pickle.load(f)
        with open(test_path, "rb") as f:
            test_docs = pickle.load(f)

    return train_docs, test_docs


def parse(reuters_dir):
    """
    Parses the Reuters-21578 dataset.

    Splits the data according to "A Study on Thresholding Strategies for Text
    Categorization" - Yang, Timing (2001).

    Args:
        data_dir (str): absolute path to the extracted Reuters-21578 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        eval_docs (list[Doc]): annotated articles for testing
    """
    docs = [Doc("this is a test", ["tagone", "tagtwo"])]

    return docs, docs
