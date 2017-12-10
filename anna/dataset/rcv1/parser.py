"""Reads the RCV1-v2 dataset for Multi-label Clasification

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.
"""

import os
import pickle
import collections
from api.doc import Doc
from . import fetcher

TRAIN_PICKLE = "train.pickle"
TEST_PICKLE = "test.pickle"
UNUSED_PICKLE = "unused.pickle"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the RCV1-v2 dataset. The dataset is also cached
    as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs acording to the "ModApte" split
    """
    rcv1_dir = fetcher.fetch(data_dir)

    train_path = os.path.join(rcv1_dir, TRAIN_PICKLE)
    test_path = os.path.join(rcv1_dir, TEST_PICKLE)
    unused_path = os.path.join(rcv1_dir, UNUSED_PICKLE)

    if not os.path.isfile(train_path):
        train_docs, test_docs, unused_docs = parse(rcv1_dir)

        with open(train_path, "wb") as f:
            pickle.dump(train_docs, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_docs, f)
        with open(unused_path, "wb") as f:
            pickle.dump(unused_docs, f)
    else:
        with open(train_path, "rb") as f:
            train_docs = pickle.load(f)
        with open(test_path, "rb") as f:
            test_docs = pickle.load(f)
        with open(unused_path, "rb") as f:
            unused_docs = pickle.load(f)

    # Switch order of original train and test
    return test_docs, train_docs, unused_docs


def parse(rcv1_dir):
    """
    Parses the RCV1-v2 dataset.

    Args:
        rcv1_dir (str): absolute path to the extracted RCV1-v2 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs
    """
    train_docs = []
    test_docs = []
    unused_docs = []

    topics = collections.defaultdict(set)
    topics_path = os.path.join(rcv1_dir, "topics.dat")
    with open(topics_path, "r") as f:
        for line in f:
            split = line.split(" ")
            topic = split[0].strip()
            doc_id = split[1].strip()
            topics[doc_id].add(topic)

    for path, docs in [("train.dat", train_docs), ("test.dat", test_docs)]:
        path = os.path.join(rcv1_dir, path)
        with open(path, "r") as f:
            doc_id = None
            text = None
            for line in f:
                if line == "\n":
                    if not doc_id or not text:
                        print("What!")
                        exit(0)
                    labels = list(topics[doc_id])
                    docs.append(Doc(None, None, None, text, labels))
                    doc_id = None
                    text = None
                elif line.startswith(".I"):
                    doc_id = line.split(" ")[1].strip()
                elif line.startswith(".W"):
                    pass
                elif not text:
                    text = line
                else:
                    text += " " + line

    return train_docs, test_docs, unused_docs
