"""Reads the Reuters dataset for Multi-label Clasification

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.
""" # noqa

import os
import pickle
from bs4 import BeautifulSoup
from api.doc import Doc
from . import fetcher

TRAIN_PICKLE = "train.pickle"
TEST_PICKLE = "test.pickle"
UNUSED_PICKLE = "unused.pickle"

TEST_DATES = \
    ["1996{:02}{:02}".format(month, day)
     for month in range(8, 9)
     for day in range(20, 32)]
TRAIN_DATES = \
    ["1996{:02}{:02}".format(month, day)
     for month in range(9, 12)
     for day in range(1, 32)] + \
    ["1997{:02}{:02}".format(month, day)
     for month in range(1, 8)
     for day in range(1, 32)]


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
    reuters_dir = fetcher.fetch(data_dir)

    train_path = os.path.join(reuters_dir, TRAIN_PICKLE)
    test_path = os.path.join(reuters_dir, TEST_PICKLE)
    unused_path = os.path.join(reuters_dir, UNUSED_PICKLE)

    if not os.path.isfile(train_path):
        train_docs, test_docs, unused_docs = parse(reuters_dir)

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

    return train_docs, test_docs, unused_docs


def parse(reuters_dir):
    """
    Parses the RCV1-v2 dataset.

    Args:
        reuters_dir (str): absolute path to the extracted RCV1-v2 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs
    """
    train_docs = []
    test_docs = []
    unused_docs = []

    for date in TEST_DATES:
        date_dir = os.path.join(reuters_dir, date)
        for path in os.listdir(date_dir):
            path = os.path.join(date_dir, path)
            test_docs.append(parse_file(path))

    for date in TRAIN_DATES:
        date_dir = os.path.join(reuters_dir, date)
        if not os.path.isdir(date_dir):
            continue
        for path in os.listdir(date_dir):
            path = os.path.join(date_dir, path)
            train_docs.append(parse_file(path))

    return train_docs, test_docs, unused_docs


def parse_file(path):
    """
    Parses a single xml file from the RCV1-v2 dataset.

    Args:
        path (str): absolute path to the an article from RCV1-v2

    Returns:
        doc (Doc): annotated Reuters article
    """
    with open(path, encoding="iso-8859-1") as fp:
        article = BeautifulSoup(fp, "html5lib")

        title = article.find("title").get_text()
        text = article.find("text").get_text()
        topics = article.find(class_="bip:topics:1.0")
        labels = []
        if topics:
            for topic in topics.find_all("code"):
                labels.append(topic["code"])

        return Doc(title, text, labels)
