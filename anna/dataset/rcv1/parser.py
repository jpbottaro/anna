"""Reads the RCV1-v2 dataset for Multi-label Clasification

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.
""" # noqa

from . import fetcher


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
    return parse(fetcher.fetch(data_dir))


def parse(rcv1_dir):
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

    return train_docs, test_docs, unused_docs
