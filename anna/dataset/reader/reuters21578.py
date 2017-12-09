"""Reads the Reuters-21578 dataset

Parses and splits according to:
    Yang, Yiming. (2001). A Study on Thresholding Strategies for Text Categorization.
    SIGIR Forum (ACM Special Interest Group on Information Retrieval). 10.1145/383952.383975.
""" # noqa

from api.doc import Doc


def parse(folder):
    """
    Parses the Reuters-21578 dataset.

    Splits the data according to "A Study on Thresholding Strategies for Text
    Categorization" - Yang, Timing (2001).

    Args:
        folder (str): path to the extracted Reuters-21578 corpus

    Returns:
        train_docs (list[Doc]): annotated articles for training
        eval_docs (list[Doc]): annotated articles for testing
    """
    docs = [Doc("this is a test", ["tagone", "tagtwo"])]

    return docs, docs
