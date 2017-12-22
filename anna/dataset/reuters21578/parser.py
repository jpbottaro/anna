"""Reads the Reuters-21578 dataset

Parses and splits according to:
    Yang, Yiming. (2001)
    A Study on Thresholding Strategies for Text Categorization.
    SIGIR Forum (ACM Special Interest Group on Information Retrieval)
    10.1145/383952.383975.
"""

import os
import dataset.utils as utils
from bs4 import BeautifulSoup
from api.doc import Doc
from . import fetcher

REUTER_SGML = "reut2-{:03}.sgm"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the Reuters-21578 dataset. The dataset is also cached
    as a pickle for further calls.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs acording to the "ModApte" split
    """
    reuters_dir = fetcher.fetch(data_dir)
    return parse(reuters_dir)


@utils.cache(3)
def parse(reuters_dir):
    """
    Parses the Reuters-21578 dataset.

    Splits the data according to "A Study on Thresholding Strategies for Text
    Categorization" - Yang, Timing (2001). This is a slightly modified version
    of "ModApte" split.

    Args:
        reuters_dir (str): absolute path to the extracted Reuters-21578 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs acording to the "ModApte" split
    """
    train_docs = []
    test_docs = []
    unused_docs = []
    for i in range(22):
        path = os.path.join(reuters_dir, REUTER_SGML.format(i))
        with open(path, encoding="latin1") as fp:
            soup = BeautifulSoup(fp, "html5lib")
            for article in soup.find_all("reuters"):
                title = article.find("title")
                if title:
                    text = str(title.find_next_sibling(string=True))
                    title = title.get_text()

                dateline = article.find("dateline")
                if dateline:
                    text = str(dateline.find_next_sibling(string=True))
                    dateline = dateline.get_text()

                if not text:
                    text = article.find("text").get_text()

                labels = [t.get_text() for t in article.topics.find_all("d")]

                doc = Doc(title, None, dateline, text, labels)

                lewis_split = article.get("lewissplit")
                is_topics = article.get("topics")
                if lewis_split == "TRAIN" and is_topics == "YES":
                    train_docs.append(doc)
                elif lewis_split == "TEST" and is_topics == "YES":
                    test_docs.append(doc)
                else:
                    unused_docs.append(doc)

    # Removes unlabelled docs and labels that don't appear in both train & test
    return yang_filter(train_docs, test_docs, unused_docs)


def yang_filter(train_docs, test_docs, unused_docs):
    """
    Splits the data according to "A Study on Thresholding Strategies for Text
    Categorization" - Yang, Timing (2001). This is a slightly modified version
    of "ModApte" split.

    Main difference (quote from the author):
        "[..] eliminating unlabelled documents and selecting the categories
        which have at least one document in the training set and one in the
        test set. [..]"

    Args:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs acording to the "ModApte" split

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs acording to the "ModApte" AND
                                 Yang's extra filter
    """
    # Get labels that don't appear in _both_ train and test
    train_labels = set([l for d in train_docs for l in d.labels])
    test_labels = set([l for d in test_docs for l in d.labels])
    bad_labels = train_labels ^ test_labels

    # Remove all bad labels from documents
    for doc in train_docs + test_docs:
        doc.labels = [l for l in doc.labels if l not in bad_labels]

    # Remove all docs that have no labels
    bad_docs = [d for d in train_docs + test_docs if not d.labels]
    train_docs = [d for d in train_docs if d not in bad_docs]
    test_docs = [d for d in test_docs if d not in bad_docs]
    unused_docs = unused_docs + bad_docs

    return train_docs, test_docs, unused_docs
