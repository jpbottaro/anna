"""Fetches and parses the Reuters-21578 dataset

Parses and splits according to:
    Yang, Yiming. (2001)
    A Study on Thresholding Strategies for Text Categorization.
    SIGIR Forum (ACM Special Interest Group on Information Retrieval)
    10.1145/383952.383975.

Visit: https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection"""  # noqa

import os
import anna.data.utils as utils
from anna.data.api import Doc
from collections import Counter
from bs4 import BeautifulSoup

DESTINATION = "reuters21578"
TAR_NAME = DESTINATION + ".tar.gz"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
      + "reuters21578-mld/" + TAR_NAME
REUTER_SGML = "reut2-{:03}.sgm"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the Reuters-21578 dataset.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (tf.data.Dataset): annotated articles for training
        test_docs (tf.data.Dataset): annotated articles for testing
        unused_docs (tf.data.Dataset): unused docs following the "ModApte" split
        labels (list[str]): final list of labels, from most to least frequent
    """
    reuters_dir = os.path.join(data_dir, DESTINATION)

    return utils.mlc_tfrecords(reuters_dir, lambda: parse(fetch(data_dir)))


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
        labels (list[str]): final list of labels, from most to least frequent
    """
    train_docs = []
    test_docs = []
    unused_docs = []
    for i in range(22):
        path = os.path.join(reuters_dir, REUTER_SGML.format(i))
        with open(path, encoding="latin1") as fp:
            soup = BeautifulSoup(fp, "html5lib")
            for article in soup.find_all("reuters"):
                doc_id = article.find("reuters")["newid"]
                text = None
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

                doc = Doc(doc_id, title, None, dateline, text, labels)

                lewis_split = article.get("lewissplit")
                is_topics = article.get("topics")
                if lewis_split == "TRAIN" and is_topics == "YES":
                    train_docs.append(doc)
                elif lewis_split == "TEST" and is_topics == "YES":
                    test_docs.append(doc)
                else:
                    unused_docs.append(doc)

    # Removes unlabelled docs and labels that don't appear in both train & test
    train_docs, test_docs, unused_docs = yang_filter(train_docs,
                                                     test_docs,
                                                     unused_docs)

    # Get list of labels, from frequent to rare
    label_counts = Counter()
    for d in train_docs:
        label_counts.update(d.labels)
    labels = [l[0] for l in label_counts.most_common()]

    return train_docs, test_docs, unused_docs, labels


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
        unused_docs (list[Doc]): unused docs according to the "ModApte" split

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs according to the "ModApte" AND
                                 Yang's extra filter
    """
    # Get labels that don't appear in _both_ train and test
    train_labels = set([l for d in train_docs for l in d.labels])
    test_labels = set([l for d in test_docs for l in d.labels])
    bad_labels = train_labels ^ test_labels

    # Remove all bad labels from documents
    for doc in train_docs + test_docs:
        doc.labels = [l for l in doc.labels if l not in bad_labels]

    # Find all docs that have no labels
    bad_docs = [d for d in train_docs + test_docs if not d.labels]
    bad_docs_set = set(bad_docs)

    # Remove them from train/test
    train_docs = [d for d in train_docs if d not in bad_docs_set]
    test_docs = [d for d in test_docs if d not in bad_docs_set]
    unused_docs = unused_docs + bad_docs

    return train_docs, test_docs, unused_docs


def fetch(data_dir):
    """
    Fetches and extracts the Reuters-21578 dataset.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        reuters_dir (str): absolute path to the folder where reuters is stored
    """
    file_path = os.path.join(data_dir, DESTINATION, TAR_NAME)
    return utils.fetch(URL, file_path)
