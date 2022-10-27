"""Reads the Reuters dataset for Multi-label Classification

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.
"""

import os
import tarfile
import anna.data.utils as utils
from anna.data.api import Doc
from collections import Counter
from bs4 import BeautifulSoup

NAME = "reuters"

REUTERS_FINAL_DIR = "rcv1"
REUTERS_FILE = "rcv1.tar.xz"
REUTERS_TEXT = """To download the Reuters corpus, follow the instructions at:

    http://trec.nist.gov/data/reuters/reuters.html

Once you have the RC1 file, put it at:

    {}"""

TEST_DATES = \
    ["1996{:02}{:02}".format(month, day)
     for month in range(8, 9)
     for day in range(20, 32)]
TRAIN_DATES = \
    ["1996{:02}{:02}".format(month, day)
     for month in range(9, 13)
     for day in range(1, 32)] + \
    ["1997{:02}{:02}".format(month, day)
     for month in range(1, 13)
     for day in range(1, 32)]


def fetch_and_parse(data_dir):
    """
    Fetches and parses the Reuters dataset.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (tf.data.Dataset): annotated articles for training
        test_docs (tf.data.Dataset): annotated articles for testing
        unused_docs (tf.data.Dataset): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    reuters_dir = os.path.join(data_dir, NAME)

    return utils.mlc_tfrecords(reuters_dir, lambda: parse(fetch(data_dir)))


def parse(reuters_dir):
    """
    Parses the RCV1-v2 dataset.

    Args:
        reuters_dir (str): absolute path to the extracted RCV1-v2 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    train_docs = []
    test_docs = []
    unused_docs = []
    label_counts = Counter()

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
            doc = parse_file(path)
            label_counts.update(doc.labels)
            train_docs.append(doc)

    # Get list of labels, from frequent to rare
    labels = [l[0] for l in label_counts.most_common()]

    return train_docs, test_docs, unused_docs, labels


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

        doc_id = article.find("newsitem")["itemid"]
        title = article.find("title").get_text()
        text = article.find("text").get_text()
        headline = article.find("headline")
        if headline:
            headline = headline.get_text()
        dateline = article.find("dateline")
        if dateline:
            dateline = dateline.get_text()
        topics = article.find(class_="bip:topics:1.0")
        labels = []
        if topics:
            for topic in topics.find_all("code"):
                labels.append(str(topic["code"]))

        return Doc(doc_id, title, headline, dateline, text, labels)


def fetch(data_dir):
    """
    If the Reuters dataset is not available, prints instructions on how to
    get it. Otherwise, returns the folder with the file.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        reuters_dir (str): absolute path to the folder where datasets are stored
    """
    # Create folder
    reuters_dir = os.path.join(data_dir, NAME)
    utils.create_folder(reuters_dir)

    # Show instructions to fetch the dataset if it's not available
    reuters_file = os.path.join(reuters_dir, REUTERS_FILE)
    if not os.path.exists(reuters_file):
        print(REUTERS_TEXT.format(reuters_file))
        exit(0)

    # Extract the file
    final_dir = os.path.join(reuters_dir, REUTERS_FINAL_DIR)
    if not os.path.exists(final_dir):
        with tarfile.open(reuters_file) as reuters:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(reuters, reuters_dir)

    return final_dir
