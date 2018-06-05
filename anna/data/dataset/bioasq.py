"""Fetches and parses the BioASQ dataset from 2018.

Parses and splits according to (using 2017/2018 for test instead of 2014/2015):
    Jinseok Nam, Eneldo Loza Mencía, Hyunwoo J. Kim, Johannes Fürnkranz
    Maximizing Subset Accuracy with Recurrent Neural Networks in
    Multi-label Classification.
    NIPS 2017: 5419-5429

Visit: http://participants-area.bioasq.org/general_information/Task6a/"""  # noqa

import os
import ijson
import anna.data.utils as utils
from anna.data.api import Doc
from collections import Counter

TEST_YEARS = ["2017", "2018"]

DESTINATION = "bioasq"
NAME = "allMeSH_2018"
JSON_NAME = NAME + ".json"
ZIP_NAME = NAME + ".zip"
URL = "http://participants-area.bioasq.org/Tasks/6a/trainingDataset" + \
      "/raw/allMeSH/"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the BioASQ dataset.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (tf.data.Dataset): annotated articles for training
        test_docs (tf.data.Dataset): annotated articles for testing
        unused_docs (tf.data.Dataset): empty dataset
        labels (list[str]): final list of labels, from most to least frequent
    """
    bioasq_dir = os.path.join(data_dir, DESTINATION)

    return utils.mlc_tfrecords(bioasq_dir, lambda: parse(fetch(data_dir)))


def parse(bioasq_dir):
    """
    Parses the BioASQ dataset.

    Splits the data according to "Maximizing Subset Accuracy with Recurrent
    Neural Networks in Multi-label Classification" - Jinseok Nam et al. (2017),
    but using docs from 2017/2018 instead of 2014/2015.

    All documents from years other than 2017 and 2018 become the training set.

    Args:
        bioasq_dir (str): absolute path to the extracted BioASQ dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): empty list
        labels (list[str]): final list of labels, from most to least frequent
    """
    train_docs = []
    test_docs = []
    unused_docs = []
    label_counts = Counter()

    bioasq_json = os.path.join(bioasq_dir, JSON_NAME)
    with open(bioasq_json) as f:
        for article in ijson.items(f, "articles.item"):
            title = article["title"]
            text = article["abstractText"]
            journal = article["journal"]
            year = article["year"]
            labels = article["meshMajor"]

            doc = Doc(title, journal, None, text, labels)

            if year in TEST_YEARS:
                test_docs.append(doc)
            else:
                train_docs.append(doc)

            label_counts.update(labels)

    # Get list of labels, from frequent to rare
    labels = [l[0] for l in label_counts.most_common()]

    return train_docs, test_docs, unused_docs, labels


def fetch(data_dir):
    """
    Fetches and extracts the BioASQ dataset.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        bioasq_dir (str): absolute path to the folder where BioASQ is stored
    """
    file_path = os.path.join(data_dir, DESTINATION, ZIP_NAME)
    result_path = os.path.join(data_dir, DESTINATION, JSON_NAME)
    return utils.fetch(URL, file_path, result_path)
