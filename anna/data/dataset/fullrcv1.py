"""Reads the RCV1-v2 dataset for Multi-label Classification

This version builds the full Reuters dataset (with original title and text),
using the RCV1-v2 test/training split _switched_, as done in:

    Jinseok Nam, Eneldo Loza Mencía, Hyunwoo J. Kim, Johannes Fürnkranz
    Maximizing Subset Accuracy with Recurrent Neural Networks in
    Multi-label Classification.
    NIPS 2017: 5419-5429

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.

Visit: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm"""  # noqa
import os
import anna.data.utils as utils
import anna.data.dataset.reuters as reuters
import anna.data.dataset.rcv1 as rcv1

NAME = "full-rcv1-v2"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the RCV1-v2 dataset, with the original title/text (not
    the stemmed/lower-cased version).

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (tf.data.Dataset): annotated articles for training
        test_docs (tf.data.Dataset): annotated articles for testing
        unused_docs (tf.data.Dataset): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    rcv_dir = os.path.join(data_dir, NAME)

    return utils.mlc_tfrecords(rcv_dir, lambda: parse(data_dir))


def parse(data_dir):
    """
    Parses both the original Reuters, and the RCV1-v2. Builds a dataset using
    Reuters documents, with the RCV1-v2 train/test split switched.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    reut_data = reuters.parse(reuters.fetch(data_dir))
    rcv1_data = rcv1.parse(rcv1.fetch(data_dir))

    reut_train, reut_test, _, reut_labels = reut_data
    rcv1_train, rcv1_test, _, rcv1_labels = rcv1_data

    # Index Reuters docs by id
    reut_all = {doc.doc_id: doc for doc in reut_train + reut_test}

    # Fetch ids for train/test splits in RCV1-v2
    rcv1_train = [doc.doc_id for doc in rcv1_train]
    rcv1_test = [doc.doc_id for doc in rcv1_test]

    # Re-build train/test for Reuters using RCV1-v2 splits
    train = [reut_all[i] for i in rcv1_train]
    test = [reut_all[i] for i in rcv1_test]

    return train, test, [], reut_labels
