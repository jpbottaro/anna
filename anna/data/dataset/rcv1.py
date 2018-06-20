"""Reads the RCV1-v2 dataset for Multi-label Classification

Important: Train and test sets are _switched_, since the original split leaves
the sides unbalanced.

Visit: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm"""  # noqa
import os
import gzip
import collections
import anna.data.utils as utils
from anna.data.api import Doc

NAME = "rcv1-v2"

HOST = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/"
FILE_URL_FORMAT = HOST + "a12-token-files/{}"
TOPIC_URL_FORMAT = HOST + "a08-topic-qrels/{}"

TEST_FILES = ["lyrl2004_tokens_test_pt0.dat.gz",
              "lyrl2004_tokens_test_pt1.dat.gz",
              "lyrl2004_tokens_test_pt2.dat.gz",
              "lyrl2004_tokens_test_pt3.dat.gz"]
TRAIN_FILE = "lyrl2004_tokens_train.dat.gz"
TOPICS_FILE = "rcv1-v2.topics.qrels.gz"

FILE_URLS = {f: FILE_URL_FORMAT.format(f) for f in TEST_FILES + [TRAIN_FILE]}
FILE_URLS[TOPICS_FILE] = TOPIC_URL_FORMAT.format(TOPICS_FILE)

TEST_FINAL = "test.dat"
TRAIN_FINAL = "train.dat"
TOPICS_FINAL = "topics.dat"


def fetch_and_parse(data_dir):
    """
    Fetches and parses the RCV1-v2 dataset.

    Args:
        data_dir (str): absolute path to the dir where datasets are stored

    Returns:
        train_docs (tf.data.Dataset): annotated articles for training
        test_docs (tf.data.Dataset): annotated articles for testing
        unused_docs (tf.data.Dataset): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    rcv_dir = os.path.join(data_dir, NAME)

    return utils.mlc_tfrecords(rcv_dir, lambda: parse(fetch(data_dir)))


def parse(rcv1_dir):
    """
    Parses the RCV1-v2 dataset.

    Args:
        rcv1_dir (str): absolute path to the extracted RCV1-v2 dir

    Returns:
        train_docs (list[Doc]): annotated articles for training
        test_docs (list[Doc]): annotated articles for testing
        unused_docs (list[Doc]): unused docs
        labels (list[str]): final list of labels, from most to least frequent
    """
    train_docs = []
    test_docs = []
    unused_docs = []
    label_counts = collections.Counter()

    topics = collections.defaultdict(set)
    topics_path = os.path.join(rcv1_dir, TOPICS_FINAL)
    with open(topics_path, "r") as f:
        for line in f:
            split = line.split(" ")
            topic = split[0].strip()
            doc_id = split[1].strip()
            topics[doc_id].add(topic)

    # IMPORTANT: we switch the order of original train and test
    for path, docs, is_train in [(TEST_FINAL, train_docs, True),
                                 (TRAIN_FINAL, test_docs, False)]:
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
                    if is_train:
                        label_counts.update(labels)
                    docs.append(Doc(doc_id, None, None, None, text, labels))
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

    # Get list of labels, from frequent to rare
    labels = [l[0] for l in label_counts.most_common()]

    return train_docs, test_docs, unused_docs, labels


def fetch(data_dir):
    """
    Fetches the tokenized RCV1-v2 dataset.

    Args:
        data_dir (str): absolute path to the folder where datasets are stored

    Returns:
        final_dir (str): absolute path where RCV1-v2 was extracted
    """
    # Create folder
    rcv1_dir = os.path.join(data_dir, NAME)
    utils.create_folder(rcv1_dir)

    # Download all datasets
    for f, url in FILE_URLS.items():
        path = os.path.join(rcv1_dir, f)
        if not os.path.exists(path):
            utils.urlretrieve(url, path)

    # Extract topics
    path = os.path.join(rcv1_dir, TOPICS_FINAL)
    if not os.path.exists(path):
        src = os.path.join(rcv1_dir, TOPICS_FILE)
        with open(path, "wb") as o, gzip.open(src, "rb") as i:
            o.write(i.read())

    # Extract train
    path = os.path.join(rcv1_dir, TRAIN_FINAL)
    if not os.path.exists(path):
        src = os.path.join(rcv1_dir, TRAIN_FILE)
        with open(path, "wb") as o, gzip.open(src, "rb") as i:
            o.write(i.read())

    # Extract test
    path = os.path.join(rcv1_dir, TEST_FINAL)
    if not os.path.exists(path):
        with open(path, "wb") as o:
            for p in TEST_FILES:
                src = os.path.join(rcv1_dir, p)
                with gzip.open(src, "rb") as i:
                    o.write(i.read())

    return rcv1_dir
