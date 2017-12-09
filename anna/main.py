"""Main entry point for any experiments."""

import os
import pickle
import dataset.reader.reuters21578 as reuters

REUTERS_PATH = "dataset/data/reuters21578"

TRAIN_PICKLE = "train.pickle"
TEST_PICKLE = "test.pickle"


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(curr_path, REUTERS_PATH)
    train_path = os.path.join(path, TRAIN_PICKLE)
    test_path = os.path.join(path, TEST_PICKLE)

    if not os.path.isfile(train_path):
        train_docs, test_docs = reuters.parse(path)

        with open(train_path, "wb") as f:
            pickle.dump(train_docs, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_docs, f)
    else:
        with open(train_path, "rb") as f:
            train_docs = pickle.load(f)
        with open(test_path, "rb") as f:
            test_docs = pickle.load(f)

    doc = test_docs[0]
    print("Text: " + doc.text[:100])
    print("Labels: " + str(doc.labels))
