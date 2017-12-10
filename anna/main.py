"""Main entry point for any experiments."""

import os
import sys
import dataset.reuters21578.parser as reuters


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py DATA_FOLDER")
        exit(1)

    data_dir = os.path.abspath(sys.argv[1])

    train_docs, test_docs, unused_docs = reuters.fetch_and_parse(data_dir)

    doc = test_docs[0]
    print("Text: " + doc.text[:100])
    print("Labels: " + str(doc.labels))
