"""Main entry point for any experiments."""

import os
import sys
import dataset.reuters21578.parser as data
from model.mlp import MLP


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py DATA_FOLDER")
        exit(1)

    # Resolve data folder
    data_dir = os.path.abspath(sys.argv[1])

    # Fetch and preprocess dataset
    train_docs, test_docs, unused_docs = data.fetch_and_parse(data_dir)
    labels = []
    for d in train_docs + test_docs:
        for l in d.labels:
            if l not in labels:
                labels.append(l)

    # Create MLP with 2 hidden layer
    model = MLP(data_dir, labels, verbose=True)

    # Train model
    model.train(train_docs, test_docs)
