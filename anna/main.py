"""Main entry point for any experiments."""

import os
import sys
import dataset.reuters21578.parser as data
import nlp.utils as nlp
from evaluation.mlc import evaluate
from model.binary_classifier import BinaryClassifierLearner as Learner


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

    # Create and train model
    model = Learner(data_dir, labels, verbose=True)
    model.train(train_docs, test_docs=test_docs)
    model.save()

    # Predict labels for the test set
    predicted_docs = model.predict(nlp.clean(test_docs))

    # Print evaluation metrics
    print(evaluate(test_docs, predicted_docs, labels))
