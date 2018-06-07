"""Main entry point for any experiments."""

import os
import sys
import anna.model.premade as models
import anna.data.dataset.reuters21578 as reuters
import anna.data.dataset.rcv1 as rcv1
import anna.data.dataset.bioasq as bioasq


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py DATA_FOLDER [reuters|rcv1|bioasq]")
        exit(1)

    dataset = "reuters"
    if len(sys.argv) == 3:
        dataset = sys.argv[2]

    if dataset == "reuters":
        data = reuters
        folder = "model"
        val_size = 777
        epochs = 40
    elif dataset == "rcv1":
        data = rcv1
        folder = "model-rcv1"
        val_size = 78126
        epochs = 5
    elif dataset == "bioasq":
        data = bioasq
        folder = "model-bioasq"
        val_size = 50000
        epochs = 3
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    # Resolve data folders
    data_dir = os.path.abspath(sys.argv[1])

    # Fetch and preprocess dataset
    train_docs, test_docs, unused_docs, labels = data.fetch_and_parse(data_dir)

    for builder in models.ALL:
        # Create default trainer
        model = builder(data_dir, labels, folder_name=folder)

        # Train and evaluate
        print("Model: {}".format(model))
        model.train(train_docs, test_docs, val_size=val_size, epochs=epochs)

        # Delete to save memory
        del model
