"""Main entry point for all experiments."""

import os
import sys
import anna.model.premade as models
import anna.data.dataset.reuters21578 as reuters
import anna.data.dataset.reuters as rcv1
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
        shuffle = 10000
        input_names = ["title", "text"]
    elif dataset == "rcv1":
        data = rcv1
        folder = "model-rcv1"
        val_size = 75000
        epochs = 5
        shuffle = 750000
        input_names = ["text"]
    elif dataset == "bioasq":
        data = bioasq
        folder = "model-bioasq"
        val_size = 100000
        epochs = 2
        shuffle = 1000000
        input_names = ["title", "text"]
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    # Resolve data folders
    data_dir = os.path.abspath(sys.argv[1])

    # Fetch and preprocess dataset
    train_docs, test_docs, unused_docs, labels = data.fetch_and_parse(data_dir)

    for builder in models.BEST:
        # Create default trainer
        model = builder(data_dir, labels,
                        input_names=input_names,
                        folder_name=folder)

        # Train and evaluate
        print("Model: {}".format(model))
        model.train(train_docs, test_docs,
                    val_size=val_size, epochs=epochs, shuffle=shuffle)

        # Delete to save memory
        del model
