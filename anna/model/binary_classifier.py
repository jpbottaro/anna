"""Model using independent binary classifiers for each label"""

import os
import numpy as np
import dataset.utils as utils
import tensorflow as tf
from model.encoder.naive import NaiveEmbeddingEncoder
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import concatenate

# Use TFOptimizer as keras' cannot handle sparcity in the embedding layer well,
# which results in big slowdowns. Replace this with stock keras optimizers when
# this is fixed
from tensorflow.python.keras._impl.keras.optimizers import TFOptimizer


class BinaryClassifierLearner():
    """
    Learner for Multi-label Classification using binary classifiers.

    It also allows to create a classifier chain, adding dependency between
    each classification in the output.
    """

    def __init__(self,
                 data_dir,
                 output_labels,
                 name="binary_model",
                 max_words=300,
                 confidence_threshold=0.3,
                 chain=False,
                 verbose=False):
        """
        Maps a Multi-label classification problem into binary classifiers,
        having one independent classifier for each label. All models share
        the core embedding layer.

        Args:
            data_dir (str): path to the folder where datasets are stored
            output_labels (list[str]): list of possible outputs
            name (str): name of the model (used when serializing to disk)
            max_words (int): number of words to use when embedding text fields
            confidence_threshold (float): threshold to use to select labels
                                          based on the classifier's confidence
            verbose (bool): print messages of progress (default: False)
        """
        self.data_dir = data_dir
        self.max_words = max_words
        self.verbose = verbose
        self.output_labels = output_labels
        self.confidence_threshold = confidence_threshold
        self.name = name
        self.model_dir = os.path.join(data_dir, "models")
        self.model_path = os.path.join(self.model_dir, self.name)
        self.encoder = NaiveEmbeddingEncoder(data_dir, max_words)

        if os.path.isfile(self.model_path):
            self._log("Loading pretrained model")
            self.model = load_model(self.model_path, custom_objects={"tf": tf})
        else:
            self._log("Building model")
            inputs, emb = self.encoder.build()

            # Each label gets a different, non-shared, fully connected layer
            outputs = []
            for i, label in enumerate(output_labels):
                name = "label_" + label
                if i > 0 and chain:
                    emb = concatenate([emb, outputs[i-1]])
                outputs.append(Dense(1, activation="sigmoid", name=name)(emb))

            self.model = Model(inputs=inputs, outputs=outputs)

        self._log("Compiling model")
        optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.01))
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy")

    def train(self, train_docs, test_docs=None, epochs=3):
        """
        Trains model with the data in `train_docs`.

        Args:
            train_docs (list[Doc]): list of document for training
            test_docs (list[Doc]): list of document for testing. Only for
                                   metrics, not use in the learning process
        """
        input_data = self.encoder.encode(train_docs)
        output_data = self._encode_output(train_docs)

        self.model.fit(input_data, output_data, epochs=epochs)

    def predict(self, docs):
        """
        Adds predicted labels in `docs`.

        Args:
            docs (list[Doc]): list of document to predict labels for

        Returns:
            analyzed_docs (list[Doc]): same as `docs`, with the predictions
        """
        input_data = self.encoder.encode(docs)
        y = self.model.predict(input_data)
        labels = self._decode_output(y)

        for i, doc in enumerate(docs):
            doc.labels = labels[i]

        return docs

    def save(self):
        """
        Saves the model in the model directory, with the given `name`.

        Args:
            name (str): the name for the model
        """
        utils.create_folder(self.model_dir)
        self.model.save(self.model_path)

    def _encode_output(self, docs):
        """
        Encodes the labels of the given `docs` into vector representations,
        to feed into the model.

        Args:
            docs (list[Doc]): list of document to encode labels for

        Returns:
            output_tensors (list[np.array]): expected true output for the `docs`
        """
        labels = []
        for label in self.output_labels:
            labels.append(
                np.array([[1. if label in d.labels else 0.] for d in docs]))

        return labels

    def _decode_output(self, outputs):
        """
        Decodes the output of the model, returning the labels with confidence
        above the threshold provided.

        Args:
            outputs (list[np.array]): batch of vector outputs from the model

        Returns:
            labels (list[list[str]]): labels for each output provided
        """
        num_docs = outputs[0].shape[0]
        labels = [[] for _ in range(num_docs)]

        # Loops on each binary classifier (representing each label)
        for i, tensor in enumerate(outputs):
            label = self.output_labels[i]
            for j in range(num_docs):
                if tensor[j] > self.confidence_threshold:
                    labels[j].append(label)

        return labels

    def _log(self, text):
        """
        Prints the provided `text`, only if the model was configured as
        `verbose`.

        Args:
            text (str): the text to print
        """
        if self.verbose:
            print(text)
