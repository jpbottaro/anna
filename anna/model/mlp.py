"""Multi layer perceptron using different binary classifiers for each label"""

import os
import numpy as np
import tensorflow as tf
import dataset.utils
from evaluation.mlc import clean, evaluate
from model.encoder.naive import NaiveEmbeddingEncoder
from model.decoder.feedforward import FeedForwardDecoder

# Use TFOptimizer as keras' cannot handle sparcity in the embedding layer well,
# which results in big slowdowns. Replace this with stock keras optimizers when
# this is fixed
from tensorflow.python.keras._impl.keras.optimizers import TFOptimizer


class MLPLearner():
    """
    Learner for Multi-label Classification using a MLP.

    It also allows to create a classifier chain, adding dependency between
    each classification in the output.
    """

    def __init__(self,
                 data_dir,
                 labels,
                 name="mlp",
                 max_words=300,
                 confidence_threshold=0.5,
                 num_layers=0,
                 hidden_size=1024,
                 chain=False,
                 verbose=False):
        """
        Maps a Multi-label classification problem into binary classifiers,
        having one independent classifier for each label. All models share
        the core embedding layer.

        Args:
            data_dir (str): path to the folder where datasets are stored
            labels (list[str]): list of possible outputs
            name (str): name of the model (used when serializing to disk)
            max_words (int): number of words to use when embedding text fields
            confidence_threshold (float): threshold to use to select labels
                                          based on the classifier's confidence
            num_layers (int): number of hidden layers in the MLP
                              (default: 0)
            hidden_size (int): size of the hidden units on each hidden layer
                               (default: 1024)
            chain (bool): True if classifiers' output should be chained
                          (default: False)
            verbose (bool): print messages of progress (default: False)
        """
        self.data_dir = data_dir
        self.max_words = max_words
        self.verbose = verbose
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.name = name
        self.model_dir = os.path.join(data_dir, "models")
        self.model_path = os.path.join(self.model_dir, name)
        self.checkpointer = \
            tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                               verbose=1,
                                               save_best_only=True)

        self.encoder = NaiveEmbeddingEncoder(data_dir, max_words)
        self.decoder = FeedForwardDecoder(data_dir,
                                          labels,
                                          confidence_threshold,
                                          num_layers,
                                          hidden_size,
                                          chain)

        dataset.utils.create_folder(self.model_dir)
        if os.path.isfile(self.model_path):
            self._log("Loading pretrained model")
            self.model = tf.keras.models.load_model(self.model_path,
                                                    custom_objects={"tf": tf})
        else:
            self._log("Building model")
            inputs, emb = self.encoder.build()
            outputs = self.decoder.build(inputs, emb)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        self._log("Compiling model")
        optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.01))
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy")

        # TODO: Add metric filtering to Keras
        self._fix_metrics(self.model)

    def train(self, train_docs, callbacks=None, val_split=0.1, epochs=5):
        """
        Trains model with the data in `train_docs`.

        Args:
            train_docs (list[Doc]): list of document for training
            callbacks (list[Callback]): list of keras callbacks to add for
                                        training
            val_split (float): fraction of `train_docs` to use for validation
            epochs (int): number of epochs to run the data for training

        Returns:
            history (History): keras' history, with record of loss values, etc.
        """
        input_data = self.encoder.encode(train_docs)
        output_data = self.decoder.encode([d.labels for d in train_docs])
        return self.model.fit(input_data, output_data,
                              epochs=epochs,
                              validation_split=val_split,
                              callbacks=callbacks + [self.checkpointer])

    def predict(self, docs):
        """
        Adds predicted labels in `docs`.

        Args:
            docs (list[Doc]): list of document to predict labels for

        Returns:
            analyzed_docs (list[Doc]): same as `docs`, with the predictions
        """
        input_data = self.encoder.encode(docs)
        output_data = self.model.predict(input_data)
        labels = self.decoder.decode(output_data)

        for i, doc in enumerate(docs):
            doc.labels = labels[i]

        return docs

    def save(self):
        """
        Saves the model in the model directory, with the given `name`.

        Args:
            name (str): the name for the model
        """
        self.model.save(self.model_path)

    def _fix_metrics(self, model):
        """
        Removes unneded metrics in the model.

        Args:
            model (tf.keras.models.Model): keras model to modify
        """
        new_names = []
        new_tensors = []
        for name, tensor in zip(model.metrics_names, model.metrics_tensors):
            if "label_" not in name:
                new_names.append(name)
                new_tensors.append(tensor)
        model.metrics_names = new_names
        model.metrics_tensors = new_tensors

    def _log(self, text):
        """
        Prints the provided `text`, only if the model was configured as
        `verbose`.

        Args:
            text (str): the text to print
        """
        if self.verbose:
            print(text)
