"""Multi layer perceptron using different binary classifiers for each label"""

import os
import numpy as np
import tensorflow as tf
import dataset.utils
from evaluation.mlc import Evaluator
from model.encoder.naive import NaiveEmbeddingEncoder
from model.decoder.feedforward import FeedForwardDecoder

# Use TFOptimizer as keras' cannot handle sparcity in the embedding layer well,
# which results in big slowdowns. Replace this with stock keras optimizers when
# this is fixed
from tensorflow.python.keras._impl.keras.optimizers import TFOptimizer


class Trainer():
    """
    Trains Multi-label Classification models. Generalizes most common models
    using:
        - Encoder: Takes the input and creates a tensor representation (e.g.
                   average word embeddings, RNN, etc.)
        - Decoder: Original input and output from the encoder, and builds the
                   result (e.g. feedforward, classifier chains, RNN, etc.)
    """

    def __init__(self, data_dir, labels, name, encoder, decoder,
                 optimizer, verbose=True):
        """
        Maps a Multi-label classification problem into binary classifiers,
        having one independent classifier for each label. All models share
        the core embedding layer.

        Args:
            data_dir (str): path to the folder where datasets are stored
            labels (list[str]): list of possible outputs
            name (str): name of the model (used when serializing to disk)
            encoder (Encoder): encoder model to process the input
            decoder (Decoder): decoder model to process encoded input and
                               produce the MLC labels
            optimizer (Optimizer): Keras optimizer to use for training
            verbose (bool): print messages of progress (default: False)
        """
        self.data_dir = data_dir
        self.labels = labels
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.verbose = 1 if verbose else 0
        self.model_dir = os.path.join(data_dir, "models")
        self.model_path = os.path.join(self.model_dir, name)

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
        self.model.compile(optimizer=optimizer, loss=decoder.get_loss())

        # TODO: Add metric filtering to Keras
        self._fix_metrics(self.model)

    def train(self, docs, test_docs=None, val_split=0.1, epochs=50):
        """
        Trains model with the data in `train_docs`.

        Args:
            docs (list[Doc]): list of document for training
            test_docs (list[Doc]): list of document for test evaluation (only
                                   for reporting, no training decisions are
                                   made with this set).
            val_split (float): fraction of `docs` to use for validation
            epochs (int): number of epochs to run the data for training

        Returns:
            history (History): keras' history, with record of loss values, etc.
        """
        split = int(len(docs) * (1. - val_split))
        train_docs = docs[0:split]
        val_docs = docs[split:]

        val_eval = Evaluator("val", self.predict, val_docs, self.labels)
        test_eval = Evaluator("test", self.predict, test_docs, self.labels)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_acc",
                                                patience=20,
                                                verbose=self.verbose)
        save = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                                  monitor="val_acc",
                                                  verbose=self.verbose,
                                                  save_best_only=True)

        train_x = self.encoder.encode(train_docs)
        train_y = self.decoder.encode([d.labels for d in train_docs])
        val_x = self.encoder.encode(val_docs)
        val_y = self.decoder.encode([d.labels for d in val_docs])
        return self.model.fit(train_x, train_y,
                              epochs=epochs,
                              validation_data=(val_x, val_y),
                              callbacks=[val_eval, test_eval, stop, save])

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
        Removes unneded metrics in the model. Very hacky, might be a possible
        addition to Keras.

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


class MLP(Trainer):
    """
    Maps a Multi-label classification problem into binary classifiers,
    having one independent classifier for each label. All outputs share
    the core embedding layer.
    """

    def __init__(self,
                 data_dir,
                 labels,
                 name="mlp",
                 max_words=300,
                 confidence_threshold=0.5,
                 num_layers=2,
                 hidden_size=1024,
                 voc_size=200000,
                 chain=False,
                 train_emb=True,
                 verbose=True):
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
                              (default: 2)
            hidden_size (int): size of the hidden units on each hidden layer
                               (default: 1024)
            chain (bool): True if classifiers' output should be chained
                          (default: False)
            train_emb (bool): True if word embeddings should be trainable
                          (default: True)
            verbose (bool): print messages of progress (default: True)
        """
        # Encode doc as average of its initial `max_words` word embeddings
        encoder = NaiveEmbeddingEncoder(data_dir, max_words,
                                        train_emb, voc_size)

        # Classify labels with independent logistic regressions
        decoder = FeedForwardDecoder(data_dir, labels, confidence_threshold,
                                     num_layers, hidden_size, chain)

        # Optimizer (wraps a TF optimizer as keras' is bad with sparce updates)
        optimizer = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=0.001))

        super().__init__(data_dir, labels, name, encoder, decoder, optimizer,
                         verbose)
