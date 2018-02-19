"""Train Encoder/Decoder models for Multi-label Classification tasks."""

import os
import tensorflow as tf
import anna.dataset.utils as utils
from anna.evaluation.mlc import Evaluator

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
                 optimizer, metric, save, verbose=True):
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
            optimizer (str): one of: adam, rmsprop, momentum (default: adam)
            metric (str): metric to optimize (e.g. val_loss, val_ebf1, etc.)
            save (bool): save the best resulting model
            verbose (bool): print messages of progress (default: False)
        """
        self.data_dir = data_dir
        self.labels = labels
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.metric = metric
        self.save = save
        self.verbose = 1 if verbose else 0
        self.model_dir = os.path.join(data_dir, "models")
        self.model_path = os.path.join(self.model_dir, name)

        utils.create_folder(self.model_dir)

        # Optimizer (use TF optimizer as keras' is bad with sparce updates)
        lr = 0.001
        if optimizer == "adam":
            opt = TFOptimizer(tf.train.AdamOptimizer(learning_rate=lr))
        elif optimizer == "rmsprop":
            opt = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=lr))
        elif optimizer == "momentum":
            opt = TFOptimizer(tf.train.MomentumOptimizer(learning_rate=lr,
                                                         momentum=0.9,
                                                         use_nesterov=True))
        else:
            raise ValueError("Unrecognized optimizer: {}".format(optimizer))

        if os.path.isfile(self.model_path):
            self._log("Loading pretrained model")
            self.model = tf.keras.models.load_model(self.model_path,
                                                    custom_objects={"tf": tf})
        else:
            self._log("Building model")
            inputs, fixed_emb = self.encoder.build()
            outputs = self.decoder.build(inputs, fixed_emb)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # The optimizer isn't Keras, so even loaded models have to be compiled
        self._log("Compiling model")
        self.model.compile(optimizer=opt, loss=decoder.loss)

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
        # Split train and validation docs
        split = int(len(docs) * (1. - val_split))
        train_docs = docs[0:split]
        val_docs = docs[split:]

        # Callbacks for evaluation, early-stopping and learning-rate schedule
        mode = "min" if "loss" in self.metric else "max"
        val_eval = Evaluator("val", self.predict, val_docs, self.labels)
        test_eval = Evaluator("test", self.predict, test_docs, self.labels)
        stop = tf.keras.callbacks.EarlyStopping(monitor=self.metric,
                                                patience=20,
                                                mode=mode,
                                                verbose=self.verbose)
        callbacks = [val_eval, test_eval, stop]

        # Save best performing model
        if self.save:
            saver = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                                       monitor=self.metric,
                                                       save_best_only=True,
                                                       mode=mode,
                                                       verbose=self.verbose)
            callbacks.append(saver)

        # Encode input and train!
        train_x = self.encoder.encode(train_docs)
        train_y = self.decoder.encode([d.labels for d in train_docs])
        val_x = self.encoder.encode(val_docs)
        val_y = self.decoder.encode([d.labels for d in val_docs])
        return self.model.fit(train_x, train_y,
                              epochs=epochs,
                              validation_data=(val_x, val_y),
                              callbacks=callbacks)

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
        new_names = ['loss']
        new_tensors = []
        for name, tensor in zip(model.metrics_names[1:],
                                model.metrics_tensors):
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
