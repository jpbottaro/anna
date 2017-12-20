"""Model using independent binary classifiers for each label"""

import os
import numpy as np
import nlp.utils as nlp
import dataset.utils as utils
import dataset.fasttext.parser as embeddings
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding, Dense, Lambda
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model, load_model

# Use TFOptimizer as keras' cannot handle sparcity in the embedding layer well,
# which results in big slowdowns. Replace this with stock keras optimizers when
# this is fixed
from tensorflow.python.keras._impl.keras.optimizers import TFOptimizer


class BinaryClassifierLearner():
    """
    Learner for Multi-label Classification using independent binary classifiers.
    """

    def __init__(self,
                 data_dir,
                 output_labels,
                 name="binary_model",
                 max_words=200,
                 confidence_threshold=0.3,
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
        utils.create_folder(self.model_dir)

        self._log("Fetching pretrained embeddings")
        self.voc, self.emb = embeddings.fetch_and_parse(data_dir)

        model_path = os.path.join(self.model_dir, self.name)
        if os.path.isfile(model_path):
            self._log("Loading pretrained model")
            self.model = load_model(model_path, custom_objects={'tf': tf})
        else:
            self._log("Building input layer")
            x1 = Input(shape=(max_words,), dtype='int32', name='title_input')
            x2 = Input(shape=(max_words,), dtype='int32', name='text_input')

            # Get embeddings
            emb_layer = Embedding(self.emb.shape[0],
                                  self.emb.shape[1],
                                  weights=[self.emb],
                                  input_length=max_words,
                                  mask_zero=True)
            x1_emb = emb_layer(x1)
            x2_emb = emb_layer(x2)

            # Average all embeddings to create each text representation
            avg_layer = Lambda(lambda x: tf.reduce_mean(x, 1))
            x1_emb = avg_layer(x1_emb)
            x2_emb = avg_layer(x2_emb)

            # Concatenate all inputs
            x = concatenate([x1_emb, x2_emb], axis=-1)

            # Each label gets a different, non-shared, fully connected layer
            outputs = []
            self._log("Building output layers")
            for i, label in enumerate(output_labels):
                name = "label_" + str(i)
                outputs.append(Dense(1, activation='sigmoid', name=name)(x))

            self.model = Model(inputs=[x1, x2], outputs=outputs)

        self._log("Compiling model")
        optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.01))
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')

    def train(self, train_docs, test_docs=None, epochs=10):
        """
        Trains model with the data in `train_docs`.

        Args:
            train_docs (list[Doc]): list of document for training
            test_docs (list[Doc]): list of document for testing. Only for
                                   metrics, not use in the learning process
        """
        input_data, output_data = self._encode_input(train_docs)

        self._log("Training")
        self.model.fit(input_data, output_data, epochs=epochs)
        self.model.save(os.path.join(self.model_dir, self.name))

    def predict(self, docs):
        """
        Adds predicted labels in `docs`.

        Args:
            docs (list[Doc]): list of document to predict labels for

        Returns:
            analyzed_docs (list[Doc]): same as `docs`, with the predictions
        """
        self._log("Building input data for prediction")
        input_data, output_data = self._encode_input(docs)

        self._log("Predicting labels")
        y = self.model.predict(input_data)

        self._log("Decoding output")
        labels = self._decode_output(y)

        for i, doc in enumerate(docs):
            doc.labels = labels[i]
        return docs

    def _encode_input(self, docs):
        """
        Encodes the given document into vector representations, to feed into
        the model.

        Args:
            docs (list[Doc]): list of document to predict labels for

        Returns:
            input_tensors (list[np.array]): tensors representing the `docs`
            output_tensors (list[np.array]): expected true output for the `docs`
        """
        self._log("Mapping docs to vectors")
        title = nlp.text_to_sequence([d.title for d in docs],
                                     self.voc, self.max_words)
        text = nlp.text_to_sequence([d.text for d in docs],
                                    self.voc, self.max_words)

        labels = []
        for label in self.output_labels:
            labels.append(
                np.array([[1. if label in d.labels else 0.] for d in docs]))

        return [title, text], labels

    def _decode_output(self, outputs):
        """
        Decodes the output of the model, returning the labels with confidence
        above the threshold provided.

        Args:
            outputs (list[np.array]): batch of vector outputs from the model

        Returns:
            docs_labels (list[list[str]]): labels for each output provided
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
