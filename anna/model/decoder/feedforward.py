"""MLP/FFN-based decoder for Multi-label Classification."""

import os
import numpy as np
import tensorflow as tf


class FeedForwardDecoder():
    """
    MLP-based decoder for Multi-label Classification, mapping each possible
    label with a single 1/0 classifier.

    The resulting label set is the set of classifiers that output confidence
    above a pre-defined threshold.
    """

    def __init__(self,
                 data_dir,
                 labels,
                 confidence_threshold,
                 num_layers,
                 hidden_size,
                 chain,
                 hinge):
        """
        Maps a Multi-label classification problem into binary classifiers,
        having one independent MLP-like classifier for each label.

        Args:
            data_dir (str): path to the folder where datasets are stored
            labels (list[str]): list of possible outputs
            confidence_threshold (float): threshold to use to select labels
                                          based on the classifier's confidence
            num_layers (int): number of hidden layers in the MLP
            hidden_size (int): size of the hidden units on each hidden layer
            chain (bool): True if classifiers' output should be chained
            hinge (bool): True if loss should be hinge (i.e. maximum margin)
        """
        self.data_dir = data_dir
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.chain = chain
        self.loss = "binary_crossentropy"

    def build(self, inputs, x):
        """
        Builds the MLP classification layer, as a series of binary classifiers
        for each possible label.

        Args:
            inputs (list[tf.keras.layers.Input]): list of inputs of the model
            embedding (tf.keras.layers.Layer): a layer/tensor with the doc
                                               embedding to use as starting
                                               ground.

        Returns:
            outputs (list[tf.keras.layers.Layer]): list of binary classifier
                                                   outputs, one per label
        """
        for i in range(self.num_layers):
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)

        # Each label gets a different, non-shared, fully connected layer
        outputs = []
        for i, label in enumerate(self.labels):
            name = "label_" + label
            if i > 0 and self.chain:
                x = tf.keras.layers.concatenate([x, outputs[i-1]])
            outputs.append(tf.keras.layers.Dense(1, activation="sigmoid",
                                                 name=name)(x))

        return outputs

    def encode(self, labels):
        """
        Encodes the label sets into vector representations,
        to feed into the model as the expected output.

        Args:
            labels (list[list[str]]): list of `Doc` to encode as vector

        Returns:
            output (list[np.array]): encoded label sets from `labels`
        """
        output = []

        for label in self.labels:
            output.append(
                np.array([[1. if label in l else 0.] for l in labels]))

        return output

    def decode(self, outputs):
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
            label = self.labels[i]
            for j in range(num_docs):
                if tensor[j] > self.confidence_threshold:
                    labels[j].append(label)

        return labels
