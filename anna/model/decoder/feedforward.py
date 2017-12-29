"""MLP/FFN-based decoder for Multi-label Classification."""

import os
import numpy as np
import tensorflow as tf


class FeedForwardDecoder():
    """
    MLP-based decoder for Multi-label Classification, with an independent
    classifier per label.
    """

    def __init__(self,
                 data_dir,
                 labels,
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
            num_layers (int): number of hidden layers in the MLP
            hidden_size (int): size of the hidden units on each hidden layer
            chain (bool): True if classifiers' output should be chained
            hinge (bool): True if loss should be hinge (i.e. large margin),
                          otherwise crossentropy is used
        """
        self.data_dir = data_dir
        self.labels = labels
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.chain = chain
        self.hinge = hinge

        if hinge:
            self.loss = binary_hinge
            self.output_reg = tf.keras.regularizers.l2(0.001)
        else:
            self.loss = binary_crossentropy
            self.output_reg = None

    def build(self, inputs, fixed_emb, var_emb):
        """
        Builds the MLP classification layer, as a series of binary classifiers
        for each possible label.

        Args:
            inputs (list[tf.keras.layers.Input]): list of inputs of the model
            fixed_emb (tf.keras.layers.Layer): a layer/tensor with a fixed
                                               embedding of the input
            var_emb (tf.keras.layers.Layer): a layer/tensor with an variable
                                             embedding of the input, decoder
                                             would need to process it (e.g.
                                             using attention)

        Returns:
            outputs (list[tf.keras.layers.Layer]): list of binary classifier
                                                   outputs, one per label
        """
        # (batch, emb_size)
        x = fixed_emb

        # (batch, hidden_size)
        for i in range(self.num_layers):
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)

        # (batch, hidden_size)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Each label gets a different, non-shared, fully connected layer
        outputs = []
        for i, label in enumerate(self.labels):
            if i > 0 and self.chain:
                x = tf.keras.layers.concatenate([x, outputs[i-1]])

            output = tf.keras.layers.Dense(1,
                                           activation="linear",
                                           kernel_regularizer=self.output_reg,
                                           name="label_" + label)(x)
            outputs.append(output)

        # [(batch, 1)]
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
        outputs = []

        for label in self.labels:
            output = [[1. if label in l else 0.] for l in labels]
            outputs.append(np.array(output))

        return outputs

    def decode(self, outputs):
        """
        Decodes the output of the model, returning the expected labels.

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
                if tensor[j] > 0.:
                    labels[j].append(label)

        return labels


def binary_crossentropy(y_true, y_pred):
    """
    Standard binary crossentropy, assuming y_true are probabilities and y_pred
    are logits.

    Args:
        y_true (Tensor): expected values as probabilities (shape=[batch, 1])
        y_pred (Tensor): predicted values as logits (shape=[batch, 1])

    Returns:
        loss (Tensor): cross entropy loss of y_pred against y_true
    """
    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(l, axis=-1)


def binary_hinge(y_true, y_pred):
    """
    Hinge loss, assuming y_true are binary probabilities and y_pred are logits.

    Args:
        y_true (Tensor): expected values as probabilities (shape=[batch, 1])
        y_pred (Tensor): predicted values as logits (shape=[batch, 1])

    Returns:
        loss (Tensor): hinge loss of y_pred against y_true
    """
    # Turn y_true from probability distribution [0, 1], to [-1, 1]
    y_true = y_true * 2. - 1.
    l = tf.maximum(1. - y_true * y_pred, 0.)
    return tf.reduce_mean(l, axis=-1)
