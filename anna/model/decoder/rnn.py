"""MLP/FFN-based decoder for Multi-label Classification."""

import os
import numpy as np
import tensorflow as tf


class RNNDecoder():
    """
    RNN-based decoder for Multi-label Classification.
    """

    def __init__(self,
                 data_dir,
                 labels,
                 hidden_size=256,
                 max_steps=50,
                 max_output_labels=20):
        """
        Maps a Multi-label classification problem into a series of individual
        label prediction, stopping when reaching a sentinel label.

        Args:
            data_dir (str): path to the folder where datasets are stored
            labels (list[str]): list of possible outputs
            hidden_size (int): size of the hidden units on each hidden layer
            max_steps (int): maximum number of steps the RNN should make
            max_output_labels (int): maximum number of labels to output
        """
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.max_output_labels = max_output_labels
        self.special_labels = ["_PAD_", "_UNK_", "_END_"]
        self.id2labels = dict(enumerate(self.special_labels + labels))
        self.labels = {c: i for i, c in self.id2labels.items()}
        self.loss = rnn_loss

    def build(self, inputs, fixed_emb, var_emb):
        """
        Builds the RNN decoding layer, as a series of recurrent softmax
        classifiers that find one label at a time.

        Args:
            inputs (list[tf.keras.layers.Input]): list of inputs of the model
            fixed_emb (tf.keras.layers.Layer): a layer/tensor with a fixed
                                               embedding of the input
            var_emb (tf.keras.layers.Layer): a layer/tensor with an variable
                                             embedding of the input, decoder
                                             would need to process it (e.g.
                                             using attention)

        Returns:
            outputs (tf.keras.layers.Layer): RNN decoder for the label set
        """
        # (batch, max_topics, nr_topics)
        return RNNDecoderLayer(self.max_output_labels,
                               len(self.id2labels),
                               self.hidden_size)(fixed_emb)

    def encode(self, labels):
        """
        Encodes the label sets into vector representations,
        to feed into the model as the expected output.

        Args:
            labels (list[list[str]]): list of labels to encode as vector

        Returns:
            output (np.array): encoded label sets from `labels`
        """
        # Labels to ids
        ids = [[self.labels[label] for label in doc_labels]
                                   for doc_labels in labels]

        # Order (random for now)
        ids = [sorted(i) for i in ids]

        # Add special ending step
        ids = [i + [self.labels["_END_"]] for i in ids]

        # Pad to make it a squared matrix
        maxlen = self.max_output_labels
        ids = tf.keras.preprocessing.sequence.pad_sequences(ids,
                                                            maxlen=maxlen,
                                                            padding="post")

        # Turn ids into one-hot vectors
        return np.array([tf.keras.utils.to_categorical(i, len(self.labels))
                         for i in ids])

    def decode(self, outputs):
        """
        Decodes the output of the model, returning the resulting labels.

        Args:
            outputs (np.array): batch of vector outputs from the model

        Returns:
            labels (list[list[str]]): labels for each output provided
        """
        num_docs = outputs.shape[0]
        num_steps = outputs.shape[1]

        # Retrieve max indexes from each softmax
        max_ids = np.argmax(outputs, axis=2)

        result = []
        for doc in range(num_docs):
            labels = []
            for step in range(num_steps):
                label = self.id2labels[max_ids[doc, step]]
                if label in self.special_labels:
                    break
                labels.append(label)
            result.append(labels)

        return result


class RNNDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, max_labels, nr_labels, hidden_size, **kwargs):
        self.max_labels = max_labels
        self.nr_labels = nr_labels
        self.hidden_size = hidden_size
        super(RNNDecoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hidden_projection = self.add_weight(name='hidden_projection',
                                                 shape=(input_shape[1],
                                                        self.hidden_size),
                                                 initializer='uniform',
                                                 trainable=True)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.hidden_size, self.nr_labels),
                                      initializer='uniform',
                                      trainable=True)
        self.cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        super(RNNDecoderLayer, self).build(input_shape)

    def call(self, x):
        shape = tf.shape(x)

        # Initial state and input
        state = tf.matmul(x, self.hidden_projection)
        output = tf.ones([shape[0], self.nr_labels]) / self.nr_labels
        outputs = []

        for i in range(self.max_labels):
            output, state = self.cell(output, state)
            output = tf.matmul(output, self.kernel)
            outputs.append(output)

        # [max_labels, batch_size, nr_labels]
        outputs = tf.stack(outputs)

        # [batch_size, max_labels, nr_labels]
        return tf.transpose(outputs, [1, 0, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_labels, self.nr_labels)


def rnn_loss(y_true, y_pred):
    """
    Loss for decoding RNN.

    Args:
        y_true (Tensor): expected values as probabilities (shape=[batch, 1])
        y_pred (Tensor): predicted values as logits (shape=[batch, 1])

    Returns:
        loss (Tensor): hinge loss of y_pred against y_true
    """
    # [batch_size * num_steps] - Pads get 0, others 1
    mask = tf.reduce_max(y_true, axis=2)
    mask = tf.sign(mask)
    mask = tf.reshape(mask, [-1])

    # [batch_size * num_steps, nr_labels] - Cross entropy on the predictions
    shape = tf.shape(y_true)
    y_pred = tf.reshape(y_pred, [-1, shape[2]])
    y_true = tf.reshape(y_true, [-1, shape[2]])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                      labels=y_true)

    # Mask loss
    return tf.reshape(loss * mask, [shape[0], shape[1]])
