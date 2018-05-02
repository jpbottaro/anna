import os
import numpy as np
import tensorflow as tf
import anna.data.utils as utils
from tensorflow.contrib.tensorboard.plugins import projector


class Encoder():
    """
    Encoder that takes the input features, and produces a
    vector representation of them.
    """

    def __init__(self, model_dir, words, emb,
                 input_names=None, max_size=None, oov_buckets=10000):
        """
        Creates an encoder with the given embeddings and maximum size
        for the input.

        Args:
            model_dir (str): path to the folder where the model will be stored
            words (list[str]): list of strings as vocabulary
            emb (np.array): initialization for the word embeddings
            max_size (int): maximum size to use from the input sequence
            oov_buckets (int): nr of buckets to use for out-of-vocabulary words
        """
        if not input_names:
            input_names = ["title", "text"]

        self.emb = emb
        self.words = words
        self.input_names = input_names
        self.oov_buckets = oov_buckets
        self.max_size = max_size
        self.model_dir = model_dir
        self.metadata_path = self.write_words(model_dir)

        if oov_buckets > 0:
            extra_emb = np.random.normal(size=[oov_buckets, emb.shape[1]])
            self.emb = np.concatenate([self.emb, extra_emb])

    def __call__(self, features, mode):
        """
        Builds the encoder for a specific feature `name` in `features`.

        Args:
            features (dict): dictionary of input features
            mode (tf.estimator.ModeKeys): the mode we are on

        Returns:
            y (tf.Tensor): the final representation of `x`
        """
        emb = tf.get_variable("word_embeddings",
                              self.emb.shape,
                              initializer=tf.constant_initializer(self.emb))
        self.write_metadata(emb.name)

        with tf.name_scope("encoder"):
            # Encode all inputs
            inputs = []
            for name in self.input_names:
                with tf.name_scope("input_" + name):
                    x, x_len = get_input(features,
                                         name,
                                         self.words,
                                         emb,
                                         self.max_size,
                                         self.oov_buckets)
                    inputs.append(self.encode(x, x_len, name))

            # Concatenate inputs, two options:
            # fixed: (batch, len(input_names) * emb_size)
            # variable: (batch, sum(input_sizes), emb_size)
            return tf.concat(inputs, 1)

    def write_words(self, model_dir):
        """
        Writes the embedding names for later use in tensorboard.

        Args:
            model_dir (str): path to the folder where the model will be stored
        """
        path = os.path.join(model_dir, "metadata.tsv")
        utils.create_folder(model_dir)
        with open(path, "w") as f:
            for name in self.words:
                print(name, file=f)

        return path

    def write_metadata(self, emb_name):
        """
        Points the variable named `emb_name` to the embedding names.

        Args:
            emb_name (str): name of the tensor with the embeddings
        """
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = emb_name
        embedding.metadata_path = self.metadata_path
        summary_writer = tf.summary.FileWriter(self.model_dir)
        projector.visualize_embeddings(summary_writer, config)

    def encode(self, x, x_len, name):
        """
        Encode a given tensor `x` with length `x_len`.

        Args:
            x (tf.Tensor): the tensor we want to encode
            x_len (tf.Tensor): the length `x`
            name (str): name of the tensor

        Returns:
            y (tf.Tensor): the final representation of `x`
        """
        raise NotImplementedError


def get_input(features, name, words, emb, max_size=None, oov_buckets=0):
    """
    Gets the sequence feature `name` from the `features`,
    trims the size if necessary, and maps it to its list
    of embeddings.

    Args:
        features (dict): dictionary of input features
        name (str): name of the feature to encode
        words (list[str]): list of strings as vocabulary
        emb (tf.Tensor): initialization for the word embeddings
        max_size (int): maximum size to use from the input sequence

    Returns:
        x (tf.Tensor): the tensor of embeddings for the feature `name`
        x_len (tf.Tensor): the length each `x`
    """
    x = features[name]
    x_mask = features[name + "_mask"]

    # Limit size
    # (batch, max_size)
    if max_size:
        with tf.name_scope("trim"):
            x = x[:,:max_size]
            x_mask = x_mask[:,:max_size]

    # Length of each sequence
    # (batch)
    with tf.name_scope("length"):
        x_len = tf.reduce_sum(x_mask, 1)

    with tf.name_scope("embed"):
        # Convert strings to ids
        # (batch, max_size)
        x = tf.contrib.lookup.index_table_from_tensor(
            mapping=words,
            default_value=0,
            num_oov_buckets=oov_buckets).lookup(x)

        # Replace with embeddings
        # (batch, max_size, emb_size)
        x = tf.nn.embedding_lookup(emb, x)

        # Clear embeddings for pads
        # (batch, max_size, emb_size)
        x = tf.multiply(x, tf.expand_dims(x_mask, -1))

    return x, x_len


class EncoderAvg(Encoder):
    """
    Encodes the input as an average of its word embeddings.
    """

    def encode(self, x, x_len, name):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        l = tf.expand_dims(x_len, -1)
        ones = tf.ones_like(l)
        return tf.reduce_sum(x, 1) / tf.where(tf.less(l, 1e-7), ones, l)


class EncoderCNN(Encoder):
    """
    Encodes the input using a simple CNN and max-over-time pooling.
    """

    def encode(self, x, x_len, name):
        pools = []
        for size in [2, 3, 4]:
            # Run CNN over words
            # (batch, input_len, filters)
            pool = tf.layers.conv1d(
                    name="{}_conv_{}".format(name, size),
                    inputs=x,
                    filters=256,
                    kernel_size=size,
                    padding="same",
                    activation=tf.nn.relu)

            # Max over-time pooling
            # (batch, filters)
            pool = tf.reduce_max(pool, 1)

            pools.append(pool)

        # Max over-time pooling
        # (batch, cnns * filters)
        return tf.concat(pools, 1)


class EncoderRNN(Encoder):
    """
    Encodes the input using a simple LSTM, returning the output
    from the last RNN step.
    """

    def encode(self, x, x_len, name):
        # Run encoding RNN
        # (batch_size, size, rnn_hidden_size)
        cell = tf.nn.rnn_cell.LSTMCell(256)
        outputs, state = tf.nn.dynamic_rnn(cell, x,
                                           name="{}_rnn".format(name),
                                           sequence_length=x_len,
                                           dtype=tf.float32)

        # Take last rnn output
        # (batch, rnn_hidden_size)
        return state.h
