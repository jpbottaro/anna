"""Text encoders using neural network-inspired architectures.

Several encoders are defined, all using works as tokens (i.e. the atomic text
unit).

## Non-parametric encoders

@@EncoderAvg
@@EncoderMax

## CNN-based encoder

@@EncoderCNN

## RNN-based encoder (using GRU)

@@EncoderRNN
@@EncoderRNNAvg
@@EncoderRNNLast
"""
import numpy as np
import tensorflow as tf
import anna.data.dataset.fasttext as embeddings


class Encoder:
    """
    Encoder that takes the input features, and produces a
    vector representation of them.
    """

    def __init__(self,
                 data_dir,
                 input_names=None,
                 input_limit=None,
                 emb_size=20000,
                 oov_buckets=10000):
        """
        Creates an encoder with the given embeddings and maximum size
        for the input.

        Args:
            data_dir (str): path to the data folder
            input_names (list[str]): names of the string inputs to encode
            input_limit (int): maximum size to use from the input sequence
            emb_size (int): nr of embeddings to store (i.e. size of vocabulary)
            oov_buckets (int): nr of buckets to use for out-of-vocabulary words
        """
        if not input_names:
            input_names = ["title", "text"]

        self.data_dir = data_dir
        self.input_names = input_names
        self.input_limit = input_limit
        self.oov_buckets = oov_buckets

        # Fetch pre-trained word embeddings
        self.words, self.emb = embeddings.fetch_and_parse(data_dir,
                                                          voc_size=emb_size)

        if oov_buckets > 0:
            extra_emb = np.random.uniform(-1, 1, size=[oov_buckets,
                                                       self.emb.shape[1]])
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

        with tf.name_scope("encoder"):
            # Encode all inputs
            inputs = []
            for name in self.input_names:
                with tf.name_scope("input_" + name):
                    x, x_len = get_input(features,
                                         name,
                                         self.words,
                                         emb,
                                         self.input_limit,
                                         self.oov_buckets)
                    inputs.append(self.encode(x, x_len, name))

            # Concatenate inputs, two options:
            # fixed: (batch, len(input_names) * emb_size)
            # variable: (batch, sum(input_sizes), emb_size)
            return tf.concat(inputs, 1)

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


def get_input(features, name, words, emb, input_limit=None, oov_buckets=0):
    """
    Gets the sequence feature `name` from the `features`,
    trims the size if necessary, and maps it to its list
    of embeddings.

    Args:
        features (dict): dictionary of input features
        name (str): name of the feature to encode
        words (list[str]): list of strings as vocabulary
        emb (tf.Tensor): initialization for the word embeddings
        input_limit (int): maximum size to use from the input sequence
        oov_buckets (int): nr of buckets to use for out-of-vocabulary words

    Returns:
        x (tf.Tensor): the tensor of embeddings for the feature `name`
        x_len (tf.Tensor): the length each `x`
    """
    x = features[name]
    x_mask = features[name + "_mask"]

    # Limit size
    # (batch, input_limit)
    if input_limit:
        with tf.name_scope("trim"):
            x = x[:, :input_limit]
            x_mask = x_mask[:, :input_limit]

    # Length of each sequence
    # (batch)
    with tf.name_scope("length"):
        x_len = tf.reduce_sum(x_mask, 1)

    with tf.name_scope("embed"):
        # Convert strings to ids
        # (batch, input_limit)
        x = tf.contrib.lookup.index_table_from_tensor(
            mapping=words,
            default_value=1,
            num_oov_buckets=oov_buckets).lookup(x)

        # Replace with embeddings
        # (batch, input_limit, emb_size)
        x = tf.nn.embedding_lookup(emb, x)

        # Clear embeddings for pads
        # (batch, input_limit, emb_size)
        x = tf.multiply(x, x_mask[:, :, tf.newaxis])

    return x, x_len


class EncoderAvg(Encoder):
    """
    Encodes the input as an average of its word embeddings.
    """

    def encode(self, x, x_len, name):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        x_len = x_len[:, tf.newaxis]
        return tf.reduce_sum(x, 1) / tf.maximum(x_len, tf.ones_like(x_len))


class EncoderMax(Encoder):
    """
    Encodes the input taking the max of each of its word embedding dimentions.
    """

    def encode(self, x, x_len, name):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        return tf.reduce_max(x, 1)


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
    Encodes the input using a bidirectional GRU, returning the the outputs for
    all steps, and the last two states in both directions.
    """

    def encode(self, x, x_len, name):
        """
        Uses a bidirectional GRU to encode the input `x`.

        Args:
            x (tf.Tensor [batch, len, emb_size]): the padded input documents, as
              a list of embeddings.
            x_len (tf.Tensor [batch]): the size of each document.
            name (str, optional): name for this operation.

        Returns:
            outputs (tf.Tensor [batch, len, rnn_size]): the output of the rnn
              at each step.
            states (tf.Tensor [batch, rnn_size]): the last rnn output of each
              document.
        """
        hidden_size = 1024
        x_len = tf.cast(x_len, tf.int32)
        c_fw = tf.nn.rnn_cell.GRUCell(hidden_size,
                                      name="rnn_fw",
                                      reuse=tf.AUTO_REUSE)
        c_bw = tf.nn.rnn_cell.GRUCell(hidden_size,
                                      name="rnn_bw",
                                      reuse=tf.AUTO_REUSE)

        # Runs encoding RNN
        # outputs: 2 x (batch_size, size, hidden_size)
        # states: 2 x (batch_size, hidden_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(c_fw,
                                                          c_bw,
                                                          x,
                                                          sequence_length=x_len,
                                                          dtype=tf.float32)

        # Concatenate forward and backward passes
        # first: (batch_size, size, 2 * hidden_size)
        # second: (batch_size, 2 * hidden_size)
        return tf.concat(outputs, 2), tf.concat(states, 1)


class EncoderRNNLast(EncoderRNN):
    """
    Encodes the input using a bidirectional GRU, returning the output
    from the last RNN steps concatenated.
    """

    def encode(self, x, x_len, name):
        _, states = super().encode(x, x_len, name)

        return states


class EncoderRNNAvg(EncoderRNN):
    """
    Encodes the input using a bidirectional GRU, returning the
    average output value in both directions.
    """

    def encode(self, x, x_len, name):
        outputs, _ = super().encode(x, x_len, name)

        # Avoid zero division
        # (batch, 1)
        x_len = tf.maximum(x_len, tf.ones_like(x_len))
        x_len = x_len[:, tf.newaxis]

        # Average all hidden vectors
        # (batch_size, rnn_hidden_size)
        return tf.reduce_sum(outputs, 1) / x_len
