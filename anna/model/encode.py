import tensorflow as tf


class Encoder():
    """
    Encoder that takes the input features, and produces a
    vector representation of them.
    """

    def __init__(self, vocab, emb, input_names=None, max_size=None):
        """
        Creates an encoder with the given embeddings and maximum size
        for the input.

        Args:
            vocab (list[str]): list of strings as vocabulary
            emb (np.array): initialization for the word embeddings
            max_size (int): maximum size to use from the input sequence
        """
        if not input_names:
            input_names = ["title", "text"]

        self.input_names = input_names
        self.vocab = vocab
        self.emb = emb
        self.max_size = max_size

    def __call__(self, features, mode):
        """
        Builds the encoder for a specific feature `name` in `features`.

        Args:
            features (dict): dictionary of input features
            mode (tf.estimator.ModeKeys): the mode we are on

        Returns:
            y (tf.Tensor): the final representation of `x`
        """
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
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
                                         self.vocab,
                                         emb,
                                         self.max_size)
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


def get_input(features, name, vocab, emb, max_size=None):
    """
    Gets the sequence feature `name` from the `features`,
    trims the size if necessary, and maps it to its list
    of embeddings.

    Args:
        features (dict): dictionary of input features
        name (str): name of the feature to encode
        vocab (list[str]): list of strings as vocabulary
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
            mapping=vocab,
            default_value=0).lookup(x)

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
