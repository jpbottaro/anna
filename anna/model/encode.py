"""Text encoders using neural network-inspired architectures.

Several encoders are defined, all using works as tokens (i.e. the atomic text
unit).

## Non-parametric encoders

@@EncoderAvg
@@EncoderMax

## CNN-based encoder

@@EncoderCNN

## RNN-based encoder

@@EncoderUniRNN
@@EncoderBiRNN
"""
import numpy as np
import tensorflow as tf
import anna.model.utils as utils
import anna.data.dataset.fasttext as embeddings


class Encoder:
    """
    Encoder that takes the input features, and produces a
    vector representation of them.
    """

    def __init__(self,
                 data_dir,
                 fixed_embeddings=False,
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
        self.fixed_emb = fixed_embeddings

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
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
        """
        emb = tf.get_variable("word_embeddings",
                              self.emb.shape,
                              initializer=tf.constant_initializer(self.emb),
                              trainable=self.fixed_emb)

        with tf.name_scope("encoder"):
            # Encode all inputs
            memory = []
            memory_len = []
            memory_fixed = []
            for name in self.input_names:
                with tf.name_scope("input_" + name):
                    x, x_len = get_input(features,
                                         name,
                                         self.words,
                                         emb,
                                         self.input_limit,
                                         self.oov_buckets)
                    mem, mem_len, mem_fixed = self.encode(x, x_len, mode, name)
                    # (batch, seq_len, size)
                    memory.append(mem)
                    # (batch)
                    memory_len.append(mem_len)
                    # (batch, emb_size)
                    memory_fixed.append(mem_fixed)

            # Concatenate variable memory:
            # (batch, sum(seq_len), size)
            final_memory, final_memory_len = seq_concat(memory, memory_len)

            # Concatenate fixed memory:
            # (batch, len(input_names) * emb_size)
            final_memory_fixed = tf.concat(memory_fixed, -1)

            return final_memory, final_memory_len, final_memory_fixed

    def encode(self, x, x_len, mode, name):
        """
        Encode a given tensor `x` with length `x_len`.

        Args:
            x (tf.Tensor): the tensor we want to encode
            x_len (tf.Tensor): the length `x`
            mode (tf.estimator.ModeKeys): the mode we are on
            name (str): name of the tensor

        Returns:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
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
        x_len = tf.to_int32(x_len)

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


def seq_pad(x, size):
    """

    Args:
        x (tf.Tensor): the tensor we want to pad
          [batch, steps, emb_size]
        size (int): size to pad

    Returns:
        x (tf.Tensor): the padded tensor
          [batch, size, emb_size]
    """
    emb_size = x.get_shape()[-1].value
    x = tf.pad(x, [[0, 0], [0, size], [0, 0]])
    x = x[:, :size, :]
    x.set_shape([None, None, emb_size])

    return x


def seq_roll(x, size):
    """
    Rolls the input `x` by `size` positions to the right, wrapping around.

    Similar to OpenNMT's roll_sequence.

    Args:
        x (tf.Tensor): the tensor we want to roll
          [batch, steps, emb_size]
        size (tf.Tensor): how many positions we want to roll `x` to the right
          [batch]

    Returns:
        y (tf.Tensor): the rolled tensor
          [batch, steps, emb_size]
    """
    batch_size = tf.shape(x)[0]
    steps = tf.shape(x)[1]

    # Build grid with all indices in the tensor `x`
    # x = [1 3 3 7]
    # size = 2
    cols, rows = tf.meshgrid(tf.range(steps), tf.range(batch_size))

    # Substract the amount we want to roll
    # [-2 -1 0 1]
    cols -= tf.expand_dims(size, 1)

    # Take the remainder of the division. This returns the rolled indices, e.g.
    # [2 3 0 1]
    cols = tf.floormod(cols, steps)

    # Build back the indices using the rolled targets
    indices = tf.stack([rows, cols], axis=-1)

    # Shuffle the input with the constructed indices
    return tf.gather_nd(x, indices)


def seq_concat(memory, memory_len):
    """
    Concatenates a list of sequential memories into a single sequence.

    Similar to OpenNMT's ConcatReducer.

    Args:
        memory (list[tf.Tensor]): a list of sequential tensors to concatenate.
          [batch, len, size]
        memory_len (list[tf.Tensor]): the length tensor in `memory`.
          [batch]

    Returns:
        final_mem (tf.Tensor): sequential representation of the input.
          [batch, sum(len), size]
        final_mem_len (tf.Tensor): length of `final_mem`.
          [batch]
    """
    # Calculate final length of each instance in the batch
    # [batch]
    final_len = tf.add_n(memory_len)

    # Calculate largest instance
    max_len = tf.reduce_max(final_len)

    # Pad memory to the largest combined sequence
    # list([batch, max_len, emb_size])
    memory = [seq_pad(m, max_len) for m in memory]

    # Build padded final memory
    # [batch, max_len, emb_size]
    final_mem = None
    accum_len = None

    # Roll and accumulate sequences to obtain the final concatenation
    for x, x_len in zip(memory, memory_len):
        if final_mem is None:
            final_mem = x
            accum_len = x_len
        else:
            final_mem += seq_roll(x, accum_len)
            accum_len += x_len

    return final_mem, final_len


class EncoderAvg(Encoder):
    """
    Encodes the input as an average of its word embeddings.
    """

    def encode(self, x, x_len, mode, name):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        div = tf.to_float(x_len[:, tf.newaxis])
        result = tf.reduce_sum(x, 1) / tf.maximum(div, tf.ones_like(div))
        return x, x_len, result


class EncoderMax(Encoder):
    """
    Encodes the input taking the max of each of its word embedding dimentions.
    """

    def encode(self, x, x_len, mode, name):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        return x, x_len, tf.reduce_max(x, 1)


class EncoderCNN(Encoder):
    """
    Encodes the input using a simple CNN and max-over-time pooling.
    """

    def encode(self, x, x_len, mode, name):
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
        result = tf.concat(pools, 1)

        return x, x_len, result


class EncoderRNN(Encoder):
    """
    Encodes the input using a bidirectional RNN, returning the outputs for
    all steps.
    """

    def __init__(self,
                 data_dir,
                 input_names=None,
                 input_limit=None,
                 emb_size=20000,
                 oov_buckets=10000,
                 rnn_type="lstm",
                 hidden_size=256,
                 dropout=.5):
        super().__init__(
            data_dir, input_names, input_limit, emb_size, oov_buckets)

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn_type = rnn_type

    def encode(self, x, x_len, mode, name):
        raise NotImplementedError


class EncoderUniRNN(EncoderRNN):
    """
    Encodes the input using a unidirectional RNN.
    """

    def encode(self, x, x_len, mode, name):
        cell = utils.rnn_cell(self.rnn_type,
                              self.hidden_size,
                              mode,
                              self.dropout)

        # Runs encoding RNN
        # out: (batch_size, size, hidden_size)
        # states: (2 or 0, batch_size, hidden_size)
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           x,
                                           sequence_length=x_len,
                                           swap_memory=True,
                                           dtype=tf.float32)

        # Get the right part of the state (useful for tuple states like in LSTM)
        # (batch_size, 2 * hidden_size)
        state = utils.rnn_state_trim(self.rnn_type, state)

        return outputs, x_len, state


class EncoderBiRNN(EncoderRNN):
    """
    Encodes the input using a bidirectional RNN.
    """

    def encode(self, x, x_len, mode, name):
        c_fw = utils.rnn_cell(self.rnn_type,
                              self.hidden_size // 2,
                              mode,
                              self.dropout)
        c_bw = utils.rnn_cell(self.rnn_type,
                              self.hidden_size // 2,
                              mode,
                              self.dropout)

        # Runs encoding RNN
        # out: 2 x (batch_size, size, hidden_size)
        # states: 2 x (2 or 0, batch_size, hidden_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(c_fw,
                                                          c_bw,
                                                          x,
                                                          sequence_length=x_len,
                                                          swap_memory=True,
                                                          dtype=tf.float32)

        # Concatenates outputs
        # out: (batch_size, size, 2 * hidden_size)
        # states: (batch_size, 2 * hidden_size)
        outputs = tf.concat(outputs, axis=-1)
        states = tf.concat(states, axis=-1)

        # Get the right part of the state (useful for tuple states like in LSTM)
        # (batch_size, 2 * hidden_size)
        state = utils.rnn_state_trim(self.rnn_type, states)

        return outputs, x_len, state
