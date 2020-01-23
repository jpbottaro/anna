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
import tensorflow.keras as tfk
import anna.model.utils as utils
import anna.data.dataset.glove as glove
import anna.data.dataset.fasttext as fasttext


class Encoder(tfk.layers.Layer):
    """
    Encoder that takes the input features, and produces a
    vector representation of them.
    """

    def __init__(self,
                 data_dir,
                 name="encoder",
                 input_limit=None,
                 voc_size=100000,
                 oov_size=1000,
                 fixed_embeddings=False,
                 pretrained_embeddings="glove",
                 *args,
                 **kwargs):
        """
        Creates an encoder with the given embeddings and maximum size
        for the input.

        Args:
            data_dir (str): path to the data folder
            input_names (list[str]): names of the string inputs to encode
            input_limit (int): maximum size to use from the input sequence
            voc_size (int): nr of embeddings to store (i.e. size of vocabulary)
            oov_size (int): nr of buckets to use for out-of-vocabulary words
            fixed_embeddings (bool): whether the embeddings should be trained
            pretrained_embeddings (str): which pretrained embeddings to use.
              options are "glove", "fasttext" (default "glove")
        """
        super(Encoder, self).__init__(name=name)

        self.data_dir = data_dir
        self.input_limit = input_limit
        self.voc_size = voc_size
        self.oov_size = oov_size
        self.fixed_emb = fixed_embeddings
        self.pretrained_emb = pretrained_embeddings

        # Fetch pre-trained word embeddings
        if pretrained_embeddings == "glove":
            self.words, self.emb = glove.fetch_and_parse(data_dir,
                                                         voc_size=voc_size)
        elif pretrained_embeddings == "fasttext":
            self.words, self.emb = fasttext.fetch_and_parse(data_dir,
                                                            voc_size=voc_size)
        else:
            raise ValueError("Unknown pretrained_embeddings: {}"
                             .format(pretrained_embeddings))

        if oov_size > 0:
            extra_emb = np.random.uniform(-.1, .1, size=[oov_size,
                                                         self.emb.shape[1]])
            self.emb = np.concatenate([self.emb, extra_emb])

    def get_config(self):
        """
        Gets the configuration of the encoder

        Returns:
            config (dict[str, str]): configuration of the encoder
        """
        return {
            'type': str(type(self)),
            'input_limit': self.input_limit,
            'voc_size': self.voc_size,
            'oov_size': self.oov_size,
            'fixed_emb': self.fixed_emb,
            'pretrained_embeddings': self.pretrained_emb
        }

    def build(self, input_shape):
        self.emb_tensor = self.add_weight(name="word_embeddings",
                                          shape=self.emb.shape,
                                          initializer=tf.initializers.constant(self.emb),
                                          trainable=not self.fixed_emb)

    def call(self, inputs, training=None):
        """
        Builds the encoder for a specific feature `name` in `features`.

        Args:
            inputs (dict[tf.Tensor]): a dict of tensors, expecting:
                - title (tf.Tensor): input words in the title
                  [batch, title_len, word_id]
                - title_mask (tf.Tensor): mask for words in the title
                  [batch, title_len]
                - title (tf.Tensor): input words in the title
                  [batch, title_len, word_id]
                - title_mask (tf.Tensor): mask for words in the title
                  [batch, title_len]
            training (bool): if this is training or eval

        Returns:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
        """

        title, title_mask, text, text_mask = inputs
        with tf.name_scope("encoder"):
            # Encode all inputs
            mem = []
            mem_len = []
            mem_fixed = []
            for name, feature, feature_mask in [("title", title, title_mask),
                                                ("text", text, text_mask)]:
                with tf.name_scope(name):
                    x, x_len = get_input(feature,
                                         feature_mask,
                                         self.words,
                                         self.emb_tensor,
                                         self.input_limit,
                                         self.oov_size)

                    m, m_len, m_fixed = self.encode(x, x_len, training)

                    # (batch, seq_len, size)
                    mem.append(m)
                    # (batch)
                    mem_len.append(m_len)
                    # (batch, emb_size)
                    mem_fixed.append(m_fixed)

            # Concatenate variable memory:
            # (batch, sum(seq_len), size)
            final_mem, final_mem_len = utils.seq_concat(mem, mem_len)

            # Concatenate fixed memory:
            # (batch, len(input_names) * emb_size)
            final_mem_fixed = tf.concat(mem_fixed, -1)

            return final_mem, final_mem_len, final_mem_fixed

    def encode(self, x, x_len, training):
        """
        Encode a given tensor `x` with length `x_len`.

        Args:
            x (tf.Tensor): the tensor we want to encode
            x_len (tf.Tensor): the length `x`
            training (bool): if this is training or eval

        Returns:
            mem (tf.Tensor): sequential representation of the input.
              [batch, sum(len), size]
            mem_len (tf.Tensor): length of `mem`.
              [batch]
            mem_fixed (tf.Tensor): fixed-sized representation of the input.
              [batch, size]
        """
        raise NotImplementedError


def get_input(x, x_mask, words, emb, input_limit=None, oov_size=0):
    """
    Trims the size of the feature `x` if necessary, and maps it to a list
    of embeddings.

    Args:
        x (tf.Tensor): the feature tensor
          [batch, len]
        x_mask (tf.Tensor): the mask for the given feature
          [batch]
        words (list[str]): list of strings as vocabulary
        emb (tf.Variable): initialization for the word embeddings
        input_limit (int): maximum size to use from the input sequence
        oov_size (int): nr of buckets to use for out-of-vocabulary words

    Returns:
        x (tf.Tensor): the tensor of embeddings for the feature `x`
          [batch, len, emb_size]
        x_len (tf.Tensor): the length each `x`
          [batch]
    """
    # Limit size
    # (batch, input_limit)
    if input_limit:
        with tf.name_scope("trim"):
            x = x[:, :input_limit]
            x_mask = x_mask[:, :input_limit]

    # Length of each sequence
    # (batch)
    with tf.name_scope("length"):
        x_len = tf.reduce_sum(input_tensor=x_mask, axis=1)
        x_len = tf.cast(x_len, tf.int32)

    with tf.name_scope("embed"):
        # Convert strings to ids
        # (batch, input_limit)
        init = tf.lookup.KeyValueTensorInitializer(
                words,
                list(range(len(words))),
                key_dtype=tf.string,
                value_dtype=tf.int64)
        if oov_size > 0:
            lookup_table = tf.lookup.StaticVocabularyTable(init, oov_size)
        else:
            lookup_table = tf.lookup.StaticHashTable(init, 1)
        x = lookup_table.lookup(x)

        # Count how many out-of-vocabulary (OOV) words are in the input
        num_oov = tf.cast(x, tf.float32) * x_mask
        num_oov = tf.logical_or(tf.equal(num_oov, 1),
                                tf.greater_equal(num_oov, len(words)))
        num_oov = tf.reduce_sum(input_tensor=tf.cast(num_oov, tf.float32), axis=1)

        # Replace with embeddings
        # (batch, input_limit, emb_size)
        x = tf.nn.embedding_lookup(params=emb, ids=x)

        # Clear embeddings for pads
        # (batch, input_limit, emb_size)
        x = x * x_mask[:, :, tf.newaxis]

    tf.summary.scalar("n_words", tf.reduce_mean(input_tensor=x_len))
    tf.summary.scalar("n_oov_words", tf.reduce_mean(input_tensor=num_oov))

    return x, x_len


class EncoderAvg(Encoder):
    """
    Encodes the input as an average of its word embeddings.
    """

    def encode(self, x, x_len, training):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        div = tf.cast(x_len[:, tf.newaxis], tf.float32)
        result = tf.reduce_sum(input_tensor=x, axis=1) / tf.maximum(div, tf.ones_like(div))
        return x, x_len, result


class EncoderMax(Encoder):
    """
    Encodes the input taking the max of each of its word embedding dimensions.
    """

    def encode(self, x, x_len, training):
        # Average embeddings, avoiding zero division when the input is empty
        # (batch, emb_size)
        return x, x_len, tf.reduce_max(input_tensor=x, axis=1)


class EncoderCNN(Encoder):
    """
    Encodes the input using a simple CNN and max-over-time pooling.
    """

    def encode(self, x, x_len, training):
        pools = []
        for size in [2, 3, 4]:
            # Run CNN over words
            # (batch, input_len, filters)
            pool = tfk.layers.Conv1D(
                name="conv_{}".format(size),
                filters=256,
                kernel_size=size,
                padding="same",
                activation=tf.nn.relu)(x)

            # Max over-time pooling
            # (batch, filters)
            pool = tfk.layers.GlobalMaxPool1D(pool)

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
                 rnn_type="gru",
                 hidden_size=1024,
                 dropout=.3,
                 *args,
                 **kwargs):
        super().__init__(data_dir, *args, **kwargs)
        self.cell = utils.rnn_cell(rnn_type,
                                   hidden_size,
                                   dropout)

    def encode(self, x, x_len, training):
        raise NotImplementedError


class EncoderUniRNN(EncoderRNN):
    """
    Encodes the input using a unidirectional RNN.
    """

    def encode(self, x, x_len, training):
        # Runs encoding RNN
        # result: [(batch_size, size, hidden_size), state1, state2, ...]
        # state: (batch_size, hidden_size)
        result = tfk.layers.RNN(self.cell,
                                return_sequences=True,
                                return_state=True)(x, tf.sequence_mask(x_len))

        # outputs: (batch_size, size, hidden_size)
        # states: [(batch_size, hidden_size)]
        outputs, states = result[0], result[1:]

        # Get the right part of the state (useful for tuple states like in LSTM)
        # (batch_size, hidden_size)
        state = utils.rnn_state_trim(self.rnn_type, states)

        return outputs, x_len, state


class EncoderBiRNN(EncoderRNN):
    """
    Encodes the input using a bidirectional RNN.
    """

    def encode(self, x, x_len, training):
        # Runs encoding Bidirectional RNN
        # result: [(batch_size, size, hidden_size), state1, state2, ...]
        # state: (batch_size, hidden_size)
        rnn = tfk.layers.RNN(self.cell,
                             return_sequences=True,
                             return_state=True)
        result = tfk.layers.Bidirectional(rnn, merge_mode="concat")(x, tf.sequence_mask(x_len))

        # outputs: (batch_size, size, hidden_size)
        # states: [(batch_size, hidden_size)]
        outputs, states = result[0], result[1:]

        return outputs, x_len, tf.concat(states, axis=-1)
