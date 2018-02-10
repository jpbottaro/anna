"""Encodes text using an RNN."""

import tensorflow as tf
import anna.nlp.utils as nlp
import anna.dataset.fasttext.parser as embeddings


class RNNEncoder():
    """
    Encodes a given document using an RNN.
    """

    def __init__(self, data_dir, hidden_size, max_words, fixed_emb, voc_size):
        """
        Creates an RNN encoder.

        Args:
            data_dir (str): path to the folder where datasets are stored
            hidden_size (int): size of the hidden units on each hidden layer
            max_words (int): number of words to use when embedding text fields
            fixed_emb (bool): True if word embeddings shouldn't be trainable
            voc_size (int): maximum size of the word vocabulary
        """
        self.data_dir = data_dir
        self.max_words = max_words
        self.train_emb = not fixed_emb
        self.hidden_size = hidden_size
        self.voc, self.emb = embeddings.fetch_and_parse(data_dir, voc_size)

    def build(self):
        """
        Builds the encoding layer, returning the list of expected inputs (which
        can be populated with `#encode()`), and the resulting embedding as a
        layer to be used in a model.

        Returns:
            inputs (list[tf.keras.layers.Input]): list of inputs to be used to
                                                  build the doc embedding
            fixed_emb (tf.keras.layers.Layer): a layer/tensor with a fixed
                                               embedding of the input
            var_emb (tf.keras.layers.Layer): a layer/tensor with an variable
                                             embedding of the input, decoder
                                             would need to process it (e.g.
                                             using attention)
        """
        # (batch, word)
        x1 = tf.keras.layers.Input(shape=(self.max_words,),
                                   dtype="int32", name="title_input")
        x2 = tf.keras.layers.Input(shape=(self.max_words,),
                                   dtype="int32", name="text_input")

        # (batch, word, emb)
        emb_layer = tf.keras.layers.Embedding(self.emb.shape[0],
                                              self.emb.shape[1],
                                              weights=[self.emb],
                                              input_length=self.max_words,
                                              mask_zero=True,
                                              trainable=self.train_emb)
        x1_emb = emb_layer(x1)
        x2_emb = emb_layer(x2)

        # Feed input to the RNN, keeping only the last output (both directions
        # are concatenated)
        # (batch, steps, hidden_size * 2)
        rnn = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)
        rnn = tf.keras.layers.Bidirectional(rnn)
        x1_out = rnn(x1_emb)
        x2_out = rnn(x2_emb)

        # Fixed embedding using last output of each direction
        def keep_last(x):
            return tf.concat([x[:, 0, :], x[:, -1, :]], axis=-1)
        keeper = tf.keras.layers.Lambda(keep_last)

        # Fixed embedding using last output in each direction of each input
        # (batch, num_inputs * 2 * hidden_size * 2)
        x = tf.keras.layers.concatenate([keeper(x1_out), keeper(x2_out)])

        return [x1, x2], x, x2_out

    def encode(self, docs):
        """
        Encodes the given document into vector representations, to feed into
        the model.

        Args:
            docs (list[Doc]): list of document to predict labels for

        Returns:
            input_tensors (map[str->np.array]): tensors representing the `docs`
        """
        title = nlp.text_to_sequence([d.title for d in docs],
                                     self.voc, self.max_words)
        text = nlp.text_to_sequence([d.text for d in docs],
                                    self.voc, self.max_words)

        return {"title_input": title, "text_input": text}
