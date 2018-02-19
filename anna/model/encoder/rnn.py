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

        # Feed input to the RNN
        # (batch, steps, hidden_size)
        rnn = tf.keras.layers.GRU(self.hidden_size, return_state=True)
        _, x1_h = rnn(x1_emb)
        _, x2_h = rnn(x2_emb)

        # Concatenate all last hidden states
        # (batch, num_inputs * hidden_size)
        x = tf.keras.layers.concatenate([x1_h, x2_h])

        return [x1, x2], x

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
