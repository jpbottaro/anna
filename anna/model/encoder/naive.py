"""Simple document encoder as average of its word embeddings."""

import tensorflow as tf
import anna.nlp.utils as nlp
import anna.dataset.fasttext.parser as embeddings


class NaiveEmbeddingEncoder():
    """
    Encodes a given document as an average of its word embeddings.
    """

    def __init__(self, data_dir, max_words, fixed_emb, voc_size):
        """
        Creates an encoder that represents a document as the average of its
        first `max_words` words' embeddings. This is done for each field in the
        document (e.g. title, text, etc.), and all of those embeddings are
        concatenated.

        Args:
            data_dir (str): path to the folder where datasets are stored
            max_words (int): number of words to use when embedding text fields
            fixed_emb (bool): True if word embeddings shouldn't be trainable
            voc_size (int): maximum size of the word vocabulary
        """
        self.data_dir = data_dir
        self.max_words = max_words
        self.train_emb = not fixed_emb
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

        # Average all embeddings to create each text representation
        # (batch, emb)
        avg_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
        x1_emb = avg_layer(x1_emb)
        x2_emb = avg_layer(x2_emb)

        # Concatenate all inputs
        # (batch, emb * num_inputs)
        x = tf.keras.layers.concatenate([x1_emb, x2_emb])

        return [x1, x2], x, x2_emb

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
