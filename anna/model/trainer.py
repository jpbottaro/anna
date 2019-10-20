"""Train Encoder/Decoder models for Multi-label Classification tasks."""

import os
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import anna.model.metrics as metrics
import anna.data.utils as datautils


class Trainer:
    """
    Trains Multi-label Classification models. Generalizes most common models
    using:
        - Encoder: Takes the input and creates a tensor representation (e.g.
                   average word embeddings, RNN, etc.)
        - Decoder: Builds the output and loss functions (e.g. feedforward,
                   classifier chains, RNN, etc.)
    """

    def __init__(self,
                 data_dir,
                 label_vocab,
                 encoder,
                 decoder,
                 folder_name="model",
                 name="unnamed",
                 batch_size=64,
                 learning_rate=0.001,
                 grad_clip=0.,
                 *args,
                 **kwargs):
        """
        Trains Multi-label Classification models.

        Args:
            data_dir (str): path to the data folder
            label_vocab (list[str]): list of possible labels
            encoder (Encoder): transforms the input text into numbers
            decoder (Decoder): takes the encoded input and produces labels
            folder_name (str): name of the folder where to save the model
            name (str): name for the model (used to save checkpoints/summaries)
            batch_size (int): batch size for training
            learning_rate (float): training learning rate
            grad_clip (float): maximum norm for gradients when optimizing
        """
        self.model_dir = os.path.join(data_dir, folder_name, name)
        self.batch_size = batch_size
        self.label_vocab = label_vocab

        title = tfk.Input(shape=(None,), dtype=tf.string, name='title')
        title_mask = tfk.Input(shape=(None,), dtype=tf.float32, name='title_mask')
        text = tfk.Input(shape=(None,), dtype=tf.string, name='text')
        text_mask = tfk.Input(shape=(None,), dtype=tf.float32, name='text_mask')
        labels = tfk.Input(shape=(None,), dtype=tf.float32, name='labels')

        mem, mem_len, mem_fixed = encoder([title, title_mask, text, text_mask])
        pred = decoder([mem, mem_len, mem_fixed, labels])

        # The loss is defined in the decoder, add dummy loss here to make Keras happy
        self.model = tfk.Model(inputs=[title, title_mask, text, text_mask, labels], outputs=pred)
        self.model.compile(optimizer=tfa.optimizers.LazyAdam(learning_rate=learning_rate,
                                                             clipnorm=grad_clip),
                           loss=lambda a, b: 0.)
        self.model.summary()

    def train(self, docs_path, test_docs_path=None,
              val_size=500, shuffle=10000, epochs=10):
        """
        Train model on `docs`, and run evaluations on `test_docs`.

        The datasets are expected to produce a tuple of:
            - dict(str -> tf.Tensor): having tokenized `title` and `text`
            - tf.Tensor: the list of labels as strings

        Args:
            docs_path (str): the path to the documents for training
            test_docs_path (str): the path to the documents for evaluation
            val_size (int): size of the validation set, in nr of docs
            shuffle (int): size of the buffer use to shuffle the training set
            epochs (int): max number of epochs to run
        """
        docs = tf.data.TFRecordDataset([docs_path]).map(datautils.parse_example)
        test_docs = tf.data.TFRecordDataset([test_docs_path]).map(datautils.parse_example)

        train_data = input_fn("train_input",
                              docs.skip(val_size),
                              self.label_vocab,
                              batch_size=self.batch_size,
                              shuffle=shuffle)

        val_data = input_fn("val_input",
                            docs.take(val_size),
                            self.label_vocab,
                            batch_size=self.batch_size)

        test_data = input_fn("test_input",
                             test_docs,
                             self.label_vocab,
                             batch_size=self.batch_size)

        tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=self.model_dir)

        self.model.fit(train_data,
                       epochs=epochs,
                       validation_data=val_data,
                       callbacks=[tensorboard_callback])


def input_fn(name, docs, label_vocab, batch_size=64, shuffle=None, repeat=None):
    with tf.name_scope(name):
        def pad_empty(doc, labels):
            """
            Pads the input fields with an empty string if they are 0-length
            """
            doc = {k: tf.cond(pred=tf.equal(tf.shape(input=v)[0], 0),
                              true_fn=lambda: tf.constant([""]),
                              false_fn=lambda: v)
                   for k, v in doc.items()}
            return doc, labels

        def mask(doc, labels):
            """
            Adds a mask for each field before they are padded
            """
            ret = {k + "_mask": tf.ones_like(v, dtype=tf.float32)
                   for k, v in doc.items()}
            ret.update(doc)
            return ret, labels

        label_dict_init = tf.lookup.KeyValueTensorInitializer(
            label_vocab,
            list(range(len(label_vocab))),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        label_dict = tf.lookup.StaticHashTable(label_dict_init, -1)

        def labels_to_multihot(doc, labels):
            """
            Coverts the labels from a list of strings to a multi-hot vector.
            """
            # Convert strings to ids (pads become -1)
            # (batch, n_positive_labels)
            labels = label_dict.lookup(labels)

            # Turn indexes to one-hot vectors (pads become all 0)
            # (batch, n_positive_labels, n_classes)
            labels = tf.one_hot(labels, len(label_vocab), dtype=tf.float32)

            # Turn labels into fixed-sized 1/0 vector
            # (batch, n_classes)
            labels = tf.reduce_sum(input_tensor=labels, axis=1)

            return doc, labels

        def add_output_to_input(doc, labels):
            """
            Adds the labels to the layer's input too (e.g. used for seq2seq teacher training)
            """
            doc["labels"] = labels
            return doc, labels

        docs = docs.map(pad_empty)
        docs = docs.map(mask)
        docs = docs.map(labels_to_multihot)
        docs = docs.map(add_output_to_input)

        if repeat:
            docs = docs.repeat(repeat)

        if shuffle:
            docs = docs.shuffle(buffer_size=shuffle)

        pads = ({
                    "title": [None],
                    "title_mask": [None],
                    "text": [None],
                    "text_mask": [None],
                    "labels": len(label_vocab)
                }, len(label_vocab))
        return docs.padded_batch(batch_size, padded_shapes=pads)
