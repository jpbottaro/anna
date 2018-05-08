"""Train Encoder/Decoder models for Multi-label Classification tasks."""

import os
import tensorflow as tf
import anna.model.metrics as metrics


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
                 labels,
                 encoder,
                 decoder,
                 name="unnamed",
                 batch_size=32):
        """
        Trains Multi-label Classification models.

        Args:
            data_dir (str): path to the data folder
            labels (list[str]): list of possible labels
            encoder (Encoder): transforms the input text into numbers
            decoder (Decoder): takes the encoded input and produces labels
        """
        config = tf.estimator.RunConfig(
            keep_checkpoint_max=1
        )
        model_dir = os.path.join(data_dir, "model", name)
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            params={
                "encoder": encoder,
                "decoder": decoder,
                "label_vocab": labels
            })

    def train(self, docs, test_docs=None, val_size=500, epochs=50):
        """
        Train model on `docs`, and run evaluations on `test_docs`.

        The datasets are expected to produce a tuple of:
            - dict(str -> tf.Tensor): having tokenized `title` and `text`
            - tf.Tensor: the list of labels as strings

        Args:
            docs (tf.data.Dataset): the documents for training
            test_docs (tf.data.Dataset): the documents for evaluation
            val_size (int): size of the validation set, in nr of docs
            epochs (int): max number of epochs to run
        """
        def train_input():
            return input_fn(docs.skip(val_size),
                            batch_size=self.batch_size,
                            shuffle=10000)

        def val_input():
            return input_fn(docs.take(val_size), batch_size=self.batch_size)

        def test_input():
            return input_fn(test_docs, batch_size=self.batch_size)

        i = 0
        while i < epochs:
            print("Starting epoch #{}".format(i))
            self.estimator.train(input_fn=train_input)
            if val_size:
                val_m = self.estimator.evaluate(val_input, name="val")
                metrics.display("val", val_m)
            if test_docs:
                test_m = self.estimator.evaluate(test_input, name="test")
                metrics.display("test", test_m)
            i += 1


def input_fn(docs, batch_size=32, shuffle=None, repeat=None):
    with tf.name_scope("input_processing"):
        def mask(doc, labels):
            title_mask = tf.ones_like(doc["title"], dtype=tf.float32)
            text_mask = tf.ones_like(doc["text"], dtype=tf.float32)
            doc["title_mask"] = title_mask
            doc["text_mask"] = text_mask
            return doc, labels

        docs = docs.map(mask)

        if repeat:
            docs = docs.repeat(repeat)

        if shuffle:
            docs = docs.shuffle(buffer_size=shuffle)

        pads = ({
                    "title": [None],
                    "title_mask": [None],
                    "text": [None],
                    "text_mask": [None]
                }, [None])
        docs = docs.padded_batch(batch_size, padded_shapes=pads)

        return docs.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    encoder = params["encoder"]
    decoder = params["decoder"]
    label_vocab = params["label_vocab"]

    with tf.name_scope("expected_output"):
        if labels is not None:
            labels = label_idx_to_hot(labels, label_vocab)

    with tf.name_scope("model"):
        net = encoder(features, mode)
        predictions, loss = decoder(net, labels, mode)

    eval_metric_ops = None
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
        eval_metric_ops = metrics.create(labels, predictions, label_vocab)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_optimizer(loss)

    pred = {"predictions": predictions}
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      predictions=pred)


def label_idx_to_hot(labels, vocab):
    # Convert strings to ids (pads become -1)
    # (batch, n_positive_labels)
    labels = tf.contrib.lookup.index_table_from_tensor(
        mapping=vocab,
        default_value=-1).lookup(labels)

    # Turn indexes to one-hot vectors (pads become all 0)
    # (batch, n_positive_labels, n_classes)
    labels = tf.one_hot(labels, len(vocab), dtype=tf.float32)

    # Turn labels into fixed-sized 1/0 vector
    # (batch, n_classes)
    return tf.reduce_sum(labels, 1)


def create_optimizer(loss):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss, global_step=tf.train.get_global_step())
