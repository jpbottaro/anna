"""Train Encoder/Decoder models for Multi-label Classification tasks."""

import os
import numpy as np
import tensorflow as tf


class Trainer():
    """
    Trains Multi-label Classification models. Generalizes most common models
    using:
        - Encoder: Takes the input and creates a tensor representation (e.g.
                   average word embeddings, RNN, etc.)
        - Decoder: Original input and output from the encoder, and builds the
                   result (e.g. feedforward, classifier chains, RNN, etc.)
    """

    def __init__(self,
                 model_dir,
                 labels,
                 encoder,
                 decoder,
                 batch_size=32):
        """
        Trains Multi-label Classification models.

        Args:
            model_dir (str): path to the folder where the model will be stored
            labels (list[str]): list of possible labels
            encoder (Encoder): transforms the input text into numbers
            decoder (Decoder): takes the encoded input and produces labels
        """
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params={
                "encoder": encoder,
                "decoder": decoder,
                "label_vocab": labels
            })

    def train(self, docs, test_docs):
        """
        Train model on `docs`, and run evaluations on `test_docs`.

        The datasets are expected to produce a tuple of:
            - dict(str -> tf.Tensor): having tokenized `title` and `text`
            - tf.Tensor: the list of labels as strings

        Args:
            docs (tf.data.Dataset): the documents for training
            test_docs (tf.data.Dataset): the documents for evaluation
        """
        train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: input_fn(docs,
                                          batch_size=self.batch_size,
                                          shuffle=1000))

        eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: input_fn(test_docs,
                                          batch_size=self.batch_size))

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


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

        docs = docs.padded_batch(batch_size, padded_shapes=({
            "title": [None],
            "title_mask": [None],
            "text": [None],
            "text_mask": [None]
        }, [None]))

        # Return the read end of the pipeline.
        next_fn = docs.make_one_shot_iterator().get_next()

    return next_fn


def model_fn(features, labels, mode, params):
    encoder = params["encoder"]
    decoder = params["decoder"]
    label_vocab = params["label_vocab"]

    with tf.name_scope("expected_output"):
        labels = label_idx_to_hot(labels, label_vocab)

    with tf.name_scope("model"):
        net = encoder(features, mode)
        predictions, loss = decoder(net, labels, mode)

    pred = None
    metrics = None
    train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
        pred = {
            "class_ids": predictions,
            "probabilities": probabilities,
            "logits": logits,
        }
    else:
        metrics = create_metrics(labels, predictions, label_vocab)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(loss)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics,
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


def label_hot_to_idx(name, labels, vocab):
    # Find all positive labels (ignoring which document they come from)
    idx = tf.cast(tf.where(tf.equal(labels, 1.)), tf.int64)
    idx = idx[:,1]

    # Fetch string labels for the first document
    first_idx = tf.cast(tf.where(tf.equal(labels[0], 1.)), tf.int64)
    names = tf.contrib.lookup.index_to_string_table_from_tensor(
        vocab,
        default_value="_UNK_",
        name="{}_output".format(name)).lookup(first_idx)

    return idx, names


def create_metrics(labels, predictions, vocab):
    with tf.name_scope("metrics"):
        expected_labels_idx, expected_labels_str = label_hot_to_idx(
                "expected", labels, vocab)
        predicted_labels_idx, predicted_labels_str = label_hot_to_idx(
                "predicted", predictions, vocab)

        n_expected_labels = tf.metrics.mean(tf.reduce_sum(labels, 1))
        n_predicted_labels = tf.metrics.mean(tf.reduce_sum(predictions, 1))
        precision = tf.metrics.precision(labels, predictions)
        recall = tf.metrics.recall(labels, predictions)
        accuracy = tf.metrics.mean(
                tf.reduce_all(tf.equal(labels, predictions), 1))

    tf.summary.scalar("out/n_expected_labels", n_expected_labels[1])
    tf.summary.scalar("out/n_predicted_labels", n_predicted_labels[1])
    tf.summary.scalar("perf/accuracy", accuracy[1])
    tf.summary.scalar("perf/precision", precision[1])
    tf.summary.scalar("perf/recall", recall[1])
    tf.summary.text("out/expected_labels_examples", expected_labels_str)
    tf.summary.text("out/predicted_labels_examples", predicted_labels_str)
    tf.summary.histogram("out/expected_labels_dist", expected_labels_idx)
    tf.summary.histogram("out/predicted_labels_dist", predicted_labels_idx)

    return {
        "out/n_expected_labels": n_expected_labels,
        "out/n_predicted_labels": n_predicted_labels,
        "perf/accuracy": accuracy,
        "perf/precision": precision,
        "perf/recall": recall,
    }


def create_optimizer(loss):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss, global_step=tf.train.get_global_step())
