"""Evaluation metrics for Multi-label Classification"""

import copy
from collections import Counter
import tensorflow as tf


def subset_accuracy(expected, predicted, labels):
    """Accuracy of labels, with accuracy defined as returning the exact
    set of labels expected.

    Args:
        expected (list[list[str]]): list of expected label lists per doc
        predicted (list[list[str]]): list of predicted label lists per doc
        labels (list[str]): all the possible labels in the task
    """
    if len(expected) == 0 or len(labels) == 0:
        return 1.

    correct = 0
    for e, p in zip(expected, predicted):
        if Counter(e) == Counter(p):
            correct += 1
    return correct / len(expected)


def hamming_accuracy(expected, predicted, labels):
    """Hamming accuracy computes how many labels are correctly predicted, both
    positive and negative, per doc.

    Args:
        expected (list[list[str]]): list of expected label lists per doc
        predicted (list[list[str]]): list of predicted label lists per doc
        labels (list[str]): all the possible labels in the task

    Returns:
        value (float): result of the metric against `predicted`
    """
    if len(expected) == 0 or len(labels) == 0:
        return 1.

    correct = 0
    for e, p in zip(expected, predicted):
        for l in labels:
            in_e = l in e
            in_p = l in p
            if (in_e and in_p) or (not in_e and not in_p):
                correct += 1
    return correct / (len(expected) * len(labels))


def example_based_f1(expected, predicted, labels):
    """Compromise between Subset Accuracy and Hamming, measuring disparities
    without being too strict (like Accuracy), nor too rewarding for negative
    labels (like Hamming).

    Args:
        expected (list[list[str]]): list of expected label lists per doc
        predicted (list[list[str]]): list of predicted label lists per doc
        labels (list[str]): all the possible labels in the task

    Returns:
        value (float): result of the metric against `predicted`
    """
    if len(expected) == 0 or len(labels) == 0:
        return 1.

    value = 0
    for e, p in zip(expected, predicted):
        total_positives = 0
        matching_positives = 0
        for l in labels:
            in_e = 1 if l in e else 0
            in_p = 1 if l in p else 0
            total_positives += in_e + in_p
            matching_positives += in_e * in_p
        if total_positives == 0:
            value += 1
        else:
            value += (2 * matching_positives) / total_positives
    return value / len(expected)


def micro_f1(expected, predicted, labels):
    """
    Treats each label as a separate two-class prediction problem, computing
    true-positive, false-positive and false-negatives, averaging all
    predictions for all test instances (not per-label).

    This metric favors systems with better performance on frequent labels.

    Args:
        expected (list[list[str]]): list of expected label lists per doc
        predicted (list[list[str]]): list of predicted label lists per doc
        labels (list[str]): all the possible labels in the task

    Returns:
        value (float): result of the metric against `predicted`
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for e, p in zip(expected, predicted):
        for l in labels:
            in_e = l in e
            in_p = l in p
            if in_e and in_p:
                true_pos += 1
            elif in_e and not in_p:
                false_pos += 1
            elif in_p and not in_e:
                false_pos += 1
    div = (2 * true_pos) + false_neg + false_pos
    if div == 0:
        return 1.
    return (2 * true_pos) / div


def macro_f1(expected, predicted, labels):
    """
    Treats each label as a separate two-class prediction problem, computing
    true-positive, false-positive and false-negatives. As opposed to
    Micro-average, this metrics averages predictions per label, and later
    averages all labels together.

    This metric favors systems with better performance on rare labels.

    Args:
        expected (list[list[str]]): list of expected label lists per doc
        predicted (list[list[str]]): list of predicted label lists per doc
        labels (list[str]): all the possible labels in the task

    Returns:
        value (float): result of the metric against `predicted`
    """
    if len(labels) == 0:
        return 1.

    value = 0
    for l in labels:
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for e, p in zip(expected, predicted):
            in_e = l in e
            in_p = l in p
            if in_e and in_p:
                true_pos += 1
            elif in_e and not in_p:
                false_pos += 1
            elif in_p and not in_e:
                false_pos += 1
        div = (2 * true_pos) + false_neg + false_pos
        if div == 0:
            value += 1.
        else:
            value += (2 * true_pos) / div
    return value / len(labels)


all_metrics = {
    "acc": subset_accuracy,
    "hamming": hamming_accuracy,
    "ebf1": example_based_f1,
    "mif1": micro_f1,
    "maf1": macro_f1
}


def evaluate(expected, predicted, labels):
    """
    Evaluates the predicted documents against the expected ones, returning
    metrics for accuracy, label micro/macro f1, and more.

    Args:
        expected (list[Doc]): list of document with true labels
        predicted (list[Doc]): list of document with predicted labels
        labels (list[str]): all the possible labels in the task

    Returns:
        metrics (Metrics): metrics comparing `predicted` with `expected`
    """
    exp = [doc.labels for doc in expected]
    pred = [doc.labels for doc in predicted]
    return {metric: value(exp, pred, labels)
            for metric, value in all_metrics.items()}


def clean(docs):
    """
    Returns a new set of documents like `docs`, but without the labels.

    Args:
        docs (list[Doc]): list of document to clean

    Returns:
        analyzed_docs (list[Doc]): same as `docs`, without labels
    """
    new_docs = [copy.copy(d) for d in docs]
    for doc in new_docs:
        doc.labels = []
    return new_docs


class Evaluator(tf.keras.callbacks.Callback):
    """Keras callback to report metrics on test documents"""

    def __init__(self, prefix, predict, docs, labels):
        """
        Evaluates the given `model` against `docs`.

        Args:
            prefix (str): prefix name for all metrics added to `logs`
            predict (callable): a function that returns predictions for a
                                set of documents
            docs (list[Doc]): list of document with true labels
            labels (list[str]): all the possible labels in the task

        Returns:
            metrics (Metrics): metrics evaluating `model` on `docs`
        """
        super().__init__()
        self.prefix = prefix
        self.predict = predict
        self.docs = docs
        self.labels = labels

    def evaluate(self):
        predicted_docs = self.predict(clean(self.docs))
        return evaluate(self.docs, predicted_docs, self.labels)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = self.evaluate()
        for name, value in metrics.items():
            logs[self.prefix + "_" + name] = value
        print()
        print(self.prefix + " - " + ", ".join(["{}: {:.4f}".format(n, v)
                                               for n, v in metrics.items()]))
