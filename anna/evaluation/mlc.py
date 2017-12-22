"""Evaluation metrics for Multi-label Classification"""

import copy
from collections import Counter
import tensorflow as tf


class Metric:
    """Single metric for a task"""

    def __init__(self, name, expected, predicted, labels):
        """Metrics for a particular prediction.

        Args:
            name (str): name of the metric
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        if len(expected) != len(predicted):
            raise ValueError("expected and predicted labels not equal length")
        self.name = name
        self.expected = expected
        self.predicted = predicted
        self.labels = labels

    def __str__(self):
        return "{} = {:.4f}".format(self.name, self.value())


class Accuracy(Metric):
    """Accuracy of labels, with accuracy defined as returning the exact
    set of labels expected."""

    def __init__(self, expected, predicted, labels):
        """Creates a Subset Accuracy metric

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        super().__init__("acc", expected, predicted, labels)

    def value(self):
        correct = 0
        for e, p in zip(self.expected, self.predicted):
            if Counter(e) == Counter(p):
                correct += 1
        return correct / len(self.expected)


class Hamming(Metric):
    """Hamming accuracy computes how many labels are correctly predicted, both
    positive and negative, per doc"""

    def __init__(self, expected, predicted, labels):
        """Creates a Hamming Accuracy metric

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        super().__init__("hamming", expected, predicted, labels)

    def value(self):
        correct = 0
        for e, p in zip(self.expected, self.predicted):
            for l in self.labels:
                in_e = l in e
                in_p = l in p
                if (in_e and in_p) or (not in_e and not in_p):
                    correct += 1
        return correct / (len(self.expected) * len(self.labels))


class ExampleBasedF1(Metric):
    """Compromise between Subset Accuracy and Hamming, measuring disparities
    without being too strict (like Accuracy), nor too rewarding for negative
    labels (like Hamming)."""

    def __init__(self, expected, predicted, labels):
        """Creates an Example Based F1 metric

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        super().__init__("example-based F1", expected, predicted, labels)

    def value(self):
        value = 0
        for e, p in zip(self.expected, self.predicted):
            total_positives = 0
            matching_positives = 0
            for l in self.labels:
                in_e = 1 if l in e else 0
                in_p = 1 if l in p else 0
                total_positives += in_e + in_p
                matching_positives += in_e * in_p
            if total_positives == 0:
                value += 1
            else:
                value += (2 * matching_positives) / total_positives
        return value / len(self.expected)


class MicroF1(Metric):
    """Treats each label as a separate two-class prediction problem, computing
    true-positive, false-positive and false-negatives, averaging all
    predictions for all test instances (not per-label).

    This metric favors systems with better performance on frequent labels."""

    def __init__(self, expected, predicted, labels):
        """Creates a Label Based Micro F1 metric

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        super().__init__("micro F1", expected, predicted, labels)

    def value(self):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for e, p in zip(self.expected, self.predicted):
            for l in self.labels:
                in_e = l in e
                in_p = l in p
                if in_e and in_p:
                    true_pos += 1
                elif in_e and not in_p:
                    false_pos += 1
                elif in_p and not in_e:
                    false_pos += 1
        return (2 * true_pos) / ((2 * true_pos) + false_neg + false_pos)


class MacroF1(Metric):
    """Treats each label as a separate two-class prediction problem, computing
    true-positive, false-positive and false-negatives. As opposed to
    Micro-average, this metrics averages predictions per label, and later
    averages all labels together.

    This metric favors systems with better performance on rare labels."""

    def __init__(self, expected, predicted, labels):
        """Creates a Label Based Macro F1 metric

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        super().__init__("macro F1", expected, predicted, labels)

    def value(self):
        value = 0
        for l in self.labels:
            true_pos = 0
            false_pos = 0
            false_neg = 0
            for e, p in zip(self.expected, self.predicted):
                in_e = l in e
                in_p = l in p
                if in_e and in_p:
                    true_pos += 1
                elif in_e and not in_p:
                    false_pos += 1
                elif in_p and not in_e:
                    false_pos += 1
            value += (2 * true_pos) / ((2 * true_pos) + false_neg + false_pos)
        return value / len(self.labels)


class Metrics():
    """Evaluation metrics for Multi-label Classification"""

    def __init__(self, expected, predicted, labels):
        """Metrics for a particular prediction.

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        self.acc = Accuracy(expected, predicted, labels)
        self.hamming = Hamming(expected, predicted, labels)
        self.ebf1 = ExampleBasedF1(expected, predicted, labels)
        self.mif1 = MicroF1(expected, predicted, labels)
        self.maf1 = MacroF1(expected, predicted, labels)

    def __str__(self):
        metrics = [self.acc, self.hamming, self.ebf1, self.mif1, self.maf1]
        return ", ".join([str(m) for m in metrics])


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
    return Metrics(exp, pred, labels)


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


class EvaluationCallback(tf.keras.callbacks.Callback):
    """Keras callback to report metrics on test documents"""

    def __init__(self, predictor, docs, labels):
        """
        Evaluates the given `model` against `docs`.

        Args:
            predictor (callable): a function that returns predictions for
                                  a set of documents
            docs (list[Doc]): list of document with true labels
            labels (list[str]): all the possible labels in the task

        Returns:
            metrics (Metrics): metrics evaluating `model` on `docs`
        """
        super().__init__()
        self.predictor = predictor
        self.docs = docs
        self.labels = labels

    def evaluate(self):
        predicted_docs = self.predictor(clean(self.docs))
        return evaluate(self.docs, predicted_docs, self.labels)

    def on_train_begin(self, logs={}):
        self.metrics = []

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.evaluate()
        self.metrics.append(metrics)
        print()
        print(metrics)
