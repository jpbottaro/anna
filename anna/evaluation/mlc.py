"""Evaluation metrics for Multi-label Classification"""

import copy
from collections import Counter


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
        """Creates subset accuracy metric

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
        """Creates Hamming accuracy metric

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


class Metrics():
    """Evaluation metrics for Multi-label Classification"""

    def __init__(self, expected, predicted, labels):
        """Metrics for a particular prediction.

        Args:
            expected (list[list[str]]): list of expected label lists per doc
            predicted (list[list[str]]): list of predicted label lists per doc
            labels (list[str]): all the possible labels in the task
        """
        self.accuracy = Accuracy(expected, predicted, labels)
        self.hamming = Hamming(expected, predicted, labels)

    def __str__(self):
        metrics = [self.accuracy, self.hamming]
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
