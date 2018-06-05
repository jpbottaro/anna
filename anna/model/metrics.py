"""Tensorflow implementation of metrics for Multi-label Classification.

Metrics: Subset Accuracy, Hamming accuracy, example-based f1, and label-based
         micro/macro f1.

Definitions from: https://papers.nips.cc/paper/7125-maximizing-subset-accuracy-with-recurrent-neural-networks-in-multi-label-classification.pdf"""  # noqa

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


def create(labels, predictions, vocab):
    num_classes = len(vocab)
    with tf.name_scope("metrics"):
        expected_labels_idx, expected_labels_str = _label_hot_to_idx(
            "expected", labels, vocab)
        predicted_labels_idx, predicted_labels_str = _label_hot_to_idx(
            "predicted", predictions, vocab)

        n_expected_labels = tf.metrics.mean(tf.reduce_sum(labels, 1))
        n_predicted_labels = tf.metrics.mean(tf.reduce_sum(predictions, 1))

        f_micro = f_measure(labels, predictions, num_classes)
        f_macro = f_measure(labels, predictions, num_classes, micro=False)
        f_ex = f_example(labels, predictions)
        hamming = tf.metrics.accuracy(labels, predictions)
        accuracy = tf.metrics.mean(
            tf.reduce_all(tf.equal(labels, predictions), 1))

    metrics = {
        "out/n_expected_labels": n_expected_labels,
        "out/n_predicted_labels": n_predicted_labels,
        "perf/miF1": f_micro,
        "perf/maF1": f_macro,
        "perf/ebF1": f_ex,
        "perf/hamming": hamming,
        "perf/accuracy": accuracy,
    }

    for name, value in metrics.items():
        tf.summary.scalar(name, value[1])

    tf.summary.text("out/expected_labels_examples", expected_labels_str)
    tf.summary.text("out/predicted_labels_examples", predicted_labels_str)
    tf.summary.histogram("out/expected_labels_dist", expected_labels_idx)
    tf.summary.histogram("out/predicted_labels_dist", predicted_labels_idx)

    return metrics


def display(name, metrics):
    message = "\t{}".format(name)
    message += "\tloss: {:.6f}".format(metrics["loss"])
    message += "\t" + "\t".join(["{}: {:.4f}".format(k[5:], v)
                                 for k, v in metrics.items() if "perf" in k])
    print(message)


def _label_hot_to_idx(name, labels, vocab):
    # Find all positive labels (ignoring which document they come from)
    idx = tf.cast(tf.where(tf.equal(labels, 1.)), tf.int64)
    idx = idx[:, 1]

    # Fetch string labels for the first document
    first_idx = tf.cast(tf.where(tf.equal(labels[0], 1.)), tf.int64)
    names = tf.contrib.lookup.index_to_string_table_from_tensor(
        vocab,
        default_value="_UNK_",
        name="{}_output".format(name)).lookup(first_idx)

    return idx, names


def f_measure(labels,
              predictions,
              num_classes,
              micro=True,
              name=None):
    """Computes the label-based f1 score of the predictions with respect to
    the labels.

    The `f_measure` function creates three local variables,
    `true_positives`, `false_positives` and `false_negatives`, that are used to
    compute the f measure. This value is ultimately returned as `f_measure`, an
    idempotent operation.

    For estimation of the metric over a stream of data, the function creates an
    `update_op` that updates these variables and returns the `f_measure`.

    Args:
      labels (tf.Tensor): the ground truth values, a `Tensor` whose dimensions
        must match `predictions`.
      predictions (tf.Tensor): the predicted values, a `Tensor` of arbitrary
        dimensions.
      num_classes (int): the possible number of labels the prediction task can
        have.
      micro (bool, optional): Whether the f measure should be taken globally
        (i.e. micro), or averaged per class (i.e. macro).
      name: An optional variable_scope name.

    Returns:
      f_measure: Scalar float `Tensor` with the f measure.
      update_op: `Operation` that increments `true_positives`,
        `false_positives` and `false_negatives` variables appropriately and
        whose value matches `f_measure`.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.f_measure is not '
                           'supported when eager execution is enabled.')

    with variable_scope.variable_scope(name, 'f_measure',
                                       (predictions, labels)):

        def count_hits(expected, predicted):
            hits = math_ops.logical_and(expected, predicted)
            hits = math_ops.cast(hits, dtypes.float32)
            return math_ops.reduce_sum(hits, axis=0)

        is_true_positive = count_hits(
            math_ops.equal(labels, 1.),
            math_ops.equal(predictions, 1.))

        is_false_positive = count_hits(
            math_ops.equal(labels, 0.),
            math_ops.equal(predictions, 1.))

        is_false_negative = count_hits(
            math_ops.equal(labels, 1.),
            math_ops.equal(predictions, 0.))

        tp_var = metric_variable([num_classes], dtypes.float32)
        fp_var = metric_variable([num_classes], dtypes.float32)
        fn_var = metric_variable([num_classes], dtypes.float32)

        tp_up = state_ops.assign_add(tp_var, is_true_positive)
        fp_up = state_ops.assign_add(fp_var, is_false_positive)
        fn_up = state_ops.assign_add(fn_var, is_false_negative)

        def compute_f_measure(tp, fp, fn, micro, name):
            if micro:
                tp = math_ops.reduce_sum(tp)
                fp = math_ops.reduce_sum(fp)
                fn = math_ops.reduce_sum(fn)
            value = 2 * tp
            den = 2 * tp + fp + fn
            res = array_ops.where(math_ops.greater(den, 0),
                                  math_ops.div(value, den),
                                  array_ops.ones_like(value))
            return math_ops.reduce_mean(res, name=name)

        f = compute_f_measure(tp_var, fp_var, fn_var, micro, 'value')
        update_op = compute_f_measure(tp_up, fp_up, fn_up, micro, 'update_op')

        return f, update_op


def f_example(labels,
              predictions,
              metrics_collections=None,
              updates_collections=None,
              name=None):
    """Computes the example-based f1 score of the predictions with respect to
    the labels.

    The `f_measure` uses the `tf.metrics.mean` to store the streaming counts.

    For estimation of the metric over a stream of data, the function creates an
    `update_op` that updates these variables and returns the `f_measure`.

    Args:
      labels: The ground truth values, a `Tensor` whose dimensions must match
        `predictions`. Will be cast to `bool`.
      predictions: The predicted values, a `Tensor` of arbitrary dimensions.
        Will be cast to `bool`.
      metrics_collections: An optional list of collections that `f_measure`
        should be added to.
      updates_collections: An optional list of collections that `update_op`
        should be added to.
      name: An optional variable_scope name.

    Returns:
      f_measure: Scalar float `Tensor` with the f measure.
      update_op: `Operation` that increments `true_positives`,
        `false_positives` and `false_negatives` variables appropriately and
        whose value matches `f_measure`.

    Raises:
      ValueError: If `predictions` and `labels` have mismatched shapes, or if
        either `metrics_collections` or `updates_collections` are not a list or
        tuple.
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.f_measure is not '
                           'supported when eager execution is enabled.')

    # Calculate the double of the true positives
    value = 2. * math_ops.reduce_sum(labels * predictions, 1)

    # Calculate denominator as sum of non-zero values of both matrices
    den = math_ops.count_nonzero(labels, 1) + \
          math_ops.count_nonzero(predictions, 1)
    den = tf.cast(den, dtypes.float32)

    # Avoid division by zero
    res = array_ops.where(math_ops.greater(den, 0),
                          math_ops.div(value, den),
                          array_ops.ones_like(value), name)

    return tf.metrics.mean(res,
                           metrics_collections=metrics_collections,
                           updates_collections=updates_collections,
                           name=name)


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections.

    Taken from tf.metrics directly, as the function is not exposed."""

    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        name=name)
