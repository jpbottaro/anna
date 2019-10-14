"""Train Encoder/Decoder models for Multi-label Classification tasks."""

import os
import tensorflow as tf
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
                 labels,
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
            labels (list[str]): list of possible labels
            encoder (Encoder): transforms the input text into numbers
            decoder (Decoder): takes the encoded input and produces labels
            folder_name (str): name of the folder where to save the model
            name (str): name for the model (used to save checkpoints/summaries)
            batch_size (int): batch size for training
            learning_rate (float): training learning rate
            grad_clip (float): maximum norm for gradients when optimizing
        """
        model_dir = os.path.join(data_dir, folder_name, name)
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=tf.estimator.RunConfig(keep_checkpoint_max=0),
            params={
                "encoder": encoder,
                "decoder": decoder,
                "label_vocab": labels,
                "learning_rate": learning_rate,
                "grad_clip": grad_clip
            })

    def train(self, docs_path, test_docs_path=None,
              val_size=500, shuffle=10000, epochs=10, eval_every=100):
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
            eval_every (int): number of training steps to wait for each eval
        """

        def train_input():
            docs = tf.data.TFRecordDataset([docs_path])\
                          .map(datautils.parse_example)
            return input_fn("train_input",
                            docs.skip(val_size),
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            repeat=epochs)

        def val_input():
            docs = tf.data.TFRecordDataset([docs_path])\
                          .map(datautils.parse_example)
            return input_fn("val_input",
                            docs.take(val_size),
                            batch_size=self.batch_size)

        def test_input():
            test_docs = tf.data.TFRecordDataset([test_docs_path])\
                               .map(datautils.parse_example)
            return input_fn("test_input",
                            test_docs,
                            batch_size=self.batch_size)

        hooks = []
        if val_size:
            hooks.append(tf.estimator.experimental.InMemoryEvaluatorHook(
                self.estimator,
                val_input,
                name="val",
                every_n_iter=eval_every))
        if test_docs_path:
            hooks.append(tf.estimator.experimental.InMemoryEvaluatorHook(
                self.estimator,
                test_input,
                name="test",
                every_n_iter=eval_every))

        self.estimator.train(input_fn=train_input, hooks=hooks)


def input_fn(name, docs, batch_size=64, shuffle=None, repeat=None):
    with tf.name_scope(name):
        def pad_empty(doc, labels):
            doc = {k: tf.cond(pred=tf.equal(tf.shape(input=v)[0], 0),
                              true_fn=lambda: tf.constant([""]),
                              false_fn=lambda: v)
                   for k, v in doc.items()}
            return doc, labels

        def mask(doc, labels):
            ret = {k + "_mask": tf.ones_like(v, dtype=tf.float32)
                   for k, v in doc.items()}
            ret.update(doc)
            return ret, labels

        docs = docs.map(pad_empty)
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
        return docs.padded_batch(batch_size, padded_shapes=pads)


def model_fn(features, labels, mode, params):
    encoder = params["encoder"]
    decoder = params["decoder"]
    label_vocab = params["label_vocab"]
    lr = params["learning_rate"]
    clipping = params["grad_clip"]

    with tf.name_scope("expected_output"):
        if labels is not None:
            labels = label_idx_to_hot(labels, label_vocab)

    with tf.name_scope("model"):
        mem, mem_len, mem_fixed = encoder(features, mode)
        predictions, loss = decoder(mem, mem_len, mem_fixed, labels, mode)

    eval_metric_ops = None
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
        eval_metric_ops = metrics.create(labels, predictions, label_vocab)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tfa.optimizers.LazyAdam(learning_rate=lr, clipnorm=clipping)
        train_vars = encoder.trainable_variables + decoder.trainable_variables
        train_op = opt.minimize(lambda: loss, train_vars)

    pred = {"predictions": predictions}
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      predictions=pred)


def label_idx_to_hot(labels, vocab):
    """
    Coverts the labels from a list of strings to a multi-hot vector.

    Args:
        labels (tf.Tensor): expected labels as a list of strings.
          [batch, nr_positive_labels]
        vocab (list[str]): vocabulary of labels

    Returns:
        labels (tf.Tensor): expected labels as multi-hot tensor
          [batch, nr_labels]
    """
    # Convert strings to ids (pads become -1)
    # (batch, n_positive_labels)
    init = tf.lookup.KeyValueTensorInitializer(
            vocab,
            list(range(len(vocab))),
            key_dtype=tf.string,
            value_dtype=tf.int64)
    labels = tf.lookup.StaticHashTable(init, -1).lookup(labels)

    # Turn indexes to one-hot vectors (pads become all 0)
    # (batch, n_positive_labels, n_classes)
    labels = tf.one_hot(labels, len(vocab), dtype=tf.float32)

    # Turn labels into fixed-sized 1/0 vector
    # (batch, n_classes)
    return tf.reduce_sum(input_tensor=labels, axis=1)
