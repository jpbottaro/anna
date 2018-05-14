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
        def pad_empty(doc, labels):
            doc = {k: tf.cond(tf.equal(tf.shape(v)[0], 0),
                              lambda: tf.constant([""]),
                              lambda: v)
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


def create_optimizer(loss, learning_rate=0.001, max_norm=5.0):
    """
    Creates an optimizing operation for the given loss function.

    Args:
        loss (tf.Tensor): a loss function to optimize against
        learning_rate (float): the learning rate for the optimizer
        max_norm (float): maximum norm for any gradient when doing the update

    Returns:
        updater (tf.Tensor): operation that updates a single step of the network
    """
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grad, var = zip(*opt.compute_gradients(loss))

    if max_norm > 0.:
        grad = clip_gradients(grad, max_norm)

    return opt.apply_gradients(zip(grad, var),
                               global_step=tf.train.get_global_step())


def clip_gradients(gradients, max_norm):
    """
    Clips gradients to the maximum `max_norm`.

    Args:
        gradients: the gradients to be applied when optimizing
        max_norm: the maximum norm for each gradient

    Returns:
        gradients: the gradients to be applied when optimizing
    """
    gradients, norm = tf.clip_by_global_norm(gradients, max_norm)

    tf.summary.scalar("misc/grad_norm", norm)
    tf.summary.scalar("misc/clipped_gradient", tf.global_norm(gradients))

    return gradients
