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
                 folder_name="model",
                 name="unnamed",
                 batch_size=64,
                 learning_rate=0.001,
                 grad_clip=0.,
                 decay_rate=1.,
                 decay_steps=0,
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
            decay_rate (float): the factor to decay the learning rate
            decay_steps (int): how many steps to wait for each decay
        """
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 1
        config = tf.estimator.RunConfig(
            session_config=session_config,
            keep_checkpoint_max=1
        )
        model_dir = os.path.join(data_dir, folder_name, name)
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            params={
                "encoder": encoder,
                "decoder": decoder,
                "label_vocab": labels,
                "learning_rate": learning_rate,
                "grad_clip": grad_clip,
                "decay_rate": decay_rate,
                "decay_steps": decay_steps
            })

    def train(self, docs, test_docs=None,
              val_size=500, shuffle=10000, epochs=10, repeat=1):
        """
        Train model on `docs`, and run evaluations on `test_docs`.

        The datasets are expected to produce a tuple of:
            - dict(str -> tf.Tensor): having tokenized `title` and `text`
            - tf.Tensor: the list of labels as strings

        Args:
            docs (tf.data.Dataset): the documents for training
            test_docs (tf.data.Dataset): the documents for evaluation
            val_size (int): size of the validation set, in nr of docs
            shuffle (int): size of the buffer use to shuffle the training set
            epochs (int): max number of epochs to run
            repeat (int): how many times to repeat the training data per epoch
        """

        def train_input():
            return input_fn(docs.skip(val_size),
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            repeat=repeat)

        def val_input():
            return input_fn(docs.take(val_size), batch_size=self.batch_size)

        def test_input():
            return input_fn(test_docs, batch_size=self.batch_size)

        for i in range(epochs):
            print("Starting epoch #{}".format(i))
            self.estimator.train(input_fn=train_input)
            if val_size:
                val_m = self.estimator.evaluate(val_input, name="val")
                metrics.display("val", val_m)
            if test_docs:
                test_m = self.estimator.evaluate(test_input, name="test")
                metrics.display("test", test_m)


def input_fn(docs, batch_size=64, shuffle=None, repeat=None):
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
    lr = params["learning_rate"]
    clipping = params["grad_clip"]
    decay_rate = params["decay_rate"]
    decay_steps = params["decay_steps"]

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
        train_op = create_optimizer(loss, lr, clipping, decay_rate, decay_steps)

    pred = {"predictions": predictions}
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      predictions=pred)


def label_idx_to_hot(labels, vocab):
    """
    Coverts the labels from a list of ids to a multi-hot vector.

    Args:
        labels (tf.Tensor): expected labels as a list of ids.
          [batch, nr_positive_labels]
        vocab (list[str]): vocabulary of labels

    Returns:
        labels (tf.Tensor): expected labels as a list of ids.
          [batch, nr_labels]
    """
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


def create_optimizer(loss, learning_rate, max_norm, decay_rate, decay_steps):
    """
    Creates an optimizing operation for the given loss function.

    Args:
        loss (tf.Tensor): a loss function to optimize against
        learning_rate (float): the learning rate for the optimizer
        max_norm (float): maximum norm for any gradient when doing the update
        decay_rate (float): the factor to decay the learning rate
        decay_steps (int): how many steps to wait for each decay

    Returns:
        updater (tf.Tensor): operation that updates a single step of the network
    """
    global_step = tf.train.get_or_create_global_step()

    if decay_rate < 1.:
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, decay_steps, decay_rate, staircase=True)

    tf.summary.scalar("misc/learning_rate", learning_rate)

    opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
    grad, var = zip(*opt.compute_gradients(loss))

    if max_norm > 0.:
        grad = clip_gradients(grad, max_norm)

    return opt.apply_gradients(zip(grad, var), global_step=global_step)


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
